#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-TTS Norwegian Fine-Tuning (Scratch / Stable)
==================================================
- Multi JSONL support (no filename collisions)
- LoRA on talker.model (attention by default)
- Train text_projection (critical for pronunciation)
- Sub-talker loss (codebook 2-16) to reduce noise/skurring
- Forces 24kHz reference audio for speaker embedding

Example:
  accelerate launch src/train_norwegian_scratch_stable.py \
    --train_jsonl /workspace/data/train_with_codes_librivox.jsonl \
    --train_jsonl /workspace/data/train_with_codes_npsc.jsonl \
    --init_model_path /workspace/base_model \
    --output_model_path /workspace/output/run_scratch_v1 \
    --batch_size 4 \
    --grad_accum 4 \
    --num_epochs 10 \
    --save_every 1 \
    --hf_repo_id telvenes/qwen3-tts-norsk-finetune
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig
from huggingface_hub import HfApi


AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]


def _to_bool(x: str) -> bool:
    return x.lower() in ("1", "true", "yes", "y", "on")


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class NorwegianTTSDataset(Dataset):
    def __init__(self, jsonl_paths: List[str], processor, config, max_audio_seconds: float = 15.0):
        self.processor = processor
        self.config = config
        self.max_audio_seconds = max_audio_seconds

        self.items: List[Dict[str, Any]] = []
        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.items.append(json.loads(line))

        print(f"✅ Lastet {len(self.items)} eksempler fra {len(jsonl_paths)} datasett.")

    def __len__(self):
        return len(self.items)

    def _load_audio_24k(self, path: str) -> Tuple[np.ndarray, int]:
        try:
            wav, sr = librosa.load(path, sr=24000, mono=True)
            if self.max_audio_seconds > 0:
                max_len = int(24000 * self.max_audio_seconds)
                if len(wav) > max_len:
                    wav = wav[:max_len]
            return wav.astype(np.float32), 24000
        except Exception:
            return np.zeros(24000, dtype=np.float32), 24000

    def _build_assistant_text(self, text: str) -> str:
        # Same pattern you used; keeps it consistent with your previous runs
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_texts(self, text: str) -> torch.Tensor:
        inp = self.processor(text=text, return_tensors="pt", padding=True)
        ids = inp["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray):
        # Import mel_spectrogram from Qwen3-TTS repo
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)  # (1, T, 128)
        return mels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]

        text = it.get("text", "")
        if not text:
            # skip empty
            return self.__getitem__((idx + 1) % len(self.items))

        # target audio codes
        audio_codes = it.get("audio_codes")
        if audio_codes is None:
            return self.__getitem__((idx + 1) % len(self.items))

        # ref audio path
        ref_path = it.get("ref_audio") or it.get("ref_audio_path") or it.get("audio")
        if not ref_path:
            return self.__getitem__((idx + 1) % len(self.items))

        txt = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(txt)

        codes = torch.tensor(audio_codes, dtype=torch.long)  # (T, 16)
        wav, _ = self._load_audio_24k(ref_path)
        ref_mel = self.extract_mels(wav)

        # your earlier training removed last 5 tokens; keep consistent
        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": codes,
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        item_len = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_len) + 8
        B, T = len(batch), max_length

        input_ids = torch.zeros((B, T, 2), dtype=torch.long)
        codec_ids = torch.zeros((B, T, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_mask = torch.zeros((B, T), dtype=torch.bool)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        codec_0_labels = torch.full((B, T), -100, dtype=torch.long)

        for i, d in enumerate(batch):
            text_ids = d["text_ids"]                  # (1, L)
            audio_codecs = d["audio_codes"]           # (C, 16)
            audio_codec_0 = audio_codecs[:, 0]        # (C,)

            L = text_ids.shape[1]
            C = audio_codec_0.shape[0]

            # Text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8 + L - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + L - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + L - 2:8 + L + C, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8 + L + C] = True

            # Codec channel
            input_ids[i, 3:8, 1] = torch.tensor([
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,
                self.config.talker_config.codec_pad_id,
            ], dtype=torch.long)

            input_ids[i, 8:8 + L - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + L - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + L - 2, 1] = self.config.talker_config.codec_bos_id

            # codec_0 tokens occupy positions:
            start = 8 + L - 1
            end = start + C
            input_ids[i, start:end, 1] = audio_codec_0
            input_ids[i, end, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, start:end] = audio_codec_0
            # (optional) you can also train eos, but codec_mask excludes it for sub-loss:
            codec_0_labels[i, end] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, start:end, :] = audio_codecs

            codec_embedding_mask[i, 3:8 + L + C] = True
            codec_embedding_mask[i, 6] = False

            codec_mask[i, start:end] = True
            attention_mask[i, :8 + L + C] = 1

        ref_mels = torch.cat([d["ref_mel"] for d in batch], dim=0)  # (B, Tm, 128)

        return {
            "input_ids": input_ids,
            "codec_ids": codec_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_mask": codec_mask,
        }


# -----------------------------
# Train
# -----------------------------
def train():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_jsonl", action="append", required=True,
                    help="Can be repeated: --train_jsonl a.jsonl --train_jsonl b.jsonl")
    ap.add_argument("--init_model_path", type=str, required=True)
    ap.add_argument("--output_model_path", type=str, default="./output/run_scratch")

    ap.add_argument("--qwen_path", type=str, default="/workspace/Qwen3-TTS")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    ap.add_argument("--lora_lr", type=float, default=1e-5)
    ap.add_argument("--text_proj_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--sub_loss_weight", type=float, default=0.2)
    ap.add_argument("--disable_subtalker", action="store_true")

    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--train_mlp_lora", type=str, default="false")

    ap.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))

    args = ap.parse_args()
    set_seed(args.seed)

    # Import Qwen3-TTS
    if os.path.isdir(args.qwen_path):
        sys.path.append(args.qwen_path)
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except Exception:
        print("❌ Kunne ikke importere Qwen3TTSModel. Sjekk --qwen_path")
        raise

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_model_path,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_model_path, exist_ok=True)
        print("🚀 Starter norsk TTS-trening (Scratch / Stable)")
        print(f"   datasets: {args.train_jsonl}")
        print(f"   batch_size={args.batch_size} grad_accum={args.grad_accum} epochs={args.num_epochs}")
        print(f"   lora_lr={args.lora_lr} text_proj_lr={args.text_proj_lr} sub_loss_weight={args.sub_loss_weight}")
        print(f"   lora_r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} train_mlp_lora={_to_bool(args.train_mlp_lora)}")
        print(f"   mixed_precision={args.mixed_precision}")

    hf_api = None
    if args.hf_repo_id and accelerator.is_main_process:
        try:
            hf_api = HfApi()
            print(f"☁️  HF Upload aktivert: {args.hf_repo_id}")
        except Exception as e:
            print(f"⚠️  HF Error: {e}")

    # Load model
    qwen = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        dtype=torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32),
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model = qwen.model

    # Freeze all
    model.requires_grad_(False)

    # Fix peft edge-cases (some peft versions expect this attr)
    if not hasattr(model.talker.model, "prepare_inputs_for_generation"):
        model.talker.model.prepare_inputs_for_generation = lambda *a, **k: k

    # LoRA targets (default: attention only; optional MLP)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if _to_bool(args.train_mlp_lora):
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model.talker.model = get_peft_model(model.talker.model, peft_config)

    # Ensure hidden states are enabled (sub-loss needs them)
    try:
        model.talker.config.output_hidden_states = True
    except Exception:
        pass
    if hasattr(model.talker.model, "config"):
        try:
            model.talker.model.config.output_hidden_states = True
        except Exception:
            pass

    # Unfreeze text_projection (critical)
    if hasattr(model.talker, "text_projection"):
        for p in model.talker.text_projection.parameters():
            p.requires_grad = True

    # Build optimizer with 2 param groups (LoRA vs text_projection)
    lora_params = []
    tp_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "text_projection" in n:
            tp_params.append(p)
        else:
            lora_params.append(p)

    optimizer = AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": args.weight_decay},
            {"params": tp_params, "lr": args.text_proj_lr, "weight_decay": args.weight_decay},
        ]
    )

    # Config for dataset tokens
    try:
        cfg = AutoConfig.from_pretrained(args.init_model_path)
    except Exception:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    ds = NorwegianTTSDataset(args.train_jsonl, qwen.processor, cfg)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ds.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    # Logging helpers
    def count_trainable(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    if accelerator.is_main_process:
        print(f"✅ Trainable params: {count_trainable(model):,}")

    global_step = 0

    for epoch in range(args.num_epochs):
        for batch in dl:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(model.device)
                codec_ids = batch["codec_ids"].to(model.device)
                ref_mels = batch["ref_mels"].to(model.device, dtype=model.dtype)
                text_mask = batch["text_embedding_mask"].to(model.device)
                codec_mask_emb = batch["codec_embedding_mask"].to(model.device)
                attn_mask = batch["attention_mask"].to(model.device)
                labels = batch["codec_0_labels"].to(model.device)
                codec_mask = batch["codec_mask"].to(model.device)

                # Speaker embedding
                speaker_embedding = model.speaker_encoder(ref_mels).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                raw_text_embeds = model.talker.model.text_embedding(input_text_ids)
                proj_text_embeds = model.talker.text_projection(raw_text_embeds)
                text_emb = proj_text_embeds * text_mask

                codec_emb = model.talker.model.codec_embedding(input_codec_ids) * codec_mask_emb
                codec_emb[:, 6, :] = speaker_embedding

                inputs_embeds = text_emb + codec_emb

                outputs = model.talker(
                    inputs_embeds=inputs_embeds[:, :-1, :],
                    attention_mask=attn_mask[:, :-1],
                    labels=labels[:, 1:],
                    output_hidden_states=True,
                )

                loss_main = outputs.loss
                loss = loss_main

                sub_loss_val = None
                if (not args.disable_subtalker) and hasattr(model.talker, "forward_sub_talker_finetune"):
                    hs_list = getattr(outputs, "hidden_states", None)
                    if hs_list is not None and len(hs_list) > 0 and hs_list[-1] is not None:
                        last_h = hs_list[-1]  # expected (B, T-1, H)

                        # Align: hidden pos i predicts token at pos i+1
                        active = codec_mask[:, 1:]  # (B, T-1), only true where next token is a codec token

                        if active.any():
                            hs = last_h[active]                 # (N, H)
                            codes = codec_ids[:, 1:, :][active] # (N, 16)

                            _, sub_loss = model.talker.forward_sub_talker_finetune(
                                codec_ids=codes,
                                talker_hidden_states=hs,
                            )

                            sub_loss_val = sub_loss
                            loss = loss_main + (args.sub_loss_weight * sub_loss)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if accelerator.is_main_process and global_step % 25 == 0:
                    sub_print = "n/a" if sub_loss_val is None else f"{sub_loss_val.detach().float().item():.4f}"
                    print(
                        f"step={global_step} | loss={loss.detach().float().item():.4f} "
                        f"(main={loss_main.detach().float().item():.4f}, sub={sub_print})"
                    )

        # Save each epoch
        if accelerator.is_main_process and ((epoch + 1) % args.save_every == 0):
            save_path = os.path.join(args.output_model_path, f"epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)

            # Save LoRA adapter
            unwrapped.talker.model.save_pretrained(save_path)

            # Save text_projection
            if hasattr(unwrapped.talker, "text_projection"):
                torch.save(
                    unwrapped.talker.text_projection.state_dict(),
                    os.path.join(save_path, "text_projection.bin"),
                )

            meta = {
                "epoch": epoch + 1,
                "train_jsonl": args.train_jsonl,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lora_lr": args.lora_lr,
                "text_proj_lr": args.text_proj_lr,
                "sub_loss_weight": args.sub_loss_weight,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "train_mlp_lora": _to_bool(args.train_mlp_lora),
                "mixed_precision": args.mixed_precision,
                "seed": args.seed,
            }
            with open(os.path.join(save_path, "train_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            readme = [
                "---",
                "base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "library_name: peft",
                f"tags: [text-to-speech, norwegian, lora, epoch-{epoch+1}]",
                "---",
                f"# Qwen3-TTS Norwegian - Epoch {epoch+1}",
                "",
                "This checkpoint contains:",
                "- LoRA adapter weights",
                "- text_projection.bin (critical for pronunciation adaptation)",
                "- train_meta.json",
                "",
                "Inference: load adapter into base model + load text_projection.bin.",
            ]
            with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(readme))

            print(f"💾 Lagret: {save_path}")

            if hf_api and args.hf_repo_id:
                try:
                    hf_api.upload_folder(
                        folder_path=save_path,
                        repo_id=args.hf_repo_id,
                        path_in_repo=f"checkpoints/epoch_{epoch+1}",
                        repo_type="model",
                    )
                    print("☁️  HF Upload OK")
                except Exception as e:
                    print(f"⚠️  HF Upload Feil: {e}")


if __name__ == "__main__":
    train()
