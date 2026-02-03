#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-tuning (Scratch/Stable) - V2
=====================================================
Fixes:
- Pads ref_mels in collate_fn (prevents torch.cat/stack crash when mel lengths differ).
- PEFT shim: adds prepare_inputs_for_generation if missing (prevents LoRA/PEFT crash).

Train:
  accelerate launch --num_processes 1 train_norwegian_scratch_stable.py \
    --train_jsonl /workspace/data/train_with_codes_librivox.jsonl \
    --init_model_path /workspace/base_model \
    --output_model_path /workspace/output/run_scratch_v1 \
    --batch_size 4 --grad_accum 4 --num_epochs 10 --save_every 1 \
    --lora_lr 1e-5 --text_proj_lr 5e-5 --sub_loss_weight 0.2 \
    --lora_r 4 --lora_alpha 8 --lora_dropout 0.1 --train_mlp_lora false \
    --mixed_precision bf16 --hf_repo_id telvenes/qwen3-tts-norsk-finetune

Multi-dataset (repeat --train_jsonl):
  ... --train_jsonl A.jsonl --train_jsonl B.jsonl ...
"""

import argparse
import json
import os
import sys
from typing import List, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from accelerate import Accelerator
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi

# Qwen imports
sys.path.append("/workspace/Qwen3-TTS")
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]


class NorwegianTTSDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: List[str],
        processor,
        config,
        max_ref_seconds: float = 12.0,
        max_audio_seconds: float = 15.0,
    ):
        self.processor = processor
        self.config = config
        self.max_ref_seconds = float(max_ref_seconds)
        self.max_audio_seconds = float(max_audio_seconds)

        self.items = []
        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.items.append(json.loads(line))

        print(f"✅ Lastet {len(self.items)} eksempler fra {len(jsonl_paths)} datasett.")

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: str) -> np.ndarray:
        audio, _sr = librosa.load(path, sr=24000, mono=True)
        max_len = int(24000 * self.max_audio_seconds)
        if len(audio) > max_len:
            audio = audio[:max_len]
        return audio.astype(np.float32)

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize(self, text: str) -> torch.Tensor:
        out = self.processor(text=text, return_tensors="pt", padding=True)
        ids = out["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    @torch.inference_mode()
    def _mel(self, audio_np: np.ndarray) -> torch.Tensor:
        m = mel_spectrogram(
            torch.from_numpy(audio_np).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)  # (1, T, 128)

        # Cap length for speaker encoder stability
        max_frames = int(self.max_ref_seconds * 24000 / 256)
        if m.shape[1] > max_frames:
            m = m[:, :max_frames, :]

        return m.squeeze(0)  # (T, 128)

    def __getitem__(self, idx: int):
        # avoid recursion; retry a few times
        for _ in range(8):
            item = self.items[idx]
            ref_path = item.get("ref_audio") or item.get("ref_audio_path") or item.get("audio_path")
            text = item.get("text", "")
            codes = item.get("audio_codes")

            if ref_path and text and codes and os.path.exists(ref_path):
                text_ids = self._tokenize(self._build_assistant_text(text))[:, :-5]
                audio_codes = torch.tensor(codes, dtype=torch.long)

                wav = self._load_audio(ref_path)
                ref_mel = self._mel(wav)  # (Tm, 128)

                return {"text_ids": text_ids, "audio_codes": audio_codes, "ref_mel": ref_mel}

            idx = (idx + 1) % len(self.items)

        raise RuntimeError("Could not find a valid sample after several retries.")


def make_collate_fn(config):
    def collate_fn(batch):
        B = len(batch)
        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        T = max(item_length) + 8

        input_ids = torch.zeros((B, T, 2), dtype=torch.long)
        codec_ids = torch.zeros((B, T, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_mask = torch.zeros((B, T), dtype=torch.bool)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        codec_0_labels = torch.full((B, T), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codes = data["audio_codes"]
            audio_codec_0 = audio_codes[:, 0]

            Lt = text_ids.shape[1]
            Lc = audio_codec_0.shape[0]

            # Text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = config.tts_pad_token_id
            input_ids[i, 7, 0] = config.tts_bos_token_id
            input_ids[i, 8:8 + Lt - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + Lt - 3, 0] = config.tts_eos_token_id
            input_ids[i, 8 + Lt - 2:8 + Lt + Lc, 0] = config.tts_pad_token_id
            text_embedding_mask[i, :8 + Lt + Lc] = True

            # Codec channel
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    config.talker_config.codec_nothink_id,
                    config.talker_config.codec_think_bos_id,
                    config.talker_config.codec_think_eos_id,
                    0,
                    config.talker_config.codec_pad_id,
                ],
                dtype=torch.long,
            )
            input_ids[i, 8:8 + Lt - 3, 1] = config.talker_config.codec_pad_id
            input_ids[i, 8 + Lt - 3, 1] = config.talker_config.codec_pad_id
            input_ids[i, 8 + Lt - 2, 1] = config.talker_config.codec_bos_id
            input_ids[i, 8 + Lt - 1:8 + Lt - 1 + Lc, 1] = audio_codec_0
            input_ids[i, 8 + Lt - 1 + Lc, 1] = config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + Lt - 1:8 + Lt - 1 + Lc] = audio_codec_0
            codec_0_labels[i, 8 + Lt - 1 + Lc] = config.talker_config.codec_eos_token_id

            codec_ids[i, 8 + Lt - 1:8 + Lt - 1 + Lc, :] = audio_codes

            codec_embedding_mask[i, 3:8 + Lt + Lc] = True
            codec_embedding_mask[i, 6] = False  # speaker slot
            codec_mask[i, 8 + Lt - 1:8 + Lt - 1 + Lc] = True
            attention_mask[i, :8 + Lt + Lc] = True

        # ✅ PAD ref_mels so stack won't crash
        ref_mels = [d["ref_mel"] for d in batch]  # each (Tm, 128)
        max_tm = max(m.shape[0] for m in ref_mels)
        padded = []
        for m in ref_mels:
            pad_t = max_tm - m.shape[0]
            if pad_t > 0:
                m = F.pad(m, (0, 0, 0, pad_t))  # pad time dim at end
            padded.append(m)
        ref_mels = torch.stack(padded, dim=0)  # (B, Tm, 128)

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }

    return collate_fn


def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", action="append", required=True)
    ap.add_argument("--init_model_path", type=str, required=True)
    ap.add_argument("--output_model_path", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=1)

    ap.add_argument("--lora_lr", type=float, default=1e-5)
    ap.add_argument("--text_proj_lr", type=float, default=5e-5)
    ap.add_argument("--sub_loss_weight", type=float, default=0.2)

    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--train_mlp_lora", type=str, default="false")

    ap.add_argument("--mixed_precision", type=str, default="bf16")  # bf16/fp16/no
    ap.add_argument("--hf_repo_id", type=str, default=None)

    ap.add_argument("--max_ref_seconds", type=float, default=12.0)
    ap.add_argument("--max_audio_seconds", type=float, default=15.0)

    args = ap.parse_args()
    args.train_mlp_lora = _str2bool(args.train_mlp_lora)

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
        print(f"   lora_r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} train_mlp_lora={args.train_mlp_lora}")
        print(f"   mixed_precision={args.mixed_precision}")

    hf_api = None
    if args.hf_repo_id and accelerator.is_main_process:
        try:
            hf_api = HfApi()
            print(f"☁️  HF Upload aktivert: {args.hf_repo_id}")
        except Exception as e:
            print(f"⚠️  HF Upload deaktivert: {e}")

    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    qwen_wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        dtype=dtype,
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model = qwen_wrapper.model

    model.requires_grad_(False)

    # PEFT shim (some peft versions require this attr)
    if not hasattr(model.talker.model, "prepare_inputs_for_generation"):

        def _pifg(input_ids=None, **kwargs):
            if input_ids is not None:
                kwargs["input_ids"] = input_ids
            return kwargs

        model.talker.model.prepare_inputs_for_generation = _pifg  # type: ignore

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.train_mlp_lora:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model.talker.model = get_peft_model(model.talker.model, peft_config)

    # Unfreeze text_projection
    if hasattr(model.talker, "text_projection"):
        for p in model.talker.text_projection.parameters():
            p.requires_grad = True

    # Try force hidden states (best effort)
    try:
        model.talker.config.output_hidden_states = True
    except Exception:
        pass
    if hasattr(model.talker.model, "config"):
        try:
            model.talker.model.config.output_hidden_states = True
        except Exception:
            pass

    try:
        cfg = AutoConfig.from_pretrained(args.init_model_path)
    except Exception:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    ds = NorwegianTTSDataset(
        jsonl_paths=args.train_jsonl,
        processor=qwen_wrapper.processor,
        config=cfg,
        max_ref_seconds=args.max_ref_seconds,
        max_audio_seconds=args.max_audio_seconds,
    )

    num_workers = 2
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(cfg),
        persistent_workers=(num_workers > 0),
    )

    # param groups: LoRA vs text_projection
    lora_params, tp_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "text_projection" in n:
            tp_params.append(p)
        else:
            lora_params.append(p)

    optimizer = AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.01},
            {"params": tp_params, "lr": args.text_proj_lr, "weight_decay": 0.0},
        ]
    )

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        for batch in dl:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(model.device)
                codec_ids = batch["codec_ids"].to(model.device)
                ref_mels = batch["ref_mels"].to(model.device, dtype=model.dtype)
                text_embedding_mask = batch["text_embedding_mask"].to(model.device)
                codec_embedding_mask = batch["codec_embedding_mask"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                codec_0_labels = batch["codec_0_labels"].to(model.device)
                codec_mask = batch["codec_mask"].to(model.device)

                speaker_embedding = model.speaker_encoder(ref_mels).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                raw_text = model.talker.model.text_embedding(input_text_ids)
                proj_text = model.talker.text_projection(raw_text)
                input_text_embedding = proj_text * text_embedding_mask

                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                loss_main = outputs.loss
                loss = loss_main

                # Sub-talker (best effort)
                if args.sub_loss_weight > 0 and getattr(outputs, "hidden_states", None):
                    last_hidden = outputs.hidden_states[-1]
                    if last_hidden is not None:
                        active_mask = codec_mask[:, :-1]
                        if active_mask.sum() > 0:
                            hs = last_hidden[:, :-1][active_mask]
                            codes = codec_ids[:, :-1, :][active_mask]
                            _, sub_loss = model.talker.forward_sub_talker_finetune(
                                codec_ids=codes,
                                talker_hidden_states=hs,
                            )
                            loss = loss_main + args.sub_loss_weight * sub_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                if accelerator.is_main_process and global_step % 25 == 0:
                    print(f"step={global_step} | loss={loss.detach().item():.4f}")

        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch}/{args.num_epochs} done")

        if accelerator.is_main_process and (epoch % args.save_every == 0):
            save_path = os.path.join(args.output_model_path, f"epoch-{epoch}")
            os.makedirs(save_path, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)
            unwrapped.talker.model.save_pretrained(save_path)

            if hasattr(unwrapped.talker, "text_projection"):
                torch.save(unwrapped.talker.text_projection.state_dict(), os.path.join(save_path, "text_projection.bin"))

            with open(os.path.join(save_path, "train_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "datasets": args.train_jsonl,
                        "lora_lr": args.lora_lr,
                        "text_proj_lr": args.text_proj_lr,
                        "sub_loss_weight": args.sub_loss_weight,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "train_mlp_lora": args.train_mlp_lora,
                        "mixed_precision": args.mixed_precision,
                        "max_ref_seconds": args.max_ref_seconds,
                        "max_audio_seconds": args.max_audio_seconds,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"💾 Lagret: {save_path}")

            if hf_api and args.hf_repo_id:
                try:
                    hf_api.upload_folder(
                        folder_path=save_path,
                        repo_id=args.hf_repo_id,
                        path_in_repo=f"checkpoints/epoch_{epoch}",
                        repo_type="model",
                    )
                    print("☁️  HF Upload OK")
                except Exception as e:
                    print(f"⚠️  HF Upload Feil: {e}")


if __name__ == "__main__":
    train()
