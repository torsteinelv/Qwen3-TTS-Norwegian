#!/usr/bin/env python3
"""
Norwegian Qwen3-TTS finetune (Scratch / Stable / Upload-safe)

Key fixes:
- Stable ref_mels: enforce fixed ref length (seconds) + padding safety
- LoRA + text_projection training with separate LRs
- Optional Sub-talker loss (best effort; skips if hidden_states missing)
- HF upload: valid README frontmatter (base_model never empty)
- Multiple datasets: --train_jsonl can be passed multiple times
"""

import argparse
import json
import os
import sys
import time
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import AutoConfig
from huggingface_hub import HfApi

from peft import LoraConfig, get_peft_model


# -----------------------------
# Qwen import (repo on disk)
# -----------------------------
def _ensure_qwen_import(qwen_path: str):
    if qwen_path and qwen_path not in sys.path:
        sys.path.append(qwen_path)
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        return Qwen3TTSModel, mel_spectrogram
    except ImportError as e:
        raise SystemExit(
            f"Could not import Qwen3-TTS from {qwen_path}. "
            f"Set --qwen_path correctly. Import error: {e}"
        )


AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]


@dataclass
class TrainMeta:
    base_model_id: str
    init_model_path: str
    train_jsonls: List[str]
    batch_size: int
    grad_accum: int
    num_epochs: int
    save_every: int
    lora_lr: float
    text_proj_lr: float
    sub_loss_weight: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    train_mlp_lora: bool
    ref_seconds: float
    max_audio_seconds: float
    mixed_precision: str
    seed: int


class NorwegianTTSDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: List[str],
        processor,
        config,
        mel_spectrogram_fn,
        ref_seconds: float = 6.0,
        max_audio_seconds: float = 15.0,
        sample_rate: int = 24000,
    ):
        self.processor = processor
        self.config = config
        self.mel_spectrogram = mel_spectrogram_fn
        self.ref_seconds = float(ref_seconds)
        self.max_audio_seconds = float(max_audio_seconds)
        self.sr = int(sample_rate)

        self.items: List[Dict[str, Any]] = []
        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.items.append(json.loads(line))
                    except Exception:
                        continue

        print(f"✅ Lastet {len(self.items)} eksempler fra {len(jsonl_paths)} datasett.")

        self.ref_n = int(self.ref_seconds * self.sr)
        self.max_n = int(self.max_audio_seconds * self.sr)

    def __len__(self):
        return len(self.items)

    def _load_audio(self, path: str) -> np.ndarray:
        # Force 24kHz, mono
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        if audio is None:
            return np.zeros(self.sr, dtype=np.float32)
        audio = audio.astype(np.float32)

        # Hard cap to avoid huge refs
        if audio.shape[0] > self.max_n:
            audio = audio[: self.max_n]

        # Make fixed-length ref for stable speaker embedding
        if audio.shape[0] >= self.ref_n:
            # random crop for augmentation (helps not overfit)
            start = np.random.randint(0, audio.shape[0] - self.ref_n + 1)
            audio = audio[start : start + self.ref_n]
        else:
            pad = self.ref_n - audio.shape[0]
            audio = np.pad(audio, (0, pad), mode="constant")

        return audio

    def _build_assistant_text(self, text: str) -> str:
        # matches your existing formatting
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_text(self, text: str) -> torch.Tensor:
        toks = self.processor(text=text, return_tensors="pt", padding=False)
        input_ids = toks["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids  # (1, T)

    @torch.inference_mode()
    def _extract_mels(self, audio: np.ndarray) -> torch.Tensor:
        # mel_spectrogram expects torch audio [B, T]
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        mels = self.mel_spectrogram(
            wav,
            n_fft=1024,
            num_mels=128,
            sampling_rate=self.sr,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)  # (1, Tm, 128)
        return mels.squeeze(0).contiguous()  # (Tm, 128)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        # robust audio path keys
        ref_path = item.get("ref_audio") or item.get("ref_audio_path") or item.get("audio_path")
        if not ref_path or not os.path.exists(ref_path):
            # skip to a nearby sample to avoid crashing dataloader
            return self.__getitem__((idx + 1) % len(self.items))

        text = item.get("text", "")
        if not isinstance(text, str) or len(text.strip()) == 0:
            return self.__getitem__((idx + 1) % len(self.items))

        text = self._build_assistant_text(text.strip())
        text_ids = self._tokenize_text(text)  # (1, T)

        audio_codes = item.get("audio_codes", None)
        if audio_codes is None:
            return self.__getitem__((idx + 1) % len(self.items))

        audio_codes_t = torch.tensor(audio_codes, dtype=torch.long)
        if audio_codes_t.dim() == 1:
            # attempt reshape (rare)
            if audio_codes_t.numel() % 16 == 0:
                audio_codes_t = audio_codes_t.view(-1, 16)
            else:
                return self.__getitem__((idx + 1) % len(self.items))
        if audio_codes_t.shape[-1] != 16:
            return self.__getitem__((idx + 1) % len(self.items))

        wav = self._load_audio(ref_path)
        ref_mel = self._extract_mels(wav)  # (Tm, 128)

        # In your earlier code you used text_ids[:,:-5], keep it (avoid trailing tokens)
        if text_ids.shape[1] > 6:
            text_ids = text_ids[:, :-5]

        return {
            "text_ids": text_ids,           # (1, Ttxt)
            "audio_codes": audio_codes_t,   # (Tcodec, 16)
            "ref_mel": ref_mel,             # (Tm, 128)
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Ensure ref_mel shapes can stack
        # (Even with fixed ref_seconds, mel frame count can differ by 1-2 due to STFT rounding)
        ref_mels = [b["ref_mel"] for b in batch]
        max_tm = max(m.shape[0] for m in ref_mels)
        ref_mel_padded = []
        for m in ref_mels:
            if m.shape[0] < max_tm:
                pad = max_tm - m.shape[0]
                m = torch.nn.functional.pad(m, (0, 0, 0, pad))  # pad time
            ref_mel_padded.append(m)
        ref_mels = torch.stack(ref_mel_padded, dim=0)  # (B, Tm, 128)

        item_len = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_len = max(item_len) + 8

        B = len(batch)
        T = max_len

        input_ids = torch.zeros((B, T, 2), dtype=torch.long)
        codec_ids = torch.zeros((B, T, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_mask = torch.zeros((B, T), dtype=torch.bool)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        codec_0_labels = torch.full((B, T), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]      # (1, Ttxt)
            codes = data["audio_codes"]      # (Tcodec, 16)
            codec0 = codes[:, 0]             # (Tcodec,)

            tlen = text_ids.shape[1]
            clen = codec0.shape[0]

            # ---- text channel (0) ----
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + tlen - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + tlen - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + tlen - 2 : 8 + tlen + clen, 0] = self.config.tts_pad_token_id

            text_embedding_mask[i, : 8 + tlen + clen] = True

            # ---- codec channel (1) ----
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,
                    self.config.talker_config.codec_pad_id,
                ],
                dtype=torch.long,
            )

            input_ids[i, 8 : 8 + tlen - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + tlen - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + tlen - 2, 1] = self.config.talker_config.codec_bos_id

            # codec tokens
            start = 8 + tlen - 1
            input_ids[i, start : start + clen, 1] = codec0
            input_ids[i, start + clen, 1] = self.config.talker_config.codec_eos_token_id

            # labels for codec0
            codec_0_labels[i, start : start + clen] = codec0
            codec_0_labels[i, start + clen] = self.config.talker_config.codec_eos_token_id

            # full codec ids (all 16 codebooks)
            codec_ids[i, start : start + clen, :] = codes

            # masks
            codec_embedding_mask[i, 3 : 8 + tlen + clen] = True
            codec_embedding_mask[i, 6] = False  # speaker embedding slot
            codec_mask[i, start : start + clen] = True
            attention_mask[i, : 8 + tlen + clen] = 1

        return {
            "input_ids": input_ids,  # (B,T,2)
            "codec_ids": codec_ids,  # (B,T,16)
            "ref_mels": ref_mels,    # (B,Tm,128)
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_mask": codec_mask,
        }


def _make_readme(base_model_id: str, epoch: int) -> str:
    # HF frontmatter must be valid and base_model must not be empty
    return "\n".join(
        [
            "---",
            f"base_model: {base_model_id}",
            "library_name: peft",
            "license: apache-2.0",
            "tags:",
            "  - text-to-speech",
            "  - norwegian",
            "  - lora",
            "  - qwen3-tts",
            "---",
            f"# Qwen3-TTS Norwegian LoRA - Epoch {epoch}",
            "",
            "This checkpoint contains:",
            "- adapter_model.safetensors (LoRA adapter)",
            "- adapter_config.json",
            "- text_projection.bin (fine-tuned text projection layer)",
            "",
        ]
    ) + "\n"


def train():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", action="append", required=True, help="Can be repeated: --train_jsonl a --train_jsonl b")
    p.add_argument("--init_model_path", type=str, required=True)
    p.add_argument("--base_model_id", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="For README/HF metadata")
    p.add_argument("--output_model_path", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1)

    p.add_argument("--lora_lr", type=float, default=1e-5)
    p.add_argument("--text_proj_lr", type=float, default=5e-5)
    p.add_argument("--sub_loss_weight", type=float, default=0.2)

    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--train_mlp_lora", type=str, default="false", help="true/false")

    p.add_argument("--ref_seconds", type=float, default=6.0)
    p.add_argument("--max_audio_seconds", type=float, default=15.0)

    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))
    p.add_argument("--qwen_path", type=str, default="/workspace/Qwen3-TTS")

    args = p.parse_args()

    train_mlp_lora = str(args.train_mlp_lora).lower() in ("1", "true", "yes", "y")

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
        print(
            f"   lora_lr={args.lora_lr} text_proj_lr={args.text_proj_lr} sub_loss_weight={args.sub_loss_weight}\n"
            f"   lora_r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} train_mlp_lora={train_mlp_lora}\n"
            f"   mixed_precision={args.mixed_precision}"
        )

    set_seed(args.seed)

    # HF
    hf_api = None
    if args.hf_repo_id and accelerator.is_main_process:
        try:
            hf_api = HfApi()
            hf_api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
            print(f"☁️  HF Upload aktivert: {args.hf_repo_id}")
        except Exception as e:
            print(f"⚠️  HF init error: {e}")
            hf_api = None

    Qwen3TTSModel, mel_spectrogram_fn = _ensure_qwen_import(args.qwen_path)

    # Load model
    qwen_wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32,
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model = qwen_wrapper.model
    model.requires_grad_(False)

    # Config used by collator
    try:
        cfg = AutoConfig.from_pretrained(args.init_model_path)
    except Exception:
        cfg = AutoConfig.from_pretrained(args.base_model_id)

    # PEFT: protect against newer peft expecting prepare_inputs_for_generation
    if not hasattr(model.talker.model, "prepare_inputs_for_generation"):
        def _dummy_prepare_inputs_for_generation(*_a, **kw):
            return kw
        setattr(model.talker.model, "prepare_inputs_for_generation", _dummy_prepare_inputs_for_generation)

    # LoRA target modules
    attn_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_targets = ["gate_proj", "up_proj", "down_proj"] if train_mlp_lora else []
    target_modules = attn_targets + mlp_targets

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    model.talker.model = get_peft_model(model.talker.model, peft_config)

    # Unfreeze text_projection properly
    if hasattr(model.talker, "text_projection"):
        for p in model.talker.text_projection.parameters():
            p.requires_grad = True

    # Trainable params report
    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Trainable params: {trainable:,}")

    # Dataset & loader
    ds = NorwegianTTSDataset(
        jsonl_paths=args.train_jsonl,
        processor=qwen_wrapper.processor,
        config=cfg,
        mel_spectrogram_fn=mel_spectrogram_fn,
        ref_seconds=args.ref_seconds,
        max_audio_seconds=args.max_audio_seconds,
        sample_rate=24000,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ds.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Optimizer with param groups (LoRA vs text_projection)
    lora_params = []
    tp_params = []
    for n, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        if "text_projection" in n:
            tp_params.append(p_)
        else:
            lora_params.append(p_)

    optim = AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0},
            {"params": tp_params, "lr": args.text_proj_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    model, optim, dl = accelerator.prepare(model, optim, dl)
    model.train()

    global_step = 0
    for epoch in range(args.num_epochs):
        t0 = time.time()
        running = 0.0
        nsteps = 0

        for batch in dl:
            global_step += 1
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(model.device)
                codec_ids = batch["codec_ids"].to(model.device)
                ref_mels = batch["ref_mels"].to(model.device, dtype=model.dtype)

                text_mask = batch["text_embedding_mask"].to(model.device)
                codec_mask_embed = batch["codec_embedding_mask"].to(model.device)
                attn = batch["attention_mask"].to(model.device)
                labels = batch["codec_0_labels"].to(model.device)
                codec_mask = batch["codec_mask"].to(model.device)

                # Speaker embedding
                speaker_embedding = model.speaker_encoder(ref_mels).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                raw_text_embeds = model.talker.model.text_embedding(input_text_ids)
                proj_text = model.talker.text_projection(raw_text_embeds)
                text_emb = proj_text * text_mask

                codec_emb = model.talker.model.codec_embedding(input_codec_ids) * codec_mask_embed
                codec_emb[:, 6, :] = speaker_embedding  # speaker slot

                inputs_embeds = text_emb + codec_emb

                outputs = model.talker(
                    inputs_embeds=inputs_embeds[:, :-1, :],
                    attention_mask=attn[:, :-1],
                    labels=labels[:, 1:],
                    output_hidden_states=True,
                )

                loss_main = outputs.loss
                loss = loss_main

                # Sub-talker loss best-effort
                if args.sub_loss_weight > 0 and hasattr(model.talker, "forward_sub_talker_finetune"):
                    hs_ok = getattr(outputs, "hidden_states", None)
                    if hs_ok is not None and len(hs_ok) > 0 and hs_ok[-1] is not None:
                        last_hidden = hs_ok[-1]  # (B, T-1, H)
                        active_mask = codec_mask[:, :-1]  # (B, T-1)
                        if active_mask.any():
                            hs = last_hidden[active_mask]              # (N, H)
                            codes = codec_ids[:, :-1, :][active_mask]  # (N, 16)
                            try:
                                _, sub_loss = model.talker.forward_sub_talker_finetune(
                                    codec_ids=codes,
                                    talker_hidden_states=hs,
                                )
                                loss = loss_main + args.sub_loss_weight * sub_loss
                            except Exception:
                                pass  # keep training on main loss only

                accelerator.backward(loss)
                optim.step()
                optim.zero_grad()

            running += float(loss.detach().item())
            nsteps += 1

            if accelerator.is_main_process and global_step % 25 == 0:
                avg = running / max(1, nsteps)
                print(f"step={global_step} | loss={avg:.4f}")

        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1}/{args.num_epochs} done ({time.time()-t0:.1f}s)")

        # Save & upload
        if (epoch + 1) % args.save_every == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_model_path, f"epoch-{epoch+1}")
                os.makedirs(save_path, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)

                # Save adapter
                unwrapped.talker.model.save_pretrained(save_path, safe_serialization=True)

                # Save text_projection
                if hasattr(unwrapped.talker, "text_projection"):
                    torch.save(
                        unwrapped.talker.text_projection.state_dict(),
                        os.path.join(save_path, "text_projection.bin"),
                    )

                # Save meta
                meta = TrainMeta(
                    base_model_id=args.base_model_id,
                    init_model_path=args.init_model_path,
                    train_jsonls=args.train_jsonl,
                    batch_size=args.batch_size,
                    grad_accum=args.grad_accum,
                    num_epochs=args.num_epochs,
                    save_every=args.save_every,
                    lora_lr=args.lora_lr,
                    text_proj_lr=args.text_proj_lr,
                    sub_loss_weight=args.sub_loss_weight,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    train_mlp_lora=train_mlp_lora,
                    ref_seconds=args.ref_seconds,
                    max_audio_seconds=args.max_audio_seconds,
                    mixed_precision=args.mixed_precision,
                    seed=args.seed,
                )
                with open(os.path.join(save_path, "train_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(asdict(meta), f, indent=2, ensure_ascii=False)

                # HF-safe README
                with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write(_make_readme(args.base_model_id, epoch + 1))

                print(f"💾 Lagret: {save_path}")

                if hf_api and args.hf_repo_id:
                    try:
                        hf_api.upload_folder(
                            folder_path=save_path,
                            repo_id=args.hf_repo_id,
                            repo_type="model",
                            path_in_repo=f"checkpoints/epoch_{epoch+1}",
                        )
                        print("☁️  HF Upload OK")
                    except Exception as e:
                        print(f"⚠️  HF Upload Feil: {e}")


if __name__ == "__main__":
    train()
