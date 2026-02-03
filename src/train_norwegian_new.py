#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-Tuning (Scratch / Stable / Language-focused)
====================================================================
Mål: Bedre norsk uttale (språk/uttale) uten å ødelegge audio (skurring).

Key ideas:
- LoRA kun på attention-proj (q/k/v/o) -> mindre drift/støy enn å trene MLP også
- Separate LR: LoRA lav LR, text_projection litt høyere
- Sub-talker loss (codebooks 2-16) for å stabilisere detalj-kodene
- Ref-audio tvinges til fast lengde (pad/crop) @ 24kHz -> stabil speaker embedding

Kjøring (Librivox-only):
  accelerate launch src/train_norwegian_scratch_opt.py \
    --train_jsonls /workspace/data/train_with_codes_librivox.jsonl \
    --init_model_path /workspace/base_model \
    --output_model_path /workspace/output/run_nb_librivox \
    --batch_size 6 \
    --num_epochs 20 \
    --save_every 1 \
    --hf_repo_id telvenes/qwen3-tts-norsk-finetune

Kjøring (Librivox + NPSC, balansert):
  accelerate launch src/train_norwegian_scratch_opt.py \
    --train_jsonls /workspace/data/train_with_codes_librivox.jsonl /workspace/data/train_with_codes_npsc.jsonl \
    --balanced_sampling \
    --init_model_path /workspace/base_model \
    --output_model_path /workspace/output/run_nb_mix \
    --batch_size 6 \
    --num_epochs 20
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Tuple, Union, Optional, Dict

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import AutoConfig
try:
    from transformers import get_cosine_schedule_with_warmup
except Exception:
    get_cosine_schedule_with_warmup = None

from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi


# -------------------------
# Qwen3-TTS imports (local repo)
# -------------------------
def ensure_qwen_import(qwen_repo_path: str):
    if qwen_repo_path and os.path.isdir(qwen_repo_path) and qwen_repo_path not in sys.path:
        sys.path.append(qwen_repo_path)

@dataclass
class TrainMeta:
    base_model: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_lr: float
    text_proj_lr: float
    sub_loss_weight: float
    ref_audio_sec: float
    seed: int
    num_epochs: int


# -------------------------
# Dataset
# -------------------------
class MultiJsonlNorwegianTTSDataset(Dataset):
    """
    Leser én eller flere jsonl og concat'er dem.
    Støtter balanced sampling via weights uten å endre __getitem__.
    """

    def __init__(
        self,
        jsonl_paths: List[str],
        processor,
        config,
        ref_audio_sec: float = 6.0,
        max_audio_sec: float = 15.0,
        seed: int = 1234,
    ):
        self.processor = processor
        self.config = config

        self.sr = 24000
        self.ref_audio_len = int(self.sr * ref_audio_sec)
        self.max_audio_len = int(self.sr * max_audio_sec)

        self.rng = np.random.default_rng(seed)

        self.items: List[Dict] = []
        self.source_ids: List[int] = []  # hvilket jsonl et item kom fra (0..n-1)

        for si, p in enumerate(jsonl_paths):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                        self.items.append(ex)
                        self.source_ids.append(si)
                    except Exception:
                        continue

        if len(self.items) == 0:
            raise ValueError("Fant 0 eksempler i train_jsonls. Sjekk filene.")

        print(f"✅ Lastet {len(self.items)} eksempler fra {len(jsonl_paths)} datasett.")

    def __len__(self):
        return len(self.items)

    def _build_assistant_text(self, text: str) -> str:
        # samme format som du har brukt
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize(self, text: str) -> torch.Tensor:
        out = self.processor(text=text, return_tensors="pt", padding=True)
        ids = out["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    def _load_ref_audio_fixed(self, path: str) -> np.ndarray:
        """
        Last @ 24kHz, og gjør ref-audio FAST LENGDE:
        - hvis lang: random crop (augment / mindre overfitting)
        - hvis kort: pad med nuller
        """
        audio, _sr = librosa.load(path, sr=self.sr, mono=True)

        # hard cap (unngå ekstremt lange filer i minnet)
        if audio.shape[0] > self.max_audio_len:
            audio = audio[: self.max_audio_len]

        if audio.shape[0] >= self.ref_audio_len:
            start = int(self.rng.integers(0, audio.shape[0] - self.ref_audio_len + 1))
            audio = audio[start : start + self.ref_audio_len]
        else:
            pad = self.ref_audio_len - audio.shape[0]
            audio = np.pad(audio, (0, pad), mode="constant")

        return audio.astype(np.float32)

    @torch.inference_mode()
    def extract_mels(self, audio_24k: np.ndarray) -> torch.Tensor:
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        mels = mel_spectrogram(
            torch.from_numpy(audio_24k).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)  # (1, T, 128)
        return mels

    def __getitem__(self, idx: int):
        item = self.items[idx]

        ref_path = item.get("ref_audio") or item.get("ref_audio_path") or item.get("audio_path")
        if not ref_path:
            # fallback: hopp til neste
            return self.__getitem__((idx + 1) % len(self.items))

        text = item.get("text", "")
        if not isinstance(text, str) or len(text.strip()) == 0:
            return self.__getitem__((idx + 1) % len(self.items))

        codes = item.get("audio_codes", None)
        if codes is None:
            raise ValueError("Mangler audio_codes i jsonl. Du må generere audio_codes først.")

        audio_codes = torch.tensor(codes, dtype=torch.long)
        if audio_codes.dim() != 2 or audio_codes.shape[1] != 16:
            # feil format -> hopp
            return self.__getitem__((idx + 1) % len(self.items))

        # tokenize
        prompt = self._build_assistant_text(text)
        text_ids = self._tokenize(prompt)

        # ref mel
        wav = self._load_ref_audio_fixed(ref_path)
        ref_mel = self.extract_mels(wav)

        # behold kompat med ditt tidligere oppsett
        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
            "source_id": self.source_ids[idx],
        }

    def collate_fn(self, batch: List[Dict]):
        # fjern evt None
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            raise RuntimeError("Tom batch etter filtrering.")

        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 8
        B, T = len(batch), max_length

        input_ids = torch.zeros((B, T, 2), dtype=torch.long)
        codec_ids = torch.zeros((B, T, 16), dtype=torch.long)

        text_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_mask = torch.zeros((B, T), dtype=torch.bool)
        attention_mask = torch.zeros((B, T), dtype=torch.long)

        codec_0_labels = torch.full((B, T), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codes = data["audio_codes"]      # (N,16)
            audio_codec_0 = audio_codes[:, 0]      # (N,)

            text_len = text_ids.shape[1]
            codec_len = audio_codec_0.shape[0]

            # TEXT CHANNEL (0)
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_len - 2 : 8 + text_len + codec_len, 0] = self.config.tts_pad_token_id

            text_embedding_mask[i, : 8 + text_len + codec_len] = True

            # CODEC CHANNEL (1)
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

            input_ids[i, 8 : 8 + text_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_len - 2, 1] = self.config.talker_config.codec_bos_id

            start = 8 + text_len - 1
            input_ids[i, start : start + codec_len, 1] = audio_codec_0
            input_ids[i, start + codec_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, start : start + codec_len] = audio_codec_0
            codec_0_labels[i, start + codec_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, start : start + codec_len, :] = audio_codes

            codec_embedding_mask[i, 3 : 8 + text_len + codec_len] = True
            codec_embedding_mask[i, 6] = False  # speaker slot

            codec_mask[i, start : start + codec_len] = True
            attention_mask[i, : 8 + text_len + codec_len] = 1

        # ref_mel har nå fast lengde -> kan stackes
        ref_mels = torch.cat([b["ref_mel"] for b in batch], dim=0)  # (B, Tmel, 128)

        return {
            "input_ids": input_ids,
            "codec_ids": codec_ids,
            "codec_0_labels": codec_0_labels,
            "codec_mask": codec_mask,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "ref_mels": ref_mels,
        }


# -------------------------
# Outputs hidden state extraction (Qwen3-TTS quirk)
# -------------------------
def extract_last_hidden(outputs) -> Optional[torch.Tensor]:
    """
    Qwen3TTSTalkerOutputWithPast.hidden_states = (decoder_hidden_states, codec_ids)
    decoder_hidden_states er tuple(layer0,...,last).
    """
    hs = getattr(outputs, "hidden_states", None)
    if hs is None:
        return None

    # forventet: (decoder_hidden_states, codec_ids)
    if isinstance(hs, (tuple, list)) and len(hs) == 2:
        decoder_hs = hs[0]
    else:
        decoder_hs = hs

    if decoder_hs is None:
        return None

    if isinstance(decoder_hs, (tuple, list)) and len(decoder_hs) > 0 and torch.is_tensor(decoder_hs[-1]):
        return decoder_hs[-1]

    if torch.is_tensor(decoder_hs):
        return decoder_hs

    return None


# -------------------------
# Main train
# -------------------------
def train():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_jsonls", type=str, nargs="+", required=True,
                        help="Én eller flere jsonl med audio_codes (Librivox alene eller Librivox + NPSC).")
    parser.add_argument("--balanced_sampling", action="store_true",
                        help="Hvis flere datasett: sample dem mer jevnt (anbefalt hvis NPSC er mye større).")

    # model paths
    parser.add_argument("--init_model_path", type=str, required=True,
                        help="Local base model path eller HF id")
    parser.add_argument("--output_model_path", type=str, required=True)

    # training
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)

    # LR / stability
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--text_proj_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--sub_loss_weight", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.10)
    parser.add_argument("--train_mlp_lora", action="store_true",
                        help="Hvis du vil: inkluder gate/up/down LoRA også (mer effekt, mer risiko for skurring).")

    # audio conditioning
    parser.add_argument("--ref_audio_sec", type=float, default=6.0,
                        help="Fast lengde for reference audio til speaker embedding (sekunder).")

    # perf / runtime
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_every", type=int, default=25)

    # scheduler
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Brukes hvis transformers scheduler finnes. 0 = ingen warmup.")
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Slå på cosine scheduler med warmup hvis tilgjengelig.")

    # precision
    parser.add_argument("--mixed_precision", type=str, default=None, choices=[None, "no", "fp16", "bf16"],
                        help="Overstyr Accelerator mixed_precision. Default: bf16 på CUDA, ellers no.")

    # hub
    parser.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))
    args = parser.parse_args()

    # mixed precision default
    if args.mixed_precision is None:
        args.mixed_precision = "bf16" if torch.cuda.is_available() else "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_model_path,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_model_path, exist_ok=True)
        print("🚀 Starter norsk TTS-trening (Scratch / Stable)")
        print(f"   datasets: {args.train_jsonls}")
        print(f"   batch_size={args.batch_size} grad_accum={args.grad_accum} epochs={args.num_epochs}")
        print(f"   lora_lr={args.lora_lr} text_proj_lr={args.text_proj_lr} sub_loss_weight={args.sub_loss_weight}")
        print(f"   lora_r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} train_mlp_lora={args.train_mlp_lora}")
        print(f"   mixed_precision={args.mixed_precision}")

    set_seed(args.seed)

    # Qwen import path
    qwen_repo_path = os.environ.get("QWEN_TTS_PATH", "/workspace/Qwen3-TTS")
    ensure_qwen_import(qwen_repo_path)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    # HF upload
    hf_api = HfApi() if (args.hf_repo_id and accelerator.is_main_process) else None
    if hf_api and accelerator.is_main_process:
        print(f"☁️  HF Upload aktivert: {args.hf_repo_id}")

    # Load model
    qwen_wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        dtype=torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32),
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model = qwen_wrapper.model
    model.requires_grad_(False)

    # LoRA target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.train_mlp_lora:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    # IMPORTANT: FEATURE_EXTRACTION (Qwen talker mangler prepare_inputs_for_generation)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION",
    )
    model.talker.model = get_peft_model(model.talker.model, peft_config)

    # Unfreeze text_projection
    if hasattr(model.talker, "text_projection"):
        for p in model.talker.text_projection.parameters():
            p.requires_grad = True

    # Print trainables
    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Trainable params: {trainable:,}")
        # hvis peft har helper:
        try:
            model.talker.model.print_trainable_parameters()
        except Exception:
            pass

    # Config for special token IDs
    try:
        config = AutoConfig.from_pretrained(args.init_model_path)
    except Exception:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    # Dataset
    dataset = MultiJsonlNorwegianTTSDataset(
        jsonl_paths=args.train_jsonls,
        processor=qwen_wrapper.processor,
        config=config,
        ref_audio_sec=args.ref_audio_sec,
        seed=args.seed,
    )

    # Balanced sampling (valgfritt)
    sampler = None
    if args.balanced_sampling and len(args.train_jsonls) > 1:
        # weight per item: 1 / count_in_source
        counts = {}
        for sid in dataset.source_ids:
            counts[sid] = counts.get(sid, 0) + 1
        weights = [1.0 / counts[sid] for sid in dataset.source_ids]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        if accelerator.is_main_process:
            print("⚖️ balanced_sampling aktiv: hver kilde får mer lik påvirkning per epoch.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer param groups (LoRA vs text_projection)
    tp_params = []
    if hasattr(model.talker, "text_projection"):
        tp_params = list(model.talker.text_projection.parameters())

    # LoRA params = alt trainable uten text_projection
    lora_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and ("talker.text_projection" not in name):
            lora_params.append(p)

    optimizer = AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": args.weight_decay},
            {"params": tp_params, "lr": args.text_proj_lr, "weight_decay": 0.0},
        ],
        lr=args.lora_lr,
    )

    # Scheduler (valgfritt)
    scheduler = None
    if args.use_scheduler and (get_cosine_schedule_with_warmup is not None):
        # ant optimizer steps
        steps_per_epoch = int(np.ceil(len(dataloader) / args.grad_accum))
        total_steps = steps_per_epoch * args.num_epochs
        warmup_steps = int(total_steps * max(0.0, args.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        if accelerator.is_main_process:
            print(f"📉 Scheduler: cosine | warmup_steps={warmup_steps} total_steps={total_steps}")
    elif args.use_scheduler and accelerator.is_main_process:
        print("⚠️ Scheduler requested, men transformers scheduler ikke tilgjengelig. Fortsetter uten scheduler.")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # Train loop
    global_step = 0
    model.train()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_main = 0.0
        epoch_sub = 0.0
        steps = 0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(model.device)
                codec_ids = batch["codec_ids"].to(model.device)
                ref_mels = batch["ref_mels"].to(model.device, dtype=model.dtype)

                text_embedding_mask = batch["text_embedding_mask"].to(model.device)
                codec_embedding_mask = batch["codec_embedding_mask"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                codec_0_labels = batch["codec_0_labels"].to(model.device)
                codec_mask = batch["codec_mask"].to(model.device)

                # speaker embedding
                speaker_embedding = model.speaker_encoder(ref_mels).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # text embedding -> projection (trainable)
                raw_text = model.talker.model.text_embedding(input_text_ids)
                proj_text = model.talker.text_projection(raw_text)
                text_emb = proj_text * text_embedding_mask

                # codec embedding + speaker injection
                codec_emb = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                codec_emb[:, 6, :] = speaker_embedding

                inputs_embeds = text_emb + codec_emb

                outputs = model.talker(
                    inputs_embeds=inputs_embeds[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],       # shift
                    output_hidden_states=True,
                )

                loss_main = outputs.loss
                loss = loss_main
                sub_loss_val = torch.tensor(0.0, device=model.device)

                last_hidden = extract_last_hidden(outputs)  # (B, L, H), L=T-1

                if last_hidden is not None:
                    active_mask = codec_mask[:, 1:]  # align med labels-shift, shape (B, L)
                    if active_mask.any():
                        hs = last_hidden[active_mask]             # (N, H)
                        codes = codec_ids[:, 1:, :][active_mask]  # (N, 16)

                        _, sub_loss = model.talker.forward_sub_talker_finetune(
                            codec_ids=codes,
                            talker_hidden_states=hs,
                        )

                        if sub_loss is not None:
                            sub_loss_val = sub_loss
                            loss = loss_main + args.sub_loss_weight * sub_loss

                # safety
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    if accelerator.is_main_process:
                        print("❌ NaN/Inf loss oppdaget. Avbryter for å unngå ødelagt checkpoint.")
                    raise RuntimeError("NaN/Inf loss")

                accelerator.backward(loss)

                # grad clipping (stabilitet)
                if args.max_grad_norm > 0:
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                # logging
                epoch_loss += float(loss.detach().item())
                epoch_main += float(loss_main.detach().item())
                epoch_sub += float(sub_loss_val.detach().item()) if sub_loss_val is not None else 0.0
                steps += 1
                global_step += 1

                if accelerator.is_main_process and (global_step % args.log_every == 0):
                    lr0 = optimizer.param_groups[0]["lr"]
                    lr1 = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr0
                    print(
                        f"step={global_step} | loss={loss.item():.4f} "
                        f"(main={loss_main.item():.4f}, sub={float(sub_loss_val):.4f}) | "
                        f"lr_lora={lr0:.2e} lr_tp={lr1:.2e}"
                    )

        avg_loss = epoch_loss / max(1, steps)
        avg_main = epoch_main / max(1, steps)
        avg_sub = epoch_sub / max(1, steps)

        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1}/{args.num_epochs} | loss={avg_loss:.4f} main={avg_main:.4f} sub={avg_sub:.4f}")

        # save
        if accelerator.is_main_process and ((epoch + 1) % args.save_every == 0):
            save_path = os.path.join(args.output_model_path, f"epoch-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)

            # save LoRA adapter
            unwrapped.talker.model.save_pretrained(save_path)

            # save text_projection
            if hasattr(unwrapped.talker, "text_projection"):
                torch.save(unwrapped.talker.text_projection.state_dict(), os.path.join(save_path, "text_projection.bin"))

            # save metadata
            meta = TrainMeta(
                base_model=str(args.init_model_path),
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_lr=args.lora_lr,
                text_proj_lr=args.text_proj_lr,
                sub_loss_weight=args.sub_loss_weight,
                ref_audio_sec=args.ref_audio_sec,
                seed=args.seed,
                num_epochs=args.num_epochs,
            )
            with open(os.path.join(save_path, "train_meta.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

            with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                f.write(
                    "\n".join(
                        [
                            "---",
                            "library_name: peft",
                            "tags: [text-to-speech, norwegian, lora, qwen3-tts]",
                            "---",
                            f"# Qwen3-TTS Norwegian (Scratch Stable) - Epoch {epoch+1}",
                            "",
                            f"- datasets: {args.train_jsonls}",
                            f"- balanced_sampling: {args.balanced_sampling}",
                            f"- lora target_modules: {target_modules}",
                            f"- lora_lr: {args.lora_lr}",
                            f"- text_proj_lr: {args.text_proj_lr}",
                            f"- sub_loss_weight: {args.sub_loss_weight}",
                            f"- ref_audio_sec: {args.ref_audio_sec}",
                            "",
                            "## Load",
                            "Load adapter into model.talker.model, and load text_projection.bin into model.talker.text_projection.",
                        ]
                    )
                )

            print(f"💾 Lagret: {save_path}")

            # HF upload
            if hf_api:
                try:
                    hf_api.upload_folder(
                        folder_path=save_path,
                        repo_id=args.hf_repo_id,
                        path_in_repo=f"checkpoints/epoch_{epoch+1}",
                        repo_type="model",
                    )
                    print("☁️ HF Upload OK")
                except Exception as e:
                    print(f"⚠️ HF Upload Feil: {e}")


if __name__ == "__main__":
    train()
