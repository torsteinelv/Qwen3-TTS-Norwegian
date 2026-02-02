#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-Tuning (RUN FIX - PEFT COMPAT)
======================================================
Fixer:
1) PEFT crash: bruker task_type="FEATURE_EXTRACTION" (Qwen talker mangler prepare_inputs_for_generation)
2) Sub-talker loss: riktig hidden-state extraction + riktig shift
3) text_projection trenes faktisk
4) 24 kHz audio (stabil speaker embedding)

Kjøring:
  accelerate launch src/train_norwegian_fixed_run.py \
    --train_jsonl /workspace/data/train_with_codes.jsonl \
    --init_model_path /workspace/base_model \
    --output_model_path /workspace/output/run_long \
    --batch_size 6 \
    --num_epochs 100 \
    --lr 1e-5 \
    --sub_loss_weight 0.25 \
    --save_every 1 \
    --hf_repo_id telvenes/qwen3-tts-norsk-finetune
"""

import argparse
import json
import os
import sys
from typing import List, Tuple, Union, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from accelerate import Accelerator
from transformers import AutoConfig
from huggingface_hub import HfApi

from peft import LoraConfig, get_peft_model


# =========================
# 0) IMPORT QWEN3-TTS
# =========================
sys.path.append("/workspace/Qwen3-TTS")
try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
except ImportError:
    print("⚠️ Kunne ikke importere Qwen3TTSModel. Sjekk at /workspace/Qwen3-TTS finnes.")
    raise


AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]


# =========================
# 1) DATASET
# =========================
class NorwegianTTSDataset(Dataset):
    def __init__(self, jsonl_path: str, processor, config, max_audio_sec: float = 15.0):
        self.processor = processor
        self.config = config
        self.max_audio_samples = int(24000 * max_audio_sec)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f]
        print(f"✅ Lastet {len(self.items)} eksempler.")

    def __len__(self):
        return len(self.items)

    def _load_audio_24k(self, path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(path, sr=24000, mono=True)  # tving 24k
            if audio.shape[0] > self.max_audio_samples:
                audio = audio[: self.max_audio_samples]
            return audio.astype(np.float32), int(sr)
        except Exception as e:
            print(f"⚠️ Audio load error ({path}): {e}")
            return np.zeros(24000, dtype=np.float32), 24000

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _tokenize_text(self, text: str) -> torch.Tensor:
        out = self.processor(text=text, return_tensors="pt", padding=True)
        ids = out["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray) -> torch.Tensor:
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

    def __getitem__(self, idx: int):
        item = self.items[idx]

        ref_path = item.get("ref_audio") or item.get("ref_audio_path") or item.get("audio_path")
        if not ref_path:
            return self.__getitem__((idx + 1) % len(self.items))

        text = self._build_assistant_text(item["text"])
        text_ids = self._tokenize_text(text)

        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)  # (N, 16)

        wav, _sr = self._load_audio_24k(ref_path)
        ref_mel = self.extract_mels(wav)

        return {
            "text_ids": text_ids[:, :-5],  # behold for kompat
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch):
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
            audio_codes = data["audio_codes"]       # (N,16)
            audio_codec_0 = audio_codes[:, 0]       # (N,)

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

        ref_mels = torch.cat([b["ref_mel"] for b in batch], dim=0)

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


# =========================
# 2) HELPERS
# =========================
def extract_last_hidden(outputs) -> Optional[torch.Tensor]:
    """
    Qwen3TTSTalkerOutputWithPast.hidden_states = (decoder_hidden_states, codec_ids)
    decoder_hidden_states er tuple(layer0,...,last). Vi vil ha last.
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


# =========================
# 3) TRAIN
# =========================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, default="./qwen3-tts-norwegian-lora")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--sub_loss_weight", type=float, default=0.25)
    parser.add_argument("--max_audio_sec", type=float, default=15.0)
    parser.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_model_path, exist_ok=True)
        print("🚀 Starter norsk TTS-trening (PEFT COMPAT)")
        print(f"   LR: {args.lr} | sub_loss_weight={args.sub_loss_weight}")
        if args.hf_repo_id:
            print(f"☁️  HF Upload: {args.hf_repo_id}")

    hf_api = HfApi() if (args.hf_repo_id and accelerator.is_main_process) else None

    qwen_wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        dtype=torch.bfloat16,
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model = qwen_wrapper.model
    model.requires_grad_(False)

    # ✅ CRITICAL FIX: FEATURE_EXTRACTION så PEFT ikke krever prepare_inputs_for_generation
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    model.talker.model = get_peft_model(model.talker.model, peft_config)

    # Unfreeze text_projection (språk/uttale)
    if hasattr(model.talker, "text_projection"):
        if accelerator.is_main_process:
            print("🔓 Unfreezing text_projection parameters...")
        for p in model.talker.text_projection.parameters():
            p.requires_grad = True

    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Trainable params: {trainable:,}")

    # config
    try:
        config = AutoConfig.from_pretrained(args.init_model_path)
    except Exception:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    dataset = NorwegianTTSDataset(args.train_jsonl, qwen_wrapper.processor, config, max_audio_sec=args.max_audio_sec)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    warned_once = False
    model.train()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        n_steps = 0

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

                speaker_embedding = model.speaker_encoder(ref_mels).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # Text embeds
                raw_text = model.talker.model.text_embedding(input_text_ids)
                proj_text = model.talker.text_projection(raw_text)
                text_emb = proj_text * text_embedding_mask

                # Codec embeds + speaker injection
                codec_emb = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                codec_emb[:, 6, :] = speaker_embedding

                inputs_embeds = text_emb + codec_emb

                # Main loss: codec_0
                outputs = model.talker(
                    inputs_embeds=inputs_embeds[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],       # shift
                    output_hidden_states=True,
                )

                loss_main = outputs.loss
                loss = loss_main

                # Sub-talker loss (code groups 2-16)
                last_hidden = extract_last_hidden(outputs)  # (B, L, H)

                if last_hidden is None:
                    if (not warned_once) and accelerator.is_main_process:
                        print("⚠️ Hidden states ikke tilgjengelig -> sub-talker loss hoppes over.")
                        warned_once = True
                else:
                    active_mask = codec_mask[:, 1:]  # align med labels-shift (B, L)
                    if active_mask.any().item():
                        hs = last_hidden[active_mask]               # (N, H)
                        codes = codec_ids[:, 1:, :][active_mask]    # (N, 16)

                        _, sub_loss = model.talker.forward_sub_talker_finetune(
                            codec_ids=codes,
                            talker_hidden_states=hs,
                        )
                        loss = loss_main + args.sub_loss_weight * sub_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += float(loss.detach().item())
                n_steps += 1

        avg_loss = epoch_loss / max(1, n_steps)
        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f}")

            if (epoch + 1) % args.save_every == 0:
                save_path = os.path.join(args.output_model_path, f"epoch-{epoch+1}")
                os.makedirs(save_path, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)

                # Lagre LoRA adapter (talker.model)
                unwrapped.talker.model.save_pretrained(save_path)

                # Lagre text_projection
                if hasattr(unwrapped.talker, "text_projection"):
                    torch.save(
                        unwrapped.talker.text_projection.state_dict(),
                        os.path.join(save_path, "text_projection.bin"),
                    )

                with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write(
                        "\n".join(
                            [
                                "---",
                                "base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                "library_name: peft",
                                f"tags: [text-to-speech, norwegian, lora, epoch-{epoch+1}]",
                                "---",
                                f"# Qwen3-TTS Norwegian - Epoch {epoch+1}",
                                f"- lr: {args.lr}",
                                f"- sub_loss_weight: {args.sub_loss_weight}",
                                "- includes: sub-talker loss + text_projection training",
                            ]
                        )
                    )

                print(f"💾 Lagret: {save_path}")

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
