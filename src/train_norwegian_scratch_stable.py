#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-tuning (Scratch/Stable) - V2
- Fix: pad ref_mels in collate to avoid torch.cat/stack crashes.
- PEFT shim: add prepare_inputs_for_generation if missing.
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

        # cap length for speaker encoder stability
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
            codec_embedding_mask[i, 6] = False
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
