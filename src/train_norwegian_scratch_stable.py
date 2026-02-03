#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-tuning (Scratch/Stable) - V2
=====================================================
Fixes DataLoader crash by padding ref_mels to same length in batch.

- Supports multiple --train_jsonl (repeat flag)
- LoRA on attention (optionally MLP)
- Trains text_projection (critical for language shift)
- Best-effort sub-talker loss (skips if hidden_states missing)

Run (example):
accelerate launch --num_processes 1 /workspace/src/train_norwegian_scratch_stable.py \
  --train_jsonl /workspace/data/train_with_codes_librivox.jsonl \
  --init_model_path /workspace/base_model \
  --output_model_path /workspace/output/run_scratch_v1 \
  --batch_size 4 --grad_accum 4 --num_epochs 10 --save_every 1 \
  --lora_lr 1e-5 --text_proj_lr 5e-5 --sub_loss_weight 0.2 \
  --lora_r 4 --lora_alpha 8 --lora_dropout 0.1 --train_mlp_lora false \
  --mixed_precision bf16 \
  --hf_repo_id telvenes/qwen3-tts-norsk-finetune
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

# -------------------------
# Qwen imports
# -------------------------
sys.path.append("/workspace/Qwen3-TTS")
try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
except ImportError:
    print("⚠️ Kunne ikke importere Qwen3TTSModel. Sjekk /workspace/Qwen3-TTS.")
    raise

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]


# ============================================================
# Dataset
# ============================================================
class NorwegianTTSDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: List[str],
        processor,
        config,
        max_ref_seconds: float = 12.0,   # keep speaker embed stable
        max_audio_secon
