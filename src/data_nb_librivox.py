#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build train_raw_librivox.jsonl (Norwegian LibriVox / NB)
========================================================
- Writes audio to WAV 24kHz mono (stable for Qwen3-TTS)
- Outputs JSONL with fields: audio, text, ref_audio, id, dataset
- Uses unique default output filename to avoid collisions.

Example:
  python3 src/data_nb_librivox.py \
    --out_jsonl /workspace/data/train_raw_librivox.jsonl \
    --out_audio_dir /workspace/data/audio_librivox \
    --max_hours 12 \
    --ref_audio_strategy same
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    from datasets import Audio, load_dataset
except Exception as e:
    print("❌ Mangler 'datasets'. Installer: pip install datasets soundfile librosa")
    raise

try:
    import librosa
except Exception:
    print("❌ Mangler 'librosa'. Installer: pip install librosa")
    raise


TEXT_CANDIDATES = ["text", "sentence", "transcript", "normalized_text", "raw_text"]


def norm_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def pick_text(ex: Dict[str, Any], preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in ex and ex[preferred]:
        return str(ex[preferred])
    for k in TEXT_CANDIDATES:
        if k in ex and ex[k]:
            return str(ex[k])
    return None


def ensure_24k_mono(wav: np.ndarray, sr: int, target_sr: int = 24000) -> Tuple[np.ndarray, int]:
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        wav = librosa.resample(y=wav, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
        sr = target_sr
    # Hard clamp just in case
    wav = np.clip(wav, -1.0, 1.0)
    return wav, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", type=str, default="NbAiLab/nb-librivox")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--streaming", action="store_true", help="Stream dataset (for very large sets).")
    ap.add_argument("--out_jsonl", type=str, default="/workspace/data/train_raw_librivox.jsonl")
    ap.add_argument("--out_audio_dir", type=str, default="/workspace/data/audio_librivox")
    ap.add_argument("--max_hours", type=float, default=0.0, help="0 = no limit")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = no limit")
    ap.add_argument("--min_seconds", type=float, default=0.6)
    ap.add_argument("--max_seconds", type=float, default=15.0)
    ap.add_argument("--text_field", type=str, default=None)

    ap.add_argument("--ref_audio_strategy", type=str, choices=["same", "fixed"], default="same",
                    help="same=ref_audio=audio (multi-speaker OK). fixed=use one ref for all (single-speaker clone).")
    ap.add_argument("--fixed_ref_audio", type=str, default=None)
    ap.add_argument("--skip_existing_wav", action="store_true")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(args.out_audio_dir, exist_ok=True)

    if args.ref_audio_strategy == "fixed" and not args.fixed_ref_audio:
        print("❌ ref_audio_strategy=fixed krever --fixed_ref_audio")
        sys.exit(1)

    print(f"📥 Laster dataset: {args.hf_dataset} split={args.hf_split} streaming={args.streaming}")
    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=args.streaming)

    # Hvis ikke streaming: cast audio til 24k for stabil skriving
    if not args.streaming:
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        except Exception:
            pass

    total_seconds = 0.0
    written = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if args.max_rows and written >= args.max_rows:
                break
            if args.max_hours and (total_seconds / 3600.0) >= args.max_hours:
                break

            text = pick_text(ex, preferred=args.text_field)
            if not text:
                continue
            text = norm_text(text)
            if len(text) < 2:
                continue

            audio_obj = ex.get("audio")
            if not audio_obj:
                continue

            wav = audio_obj["array"]
            sr = int(audio_obj["sampling_rate"])
            wav, sr = ensure_24k_mono(np.asarray(wav), sr, target_sr=24000)

            dur = float(len(wav)) / float(sr)
            if dur < args.min_seconds or dur > args.max_seconds:
                continue

            uid = f"librivox_{written:07d}"
            wav_path = os.path.join(args.out_audio_dir, f"{uid}.wav")

            if (not args.skip_existing_wav) or (not os.path.exists(wav_path)):
                sf.write(wav_path, wav, sr)

            ref_audio = wav_path if args.ref_audio_strategy == "same" else args.fixed_ref_audio

            row = {
                "id": uid,
                "dataset": "nb-librivox",
                "audio": wav_path,
                "text": text,
                "ref_audio": ref_audio,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            written += 1
            total_seconds += dur

            if written % 500 == 0:
                print(f"  ... {written} skrevet | ~{total_seconds/3600.0:.2f} timer")

    print(f"✅ Ferdig: {args.out_jsonl}")
    print(f"   Rader: {written}")
    print(f"   Timer: {total_seconds/3600.0:.2f}")


if __name__ == "__main__":
    main()
