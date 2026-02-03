#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare audio_codes using Qwen3-TTS-Tokenizer-12Hz
==================================================
Input : train_raw_*.jsonl (audio, text, ref_audio)
Output: train_with_codes_*.jsonl (same + audio_codes)

Example:
  python3 src/prepare_codes_12hz.py \
    --in_jsonl /workspace/data/train_raw_librivox.jsonl \
    --out_jsonl /workspace/data/train_with_codes_librivox.jsonl \
    --qwen_path /workspace/Qwen3-TTS \
    --tokenizer_model Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --device_map cuda:0 \
    --batch_size 8
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch

# Optional, but helps
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--qwen_path", type=str, default="/workspace/Qwen3-TTS")
    ap.add_argument("--tokenizer_model", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    ap.add_argument("--device_map", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--skip_if_has_codes", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.qwen_path):
        sys.path.append(args.qwen_path)

    try:
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    except Exception as e:
        print("❌ Klarte ikke importere Qwen3TTSTokenizer.")
        print("   Sjekk at --qwen_path peker på Qwen3-TTS repo, eller at qwen-tts er installert.")
        raise

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"🔤 Laster tokenizer: {args.tokenizer_model}")
    tok = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model,
        device_map=args.device_map,
        torch_dtype=dtype,
    )

    # Load input rows
    rows: List[Dict[str, Any]] = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    print(f"🎛️  Encoding audio_codes for {len(rows)} rows | batch_size={args.batch_size} | device={args.device_map}")
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        batch_paths: List[str] = []
        batch_rows: List[Dict[str, Any]] = []

        def flush():
            nonlocal batch_paths, batch_rows
            if not batch_rows:
                return

            enc = tok.encode(batch_paths, return_dict=True)
            codes_list = enc.audio_codes  # List[Tensor] each (T, 16) for 12Hz :contentReference[oaicite:2]{index=2}

            for r, codes in zip(batch_rows, codes_list):
                r2 = dict(r)
                r2["audio_codes"] = codes.detach().to("cpu", dtype=torch.long).numpy().tolist()
                out.write(json.dumps(r2, ensure_ascii=False) + "\n")

            batch_paths = []
            batch_rows = []

        for r in tqdm(rows):
            if args.skip_if_has_codes and "audio_codes" in r and r["audio_codes"]:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue

            apath = r.get("audio") or r.get("audio_path")
            if not apath or not os.path.exists(apath):
                continue

            batch_paths.append(apath)
            batch_rows.append(r)

            if len(batch_rows) >= args.batch_size:
                flush()

        flush()

    print(f"✅ Ferdig: {args.out_jsonl}")


if __name__ == "__main__":
    main()
