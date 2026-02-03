#!/usr/bin/env python3
import os
import json
import argparse
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


TARGET_SR_DEFAULT = 24000


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("<hesitation>", "")
    t = t.replace("\u00a0", " ")
    t = " ".join(t.split()).strip()
    return t


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def decode_audio(sample_audio) -> tuple[np.ndarray, int]:
    """
    Handles common HF streaming audio formats:
    - {"array": ..., "sampling_rate": ...}
    - {"path": "..."} (sometimes)
    """
    if isinstance(sample_audio, dict):
        if "array" in sample_audio and sample_audio["array"] is not None:
            y = np.asarray(sample_audio["array"], dtype=np.float32)
            sr = int(sample_audio.get("sampling_rate", TARGET_SR_DEFAULT))
            # downmix if needed
            if y.ndim == 2:
                y = np.mean(y, axis=1).astype(np.float32)
            return y, sr

        if "path" in sample_audio and sample_audio["path"]:
            path = sample_audio["path"]
            y, sr = librosa.load(path, sr=None, mono=True)
            return y.astype(np.float32), int(sr)

    raise ValueError("Unsupported audio format in sample['audio'].")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--hf_dataset", type=str, default="NbAiLab/NPSC")
    ap.add_argument("--config", type=str, default="16K_mp3_bokmaal")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--streaming", action="store_true", help="Use streaming=True")
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_audio_dir", type=str, required=True)

    ap.add_argument("--target_sr", type=int, default=TARGET_SR_DEFAULT)
    ap.add_argument("--min_seconds", type=float, default=1.5)
    ap.add_argument("--max_seconds", type=float, default=15.0)

    ap.add_argument("--max_hours", type=float, default=0.0, help="0 = no hour limit")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = no sample limit")

    ap.add_argument("--text_field", type=str, default="text")

    # ref_audio options:
    #   fixed_first: first valid sample becomes ref for all
    #   same: ref_audio = audio (per sample)
    #   fixed_file: use a user-provided file as ref for all
    ap.add_argument("--ref_audio_strategy", type=str, default="fixed_first",
                    choices=["fixed_first", "same", "fixed_file"])
    ap.add_argument("--fixed_ref_audio", type=str, default=None)

    ap.add_argument("--skip_existing_wav", action="store_true",
                    help="Skip writing wav if it already exists (resume runs)")

    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out_jsonl) or ".")
    ensure_dir(args.out_audio_dir)

    if args.ref_audio_strategy == "fixed_file":
        if not args.fixed_ref_audio or not os.path.exists(args.fixed_ref_audio):
            raise FileNotFoundError("--ref_audio_strategy fixed_file requires --fixed_ref_audio to exist.")

    print(f"--- STARTER NEDLASTING AV {args.hf_dataset} ({args.config}) split={args.split} ---")
    print(f"streaming={args.streaming} trust_remote_code={args.trust_remote_code}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"out_audio_dir={args.out_audio_dir}")

    ds = load_dataset(
        args.hf_dataset,
        args.config,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code,
    )

    max_total_seconds = args.max_hours * 3600.0 if args.max_hours and args.max_hours > 0 else None
    total_seconds = 0.0
    written = 0
    seen = 0

    ref_audio_path = None
    if args.ref_audio_strategy == "fixed_file":
        ref_audio_path = os.path.abspath(args.fixed_ref_audio)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, desc="Building NPSC", unit="ex"):
            seen += 1

            if args.max_samples and args.max_samples > 0 and written >= args.max_samples:
                break

            # text
            text = sample.get(args.text_field, "")
            text = clean_text(text)
            if not text:
                continue

            # audio
            try:
                audio_obj = sample.get("audio", None)
                if audio_obj is None:
                    continue

                y, sr = decode_audio(audio_obj)
                duration = float(len(y) / sr)

                if duration < args.min_seconds or duration > args.max_seconds:
                    continue

                # resample to target_sr
                if sr != args.target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=args.target_sr).astype(np.float32)
                    sr = args.target_sr

                # save wav
                wav_name = f"npsc_{written:09d}.wav"
                wav_path = os.path.abspath(os.path.join(args.out_audio_dir, wav_name))

                if args.skip_existing_wav and os.path.exists(wav_path):
                    # assume it's correct
                    pass
                else:
                    y = np.clip(y, -1.0, 1.0)
                    sf.write(wav_path, y, sr, subtype="PCM_16")

                # set ref audio
                if args.ref_audio_strategy == "fixed_first":
                    if ref_audio_path is None:
                        ref_audio_path = wav_path
                        print(f"✅ Set reference audio to FIRST valid sample: {ref_audio_path}")
                elif args.ref_audio_strategy == "same":
                    ref_audio_path = wav_path

                if ref_audio_path is None:
                    # should not happen unless fixed_first but no valid sample yet
                    continue

                # JSONL entry (IMPORTANT: use key 'audio' for prepare_data.py compatibility)
                entry = {
                    "audio": wav_path,
                    "text": text,
                    "ref_audio": ref_audio_path,
                }

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
                total_seconds += float(len(y) / sr)

                if max_total_seconds is not None and total_seconds >= max_total_seconds:
                    break

            except Exception as e:
                # keep going on noisy/bad samples
                print(f"⚠️ Error on sample #{seen}: {e}")
                continue

    print(f"\n✅ Ferdig!")
    print(f"   wrote: {written} lines")
    print(f"   hours: {total_seconds/3600.0:.2f}")
    print(f"   jsonl: {args.out_jsonl}")


if __name__ == "__main__":
    main()
