#!/usr/bin/env python3
"""
Build NPSC JSONL dataset for Qwen3-TTS fine-tuning.

Outputs JSONL where each line contains at minimum:
  - "text": str
  - "audio_path": str (local WAV path, 24kHz mono)
  - "ref_audio": str (path to reference WAV)
Optionally:
  - "ref_text": str

Example line:
{"text":"hei ...","audio_path":"/workspace/data/npsc_wavs/npsc_00000001.wav","ref_audio":"/workspace/data/npsc_wavs/npsc_00000001.wav","ref_text":"hei ..."}

Notes:
- NPSC requires trust_remote_code=True in many environments (non-interactive containers).
- This script resamples audio to 24kHz to match Qwen3-TTS expectations.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


DEFAULT_REQUIRE_CHARS = (
    "abcdefghijklmnopqrstuvwxyzæøå"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ"
    " .,;:!?\"'()-—–…/\\"
    "\n\t"
    "0123456789"
)


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sanitize_text(text: str) -> str:
    # basic cleanup: collapse whitespace
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _allowed_chars_ok(text: str, allowed_chars: Optional[str]) -> bool:
    if not allowed_chars:
        return True
    allowed = set(allowed_chars)
    for ch in text:
        if ch not in allowed:
            return False
    return True


def _extract_audio_blob(example: Dict[str, Any]) -> Any:
    """
    Try common keys. NPSC typically uses 'audio' but we keep it robust.
    Returns whatever we find; later decode handles formats.
    """
    for k in ("audio", "speech", "wav", "sound"):
        if k in example and example[k] is not None:
            return example[k]
    return None


def _decode_to_wave(audio_obj: Any) -> Tuple[np.ndarray, int]:
    """
    audio_obj can be:
      - dict with 'array' and 'sampling_rate' (decoded HF Audio)
      - dict with 'path' (some streaming cases)
      - string path
    Returns (float32 waveform mono, sr).
    """
    if audio_obj is None:
        raise ValueError("No audio object found in example.")

    # HF decoded Audio typically: {"array": np.array, "sampling_rate": int, ...}
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            y = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj.get("sampling_rate", 24000))
            # if multi-channel, downmix
            if y.ndim == 2:
                y = np.mean(y, axis=1).astype(np.float32)
            return y, sr

        # sometimes provides a path
        if "path" in audio_obj and audio_obj["path"]:
            path = audio_obj["path"]
            y, sr = librosa.load(path, sr=None, mono=True)
            return y.astype(np.float32), int(sr)

    # raw path string
    if isinstance(audio_obj, str):
        y, sr = librosa.load(audio_obj, sr=None, mono=True)
        return y.astype(np.float32), int(sr)

    raise TypeError(f"Unsupported audio object type: {type(audio_obj)}")


def _resample_to_24k(y: np.ndarray, sr: int, target_sr: int = 24000) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def _write_wav_24k(out_path: str, y: np.ndarray, sr: int = 24000) -> float:
    # clip to safe range
    y = np.clip(y, -1.0, 1.0)
    sf.write(out_path, y, sr, subtype="PCM_16")
    return float(len(y) / sr)


def build_jsonl(
    hf_dataset: str,
    hf_split: str,
    streaming: bool,
    trust_remote_code: bool,
    out_jsonl: str,
    out_audio_dir: str,
    max_hours: float,
    max_rows: int,
    min_seconds: float,
    max_seconds: float,
    text_field: str,
    ref_audio_strategy: str,
    fixed_ref_audio: Optional[str],
    skip_existing_wav: bool,
    require_chars: Optional[str],
) -> None:
    _safe_mkdir(os.path.dirname(out_jsonl) or ".")
    _safe_mkdir(out_audio_dir)

    if ref_audio_strategy == "fixed":
        if not fixed_ref_audio or not os.path.exists(fixed_ref_audio):
            raise FileNotFoundError(
                "ref_audio_strategy=fixed requires --fixed_ref_audio pointing to an existing file."
            )

    ds = load_dataset(
        hf_dataset,
        split=hf_split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )

    # We stream-write JSONL to avoid memory spikes.
    total_target_seconds = max_hours * 3600.0 if max_hours and max_hours > 0 else None
    total_seconds = 0.0
    written = 0
    seen = 0

    # tqdm needs total; for streaming we don't know it.
    pbar = tqdm(total=None, desc="Building NPSC JSONL", unit="ex")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in ds:
            seen += 1
            pbar.update(1)

            if max_rows and max_rows > 0 and seen > max_rows:
                break

            text = ex.get(text_field, None)
            if not text or not isinstance(text, str):
                continue

            text = _sanitize_text(text)
            if not text:
                continue

            if not _allowed_chars_ok(text, require_chars):
                continue

            audio_obj = _extract_audio_blob(ex)
            if audio_obj is None:
                continue

            # deterministic filename (index-based) is fine; collisions avoided with seen counter
            wav_name = f"npsc_{seen:09d}.wav"
            wav_path = os.path.join(out_audio_dir, wav_name)

            # If file exists and we're skipping, we still need duration for hour cap.
            duration = None
            if skip_existing_wav and os.path.exists(wav_path):
                try:
                    info = sf.info(wav_path)
                    duration = float(info.frames / info.samplerate)
                except Exception:
                    duration = None

            if duration is None:
                try:
                    y, sr = _decode_to_wave(audio_obj)
                    y = _resample_to_24k(y, sr, 24000)
                    duration = float(len(y) / 24000.0)

                    # length filter before writing (saves time/space)
                    if min_seconds and duration < float(min_seconds):
                        continue
                    if max_seconds and duration > float(max_seconds):
                        continue

                    _write_wav_24k(wav_path, y, 24000)
                except Exception:
                    # skip bad samples
                    continue
            else:
                # if duration known from existing file, apply filters
                if min_seconds and duration < float(min_seconds):
                    continue
                if max_seconds and duration > float(max_seconds):
                    continue

            # ref audio choice
            if ref_audio_strategy == "same":
                ref_audio = wav_path
                ref_text = text
            else:
                ref_audio = fixed_ref_audio
                ref_text = None

            row = {
                "text": text,
                "audio_path": wav_path,
                "ref_audio": ref_audio,
            }
            if ref_text is not None:
                row["ref_text"] = ref_text

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            total_seconds += float(duration)

            # stop if we reached hour cap
            if total_target_seconds is not None and total_seconds >= total_target_seconds:
                break

            if written % 200 == 0:
                pbar.set_postfix({"written": written, "hours": round(total_seconds / 3600.0, 2)})

    pbar.close()
    print("\n✅ Done.")
    print(f"   wrote: {written} examples")
    print(f"   hours: {total_seconds / 3600.0:.2f}")
    print(f"   jsonl: {out_jsonl}")
    print(f"   wavs : {out_audio_dir}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--hf_dataset", type=str, default="NbAiLab/NPSC")
    ap.add_argument("--hf_split", type=str, default="train")
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode (no full download).")

    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Required for datasets with custom loading code (NPSC often needs this in containers).",
    )

    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_audio_dir", type=str, default="./npsc_wavs")

    ap.add_argument("--max_hours", type=float, default=0.0, help="Stop after N hours of audio (0 = no limit).")
    ap.add_argument("--max_rows", type=int, default=0, help="Stop after N rows (0 = no limit).")

    ap.add_argument("--min_seconds", type=float, default=1.0)
    ap.add_argument("--max_seconds", type=float, default=15.0)

    ap.add_argument("--text_field", type=str, default="text")

    ap.add_argument("--ref_audio_strategy", type=str, choices=["same", "fixed"], default="same")
    ap.add_argument("--fixed_ref_audio", type=str, default=None)

    ap.add_argument("--skip_existing_wav", action="store_true", help="Skip writing WAV if file exists.")
    ap.add_argument(
        "--require_chars",
        type=str,
        default=DEFAULT_REQUIRE_CHARS,
        help="Whitelist of allowed characters. Set to '' to disable filtering.",
    )

    args = ap.parse_args()

    require_chars = args.require_chars if args.require_chars != "" else None

    build_jsonl(
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code,
        out_jsonl=args.out_jsonl,
        out_audio_dir=args.out_audio_dir,
        max_hours=float(args.max_hours),
        max_rows=int(args.max_rows),
        min_seconds=float(args.min_seconds),
        max_seconds=float(args.max_seconds),
        text_field=args.text_field,
        ref_audio_strategy=args.ref_audio_strategy,
        fixed_ref_audio=args.fixed_ref_audio,
        skip_existing_wav=bool(args.skip_existing_wav),
        require_chars=require_chars,
    )


if __name__ == "__main__":
    main()
