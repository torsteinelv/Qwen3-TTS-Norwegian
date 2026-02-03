#!/bin/bash
set -euo pipefail

# -----------------------------
# 1) LOGGING
# -----------------------------
LOG_FILE="/tmp/console_log.txt"
mkdir -p /workspace/output
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "  NORWEGIAN QWEN3-TTS FINETUNE (SCRATCH/STABLE)   "
echo "=================================================="

upload_logs() {
  echo "🏁 Script finished/exited. Uploading logs..."
  python3 - <<'PY'
from huggingface_hub import HfApi
import os, pathlib

repo = os.getenv("HF_REPO_ID")
token = os.getenv("HF_TOKEN")
log_file = os.getenv("LOG_FILE", "/workspace/output/console_log.txt")
pod = os.getenv("HOSTNAME", "unknown")

try:
    if repo and token and pathlib.Path(log_file).exists():
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=log_file,
            path_in_repo=f"logs/console_{pod}.txt",
            repo_id=repo,
            repo_type="model"
        )
        print("✅ Log uploaded!")
    else:
        print("ℹ️ Skipping log upload (missing HF_REPO_ID/HF_TOKEN or log file).")
except Exception as e:
    print(f"❌ Log upload failed: {e}")
PY
}
export LOG_FILE
trap upload_logs EXIT

# -----------------------------
# 2) CONFIG (ENV OVERRIDES)
# -----------------------------
WORKDIR="/workspace"
REPO_DIR="$WORKDIR/Qwen3-TTS"
DATA_DIR="$WORKDIR/data"
SRC_DIR="$WORKDIR/src"
MODEL_LOCAL_DIR="$WORKDIR/base_model"

# Training
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_EVERY="${SAVE_EVERY:-1}"

LORA_LR="${LORA_LR:-1e-5}"
TEXT_PROJ_LR="${TEXT_PROJ_LR:-5e-5}"
SUB_LOSS_WEIGHT="${SUB_LOSS_WEIGHT:-0.2}"

LORA_R="${LORA_R:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
TRAIN_MLP_LORA="${TRAIN_MLP_LORA:-false}"   # true/false

MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # bf16/fp16/no

# Dataset sizing
LIBRIVOX_HOURS="${LIBRIVOX_HOURS:-12}"      # increase to "as much as possible"
NPSC_HOURS="${NPSC_HOURS:-0}"               # 0 = skip, else add hours
MAX_SECONDS="${MAX_SECONDS:-15}"
MIN_SECONDS="${MIN_SECONDS:-0.6}"

# Codes extraction
CODES_DEVICE="${CODES_DEVICE:-cuda:0}"
CODES_BATCH_SIZE="${CODES_BATCH_SIZE:-8}"
CODES_DTYPE="${CODES_DTYPE:-bf16}"

# Output run
RUN_DIR="${RUN_DIR:-/workspace/output/run_scratch_v1}"

mkdir -p "$DATA_DIR" "$RUN_DIR"

# -----------------------------
# 3) DEPENDENCIES (best-effort)
# -----------------------------
python3 -c "import peft" >/dev/null 2>&1 || pip install -U peft
python3 -c "import datasets" >/dev/null 2>&1 || pip install -U datasets
python3 -c "import librosa" >/dev/null 2>&1 || pip install -U librosa
python3 -c "import soundfile" >/dev/null 2>&1 || pip install -U soundfile
python3 -c "import accelerate" >/dev/null 2>&1 || pip install -U accelerate

# -----------------------------
# 4) PREPARE REPO
# -----------------------------
if [ ! -d "$REPO_DIR" ]; then
  echo "[1/6] Cloning Qwen3-TTS repository..."
  git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
fi

# -----------------------------
# 5) HF AUTH (avoid deprecated login command)
# -----------------------------
if [ -n "${HF_TOKEN:-}" ]; then
  echo "🔑 Hugging Face auth (token via env)"
  python3 - <<'PY'
import os
from huggingface_hub import HfApi
HfApi(token=os.getenv("HF_TOKEN")).whoami()
print("✅ HF token OK")
PY
else
  echo "⚠️ HF_TOKEN not set. HF downloads/uploads may fail."
fi

# -----------------------------
# 6) BASE MODEL
# -----------------------------
echo "[2/6] Checking base model..."
CONFIG_FILE="$MODEL_LOCAL_DIR/config.json"
if [ ! -s "$CONFIG_FILE" ]; then
  echo "⬇️ Downloading base model to $MODEL_LOCAL_DIR ..."
  rm -rf "$MODEL_LOCAL_DIR"
  mkdir -p "$MODEL_LOCAL_DIR"
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
token = os.getenv("HF_TOKEN")
local_dir = "/workspace/base_model"
snapshot_download(repo_id=repo, local_dir=local_dir, token=token, local_dir_use_symlinks=False)
print("✅ Base model downloaded")
PY
else
  echo "✅ Base model already present."
fi

# -----------------------------
# 7) BUILD DATASETS (unique filenames)
# -----------------------------
echo "[3/6] Building datasets..."

LIB_RAW="$DATA_DIR/train_raw_librivox.jsonl"
NPSC_RAW="$DATA_DIR/train_raw_npsc.jsonl"

if [ ! -f "$LIB_RAW" ]; then
  echo "📚 Building nb-librivox -> $LIB_RAW  (hours=$LIBRIVOX_HOURS)"
  python3 "$SRC_DIR/data_nb_librivox.py" \
    --out_jsonl "$LIB_RAW" \
    --out_audio_dir "$DATA_DIR/audio_librivox" \
    --max_hours "$LIBRIVOX_HOURS" \
    --min_seconds "$MIN_SECONDS" \
    --max_seconds "$MAX_SECONDS" \
    --ref_audio_strategy same
else
  echo "✅ Found $LIB_RAW"
fi

if [ "$NPSC_HOURS" != "0" ]; then
  if [ ! -f "$NPSC_RAW" ]; then
    echo "🎧 Building NPSC -> $NPSC_RAW (hours=$NPSC_HOURS)"
    python3 "$SRC_DIR/data_npsc.py" \
      --out_jsonl "$NPSC_RAW" \
      --out_audio_dir "$DATA_DIR/audio_npsc" \
      --max_hours "$NPSC_HOURS" \
      --min_seconds "$MIN_SECONDS" \
      --max_seconds "$MAX_SECONDS" \
      --ref_audio_strategy same
  else
    echo "✅ Found $NPSC_RAW"
  fi
else
  echo "ℹ️ NPSC disabled (NPSC_HOURS=0)."
fi

# -----------------------------
# 8) PREPARE AUDIO CODES (unique filenames)
# -----------------------------
echo "[4/6] Preparing audio codes..."

LIB_CODES="$DATA_DIR/train_with_codes_librivox.jsonl"
NPSC_CODES="$DATA_DIR/train_with_codes_npsc.jsonl"

if [ ! -f "$LIB_CODES" ]; then
  echo "🔢 Encoding codes for LibriVox -> $LIB_CODES"
  python3 "$SRC_DIR/prepare_codes_12hz.py" \
    --in_jsonl "$LIB_RAW" \
    --out_jsonl "$LIB_CODES" \
    --qwen_path "$REPO_DIR" \
    --device_map "$CODES_DEVICE" \
    --batch_size "$CODES_BATCH_SIZE" \
    --dtype "$CODES_DTYPE"
else
  echo "✅ Found $LIB_CODES"
fi

TRAIN_JSONL_ARGS=( --train_jsonl "$LIB_CODES" )

if [ "$NPSC_HOURS" != "0" ]; then
  if [ ! -f "$NPSC_CODES" ]; then
    echo "🔢 Encoding codes for NPSC -> $NPSC_CODES"
    python3 "$SRC_DIR/prepare_codes_12hz.py" \
      --in_jsonl "$NPSC_RAW" \
      --out_jsonl "$NPSC_CODES" \
      --qwen_path "$REPO_DIR" \
      --device_map "$CODES_DEVICE" \
      --batch_size "$CODES_BATCH_SIZE" \
      --dtype "$CODES_DTYPE"
  else
    echo "✅ Found $NPSC_CODES"
  fi
  TRAIN_JSONL_ARGS+=( --train_jsonl "$NPSC_CODES" )
fi

# -----------------------------
# 9) TRAIN
# -----------------------------
echo "[5/6] Training..."
echo "RUN_DIR=$RUN_DIR"
echo "TRAIN_JSONL_ARGS=${TRAIN_JSONL_ARGS[*]}"

accelerate launch --num_processes 1 "$SRC_DIR/train_norwegian_scratch_stable.py" \
  "${TRAIN_JSONL_ARGS[@]}" \
  --init_model_path "$MODEL_LOCAL_DIR" \
  --output_model_path "$RUN_DIR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --num_epochs "$NUM_EPOCHS" \
  --save_every "$SAVE_EVERY" \
  --lora_lr "$LORA_LR" \
  --text_proj_lr "$TEXT_PROJ_LR" \
  --sub_loss_weight "$SUB_LOSS_WEIGHT" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --train_mlp_lora "$TRAIN_MLP_LORA" \
  --mixed_precision "$MIXED_PRECISION" \
  --hf_repo_id "${HF_REPO_ID:-}"

echo "[6/6] ✅ JOBB FERDIG!"
