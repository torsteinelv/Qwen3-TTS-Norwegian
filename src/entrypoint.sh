#!/bin/bash
set -e

# --- 1. SETUP LOGGING (fresh each container run) ---
LOG_FILE="/tmp/console_log.txt"
rm -f "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "  NORWEGIAN QWEN3-TTS FINETUNE (SCRATCH/STABLE)   "
echo "=================================================="

upload_logs() {
  echo "🏁 Script finished/exited. Uploading logs..."
  python3 - << 'PY'
from huggingface_hub import HfApi, login
import os
try:
    token = os.getenv("HF_TOKEN")
    repo = os.getenv("HF_REPO_ID")
    log_file = "/tmp/console_log.txt"
    pod_name = os.getenv("HOSTNAME", "unknown")
    if token and repo and os.path.exists(log_file):
        login(token=token)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=log_file,
            path_in_repo=f"logs/console_{pod_name}.txt",
            repo_id=repo,
            repo_type="model"
        )
        print("✅ Log uploaded!")
    else:
        print("ℹ️ Skipped log upload (missing HF_TOKEN/HF_REPO_ID or log file).")
except Exception as e:
    print(f"❌ Log upload failed: {e}")
PY
}
trap upload_logs EXIT

# --- 2. CONFIG ---
WORKDIR="/workspace"
REPO_DIR="$WORKDIR/Qwen3-TTS"
FINETUNE_DIR="$REPO_DIR/finetuning"
DATA_DIR="/workspace/data"
MODEL_LOCAL_DIR="/workspace/base_model"

# Train params (override via env)
export NUM_EPOCHS=${NUM_EPOCHS:-10}
export BATCH_SIZE=${BATCH_SIZE:-4}
export GRAD_ACCUM=${GRAD_ACCUM:-4}

export LORA_LR=${LORA_LR:-"1e-5"}
export TEXT_PROJ_LR=${TEXT_PROJ_LR:-"5e-5"}
export SUB_LOSS_WEIGHT=${SUB_LOSS_WEIGHT:-"0.2"}

export LORA_R=${LORA_R:-4}
export LORA_ALPHA=${LORA_ALPHA:-8}
export LORA_DROPOUT=${LORA_DROPOUT:-0.1}
export TRAIN_MLP_LORA=${TRAIN_MLP_LORA:-false}

export MIXED_PRECISION=${MIXED_PRECISION:-bf16}

# Dataset toggles
# Default ON: 1000 hours. Set to 0 to disable.
export NPSC_HOURS=${NPSC_HOURS:-1000}

# ✅ REQUIRED: prepare_data batch size = 16
export PREPARE_BATCH_SIZE=16

mkdir -p "$DATA_DIR"

# --- 3. REPO ---
if [ ! -d "$REPO_DIR" ]; then
  echo "[1/6] Cloning Qwen3-TTS repository..."
  git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
fi

cd "$FINETUNE_DIR"

echo "🔑 Hugging Face auth (token via env)"
python3 - << 'PY'
import os
assert os.getenv("HF_TOKEN"), "HF_TOKEN missing"
assert os.getenv("HF_REPO_ID"), "HF_REPO_ID missing"
print("✅ HF token/repo OK")
PY

# --- 4. BASE MODEL ---
echo "[2/6] Checking base model..."
if [ ! -s "$MODEL_LOCAL_DIR/config.json" ]; then
  echo "⚠️  Base model missing. Downloading..."
  rm -rf "$MODEL_LOCAL_DIR"
  mkdir -p "$MODEL_LOCAL_DIR"
  huggingface-cli download \
    --token "$HF_TOKEN" \
    --resume-download "Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
    --local-dir "$MODEL_LOCAL_DIR"
else
  echo "✅ Base model already present."
fi

# --- 5. BUILD DATASETS ---
echo "[3/6] Building datasets..."
RAW_LIB="$DATA_DIR/train_raw_librivox.jsonl"
RAW_NPSC="$DATA_DIR/train_raw_npsc.jsonl"

if [ ! -f "$RAW_LIB" ]; then
  echo "📚 Building LibriVox dataset..."
  # run builder (assumes it writes train_raw.jsonl in cwd)
  python3 /workspace/src/data_nb_librivox.py
  if [ -f "train_raw.jsonl" ]; then
    mv train_raw.jsonl "$RAW_LIB"
  elif [ -f "/workspace/train_raw.jsonl" ]; then
    mv /workspace/train_raw.jsonl "$RAW_LIB"
  fi
  test -f "$RAW_LIB" || (echo "❌ LibriVox builder did not produce train_raw.jsonl" && exit 1)
else
  echo "✅ Found $RAW_LIB"
fi

if [ "$NPSC_HOURS" != "0" ]; then
  if [ ! -f "$RAW_NPSC" ]; then
    echo "🎧 Building NPSC dataset (max_hours=$NPSC_HOURS)..."
    python3 /workspace/src/data_nb_npsc.py \
      --out_jsonl "$RAW_NPSC" \
      --out_audio_dir "$DATA_DIR/npsc_wavs" \
      --max_hours "$NPSC_HOURS" \
      --min_seconds 1 \
      --max_seconds 15 \
      --trust_remote_code

  else
    echo "✅ Found $RAW_NPSC"
  fi
else
  echo "ℹ️ NPSC disabled (NPSC_HOURS=0)."
fi

else
  echo "ℹ️ NPSC disabled (NPSC_HOURS=0)."
fi

# --- 6. PREPARE CODES ---
echo "[4/6] Preparing audio codes..."
sed -i "s/BATCH_INFER_NUM = 32/BATCH_INFER_NUM = $PREPARE_BATCH_SIZE/g" prepare_data.py

CODES_LIB="$DATA_DIR/train_with_codes_librivox.jsonl"
CODES_NPSC="$DATA_DIR/train_with_codes_npsc.jsonl"

if [ ! -f "$CODES_LIB" ]; then
  python3 prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path "Qwen/Qwen3-TTS-Tokenizer-12Hz" \
    --input_jsonl "$RAW_LIB" \
    --output_jsonl "$CODES_LIB"
else
  echo "✅ Found $CODES_LIB"
fi

if [ "$NPSC_HOURS" != "0" ] && [ -f "$RAW_NPSC" ]; then
  if [ ! -f "$CODES_NPSC" ]; then
    python3 prepare_data.py \
      --device cuda:0 \
      --tokenizer_model_path "Qwen/Qwen3-TTS-Tokenizer-12Hz" \
      --input_jsonl "$RAW_NPSC" \
      --output_jsonl "$CODES_NPSC"
  else
    echo "✅ Found $CODES_NPSC"
  fi
fi

# --- 7. TRAIN ---
echo "[5/6] Training..."
RUN_DIR="/workspace/output/run_scratch_v1"
mkdir -p "$RUN_DIR"

TRAIN_JSONL_ARGS="--train_jsonl $CODES_LIB"
if [ "$NPSC_HOURS" != "0" ] && [ -f "$CODES_NPSC" ]; then
  TRAIN_JSONL_ARGS="$TRAIN_JSONL_ARGS --train_jsonl $CODES_NPSC"
fi

echo "RUN_DIR=$RUN_DIR"
echo "TRAIN_JSONL_ARGS=$TRAIN_JSONL_ARGS"
echo "PREPARE_BATCH_SIZE=$PREPARE_BATCH_SIZE"

# fail fast on syntax before we start
python3 -m py_compile /workspace/src/train_norwegian_scratch_stable.py

accelerate launch \
  --num_processes 1 \
  --mixed_precision "$MIXED_PRECISION" \
  /workspace/src/train_norwegian_scratch_stable.py \
  $TRAIN_JSONL_ARGS \
  --init_model_path "$MODEL_LOCAL_DIR" \
  --base_model_id "Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
  --output_model_path "$RUN_DIR" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --num_epochs "$NUM_EPOCHS" \
  --save_every 1 \
  --lora_lr "$LORA_LR" \
  --text_proj_lr "$TEXT_PROJ_LR" \
  --sub_loss_weight "$SUB_LOSS_WEIGHT" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --train_mlp_lora "$TRAIN_MLP_LORA" \
  --mixed_precision "$MIXED_PRECISION" \
  --hf_repo_id "$HF_REPO_ID"

echo "✅ JOBB FERDIG!"
