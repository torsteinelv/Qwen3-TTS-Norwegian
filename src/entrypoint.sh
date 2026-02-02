#!/bin/bash
set -e

# --- 1. SETUP LOGGING ---
LOG_FILE="/workspace/output/console_log.txt"
mkdir -p /workspace/output
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "   NORWEGIAN QWEN3-TTS LORA FINETUNER (vTurbo)    "
echo "=================================================="

# Funksjon for å laste opp loggen ved krasj/slutt
upload_logs() {
    echo "🏁 Script finished/exited. Uploading logs..."
    python3 -c "
from huggingface_hub import HfApi, login
import os
try:
    token = os.getenv('HF_TOKEN')
    if token:
        login(token=token)
        api = HfApi()
        repo = os.getenv('HF_REPO_ID')
        log_file = '$LOG_FILE'
        pod_name = os.getenv('HOSTNAME', 'unknown')
        if os.path.exists(log_file) and repo:
            api.upload_file(path_or_fileobj=log_file, path_in_repo=f'logs/console_{pod_name}.txt', repo_id=repo)
            print('✅ Log uploaded!')
except Exception as e:
    print(f'❌ Log upload failed: {e}')
"
}
trap upload_logs EXIT

# --- 2. CONFIG ---
WORKDIR="/workspace"
REPO_DIR="$WORKDIR/Qwen3-TTS"
FINETUNE_DIR="$REPO_DIR/finetuning"
DATA_DIR="/workspace/data"

# Standard parametere (Kan overstyres av environment variables)
export NUM_EPOCHS=${NUM_EPOCHS:-30} 
export LEARNING_RATE=${LEARNING_RATE:-"1e-4"} 
export BATCH_SIZE=${BATCH_SIZE:-4}
export PREPARE_BATCH_SIZE=${PREPARE_BATCH_SIZE:-16}

# Sikkerhetssjekk: Installer PEFT hvis det mangler i imaget
if ! python3 -c "import peft" &> /dev/null; then
    echo "📦 PEFT mangler, installerer..."
    pip install peft
fi

# --- 3. PREPARE REPO (Trengs for prepare_data.py) ---
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/6] Cloning Qwen3-TTS repository..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
fi
cd "$FINETUNE_DIR"

echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# --- 4. PREPARE MODEL & DATA ---

# Steg A: Last ned BASIS-modellen
echo "[2/6] Sjekker basismodell..."
MODEL_LOCAL_DIR="/workspace/base_model"
CONFIG_FILE="$MODEL_LOCAL_DIR/config.json"

if [ ! -s "$CONFIG_FILE" ]; then
    echo "⚠️  Fant ingen gyldig modell. Laster ned..."
    rm -rf "$MODEL_LOCAL_DIR"/*
    huggingface-cli download \
        --token "$HF_TOKEN" \
        --resume-download "Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
        --local-dir "$MODEL_LOCAL_DIR"
else
    echo "✅ Gyldig basismodell funnet på disk."
fi

# Steg B: Bygg jsonl-filen
RAW_JSONL="$DATA_DIR/train_raw.jsonl"

if [ ! -f "$RAW_JSONL" ]; then
    echo "[3/6] Bygger LibriVox datasett..."
    # Kjører scriptet direkte fra src-mappen
    python3 /workspace/src/data_nb_librivox.py
    
    if [ -f "train_raw.jsonl" ]; then
        mv train_raw.jsonl "$RAW_JSONL"
        echo "✅ Flyttet train_raw.jsonl til $DATA_DIR"
    fi
else
    echo "✅ Fant eksisterende datasett ($RAW_JSONL)."
fi

# Steg C: Patching for Audio Codes (Optimalisering)
echo "[INFO] Patcher scripts..."
sed -i "s/BATCH_INFER_NUM = 32/BATCH_INFER_NUM = $PREPARE_BATCH_SIZE/g" prepare_data.py

# Steg D: Ekstraher audio codes
CODES_JSONL="$DATA_DIR/train_with_codes.jsonl"

if [ ! -f "$CODES_JSONL" ]; then
    echo "[4/6] Ekstraherer audio codes..."
    python3 prepare_data.py \
      --device cuda:0 \
      --tokenizer_model_path "Qwen/Qwen3-TTS-Tokenizer-12Hz" \
      --input_jsonl "$RAW_JSONL" \
      --output_jsonl "$CODES_JSONL"
else
    echo "✅ Audio codes allerede generert. Hopper over."
fi



# Vi kjører direkte mot filen som start.sh har hentet ned
accelerate launch --num_processes 1 /workspace/src/train_norwegian_new.py \
  --train_jsonl "$CODES_JSONL" \
  --init_model_path "$MODEL_LOCAL_DIR" \
  --output_model_path /workspace/output/run_long \
  --batch_size 4 \
  --num_epochs 10

echo "✅ JOBB FERDIG!"
