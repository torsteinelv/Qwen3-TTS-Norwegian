#!/bin/bash
set -e

# --- 1. SETUP LOGGING ---
LOG_FILE="/workspace/output/console_log.txt"
mkdir -p /workspace/output
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "   NORWEGIAN QWEN3-TTS LIBRIVOX FINETUNER vTurbo  "
echo "=================================================="

# Funksjon for å laste opp loggen hvis scriptet dør
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

# Dagens parametere
export NUM_EPOCHS=${NUM_EPOCHS:-10} 
export LEARNING_RATE=${LEARNING_RATE:-"2e-6"}
export BATCH_SIZE=${BATCH_SIZE:-4}
export PREPARE_BATCH_SIZE=${PREPARE_BATCH_SIZE:-16}

# --- 3. PREPARE REPO ---
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/6] Cloning Qwen3-TTS repository..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
fi
cd "$FINETUNE_DIR"

echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# --- 4. PREPARE MODEL & DATA ---

# Steg A: Last ned BASIS-modellen lokalt
echo "[2/6] Laster ned basismodell til disk..."
MODEL_LOCAL_DIR="/workspace/base_model"
if [ ! -d "$MODEL_LOCAL_DIR" ]; then
    huggingface-cli download \
        --token "$HF_TOKEN" \
        --resume-download "Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
        --local-dir "$MODEL_LOCAL_DIR"
else
    echo "✅ Basismodell ligger allerede lokalt."
fi

# Steg B: Bygg jsonl-filen (hvis den mangler)
if [ ! -f "train_raw.jsonl" ]; then
    echo "[3/6] Bygger LibriVox datasett..."
    # Vi bruker full path til scriptet ditt
    python3 /workspace/src/data_nb_librivox.py
fi

# Steg C: Patching av scripts (fremdeles nødvendig for prepare_data)
echo "[INFO] Patcher scripts..."
sed -i "s/BATCH_INFER_NUM = 32/BATCH_INFER_NUM = $PREPARE_BATCH_SIZE/g" prepare_data.py

# Steg D: Ekstraher audio codes (Tung prosess, kjøres kun hvis train_with_codes mangler)
if [ ! -f "train_with_codes.jsonl" ]; then
    echo "[4/6] Ekstraherer audio codes..."
    python3 prepare_data.py \
      --device cuda:0 \
      --tokenizer_model_path "Qwen/Qwen3-TTS-Tokenizer-12Hz" \
      --input_jsonl train_raw.jsonl \
      --output_jsonl train_with_codes.jsonl
else
    echo "✅ Audio codes allerede generert. Hopper over."
fi

# --- 5. TRAINING ---
echo "[5/6] Starter trening med custom norsk-optimalisert script..."

# Kopierer vårt nye script inn i arbeidsmappen
cp /workspace/src/train_norwegian.py . 

# Sjekk om vi skal RESUME (Fortsette) fra en tidligere jobb
EXTRA_ARGS=""
if [ -n "$RESUME_REPO_ID" ]; then
    echo "🔄 RESUME DETECTED: Vil fortsette fra $RESUME_REPO_ID"
    EXTRA_ARGS="$EXTRA_ARGS --resume_repo_id $RESUME_REPO_ID"
fi

echo "🚀 Kjører trening..."
echo "   Batch Size: $BATCH_SIZE"
echo "   Epochs:     $NUM_EPOCHS"

# Vi bruker 'accelerate launch' for bedre GPU-styring
accelerate launch --num_processes 1 train_norwegian.py \
  --init_model_path "$MODEL_LOCAL_DIR" \
  --output_model_path /workspace/output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --speaker_name "Kathrine" \
  $EXTRA_ARGS

echo "✅ JOBB FERDIG! (Alle epoker er lastet opp underveis)"
