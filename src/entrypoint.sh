#!/bin/bash
set -e

# --- 1. SETUP LOGGING ---
LOG_FILE="/workspace/output/console_log.txt"
mkdir -p /workspace/output
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "   NORWEGIAN QWEN3-TTS LIBRIVOX FINETUNER vTurbo  "
echo "=================================================="

# Funksjon for å laste opp loggen
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
DATA_DIR="/workspace/data"  # <--- VIKTIG: Alt lagres på den persistente disken!

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

# Steg A: Last ned BASIS-modellen (Til PVC) - MED SJEKK FOR KORRUPT FIL
echo "[2/6] Sjekker basismodell..."
MODEL_LOCAL_DIR="/workspace/base_model"
CONFIG_FILE="$MODEL_LOCAL_DIR/config.json"

# Sjekk: Finnes config.json og er den større enn 0 bytes?
if [ ! -s "$CONFIG_FILE" ]; then
    echo "⚠️  Fant ingen gyldig modell (eller korrupt fil). Renser opp og laster ned..."
    rm -rf "$MODEL_LOCAL_DIR"/* # Slett gammelt rask
    
    huggingface-cli download \
        --token "$HF_TOKEN" \
        --resume-download "Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
        --local-dir "$MODEL_LOCAL_DIR"
else
    echo "✅ Gyldig basismodell funnet på disk. Skipper nedlasting."
fi

# Steg B: Bygg jsonl-filen (Sjekker nå på PVC!)
RAW_JSONL="$DATA_DIR/train_raw.jsonl"

if [ ! -f "$RAW_JSONL" ]; then
    echo "[3/6] Bygger LibriVox datasett..."
    python3 /workspace/src/data_nb_librivox.py
    
    # Siden python-scriptet lagrer i CWD, flytter vi den til safe-zone (PVC)
    if [ -f "train_raw.jsonl" ]; then
        mv train_raw.jsonl "$RAW_JSONL"
        echo "✅ Flyttet train_raw.jsonl til $DATA_DIR"
    fi
else
    echo "✅ Fant eksisterende datasett ($RAW_JSONL). Skipper bygging."
fi

# Steg C: Patching
echo "[INFO] Patcher scripts..."
sed -i "s/BATCH_INFER_NUM = 32/BATCH_INFER_NUM = $PREPARE_BATCH_SIZE/g" prepare_data.py

# Steg D: Ekstraher audio codes (Sjekker på PVC!)
CODES_JSONL="$DATA_DIR/train_with_codes.jsonl"

if [ ! -f "$CODES_JSONL" ]; then
    echo "[4/6] Ekstraherer audio codes..."
    python3 prepare_data.py \
      --device cuda:0 \
      --tokenizer_model_path "Qwen/Qwen3-TTS-Tokenizer-12Hz" \
      --input_jsonl "$RAW_JSONL" \
      --output_jsonl "$CODES_JSONL"
else
    echo "✅ Audio codes allerede generert ($CODES_JSONL). Hopper over."
fi

# --- 5. TRAINING ---
echo "[5/6] Starter trening..."
cp /workspace/src/train_norwegian.py . 

EXTRA_ARGS=""
if [ -n "$RESUME_REPO_ID" ]; then
    echo "🔄 RESUME DETECTED: Vil fortsette fra $RESUME_REPO_ID"
    EXTRA_ARGS="$EXTRA_ARGS --resume_repo_id $RESUME_REPO_ID"
fi

echo "🚀 Kjører trening..."
# Bruker accelerate launch for bedre minnestyring + små bokstaver på navn
accelerate launch --num_processes 1 train_norwegian.py \
  --init_model_path "$MODEL_LOCAL_DIR" \
  --output_model_path /workspace/output \
  --train_jsonl "$CODES_JSONL" \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --speaker_name "kathrine" \
  $EXTRA_ARGS

echo "✅ JOBB FERDIG!"
