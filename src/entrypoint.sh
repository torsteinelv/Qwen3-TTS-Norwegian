#!/bin/bash
set -e

# --- 1. SETUP LOGGING ---
LOG_FILE="/workspace/output/console_log.txt"
mkdir -p /workspace/output
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "   NORWEGIAN QWEN3-TTS LIBRIVOX FINETUNER vFinal  "
echo "=================================================="

upload_logs() {
    echo "🏁 Script finished. Uploading logs..."
    python3 -c "
from huggingface_hub import HfApi, login
import os
try:
    login(token=os.getenv('HF_TOKEN'))
    api = HfApi()
    repo = os.getenv('HF_REPO_ID')
    log_file = '$LOG_FILE'
    pod_name = os.getenv('HOSTNAME', 'unknown')
    if os.path.exists(log_file):
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

# Dagens parametere (Batch size 1 for sikkerhet)
export NUM_EPOCHS=${NUM_EPOCHS:-4} 
export LEARNING_RATE=${LEARNING_RATE:-"2e-6"}
export BATCH_SIZE=${BATCH_SIZE:-1}
export PREPARE_BATCH_SIZE=${PREPARE_BATCH_SIZE:-16}

# --- 3. PREPARE ---
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/6] Cloning Qwen3-TTS repository..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
fi
cd "$FINETUNE_DIR"

echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

# Steg A: Last ned BASIS-modellen lokalt (Kopi fra i går!)
echo "[2/6] Laster ned basismodell til disk..."
MODEL_LOCAL_DIR="/workspace/base_model"
if [ ! -d "$MODEL_LOCAL_DIR" ]; then
    huggingface-cli download \
        --token "$HF_TOKEN" \
        --resume-download "Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
        --local-dir "$MODEL_LOCAL_DIR"
else
    echo "Basismodell ligger allerede lokalt."
fi

echo "[3/6] Bygger LibriVox datasett..."
python3 /workspace/src/data_nb_librivox.py

# --- PATCHING ---
echo "[INFO] Patcher scripts..."

# Fiks 1: Minne-fiks for prepare_data (Viktig i dag!)
sed -i "s/BATCH_INFER_NUM = 32/BATCH_INFER_NUM = $PREPARE_BATCH_SIZE/g" prepare_data.py

# Fiks 2: Tensorboard-sti (Viktig for at sft ikke skal krasje)
sed -i 's/log_with="tensorboard")/log_with="tensorboard", project_dir="\/workspace\/output")/g' sft_12hz.py

# --- KJØRING ---

# Steg B: prepare_data bruker REMOTE tokenizer (Slik dere gjorde i går!)
# Dette unngår "Unknown Architecture" feilen
echo "[4/6] Ekstraherer audio codes..."
python3 prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path "Qwen/Qwen3-TTS-Tokenizer-12Hz" \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# Steg C: sft_12hz bruker LOKAL modell (Slik dere gjorde i går!)
# Dette unngår "FileNotFound" ved lagring
#echo "[5/6] Starter trening..."
#python3 sft_12hz.py \
#  --init_model_path "$MODEL_LOCAL_DIR" \
#  --output_model_path /workspace/output \
#  --train_jsonl train_with_codes.jsonl \
#  --batch_size $BATCH_SIZE \
#  --lr $LEARNING_RATE \
#  --num_epochs $NUM_EPOCHS \
#  --speaker_name norsk_taler

# Nytt
echo "[5/6] Starter trening med custom norsk-optimalisert script..."

# Kopierer vårt script inn i mappen der Qwen-bibliotekene er tilgjengelige
cp /workspace/src/train_norwegian.py . 

python3 train_norwegian.py \
  --init_model_path "$MODEL_LOCAL_DIR" \
  --output_model_path /workspace/output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --speaker_name norsk_taler

# --- 5. UPLOAD ---
LAST_EPOCH_IDX=$((NUM_EPOCHS - 1))
CHECKPOINT_DIR="/workspace/output/checkpoint-epoch-$LAST_EPOCH_IDX"
export CHECKPOINT_DIR="$CHECKPOINT_DIR"

echo "[6/6] Ser etter resultater..."
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✅ Fant ferdig modell: $CHECKPOINT_DIR. Laster opp..."
    
    if [ -f "/workspace/src/upload_to_hf.py" ]; then
        python3 /workspace/src/upload_to_hf.py --local_dir "$CHECKPOINT_DIR" --repo_id "$HF_REPO_ID"
    else
        # Fallback hvis upload script mangler
        cat << EOF > upload_simple.py
import os
from huggingface_hub import HfApi
api = HfApi()
repo = os.getenv('HF_REPO_ID')
src = "$CHECKPOINT_DIR"
tgt = f"librivox_model_final/{os.path.basename(src)}"
print(f"🚀 Uploading {src} -> {repo}/{tgt}")
api.upload_folder(folder_path=src, repo_id=repo, path_in_repo=tgt)
EOF
        python3 upload_simple.py
    fi
else
    echo "❌ CHECKPOINT NOT FOUND!"
    exit 1
fi

echo "✅ JOBB FERDIG!"
