#!/bin/bash
set -e

# --- 1. SETUP LOGGING ---
# Vi lagrer alt som skjer i console til en fil
LOG_FILE="/workspace/output/console_log.txt"
mkdir -p /workspace/output

# Starter logging: Alt som skrives til stdout/stderr går nå OGSÅ til filen
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🚀 Starting Entrypoint Script (Logging enabled)"

# --- 2. FUNKSJON FOR LOGG-OPPLASTING (Kjøres alltid ved slutt) ---
upload_logs() {
    EXIT_CODE=$?
    echo "🏁 Script finished with exit code: $EXIT_CODE"
    echo "📤 Uploading console log to Hugging Face..."
    
    # Python one-liner for å laste opp loggen sikkert
    python3 -c "
from huggingface_hub import HfApi, login
import os

try:
    login(token=os.getenv('HF_TOKEN'))
    api = HfApi()
    repo = os.getenv('HF_REPO_ID')
    log_file = '$LOG_FILE'
    
    # Vi bruker pod-navnet (HOSTNAME) for å skille logger fra hverandre
    pod_name = os.getenv('HOSTNAME', 'unknown-pod')
    target_path = f'logs/console_{pod_name}.txt'
    
    print(f'   Loggfil: {log_file}')
    print(f'   Target: {repo}/{target_path}')
    
    if os.path.exists(log_file):
        api.upload_file(
            path_or_fileobj=log_file,
            path_in_repo=target_path,
            repo_id=repo
        )
        print('✅ Log uploaded successfully!')
    else:
        print('⚠️ Log file not found!')
except Exception as e:
    print(f'❌ Failed to upload log: {e}')
"
}

# Sett fellen: Kjør upload_logs når scriptet avslutter (uansett grunn)
trap upload_logs EXIT

# --- 3. KONFIGURASJON ---
WORKDIR="/workspace"
REPO_DIR="$WORKDIR/Qwen3-TTS"
FINETUNE_DIR="$REPO_DIR/finetuning"

export NUM_EPOCHS=${NUM_EPOCHS:-4} 
export LEARNING_RATE=${LEARNING_RATE:-"2e-6"}
export BATCH_SIZE=${BATCH_SIZE:-2}

echo "⚙️ Config: Epochs=$NUM_EPOCHS, LR=$LEARNING_RATE, Batch=$BATCH_SIZE"

# --- 4. PREPARERING ---
if [ ! -d "$REPO_DIR" ]; then
    echo "📦 Cloning Qwen3-TTS repository..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
fi
cd "$FINETUNE_DIR"

echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

echo "📚 Building Dataset (NbAiLab/nb-librivox)..."
# Vi bruker scriptet som bypasser load_dataset-feilen
python3 /workspace/src/data_nb_librivox.py

echo "⚙️ Running prepare_data.py..."
python3 prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# --- 5. TRENING 🔥 ---
echo "🔥 Starting SFT Training..."
python3 sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path /workspace/output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --speaker_name norsk_taler

# --- 6. OPPLASTING AV MODELL ---
LAST_EPOCH_IDX=$((NUM_EPOCHS - 1))
CHECKPOINT_DIR="/workspace/output/checkpoint-epoch-$LAST_EPOCH_IDX"
export CHECKPOINT_DIR="$CHECKPOINT_DIR" # Trengs for test-scriptet

echo "🎯 Forventet slutt-mappe: $CHECKPOINT_DIR"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✅ Checkpoint found! Uploading model..."
    
    cat << EOF > upload_simple.py
import os
from huggingface_hub import HfApi, login

login(token=os.getenv('HF_TOKEN'))
api = HfApi()
repo_id = os.getenv('HF_REPO_ID')

source_dir = "$CHECKPOINT_DIR"
folder_name = os.path.basename(source_dir)
target_path = f"librivox_model_final/{folder_name}"

print(f"🚀 Uploading {source_dir} -> {repo_id}/{target_path}")

api.upload_folder(
    folder_path=source_dir,
    repo_id=repo_id,
    path_in_repo=target_path
)
print("✨ Model upload complete!")
EOF
    python3 upload_simple.py

    # --- 7. VALIDEINGSTEST (Kun hvis opplasting gikk bra) ---
    echo "🔬 Running validation test..."
    python3 /workspace/src/test_final_model.py || echo "⚠️ Validation failed, but model is safe."

else
    echo "❌ CRITICAL ERROR: Checkpoint folder not found!"
    exit 1
fi

echo "🎉 JOB DONE!"
# (Her slutter scriptet, og 'trap upload_logs EXIT' slår inn og laster opp loggen)
