#!/bin/bash
set -e

echo "🚀 Starting Entrypoint Script for Qwen3-TTS (Smart Config Edition)"

# --- KONFIGURASJON ---
WORKDIR="/workspace"
REPO_DIR="$WORKDIR/Qwen3-TTS"
FINETUNE_DIR="$REPO_DIR/finetuning"

# Definerer Defaults (Disse kan overstyres via 'env' i Kubernetes YAML)
# Vi setter default til 4 epochs basert på anbefalingen om lav learning rate.
export NUM_EPOCHS=${NUM_EPOCHS:-4} 
export LEARNING_RATE=${LEARNING_RATE:-"2e-6"}
export BATCH_SIZE=${BATCH_SIZE:-2}

echo "⚙️ Config loaded: Epochs=$NUM_EPOCHS, LR=$LEARNING_RATE, Batch=$BATCH_SIZE"

# 1. Clone Repo
if [ ! -d "$REPO_DIR" ]; then
    echo "📦 Cloning Qwen3-TTS repository..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git "$REPO_DIR"
fi
cd "$FINETUNE_DIR"

# 2. Login
echo "🔑 Logging into Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

# 3. Bygg Dataset
echo "📚 Building Dataset (Target: Kathrine Engan)..."
python3 /workspace/src/data_nb_librivox.py

# 4. Prepare Data
echo "⚙️ Running prepare_data.py..."
python3 prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# 5. START TRENING 🔥
echo "🔥 Starting SFT Training..."
python3 sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path /workspace/output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --speaker_name norsk_taler

# 6. SLIM UPLOAD LOGIC 📤
# Siden vi vet NUM_EPOCHS, vet vi nøyaktig hva siste mappe heter.
# Hvis NUM_EPOCHS er 4, vil mappene hete 0, 1, 2, 3.
# Siste mappe er altså (NUM_EPOCHS - 1).

LAST_EPOCH_IDX=$((NUM_EPOCHS - 1))
CHECKPOINT_DIR="/workspace/output/checkpoint-epoch-$LAST_EPOCH_IDX"

echo "🎯 Forventet slutt-mappe: $CHECKPOINT_DIR"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✅ Mappen finnes! Starter opplasting..."
    
    cat << EOF > upload_simple.py
import os
from huggingface_hub import HfApi, login

login(token=os.getenv('HF_TOKEN'))
api = HfApi()
repo_id = os.getenv('HF_REPO_ID')

source_dir = "$CHECKPOINT_DIR"
folder_name = os.path.basename(source_dir)
target_path = f"librivox_model_v3/{folder_name}"

print(f"🚀 Laster opp {source_dir} -> {repo_id}/{target_path}")

api.upload_folder(
    folder_path=source_dir,
    repo_id=repo_id,
    path_in_repo=target_path
)
print("✨ Opplasting fullført!")
EOF

    python3 upload_simple.py

else
    echo "❌ KRITISK FEIL: Fant ikke mappen $CHECKPOINT_DIR etter trening!"
    echo "   Sjekk om treningen kræsjet før den ble ferdig."
    exit 1
fi

echo "🎉 JOB DONE!"
