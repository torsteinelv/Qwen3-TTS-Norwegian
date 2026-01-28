#!/bin/bash
set -e

echo "=================================================="
echo "   NORWEGIAN QWEN3-TTS FINETUNER (FIXED)          "
echo "=================================================="

# 1. Klon Qwen3-TTS repoet
if [ ! -d "Qwen3-TTS" ]; then
    echo "[1/5] Cloner Qwen3-TTS repo..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git
else
    echo "[1/5] Qwen3-TTS repo finnes allerede."
fi

# Gå til finetuning mappen
cd Qwen3-TTS/finetuning

# 2. Generer datasett (WAV + JSONL)
echo "[2/5] Bygger datasett fra NPSC..."
python /workspace/src/data_npsc.py

# 3. Preprosesser data (Audio -> Codes)
echo "[3/5] Ekstraherer audio codes (Tokenizing)..."
python prepare_data.py \
    --input_jsonl /workspace/data/train.jsonl \
    --output_jsonl /workspace/data/train_with_codes.jsonl \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --device cuda:0

# --- FIX: Patch scriptet for å øke gradient accumulation ---
# Scriptet har hardkodet 'gradient_accumulation_steps=4'. Vi endrer det til 8 for bedre stabilitet.
echo "[INFO] Patcher sft_12hz.py for å bruke accumulation=8..."
sed -i 's/gradient_accumulation_steps=4/gradient_accumulation_steps=8/g' sft_12hz.py

# 4. Start Trening (SFT)
# Fjernet --gradient_accumulation_steps og --save_steps da scriptet ikke støtter dem som argumenter.
# Scriptet lagrer automatisk en checkpoint etter hver epoch.
echo "[4/5] Starter trening..."
accelerate launch sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl /workspace/data/train_with_codes.jsonl \
    --output_model_path /workspace/output \
    --batch_size 2 \
    --lr 1e-5 \
    --num_epochs 5 \
    --speaker_name "norsk_taler"

# 5. Last opp til Hugging Face
echo "[5/5] Ser etter resultater for opplasting..."

if [ -n "$HF_REPO_ID" ] && [ -n "$HF_TOKEN" ]; then
    # Finn mappen med det siste sjekkpunktet
    LAST_CHECKPOINT=$(ls -dt /workspace/output/checkpoint-epoch-* | head -1)
    
    if [ -z "$LAST_CHECKPOINT" ]; then
        echo "❌ Fant ingen checkpoints i /workspace/output! Treningen kan ha feilet."
    else
        echo "📤 Fant ferdig modell: $LAST_CHECKPOINT"
        echo "🚀 Laster opp til Hugging Face: $HF_REPO_ID"
        
        python /workspace/src/upload_to_hf.py \
            --local_dir "$LAST_CHECKPOINT" \
            --repo_id "$HF_REPO_ID"
    fi
else
    echo "⚠️  HF_REPO_ID eller HF_TOKEN mangler. Hopper over opplasting."
    echo "   (Modellen ligger lokalt i output-mappen)"
fi

echo "✅ JOBB FERDIG!"
