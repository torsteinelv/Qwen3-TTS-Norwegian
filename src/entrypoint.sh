#!/bin/bash
set -e

echo "=================================================="
echo "   NORWEGIAN QWEN3-TTS FINETUNER (FINAL V4)       "
echo "=================================================="

# 1. Klon Qwen3-TTS repoet
if [ ! -d "Qwen3-TTS" ]; then
    echo "[1/6] Cloner Qwen3-TTS repo..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git
else
    echo "[1/6] Qwen3-TTS repo finnes allerede."
fi

# 2. Last ned BASIS-modellen lokalt (for å unngå krasj etter trening)
echo "[2/6] Laster ned basismodell til disk..."
if [ ! -d "/workspace/base_model" ]; then
    huggingface-cli download \
        --token "$HF_TOKEN" \
        --resume-download "Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
        --local-dir /workspace/base_model
else
    echo "Basismodell ligger allerede lokalt."
fi

# Gå til finetuning mappen
cd Qwen3-TTS/finetuning

# 3. Generer datasett (WAV + JSONL)
echo "[3/6] Bygger datasett fra NPSC..."
python /workspace/src/data_npsc.py

# 4. Preprosesser data (Audio -> Codes)
# VIKTIG ENDRING HER: Vi peker tilbake til Tokenizer-repoet på HF!
echo "[4/6] Ekstraherer audio codes (Tokenizing)..."
python prepare_data.py \
    --input_jsonl /workspace/data/train.jsonl \
    --output_jsonl /workspace/data/train_with_codes.jsonl \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --device cuda:0

# --- FIXES FOR QWEN TRAINING SCRIPT ---
echo "[INFO] Patcher sft_12hz.py..."

# Fix 1: Øk gradient accumulation (4 -> 8)
sed -i 's/gradient_accumulation_steps=4/gradient_accumulation_steps=8/g' sft_12hz.py

# Fix 2: Legg til project_dir for logging
sed -i 's/log_with="tensorboard"/log_with="tensorboard", project_dir=args.output_model_path/g' sft_12hz.py

# Fix 3: Bruk SDPA istedenfor Flash Attention 2
sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/g' sft_12hz.py

# 5. Start Trening (SFT)
# HER bruker vi den lokale mappen vi lastet ned i steg 2
echo "[5/6] Starter trening..."
accelerate launch sft_12hz.py \
    --init_model_path /workspace/base_model \
    --train_jsonl /workspace/data/train_with_codes.jsonl \
    --output_model_path /workspace/output \
    --batch_size 2 \
    --lr 1e-5 \
    --num_epochs 5 \
    --speaker_name "norsk_taler"

# 6. Last opp til Hugging Face
echo "[6/6] Ser etter resultater for opplasting..."
python /workspace/src/upload_to_hf.py \
    --local_dir "/workspace/output" \
    --repo_id "$HF_REPO_ID"

echo "✅ JOBB FERDIG!"
