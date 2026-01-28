#!/bin/bash
set -e

echo "=================================================="
echo "   NORWEGIAN QWEN3-TTS FINETUNER (20GB VRAM)      "
echo "=================================================="

# 1. Klon Qwen3-TTS repoet (vi trenger trenings-scriptene derfra)
if [ ! -d "Qwen3-TTS" ]; then
    echo "[1/4] Cloner Qwen3-TTS repo..."
    git clone https://github.com/QwenLM/Qwen3-TTS.git
else
    echo "[1/4] Qwen3-TTS repo finnes allerede."
fi

# Gå til finetuning mappen for å ha tilgang til scripts
cd Qwen3-TTS/finetuning

# 2. Generer datasett (WAV + JSONL)
echo "[2/4] Bygger datasett fra NPSC..."
python /workspace/src/data_npsc.py

# 3. Preprosesser data (Audio -> Codes)
# Dette bruker prepare_data.py fra Qwen3 repoet
echo "[3/4] Ekstraherer audio codes (Tokenizing)..."
python prepare_data.py \
    --input_jsonl /workspace/data/train.jsonl \
    --output_jsonl /workspace/data/train_with_codes.jsonl \
    --model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --device cuda:0

# 4. Start Trening (SFT)
# Innstillinger optimalisert for 20GB VRAM:
# - Batch size 2 (øker stabilitet over 1)
# - Gradient accumulation 8 (effektiv batch size = 16)
# - Epochs 5 (Nok til å høre forskjell)
echo "[4/4] Starter trening..."
accelerate launch sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl /workspace/data/train_with_codes.jsonl \
    --output_model_path /workspace/output \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 1e-5 \
    --num_epochs 5 \
    --save_steps 100 \
    --speaker_name "norsk_taler"

echo "✅ TRENING FERDIG! Sjekk /workspace/output for resultater."
