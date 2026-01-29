#!/bin/bash
set -e

# 1. Hent selve trenings-scriptene (de følger ikke med pip-pakken)
echo "📦 Cloner Qwen3-TTS repo for å få tak i trenings-koden..."
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning

# 2. Logg inn på HF (Tokenet hentes fra env variabler i k8s)
huggingface-cli login --token $HF_TOKEN

# 3. Kjør ditt Python-script som bygger dataset (LibriVox)
# (Antar at du legger build_dataset.py i /workspace/src/ og flytter den hit,
# eller kjører den direkte)
echo "📚 Bygger dataset..."
python3 /workspace/src/build_dataset.py

# 4. Prepare data (Lager 12Hz codes)
echo "⚙️ Prosesserer data..."
python3 prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# 5. Start trening
echo "🚀 Starter trening..."
python3 sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path /workspace/output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 2 \
  --speaker_name norsk_taler

# 6. Last opp
echo "📤 Laster opp..."
# (Legg inn opplastingskoden din her eller kjør et python-script)
