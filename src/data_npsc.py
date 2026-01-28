import os
import json
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

# --- KONFIGURASJON ---
DATASET_ID = "NbAiLab/NPSC"
CONFIG = "16K_mp3_bokmaal" 
OUTPUT_DIR = "/workspace/data"
AUDIO_SUBDIR = "audio_files"
JSONL_FILE = "train.jsonl"

# Qwen3-TTS standard
TARGET_SR = 24000  
MIN_DURATION = 1.5 
MAX_DURATION = 15.0
MAX_SAMPLES = 5000  # Juster opp når du ser at det virker

print(f"--- STARTER NEDLASTING AV {DATASET_ID} ---")

os.makedirs(os.path.join(OUTPUT_DIR, AUDIO_SUBDIR), exist_ok=True)

# Laster dataset (streaming mode for å spare disk i starten)
dataset = load_dataset(DATASET_ID, CONFIG, split="train", streaming=True, trust_remote_code=True)

data_entries = []
success_count = 0
ref_audio_path = None

for i, sample in tqdm(enumerate(dataset)):
    if success_count >= MAX_SAMPLES:
        break

    try:
        text = sample.get("text", "")
        if not text: continue
        
        # Enkel tekstvask
        text = text.replace("<hesitation>", "").strip()
        
        audio_array = sample["audio"]["array"]
        orig_sr = sample["audio"]["sampling_rate"]

        # Lengdesjekk (sekunder)
        duration = len(audio_array) / orig_sr
        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue

        # Resample til 24kHz (Qwen3 krav)
        if orig_sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=TARGET_SR)

        # Lagre fil
        filename = f"npsc_{success_count}.wav"
        abs_audio_path = os.path.abspath(os.path.join(OUTPUT_DIR, AUDIO_SUBDIR, filename))
        sf.write(abs_audio_path, audio_array, TARGET_SR)

        # Vi bruker den FØRSTE gyldige filen som referanse-lyd for alle.
        # Dette tvinger modellen til å lære språket, ikke bytte stemme hele tiden.
        if ref_audio_path is None:
            ref_audio_path = abs_audio_path
            print(f"Set reference audio to: {ref_audio_path}")

        # Qwen3-TTS Format
        entry = {
            "audio": abs_audio_path,
            "text": text,
            "ref_audio": ref_audio_path 
        }
        
        data_entries.append(entry)
        success_count += 1

    except Exception as e:
        print(f"Error on sample {i}: {e}")

# Lagre JSONL
out_file = os.path.join(OUTPUT_DIR, JSONL_FILE)
with open(out_file, "w", encoding="utf-8") as f:
    for entry in data_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Ferdig! Lagret {len(data_entries)} linjer til {out_file}")
