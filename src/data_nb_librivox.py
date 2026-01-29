import os
import json
import csv
import soundfile as sf
import librosa
import numpy as np
from huggingface_hub import hf_hub_download, login

# --- KONFIGURASJON ---
OUTPUT_DIR = "/workspace/data" 
JSONL_PATH = "train_raw.jsonl"
TARGET_SPEAKER = "Kathrine Engan"
# MAX_SAMPLES = 2000  <-- KOMMENTERT UT (Vi vil ha alt!)
REF_FILENAME = "reference_master.wav"
REPO_ID = "NbAiLab/nb-librivox"
TARGET_SR = 24000

def build_librivox_dataset():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print(f"✅ Logget inn på Hugging Face.")
    else:
        print("⚠️ ADVARSEL: HF_TOKEN mangler.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Lagrer lydfiler til: {OUTPUT_DIR}")

    print(f"📡 Henter metadata.csv fra {REPO_ID}...")
    try:
        meta_local_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename="metadata.csv", 
            repo_type="dataset", 
            token=token
        )
    except Exception as e:
        print(f"❌ Klarte ikke hente metadata: {e}")
        return

    jsonl_data = []
    count = 0
    ref_saved = False
    
    print(f"🔍 Starter filtrering av CSV for: '{TARGET_SPEAKER}' (Mål: {TARGET_SR}Hz)")
    print("🚀 KJØRER I FULL-MODUS (Ingen begrensning på antall filer)")

    with open(meta_local_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # --- FJERNET SPERREN HER ---
            # if count >= MAX_SAMPLES:
            #     print(f"🛑 Nådde {MAX_SAMPLES} samples.")
            #     break
            # ---------------------------

            speaker_name = row.get('speaker_name', '')
            if TARGET_SPEAKER.lower() not in str(speaker_name).lower():
                continue

            repo_filename = row['file_name']
            text = row['text']

            try:
                local_audio_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=repo_filename,
                    repo_type="dataset",
                    token=token
                )
                
                # Resampling til 24kHz
                audio_array, sr = librosa.load(local_audio_path, sr=TARGET_SR)
                
                duration = len(audio_array) / sr
                if duration < 1.0 or duration > 20.0:
                    continue

                filename = f"{OUTPUT_DIR}/ke_{count:05d}.wav"
                sf.write(filename, audio_array, sr)

                ref_path = f"{OUTPUT_DIR}/{REF_FILENAME}"
                if not ref_saved:
                    if 4.0 < duration < 10.0:
                        sf.write(ref_path, audio_array, sr)
                        ref_saved = True
                        print(f"✅ MASTER REFERANSE LAGRET (24kHz): {ref_path}")
                    else:
                        continue 
                
                entry = {
                    "audio": filename,
                    "text": text,
                    "ref_audio": ref_path 
                }
                jsonl_data.append(entry)
                
                count += 1
                if count % 50 == 0:
                    print(f"✅ Prosessert {count} filer...")

            except Exception as e:
                print(f"⚠️ Feil ved {repo_filename}: {e}")
                continue

    print(f"💾 Skriver {len(jsonl_data)} linjer til {JSONL_PATH}...")
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for line in jsonl_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"🎉 Dataset ferdig bygget! Totalt {count} filer klar for trening.")

if __name__ == "__main__":
    build_librivox_dataset()
