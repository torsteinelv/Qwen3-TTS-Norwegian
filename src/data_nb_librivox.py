import os
import json
import csv
import soundfile as sf
from huggingface_hub import hf_hub_download, login

# --- KONFIGURASJON ---
# Endre til "/workspace/data" og MAX_SAMPLES=2000 før du bygger Docker!
OUTPUT_DIR = "/workspace/data" 
JSONL_PATH = "train_raw.jsonl"
TARGET_SPEAKER = "Kathrine Engan"
MAX_SAMPLES = 1000
REF_FILENAME = "reference_master.wav"
REPO_ID = "NbAiLab/nb-librivox"

def build_librivox_dataset():
    # 1. Autentisering
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print(f"✅ Logget inn på Hugging Face.")
    else:
        print("⚠️ ADVARSEL: HF_TOKEN mangler.")

    # 2. Opprett mappe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Lagrer lydfiler til: {OUTPUT_DIR}")

    # 3. Last ned metadata manuelt (Bypasser datasets-biblioteket)
    print(f"📡 Henter metadata.csv fra {REPO_ID}...")
    try:
        meta_local_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename="metadata.csv", 
            repo_type="dataset", 
            token=token
        )
        print(f"✅ Metadata lastet ned til: {meta_local_path}")
    except Exception as e:
        print(f"❌ Klarte ikke hente metadata: {e}")
        return

    jsonl_data = []
    count = 0
    ref_saved = False
    
    print(f"🔍 Starter filtrering av CSV for: '{TARGET_SPEAKER}'")

    # 4. Les CSV-filen linje for linje
    with open(meta_local_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Sjekk at vi har de kolonnene vi forventer (basert på din sjekk)
        if 'file_name' not in reader.fieldnames or 'speaker_name' not in reader.fieldnames:
            print(f"❌ Feil kolonner i CSV! Fant: {reader.fieldnames}")
            return

        for row in reader:
            if count >= MAX_SAMPLES:
                print(f"🛑 Nådde {MAX_SAMPLES} samples.")
                break

            # Sjekk Speaker
            speaker_name = row['speaker_name']
            if TARGET_SPEAKER.lower() not in str(speaker_name).lower():
                continue

            # Sjekk filnavn
            repo_filename = row['file_name']
            text = row['text']

            # 5. Last ned selve lydfilen
            try:
                # Laster ned filen til cache og får stien
                local_audio_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=repo_filename,
                    repo_type="dataset",
                    token=token
                )
                
                # Les lyden med SoundFile for å sjekke lengde/format
                audio_array, sr = sf.read(local_audio_path)
                
                # Kast filer som er for korte
                duration = len(audio_array) / sr
                if duration < 1.0:
                    continue

                # Lagre til vår mappe (konverterer evt. mp3 til wav automatisk ved lagring)
                filename = f"{OUTPUT_DIR}/ke_{count:05d}.wav"
                sf.write(filename, audio_array, sr)

                # --- REFERANSE-LOGIKK ---
                ref_path = f"{OUTPUT_DIR}/{REF_FILENAME}"
                
                # Vi leter etter den perfekte referansefilen (mellom 4 og 10 sekunder)
                if not ref_saved:
                    if 4.0 < duration < 10.0:
                        sf.write(ref_path, audio_array, sr)
                        ref_saved = True
                        print(f"✅ MASTER REFERANSE LAGRET: {ref_path} (Varighet: {duration:.2f}s)")
                    else:
                        # Vent på en bedre kandidat før vi legger til noe i jsonl
                        continue
                
                # Legg til i dataset-listen
                entry = {
                    "audio": filename,
                    "text": text,
                    "ref_audio": ref_path 
                }
                jsonl_data.append(entry)
                
                count += 1
                if count % 10 == 0:
                    print(f"✅ Prosessert {count} filer...")

            except Exception as e:
                print(f"⚠️ Feil ved nedlasting av {repo_filename}: {e}")
                continue

    # 6. Skriv JSONL-filen
    print(f"💾 Skriver {len(jsonl_data)} linjer til {JSONL_PATH}...")
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for line in jsonl_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("🎉 Dataset ferdig bygget! Klar for prepare_data.py")

if __name__ == "__main__":
    build_librivox_dataset()
