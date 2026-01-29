import os
import json
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import login
import numpy as np

# Konfigurasjon
OUTPUT_DIR = "/workspace/data_wavs"
JSONL_PATH = "train_raw.jsonl"
TARGET_SPEAKER = "Kathrine Engan"
MAX_SAMPLES = 2000  # 2000 setninger gir ca 2-4 timer lyd, som er perfekt for fine-tuning
REF_FILENAME = "reference_master.wav" # Den ene filen som styrer alt

def build_librivox_dataset():
    # 1. Autentisering
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print(f"✅ Logget inn på Hugging Face.")
    else:
        print("⚠️ ADVARSEL: HF_TOKEN ikke funnet. Kan feile hvis dataset er privat.")

    # 2. Setup mapper
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Lagrer lydfiler til: {OUTPUT_DIR}")

    # 3. Last datasett i Streaming Mode (laster ikke ned alt, bare det vi ber om)
    print("📡 Kobler til NbAiLab/nb_librivox (Streaming)...")
    try:
        ds = load_dataset("NbAiLab/nb_librivox", split="train", streaming=True)
    except Exception as e:
        print(f"❌ Klarte ikke laste dataset: {e}")
        return

    jsonl_data = []
    count = 0
    ref_saved = False
    
    print(f"🔍 Starter søk etter speaker: '{TARGET_SPEAKER}'...")

    for i, sample in enumerate(ds):
        # Sikkerhetsstopp
        if count >= MAX_SAMPLES:
            print(f"🛑 Nådde maks antall samples ({MAX_SAMPLES}). Stopper.")
            break

        # Sjekk metadata for speaker. 
        # NbAiLab/nb_librivox har ofte speaker-info i 'metadata'-feltet eller 'speaker_name'
        # Vi sjekker begge for sikkerhets skyld.
        speaker_name = sample.get("speaker_name", "")
        if not speaker_name and "metadata" in sample:
            speaker_name = sample["metadata"].get("speaker_name", "") # Fallback
        
        # Hvis vi ikke finner navnet, hopp over (eller print for debug første gang)
        if i == 0:
            print(f"ℹ️ Første rad nøkler: {sample.keys()}")
            print(f"ℹ️ Eksempel speaker: {speaker_name}")

        # FILTRERING: Er dette Kathrine?
        if TARGET_SPEAKER.lower() not in str(speaker_name).lower():
            continue

        # Hent lyd og tekst
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"]

        # Hopp over tomme/korte filer (mindre enn 1 sekund)
        if len(audio_array) < sr * 1.0:
            continue

        # Lagre filen lokalt
        filename = f"{OUTPUT_DIR}/ke_{count:05d}.wav"
        
        # Vi må konvertere til float32 hvis det ikke allerede er det, SoundFile er kresen
        sf.write(filename, audio_array, sr)

        # LOGIKK FOR REFERANSEFIL (Viktig fra Issue #39)
        # Vi velger den første filen vi finner som master-referanse.
        # Alle treningsdata skal peke på denne ene filen for å holde stemmen stabil.
        ref_path = f"{OUTPUT_DIR}/{REF_FILENAME}"
        
        if not ref_saved:
            # Sjekk at den er av god lengde (mellom 5 og 10 sekunder er ideelt for ref)
            duration = len(audio_array) / sr
            if 4.0 < duration < 12.0:
                sf.write(ref_path, audio_array, sr)
                ref_saved = True
                print(f"✅ MASTER REFERANSE LAGRET: {ref_path} (Varighet: {duration:.2f}s)")
            else:
                # Hvis den er for kort/lang til å være ref, bruker vi den bare som training data
                # og venter på neste fil til å bli ref.
                pass
        
        # Hvis vi ikke har funnet en god ref enda, kan vi ikke legge til data i jsonl
        # for da mangler vi ref_path. Vi skipper til vi har en ref.
        if not ref_saved:
            continue

        # Legg til i listen
        entry = {
            "audio": filename,
            "text": text,
            "ref_audio": ref_path  # VIKTIG: Statisk referanse
        }
        jsonl_data.append(entry)
        
        count += 1
        if count % 100 == 0:
            print(f"✅ Prosessert {count} linjer...")

    # 4. Lagre JSONL
    print(f"💾 Skriver {len(jsonl_data)} linjer til {JSONL_PATH}...")
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for line in jsonl_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("🎉 Ferdig! Dataset er klart for prepare_data.py")

if __name__ == "__main__":
    build_librivox_dataset()
