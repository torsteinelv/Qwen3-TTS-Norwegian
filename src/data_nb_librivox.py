import os
import json
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import login

# --- KONFIGURASJON ---
# Vi bruker /workspace/data som vi lagde i Dockerfilen
OUTPUT_DIR = "/workspace/data" 
JSONL_PATH = "train_raw.jsonl"
TARGET_SPEAKER = "Kathrine Engan"
MAX_SAMPLES = 2000  # 2000 linjer gir solid trening uten å ta evigheter
REF_FILENAME = "reference_master.wav" # Den ENE filen som skal være referanse for alle

def build_librivox_dataset():
    # 1. Autentisering
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print(f"✅ Logget inn på Hugging Face.")
    else:
        print("⚠️ ADVARSEL: HF_TOKEN mangler. Scriptet kan feile.")

    # 2. Opprett mappe hvis den mangler
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📂 Lagrer lydfiler til: {OUTPUT_DIR}")

    # 3. Last datasett (Streaming = laster ned mens vi går)
    print("📡 Kobler til NbAiLab/nb_librivox...")
    try:
        ds = load_dataset("NbAiLab/nb_librivox", split="train", streaming=True)
    except Exception as e:
        print(f"❌ Kritisk feil ved lasting av dataset: {e}")
        return

    jsonl_data = []
    count = 0
    ref_saved = False
    
    print(f"🔍 Starter filtrering. Ser etter: '{TARGET_SPEAKER}'")

    for i, sample in enumerate(ds):
        if count >= MAX_SAMPLES:
            print(f"🛑 Nådde {MAX_SAMPLES} samples. Ferdig med nedlasting.")
            break

        # Hent speaker fra metadata (LibriVox-strukturen kan variere litt)
        speaker_name = sample.get("speaker_name", "")
        if not speaker_name and "metadata" in sample:
            speaker_name = sample["metadata"].get("speaker_name", "")

        # --- FILTRERING ---
        # Vi vil KUN ha Kathrine. Alle andre kastes.
        if TARGET_SPEAKER.lower() not in str(speaker_name).lower():
            continue

        # Hent lyd og tekst
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"]

        # Kast filer som er kortere enn 1 sekund (søppel-data)
        if len(audio_array) < sr * 1.0:
            continue

        # Lagre WAV-filen lokalt
        filename = f"{OUTPUT_DIR}/ke_{count:05d}.wav"
        sf.write(filename, audio_array, sr)

        # --- REFERANSE-LOGIKK (CRITICAL FIX) ---
        # Vi trenger ÉN god fil som fungerer som "anker" for hele treningen.
        # Vi lagrer den første gode filen vi finner (mellom 4 og 12 sekunder) som master.
        ref_path = f"{OUTPUT_DIR}/{REF_FILENAME}"
        
        if not ref_saved:
            duration = len(audio_array) / sr
            if 4.0 < duration < 12.0:
                sf.write(ref_path, audio_array, sr)
                ref_saved = True
                print(f"✅ MASTER REFERANSE LAGRET: {ref_path} (Varighet: {duration:.2f}s)")
            else:
                # Hvis filen er for kort/lang til å være master-ref, hopper vi over den
                # i påvente av en bedre kandidat før vi begynner å skrive til JSONL.
                continue
        
        # Hvis vi ikke har en ref enda, kan vi ikke starte treningen.
        if not ref_saved:
            continue

        # Legg til i listen. Merk: ref_audio peker ALLTID på samme fil.
        entry = {
            "audio": filename,
            "text": text,
            "ref_audio": ref_path 
        }
        jsonl_data.append(entry)
        
        count += 1
        if count % 100 == 0:
            print(f"✅ Prosessert {count} / {MAX_SAMPLES} linjer...")

    # 4. Skriv JSONL-filen
    print(f"💾 Skriver {len(jsonl_data)} linjer til {JSONL_PATH}...")
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for line in jsonl_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("🎉 Dataset ferdig bygget! Klar for prepare_data.py")

if __name__ == "__main__":
    build_librivox_dataset()
