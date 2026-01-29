import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from huggingface_hub import HfApi, login

def run_validation():
    # 1. Hent konfigurasjon
    checkpoint_dir = os.getenv("CHECKPOINT_DIR")
    repo_id = os.getenv("HF_REPO_ID")
    token = os.getenv("HF_TOKEN")

    if not checkpoint_dir:
        print("❌ FEIL: CHECKPOINT_DIR mangler. Skipper validering.")
        return

    print(f"🚀 Starter validering. Laster modell fra: {checkpoint_dir}")

    try:
        login(token=token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = Qwen3TTSModel.from_pretrained(
            checkpoint_dir, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        # 2. Definer tester
        tests = [
            ("validering_norsk.wav", "Hei! Dette er en test av den ferdigtrente modellen for å høre om jeg snakker norsk med Kathrine sin stemme."),
            ("validering_engelsk.wav", "And this is a short test in English to see how the model handles switching languages with the new voice.")
        ]

        api = HfApi()
        print("🎙️ Genererer lyd og laster opp...")

        for fname, text in tests:
            print(f"   - Genererer: '{text[:30]}...'")
            
            wavs, sr = model.generate_custom_voice(
                text=text,
                language="Auto", 
                speaker="norsk_taler" 
            )
            
            sf.write(fname, wavs[0], sr)
            
            target_path = f"final_validation_tests/{fname}"
            print(f"   - Laster opp til: {repo_id}/{target_path}")
            
            api.upload_file(
                path_or_fileobj=fname,
                path_in_repo=target_path,
                repo_id=repo_id
            )

        print("✨ Validering fullført!")

    except Exception as e:
        print(f"⚠️ ADVARSEL: Valideringstesten feilet (men modellen er trygg).")
        print(f"Feilmelding: {e}")

if __name__ == "__main__":
    run_validation()
