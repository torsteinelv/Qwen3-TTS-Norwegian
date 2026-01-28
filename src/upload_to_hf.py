import os
import argparse
from huggingface_hub import HfApi, login

def upload():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, required=True, help="Mappen som skal lastes opp")
    parser.add_argument("--repo_id", type=str, required=True, help="Navn på repoet på Hugging Face (Brukernavn/Repo)")
    args = parser.parse_args()

    # 1. Hent token og logg inn (Samme metode som Omni-scriptet)
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN miljøvariabel mangler. Kan ikke laste opp.")
        return

    print(f"🔐 Logger inn på Hugging Face...")
    try:
        login(token=token, add_to_git_credential=True)
        print("✅ Innlogging suksess!")
    except Exception as e:
        print(f"⚠️ Innlogging feilet (men prøver likevel med API-token): {e}")

    # 2. Start opplasting
    print(f"🚀 Starter opplasting av '{args.local_dir}' til 'https://huggingface.co/{args.repo_id}'...")
    
    try:
        api = HfApi(token=token)
        
        # Opprett repo hvis det ikke finnes (private som default)
        print("   Checking/Creating repo...")
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=True)
        
        # Last opp hele mappen
        print("   Uploading folder...")
        api.upload_folder(
            folder_path=args.local_dir,
            repo_id=args.repo_id,
            repo_type="model"
        )
        print("✅ Opplasting fullført! Modellen ligger nå på Hugging Face.")
        
    except Exception as e:
        print(f"❌ Noe gikk galt under opplasting: {e}")
        # Kaster feilen videre slik at Kubernetes skjønner at jobben feilet
        raise e

if __name__ == "__main__":
    upload()
