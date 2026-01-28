import os
import argparse
from huggingface_hub import HfApi

def upload():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, required=True, help="Mappen som skal lastes opp")
    parser.add_argument("--repo_id", type=str, required=True, help="Navn på repoet på Hugging Face (Brukernavn/Repo)")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN miljøvariabel mangler. Kan ikke laste opp.")
        return

    print(f"🚀 Starter opplasting av '{args.local_dir}' til 'https://huggingface.co/{args.repo_id}'...")
    
    try:
        api = HfApi(token=token)
        
        # Opprett repo hvis det ikke finnes (private som default for sikkerhets skyld)
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=True)
        
        # Last opp hele mappen
        api.upload_folder(
            folder_path=args.local_dir,
            repo_id=args.repo_id,
            repo_type="model"
        )
        print("✅ Opplasting fullført!")
        
    except Exception as e:
        print(f"❌ Noe gikk galt under opplasting: {e}")

if __name__ == "__main__":
    upload()
