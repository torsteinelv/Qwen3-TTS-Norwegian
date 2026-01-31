import os
import argparse
import torch
import json
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import TrainingArguments, Trainer, AutoProcessor
# Vi bruker Qwen3TTSModel fra biblioteket, men bytter prosessoren til AutoProcessor for å unngå feilen
from qwen_tts import Qwen3TTSModel 

# --- 1. DATASET KLASSE ---
class TTSDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        self.processor = processor
        
        # Last inn data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if os.path.exists(item["audio_path"]):
                    self.data.append(item)
                else:
                    print(f"⚠️ Mangler fil: {item['audio_path']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio_path"]
        text = item["text"]
        
        # Last lyd
        audio, sr = sf.read(audio_path)
        # Qwen forventer ofte spesifikk sample rate, prosessoren håndterer dette vanligvis,
        # men sjekk at lydfilene dine er pre-samplet (f.eks 16k eller 24k) i prepare-steget.
        
        # Prosesser tekst og lyd
        # Merk: Input-formatet varierer litt mellom versjoner, dette er standard HF-måte:
        inputs = self.processor(
            text=text,
            audios=audio, 
            return_tensors="pt", 
            sampling_rate=sr,
            padding=True
        )
        
        # Vi må returnere en dict som fjerner batch-dimensjonen (siden collator legger den på igjen)
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def collate_fn(self, features):
        # Bruk prosessorens innebygde padding hvis mulig, ellers manuell
        return self.processor.pad(features, return_tensors="pt", padding=True)

# --- 2. LAGRING OG OPPLASTING ---
def save_and_upload(trainer, args, accelerator):
    """
    Sikker lagring som fikser 'text_projection' og README-metadata.
    """
    if accelerator.is_main_process:
        print("💾 Starter lagring...")
        
        # 1. Lagre Adapter (LoRA)
        trainer.save_model(args.output_model_path)
        
        # 2. Lagre text_projection manuelt (Kritisk for ÆØÅ)
        # Dette laget trenes ofte utenfor LoRA-configen
        tp_path = os.path.join(args.output_model_path, "text_projection.bin")
        try:
            torch.save(trainer.model.talker.text_projection.state_dict(), tp_path)
            print(f"✅ Text Projection lagret separat: {tp_path}")
        except AttributeError:
            print("⚠️ Advarsel: Fant ikke model.talker.text_projection. Hopper over.")

        # 3. FIKS README.METADATA (Dette feilet sist!)
        readme_path = os.path.join(args.output_model_path, "README.md")
        readme_content = f"""---
base_model: {args.init_model_path}
library_name: peft
license: other
tags:
- text-to-speech
- qwen3-tts
- lora
- generated_from_trainer
---

# Qwen3-TTS Norsk Finetuning (LoRA)

Dette er en LoRA-adapter for Qwen3-TTS trent på norsk data.
- **Base Model:** {args.init_model_path}
- **Epochs:** {args.num_epochs}
- **Learning Rate:** {args.lr}
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("✅ README.md fikset (metadata lagt til).")

        # 4. Last opp til Hugging Face (hvis token finnes)
        if os.getenv("HF_TOKEN"):
            repo_id = os.getenv('HF_REPO_ID')
            print(f"🚀 Laster opp til Hugging Face: {repo_id}...")
            try:
                # Prøv Trainer sin innebygde push først
                trainer.push_to_hub()
            except Exception as e:
                print(f"⚠️ Trainer push feilet ({e}). Prøver direkte API opplasting...")
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    api.upload_folder(
                        folder_path=args.output_model_path,
                        repo_id=repo_id,
                        path_in_repo=".",
                    )
                    print("🎉 Opplasting fullført via HfApi!")
                except Exception as e2:
                    print(f"❌ Opplasting feilet totalt: {e2}")
                    print("DATAEN ER LAGRET LOKALT PÅ DISK. DU MISTER INGENTING.")
        else:
            print("ℹ️ Ingen HF_TOKEN, hopper over opplasting.")

# --- 3. TRENINGSLØKKE ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="/workspace/base_model")
    parser.add_argument("--output_model_path", type=str, default="/workspace/output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4) # Default 4, kan overstyres i YAML
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=15)
    
    # LoRA parametere
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Ignorer ukjente argumenter (f.eks speaker_name som var i det gamle scriptet)
    args, unknown = parser.parse_known_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16", # Ampere GPU støtter bf16
        log_with="tensorboard",
        project_dir=args.output_model_path
    )

    if accelerator.is_main_process:
        print(f"🚀 Starter LoRA-trening på {accelerator.device}")
        print(f"📂 Data: {args.train_jsonl}")
        if unknown:
            print(f"ℹ️ Ignorerer ubrukte argumenter: {unknown}")

    # 1. Last Modell og Prosessor
    # FIX: Bruk AutoProcessor istedenfor Qwen3TTSProcessor som feilet
    try:
        processor = AutoProcessor.from_pretrained(args.init_model_path, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ AutoProcessor feilet, prøver Qwen2AudioProcessor... {e}")
        from transformers import Qwen2AudioProcessor
        processor = Qwen2AudioProcessor.from_pretrained(args.init_model_path, trust_remote_code=True)

    model = Qwen3TTSModel.from_pretrained(
        args.init_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Aktiver Gradient Checkpointing for å spare VRAM
    model.talker.model.gradient_checkpointing_enable() 

    # 2. Setup LoRA Config
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        # Her treffer vi ALLE viktige lag i transformeren
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION" # Eller CAUSAL_LM, avhengig av base-klassen
    )
    
    # Påfør LoRA
    model.talker.model = get_peft_model(model.talker.model, peft_config)
    
    # VIKTIG: Tving text_projection til å være trenbar (utenfor LoRA)
    # Dette er "inngangsdøren" for tekst, viktig for norske bokstaver
    for param in model.talker.text_projection.parameters():
        param.requires_grad = True

    if accelerator.is_main_process:
        model.talker.model.print_trainable_parameters()

    # 3. Dataset
    dataset = TTSDataset(args.train_jsonl, processor)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_model_path,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        logging_dir=f"{args.output_model_path}/logs",
        report_to="tensorboard",
        bf16=True,
        dataloader_num_workers=4,
        push_to_hub=False, # Vi håndterer opplasting manuelt i save_and_upload
        save_total_limit=2,
        remove_unused_columns=False 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataset.collate_fn
    )

    # 5. Start Trening
    print("🔥 Starter trening nå...")
    trainer.train()
    
    # 6. Lagre og Last opp
    save_and_upload(trainer, args, accelerator)

if __name__ == "__main__":
    train()
