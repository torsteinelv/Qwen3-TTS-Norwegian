import os
import argparse
import torch
import json
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import TrainingArguments, Trainer
from qwen_tts import Qwen3TTSModel, Qwen3TTSProcessor

# ... (Antar at importene og Dataset-klassen din er lik som før i toppen) ...
# ... (La Dataset-klassen og collate_fn stå som de er) ...

def save_and_upload(trainer, args, accelerator):
    """
    Hjelpefunksjon for å lagre og laste opp trygt med riktig metadata.
    """
    # 1. Lagre lokalt først
    trainer.save_model(args.output_model_path)
    trainer.save_state()
    
    # 2. Lagre text_projection manuelt (viktig for ÆØÅ)
    if accelerator.is_main_process:
        tp_path = os.path.join(args.output_model_path, "text_projection.bin")
        torch.save(trainer.model.talker.text_projection.state_dict(), tp_path)
        print(f"✅ Text Projection lagret: {tp_path}")

        # 3. FIKS README.md (Dette er delen som feilet sist!)
        readme_path = os.path.join(args.output_model_path, "README.md")
        
        # Vi lager innholdet manuelt for å være 100% sikre på at HF godtar det
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

## Detaljer
- **Base Model:** {args.init_model_path}
- **Epochs:** {args.num_epochs}
- **Batch Size:** {args.batch_size}
- **Learning Rate:** {args.lr}

## Hvordan bruke
Se `test_cpu_lora.py` i GitHub-repoet for instruksjoner om hvordan du laster denne modellen sammen med base-modellen.
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("✅ README.md fikset med korrekt metadata.")

        # 4. Last opp til Hugging Face (hvis token finnes)
        if os.getenv("HF_TOKEN"):
            print(f"🚀 Laster opp til Hugging Face: {os.getenv('HF_REPO_ID')}...")
            try:
                trainer.push_to_hub()
                print("🎉 Opplasting fullført!")
            except Exception as e:
                print(f"⚠️ Automatisk opplasting feilet (men filene er lagret lokalt): {e}")
                # Fallback: Prøv å bruke HfApi direkte hvis Trainer feiler
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    repo_id = os.getenv('HF_REPO_ID')
                    if repo_id:
                        api.upload_folder(
                            folder_path=args.output_model_path,
                            repo_id=repo_id,
                            path_in_repo=".",
                        )
                        print("🎉 Opplasting fullført via HfApi fallback!")
                except Exception as e2:
                    print(f"❌ Fallback feilet også: {e2}")
        else:
            print("ℹ️ Ingen HF_TOKEN funnet, hopper over opplasting.")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Lagt til for logging
    parser.add_argument("--logging_steps", type=int, default=10)
    
    args = parser.parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path  # Fikser logging-feil
    )

    if accelerator.is_main_process:
        print(f"🚀 Starter trening på {accelerator.device}")
        print(f"📂 Data: {args.train_jsonl}")

    # 1. Last modell
    model = Qwen3TTSModel.from_pretrained(
        args.init_model_path, 
        torch_dtype=torch.bfloat16
    )
    
    # 2. Aktiver Gradient Checkpointing (Sparer minne!)
    model.talker.model.gradient_checkpointing_enable() 

    # 3. Setup LoRA
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        model.talker.model = get_peft_model(model.talker.model, peft_config)
        
        # Tving text_projection til å være trenbar (viktig for norsk!)
        for param in model.talker.text_projection.parameters():
            param.requires_grad = True

        if accelerator.is_main_process:
            model.talker.model.print_trainable_parameters()

    # 4. Dataset
    # (Antar at DanishTTSDataset er definert lenger opp i filen din)
    # dataset = DanishTTSDataset(args.train_jsonl, model.processor, model.config)
    
    # -- MIDLERTIDIG FIX FOR Å FÅ KODEN TIL Å KJØRE UTEN DATASET KLASSEN HER --
    # Du må lime inn Dataset-klassen din her hvis du overskriver hele filen!
    from dataset import TTSDataset # Eller hva du kalte den lokalt
    dataset = TTSDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=model.tokenizer,
    )
    # -----------------------------------------------------------------------

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_model_path,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        logging_dir=f"{args.output_model_path}/logs",
        report_to="tensorboard",
        bf16=True,
        dataloader_num_workers=4,
        push_to_hub=True,                    # Vi prøver auto-push
        hub_model_id=os.getenv("HF_REPO_ID"), # Henter repo navn fra env
        hub_strategy="every_save",           # Last opp hver gang vi lagrer
        save_total_limit=2,                  # Spar plass, behold kun 2 siste
        remove_unused_columns=False          # Viktig for custom datasets
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataset.collate_fn
    )

    # 6. Start Trening
    trainer.train()
    
    # 7. Lagre og Last opp (Med fiksen vår)
    save_and_upload(trainer, args, accelerator)

if __name__ == "__main__":
    train()
