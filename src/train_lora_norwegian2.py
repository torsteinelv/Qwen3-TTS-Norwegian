import os
import argparse
import torch
import torch.nn as nn
import json
import types
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, AutoProcessor, TrainerCallback
from huggingface_hub import HfApi

# Sjekk import
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("❌ Kunne ikke importere Qwen3TTSModel.")
    raise

# --- 1. SMART OPPLASTING MED README-FIKS ---
class UploadToHubCallback(TrainerCallback):
    def __init__(self, api, repo_id, output_dir, base_model_path):
        self.api = api
        self.repo_id = repo_id
        self.output_dir = output_dir
        self.base_model_path = base_model_path # Vi trenger denne for README

    def on_save(self, args, state, control, **kwargs):
        """
        Kjøres hver gang Trainer lagrer et checkpoint.
        """
        if self.repo_id and self.api:
            ckpt_name = f"checkpoint-{state.global_step}"
            ckpt_path = os.path.join(self.output_dir, ckpt_name)
            
            # --- FIKS: Skriv en gyldig README.md inn i checkpoint-mappen FØR opplasting ---
            # Dette løser "base_model is not allowed to be empty" feilen
            readme_content = f"""---
base_model: Qwen/Qwen2.5-1.5B-Instruct # Eller navnet på modellen du bruker, fyllstoff fungerer også
library_name: peft
tags:
- text-to-speech
- qwen3-tts
- norwegian
---
# Epoch {state.epoch} (Step {state.global_step})
"""
            # Skriv filen
            try:
                with open(os.path.join(ckpt_path, "README.md"), "w") as f:
                    f.write(readme_content)
            except Exception as e:
                print(f"⚠️ Kunne ikke skrive README i checkpoint: {e}")

            print(f"\n☁️ Arkiverer {ckpt_name} (Epoch {state.epoch}) til Hugging Face...")
            
            try:
                # 1. Last opp historikk
                self.api.upload_folder(
                    folder_path=ckpt_path,
                    repo_id=self.repo_id,
                    path_in_repo=ckpt_name, 
                    repo_type="model",
                    commit_message=f"Archive step {state.global_step} (Epoch {state.epoch})"
                )
                
                # 2. Oppdater hovedmodellen
                self.api.upload_folder(
                    folder_path=ckpt_path,
                    repo_id=self.repo_id,
                    path_in_repo=".", 
                    repo_type="model",
                    commit_message=f"Update main model to step {state.global_step}"
                )
                print(f"✅ {ckpt_name} er trygt lagret i skyen!")
                
            except Exception as e:
                print(f"⚠️ Opplasting feilet: {e}")

# --- 2. DATASET ---
class TTSDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        self.processor = processor
        if not os.path.exists(jsonl_path):
             raise FileNotFoundError(f"Finner ikke filen: {jsonl_path}")
             
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "audio_codes" in item:
                        self.data.append(item)
                except:
                    continue
        print(f"✅ Lastet {len(self.data)} linjer.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"<|im_start|>assistant\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.processor(text=full_text, return_tensors="pt")
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "audio_codes": torch.tensor(item["audio_codes"], dtype=torch.long)
        }

    def collate_fn(self, features):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=151671
        )
        audio_codes = [f["audio_codes"] for f in features]
        max_len = max(c.size(0) for c in audio_codes)
        
        if audio_codes[0].dim() > 1:
            num_codebooks = audio_codes[0].size(1)
            padded_audio_codes = torch.zeros((len(audio_codes), max_len, num_codebooks), dtype=torch.long)
            for i, c in enumerate(audio_codes):
                padded_audio_codes[i, :c.size(0), :] = c
        else:
            padded_audio_codes = torch.zeros((len(audio_codes), max_len, 1), dtype=torch.long)
            for i, c in enumerate(audio_codes):
                padded_audio_codes[i, :c.size(0), 0] = c

        return {"input_ids": input_ids, "audio_codes": padded_audio_codes}

# --- 3. TRENING ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="/workspace/base_model")
    parser.add_argument("--output_model_path", type=str, default="/workspace/output/long_run")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5) 
    parser.add_argument("--num_epochs", type=int, default=15) # 15 Epochs
    parser.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))
    
    args, _ = parser.parse_known_args()

    hf_api = None
    if args.hf_repo_id and os.getenv("HF_TOKEN"):
        hf_api = HfApi(token=os.getenv("HF_TOKEN"))
        print(f"🔗 Opplasting aktivert til: {args.hf_repo_id}")

    print("❄️ Laster prosessor...")
    processor = AutoProcessor.from_pretrained(args.init_model_path, trust_remote_code=True)

    print("❄️ Laster basismodell...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path, 
        dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map="auto"
    )
    model = wrapper.model.talker 

    # --- ORIGINALE 'LOSS 4' LOGIKK ---
    def custom_forward(self, input_ids=None, audio_codes=None, **kwargs):
        raw_text_embeds = self.model.text_embedding(input_ids)
        talker_hidden_states = self.text_projection(raw_text_embeds)[:, -1, :]
        _, loss = self.forward_sub_talker_finetune(
            audio_codes.view(-1, audio_codes.size(-1)), 
            talker_hidden_states.repeat_interleave(audio_codes.size(1), dim=0)
        )
        return {"loss": loss}

    model.forward = types.MethodType(custom_forward, model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="FEATURE_EXTRACTION"
    )
    
    model = get_peft_model(model, peft_config)
    
    dataset = TTSDataset(args.train_jsonl, processor)
    
    t_args = TrainingArguments(
        output_dir=args.output_model_path,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=5,
        bf16=True,
        save_strategy="epoch",       
        save_total_limit=2,          
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True
    )

    # Legg til init_model_path i callbacken
    callbacks = [UploadToHubCallback(
        api=hf_api, 
        repo_id=args.hf_repo_id, 
        output_dir=args.output_model_path,
        base_model_path=args.init_model_path
    )]

    trainer = Trainer(
        model=model, 
        args=t_args, 
        train_dataset=dataset, 
        data_collator=dataset.collate_fn,
        callbacks=callbacks
    )

    print(f"🔥 Starter trening i {args.num_epochs} epochs!")
    trainer.train()

    if trainer.is_world_process_zero():
        print("💾 Lagrer sluttresultat...")
        model.save_pretrained(args.output_model_path)
        processor.save_pretrained(args.output_model_path)
        
        readme_path = os.path.join(args.output_model_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"---\nbase_model: {args.init_model_path}\nlibrary_name: peft\ntags:\n- text-to-speech\n- qwen3-tts\n---")

        if hf_api:
            hf_api.upload_folder(
                folder_path=args.output_model_path,
                repo_id=args.hf_repo_id,
                repo_type="model",
                commit_message="Final model upload"
            )

if __name__ == "__main__":
    train()
