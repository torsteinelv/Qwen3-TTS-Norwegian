import os
import argparse
import torch
import torch.nn as nn
import json
import types
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, AutoProcessor
from huggingface_hub import HfApi

# Importerer wrapperen
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("❌ Kunne ikke importere Qwen3TTSModel. Sjekk at filen ligger i mappen.")
    raise

# --- 1. DATASET ---
class TTSDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        self.processor = processor
        
        # Sjekk at filen eksisterer
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Finner ikke dataset: {jsonl_path}")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "audio_codes" in item and "text" in item:
                        self.data.append(item)
                except json.JSONDecodeError:
                    continue

        print(f"✅ Lastet {len(self.data)} eksempler fra {jsonl_path}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # ChatML format
        full_text = f"<|im_start|>assistant\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
        
        # Merk: Pass på at prosessoren ikke trunkerer for tidlig hvis tekstene er lange
        inputs = self.processor(text=full_text, return_tensors="pt")
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "audio_codes": torch.tensor(item["audio_codes"], dtype=torch.long)
        }

    def collate_fn(self, features):
        # Padding for tekst (151671 = tts_pad_token_id for Qwen)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=151671
        )
        
        audio_codes_list = [f["audio_codes"] for f in features]
        max_len = max(c.size(0) for c in audio_codes_list)
        
        # Håndterer både 1D (single codebook) og 2D (multi codebook) audio codes
        sample_dim = audio_codes_list[0].dim()
        num_codebooks = audio_codes_list[0].size(1) if sample_dim > 1 else 1
        
        # ENDRING: Bruker -100 for padding hvis vi skal bruke dette som labels senere,
        # men siden Qwen-koden din tar inn codes direkte, bruker vi 0 her,
        # MEN vi må passe på at loss-funksjonen ignorerer disse.
        # Vi oppretter en separat 'labels' tensor for sikkerhets skyld.
        
        padded_audio_codes = torch.zeros((len(features), max_len, num_codebooks), dtype=torch.long)
        labels = torch.full((len(features), max_len, num_codebooks), -100, dtype=torch.long) # -100 er standard ignore_index

        for i, c in enumerate(audio_codes_list):
            l = c.size(0)
            if sample_dim == 1:
                padded_audio_codes[i, :l, 0] = c
                labels[i, :l, 0] = c
            else:
                padded_audio_codes[i, :l, :c.size(1)] = c
                labels[i, :l, :c.size(1)] = c
                
        # Hvis modellen forventer 2D input for single codebook, behold [batch, seq, 1]
        # Hvis den forventer flat, skviser vi. Din monkeypatch bruker .view(-1, last_dim), så 3D er trygt.
        
        return {
            "input_ids": input_ids, 
            "audio_codes": padded_audio_codes,
            "labels": labels # Sender med labels i tilfelle du vil endre loss-logikken
        }

# --- 2. TRENING ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="/workspace/base_model")
    parser.add_argument("--output_model_path", type=str, default="/workspace/output/test_run")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3) # Økte default til 3
    parser.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))
    
    args, _ = parser.parse_known_args()

    print(f"❄️ Laster prosessor fra {args.init_model_path}...")
    processor = AutoProcessor.from_pretrained(args.init_model_path, trust_remote_code=True)

    print("❄️ Laster basismodell...")
    # Last med bfloat16 for minnebesparelse på Ampere+ GPUer
    wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map="auto" # Automatisk plassering på GPU
    )
    model = wrapper.model.talker 

    # --- MONKEY PATCH ---
    # Oppdatering: Sørger for at den håndterer labels/padding korrekt hvis mulig
    def custom_forward(self, input_ids=None, audio_codes=None, labels=None, **kwargs):
        # Tekst embedding
        raw_text_embeds = self.model.text_embedding(input_ids)
        
        # Henter siste state. Sjekk om 'text_projection' er riktig lagnavn i modellen din.
        talker_hidden_states = self.text_projection(raw_text_embeds)[:, -1, :]
        
        # Flatten inputs for Qwen TTS intern logikk
        flat_audio_codes = audio_codes.view(-1, audio_codes.size(-1))
        
        # Her må vi repetere teksten for å matche antall audio frames
        # Sjekk: audio_codes er [Batch, Seq, Codes]. 
        # Vi må repetere text embed [Batch, Dim] -> [Batch * Seq, Dim]
        # Bruk repeat_interleave på batch-dimensjonen etter å ha utvidet
        
        seq_len = audio_codes.size(1)
        # Utvider [Batch, Dim] -> [Batch, Seq, Dim] -> [Batch*Seq, Dim]
        talker_hidden_states_expanded = talker_hidden_states.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, talker_hidden_states.size(-1))
        
        # Kaller intern finetune funksjon
        # MERK: Hvis forward_sub_talker_finetune ikke tar hensyn til padding (0),
        # vil loss bli forurenset. Sjekk kildekoden til Qwen3TTSModel.
        # Hvis den støtter 'labels' argument med -100, send 'labels.view(...)' istedenfor 'flat_audio_codes' som target.
        
        # Antar her at funksjonen returnerer (logits, loss)
        _, loss = self.forward_sub_talker_finetune(
            flat_audio_codes, 
            talker_hidden_states_expanded
        )
        
        return {"loss": loss}

    model.forward = types.MethodType(custom_forward, model)

    # Aktiver gradient checkpointing for å spare VRAM
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads() # Nødvendig for LoRA + Checkpointing

    # LoRA konfigurasjon
    peft_config = LoraConfig(
        r=32, # Økte rank litt for bedre kapasitet på TTS
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # La til MLP lagene
        lora_dropout=0.05,
        bias="none", 
        task_type=TaskType.FEATURE_EXTRACTION # Eller CAUSAL_LM, avhengig av basen
    )
    
    print("🚀 Aktiverer LoRA...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # Vis hvor mange params vi trener

    dataset = TTSDataset(args.train_jsonl, processor)
    
    t_args = TrainingArguments(
        output_dir=args.output_model_path,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4, # Simulerer større batch
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        bf16=True, # Bruk bfloat16
        save_strategy="epoch",
        save_total_limit=2, # Sparer plass
        remove_unused_columns=False, # VIKTIG for custom dataset
        report_to="tensorboard", # Eller "wandb"
        gradient_checkpointing=True, # VIKTIG
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model, 
        args=t_args, 
        train_dataset=dataset, 
        data_collator=dataset.collate_fn
    )

    print("🔥 Starter trening!")
    trainer.train()

    # --- 3. LAGRING ---
    if trainer.is_world_process_zero():
        print("💾 Lagrer LoRA-adapter...")
        model.save_pretrained(args.output_model_path)
        
        # Lagre README
        readme_content = f"""---
library_name: peft
tags:
- text-to-speech
- qwen
- lora
- norwegian
base_model: {args.init_model_path}
---
# Qwen3-TTS Norsk LoRA
Finetuned on dataset: {os.path.basename(args.train_jsonl)}
"""
        with open(os.path.join(args.output_model_path, "README.md"), "w") as f:
            f.write(readme_content)

        if args.hf_repo_id and os.getenv("HF_TOKEN"):
            try:
                print(f"🚀 Laster opp til {args.hf_repo_id}...")
                api = HfApi(token=os.getenv("HF_TOKEN"))
                api.upload_folder(
                    folder_path=args.output_model_path,
                    repo_id=args.hf_repo_id,
                    repo_type="model"
                )
            except Exception as e:
                print(f"⚠️ Opplasting feilet: {e}")

if __name__ == "__main__":
    train()
