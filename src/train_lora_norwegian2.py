import os
import argparse
import torch
import torch.nn as nn
import json
import types
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, AutoProcessor
from huggingface_hub import HfApi

# Importerer wrapperen
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("❌ Kunne ikke importere Qwen3TTSModel.")
    raise

# --- 1. DATASET ---
class TTSDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        self.processor = processor
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if "audio_codes" in item:
                    self.data.append(item)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # ChatML format er kritisk for Qwen3
        full_text = f"<|im_start|>assistant\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.processor(text=full_text, return_tensors="pt")
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "audio_codes": torch.tensor(item["audio_codes"], dtype=torch.long)
        }

    def collate_fn(self, features):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=151671 # tts_pad_token_id
        )
        
        audio_codes = [f["audio_codes"] for f in features]
        max_len = max(c.size(0) for c in audio_codes)
        num_codebooks = audio_codes[0].size(1) if audio_codes[0].dim() > 1 else 16
        
        padded_audio_codes = torch.zeros((len(audio_codes), max_len, num_codebooks), dtype=torch.long)
        for i, c in enumerate(audio_codes):
            l = c.size(0)
            if c.dim() == 1:
                padded_audio_codes[i, :l, 0] = c
            else:
                padded_audio_codes[i, :l, :c.size(1)] = c
                
        return {"input_ids": input_ids, "audio_codes": padded_audio_codes}

# --- 2. TRENING ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="/workspace/base_model")
    parser.add_argument("--output_model_path", type=str, default="/workspace/output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))
    
    args, _ = parser.parse_known_args()

    print("❄️ Laster prosessor...")
    processor = AutoProcessor.from_pretrained(args.init_model_path, trust_remote_code=True, fix_mistral_regex=True)

    print("❄️ Laster basismodell...")
    wrapper = Qwen3TTSModel.from_pretrained(args.init_model_path, dtype=torch.bfloat16, trust_remote_code=True)
    model = wrapper.model.talker 

    # Monkey patch for å omgå NoneType-error under trening
    def custom_forward(self, input_ids=None, audio_codes=None, **kwargs):
        raw_text_embeds = self.model.text_embedding(input_ids)
        talker_hidden_states = self.text_projection(raw_text_embeds)[:, -1, :]
        
        # Bruker den interne finetune-funksjonen for stabilitet
        _, loss = self.forward_sub_talker_finetune(
            audio_codes.view(-1, audio_codes.size(-1)), 
            talker_hidden_states.repeat_interleave(audio_codes.size(1), dim=0)
        )
        return {"loss": loss}

    model.forward = types.MethodType(custom_forward, model)

    # LoRA konfigurasjon
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", 
        task_type="FEATURE_EXTRACTION"
    )
    
    print("🚀 Aktiverer LoRA...")
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
        remove_unused_columns=False,
        report_to="none",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model, 
        args=t_args, 
        train_dataset=dataset, 
        data_collator=dataset.collate_fn
    )

    print("🔥 Starter trening!")
    trainer.train()

    # --- LAGRING OG OPPLASTING ---
    print("💾 Lagrer LoRA-adapter...")
    model.save_pretrained(args.output_model_path)
    
    if args.hf_repo_id and os.getenv("HF_TOKEN"):
        try:
            print(f"🚀 Laster opp til Hugging Face: {args.hf_repo_id}...")
            api = HfApi(token=os.getenv("HF_TOKEN"))
            api.upload_folder(
                folder_path=args.output_model_path,
                repo_id=args.hf_repo_id,
                repo_type="model"
            )
            print("✅ Opplasting fullført!")
        except Exception as e:
            print(f"⚠️ Opplasting feilet: {e}")
    else:
        print("ℹ️ Ingen HF_REPO_ID eller TOKEN funnet, hopper over opplasting.")

if __name__ == "__main__":
    train()
