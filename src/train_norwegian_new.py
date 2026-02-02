#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-Tuning (LoRA + Språkembedding)
"""

import argparse
import json
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from qwen3_tts import Qwen3TTSModel


def extend_language_embedding(model):
    current_weights = model.model.talker.language_embedding.weight.data.clone()
    current_languages = model.model.config.talker_config.languages.copy()
    
    if "Norwegian" in current_languages:
        print(f"✅ Språket 'Norwegian' finnes allerede (indeks {current_languages.index('Norwegian')})")
        return model
    
    print(f"🔧 Utvider språkembedding-tabell med 'Norwegian'...")
    
    try:
        de_idx = current_languages.index("German")
        en_idx = current_languages.index("English")
    except ValueError:
        de_idx, en_idx = 0, 1
    
    new_embedding = (current_weights[de_idx] + current_weights[en_idx]) / 2.0
    new_weight = torch.cat([current_weights, new_embedding.unsqueeze(0)], dim=0)
    
    model.model.talker.language_embedding.weight = torch.nn.Parameter(new_weight, requires_grad=True)
    model.model.config.talker_config.languages.append("Norwegian")
    
    print(f"✅ Språkembedding utvidet: {len(current_languages)} → {len(model.model.config.talker_config.languages)} språk")
    return model


class NorwegianTTSDataset(Dataset):
    def __init__(self, jsonl_path, processor, lang_size):
        self.processor = processor
        self.lang_size = lang_size
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.items = [json.loads(line) for line in f]
        print(f"✅ Lastet {len(self.items)} eksempler")
    
    def __len__(self):
        return len(self.items)
    
    def _load_audio(self, path):
        audio, sr = librosa.load(path, sr=24000, mono=True)
        if len(audio) > 24000 * 15:
            audio = audio[:24000 * 15]
        return audio
    
    def __getitem__(self, idx):
        item = self.items[idx]
        ref_audio = self._load_audio(item["ref_audio_path"])
        return {
            "text": item["text"],
            "audio_codes": np.array(item["audio_codes"], dtype=np.int64),
            "ref_audio": ref_audio,
            "language": "Norwegian"
        }
    
    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        input_ids = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"]
        
        assistant_ids = []
        for ids in input_ids:
            eos_idx = (ids == 151645).nonzero(as_tuple=True)[0]
            split_idx = eos_idx[0].item() if len(eos_idx) > 0 else 3
            new_ids = torch.cat([
                ids[:split_idx],
                torch.tensor([198, 10380, 12114, 3727, 198]),
                ids[split_idx:],
            ])
            assistant_ids.append(new_ids)
        
        max_len = max(len(ids) for ids in assistant_ids)
        padded_input_ids = torch.full((len(batch), max_len), 151643, dtype=torch.long)
        for i, ids in enumerate(assistant_ids):
            padded_input_ids[i, :len(ids)] = ids
        
        max_t = max(item["audio_codes"].shape[0] for item in batch)
        codec_ids = torch.full((len(batch), max_t, 16), 1024, dtype=torch.long)
        codec_mask = torch.zeros((len(batch), max_t), dtype=torch.bool)
        
        for i, item in enumerate(batch):
            codes = torch.from_numpy(item["audio_codes"])
            codec_ids[i, :codes.shape[0]] = codes
            codec_mask[i, :codes.shape[0]] = True
        
        ref_mels = []
        for item in batch:
            mel = librosa.feature.melspectrogram(
                y=item["ref_audio"], sr=24000, n_fft=1024, hop_length=256, n_mels=128, fmin=0, fmax=12000
            )
            mel = torch.from_numpy(mel).unsqueeze(0).transpose(1, 2)
            ref_mels.append(mel)
        ref_mels = torch.cat(ref_mels, dim=0)
        
        lang_idx = self.lang_size - 1
        language_ids = torch.full((len(batch),), lang_idx, dtype=torch.long)
        
        return {
            "input_ids": padded_input_ids,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "ref_mels": ref_mels,
            "language_ids": language_ids
        }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, default="./qwen3-tts-norwegian-lora")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        project_dir=args.output_model_path
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_model_path, exist_ok=True)
        print("🚀 Starter norsk TTS-trening")
    
    model = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    
    model = extend_language_embedding(model)
    
    model.model.requires_grad_(False)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model.model.talker.model = get_peft_model(model.model.talker.model, peft_config)
    
    model.model.talker.language_embedding.weight.requires_grad = True
    model.model.talker.text_projection.requires_grad = True
    
    if accelerator.is_main_process:
        total_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"✅ Trenbare parametere: {total_trainable:,} / {total_params:,} ({total_trainable/total_params*100:.2f}%)")
    
    lang_size = len(model.model.config.talker_config.languages)
    dataset = NorwegianTTSDataset(args.train_jsonl, model.processor, lang_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=2, pin_memory=True)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=args.lr, weight_decay=0.01)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(model.device)
                codec_ids = batch["codec_ids"].to(model.device)
                codec_mask = batch["codec_mask"].to(model.device)
                ref_mels = batch["ref_mels"].to(model.device, dtype=model.dtype)
                language_ids = batch["language_ids"].to(model.device)
                
                with torch.no_grad():
                    speaker_embedding = model.model.speaker_encoder(ref_mels).detach()
                
                text_embeds = model.model.talker.model.text_embedding(input_ids)
                text_embeds = model.model.talker.text_projection(text_embeds)
                
                lang_embeds = model.model.talker.language_embedding(language_ids)
                lang_embeds = lang_embeds.unsqueeze(1).expand(-1, text_embeds.size(1), -1)
                text_embeds = text_embeds + lang_embeds
                
                codec_input_ids = torch.full_like(input_ids, 1024)
                codec_input_ids[:, 0] = 1025
                codec_embeds = model.model.talker.model.codec_embedding(codec_input_ids)
                codec_embeds[:, 6, :] = speaker_embedding
                
                input_embeds = text_embeds + codec_embeds
                
                outputs = model.model.talker(
                    inputs_embeds=input_embeds[:, :-1, :],
                    attention_mask=torch.ones_like(input_ids)[:, :-1],
                    labels=codec_ids[:, 1:, 0],
                    output_hidden_states=True
                )
                
                sub_loss = 0.0
                if hasattr(model.model.talker, "forward_sub_talker"):
                    hidden_states = outputs.hidden_states[-1]
                    valid_hs = hidden_states[codec_mask[:, 1:]]
                    valid_codes = codec_ids[codec_mask][:, 1:]
                    if valid_hs.size(0) > 0:
                        _, sub_loss = model.model.talker.forward_sub_talker(valid_codes, valid_hs)
                        sub_loss = sub_loss.mean()
                
                loss = outputs.loss + 0.1 * sub_loss
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % args.save_every == 0:
                save_path = os.path.join(args.output_model_path, f"epoch-{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                
                accelerator.unwrap_model(model).model.talker.model.save_pretrained(save_path)
                torch.save(accelerator.unwrap_model(model).model.talker.language_embedding.state_dict(), os.path.join(save_path, "language_embedding.bin"))
                torch.save(accelerator.unwrap_model(model).model.talker.text_projection.state_dict(), os.path.join(save_path, "text_projection.bin"))
                accelerator.unwrap_model(model).model.config.save_pretrained(save_path)
                
                # ✅ 100% SYNTAKTISK KORREKT README (ingen triple quotes!)
                with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write("---\n")
                    f.write("base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base\n")
                    f.write("library_name: peft\n")
                    f.write("tags:\n")
                    f.write("- text-to-speech\n")
                    f.write("- qwen3-tts\n")
                    f.write("- norwegian\n")
                    f.write("- lora\n")
                    f.write("---\n")
                    f.write(f"# Qwen3-TTS Norwegian LoRA (Epoch {epoch+1})\n")
                    f.write("\n")
                    f.write("Trained with extended language embedding for Norwegian.\n")
                    f.write("\n")
                    f.write("## Usage\n")
                    f.write("```python\n")
                    f.write("from qwen3_tts import Qwen3TTSModel\n")
                    f.write("from peft import PeftModel\n")
                    f.write("\n")
                    f.write('model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")\n')
                    f.write('model.model.talker.model = PeftModel.from_pretrained(\n')
                    f.write('    model.model.talker.model,\n')
                    f.write(f'    "{save_path}"\n')
                    f.write(')\n')
                    f.write('\n')
                    f.write('# VIKTIG: Bruk language="Norwegian" (eksakt dette navnet!)\n')
                    f.write('wavs, sr = model.generate(\n')
                    f.write('    text="Dette er en norsk setning med æøå.",\n')
                    f.write('    language="Norwegian",\n')
                    f.write('    ref_audio="ref.wav"\n')
                    f.write(')\n')
                    f.write("```\n")
                
                print(f"💾 Lagret checkpoint: {save_path}")
    
    if accelerator.is_main_process:
        print("✅ Trening fullført!")


if __name__ == "__main__":
    train()
