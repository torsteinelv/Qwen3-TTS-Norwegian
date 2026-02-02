#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-Tuning (LoRA + Språkembedding)
========================================================

Løser de 4 kritiske problemene:
1. ✅ Utvider språkembedding-tabellen med "Norwegian"
2. ✅ Gjør språkembedding trenbar
3. ✅ Bruker riktig språkkode ("Norwegian")
4. ✅ Genererer audio_codes med offisiell tokenizer

Bruk:
  python train_norwegian.py \
    --train_jsonl ./data/norwegian_data.jsonl \
    --output_dir ./qwen3-tts-norwegian-lora \
    --num_epochs 50 \
    --batch_size 2
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
from safetensors.torch import save_file

# Installer først: git clone https://github.com/QwenLM/Qwen3-TTS && cd Qwen3-TTS && pip install -e "."
from qwen3_tts import Qwen3TTSModel, Qwen3TTSTokenizer


# ==========================================
# 1. UTVID SPRÅKEMBEDDING MED NORSK
# ==========================================
def extend_language_embedding(model):
    """
    Legger til "Norwegian" i språkembedding-tabellen.
    Initialiserer med gjennomsnitt av tysk + engelsk (fonetisk nærme språk).
    """
    current_weights = model.model.talker.language_embedding.weight.data.clone()
    current_languages = model.model.config.talker_config.languages.copy()
    
    if "Norwegian" in current_languages:
        print(f"✅ Språket 'Norwegian' finnes allerede (indeks {current_languages.index('Norwegian')})")
        return model
    
    print(f"🔧 Utvider språkembedding-tabell med 'Norwegian'...")
    
    # Finn tysk og engelsk (fonetisk nær norsk)
    try:
        de_idx = current_languages.index("German")
        en_idx = current_languages.index("English")
    except ValueError:
        # Fallback: bruk første to språk
        de_idx, en_idx = 0, 1
    
    # Lag ny embedding som gjennomsnitt
    new_embedding = (current_weights[de_idx] + current_weights[en_idx]) / 2.0
    
    # Utvid vekten
    new_weight = torch.cat([current_weights, new_embedding.unsqueeze(0)], dim=0)
    
    # Erstatt embedding-laget
    model.model.talker.language_embedding.weight = torch.nn.Parameter(
        new_weight, 
        requires_grad=True  # KRITISK: må trenes!
    )
    
    # Oppdater språklista i config
    model.model.config.talker_config.languages.append("Norwegian")
    
    print(f"✅ Språkembedding utvidet: {len(current_languages)} → {len(model.model.config.talker_config.languages)} språk")
    print(f"   Språkliste: {model.model.config.talker_config.languages}")
    return model


# ==========================================
# 2. DATASET MED RIKTIG CODEC-KODING
# ==========================================
class NorwegianTTSDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, processor):
        self.tokenizer = tokenizer
        self.processor = processor
        
        # Last data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.items = [json.loads(line) for line in f]
        
        print(f"✅ Lastet {len(self.items)} eksempler fra {jsonl_path}")
        
        # Valider første eksempel
        if len(self.items) > 0:
            item = self.items[0]
            assert "text" in item, "Mangler 'text' i JSONL"
            assert "audio_path" in item, "Mangler 'audio_path' i JSONL"
            assert "ref_audio_path" in item, "Mangler 'ref_audio_path' i JSONL"
    
    def __len__(self):
        return len(self.items)
    
    def _load_audio(self, path):
        audio, sr = librosa.load(path, sr=24000, mono=True)
        if len(audio) > 24000 * 15:  # Maks 15 sek
            audio = audio[:24000 * 15]
        return audio
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Last og kode lyd til tokens [T, 16]
        audio = self._load_audio(item["audio_path"])
        audio_codes = self.tokenizer.encode(audio)  # Dette er KRITISK – bruker riktig codec
        
        # Last referanselyd
        ref_audio = self._load_audio(item["ref_audio_path"])
        
        return {
            "text": item["text"],
            "audio_codes": audio_codes.numpy(),  # [T, 16]
            "ref_audio": ref_audio,
            "language": "Norwegian"  # EKSACT dette navnet!
        }
    
    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        input_ids = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )["input_ids"]
        
        # Bygg assistant-format (samme som Qwen3-TTS sin internformat)
        assistant_ids = []
        for ids in input_ids:
            eos_idx = (ids == 151645).nonzero(as_tuple=True)[0]
            split_idx = eos_idx[0].item() if len(eos_idx) > 0 else 3
            
            new_ids = torch.cat([
                ids[:split_idx],
                torch.tensor([198, 10380, 12114, 3727, 198]),  # \nassistant\n
                ids[split_idx:],
            ])
            assistant_ids.append(new_ids)
        
        # Pad
        max_len = max(len(ids) for ids in assistant_ids)
        padded_input_ids = torch.full((len(batch), max_len), 151643, dtype=torch.long)  # pad_token_id
        for i, ids in enumerate(assistant_ids):
            padded_input_ids[i, :len(ids)] = ids
        
        # Samle audio codes
        max_t = max(item["audio_codes"].shape[0] for item in batch)
        codec_ids = torch.full((len(batch), max_t, 16), 1024, dtype=torch.long)  # codec_pad_id=1024
        codec_mask = torch.zeros((len(batch), max_t), dtype=torch.bool)
        
        for i, item in enumerate(batch):
            codes = torch.from_numpy(item["audio_codes"])
            codec_ids[i, :codes.shape[0]] = codes
            codec_mask[i, :codes.shape[0]] = True
        
        # Lag mel-spectrogram for referanselyd
        ref_mels = []
        for item in batch:
            audio_tensor = torch.from_numpy(item["ref_audio"]).unsqueeze(0)
            mel = librosa.feature.melspectrogram(
                y=item["ref_audio"],
                sr=24000,
                n_fft=1024,
                hop_length=256,
                n_mels=128,
                fmin=0,
                fmax=12000
            )
            mel = torch.from_numpy(mel).unsqueeze(0).transpose(1, 2)
            ref_mels.append(mel)
        ref_mels = torch.cat(ref_mels, dim=0)
        
        # Språk-ID (bruk utvidet lista!)
        lang_idx = batch[0]["language"]  # "Norwegian"
        language_ids = torch.tensor([
            model.model.config.talker_config.languages.index(lang_idx)
            for _ in batch
        ], dtype=torch.long)
        
        return {
            "input_ids": padded_input_ids,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "ref_mels": ref_mels,
            "language_ids": language_ids
        }


# ==========================================
# 3. TRENING
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./qwen3-tts-norwegian-lora")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)  # Lav LR for stabilitet
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()
    
    # Setup
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print("🚀 Starter norsk TTS-trening med utvidet språkembedding")
    
    # Last modell
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    
    # ✅ KRITISK TRINN 1: Utvid språkembedding med norsk
    model = extend_language_embedding(model)
    
    # Last tokenizer for codec-koding
    tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")
    
    # ✅ KRITISK TRINN 2: FRYS alt unntatt LoRA + språkembedding + text_projection
    model.model.requires_grad_(False)
    
    # Aktiver LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model.model.talker.model = get_peft_model(model.model.talker.model, peft_config)
    
    # ✅ KRITISK TRINN 3: Gjør språkembedding trenbar
    model.model.talker.language_embedding.weight.requires_grad = True
    
    # ✅ KRITISK TRINN 4: Gjør text_projection trenbar (for norske tegn)
    model.model.talker.text_projection.requires_grad = True
    
    if accelerator.is_main_process:
        print("\n✅ Trenbare parametere:")
        print(f"   - LoRA: {sum(p.numel() for p in model.model.talker.model.parameters() if p.requires_grad):,}")
        print(f"   - Language embedding: {model.model.talker.language_embedding.weight.requires_grad}")
        print(f"   - Text projection: {model.model.talker.text_projection.requires_grad}")
        total_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"   - Total trenbare: {total_trainable:,} / {total_params:,} ({total_trainable/total_params*100:.2f}%)")
    
    # Datasett
    dataset = NorwegianTTSDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tokenizer,
        processor=model.processor
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Trening
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                # Forbered input
                input_ids = batch["input_ids"].to(model.device)
                codec_ids = batch["codec_ids"].to(model.device)
                codec_mask = batch["codec_mask"].to(model.device)
                ref_mels = batch["ref_mels"].to(model.device, dtype=model.dtype)
                language_ids = batch["language_ids"].to(model.device)
                
                # Hent speaker embedding
                with torch.no_grad():
                    speaker_embedding = model.model.speaker_encoder(ref_mels).detach()
                
                # Lag embeddings
                text_embeds = model.model.talker.model.text_embedding(input_ids)
                text_embeds = model.model.talker.text_projection(text_embeds)
                
                # ✅ KRITISK TRINN 5: Legg til språkembedding
                lang_embeds = model.model.talker.language_embedding(language_ids)
                lang_embeds = lang_embeds.unsqueeze(1).expand(-1, text_embeds.size(1), -1)
                text_embeds = text_embeds + lang_embeds
                
                # Lag codec embeddings
                codec_input_ids = torch.full_like(input_ids, 1024)  # codec_pad_id
                codec_input_ids[:, 0] = 1025  # codec_bos_id
                
                codec_embeds = model.model.talker.model.codec_embedding(codec_input_ids)
                codec_embeds[:, 6, :] = speaker_embedding
                
                # Kombiner
                input_embeds = text_embeds + codec_embeds
                
                # Forward pass
                outputs = model.model.talker(
                    inputs_embeds=input_embeds[:, :-1, :],
                    attention_mask=torch.ones_like(input_ids)[:, :-1],
                    labels=codec_ids[:, 1:, 0],  # Bare første codec-lag
                    output_hidden_states=True
                )
                
                # Sub-talker loss
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
        
        # Logging
        avg_loss = epoch_loss / len(dataloader)
        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f}")
            
            # Lagre hver 5. epoke
            if (epoch + 1) % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"epoch-{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                
                # Lagre LoRA
                accelerator.unwrap_model(model).model.talker.model.save_pretrained(save_path)
                
                # Lagre språkembedding + text_projection
                torch.save(
                    accelerator.unwrap_model(model).model.talker.language_embedding.state_dict(),
                    os.path.join(save_path, "language_embedding.bin")
                )
                torch.save(
                    accelerator.unwrap_model(model).model.talker.text_projection.state_dict(),
                    os.path.join(save_path, "text_projection.bin")
                )
                
                # Lagre config
                accelerator.unwrap_model(model).model.config.save_pretrained(save_path)
                
                # README
                with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write(f"""---
base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
library_name: peft
tags:
- text-to-speech
- qwen3-tts
- norwegian
- lora
---
# Qwen3-TTS Norwegian LoRA (Epoch {epoch+1})

Trained with extended language embedding for Norwegian.

## Usage
```python
from qwen3_tts import Qwen3TTSModel
from peft import PeftModel

model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
model.model.talker.model = PeftModel.from_pretrained(
    model.model.talker.model,
    "{save_path}"
)

# VIKTIG: Bruk language="Norwegian" (eksakt dette navnet!)
wavs, sr = model.generate(
    text="Dette er en norsk setning med æøå.",
    language="Norwegian",
    ref_audio="ref.wav"
)
