#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-Tuning (LoRA + Språkembedding)
========================================================

Løser de 4 kritiske problemene:
1. ✅ Utvider språkembedding-tabellen med "Norwegian"
2. ✅ Gjør språkembedding trenbar
3. ✅ Bruker riktig språkkode ("Norwegian")
4. ✅ Bruker riktig codec-tokenizer

Bruk:
  accelerate launch src/train_norwegian_new.py \
    --train_jsonl ./data/train_with_codes.jsonl \
    --init_model_path ./base_model \
    --output_model_path ./output/run_long \
    --batch_size 4 \
    --num_epochs 100
"""

import argparse
import json
import os
import sys
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

# ==========================================
# 0. IMPORT FIX (Peker til Qwen-koden)
# ==========================================
sys.path.append("/workspace/Qwen3-TTS")

try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
except ImportError:
    print("⚠️  Kunne ikke importere Qwen3TTSModel. Sjekk at mappen '/workspace/Qwen3-TTS' eksisterer.")
    sys.exit(1)


# ==========================================
# 1. UTVID SPRÅKEMBEDDING MED NORSK
# ==========================================
def extend_language_embedding(model):
    """
    Legger til "Norwegian" i språkembedding-tabellen.
    """
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
        print("⚠️  Fant ikke Tysk/Engelsk, bruker fallback-initialisering.")
        de_idx, en_idx = 0, 1
    
    # Lag ny embedding som gjennomsnitt
    new_embedding = (current_weights[de_idx] + current_weights[en_idx]) / 2.0
    
    # Utvid vekten
    new_weight = torch.cat([current_weights, new_embedding.unsqueeze(0)], dim=0)
    
    # Erstatt embedding-laget med det nye
    model.model.talker.language_embedding.weight = torch.nn.Parameter(
        new_weight, 
        requires_grad=True  # KRITISK: må være True for å trenes!
    )
    
    # Oppdater språklista i config
    model.model.config.talker_config.languages.append("Norwegian")
    
    print(f"✅ Språkembedding utvidet: {len(current_languages)} -> {len(model.model.config.talker_config.languages)} språk")
    return model


# ==========================================
# 2. DATASET
# ==========================================
class NorwegianTTSDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.processor = processor
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.items = [json.loads(line) for line in f]
        
        print(f"✅ Lastet {len(self.items)} eksempler fra {jsonl_path}")
        
        if len(self.items) > 0:
            item = self.items[0]
            assert "text" in item, "Mangler 'text' i JSONL"
            assert "audio_codes" in item, "Mangler 'audio_codes' i JSONL"
            assert "ref_audio_path" in item, "Mangler 'ref_audio_path' i JSONL"
    
    def __len__(self):
        return len(self.items)
    
    def _load_audio(self, path):
        try:
            audio, sr = librosa.load(path, sr=24000, mono=True)
            if len(audio) > 24000 * 15: 
                audio = audio[:24000 * 15]
            return audio
        except Exception as e:
            print(f"⚠️ Feil ved lasting av lyd {path}: {e}")
            return np.zeros(24000)
    
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
        input_ids = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )["input_ids"]
        
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
        
        return {
            "input_ids": padded_input_ids,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "ref_mels": ref_mels,
            "language": "Norwegian"
        }


# ==========================================
# 3. TRENING
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, default="./qwen3-tts-norwegian-lora")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_model_path, exist_ok=True)
        print("🚀 Starter norsk TTS-trening med utvidet språkembedding")
        print(f"   LR: {args.lr}, Epochs: {args.num_epochs}, Batch: {args.batch_size}")
    
    model = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    
    # 1. Utvid språkembedding
    model = extend_language_embedding(model)
    
    # 2. Frys modell
    model.model.requires_grad_(False)
    
    # 3. Aktiver LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model.model.talker.model = get_peft_model(model.model.talker.model, peft_config)
    
    # 4. Aktiver trening på spesifikke lag
    model.model.talker.language_embedding.weight.requires_grad = True
    model.model.talker.text_projection.requires_grad = True
    
    if accelerator.is_main_process:
        print("\n✅ Trenbare parametere:")
        print(f"   - LoRA: {sum(p.numel() for p in model.model.talker.model.parameters() if p.requires_grad):,}")
        print(f"   - Language embedding: {model.model.talker.language_embedding.weight.requires_grad}")
    
    dataset = NorwegianTTSDataset(jsonl_path=args.train_jsonl, processor=model.processor)
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
    
    # Finn Language ID
    try:
        unwrapped = accelerator.unwrap_model(model)
        norwegian_lang_id = unwrapped.model.config.talker_config.languages.index("Norwegian")
    except ValueError:
        print("❌ CRITICAL: Fant ikke 'Norwegian' i språklisten etter utvidelse!")
        return
    
    if accelerator.is_main_process:
        print(f"🎯 Trener med Language ID: {norwegian_lang_id} (Norwegian)")

    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        steps_in_epoch = 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(model.device)
                codec_ids = batch["codec_ids"].to(model.device)
                codec_mask = batch["codec_mask"].to(model.device)
                ref_mels = batch["ref_mels"].to(model.device, dtype=model.dtype)
                
                # Sett Language ID
                curr_bs = input_ids.shape[0]
                language_ids = torch.full((curr_bs,), norwegian_lang_id, dtype=torch.long, device=model.device)
                
                with torch.no_grad():
                    speaker_embedding = model.model.speaker_encoder(ref_mels).detach()
                
                # Forward Pass Logic
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
                
                # Sub-loss
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
                steps_in_epoch += 1
        
        # Logging & Lagring
        avg_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % args.save_every == 0:
                save_path = os.path.join(args.output_model_path, f"epoch-{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.model.talker.model.save_pretrained(save_path)
                
                torch.save(unwrapped_model.model.talker.language_embedding.state_dict(), os.path.join(save_path, "language_embedding.bin"))
                torch.save(unwrapped_model.model.talker.text_projection.state_dict(), os.path.join(save_path, "text_projection.bin"))
                unwrapped_model.model.config.save_pretrained(save_path)
                
                # SIKKER README-SKRIVING (INGEN F-STRING HELVETE)
                readme_lines = [
                    "---",
                    "base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "library_name: peft",
                    f"tags: [text-to-speech, qwen3-tts, norwegian, lora, epoch-{epoch+1}]",
                    "---",
                    f"# Qwen3-TTS Norwegian LoRA (Epoch {epoch+1})",
                    "",
                    "Trained with extended language embedding for Norwegian.",
                    "",
                    "## Usage",
                    "```python",
                    "from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel",
                    "from peft import PeftModel",
                    "",
                    'model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")',
                    "model.model.talker.model = PeftModel.from_pretrained(",
                    "    model.model.talker.model,",
                    f'    "{save_path}"',
                    ")",
                    "",
                    'wavs, sr = model.generate(',
                    '    text="Dette er en norsk setning.",',
                    '    language="Norwegian",',
                    '    ref_audio="ref.wav"',
                    ")",
                    "```"
                ]
                with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write("\n".join(readme_lines))
                
                print(f"💾 Checkpoint lagret: {save_path}")

if __name__ == "__main__":
    train()
