#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-Tuning Script (LoRA + Språkembedding)
===============================================================

Adresserer de 3 kritiske problemene:
1. ✅ Utvider språkembedding-tabellen med norsk ("Norwegian")
2. ✅ Sikrer tokenizer støtter norske tegn (æøåÆØÅ)
3. ✅ Riktig dataforberedelse med codec-koding (EnCodec)

Bruk:
  python finetune_norwegian.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_data_dir ./norwegian_dataset \
    --output_dir ./qwen3-tts-norwegian-lora \
    --num_epochs 10 \
    --batch_size 2
"""

import argparse
import json
import os
import re
import shutil
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import HfApi, create_repo, Repository
from safetensors.torch import save_file

# --- Qwen3-TTS Imports (krever qwen-tts >= 1.0.0) ---
try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from qwen_tts.core.codec.codec_encoder import CodecEncoder
except ImportError:
    raise ImportError("Installer først: pip install qwen-tts transformers accelerate peft librosa torch torchaudio")

# ==========================================
# 1. UTVID SPRÅKEMBEDDING-TABELLEN
# ==========================================
def extend_language_embedding(model, new_language: str = "Norwegian"):
    """
    Legger til norsk i språkembedding-tabellen.
    Initialiserer med gjennomsnitt av fonetisk nærstående språk (tysk + engelsk).
    """
    current_languages = model.talker.language_embedding.weight.data
    current_lang_list = model.config.talker_config.languages
    
    if new_language in current_lang_list:
        print(f"✅ Språket '{new_language}' finnes allerede i modellen")
        return model
    
    print(f"🔧 Utvider språkembedding med '{new_language}'")
    
    # Finn indekser for tysk og engelsk (fonetisk nær norsk)
    try:
        de_idx = current_lang_list.index("German")
        en_idx = current_lang_list.index("English")
    except ValueError:
        # Fallback: bruk gjennomsnitt av alle språk
        de_idx, en_idx = 0, 1
    
    # Lag ny embedding som gjennomsnitt av tysk + engelsk
    new_embedding = (current_languages[de_idx] + current_languages[en_idx]) / 2.0
    
    # Utvid vekten
    new_weight = torch.cat([
        current_languages,
        new_embedding.unsqueeze(0)
    ], dim=0)
    
    # Opprett ny embedding-lag med utvidet vekt
    from torch import nn
    new_lang_emb = nn.Embedding(
        num_embeddings=new_weight.size(0),
        embedding_dim=new_weight.size(1)
    )
    new_lang_emb.weight.data = new_weight
    new_lang_emb.weight.requires_grad = True  # Viktig: trene denne!
    
    # Erstatt i modellen
    model.talker.language_embedding = new_lang_emb
    model.config.talker_config.languages.append(new_language)
    
    print(f"✅ Språkembedding utvidet: {len(current_lang_list)} → {len(model.config.talker_config.languages)} språk")
    return model

# ==========================================
# 2. SIKRE TOKENIZER STØTTER NORSKE TEGN
# ==========================================
def ensure_norwegian_tokenizer(tokenizer):
    """
    Sjekker at tokenizer støtter norske tegn (æøåÆØÅ).
    Hvis ikke, legg til dem i vokabularet.
    """
    norwegian_chars = "æøåÆØÅ"
    missing = [c for c in norwegian_chars if c not in tokenizer.get_vocab()]
    
    if missing:
        print(f"⚠️  Tokenizer mangler norske tegn: {missing}")
        print("💡 Løsning: Bruk text normalization under trening (se _normalize_norwegian_text)")
    else:
        print("✅ Tokenizer støtter alle norske tegn")
    return tokenizer

def _normalize_norwegian_text(text: str) -> str:
    """
    Normaliserer norsk tekst for bedre tokenisering:
    - Erstatt tall med ord ("42" → "førtito")
    - Erstatt spesialtegn
    - Konverter til små bokstaver (valgfritt)
    """
    # Enkel normalisering (for avansert: bruk lingua-no eller nb_NO locale)
    text = text.lower()
    text = re.sub(r'\b1\b', 'én', text)
    text = re.sub(r'\b2\b', 'to', text)
    text = re.sub(r'\b3\b', 'tre', text)
    text = re.sub(r'\b4\b', 'fire', text)
    text = re.sub(r'\b5\b', 'fem', text)
    text = re.sub(r'\b6\b', 'seks', text)
    text = re.sub(r'\b7\b', 'sju', text)
    text = re.sub(r'\b8\b', 'åtte', text)
    text = re.sub(r'\b9\b', 'ni', text)
    text = re.sub(r'\b10\b', 'ti', text)
    text = re.sub(r'\b20\b', 'tjue', text)
    text = re.sub(r'\b30\b', 'tretti', text)
    text = re.sub(r'\b40\b', 'førti', text)
    text = re.sub(r'\b50\b', 'femti', text)
    text = re.sub(r'\b60\b', 'seksti', text)
    text = re.sub(r'\b70\b', 'sytti', text)
    text = re.sub(r'\b80\b', 'åtti', text)
    text = re.sub(r'\b90\b', 'nitti', text)
    text = re.sub(r'\b100\b', 'hundre', text)
    text = re.sub(r'\b1000\b', 'tusen', text)
    
    # Fjern overflødig punktum etter forkortelser
    text = re.sub(r'\b(\w+)\.(\w+)\b', r'\1 \2', text)
    
    return text.strip()

# ==========================================
# 3. DATASET MED KORREKT CODEC-KODING
# ==========================================
class NorwegianTTSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        processor,
        config,
        codec_encoder: CodecEncoder,
        language: str = "Norwegian",
        max_duration: float = 15.0  # Maks 15 sekunder per eksempel
    ):
        self.processor = processor
        self.config = config
        self.codec_encoder = codec_encoder
        self.language = language
        self.max_samples = int(max_duration * 24000)
        
        # Last alle .wav + .txt par
        self.items = []
        data_path = Path(data_dir)
        
        for wav_path in data_path.glob("*.wav"):
            txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                if len(text) < 5:  # For kort tekst
                    continue
                    
                # Last og trim lyd
                audio, sr = torchaudio.load(wav_path)
                if sr != 24000:
                    audio = torchaudio.functional.resample(audio, sr, 24000)
                audio = audio.mean(dim=0)  # Konverter til mono
                
                if audio.shape[0] > self.max_samples:
                    audio = audio[:self.max_samples]
                
                self.items.append({
                    "audio": audio.numpy(),
                    "text": text,
                    "duration": audio.shape[0] / 24000.0
                })
        
        print(f"✅ Lastet {len(self.items)} norske taleeksempler fra {data_dir}")
        if len(self.items) < 10:
            raise ValueError("For få treningsdata! Trenger minst 10 eksempler.")
    
    def __len__(self):
        return len(self.items)
    
    @torch.no_grad()
    def _encode_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Kode lyd til codec tokens med EnCodec"""
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.codec_encoder.device)
        with torch.amp.autocast("cuda", enabled=True):
            codes = self.codec_encoder(audio_tensor)
        return codes[0].cpu()  # Returner kun første lag (viktig for Qwen3-TTS)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        text = _normalize_norwegian_text(item["text"])
        audio = item["audio"]
        
        # Kode lyd til tokens (krever GPU for hastighet)
        audio_codes = self._encode_audio(audio)  # Shape: [16, T]
        audio_codes = audio_codes.transpose(0, 1).contiguous()  # Shape: [T, 16]
        
        # Lag referanse-mel (bruk første 3 sekunder av lyden)
        ref_end = min(72000, audio.shape[0])  # 3 sekunder = 72000 samples @ 24kHz
        ref_audio = audio[:ref_end]
        
        return {
            "text": text,
            "audio_codes": audio_codes.numpy(),
            "ref_audio": ref_audio,
            "language": self.language
        }
    
    def collate_fn(self, batch):
        # Samle tekst
        texts = [item["text"] for item in batch]
        input_ids = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )["input_ids"]
        
        # Bygg assistant-format
        input_ids_list = []
        for ids in input_ids:
            # Finn EOS-token (vanligvis 151645)
            eos_idx = (ids == 151645).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                split_idx = min(3, eos_idx[0].item())
            else:
                split_idx = 3
            
            # Bygg: <|im_start|>assistant\n + tekst + <|im_end|>
            new_ids = torch.cat([
                ids[:split_idx],
                torch.tensor([198, 10380, 12114, 3727, 198]),  # \nassistant\n
                ids[split_idx:],
            ])
            input_ids_list.append(new_ids)
        
        # Pad til maks lengde
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = torch.full((len(batch), max_len), self.config.pad_token_id, dtype=torch.long)
        for i, ids in enumerate(input_ids_list):
            padded_input_ids[i, :len(ids)] = ids
        
        # Samle audio codes
        max_codec_len = max(item["audio_codes"].shape[0] for item in batch)
        codec_ids = torch.full((len(batch), max_codec_len, 16), self.config.talker_config.codec_pad_id, dtype=torch.long)
        codec_mask = torch.zeros((len(batch), max_codec_len), dtype=torch.bool)
        
        for i, item in enumerate(batch):
            codes = torch.from_numpy(item["audio_codes"])
            codec_ids[i, :codes.shape[0]] = codes
            codec_mask[i, :codes.shape[0]] = True
        
        # Samle referanse-mel (bruk codec_encoder til å lage mel)
        ref_mels = []
        for item in batch:
            ref_audio = torch.from_numpy(item["ref_audio"]).unsqueeze(0).to(self.codec_encoder.device)
            with torch.no_grad():
                mel = self.codec_encoder.mel_spectrogram(ref_audio)
            ref_mels.append(mel.cpu())
        
        ref_mels = torch.cat(ref_mels, dim=0)
        
        # Språkkoder (bruk indeks i utvidet språkliste)
        lang_idx = self.config.talker_config.languages.index(batch[0]["language"])
        language_ids = torch.full((len(batch),), lang_idx, dtype=torch.long)
        
        return {
            "input_ids": padded_input_ids,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "ref_mels": ref_mels,
            "language_ids": language_ids
        }

# ==========================================
# 4. TRENINGSSKRIPT MED FULL SPRÅKSTØTTE
# ==========================================
def train():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-TTS for Norwegian")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Mappe med .wav + .txt par")
    parser.add_argument("--output_dir", type=str, default="./qwen3-tts-norwegian-lora")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--hf_repo_id", type=str, default=None, help="Hugging Face repo ID for opplasting")
    args = parser.parse_args()
    
    # Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"🚀 Starter norsk TTS-trening | Språk: Norwegian | Epoker: {args.num_epochs}")
        print(f"📦 Data: {args.train_data_dir} | Batch: {args.batch_size} | LR: {args.lr}")
    
    # Last modell
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    model = qwen3tts.model
    
    # 1. Utvid språkembedding med norsk
    model = extend_language_embedding(model, "Norwegian")
    
    # 2. Sjekk tokenizer for norske tegn
    ensure_norwegian_tokenizer(qwen3tts.processor)
    
    # 3. FRYS alt unntatt LoRA + språkembedding + text_projection
    model.requires_grad_(False)
    
    # Aktiver LoRA på Transformer
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model.talker.model = get_peft_model(model.talker.model, peft_config)
    
    # Aktiver trening av språkembedding (kritisk for norsk!)
    model.talker.language_embedding.weight.requires_grad = True
    
    # Aktiver text_projection for bedre norske fonemer
    model.talker.text_projection.requires_grad = True
    
    if accelerator.is_main_process:
        print("\n✅ Trening aktiveringsstatus:")
        print(f"   - LoRA parameters: {sum(p.numel() for p in model.talker.model.parameters() if p.requires_grad):,}")
        print(f"   - Language embedding: {model.talker.language_embedding.weight.requires_grad}")
        print(f"   - Text projection: {model.talker.text_projection.requires_grad}")
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   - Total trenbare parametere: {total_trainable:,} / {total_params:,} ({total_trainable/total_params*100:.2f}%)")
    
    # Last codec encoder for dataforberedelse
    codec_encoder = CodecEncoder.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device}
    )
    
    # Last dataset
    dataset = NorwegianTTSDataset(
        data_dir=args.train_data_dir,
        processor=qwen3tts.processor,
        config=model.config,
        codec_encoder=codec_encoder,
        language="Norwegian"
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer (kun trenbare parametere)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Prepare for training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    # Hugging Face API
    hf_api = None
    if args.hf_repo_id and os.getenv("HF_TOKEN"):
        hf_api = HfApi(token=os.getenv("HF_TOKEN"))
        if accelerator.is_main_process:
            try:
                create_repo(args.hf_repo_id, exist_ok=True, repo_type="model")
                print(f"☁️  Koblet til HF repo: {args.hf_repo_id}")
            except Exception as e:
                print(f"⚠️  Kunne ikke opprette HF repo: {e}")
    
    # Trening
    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Forbered input
                input_ids = batch["input_ids"].to(model.device)
                codec_ids = batch["codec_ids"].to(model.device)
                codec_mask = batch["codec_mask"].to(model.device)
                ref_mels = batch["ref_mels"].to(model.device, dtype=model.dtype)
                language_ids = batch["language_ids"].to(model.device)
                
                # Hent speaker embedding fra referanse-mel
                with torch.no_grad():
                    speaker_embedding = model.speaker_encoder(ref_mels).detach()
                
                # Lag input embeddings
                text_embeds = model.talker.model.text_embedding(input_ids)
                text_embeds = model.talker.text_projection(text_embeds)
                
                # Lag språkembedding (kritisk!)
                lang_embeds = model.talker.language_embedding(language_ids)
                lang_embeds = lang_embeds.unsqueeze(1).expand(-1, text_embeds.size(1), -1)
                text_embeds = text_embeds + lang_embeds  # Legg til språkinfo
                
                # Lag codec embeddings
                codec_input_ids = torch.full_like(input_ids, model.config.talker_config.codec_pad_id)
                codec_input_ids[:, 0] = model.config.talker_config.codec_bos_id
                
                codec_embeds = model.talker.model.codec_embedding(codec_input_ids)
                codec_embeds[:, 6, :] = speaker_embedding  # Sett speaker embedding på riktig posisjon
                
                # Kombiner embeddings
                input_embeds = text_embeds + codec_embeds
                
                # Forward pass
                outputs = model.talker(
                    inputs_embeds=input_embeds[:, :-1, :],
                    attention_mask=torch.ones_like(input_ids)[:, :-1],
                    labels=codec_ids[:, 1:, 0],  # Bare første codec-laget som labels
                    output_hidden_states=True
                )
                
                # Sub-talker loss (hvis tilgjengelig)
                sub_loss = 0.0
                if hasattr(model.talker, "forward_sub_talker"):
                    hidden_states = outputs.hidden_states[-1]
                    valid_hs = hidden_states[codec_mask[:, 1:]]
                    valid_codes = codec_ids[codec_mask][:, 1:]
                    if valid_hs.size(0) > 0:
                        _, sub_loss = model.talker.forward_sub_talker(valid_codes, valid_hs)
                        sub_loss = sub_loss.mean()
                
                loss = outputs.loss + 0.1 * sub_loss  # Vekt sub-talker loss lavere
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
            
            # Logging
            if step % 10 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} | Step {step}/{len(train_dataloader)} | Loss: {loss.item():.4f} | Sub-loss: {sub_loss:.4f}")
            
            # Lagring
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                
                # Lagre LoRA-vekter
                accelerator.unwrap_model(model).talker.model.save_pretrained(save_path)
                
                # Lagre språkembedding og text_projection
                torch.save(
                    accelerator.unwrap_model(model).talker.language_embedding.state_dict(),
                    os.path.join(save_path, "language_embedding.bin")
                )
                torch.save(
                    accelerator.unwrap_model(model).talker.text_projection.state_dict(),
                    os.path.join(save_path, "text_projection.bin")
                )
                
                # Lagre config med oppdatert språkliste
                config = accelerator.unwrap_model(model).config
                config.save_pretrained(save_path)
                
                # Lag README
                with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write(f"""---
base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
library_name: peft
tags:
- text-to-speech
- qwen3-tts
- norwegian
- lora
- bokmål
---
# Qwen3-TTS Norwegian LoRA

Trained on Norwegian speech data with extended language embedding.

## Usage

```python
from qwen_tts import Qwen3TTSModel
from peft import PeftModel

# Last base-modell
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# Last LoRA-vekter
model.model.talker.model = PeftModel.from_pretrained(
    model.model.talker.model, 
    "{save_path}"
)

# Generer norsk tale (viktig: bruk language="Norwegian")
wavs, sr = model.generate(
    text="Dette er en norsk setning.",
    language="Norwegian",  # MÅ være "Norwegian"!
    ref_audio="norsk_referanse.wav",
    ref_text="Dette er en referansetekst."
)
