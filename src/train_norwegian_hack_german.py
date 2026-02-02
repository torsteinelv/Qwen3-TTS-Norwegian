#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-Tuning (German Hijack + Robust Data Collator)
=======================================================================

Fixes:
1. ValueError (Batch Size Mismatch): Uses the correct complex collate_fn 
   to align text and audio sequences perfectly.
2. KeyError: Robust checking for 'ref_audio' vs 'ref_audio_path'.
3. German Hijack: Injects 'German' language ID to force Germanic prosody.

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
from typing import Any, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig

# ==========================================
# 0. IMPORT FIX
# ==========================================
sys.path.append("/workspace/Qwen3-TTS")
try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
except ImportError:
    print("⚠️  Kunne ikke importere Qwen3TTSModel. Sjekk paths.")
    sys.exit(1)

# ==========================================
# 1. DATASET & COMPLEX COLLATOR (FIXES VALUE ERROR)
# ==========================================
AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]
MaybeList = Union[Any, List[Any]]

class NorwegianTTSDataset(Dataset):
    def __init__(self, jsonl_path, processor, config):
        self.processor = processor
        self.config = config
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.items = [json.loads(line) for line in f]
        print(f"✅ Lastet {len(self.items)} eksempler.")

    def __len__(self):
        return len(self.items)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(x, sr=None, mono=True)
            if len(audio) > 24000 * 15: # Cut max 15s to avoid OOM
                audio = audio[:24000 * 15]
            return audio.astype(np.float32), int(sr)
        except Exception as e:
            print(f"⚠️ Audio load error: {e}")
            return np.zeros(24000).astype(np.float32), 24000

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list): items = audios
        else: items = [audios]
        out = []
        for a in items:
            if isinstance(a, str): out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple): out.append((a[0].astype(np.float32), int(a[1])))
            else: raise TypeError(f"Unsupported audio type: {type(a)}")
        return out
    
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        # Må bruke 24kHz for Qwen
        if sr != 24000:
            # Enkel resampling hvis nødvendig (men vi antar 24k fra load)
            pass
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000
        ).transpose(1, 2)
        return mels

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Robust path finding
        ref_path = item.get("ref_audio") or item.get("ref_audio_path")
        if not ref_path:
            return self.__getitem__((idx + 1) % len(self.items))

        text = self._build_assistant_text(item["text"])
        text_ids = self._tokenize_texts(text)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        normalized = self._normalize_audio_inputs([ref_path])
        wav, sr = normalized[0]
        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:,:-5],    # Fjerner noen slutt-tokens for korrekt fletting
            "audio_codes": audio_codes,
            "ref_mel": ref_mel
        }
        
    def collate_fn(self, batch):
        # Denne komplekse logikken er NØDVENDIG for å unngå ValueError
        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length

        input_ids = torch.zeros((b,t,2), dtype=torch.long)
        codec_ids = torch.zeros((b,t,16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b,t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b,t), dtype=torch.bool)
        codec_mask = torch.zeros((b,t), dtype=torch.bool)
        attention_mask = torch.zeros((b,t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data['text_ids']
            audio_codec_0 = data['audio_codes'][:,0]
            audio_codecs = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # Text channel setup
            input_ids[i, :3, 0] = text_ids[0,:3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0,3:]
            input_ids[i, 8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8+text_ids_len+codec_ids_len] = True

            # Codec channel setup
            input_ids[i, 3:8, 1] = torch.tensor([
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,      # for speaker embedding
                self.config.talker_config.codec_pad_id        
            ])
            input_ids[i, 8:8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8+text_ids_len-1+codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, :] = audio_codecs

            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False       # for speaker embedding

            codec_mask[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True
        
        ref_mels = [data['ref_mel'] for data in batch]
        ref_mels = torch.cat(ref_mels, dim=0)

        return {
            'input_ids': input_ids,
            'ref_mels': ref_mels,
            'attention_mask': attention_mask,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels': codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask': codec_mask
        }

# ==========================================
# 2. TRENING (Med German Hijack)
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
        print("🚀 Starter norsk TTS-trening (German Hijack + Robust Collator)")
    
    # Last modell
    qwen_wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    model = qwen_wrapper.model # Henter den ekte modellen
    
    # --- FINN GERMAN ID ---
    try:
        lang_list = model.config.talker_config.languages
        german_id = lang_list.index("German")
        if accelerator.is_main_process:
            print(f"✅ Fant 'German' på indeks {german_id}. Hijacker denne!")
    except:
        print("⚠️ Fant ikke 'German', bruker ID 2 som fallback.")
        german_id = 2

    # Frys modell
    model.requires_grad_(False)
    
    # Aktiver LoRA
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION"
    )
    model.talker.model = get_peft_model(model.talker.model, peft_config)
    
    # Aktiver text_projection
    if hasattr(model.talker, "text_projection"):
        model.talker.text_projection.requires_grad = True
        if accelerator.is_main_process: print("✅ Text Projection trenbar.")

    if accelerator.is_main_process:
        print(f"✅ LoRA aktivert. Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Config for dataset
    try:
        config = AutoConfig.from_pretrained(args.init_model_path)
    except:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    dataset = NorwegianTTSDataset(args.train_jsonl, qwen_wrapper.processor, config)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=dataset.collate_fn, num_workers=2, pin_memory=True
    )
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        steps_in_epoch = 0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids'].to(model.device)
                codec_ids = batch['codec_ids'].to(model.device)
                ref_mels = batch['ref_mels'].to(model.device, dtype=model.dtype)
                text_embedding_mask = batch['text_embedding_mask'].to(model.device)
                codec_embedding_mask = batch['codec_embedding_mask'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                codec_0_labels = batch['codec_0_labels'].to(model.device)
                codec_mask = batch['codec_mask'].to(model.device)

                # --- 1. Speaker Embedding ---
                speaker_embedding = model.speaker_encoder(ref_mels).detach()

                # --- 2. Manuell Embedding-konstruksjon (for Hijack) ---
                # Splitter input_ids i tekst og codec kanaler
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # Tekst embedding
                raw_text_embeds = model.talker.model.text_embedding(input_text_ids)
                projected_text_embeds = model.talker.text_projection(raw_text_embeds)
                input_text_embedding = projected_text_embeds * text_embedding_mask

                # >>> GERMAN HIJACK START <<<
                # Vi finner language embedding variabelen
                lang_embeds = None
                if hasattr(model.talker, "language_embedding"):
                    # Lag German ID tensor
                    bs = input_ids.shape[0]
                    g_ids = torch.full((bs,), german_id, dtype=torch.long, device=model.device)
                    # Hent embedding
                    lang_emb_vec = model.talker.language_embedding(g_ids)
                    # Legg til tekst-delen (kringkastet til sekvenslengde)
                    lang_embeds = lang_emb_vec.unsqueeze(1) # [B, 1, D]
                
                if lang_embeds is not None:
                    # Legg German-signalet KUN til der det er tekst (ikke padding/codec)
                    # text_embedding_mask sikrer at vi ikke ødelegger padding
                    input_text_embedding = input_text_embedding + (lang_embeds * text_embedding_mask)
                # >>> GERMAN HIJACK END <<<

                # Codec embedding
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                # Kombiner alt
                input_embeddings = input_text_embedding + input_codec_embedding

                # Legg til AR Codec layers (lag 1-15)
                for i in range(1, 16):
                    codec_i_emb = model.talker.code_predictor.get_input_embeddings()[i-1](codec_ids[:, :, i])
                    codec_i_emb = codec_i_emb * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_emb

                # --- 3. Forward Pass ---
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                # Sub-loss calculation
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]
                
                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                loss = outputs.loss + sub_talker_loss
                
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
                unwrapped_model.talker.model.save_pretrained(save_path)
                unwrapped_model.config.save_pretrained(save_path)
                
                readme_lines = [
                    "---",
                    "base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "library_name: peft",
                    f"tags: [text-to-speech, norwegian, lora, epoch-{epoch+1}]",
                    "---",
                    f"# Qwen3-TTS Norwegian (German Hijack) - Epoch {epoch+1}",
                    "Usage: Set language='German' to speak Norwegian."
                ]
                with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write("\n".join(readme_lines))
                print(f"💾 Checkpoint lagret: {save_path}")

if __name__ == "__main__":
    train()
