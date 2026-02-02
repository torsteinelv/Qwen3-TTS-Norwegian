#!/usr/bin/env python3
"""
Qwen3-TTS Norwegian Fine-Tuning (FINAL FIX V2)
==============================================
1. Fixes static/noise by adding sub-talker loss.
2. Fixes language learning by unfreezing text_projection.
3. Fixes 'NoneType' crash by forcing output_hidden_states in config.
4. Forces 24kHz sampling rate.
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
from huggingface_hub import HfApi

# ==========================================
# 0. IMPORT FIX
# ==========================================
sys.path.append("/workspace/Qwen3-TTS")
try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
except ImportError:
    print("⚠️ Kunne ikke importere Qwen3TTSModel. Sjekk paths.")
    sys.exit(1)

# ==========================================
# 1. DATASET & ROBUST COLLATOR
# ==========================================
AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]

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
            # Tving 24000 Hz her.
            audio, sr = librosa.load(x, sr=24000, mono=True)
            
            if len(audio) > 24000 * 15: 
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
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000
        ).transpose(1, 2)
        return mels

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Robust path finding
        ref_path = item.get("ref_audio") or item.get("ref_audio_path") or item.get("audio_path")
        if not ref_path:
            return self.__getitem__((idx + 1) % len(self.items))

        text = self._build_assistant_text(item["text"])
        text_ids = self._tokenize_texts(text)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        normalized = self._normalize_audio_inputs([ref_path])
        wav, sr = normalized[0]
        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:,:-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel
        }
        
    def collate_fn(self, batch):
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
                0, 
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
            codec_embedding_mask[i, 6] = False

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
# 2. TRENING (MAIN)
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, default="./qwen3-tts-norwegian-lora")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))
    args = parser.parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_model_path
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_model_path, exist_ok=True)
        print("🚀 Starter norsk TTS-trening (Med Sub-Talker Loss & Real Unfreeze)")
        print(f"   LR: {args.lr}, Upload hver {args.save_every}. epoch")
    
    # HF Setup
    hf_api = None
    if args.hf_repo_id and accelerator.is_main_process:
        try:
            hf_api = HfApi()
            print(f"☁️  HF Upload aktivert: {args.hf_repo_id}")
        except Exception as e:
            print(f"⚠️  HF Error: {e}")

    qwen_wrapper = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        dtype=torch.bfloat16, 
        device_map={"": accelerator.device},
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    model = qwen_wrapper.model 

    model.requires_grad_(False)
    
    peft_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="FEATURE_EXTRACTION"
    )
    model.talker.model = get_peft_model(model.talker.model, peft_config)
    
    # --- CRITICAL FIX START: FORCE CONFIG TO RETURN HIDDEN STATES ---
    # Dette fikser "NoneType is not subscriptable" krasjen.
    # Vi tvinger konfigurasjonen til å alltid beregne hidden states.
    model.talker.config.output_hidden_states = True
    if hasattr(model.talker.model, "config"):
        model.talker.model.config.output_hidden_states = True
    # --- CRITICAL FIX END ---

    # Unfreeze text_projection (for å lære norsk)
    if hasattr(model.talker, "text_projection"):
        print("🔓 Unfreezing text_projection parameters...")
        for p in model.talker.text_projection.parameters():
            p.requires_grad = True

    if accelerator.is_main_process:
        print(f"✅ LoRA aktivert. Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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

                speaker_embedding = model.speaker_encoder(ref_mels).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                raw_text_embeds = model.talker.model.text_embedding(input_text_ids)
                projected_text_embeds = model.talker.text_projection(raw_text_embeds)
                input_text_embedding = projected_text_embeds * text_embedding_mask

                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True 
                )

                loss_main = outputs.loss

                # Sjekk at vi faktisk fikk hidden states (Double safety check)
                if outputs.hidden_states is None:
                    # Hvis config-fixen feiler, bruk bare main loss for å unngå krasj
                    if step == 0 and accelerator.is_main_process:
                         print("⚠️ ADVARSEL: Fikk fortsatt ikke hidden states. Skipper sub-talker loss dette steget.")
                    loss = loss_main
                else:
                    # Sub-Talker Loss (Fjerner støy)
                    last_hidden = outputs.hidden_states[-1]
                    active_mask = codec_mask[:, :-1]
                    
                    if active_mask.sum() > 0:
                        hs = last_hidden[:, :-1][active_mask]
                        codes = codec_ids[:, :-1, :][active_mask]
                        
                        sub_logits, sub_loss = model.talker.forward_sub_talker_finetune(
                            codec_ids=codes,
                            talker_hidden_states=hs,
                        )
                        loss = loss_main + sub_loss
                    else:
                        loss = loss_main

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                steps_in_epoch += 1
        
        avg_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % args.save_every == 0:
                save_path = os.path.join(args.output_model_path, f"epoch-{epoch+1}")
                os.makedirs(save_path, exist_ok=True)
                
                model.talker.model.save_pretrained(save_path)
                
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model.talker, "text_projection"):
                     torch.save(
                        unwrapped_model.talker.text_projection.state_dict(),
                        os.path.join(save_path, "text_projection.bin")
                    )
                
                readme_lines = [
                    "---",
                    "base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "library_name: peft",
                    f"tags: [text-to-speech, norwegian, lora, pure-lora, epoch-{epoch+1}]",
                    "---",
                    f"# Qwen3-TTS Norwegian (Final Fix) - Epoch {epoch+1}",
                    "Includes Sub-Talker Loss training."
                ]
                with open(os.path.join(save_path, "README.md"), "w", encoding="utf-8") as f:
                    f.write("\n".join(readme_lines))
                
                print(f"💾 Lagret: {save_path}")

                if hf_api:
                    try:
                        hf_api.upload_folder(
                            folder_path=save_path,
                            repo_id=args.hf_repo_id,
                            path_in_repo=f"checkpoints/epoch_{epoch+1}",
                            repo_type="model"
                        )
                        print(f"☁️  HF Upload OK")
                    except Exception as e:
                        print(f"⚠️  HF Upload Feil: {e}")

if __name__ == "__main__":
    train()
