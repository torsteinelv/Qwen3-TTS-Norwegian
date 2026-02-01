import argparse
import json
import os
import shutil
import torch
import librosa
import numpy as np
from typing import Any, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig
from huggingface_hub import HfApi
from safetensors.torch import save_file

# --- Qwen Imports ---
# Vi antar at qwen-tts er installert via pip i containeren
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

# ==========================================
# 1. INLINE DATASET KLASSE (For å unngå import-feil)
# ==========================================
AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]
MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config, lag_num=-1):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]
        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out
    
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        return mels

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = item["text"]
        audio_codes = item["audio_codes"]
        ref_audio_path = item['ref_audio']

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav, sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:,:-5],    # 1 , t
            "audio_codes": audio_codes,     # t, 16
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
                0,     # for speaker embedding
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
# 2. TRAINING SCRIPT (LoRA + Korrekt Data Flow)
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5) # LoRA tåler høyere LR
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--hf_repo_id", type=str, default=os.getenv("HF_REPO_ID"))
    args = parser.parse_args()

    # 1. Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4, 
        mixed_precision="bf16", 
        log_with="tensorboard",
        project_dir=args.output_model_path
    )

    if accelerator.is_main_process:
        print(f"🚀 Starter LoRA-trening (Komplett) | Epochs: {args.num_epochs}")
        os.makedirs(args.output_model_path, exist_ok=True)

    # 2. Last Modell
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = qwen3tts.model
    
    # 3. FRYS ALT & AKTIVER LORA
    model.requires_grad_(False)
    
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="FEATURE_EXTRACTION"
    )
    
    model.talker.model = get_peft_model(model.talker.model, peft_config)
    
    # VIKTIG: Åpne text_projection for trening (norske tegn)
    model.talker.text_projection.requires_grad_(True)

    if accelerator.is_main_process:
        print("✅ LoRA aktivert. Trenbare parametere:")
        model.talker.model.print_trainable_parameters()

    # 4. Dataset & Dataloader
    try:
        config = AutoConfig.from_pretrained(args.init_model_path)
    except:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    with open(args.train_jsonl, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate_fn,
        num_workers=4
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # 5. Hugging Face API Setup
    hf_api = None
    if args.hf_repo_id and os.getenv("HF_TOKEN"):
        hf_api = HfApi(token=os.getenv("HF_TOKEN"))
        if accelerator.is_main_process:
            print(f"🔗 Koblet til Hugging Face Repo: {args.hf_repo_id}")

    # 6. Treningsløkke
    model.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        steps_in_epoch = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                
                # Hent data fra batch
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask'] 
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                # Forward pass logikk
                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                raw_text_embeds = model.talker.model.text_embedding(input_text_ids)
                projected_text_embeds = model.talker.text_projection(raw_text_embeds)
                
                input_text_embedding = projected_text_embeds * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

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

            if step % 10 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # --- SLUTT PÅ EPOCH: LAGRING & OPPLASTING ---
        if accelerator.is_main_process:
            avg_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
            print(f"📊 Epoch {epoch} ferdig. Snitt Loss: {avg_loss:.4f}")

            checkpoint_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            model.talker.model.save_pretrained(checkpoint_dir)
            
            # Lagre text_projection
            tp_path = os.path.join(checkpoint_dir, "text_projection.bin")
            torch.save(model.talker.text_projection.state_dict(), tp_path)

            # RIKTIG METADATA
            readme_content = f"""---
base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
library_name: peft
tags:
- text-to-speech
- qwen3-tts
- norwegian
- lora
- epoch-{epoch}
---
# Qwen3-TTS Norwegian LoRA (Epoch {epoch})

Trained on NPSC and LibriVox data.
- **Loss:** {avg_loss:.4f}
- **Steps:** {steps_in_epoch}
"""
            with open(os.path.join(checkpoint_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(readme_content)

            print(f"💾 Checkpoint lagret lokalt: {checkpoint_dir}")

            if hf_api:
                try:
                    print(f"☁️ Laster opp Epoch {epoch} til HF...")
                    hf_api.upload_folder(
                        folder_path=checkpoint_dir,
                        repo_id=args.hf_repo_id,
                        path_in_repo=f"checkpoints/epoch_{epoch}",
                        repo_type="model",
                        commit_message=f"Upload Epoch {epoch} (Loss: {avg_loss:.4f})"
                    )
                    hf_api.upload_folder(
                        folder_path=checkpoint_dir,
                        repo_id=args.hf_repo_id,
                        path_in_repo=".", 
                        repo_type="model",
                        commit_message=f"Update main model to Epoch {epoch}"
                    )
                    print("✅ Opplasting fullført!")
                except Exception as e:
                    print(f"⚠️ Opplasting feilet: {e}")

if __name__ == "__main__":
    train()
