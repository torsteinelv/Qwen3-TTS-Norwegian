import argparse
import json
import os
import re
import shutil
import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from huggingface_hub import HfApi, list_repo_files, snapshot_download

# --- VIKTIG: PEFT (LoRA) Biblioteket ---
from peft import LoraConfig, get_peft_model, TaskType

target_speaker_embedding = None

def get_latest_checkpoint(repo_id):
    """Sjekker Hugging Face for siste checkpoint for å kunne fortsette trening (Resume)."""
    print(f"🔍 Ser etter checkpoints i {repo_id}...")
    try:
        files = list_repo_files(repo_id)
        epoch_nums = []
        for f in files:
            # Ser etter mønsteret checkpoints/epoch_X/
            match = re.search(r"checkpoints/epoch_(\d+)/", f)
            if match:
                epoch_nums.append(int(match.group(1)))
        
        if not epoch_nums:
            print("⚠️ Fant ingen tidligere checkpoints. Starter fra scratch.")
            return None, 0
        
        latest_epoch = max(epoch_nums)
        print(f"✅ Fant siste checkpoint: Epoch {latest_epoch}")
        return latest_epoch, latest_epoch + 1
    except Exception as e:
        print(f"⚠️ Kunne ikke sjekke repo: {e}")
        return None, 0

def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--resume_repo_id", type=str, default=None)
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4) # LoRA tåler høyere learning rate enn full finetuning
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    # Initialiser Accelerator (håndterer GPU/Multi-GPU)
    accelerator = Accelerator(
        gradient_accumulation_steps=4, 
        mixed_precision="bf16", 
        log_with="tensorboard",
        project_dir=args.output_model_path
    )

    start_epoch = 0
    MODEL_PATH = args.init_model_path

    # --- RESUME LOGIKK ---
    if args.resume_repo_id:
        latest_epoch, next_epoch = get_latest_checkpoint(args.resume_repo_id)
        if latest_epoch is not None:
            accelerator.print(f"🔄 RESUMING: Laster ned Epoch {latest_epoch}...")
            try:
                download_path = snapshot_download(
                    repo_id=args.resume_repo_id,
                    allow_patterns=f"checkpoints/epoch_{latest_epoch}/*"
                )
                MODEL_PATH = os.path.join(download_path, "checkpoints", f"epoch_{latest_epoch}")
                start_epoch = next_epoch
            except Exception as e:
                accelerator.print(f"⚠️ Feil ved resume download, starter fra scratch: {e}")

    # Last modell og config
    qwen3tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    try:
        config = AutoConfig.from_pretrained(MODEL_PATH)
    except:
        config = AutoConfig.from_pretrained(args.init_model_path)

    # Last dataset
    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    model = qwen3tts.model

    # --- LORA KONFIGURASJON ---
    accelerator.print("❄️  Fryser basismodellen...")
    model.requires_grad_(False)
    
    # 1. Åpne text_projection (Critical Fix for språkforståelse ved inngangen)
    if hasattr(model.talker, "text_projection"):
        accelerator.print("🔓 Unfreezing text_projection (Input Mapping)")
        model.talker.text_projection.requires_grad_(True)

    # 2. Aktiver LoRA (Deep Learning i hjernen)
    # Target modules er basert på Qwen2.5 arkitekturen som vi inspiserte
    accelerator.print("🚀 Aktiverer LoRA på dype lag...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05, 
        bias="none",
        task_type="FEATURE_EXTRACTION" 
    )

    # Vi wrapper selve språkmodellen (talker.model)
    model.talker.model = get_peft_model(model.talker.model, peft_config)
    
    # Vis hva som faktisk trenes
    accelerator.print("📊 TRENINGS-STATUS:")
    model.talker.model.print_trainable_parameters()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model.train()

    end_epoch = start_epoch + args.num_epochs
    accelerator.print(f"🎯 Trener fra Epoch {start_epoch} til {end_epoch}")

    for epoch in range(start_epoch, end_epoch):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # --- CRITICAL FIX: Custom Forward Pass for LoRA + Projection ---
                # Siden LoRA wrapper modellen, må vi hente base_model for å nå embedding-lagene direkte
                
                # 1. Tekst Embeddings
                base_model = model.talker.model.get_base_model() if hasattr(model.talker.model, "get_base_model") else model.talker.model
                
                raw_text_embeds = base_model.text_embedding(input_text_ids)
                projected_text_embeds = model.talker.text_projection(raw_text_embeds)
                input_text_embedding = projected_text_embeds * text_embedding_mask

                # 2. Codec Embeddings
                input_codec_embedding = base_model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                # 3. Codec Predictors
                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # 4. Forward Pass (Gjennom LoRA-lagene)
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                # 5. Sub-talker Loss
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # --- LAGRING AV CHECKPOINT ---
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Lagre LoRA Adapter (Den lille filen)
            # Dette lagrer kun endringene ("Gule Lapper"), ikke hele modellen
            model.talker.model.save_pretrained(output_dir)
            
            # 2. Lagre text_projection manuelt (siden den ikke er del av LoRA-configen)
            torch.save(model.talker.text_projection.state_dict(), os.path.join(output_dir, "text_projection.bin"))
            
            # 3. Oppdater config så vi vet dette er en adapter-modell
            config_path = os.path.join(output_dir, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r+') as f:
                    data = json.load(f)
                    data["custom_usage"] = "qwen3_tts_norwegian_lora"
                    f.seek(0)
                    json.dump(data, f, indent=2)

            # 4. Lagre en config.json for repo-strukturen
            with open(os.path.join(output_dir, "config.json"), 'w') as f:
                json.dump({"model_type": "qwen3_lora_adapter", "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}, f)

            accelerator.print(f"✅ LoRA Checkpoint lagret: {output_dir}")

            # Upload til Hugging Face
            repo_id = os.getenv("HF_REPO_ID")
            token = os.getenv("HF_TOKEN")
            if repo_id and token:
                try:
                    accelerator.print(f"🚀 Laster opp Epoch {epoch}...")
                    api = HfApi(token=token)
                    api.upload_folder(
                        folder_path=output_dir,
                        repo_id=repo_id,
                        path_in_repo=f"checkpoints/epoch_{epoch}",
                        repo_type="model"
                    )
                    accelerator.print(f"☁️  Opplasting ferdig!")
                except Exception as e:
                    accelerator.print(f"⚠️ Feil ved opplasting: {e}")

if __name__ == "__main__":
    train()
