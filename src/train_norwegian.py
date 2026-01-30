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

target_speaker_embedding = None

def get_latest_checkpoint(repo_id):
    """Finner siste epoke i et Hugging Face repo"""
    print(f"🔍 Ser etter checkpoints i {repo_id}...")
    try:
        files = list_repo_files(repo_id)
        # Ser etter mapper som heter checkpoints/epoch_X
        epoch_nums = []
        for f in files:
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
        print(f"⚠️ Kunne ikke sjekke repo (kanskje det er tomt?): {e}")
        return None, 0

def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, required=True) # Basis-modell
    parser.add_argument("--resume_repo_id", type=str, default=None)   # Hvilket repo skal vi fortsette fra?
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=10)         # Hvor mange FLERE epoker skal vi kjøre?
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=4, 
        mixed_precision="bf16", 
        log_with="tensorboard",
        project_dir=args.output_model_path
    )

    # --- LOGIKK FOR RESUME ---
    start_epoch = 0
    MODEL_PATH = args.init_model_path

    if args.resume_repo_id:
        latest_epoch, next_epoch = get_latest_checkpoint(args.resume_repo_id)
        if latest_epoch is not None:
            accelerator.print(f"🔄 RESUMING: Laster ned Epoch {latest_epoch} fra Hugging Face...")
            
            # Last ned kun den mappen vi trenger til en cache-mappe
            download_path = snapshot_download(
                repo_id=args.resume_repo_id,
                allow_patterns=f"checkpoints/epoch_{latest_epoch}/*"
            )
            # snapshot_download legger det i en undermappe struktur, vi må peke på den
            MODEL_PATH = os.path.join(download_path, "checkpoints", f"epoch_{latest_epoch}")
            start_epoch = next_epoch
            accelerator.print(f"📂 Modellen er lastet ned til: {MODEL_PATH}")
            accelerator.print(f"🚀 Starter trening fra Epoch {start_epoch}")

    # Last modell
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # Prøv å laste config, hvis resume feiler/mangler config, bruk init_path sin
    try:
        config = AutoConfig.from_pretrained(MODEL_PATH)
    except:
        config = AutoConfig.from_pretrained(args.init_model_path)

    # Dataset setup
    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    model = qwen3tts.model

    # --- 1. FRYS ALT FØRST ---
    accelerator.print("❄️  Freezing ALL model parameters...")
    model.requires_grad_(False)
    
    # --- 2. FINN OG ÅPNE TEXT PROJECTION ---
    if hasattr(model.talker, "text_projection"):
        accelerator.print("🔓 Unfreezing text_projection (CORRECT PATH FOUND)")
        model.talker.text_projection.requires_grad_(True)
    else:
        accelerator.print("⚠️ ADVARSEL: Fant ikke text_projection direkte.")

    # --- 3. ÅPNE FLERE LAG (4 lag) ---
    LAYERS_TO_UNFREEZE = 0
    accelerator.print(f"🔓 Unfreezing first {LAYERS_TO_UNFREEZE} talker layers...")
    if hasattr(model.talker.model, "layers"):
        for i in range(LAYERS_TO_UNFREEZE):
            model.talker.model.layers[i].requires_grad_(True)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(f"🔥 Trenbare parametere: {len(trainable_params)} tensorer")
    
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    model.train()

    # Juster loopen til å starte fra start_epoch
    end_epoch = start_epoch + args.num_epochs
    accelerator.print(f"🎯 Skal trene fra Epoch {start_epoch} til {end_epoch}")

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

                # --- CRITICAL FIX ---
                raw_text_embeds = model.talker.model.text_embedding(input_text_ids)
                projected_text_embeds = model.talker.text_projection(raw_text_embeds)
                input_text_embedding = projected_text_embeds * text_embedding_mask
                # --------------------

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

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # --- LAGRING ---
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            
            # Kopier fra der vi hentet modellen (enten base eller lastet ned cache)
            if os.path.exists(MODEL_PATH):
                 shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)
            
            # Oppdater config (Bevar data!)
            config_path = os.path.join(output_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                config_dict["tts_model_type"] = "custom_voice"
                talker_cfg = config_dict.get("talker_config", {})
                talker_cfg["spk_id"] = {args.speaker_name: 3000}
                talker_cfg["spk_is_dialect"] = {args.speaker_name: False}
                config_dict["talker_config"] = talker_cfg

                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped.state_dict().items()}
            
            keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
            for k in keys_to_drop: del state_dict[k]

            if target_speaker_embedding is not None:
                state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().cpu()

            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            accelerator.print(f"✅ Checkpoint lagret lokalt: {output_dir}")

            # Upload til HF
            repo_id = os.getenv("HF_REPO_ID")
            if repo_id:
                try:
                    accelerator.print(f"🚀 Laster opp checkpoint for epoch {epoch} til {repo_id}...")
                    api = HfApi()
                    api.upload_folder(
                        folder_path=output_dir,
                        repo_id=repo_id,
                        path_in_repo=f"checkpoints/epoch_{epoch}",
                        repo_type="model"
                    )
                    accelerator.print(f"☁️  Opplasting ferdig! Dataen er trygg.")
                except Exception as e:
                    accelerator.print(f"⚠️ Feil ved opplasting: {e}")

if __name__ == "__main__":
    train()
