# Qwen3-TTS Norwegian Finetuning (Project Kathrine)

This repository contains the code for finetuning the **Qwen2.5-based Qwen3-TTS** model for Norwegian speech synthesis.

The objective is to retain high audio fidelity (voice cloning, breath, pauses) while adapting the model to understand Norwegian pronunciation, prosody, and grammar. The training data consists of studio-quality recordings from **NPSC** (The Norwegian Parliament) and audiobooks from **LibriVox**.

## Project Status
**Current Phase:** LoRA Training (Epoch 0-30)

## Experiment Checklist

We are iterating through different finetuning strategies to solve the language adaptation problem.

- [x] **Experiment 1: Standard Finetuning (Freeze 2)**
  - **Method:** Unfroze `text_projection` and the first 2 Transformer layers.
  - **Result:** Failed. Good voice cloning quality, but retained a "German" accent. Extended training led to unstable timing (skipping/rewinding).

- [x] **Experiment 2: Aggressive Finetuning (Freeze 4)**
  - **Method:** Unfroze the first 4 layers to force deeper language learning.
  - **Result:** Failed. Total model collapse. The dataset (~9 hours) was insufficient for this amount of trainable parameters, resulting in unintelligible output.

- [ ] **Experiment 3: LoRA (Low-Rank Adaptation)**
  - **Method:** Injecting trainable adapters into all layers (up to layer 24) while freezing the base model.
  - **Hypothesis:** This should allow the model to learn Norwegian sentence melody (prosody) across the entire network without losing stability or audio quality.
  - **Status:** In progress.

## Technical Configuration (LoRA)

We use the `peft` library to target specific projection layers throughout the model architecture.

**Current Config:**
```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    task_type="FEATURE_EXTRACTION"
)
