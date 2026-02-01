# Qwen3-TTS Norwegian Finetuning (Project Kathrine)

This repository contains the code for finetuning the **Qwen2.5-based Qwen3-TTS** model for Norwegian speech synthesis.

The objective is to retain high audio fidelity (voice cloning, breath, pauses) while adapting the model to understand Norwegian pronunciation, prosody, and grammar. The training data consists of studio-quality recordings from **NPSC** (The Norwegian Parliament) and audiobooks from **LibriVox**.

## Project Status
**Current Phase:** LoRA Training - Run 4 (Low LR / "Safe Mode")

## Experiment Checklist

We are iterating through different finetuning strategies to solve the language adaptation problem.

- [x] **Experiment 1: Standard Finetuning (Freeze 2)**
  - **Method:** Unfroze `text_projection` and the first 2 Transformer layers.
  - **Result:** Failed. Good voice cloning quality, but retained a "German" accent. Extended training led to unstable timing.

- [x] **Experiment 2: Aggressive Finetuning (Freeze 4)**
  - **Method:** Unfroze the first 4 layers to force deeper language learning.
  - **Result:** Failed. Total model collapse. The dataset (~9 hours) was insufficient for this amount of trainable parameters.

- [x] **Experiment 3: LoRA (High LR: 1e-4, Rank 32)**
  - **Method:** LoRA on all layers.
  - **Result:** Partial Success. Loss dropped from 21.0 to ~7.5.
  - **Observation:** The model learned correct Norwegian prosody/rhythm (intonation is perfect), but articulation remained "mushy" (consonants missing).
  - **Diagnosis:** Learning Rate was too high for fine-grained detail, causing the model to plateau.

- [ ] **Experiment 4: LoRA "Safe Mode" (Low LR: 5e-5, Rank 16)**
  - **Method:** Restarting with lower Learning Rate and reduced Rank to stabilize convergence.
  - **Goal:** Convert the "mushy" sound into clear Norwegian words by allowing finer weight adjustments.
  - **Status:** In progress (Target: 60 Epochs).

## Technical Configuration (LoRA)

We use the `peft` library to target specific projection layers throughout the model architecture.

**Current Config (Run 4):**
```python
# Adjusted for stability
peft_config = LoraConfig(
    r=16,              # Reduced from 32 to prevent overfitting/noise
    lora_alpha=32,     # Reduced from 64 for stability
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    task_type="FEATURE_EXTRACTION"
)

# Hyperparameters
# Learning Rate: 5e-5
# Batch Size: 4
