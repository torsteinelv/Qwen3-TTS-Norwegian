# 🇳🇴 Qwen3-TTS Norwegian Finetuning (Project Kathrine)

This repository contains the code for teaching the **Qwen2.5-based Qwen3-TTS** model to speak Norwegian.

The goal is to retain the model's high audio fidelity (voice cloning, breath, pauses) while forcing it to understand Norwegian pronunciation, prosody, and grammar. We are training on studio-quality data from **NPSC** (The Norwegian Parliament) and audiobooks from **LibriVox** (Kathrine).

---

## 🚧 Status: Migrating to LoRA (Deep Brain Surgery)
After testing standard "Freeze/Unfreeze" strategies with mixed results, we are now migrating to **LoRA (Low-Rank Adaptation)**. This allows us to modify the model's behavior deep within the neural network without causing a collapse.

---

## 🧪 Experiment Log (Key Findings)

We have conducted several experiments on our Kubernetes cluster (A40/L40S GPUs). Here are the critical findings that led us to LoRA:

### ❌ Experiment 1: Standard Finetuning ("Freeze 2")
* **Method:** We froze almost the entire model but un-froze `text_projection` (input) and the first two Transformer layers (Layer 0-1).
* **Result: Great voice, but "German" accent and unstable timing.**
    * ✅ **Audio Quality:** The voice cloning of Kathrine was excellent and clean.
    * ✅ **Progress:** We heard hints of the correct dialect features (e.g., "skarre-r") around epoch 3.
    * ⚠️ **Accent:** The model struggled with Norwegian sentence melody (prosody), sounding like a German speaker reading Norwegian text.
    * ❌ **Failure:** As we ran more epochs, the output started **"rewinding"** or "skipping". The model lost its sense of time/duration, rushing through words or repeating syllables.

### ❌ Experiment 2: Aggressive Finetuning ("Freeze 4")
* **Method:** To fix the German accent, we tried unfreezing the first **4** layers instead of 2. The theory was that we needed to go deeper into the "brain" to change the language logic.
* **Result: Total Collapse ("The Soup").**
    * The model could not handle ~200M open parameters with our limited dataset size (~10 hours).
    * ❌ **Failure:** Words melted together into unintelligible gibberish (e.g., *"Hei dette er"* became *"Hedetter"*).
    * **Conclusion:** The "Freeze" method is too blunt. If we unfreeze too little, we get a German accent. If we unfreeze too much, the model forgets how to speak.

---

## 🚀 The Solution: LoRA

To solve the dilemma above, we are now using **LoRA (Low-Rank Adaptation)** via the script `src/train_lora_norwegian.py`.

**Why LoRA?**
1.  **Depth:** We inject trainable adapters into **ALL** layers of the model (down to layer 24). This allows us to fix the sentence melody and remove the "German accent".
2.  **Stability:** Since we freeze the entire base model and only train tiny adapter layers (<1% of parameters), we avoid the "Soup" problem and the "rewinding" issues. The model retains its stability.

### Configuration (PEFT)
We use the `peft` library to train `q_proj`, `v_proj`, `gate_proj`, etc., throughout the entire model:
```python
peft_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    task_type="FEATURE_EXTRACTION"
)
