# Qwen3-TTS Norwegian Finetuning (Project Kathrine)

This repository contains the code for finetuning the **Qwen2.5-based Qwen3-TTS** model for Norwegian speech synthesis.

Goal: retain high audio fidelity (voice cloning, breath, pauses) while adapting the model to Norwegian pronunciation, prosody, and grammar. Training data consists of studio-quality recordings from **NPSC** (The Norwegian Parliament) and audiobooks from **LibriVox**.

---

## Project Status
**Current Phase:** LoRA + `text_projection` training + (best-effort) Sub-Talker loss  
**Current Conclusion:** “German accent” is mostly a **text projection / text→speech alignment** problem, while “skurring” is mainly **codec detail stability** (sub-talker + LR/overfit).

---

## Experiment Checklist (What we tried)

We are iterating through finetuning strategies to solve language adaptation + audio stability.

### ✅ Experiment 1: Standard finetuning (Freeze 2)
- **Method:** Unfroze `text_projection` + first 2 transformer layers
- **Result:** Failed  
  - Voice cloning quality OK
  - Accent remained “German-ish”
  - Longer training caused unstable timing/cadence

### ✅ Experiment 2: Aggressive finetuning (Freeze 4)
- **Method:** Unfroze first 4 layers
- **Result:** Failed (collapse)
  - Dataset (~9h at the time) too small for that many trainable params

### ✅ Experiment 3: LoRA (High LR: 1e-4, higher rank)
- **Method:** LoRA on projections across the talker transformer  
- **Result:** Partial success
  - Loss fell from ~21 → ~7.5
  - Prosody/rhythm got much better (intonation “right”)
  - Articulation stayed mushy (consonants weak/missing)
- **Diagnosis:** LR too high for fine-grained detail → plateau + early overfit/noise

### ✅ Experiment 4: “Crash-proof” training pass (stability work)
This phase was about making the pipeline run reliably and identifying the real blockers.

**4A — Hidden-states / Sub-talker crash**
- **Symptom:** training crashed when sub-talker loss tried to use `outputs.hidden_states[-1]`
- **Fix:** best-effort sub-talker: if hidden_states missing → skip sub-loss instead of crash
- **Observation:** when hidden states were missing, training continued but we only trained main talker loss → weaker improvement on noise/skurring

**4B — PEFT / LoRA attach failure**
- **Symptom:**  
  `AttributeError: 'Qwen3TTSTalkerModel' object has no attribute 'prepare_inputs_for_generation'`
- **Cause:** using a PEFT wrapper class that assumes HF generation APIs exist (CausalLM-style).  
  Qwen3TTSTalkerModel is not a standard HF generation model.
- **Fix:** ensure we use a PEFT mode that does *not* require generation hooks (FEATURE_EXTRACTION-like setup) and avoid wrappers that require `prepare_inputs_for_generation`.

**4C — Silent WAV + “spinning for 10 min” inference**
- **Symptoms:**
  - Sometimes generation hung much longer than earlier (CPU: 30s → 10min)
  - Sometimes output WAV was nearly silent / perceived silent
  - Transformers printed:  
    `The following generation flags are not valid and may be ignored: ['temperature', 'top_p']`
- **Diagnosis:**
  - In some runs `do_sample=False` → temperature/top_p ignored → unexpected decode behavior
  - Very long codec sequences from some checkpoints (esp later epochs) → slow CPU decode
  - “Silent” could be extremely low RMS audio or bad decode mode depending on checkpoint
- **Fixes used:**
  - Explicit sampling mode in test (`--greedy` vs `--temperature/top_p`)
  - Lower `max_new_tokens` during debug
  - Measure RMS/absmax stats in test script to confirm if audio is truly silent

### ✅ Experiment 5: “Scratch / Stable” LoRA + Text Projection + Sub-Talker (working training loop)
This is the “stable baseline” configuration that *actually runs end-to-end* (train → save → HF upload → test).

- **Key choices:**
  - **Force 24kHz** everywhere (dataset + ref mels)
  - **Train `text_projection`** (unfrozen)
  - **LoRA kept small** (r=4, alpha=8, dropout=0.1) to reduce overfit
  - **Separate LRs**:
    - LoRA LR low (`1e-5`)
    - text_projection LR slightly higher (`3e-5` to `5e-5`)
  - **Sub-talker loss enabled** with weight ~`0.2` (best effort)
- **Result:** training stable; HF upload works once README metadata is valid.

---

## Key Findings (What we learned)

### 1) `text_projection` is required for language/accent
If `text_projection` is not trained and loaded during inference:
- accent tends to remain “German/English-ish”
- LoRA alone mainly changes prosody/intonation, not crisp phonetics

We therefore always:
- unfreeze/train `text_projection`
- save it separately as `text_projection.bin`
- load it explicitly in the test script

### 2) Overfitting shows up as skurring/noise growth after early epochs (with higher LR)
Empirically:
- Early epochs (1–2) could sound “best” even if accent was not perfect
- With LR `1e-4`, later epochs tended to develop more skurring/noise (classic overfit/instability for codec detail)
- Smaller LoRA + lower LR behaves better long-term, but needs more/better data to fully solve accent.

### 3) Sub-talker loss helps detail, but only if hidden-states are truly available
When hidden states are missing or disabled:
- sub-loss can’t run
- improvements skew toward “structure/prosody” instead of crisp articulation/detail

### 4) HF upload failure was metadata-related (README front matter)
We hit:
- `Invalid metadata in README.md: "base_model" is not allowed to be empty`
Fix: ensure README in checkpoints has valid front matter with a non-empty `base_model`.

---

## Datasets (what we currently use)

### LibriVox (Norwegian)
- Pros: more varied text, longer contexts
- Cons: variable quality, variable speaker characteristics, can push model into voice variability instead of pure language learning

### NPSC (Parliament)
- Pros: studio-like consistency, clean speech, good for phonetics/articulation
- Cons: HF dataset requires `trust_remote_code=True` and config/split correctness

---

## Practical Issues we fixed

### A) NPSC builder flags
We initially used `--hours`, but script supported `--max_hours`.  
**Fix:** use `--max_hours`.

### B) NPSC requires trust_remote_code
Error:
`Please pass trust_remote_code=True ...`
**Fix:** pass `trust_remote_code=True` in `load_dataset(...)` and/or expose a CLI flag.

### C) ref_mels shape mismatch in collate
Crash:
`Sizes of tensors must match ... Expected 655 but got 518`
**Fix:** pad/crop ref_mels to the same time dimension inside collate before `torch.cat`.

### D) Logs should reset each container run
We moved logs to `/tmp/console_log.txt` and delete it at start of entrypoint to avoid old logs accumulating.

---

## Current “Stable Baseline” Config (the one we consider the reference)

- **Prepare codes**: PREPARE_BATCH_SIZE=16
- **Train**
  - batch_size=4
  - grad_accum=4
  - epochs: 10–20 (monitor audio, not just loss)
  - lora_lr=1e-5
  - text_proj_lr=3e-5 to 5e-5
  - sub_loss_weight=0.2
  - LoRA: r=4 alpha=8 dropout=0.1
  - mixed_precision=bf16

---

## Test/Inference Notes (why results differ by flags)

We observed that some runs print:
`generation flags are not valid and may be ignored: ['temperature', 'top_p']`

Interpretation:
- If decode is effectively greedy (`do_sample=False`), then temp/top_p won’t matter.
- Some checkpoints produce very long sequences → slow CPU decode.
- “Silent” outputs must be verified via RMS stats (not only listening).

---

## TODO / Next decisions (research questions)
- Best way to force *phonetic clarity* without blowing up noise:
  - increase NPSC hours (cleaner articulation)
  - curriculum: NPSC-first then blend in LibriVox
  - consider enabling LoRA on MLP (optional) if consonants remain weak
- Confirm hidden-state availability end-to-end for sub-talker training
- Architecture introspection script to understand where the strongest language levers are (`text_projection`, embedding paths, sub-talker heads, etc.)

---
