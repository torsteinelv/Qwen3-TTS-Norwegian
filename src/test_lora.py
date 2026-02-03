import os
import sys
import time
import argparse
import numpy as np
import torch
import soundfile as sf
from huggingface_hub import snapshot_download
from peft import PeftModel


def add_qwen_path(qwen_path: str | None):
    candidates = []
    if qwen_path:
        candidates.append(qwen_path)

    candidates += [
        os.environ.get("QWEN_TTS_PATH", ""),
        "/workspace/Qwen3-TTS",
        os.path.join(os.getcwd(), "Qwen3-TTS"),
    ]
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


def safe_torch_load(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def pick_dtype(device: str, force_fp32: bool):
    if force_fp32:
        return torch.float32
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def wav_stats(wav: np.ndarray):
    if wav is None:
        return None
    if wav.size == 0:
        return {"len": 0, "min": 0.0, "max": 0.0, "rms": 0.0, "absmax": 0.0}
    mn = float(np.min(wav))
    mx = float(np.max(wav))
    absmax = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float64))) + 1e-12))
    return {"len": int(wav.size), "min": mn, "max": mx, "rms": rms, "absmax": absmax}


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS LoRA test (fast + non-silent defaults)")

    parser.add_argument("--repo_id", type=str, default="telvenes/qwen3-tts-norsk-finetune")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/epoch_1")
    parser.add_argument("--local_checkpoint", type=str, default=None)
    parser.add_argument("--qwen_path", type=str, default=None)

    parser.add_argument("--text", type=str, default="Hei! Nå tester vi norsk uttale. Kylling, ski og kino.")
    parser.add_argument("--ref_audio", type=str, required=True)
    parser.add_argument("--ref_text", type=str, default=None)
    parser.add_argument("--x_vector_only", action="store_true")

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--language", type=str, default="Auto")

    # Viktig: CPU speed & silence
    parser.add_argument("--max_new_tokens", type=int, default=256, help="CPU: 128-256 anbefales")
    parser.add_argument("--seed", type=int, default=1234)

    # Sampling defaults (anbefalt for TTS)
    parser.add_argument("--greedy", action="store_true", help="Sett do_sample=False (kan gi stillhet/treig)")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # Subtalker sampling (codebooks 2-16)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=0.9)
    parser.add_argument("--subtalker_temperature", type=float, default=0.8)

    # Andre nyttige toggles
    parser.add_argument("--non_streaming_mode", action="store_true", help="Bruk non_streaming_mode=True (ofte raskere)")
    parser.add_argument("--force_fp32", action="store_true", help="Tving float32 selv på GPU (ikke anbefalt)")
    parser.add_argument("--no_lora", action="store_true", help="Ikke last LoRA (for A/B test)")
    parser.add_argument("--no_text_projection", action="store_true", help="Ikke last text_projection.bin (for A/B test)")

    args = parser.parse_args()

    add_qwen_path(args.qwen_path)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    if not os.path.exists(args.ref_audio):
        print(f"❌ Finner ikke ref_audio: {args.ref_audio}")
        return

    # Seed (gir reproducerbar sampling)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = pick_dtype(args.device, args.force_fp32)

    print("\n" + "=" * 60)
    print(f"🧪 TEST checkpoint: {args.checkpoint_dir}")
    print(f"🖥️  device={args.device} dtype={dtype} max_new_tokens={args.max_new_tokens} seed={args.seed}")
    print("=" * 60)

    # 1) Finn checkpoint
    t0 = time.time()
    if args.local_checkpoint:
        adapter_path = args.local_checkpoint
        print(f"📂 Bruker lokal checkpoint: {adapter_path}")
    else:
        print(f"⬇️  Laster ned fra HF: {args.repo_id} | {args.checkpoint_dir}")
        snap = snapshot_download(
            repo_id=args.repo_id,
            allow_patterns=[f"{args.checkpoint_dir}/*", f"{args.checkpoint_dir}/**"],
            repo_type="model",
        )
        adapter_path = os.path.join(snap, args.checkpoint_dir)
        print(f"✅ Checkpoint: {adapter_path}")
    print(f"⏱️  checkpoint load: {time.time()-t0:.2f}s")

    try:
        print("📦 Filer:", sorted(os.listdir(adapter_path)))
    except Exception as e:
        print(f"⚠️ Kunne ikke liste filer: {e}")

    # 2) Load base model
    t0 = time.time()
    print(f"\n❄️  Laster base model: {args.base_model}")
    model = Qwen3TTSModel.from_pretrained(
        args.base_model,
        device_map={"": args.device},
        dtype=dtype,
    )
    model.model.eval()
    torch.set_grad_enabled(False)
    print(f"⏱️  base model load: {time.time()-t0:.2f}s")

    # 3) LoRA
    if not args.no_lora:
        t0 = time.time()
        print("\n🧠 Laster LoRA...")
        model.model.talker.model = PeftModel.from_pretrained(
            model.model.talker.model,
            adapter_path,
            is_trainable=False,
        )
        model.model.talker.model.eval()
        print(f"⏱️  lora load: {time.time()-t0:.2f}s")
    else:
        print("\n🧪 Hopper over LoRA (--no_lora).")

    # 4) text_projection
    if not args.no_text_projection:
        tp_path = os.path.join(adapter_path, "text_projection.bin")
        if os.path.exists(tp_path):
            t0 = time.time()
            print("📖 Laster text_projection.bin...")
            sd = safe_torch_load(tp_path, map_location="cpu")
            model.model.talker.text_projection.load_state_dict(sd, strict=True)
            model.model.talker.text_projection.to(args.device)
            model.model.talker.text_projection.to(dtype=dtype)
            model.model.talker.text_projection.eval()
            print(f"⏱️  text_projection load: {time.time()-t0:.2f}s")
        else:
            print("⚠️ Fant ikke text_projection.bin (uttale vil ofte ligne base).")
    else:
        print("\n🧪 Hopper over text_projection (--no_text_projection).")

    # 5) Prompt
    use_xvec = args.x_vector_only or (args.ref_text is None)
    mode = "X-vector-only" if use_xvec else "ICL (ref_audio+ref_text)"
    print(f"\n🎛️  Prompt mode: {mode}")

    t0 = time.time()
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=use_xvec,
    )
    print(f"⏱️  create_voice_clone_prompt: {time.time()-t0:.2f}s")

    # 6) Generation kwargs
    do_sample = (not args.greedy)
    subtalker_dosample = (not args.greedy)

    gen_kwargs = dict(
        text=args.text,
        voice_clone_prompt=voice_prompt,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        non_streaming_mode=bool(args.non_streaming_mode),
    )

    # Bare legg til sampling-params hvis do_sample=True (slipper “ignored flags”)
    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,

            subtalker_dosample=True,
            subtalker_top_k=args.subtalker_top_k,
            subtalker_top_p=args.subtalker_top_p,
            subtalker_temperature=args.subtalker_temperature,
        )
    else:
        gen_kwargs.update(
            do_sample=False,
            subtalker_dosample=False,
        )

    print("\n🎙️  Genererer...")
    print(f"   text: {args.text}")
    print(f"   do_sample={do_sample}, subtalker_dosample={subtalker_dosample}, non_streaming_mode={bool(args.non_streaming_mode)}")
    if do_sample:
        print(f"   temp={args.temperature} top_p={args.top_p} top_k={args.top_k} rep_pen={args.repetition_penalty}")
        print(f"   sub_temp={args.subtalker_temperature} sub_top_p={args.subtalker_top_p} sub_top_k={args.subtalker_top_k}")
    else:
        print("   ⚠️ Greedy modus kan gi stillhet og være treigere på CPU.")

    t0 = time.time()
    with torch.inference_mode():
        wavs, sr = model.generate_voice_clone(**gen_kwargs)
    gen_time = time.time() - t0
    print(f"⏱️  generate_voice_clone: {gen_time:.2f}s (sr={sr})")

    wav = wavs[0] if isinstance(wavs, list) and len(wavs) > 0 else None
    st = wav_stats(wav)
    print("\n📊 WAV stats:", st)

    if wav is None or (st and st["len"] == 0):
        print("\n❌ Du fikk 0 samples ut. Hvis du kjører ICL kan det bety at modellen genererte ~0 nye koder og cuttet vekk alt.")
        print("   Prøv: --x_vector_only eller kortere ref_text, og bruk sampling (ikke --greedy).")
        return

    # Lagre
    if args.output_file:
        out = args.output_file
    else:
        safe_ckpt = args.checkpoint_dir.replace("/", "_").replace("\\", "_")
        out = f"test_{safe_ckpt}.wav"

    sf.write(out, wav.astype(np.float32), sr)
    print(f"\n✅ Lagret: {out}")

    # Stillhet-deteksjon
    if st and st["absmax"] < 1e-4:
        print("\n⚠️ Output ser ut som (nesten) stillhet.")
        print("   Dette skjer ofte hvis du kjører greedy eller for lav temperatur.")
        print("   Anbefalt CPU-test:")
        print("     - IKKE bruk --greedy")
        print("     - prøv --temperature 0.9 --top_p 0.95")
        print("     - evt øk max_new_tokens litt (256->384) hvis den stopper for tidlig")
        print("   Og sjekk at base uten LoRA faktisk gir lyd: bruk --no_lora --no_text_projection som kontroll.")


if __name__ == "__main__":
    main()
