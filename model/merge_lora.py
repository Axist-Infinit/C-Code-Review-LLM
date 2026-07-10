#!/usr/bin/env python3
"""Merge a QLoRA surface adapter into the base model for Ollama serving.

The surface lane serves through Ollama (GGUF), but training produces a PEFT LoRA
adapter. This closes that gap: load the base in fp16, apply the adapter, merge the
low-rank deltas into the weights (merge_and_unload), and save a standalone fp16
safetensors model that Ollama can import directly (Qwen2 arch) via a Modelfile
`FROM ./llm-surface-merged`, then re-quantize.

    python model/merge_lora.py --base Qwen/Qwen2.5-Coder-14B-Instruct \
        --adapter ./llm-surface-lora --out ./llm-surface-merged

Note: merges in fp16 (NOT the 4-bit training quantization) so no quant error is
baked into the merged weights; Ollama does the final Q4_K_M quantization on import.
Needs enough RAM/VRAM to hold the 14B in fp16 (~28 GB) — merge runs on CPU if VRAM
is short (slower but works).
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    ap.add_argument("--base", required=True, help="Base model dir or HF id (must match training)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir from train_surface_sft.py")
    ap.add_argument("--out", required=True, help="Output dir for the merged fp16 model")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda (cpu if VRAM is short)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not os.path.isdir(args.adapter):
        sys.exit(f"[ERR] adapter dir not found: {args.adapter}")

    base_local = os.path.isdir(args.base)
    device_map = None if args.device == "cpu" else (args.device if args.device != "auto" else "auto")
    print(f"[load] base {args.base} in fp16 (device_map={device_map})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map=device_map,
        local_files_only=base_local)
    print(f"[load] adapter {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("[merge] merge_and_unload ...")
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=base_local)
    tok.save_pretrained(args.out)
    print(f"[OK] merged model -> {args.out}")
    print("[next] import into Ollama:\n"
          f"       ollama show qwen2.5-coder:14b --modelfile > Modelfile.base\n"
          f"       # replace the FROM line with:  FROM {args.out}\n"
          f"       ollama create qwen2.5-coder-surface:14b -f Modelfile.surface")


if __name__ == "__main__":
    main()
