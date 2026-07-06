#!/usr/bin/env python3
"""Hardware profile selection for the C code review pipeline.

Profiles live in profiles.json next to this file. A profile is selected by,
in order of precedence:
  1. explicit --profile flag / select_profile(name=...)
  2. CCR_PROFILE environment variable
  3. auto-detection (GPU name via nvidia-smi, machine arch via uname)
"""
import json
import os
import platform
import subprocess

PROFILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles.json")


def load_profiles(path=PROFILES_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _gpu_mem_mib():
    """Total VRAM of the first GPU in MiB, or None if it can't be determined."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0].split()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None


# A discrete GPU with less than this much VRAM cannot comfortably hold the 14b
# model the "4090" profile assumes (~9GB weights + a 16k-ctx KV cache), so it is
# routed to the "laptop" profile (qwen2.5-coder:7b) instead.
_LAPTOP_VRAM_CEILING_MIB = 16000


def detect_profile():
    """Best-effort hardware detection. Returns a profile name."""
    gpu = _gpu_name()
    arch = platform.machine().lower()
    if gpu:
        g = gpu.lower()
        # DGX Spark: GB10 Grace Blackwell superchip, aarch64
        if "gb10" in g or "spark" in g or arch in ("aarch64", "arm64"):
            return "spark"
        if "4090" in g:
            return "4090"
        # A smaller discrete GPU (e.g. a 12GB laptop card) can't hold the 14b
        # model the 4090 profile assumes; route <16GB VRAM to the laptop profile.
        mem = _gpu_mem_mib()
        if mem is not None and mem < _LAPTOP_VRAM_CEILING_MIB:
            return "laptop"
        # Any other/unmeasurable CUDA GPU: 4090 profile is the sane default.
        return "4090"
    if arch in ("aarch64", "arm64"):
        return "spark"
    return "cpu"


def select_profile(name=None, path=PROFILES_PATH):
    """Return (profile_name, profile_dict)."""
    profiles = load_profiles(path)
    name = name or os.environ.get("CCR_PROFILE") or detect_profile()
    if name not in profiles:
        raise KeyError(
            f"Unknown profile {name!r}; available: {', '.join(sorted(profiles))}"
        )
    return name, profiles[name]


def add_profile_arg(ap):
    """Attach the standard --profile argument to an argparse parser."""
    ap.add_argument(
        "--profile", default=None,
        help="Hardware profile (4090|spark|cpu). Default: $CCR_PROFILE or auto-detect.",
    )
    return ap


# --- training-config mapping -------------------------------------------------
# Maps a profile's declared classifier dtype to the precision flag understood by
# model/train_vuln_model.py (and HF TrainingArguments). This is the single
# source of truth that wires profiles.json -> the trainer; before this existed
# the trainer silently defaulted to fp32/batch16 regardless of the profile, so
# the "per-machine training adaptation" was fiction. Pure-Python + stdlib only
# so it is unit-testable without torch.
#
#   float16  -> fp16=True,  bf16=False   (Ada/RTX 4090, sm_89)
#   bfloat16 -> fp16=False, bf16=True    (Grace-Blackwell/DGX Spark, sm_121)
#   float32  -> fp16=False, bf16=False   (CPU fallback / no mixed precision)
_DTYPE_TO_PRECISION = {
    "float16": {"fp16": True, "bf16": False},
    "fp16": {"fp16": True, "bf16": False},
    "bfloat16": {"fp16": False, "bf16": True},
    "bf16": {"fp16": False, "bf16": True},
    "float32": {"fp16": False, "bf16": False},
    "fp32": {"fp16": False, "bf16": False},
}


def precision_flags_for_dtype(dtype):
    """Return {'fp16': bool, 'bf16': bool} for a profile dtype string.

    Unknown dtypes fall back to full precision (no mixed-precision flag), which
    is always safe — it just trains slower. Never raises so the trainer cannot
    be broken by an unexpected profile value.
    """
    key = str(dtype or "").strip().lower()
    return dict(_DTYPE_TO_PRECISION.get(key, {"fp16": False, "bf16": False}))


def training_config(name=None, path=PROFILES_PATH):
    """Resolve the per-machine TRAINING configuration for a profile.

    Returns a dict with:
      profile      -> resolved profile name
      batch_size   -> per-device train/eval batch size (int)
      dtype        -> the profile's declared classifier dtype string
      fp16, bf16   -> mutually-exclusive mixed-precision flags for the trainer

    This is what train.sh / train_classifier.sh consult to pass the right
    --batch and precision flag per detected machine (overridable by env).
    """
    name, prof = select_profile(name=name, path=path)
    dtype = prof.get("classifier_dtype", "float32")
    flags = precision_flags_for_dtype(dtype)
    return {
        "profile": name,
        "batch_size": int(prof.get("classifier_batch_size", 16)),
        "dtype": dtype,
        "fp16": flags["fp16"],
        "bf16": flags["bf16"],
    }


def _emit_training_env(name=None, path=PROFILES_PATH):
    """Print shell-evalable assignments for the resolved training config.

    Used by train.sh / train_classifier.sh:  eval "$(python profiles.py --training-env)"
    Emits CCR_BATCH, CCR_FP16 (1/0), CCR_BF16 (1/0), CCR_PRECISION_FLAG, CCR_PROFILE.
    """
    cfg = training_config(name=name, path=path)
    flag = "--fp16" if cfg["fp16"] else ("--bf16" if cfg["bf16"] else "")
    print(f"CCR_PROFILE={cfg['profile']}")
    print(f"CCR_BATCH={cfg['batch_size']}")
    print(f"CCR_FP16={1 if cfg['fp16'] else 0}")
    print(f"CCR_BF16={1 if cfg['bf16'] else 0}")
    print(f"CCR_PRECISION_FLAG={flag}")


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    # `--training-env` emits shell assignments for the train scripts to eval;
    # `--training-config` dumps the resolved training config as JSON.
    forced = None
    for i, a in enumerate(argv):
        if a == "--profile" and i + 1 < len(argv):
            forced = argv[i + 1]
    if "--training-env" in argv:
        _emit_training_env(name=forced)
    elif "--training-config" in argv:
        print(json.dumps(training_config(name=forced), indent=2))
    else:
        name, prof = select_profile(name=forced)
        print(f"[profile] {name}: {prof['description']}")
        for k, v in prof.items():
            if k != "description":
                print(f"  {k} = {v}")
