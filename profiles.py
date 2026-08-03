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


# VRAM tiers for the Ollama reviewer/explainer model. What decides the tier is
# weights + KV cache, and the KV cache is sized by the profile's own
# ollama_num_ctx (surface_review.py honours it rather than forcing 16384):
#
#   qwen2.5-coder:14b q4  = 9.0GB weights; KV = 1.5GB @ 8192, 3.0GB @ 16384
#   qwen2.5-coder:7b  q4  = 4.7GB weights; KV = 0.7GB @ 8192
#
# So a 24GB card runs the 14b at 16384 ("4090"); a 12GB card runs the SAME 14b at
# 8192 for ~10.5GB resident, measured 100% on GPU ("laptop"); anything smaller
# must stay on the 7b or Ollama spills layers into host RAM and crawls
# ("laptop-small"). Below 16GB the 4090 profile's 16k context is what does not
# fit — not the 14b itself.
_LAPTOP_VRAM_CEILING_MIB = 16000
_SMALL_GPU_VRAM_MIB = 11500


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
        # A smaller discrete GPU can't hold the 16k-context KV cache the 4090
        # profile assumes; route by VRAM tier (see the constants above).
        mem = _gpu_mem_mib()
        if mem is not None and mem < _LAPTOP_VRAM_CEILING_MIB:
            return "laptop-small" if mem < _SMALL_GPU_VRAM_MIB else "laptop"
        # Any other/unmeasurable CUDA GPU: 4090 profile is the sane default.
        return "4090"
    if arch in ("aarch64", "arm64"):
        return "spark"
    return "cpu"


# --- hardware-aware model recommendation ------------------------------------
# What decides whether an Ollama model runs WELL is residency: weights + KV cache
# must fit in VRAM, or llama.cpp offloads layers to host RAM, where it does not
# fail — it just crawls (often 10-30x slower). So the recommender computes the
# real footprint instead of guessing from a GPU name.
#
# kv_bytes_per_token = 2 (K and V) * n_kv_heads * head_dim * 2 bytes (f16),
# summed over layers. Measured values for the qwen2.5-coder q4_K_M family:
#   7b : 28 layers, 4 kv heads, d=128 -> 57,344 B/token   (weights 4.7GB)
#   14b: 48 layers, 8 kv heads, d=128 -> 196,608 B/token  (weights 9.0GB)
#   32b: 64 layers, 8 kv heads, d=128 -> 262,144 B/token  (weights 20.0GB)
_MIB = 1024 * 1024
MODEL_CATALOG = [
    {"tag": "qwen2.5-coder:32b", "weights_mib": 20 * 1024, "kv_bytes_per_token": 262144, "rank": 3},
    {"tag": "qwen2.5-coder:14b", "weights_mib": 9 * 1024, "kv_bytes_per_token": 196608, "rank": 2},
    {"tag": "qwen2.5-coder:7b", "weights_mib": 4700, "kv_bytes_per_token": 57344, "rank": 1},
]

# Headroom left free for the display/compositor, CUDA context and fragmentation.
# Without it a "just fits" model thrashes the moment anything else touches the GPU.
VRAM_HEADROOM_MIB = 1200


def model_footprint_mib(entry, num_ctx):
    """VRAM a model needs fully resident at this context length."""
    kv_mib = (entry["kv_bytes_per_token"] * int(num_ctx)) / _MIB
    return int(round(entry["weights_mib"] + kv_mib))


def recommend_model(vram_mib=None, num_ctx=8192, installed=None, catalog=None):
    """Pick the strongest model that stays fully resident on this GPU.

    Returns a dict: {tag, footprint_mib, fits, installed, reason, alternatives}.
    ``installed`` is an optional list of Ollama tags already pulled; when given,
    the recommendation prefers a pulled model and reports the better one that
    would fit if it were pulled, rather than silently naming something absent.
    Pure arithmetic + data — no GPU, no network, so it is unit-testable.
    """
    catalog = catalog if catalog is not None else MODEL_CATALOG
    ranked = sorted(catalog, key=lambda e: -e["rank"])
    if vram_mib is None:                      # CPU-only / unmeasurable: smallest
        smallest = ranked[-1]
        return {"tag": smallest["tag"],
                "footprint_mib": model_footprint_mib(smallest, num_ctx),
                "fits": False, "installed": None,
                "reason": "no CUDA GPU detected (or VRAM unreadable) — using the "
                          "smallest model; expect CPU-speed inference",
                "alternatives": []}

    budget = int(vram_mib) - VRAM_HEADROOM_MIB
    fitting = [e for e in ranked if model_footprint_mib(e, num_ctx) <= budget]
    if not fitting:
        smallest = ranked[-1]
        return {"tag": smallest["tag"],
                "footprint_mib": model_footprint_mib(smallest, num_ctx),
                "fits": False, "installed": None,
                "reason": f"even {smallest['tag']} needs "
                          f"{model_footprint_mib(smallest, num_ctx)}MiB at ctx {num_ctx} "
                          f"but only {budget}MiB is usable — it will spill to host RAM",
                "alternatives": []}

    best = fitting[0]
    have = None if installed is None else [t.strip() for t in installed]
    chosen, reason = best, (f"largest model that stays fully on the GPU: "
                            f"{model_footprint_mib(best, num_ctx)}MiB of {budget}MiB usable "
                            f"at ctx {num_ctx}")
    upgrade = []
    if have is not None and not _tag_present(best["tag"], have):
        pulled = [e for e in fitting if _tag_present(e["tag"], have)]
        if pulled:
            chosen = pulled[0]
            reason = (f"{best['tag']} would fit but is not pulled; using the best "
                      f"pulled model that fits")
            upgrade = [best["tag"]]
        else:
            reason = f"{best['tag']} fits this GPU but is not pulled yet"
    return {"tag": chosen["tag"],
            "footprint_mib": model_footprint_mib(chosen, num_ctx),
            "fits": True,
            "installed": None if have is None else _tag_present(chosen["tag"], have),
            "reason": reason,
            "alternatives": upgrade}


def tag_present(tag, installed):
    """Is this Ollama tag pulled?

    Exact match, or — ONLY when the requested tag omits the ':size' suffix — the
    same base name, so "mistral" is satisfied by "mistral:latest". A different
    size of the same family must never count: having qwen2.5-coder:7b does not
    mean qwen2.5-coder:32b is available.
    """
    if any(t == tag for t in installed):
        return True
    if ":" in tag:
        return False
    return any(t.split(":")[0] == tag for t in installed)


_tag_present = tag_present   # internal alias


def detect_hardware():
    """Best-effort facts about THIS machine: {gpu, vram_mib, ram_mib, arch}."""
    ram_mib = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_mib = int(line.split()[1]) // 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    return {"gpu": _gpu_name(), "vram_mib": _gpu_mem_mib(),
            "ram_mib": ram_mib, "arch": platform.machine().lower()}


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
    elif "--recommend-model" in argv:
        # What model this machine should actually run, from measured VRAM.
        hw = detect_hardware()
        _, prof = select_profile(name=forced)
        ctx = int(prof.get("ollama_num_ctx", 8192))
        rec = recommend_model(vram_mib=hw["vram_mib"], num_ctx=ctx)
        print(json.dumps({"hardware": hw, "num_ctx": ctx, "recommendation": rec}, indent=2))
    else:
        name, prof = select_profile(name=forced)
        print(f"[profile] {name}: {prof['description']}")
        for k, v in prof.items():
            if k != "description":
                print(f"  {k} = {v}")
