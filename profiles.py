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
        # Any other CUDA GPU: 4090 profile is the sane discrete-GPU default
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


if __name__ == "__main__":
    name, prof = select_profile()
    print(f"[profile] {name}: {prof['description']}")
    for k, v in prof.items():
        if k != "description":
            print(f"  {k} = {v}")
