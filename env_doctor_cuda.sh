#!/usr/bin/env bash
# POST-INSTALL ASSERTION script — run on REAL hardware (RTX 4090 or DGX Spark)
# AFTER torch is installed. CPU-only / dev boxes cannot run this; it requires a
# CUDA GPU and a CUDA-enabled torch and will exit non-zero otherwise (by design).
#
# It asserts, with a non-zero exit on any failure:
#   1. torch imports and torch.cuda.is_available() is True
#   2. a tiny CUDA matmul runs and the result is finite (kernels actually work)
#   3. the device compute capability is one the installed wheel has kernels for,
#      and WARNS loudly if the GPU is sm_89 (4090) or sm_121 (Spark) but the
#      wheel was built without that arch (the classic "no kernel image is
#      available for execution on the device" trap).
#
# Usage:  ./env_doctor_cuda.sh        # uses .venv if present
set -euo pipefail
VENV="${VENV:-.venv}"
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"

python - <<'PY'
import sys

def fail(msg):
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except Exception as e:
    fail(f"could not import torch: {e}")

print(f"[doctor] torch {torch.__version__}")
ver = getattr(torch, "version", None)
print(f"[doctor] built for CUDA: {getattr(ver, 'cuda', None)}")

# 1) CUDA visible?
if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is False — wrong wheel (CPU build?) or driver "
         "not visible. On aarch64 install the cu130 NIGHTLY wheel; on x86_64 the "
         "channel must match your driver's CUDA (see install_torch_nightly_for_new_gpu.sh).")

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
sm = f"sm_{cap[0]}{cap[1]}"
print(f"[doctor] device 0: {name}  capability {cap[0]}.{cap[1]} ({sm})")

# 2) tiny CUDA matmul -> finite?
try:
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    c = (a @ b).sum()
    torch.cuda.synchronize()
except Exception as e:
    fail(f"CUDA matmul raised — the wheel likely lacks kernels for {sm} "
         f"('no kernel image' error). Install a wheel built for {sm}. Detail: {e}")
if not torch.isfinite(c).item():
    fail("CUDA matmul produced a non-finite result; the CUDA runtime is unhealthy.")
print(f"[doctor] CUDA matmul OK (finite, sum={c.item():.3f})")

# 3) compute-capability coverage check
try:
    archs = torch.cuda.get_arch_list()  # e.g. ['sm_80','sm_86','sm_89','sm_90']
except Exception:
    archs = []
print(f"[doctor] wheel arch list: {archs or 'unknown'}")

KNOWN = {
    "sm_89": "RTX 4090 (Ada) — needs a cu12.x/cu13.x wheel built with sm_89",
    "sm_121": "DGX Spark (Grace-Blackwell GB10) — needs a cu130 NIGHTLY wheel with sm_121",
}
if sm in KNOWN and archs and sm not in archs:
    print(f"[WARN] GPU is {sm} ({KNOWN[sm]}) but the installed wheel's arch list "
          f"{archs} does NOT include {sm}. The matmul above may have used PTX JIT; "
          f"native kernels are absent. Reinstall the correct wheel "
          f"(./install_torch_nightly_for_new_gpu.sh).", file=sys.stderr)
elif sm not in KNOWN:
    print(f"[doctor] note: {sm} is not one of the two target GPUs (sm_89 / sm_121); "
          "proceeding, but this is an untested device for this pipeline.")

print("[doctor] PASS: CUDA available, matmul finite, capability checked.")
PY
