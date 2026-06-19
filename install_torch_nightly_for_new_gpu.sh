#!/usr/bin/env bash
# Install a NIGHTLY torch for a GPU whose compute capability the stable wheels
# don't yet ship kernels for (e.g. a brand-new Ada/Blackwell card, or the DGX
# Spark's Grace-Blackwell sm_121).
#
# ARCH-AWARE (this used to hardcode cu124 x86 nightlies + torchvision/torchaudio,
# which is wrong for both targets):
#   * aarch64 (DGX Spark, sm_121) -> cu130 NIGHTLY, torch ONLY
#   * x86_64                       -> nightly channel matching the driver's CUDA
#                                     (cu130/cu128/...), torch ONLY
# torchvision/torchaudio are intentionally NOT installed: they are unused and
# torchvision >=0.25 breaks the `datasets` torch formatter.
set -euo pipefail
VENV="${VENV:-.venv}"
source "$VENV/bin/activate"

# shellcheck source=lib_torch_install.sh
source "$(dirname "$0")/lib_torch_install.sh"

# Remove any existing torch stack (incl. torchvision/torchaudio that older
# installs may have pulled in) before installing the nightly.
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

ARCH="$(uname -m)"
CUDA_VER="$(ccr_detect_cuda_ver)"
IDX="$(ccr_torch_channel "$ARCH" "$CUDA_VER" nightly)"
ccr_t_say "Host arch=$ARCH CUDA='${CUDA_VER:-unknown}' -> nightly channel: $IDX"
ccr_t_say "Installing torch ONLY (no torchvision/torchaudio)."

pip install --pre --no-cache-dir --index-url "$IDX" torch

python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
PY

ccr_t_say "Done. Validate kernels for your card with: ./env_doctor_cuda.sh"
