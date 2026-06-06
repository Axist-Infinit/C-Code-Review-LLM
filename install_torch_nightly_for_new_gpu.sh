#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
pip uninstall -y torch torchvision torchaudio nvidia-* triton || true
pip install --pre --no-cache-dir --index-url https://download.pytorch.org/whl/nightly/cu124 torch torchvision torchaudio
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
PY
