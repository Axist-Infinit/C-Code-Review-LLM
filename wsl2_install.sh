#!/usr/bin/env bash
# DEPRECATED: use ./setup_machine.sh instead — it handles newer CUDA versions
# (12.5+/13.x), DGX Spark (aarch64), and skips torchvision (which breaks
# datasets >=4.4). Kept for reference only.
set -euo pipefail
VENV="${VENV:-.venv}"; PY="${PYTHON:-python3}"
echo "[INFO] $(uname -a)"
if command -v apt-get >/dev/null; then sudo apt-get update -y && sudo apt-get install -y git git-lfs python3 python3-venv python3-pip build-essential; fi
git lfs install || true
[[ -d "$VENV" ]] || $PY -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
USE_GPU="${USE_GPU:-1}"
INDEX_URL=""
if [[ "$USE_GPU" == "1" ]] && command -v nvidia-smi >/dev/null; then
  CUDA_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9][0-9]*\.[0-9][0-9]*" | awk '{print $3}' || true)
  case "$CUDA_VER" in
    12.4*) INDEX_URL="https://download.pytorch.org/whl/cu124" ;;
    12.3*) INDEX_URL="https://download.pytorch.org/whl/cu123" ;;
    12.1*|12.2*) INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
    11.*) INDEX_URL="https://download.pytorch.org/whl/cu118" ;;
  esac
fi
if [[ -n "$INDEX_URL" ]]; then pip install --no-cache-dir --index-url "$INDEX_URL" torch torchvision torchaudio; else pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio; fi
pip install --no-cache-dir -r requirements.txt
bash ./bootstrap_data.sh || true
bash ./fetch_graphcodebert.sh || true
