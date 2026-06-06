#!/usr/bin/env bash
set -euo pipefail
VENV="${VENV:-.venv}"
detect_pm(){ command -v apt-get&&echo apt|| command -v dnf&&echo dnf|| command -v yum&&echo yum|| command -v zypper&&echo zypper|| command -v brew&&echo brew|| command -v pacman|| echo unknown; }
PM=$(detect_pm); echo "[INFO] Detected PM: $PM"
case "$PM" in
  apt) sudo apt-get update -y && sudo apt-get install -y git git-lfs python3 python3-venv python3-pip;;
  dnf) sudo dnf install -y git git-lfs python3 python3-virtualenv python3-pip || true;;
  yum) sudo yum install -y git git-lfs python3 python3-pip || true;;
  zypper) sudo zypper install -y git git-lfs python3 python3-pip python3-venv || true;;
  brew) brew update || true; brew install git git-lfs python@3 || true;;
  pacman) sudo pacman -Sy --noconfirm git git-lfs python python-pip;;
  *) echo "[WARN] Unknown PM; ensure git, git-lfs, python3, venv, pip are installed.";;
esac
git lfs install || true
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6"
else
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"
fi
pip install --upgrade "transformers<5" "datasets>=2.14" tree_sitter tree-sitter-languages huggingface_hub scikit-learn peft accelerate packaging requests
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch:", torch.__version__)
PY
