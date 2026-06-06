#!/usr/bin/env bash
# One-command machine setup for the C code review pipeline.
# Works on both target machines:
#   - RTX 4090 workstation (WSL2 / x86_64, CUDA 12.x-13.x)
#   - DGX Spark (aarch64 Grace Blackwell, CUDA 13.x)
#
# What it does: system deps -> venv -> torch (correct wheel for this machine)
# -> python deps -> GraphCodeBERT base -> Ollama check + explainer model pull.
# It does NOT train; see SETUP.md for the training / model-transfer options.
set -euo pipefail
VENV="${VENV:-.venv}"; PY="${PYTHON:-python3}"; USE_GPU="${USE_GPU:-1}"
PULL_OLLAMA="${PULL_OLLAMA:-1}"
say(){ printf "\n\033[1;36m[setup]\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }

ARCH="$(uname -m)"
say "Host: $(uname -srm)"

# --- system deps ------------------------------------------------------------
SUDO="$(command -v sudo || true)"
if command -v apt-get >/dev/null; then
  $SUDO apt-get update -y && $SUDO apt-get install -y \
    git python3 python3-venv python3-pip build-essential zstd
fi

# --- venv -------------------------------------------------------------------
[[ -d "$VENV" ]] || $PY -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- torch (torch only; torchvision >=0.25 breaks datasets' torch formatter) -
INDEX_URL="https://download.pytorch.org/whl/cpu"
if [[ "$USE_GPU" == "1" ]] && command -v nvidia-smi >/dev/null; then
  CUDA_VER=$(nvidia-smi | grep -oE "CUDA[ A-Za-z]* Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1 || true)
  case "$CUDA_VER" in
    13.*)              INDEX_URL="https://download.pytorch.org/whl/cu130" ;;
    12.8*|12.9*)       INDEX_URL="https://download.pytorch.org/whl/cu128" ;;
    12.5*|12.6*|12.7*) INDEX_URL="https://download.pytorch.org/whl/cu126" ;;
    12.*)              INDEX_URL="https://download.pytorch.org/whl/cu124" ;;
    11.*)              INDEX_URL="https://download.pytorch.org/whl/cu118" ;;
    *) [[ -n "$CUDA_VER" ]] && INDEX_URL="https://download.pytorch.org/whl/cu128" ;;
  esac
  # DGX Spark / Grace (aarch64 Blackwell sm_121): needs cu130+ aarch64 wheels
  [[ "$ARCH" == "aarch64" ]] && INDEX_URL="https://download.pytorch.org/whl/cu130"
fi
say "Installing torch from $INDEX_URL"
pip install --no-cache-dir --index-url "$INDEX_URL" torch

say "Installing python deps"
pip install --no-cache-dir -r requirements.txt

# sanity: GPU visible to torch?
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"[setup] torch {torch.__version__} | cuda: {ok}" + (f" | {torch.cuda.get_device_name(0)}" if ok else ""))
PY

# --- GraphCodeBERT base (needed for training; skipped if already present) ---
if [[ ! -f models/graphcodebert-base/config.json ]]; then
  say "Fetching GraphCodeBERT base"
  bash ./fetch_graphcodebert.sh || warn "GraphCodeBERT fetch failed (offline?); needed only for training"
fi

# --- profile + Ollama -------------------------------------------------------
say "Hardware profile:"
python profiles.py

if command -v ollama >/dev/null; then
  MODEL=$(python - <<'PY'
from profiles import select_profile
print(select_profile()[1]["ollama_model"])
PY
)
  if [[ "$PULL_OLLAMA" == "1" ]]; then
    say "Pulling explainer model: $MODEL (skip with PULL_OLLAMA=0)"
    ollama pull "$MODEL" || warn "ollama pull failed; explainer will fall back to heuristics"
  else
    say "Skipping ollama pull (PULL_OLLAMA=0). Explainer model for this machine: $MODEL"
  fi
else
  warn "Ollama not installed — LLM explanations will fall back to regex heuristics."
  warn "Install: https://ollama.com/download (native builds for x86_64 and DGX Spark)"
fi

say "Setup complete. Next steps:"
echo "  - Get a trained classifier: ./unpack_model.sh vuln-model-bigvul.tar.zst   (copied from another machine)"
echo "    or train here:            ./one_click_unlock_fetch_train_relock.sh"
echo "  - Scan:                     SRC=path/to/code ./scan_repo.sh"
echo "  - Verify pipeline:          ./smoke_test.sh"
