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

# Arch is resolved inside lib_torch_install.sh (ccr_install_torch); just report it.
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
# Arch-aware wheel selection lives in lib_torch_install.sh. On aarch64 (DGX
# Spark, sm_121) this resolves to the cu130 NIGHTLY channel (no stable cu130
# aarch64 wheel is guaranteed) and FAILS SOFT so a missing wheel does not abort
# the whole setup under `set -e`.
# shellcheck source=lib_torch_install.sh
source "$(dirname "$0")/lib_torch_install.sh"
TORCH_OK=1
ccr_install_torch || TORCH_OK=0
if [[ "$TORCH_OK" != "1" ]]; then
  warn "torch install failed for this arch/CUDA. Setup CONTINUES (deps still install)."
  warn "Resolve torch via the per-machine runbook, then run ./env_doctor_cuda.sh."
fi

say "Installing python deps"
pip install --no-cache-dir -r requirements.txt

# sanity: GPU visible to torch? (best-effort; do not abort if torch is missing)
if [[ "$TORCH_OK" == "1" ]]; then
  python - <<'PY' || warn "torch import/sanity failed; run ./env_doctor_cuda.sh on real hardware."
import torch
ok = torch.cuda.is_available()
print(f"[setup] torch {torch.__version__} | cuda: {ok}" + (f" | {torch.cuda.get_device_name(0)}" if ok else ""))
PY
else
  warn "Skipping torch sanity check (torch not installed). On real hardware run:"
  warn "  ./env_doctor_cuda.sh   (asserts CUDA + a tiny matmul + compute capability)"
fi

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
