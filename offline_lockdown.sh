#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
mkdir -p "$HF_HOME"
echo "[OK] Offline mode enforced. Cache: $HF_HOME"
