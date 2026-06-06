#!/usr/bin/env bash
set +u
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true
unset HF_DATASETS_OFFLINE || true
set -u
echo "[OK] Online mode enabled."
