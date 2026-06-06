#!/usr/bin/env bash
set -Eeuo pipefail

PYBIN="${PYBIN:-python}"
OUT_DIR="models/graphcodebert-base"
mkdir -p "$OUT_DIR"

$PYBIN - <<'PY'
import os, sys
from transformers import AutoTokenizer, AutoModel, AutoConfig

out_dir = "models/graphcodebert-base"
os.makedirs(out_dir, exist_ok=True)

name = "microsoft/graphcodebert-base"
tok = AutoTokenizer.from_pretrained(name)
base = AutoModel.from_pretrained(name)

tok.save_pretrained(out_dir)
base.save_pretrained(out_dir, safe_serialization=True)

print("[OK] Saved GraphCodeBERT under", out_dir)
PY
