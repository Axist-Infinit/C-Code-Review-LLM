#!/usr/bin/env bash
set -Eeuo pipefail

PYBIN="${PYBIN:-python}"
OUT_DIR="models/graphcodebert-base"
mkdir -p "$OUT_DIR"
# HF_REVISION pins a specific commit sha / tag for supply-chain integrity.
# Default empty = current HEAD of microsoft/graphcodebert-base (documented risk:
# pin a sha for reproducible/audited builds).
HF_REVISION="${HF_REVISION:-}"

$PYBIN - "$HF_REVISION" <<'PY'
import os, sys
from transformers import AutoTokenizer, AutoModel

out_dir = "models/graphcodebert-base"
os.makedirs(out_dir, exist_ok=True)

revision = (sys.argv[1] if len(sys.argv) > 1 else "") or None
name = "microsoft/graphcodebert-base"
tok = AutoTokenizer.from_pretrained(name, revision=revision)
base = AutoModel.from_pretrained(name, revision=revision)

tok.save_pretrained(out_dir)
base.save_pretrained(out_dir, safe_serialization=True)

print("[OK] Saved GraphCodeBERT under", out_dir, "(revision=%s)" % (revision or "<default>"))
PY
