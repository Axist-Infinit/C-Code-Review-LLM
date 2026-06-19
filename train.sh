#!/usr/bin/env bash
# Train the GraphCodeBERT classifier with the PER-MACHINE training config.
#
# Batch size and mixed-precision flag come from the detected hardware profile
# (profiles.json via profiles.py): 4090 -> batch 64 + --fp16, Spark -> batch 128
# + --bf16, cpu -> batch 8 + full precision. Override either with the BATCH /
# CCR_PRECISION_FLAG env vars (or CCR_PROFILE to force a profile).
set -euo pipefail
VENV="${VENV:-.venv}"
source "$VENV/bin/activate"
OUT="${OUT:-vuln-model}"; EPOCHS="${EPOCHS:-3}"

# Resolve per-machine training config (batch + precision flag). Env overrides win.
eval "$(python profiles.py --training-env)"
BATCH="${BATCH:-$CCR_BATCH}"
PRECISION_FLAG="${CCR_PRECISION_FLAG:-}"
echo "[train] profile=$CCR_PROFILE batch=$BATCH precision='${PRECISION_FLAG:-fp32}'"

python model/train_vuln_model.py \
  --base ./models/graphcodebert-base \
  --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl \
  --out "$OUT" --epochs "$EPOCHS" --batch "$BATCH" --class_weight $PRECISION_FLAG
