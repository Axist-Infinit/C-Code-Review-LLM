#!/usr/bin/env bash
# Train the classifier (5 epochs) with the PER-MACHINE training config.
# Batch size + precision flag come from the detected profile (profiles.py);
# override with BATCH / CCR_PRECISION_FLAG / CCR_PROFILE env vars.
set -euo pipefail
VENV="${VENV:-.venv}"
source "$VENV/bin/activate"
OUT="${OUT:-vuln-model}"; EPOCHS="${EPOCHS:-5}"

eval "$(python profiles.py --training-env)"
BATCH="${BATCH:-$CCR_BATCH}"
PRECISION_FLAG="${CCR_PRECISION_FLAG:-}"
echo "[train] profile=$CCR_PROFILE batch=$BATCH precision='${PRECISION_FLAG:-fp32}'"

python model/train_vuln_model.py \
  --base ./models/graphcodebert-base \
  --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl \
  --out "$OUT" --epochs "$EPOCHS" --batch "$BATCH" --class_weight $PRECISION_FLAG
