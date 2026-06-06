#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
OUT="${OUT:-vuln-model}"; EPOCHS="${EPOCHS:-3}"; BATCH="${BATCH:-16}"
python model/train_vuln_model.py --base ./models/graphcodebert-base --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl --out "$OUT" --epochs "$EPOCHS" --batch "$BATCH" --class_weight
