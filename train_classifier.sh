#!/usr/bin/env bash
set -euo pipefail
VENV="${VENV:-.venv}"
source "$VENV/bin/activate"
python model/train_vuln_model.py \
  --base ./models/graphcodebert-base \
  --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl \
  --out vuln-model --epochs 5 --batch 16 --class_weight
