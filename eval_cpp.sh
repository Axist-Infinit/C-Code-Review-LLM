#!/usr/bin/env bash
# Evaluate the trained classifier on the curated C++ held-out set and write a
# per-category metrics report to benchmarks/cpp_eval_metrics.json.
#
# The classifier is trained on BigVul (overwhelmingly C); this measures how it
# actually scores real C++ methods, broken down by vulnerability class, so you
# can decide whether C++ needs its own training data. Requires a trained model
# in ./vuln-model and the ML stack installed (torch/transformers).
set -euo pipefail
VENV="${VENV:-.venv}"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
MODEL="${MODEL:-./vuln-model}"
TEST="${TEST:-benchmarks/cpp_eval.jsonl}"
OUT="${OUT:-benchmarks/cpp_eval_metrics.json}"
SCORES="${SCORES:-benchmarks/cpp_eval_scores.jsonl}"
PROFILE_ARG=""
[[ -n "${CCR_PROFILE:-}" ]] && PROFILE_ARG="--profile $CCR_PROFILE"

python ./evaluate_model.py --model "$MODEL" --test "$TEST" \
  --group-by category --save-scores "$SCORES" --out "$OUT" $PROFILE_ARG
echo "[OK] C++ eval metrics  -> $OUT"
echo "[OK] per-sample scores -> $SCORES"
