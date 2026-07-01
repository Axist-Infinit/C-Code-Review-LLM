#!/usr/bin/env bash
# Evaluate the trained classifier on the C++ held-out sets and write
# per-category metrics reports under benchmarks/.
#
# Two sets are scored:
#   * benchmarks/cpp_eval.jsonl       — short synthetic snippets (kept as a
#     regression probe; a BigVul-trained model is EXPECTED to score ~random
#     here, see benchmarks/cpp_eval_findings.md);
#   * benchmarks/cpp_eval_real.jsonl  — real PrimeVul C++ CVE functions,
#     leakage-filtered against BigVul (scripts/py/build_cpp_eval_real.py).
#     Scored only when the file exists, so pre-build checkouts still work.
#
# evaluate_model.py reads the tokenizer max_length from the model's
# inference.json automatically (512 when absent) — no flag needed here.
# Requires a trained model in ./vuln-model and the ML stack (torch/transformers).
set -euo pipefail
VENV="${VENV:-.venv}"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
MODEL="${MODEL:-./vuln-model}"
TEST="${TEST:-benchmarks/cpp_eval.jsonl}"
OUT="${OUT:-benchmarks/cpp_eval_metrics.json}"
SCORES="${SCORES:-benchmarks/cpp_eval_scores.jsonl}"
REAL_TEST="${REAL_TEST:-benchmarks/cpp_eval_real.jsonl}"
REAL_OUT="${REAL_OUT:-benchmarks/cpp_eval_real_metrics.json}"
REAL_SCORES="${REAL_SCORES:-benchmarks/cpp_eval_real_scores.jsonl}"
PROFILE_ARG=""
[[ -n "${CCR_PROFILE:-}" ]] && PROFILE_ARG="--profile $CCR_PROFILE"

python ./evaluate_model.py --model "$MODEL" --test "$TEST" \
  --group-by category --save-scores "$SCORES" --out "$OUT" $PROFILE_ARG
echo "[OK] C++ eval (synthetic) metrics  -> $OUT"
echo "[OK] per-sample scores             -> $SCORES"

if [[ -f "$REAL_TEST" ]]; then
  python ./evaluate_model.py --model "$MODEL" --test "$REAL_TEST" \
    --group-by category --save-scores "$REAL_SCORES" --out "$REAL_OUT" $PROFILE_ARG
  echo "[OK] C++ eval (real) metrics       -> $REAL_OUT"
  echo "[OK] per-sample scores             -> $REAL_SCORES"
else
  echo "[skip] $REAL_TEST not found — build it with scripts/py/build_cpp_eval_real.py"
fi
