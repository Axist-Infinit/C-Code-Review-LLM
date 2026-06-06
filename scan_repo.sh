#!/usr/bin/env bash
set -euo pipefail
source ./offline_lockdown.sh
VENV="${VENV:-.venv}"
source "$VENV/bin/activate"
SRC="${SRC:-playground}"; OUTDIR="${OUT:-scan_out}"; MODEL="${MODEL:-./vuln-model}"
PROFILE_ARG=""
[[ -n "${CCR_PROFILE:-}" ]] && PROFILE_ARG="--profile $CCR_PROFILE"

python ./local_vuln_scanner.py "$SRC" -o "$OUTDIR" --model "$MODEL" $PROFILE_ARG
# Explain via local Ollama LLM (profile picks the model); auto-falls back to
# the heuristic regex explainer when Ollama is unavailable.
python ./llm_explain.py "$OUTDIR/classifier_findings.json" \
  --out "$OUTDIR/llm_findings.json" \
  --html "$OUTDIR/report.html" $PROFILE_ARG
