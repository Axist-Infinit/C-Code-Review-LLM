#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
SRC="${SRC:-playground}"; OUTDIR="${OUT:-scan_out}"; MODEL="${MODEL:-./vuln-model}"
PROFILE_ARG=""
[[ -n "${CCR_PROFILE:-}" ]] && PROFILE_ARG="--profile $CCR_PROFILE"

python ./local_vuln_scanner.py "$SRC" -o "$OUTDIR" --model "$MODEL" $PROFILE_ARG
python ./llm_explain.py "$OUTDIR/classifier_findings.json" \
  --out "$OUTDIR/llm_findings.json" \
  --html "$OUTDIR/report.html" $PROFILE_ARG
