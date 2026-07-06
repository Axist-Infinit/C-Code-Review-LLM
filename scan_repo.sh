#!/usr/bin/env bash
set -euo pipefail
source ./offline_lockdown.sh
VENV="${VENV:-.venv}"
source "$VENV/bin/activate"
SRC="${SRC:-playground}"; OUTDIR="${OUT:-scan_out}"; MODEL="${MODEL:-./vuln-model}"
PROFILE_ARG=""
[[ -n "${CCR_PROFILE:-}" ]] && PROFILE_ARG="--profile $CCR_PROFILE"
# Triage default: 0.5 (recall-oriented; the LLM explainer filters FPs downstream).
# THRESHOLD=tuned uses the model's own max-F1 threshold from inference.json.
THRESHOLD="${THRESHOLD:-0.5}"
THR_ARG="--threshold $THRESHOLD"
[[ "$THRESHOLD" == "tuned" ]] && THR_ARG=""

python ./local_vuln_scanner.py "$SRC" -o "$OUTDIR" --model "$MODEL" $THR_ARG $PROFILE_ARG
# Explain via local Ollama LLM (profile picks the model); auto-falls back to
# the heuristic regex explainer when Ollama is unavailable.
python ./llm_explain.py "$OUTDIR/classifier_findings.json" \
  --out "$OUTDIR/llm_findings.json" \
  --html "$OUTDIR/report.html" $PROFILE_ARG

# --- Attack-surface / contract review (headers + whole modules) ---
# The body-classifier above cannot see declaration-only headers; this lane reads
# them as contracts (see surface_scan.sh). Guarded + --soft-fail so a missing
# Ollama never breaks the run; findings also render to SARIF for code scanning.
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"; [[ "$OLLAMA_URL" == http*://* ]] || OLLAMA_URL="http://$OLLAMA_URL"
if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  python ./surface_review.py "$SRC" \
    --json "$OUTDIR/surface_review.json" \
    --md   "$OUTDIR/surface_report.md" \
    --pipeline-out "$OUTDIR/surface_findings.json" \
    --ollama-url "$OLLAMA_URL" --soft-fail $PROFILE_ARG
  python ./to_sarif.py "$OUTDIR/surface_findings.json" -o "$OUTDIR/surface.sarif" || true
else
  echo "[skip] surface review: no Ollama at $OLLAMA_URL (headers not contract-reviewed)"
fi
