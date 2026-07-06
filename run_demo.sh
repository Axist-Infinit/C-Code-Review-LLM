#!/usr/bin/env bash
# Demo scan of playground/ with the trained classifier + explainer. Same
# hardening as scan_repo.sh: VENV override, offline lockdown, and a 0.5 triage
# threshold (THRESHOLD=tuned uses the model's own max-F1 threshold).
set -euo pipefail
VENV="${VENV:-.venv}"
source "$VENV/bin/activate"
source ./offline_lockdown.sh
SRC="${SRC:-playground}"; OUTDIR="${OUT:-scan_out}"; MODEL="${MODEL:-./vuln-model}"
PROFILE_ARG=""
[[ -n "${CCR_PROFILE:-}" ]] && PROFILE_ARG="--profile $CCR_PROFILE"
THRESHOLD="${THRESHOLD:-0.5}"
THR_ARG="--threshold $THRESHOLD"
[[ "$THRESHOLD" == "tuned" ]] && THR_ARG=""

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "[err] no trained model at $MODEL (config.json missing)." >&2
  echo "[err] Train one (bash ./one_click_unlock_fetch_train_relock.sh) or unpack" >&2
  echo "[err] a transferred one (bash ./unpack_model.sh trained_model.tar.zst .)." >&2
  exit 1
fi

python ./local_vuln_scanner.py "$SRC" -o "$OUTDIR" --model "$MODEL" $THR_ARG $PROFILE_ARG
python ./llm_explain.py "$OUTDIR/classifier_findings.json" \
  --out "$OUTDIR/llm_findings.json" \
  --html "$OUTDIR/report.html" $PROFILE_ARG

# --- Attack-surface / contract review (headers + whole modules) ---
# Reads declaration-only headers as contracts, which the body-classifier above
# cannot (see surface_scan.sh). Guarded + --soft-fail so a missing Ollama never
# breaks the demo.
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"; [[ "$OLLAMA_URL" == http*://* ]] || OLLAMA_URL="http://$OLLAMA_URL"
if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  python ./surface_review.py "$SRC" \
    --json "$OUTDIR/surface_review.json" \
    --md   "$OUTDIR/surface_report.md" \
    --pipeline-out "$OUTDIR/surface_findings.json" \
    --ollama-url "$OLLAMA_URL" --soft-fail $PROFILE_ARG
  python ./to_sarif.py "$OUTDIR/surface_findings.json" -o "$OUTDIR/surface.sarif" || true
else
  echo "[skip] surface review: no Ollama at $OLLAMA_URL"
fi
