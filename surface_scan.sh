#!/usr/bin/env bash
# Attack-surface / contract security review for C/C++ — the lane that reads
# HEADERS and whole modules as CONTRACTS (the body-classifier lane in
# scan_repo.sh is blind to declaration-only files). It needs only a local
# Ollama server — no torch, no trained classifier — so it is the lane to run on
# any new file you want a senior-reviewer-style threat model for.
#
#   ./surface_scan.sh playground/dhcp-internal.h playground/dhcp-protocol.h
#   OUT=out ./surface_scan.sh src/                 # a whole directory, reviewed together
#   SAMPLES=2 MODELS=qwen2.5-coder:14b,qwen2.5-coder:32b ./surface_scan.sh hdr.h   # higher-recall union
#
# Outputs (under $OUT, default scan_out/surface): surface_report.md (the rich
# report), surface_review.json (raw), surface_findings.json (pipeline schema)
# and surface.sarif (for CI / IDE / GitHub code scanning).
set -euo pipefail
source ./offline_lockdown.sh 2>/dev/null || true

OUTDIR="${OUT:-scan_out/surface}"
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
# Accept Ollama's native host:port form as well as a full URL.
[[ "$OLLAMA_URL" == http*://* ]] || OLLAMA_URL="http://$OLLAMA_URL"

PROFILE_ARG=""
[[ -n "${CCR_PROFILE:-}" ]] && PROFILE_ARG="--profile $CCR_PROFILE"

# Optional higher-recall union finisher (see README): SAMPLES=N and/or MODELS=a,b
EXTRA=()
[[ -n "${MODELS:-}" ]]  && EXTRA+=(--models "$MODELS")
[[ -n "${SAMPLES:-}" ]] && EXTRA+=(--samples "$SAMPLES")

# Paths to review (default: the playground demo headers).
if [[ $# -gt 0 ]]; then PATHS=("$@"); else PATHS=(playground/dhcp-internal.h playground/dhcp-protocol.h); fi

mkdir -p "$OUTDIR"

if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  MODEL="$(python3 -c 'import profiles; print(profiles.select_profile()[1]["ollama_model"])' 2>/dev/null || echo qwen2.5-coder:7b)"
  echo "[err] No Ollama server reachable at $OLLAMA_URL." >&2
  echo "[err] This lane needs a local LLM. Start it and pull the model, then re-run:" >&2
  echo "        ollama serve &        # or launch the Ollama desktop app" >&2
  echo "        ollama pull $MODEL" >&2
  exit 1
fi

echo "[surface] reviewing: ${PATHS[*]}"
python3 ./surface_review.py "${PATHS[@]}" \
  --json         "$OUTDIR/surface_review.json" \
  --md           "$OUTDIR/surface_report.md" \
  --pipeline-out "$OUTDIR/surface_findings.json" \
  --ollama-url   "$OLLAMA_URL" \
  $PROFILE_ARG ${EXTRA[@]+"${EXTRA[@]}"}

# Bridge the findings into the standard SARIF output (best-effort).
python3 ./to_sarif.py "$OUTDIR/surface_findings.json" -o "$OUTDIR/surface.sarif" || true

echo
echo "[OK] Markdown report : $OUTDIR/surface_report.md"
echo "[OK] Raw review JSON : $OUTDIR/surface_review.json"
echo "[OK] Pipeline JSON   : $OUTDIR/surface_findings.json"
echo "[OK] SARIF           : $OUTDIR/surface.sarif"
