#!/usr/bin/env bash
# End-to-end smoke test of the pipeline. Requires a trained model in ./vuln-model.
set -euo pipefail
VENV="${VENV:-.venv}"
source "$VENV/bin/activate"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
pass(){ printf "\033[1;32m[PASS]\033[0m %s\n" "$*"; }

python profiles.py >/dev/null                                      && pass "profile detection"
python local_vuln_scanner.py playground/vuln_demo.c -o "$T" >/dev/null 2>&1 \
                                                                   && pass "scanner (function granularity)"
python local_vuln_scanner.py playground/vuln_demo.c -o "$T" --granularity window >/dev/null 2>&1 \
                                                                   && pass "scanner (window granularity)"
python explain_findings.py "$T/classifier_findings.json" --out "$T/h.json" --html "$T/h.html" >/dev/null \
                                                                   && pass "heuristic explainer + HTML"
python llm_explain.py "$T/classifier_findings.json" --out "$T/l.json" --backend heuristic >/dev/null \
                                                                   && pass "llm_explain (heuristic backend)"
python to_sarif.py "$T/l.json" -o "$T/f.sarif" >/dev/null          && pass "SARIF conversion"
# Attack-surface / contract lane (headers): --soft-fail keeps this green whether
# or not Ollama is up; its pipeline-out must always feed the SARIF bridge.
python surface_review.py playground/dhcp-internal.h playground/dhcp-protocol.h \
    --pipeline-out "$T/surface.json" --md "$T/surface.md" --soft-fail --no-critic >/dev/null 2>&1 \
  && python to_sarif.py "$T/surface.json" -o "$T/surface.sarif" >/dev/null \
                                                                   && pass "surface review (soft-fail + SARIF bridge)"
python ensemble_scan.py playground/vuln_demo.c --ml "$T/l.json" -o "$T/e.json" >/dev/null \
                                                                   && pass "ensemble merge"
python evaluate_model.py --model ./vuln-model --test data/test.jsonl >/dev/null 2>&1 \
                                                                   && pass "eval harness"
echo "All smoke tests passed."
