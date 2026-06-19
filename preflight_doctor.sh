#!/usr/bin/env bash
# Thin wrapper for preflight_doctor.py — verifies model dir + tuned threshold +
# resolved profile + Ollama reachability BEFORE a long scan/train run.
# Exit 0 = ready (warnings allowed); non-zero = a required check failed.
#
#   ./preflight_doctor.sh                      # default ./vuln-model
#   MODEL=./vuln-model-bigvul ./preflight_doctor.sh
#   ./preflight_doctor.sh --require-ollama     # also fail if Ollama is down
set -euo pipefail
VENV="${VENV:-.venv}"
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
exec python preflight_doctor.py "$@"
