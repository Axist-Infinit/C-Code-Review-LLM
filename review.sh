#!/usr/bin/env bash
# review.sh — single interactive front door for the local C/C++ review model.
#
# Everything the model can do runs from one menu:
#   1) Attack-surface / contract review  (surface_review.py — Ollama only, no classifier)
#   2) Explain existing classifier findings (llm_explain.py)
#   3) Full pipeline scan  (classifier -> explainer -> surface; needs ./vuln-model)
# plus pickers for the Ollama model + hardware profile and a status/report view.
#
# The "model that runs" here is the local Ollama LLM (qwen2.5-coder:*). The
# surface lane needs only a running Ollama server; the full pipeline lane also
# needs a trained GraphCodeBERT classifier in ./vuln-model.
#
# No arguments — just:  ./review.sh
set -uo pipefail

# Run from the repo root so the relative script/model paths resolve.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

# ---- config / session state ------------------------------------------------
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
[[ "$OLLAMA_URL" == http*://* ]] || OLLAMA_URL="http://$OLLAMA_URL"
OUTDIR="${OUT:-scan_out}"
PROFILE="${CCR_PROFILE:-}"          # empty = auto-detect via profiles.py
MODEL_TAG=""                        # empty = use the profile's default model
SAMPLES="${SAMPLES:-1}"             # >1 = higher-recall self-consistency union

# ---- pretty printing -------------------------------------------------------
if [[ -t 1 ]]; then
  B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  GRN=$'\033[1;32m'; YEL=$'\033[1;33m'; RED=$'\033[1;31m'; CYN=$'\033[1;36m'
else
  B=""; DIM=""; R=""; GRN=""; YEL=""; RED=""; CYN=""
fi
say(){ printf '%s\n' "$*"; }
hr(){ printf '%s\n' "────────────────────────────────────────────────────────────"; }

# ---- helpers ---------------------------------------------------------------
ollama_up(){ curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

ollama_models(){
  curl -fsS "$OLLAMA_URL/api/tags" 2>/dev/null \
    | python3 -c "import sys,json;[print(m['name']) for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null
}

profile_default_model(){
  CCR_PROFILE="$PROFILE" python3 -c \
    "import profiles;print(profiles.select_profile()[1]['ollama_model'])" 2>/dev/null \
    || echo "qwen2.5-coder:7b"
}

detected_profile(){
  CCR_PROFILE="$PROFILE" python3 -c \
    "import profiles;print(profiles.select_profile()[0])" 2>/dev/null || echo "?"
}

have_classifier(){ [[ -f vuln-model/config.json ]]; }

# Effective model tag actually used (chosen override, else profile default).
effective_model(){ [[ -n "$MODEL_TAG" ]] && echo "$MODEL_TAG" || profile_default_model; }

open_report(){
  local f="$1"
  [[ -f "$f" ]] || { say "${YEL}(no report at $f yet)${R}"; return; }
  say "${GRN}Report:${R} $f"
  if command -v wslview >/dev/null 2>&1; then wslview "$f" >/dev/null 2>&1 &
  elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$(wslpath -w "$f" 2>/dev/null || echo "$f")" >/dev/null 2>&1 &
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$f" >/dev/null 2>&1 &
  else say "${DIM}(open it manually)${R}"; fi
}

pause(){ read -rp $'\n'"${DIM}[enter] to return to menu…${R} " _; }

# ---- actions ---------------------------------------------------------------
run_surface(){
  ollama_up || { say "${RED}No Ollama server at $OLLAMA_URL.${R} Start it:  ollama serve &"; pause; return; }
  # Default to whatever demo input actually exists (headers may have been removed).
  local def="playground"
  for c in "playground/dhcp-internal.h playground/dhcp-protocol.h" \
           "playground/dhcp-internal.h" "playground"; do
    # shellcheck disable=SC2086
    if ls $c >/dev/null 2>&1; then def="$c"; break; fi
  done
  say "Paths to review together ${DIM}(space-separated file/dir; default: $def)${R}:"
  read -rp "> " paths
  [[ -z "${paths// }" ]] && paths="$def"
  # Reuse the tested surface_scan.sh wiring (report + SARIF bridge) via env.
  say "\n${CYN}▶ surface review${R}  model=${B}$(effective_model)${R}  samples=$SAMPLES  out=$OUTDIR/surface"
  OUT="$OUTDIR/surface" \
  OLLAMA_HOST="$OLLAMA_URL" \
  CCR_PROFILE="$PROFILE" \
  ${MODEL_TAG:+MODELS="$MODEL_TAG"} \
  ${SAMPLES:+SAMPLES="$SAMPLES"} \
    ./surface_scan.sh $paths
  open_report "$OUTDIR/surface/surface_report.md"
  pause
}

run_explain(){
  ollama_up || say "${YEL}No Ollama — falling back to the heuristic explainer.${R}"
  local inp="$OUTDIR/classifier_findings.json"
  [[ -f "$inp" ]] || { say "${RED}No $inp — run a classifier scan first (option 3).${R}"; pause; return; }
  local backend=auto; ollama_up || backend=heuristic
  say "\n${CYN}▶ explain findings${R}  model=${B}$(effective_model)${R}  backend=$backend"
  python3 ./llm_explain.py "$inp" \
    --out "$OUTDIR/llm_findings.json" --html "$OUTDIR/report.html" \
    --backend "$backend" --ollama-url "$OLLAMA_URL" \
    ${MODEL_TAG:+--model "$MODEL_TAG"} ${PROFILE:+--profile "$PROFILE"}
  open_report "$OUTDIR/report.html"
  pause
}

run_full(){
  if ! have_classifier; then
    say "${RED}No trained classifier at ./vuln-model (config.json missing).${R}"
    say "This lane needs one. Either train:"
    say "   ${DIM}bash ./one_click_unlock_fetch_train_relock.sh${R}"
    say "or unpack a transferred model:"
    say "   ${DIM}bash ./unpack_model.sh trained_model.tar.zst .${R}"
    say "${DIM}(The surface-review lane, option 1, works without a classifier.)${R}"
    pause; return
  fi
  say "Source path to scan ${DIM}(file/dir; default: playground)${R}:"
  read -rp "> " src; [[ -z "${src// }" ]] && src="playground"
  say "\n${CYN}▶ full pipeline${R}  src=$src  model=${B}$(effective_model)${R}  out=$OUTDIR"
  SRC="$src" OUT="$OUTDIR" OLLAMA_HOST="$OLLAMA_URL" CCR_PROFILE="$PROFILE" \
    ./scan_repo.sh
  open_report "$OUTDIR/report.html"
  pause
}

pick_model(){
  say "Available Ollama models:"
  local i=1; local -a models=()
  while IFS= read -r m; do models+=("$m"); printf "  %d) %s\n" "$i" "$m"; ((i++)); done < <(ollama_models)
  [[ ${#models[@]} -eq 0 ]] && say "  ${YEL}(none — is Ollama running?)${R}"
  printf "  0) %s\n" "use profile default ($(profile_default_model))"
  read -rp "> pick # " n
  if [[ "$n" == "0" || -z "${n// }" ]]; then MODEL_TAG=""; say "→ using profile default";
  elif [[ "$n" =~ ^[0-9]+$ ]] && (( n>=1 && n<=${#models[@]} )); then MODEL_TAG="${models[n-1]}"; say "→ model = $MODEL_TAG";
  else say "${YEL}invalid choice${R}"; fi
  pause
}

pick_profile(){
  say "Hardware profile (sizes batch + picks the default model):"
  say "  1) auto-detect   2) 4090   3) spark   4) laptop   5) cpu"
  read -rp "> pick # " n
  case "$n" in
    1|"") PROFILE="";;
    2) PROFILE="4090";; 3) PROFILE="spark";; 4) PROFILE="laptop";; 5) PROFILE="cpu";;
    *) say "${YEL}invalid choice${R}"; pause; return;;
  esac
  say "→ profile = ${PROFILE:-auto ($(detected_profile))}"
  pause
}

toggle_samples(){
  say "Self-consistency samples per model (higher = better recall, slower). Current: $SAMPLES"
  read -rp "> new value [1-5] " n
  [[ "$n" =~ ^[1-5]$ ]] && SAMPLES="$n" && say "→ samples = $SAMPLES" || say "${YEL}unchanged${R}"
  pause
}

# ---- menu ------------------------------------------------------------------
status_line(){
  local up prof cls
  ollama_up && up="${GRN}up${R}" || up="${RED}down${R}"
  prof="${PROFILE:-auto ($(detected_profile))}"
  have_classifier && cls="${GRN}present${R}" || cls="${DIM}absent${R}"
  hr
  printf "  Ollama:   %b  @ %s\n" "$up" "$OLLAMA_URL"
  printf "  Model:    ${B}%s${R}\n" "$(effective_model)"
  printf "  Profile:  %b     Classifier: %b\n" "$prof" "$cls"
  printf "  Out dir:  %s     Samples: %s\n" "$OUTDIR" "$SAMPLES"
  hr
}

while true; do
  clear 2>/dev/null || true
  say "${B}C/C++ Code-Review Model — control menu${R}"
  status_line
  say "  ${B}1${R}) Attack-surface / contract review   ${DIM}(headers+modules, Ollama only)${R}"
  say "  ${B}2${R}) Explain existing classifier findings ${DIM}(llm_explain)${R}"
  say "  ${B}3${R}) Full pipeline scan                 ${DIM}(classifier → explain → surface)${R}"
  say ""
  say "  ${B}m${R}) Pick Ollama model     ${B}p${R}) Pick hardware profile"
  say "  ${B}s${R}) Recall samples        ${B}r${R}) Open last surface report"
  say "  ${B}q${R}) Quit"
  read -rp $'\n> ' choice
  case "$choice" in
    1) run_surface;;
    2) run_explain;;
    3) run_full;;
    m|M) pick_model;;
    p|P) pick_profile;;
    s|S) toggle_samples;;
    r|R) open_report "$OUTDIR/surface/surface_report.md"; pause;;
    q|Q|"") say "bye."; exit 0;;
    *) say "${YEL}unknown option: $choice${R}"; pause;;
  esac
done
