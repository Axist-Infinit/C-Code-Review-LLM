#!/usr/bin/env bash
# ccr.sh — THE single interactive front door for this project.
#
#     ./ccr.sh                 # no arguments, ever
#
# One menu covers the whole lifecycle:
#   REVIEW   drop C/C++ files in review_inbox/ -> report.html + Markdown + SARIF
#   SETUP    python env, torch, tree-sitter, Ollama server + explainer models
#   TRAIN    GraphCodeBERT classifier (bootstrap smoke run or the real BigVul run)
#   TUNE     surface-review LoRA distillation + all inference settings
#
# Design rules this script follows (each one is a bug that was actually hit here):
#
#  * NO `set -e`. A failing action must drop you back at the menu, never kill it.
#  * Sub-tools are invoked as direct python entry points, never through
#    env-prefixed wrapper scripts. In bash, `FOO=1 ${X:+BAR=2} ./s.sh` does NOT
#    pass BAR: assignment prefixes are recognised before expansion, so the
#    expanded word becomes the COMMAND NAME ("BAR=2: command not found").
#    That is exactly how the old review.sh silently never ran a surface scan.
#  * Every action echoes the equivalent plain command before running it, so
#    anything the menu does can be reproduced by hand outside the menu.
#  * Preconditions are checked and explained BEFORE a long job starts, and the
#    menu is honest about what a lane can and cannot do on this machine.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

# ---------------------------------------------------------------------------
# settings (persisted in .ccr/config; env vars win on first run)
# ---------------------------------------------------------------------------
STATE_DIR=".ccr"
CONF="$STATE_DIR/config"

VENV="${VENV:-.venv}"
INBOX="${CCR_INBOX:-review_inbox}"
OUTDIR="${OUT:-scan_out}"
MODEL_DIR="${MODEL:-./vuln-model}"
PROFILE="${CCR_PROFILE:-}"      # empty = auto-detect via profiles.py
MODEL_TAG=""                    # empty = the profile's ollama_model
LANE="auto"                     # auto | fast | deep | full
SAMPLES=1                       # surface-review self-consistency passes
CRITIC=1                        # surface-review 2nd (completeness) pass
NUM_CTX=""                      # empty = the profile's ollama_num_ctx
TOPK=""                         # empty = the profile's explainer_top_k
THRESHOLD="0.5"                 # classifier triage threshold ("tuned" = model's own)
MIN_SEV=""                      # empty = report every severity

CONF_KEYS=(VENV INBOX OUTDIR MODEL_DIR PROFILE MODEL_TAG LANE SAMPLES CRITIC
           NUM_CTX TOPK THRESHOLD MIN_SEV)

load_conf(){
  [[ -f "$CONF" ]] || return 0
  local line k v
  while IFS= read -r line; do
    [[ "$line" == *=* ]] || continue
    k="${line%%=*}"; v="${line#*=}"
    local known
    for known in "${CONF_KEYS[@]}"; do
      [[ "$k" == "$known" ]] && printf -v "$k" '%s' "$v" && break
    done
  done < "$CONF"
}

save_conf(){
  mkdir -p "$STATE_DIR" || return 1
  local k
  : > "$CONF"
  for k in "${CONF_KEYS[@]}"; do printf '%s=%s\n' "$k" "${!k}" >> "$CONF"; done
}

# ---------------------------------------------------------------------------
# pretty printing
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  GRN=$'\033[1;32m'; YEL=$'\033[1;33m'; RED=$'\033[1;31m'; CYN=$'\033[1;36m'
else
  B=""; DIM=""; R=""; GRN=""; YEL=""; RED=""; CYN=""
fi
say(){ printf '%s\n' "$*"; }
hr(){ printf '%s\n' "──────────────────────────────────────────────────────────────"; }
head2(){ printf '\n%s%s%s\n' "$B" "$*" "$R"; }
ok(){   printf '%s[ok]%s %s\n'   "$GRN" "$R" "$*"; }
warn(){ printf '%s[warn]%s %s\n' "$YEL" "$R" "$*"; }
err(){  printf '%s[err]%s %s\n'  "$RED" "$R" "$*"; }
pause(){ local _x; read -rp $'\n'"${DIM}[enter] to return…${R} " _x || true; }

# Echo a command exactly as it will run, then run it. Return its status.
run(){
  local a q=""
  for a in "$@"; do q+="$(printf '%q' "$a") "; done
  printf '%s$ %s%s\n' "$DIM" "$q" "$R"
  "$@"
}

# NOTE: readline (`read -e`) miscounts cursor columns when a prompt contains raw
# colour escapes, so ask() prompts are deliberately plain text.
ask(){  # ask VAR "prompt" "default"
  local __var="$1" __prompt="$2" __def="${3:-}" __ans=""
  if [[ -n "$__def" ]]; then
    read -rep "$__prompt [$__def]: " __ans || true
  else
    read -rep "$__prompt: " __ans || true
  fi
  [[ -z "${__ans// }" ]] && __ans="$__def"
  printf -v "$__var" '%s' "$__ans"
}

confirm(){  # confirm "question"  -> 0 on yes
  local a=""
  read -rp "$1 ${DIM}[y/N]${R} " a || true
  [[ "$a" == y || "$a" == Y || "$a" == yes ]]
}

# ---------------------------------------------------------------------------
# interpreter + probes
# ---------------------------------------------------------------------------
PY="python3"
resolve_py(){
  if [[ -x "$VENV/bin/python" ]]; then PY="$VENV/bin/python"
  elif command -v python3 >/dev/null 2>&1; then PY="python3"
  else PY="python"; fi
}

# Probe facts (defaults used when the probe itself fails).
P_PY="?"; P_PY311=0; P_TORCH=0; P_TRANSFORMERS=0; P_PEFT=0; P_TS=0
P_PROFILE="?"; P_OMODEL="qwen2.5-coder:7b"; P_OCTX="8192"
P_MODEL="absent"; P_THR=""; P_TRAIN_ROWS=0; P_SFT_ROWS=0; P_INBOX_N=0
P_GPU=""; P_VRAM=""; P_RAM=""; P_REC_TAG=""; P_REC_MIB=0; P_REC_FITS=0; P_REC_WHY=""

probe(){
  resolve_py
  local out line k v
  out="$(CCR_PROFILE="$PROFILE" CCR_MODEL_DIR="$MODEL_DIR" CCR_INBOX="$INBOX" \
        "$PY" - <<'PY' 2>/dev/null
import glob, importlib.util as iu, json, os, sys

def has(mod):
    try:
        return iu.find_spec(mod) is not None
    except Exception:
        return False

def emit(k, v):
    print("%s=%s" % (k, v))

emit("P_PY", "%d.%d.%d" % sys.version_info[:3])
emit("P_PY311", 1 if sys.version_info[:2] >= (3, 11) else 0)
emit("P_TORCH", 1 if has("torch") else 0)
emit("P_TRANSFORMERS", 1 if has("transformers") else 0)
emit("P_PEFT", 1 if has("peft") else 0)
emit("P_TS", 1 if (has("tree_sitter") and has("tree_sitter_c")) else 0)

prof, omodel, octx = "?", "qwen2.5-coder:7b", "8192"
rec_tag, rec_mib, rec_fits, rec_why = "", "0", "0", ""
gpu, vram, ram = "", "", ""
try:
    sys.path.insert(0, os.getcwd())
    import profiles
    name, p = profiles.select_profile()
    prof = name
    omodel = p.get("ollama_model", omodel)
    octx = str(p.get("ollama_num_ctx", octx))
    # What this machine can ACTUALLY hold, from measured VRAM (not the GPU name).
    hw = profiles.detect_hardware()
    gpu = hw.get("gpu") or ""
    vram = "" if hw.get("vram_mib") is None else str(hw["vram_mib"])
    ram = "" if hw.get("ram_mib") is None else str(hw["ram_mib"])
    installed = [t for t in (os.environ.get("CCR_INSTALLED_TAGS") or "").split(",") if t]
    rec = profiles.recommend_model(vram_mib=hw.get("vram_mib"), num_ctx=int(octx),
                                   installed=installed or None)
    rec_tag = rec["tag"]
    rec_mib = str(rec["footprint_mib"])
    rec_fits = "1" if rec["fits"] else "0"
    rec_why = rec["reason"]
except Exception:
    pass
emit("P_PROFILE", prof)
emit("P_OMODEL", omodel)
emit("P_OCTX", octx)
emit("P_GPU", gpu)
emit("P_VRAM", vram)
emit("P_RAM", ram)
emit("P_REC_TAG", rec_tag)
emit("P_REC_MIB", rec_mib)
emit("P_REC_FITS", rec_fits)
emit("P_REC_WHY", rec_why)

# Classifier state. A dir whose trainer_state says max_steps <= 20 is the
# bootstrap smoke artifact, not a usable classifier (see README/SETUP).
mdir = os.environ.get("CCR_MODEL_DIR") or "vuln-model"
kind, thr = "absent", ""
if os.path.isfile(os.path.join(mdir, "config.json")):
    kind = "trained"
    states = glob.glob(os.path.join(mdir, "checkpoint-*", "trainer_state.json"))
    states.append(os.path.join(mdir, "trainer_state.json"))
    for s in states:
        try:
            with open(s, encoding="utf-8") as f:
                if int(json.load(f).get("max_steps") or 0) <= 20:
                    kind = "bootstrap"
        except Exception:
            pass
    try:
        with open(os.path.join(mdir, "inference.json"), encoding="utf-8") as f:
            thr = str(json.load(f).get("threshold", ""))
    except Exception:
        pass
emit("P_MODEL", kind)
emit("P_THR", thr)

def rows(path):
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0

emit("P_TRAIN_ROWS", rows("data/train.jsonl"))
emit("P_SFT_ROWS", rows("data/surface_sft_train.jsonl"))

inbox = os.environ.get("CCR_INBOX") or "review_inbox"
n = 0
if os.path.isdir(inbox):
    try:
        from local_vuln_scanner import list_sources
        n = len(list_sources(inbox))
    except Exception:
        n = 0
emit("P_INBOX_N", n)
PY
)"
  while IFS= read -r line; do
    [[ "$line" == P_*=* ]] || continue
    k="${line%%=*}"; v="${line#*=}"
    printf -v "$k" '%s' "$v"
  done <<< "$out"
}

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
# $OLLAMA_HOST may legitimately be a bare host:port (Ollama's native form),
# which urllib rejects — normalise once, keep both forms.
ollama_url(){
  local u="${OLLAMA_HOST:-http://127.0.0.1:11434}"
  [[ "$u" == http*://* ]] || u="http://$u"
  printf '%s' "$u"
}
OLLAMA_URL="$(ollama_url)"
OLLAMA_HOSTPORT="${OLLAMA_URL#*://}"

ollama_bin(){
  if command -v ollama >/dev/null 2>&1; then command -v ollama
  elif [[ -x "$HOME/.local/bin/ollama" ]]; then printf '%s' "$HOME/.local/bin/ollama"
  else return 1; fi
}
ollama_up(){ curl -fsS --max-time 3 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

ollama_models(){
  curl -fsS --max-time 5 "$OLLAMA_URL/api/tags" 2>/dev/null | "$PY" -c \
    'import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
for m in d.get("models",[]): print(m.get("name",""))' 2>/dev/null
}

# Effective explainer/reviewer tag: explicit override, else the profile default.
effective_model(){ [[ -n "$MODEL_TAG" ]] && printf '%s' "$MODEL_TAG" || printf '%s' "$P_OMODEL"; }

ollama_has_model(){  # ollama_has_model TAG
  local want="$1" m
  while IFS= read -r m; do
    [[ -z "$m" ]] && continue
    [[ "$m" == "$want" || "${m%%:*}" == "${want%%:*}" ]] && return 0
  done < <(ollama_models)
  return 1
}

start_ollama(){
  local bin
  if ! bin="$(ollama_bin)"; then
    err "ollama binary not found (looked on \$PATH and in ~/.local/bin)."
    say "Install it from the Ollama menu, or: https://ollama.com/download"
    return 1
  fi
  if ollama_up; then ok "Ollama already running at $OLLAMA_URL"; return 0; fi
  mkdir -p "$STATE_DIR"
  say "${DIM}\$ OLLAMA_HOST=$OLLAMA_HOSTPORT $bin serve >> $STATE_DIR/ollama.log 2>&1 &${R}"
  ( OLLAMA_HOST="$OLLAMA_HOSTPORT" nohup "$bin" serve >> "$STATE_DIR/ollama.log" 2>&1 & )
  local i
  for i in $(seq 1 25); do
    ollama_up && { ok "Ollama is up at $OLLAMA_URL"; return 0; }
    sleep 1
  done
  err "Ollama did not come up in 25s — last log lines:"
  tail -n 15 "$STATE_DIR/ollama.log" 2>/dev/null
  return 1
}

pull_model(){  # pull_model TAG
  local bin tag="$1"
  bin="$(ollama_bin)" || { err "ollama binary not found"; return 1; }
  ollama_up || { warn "server is down — starting it first"; start_ollama || return 1; }
  OLLAMA_HOST="$OLLAMA_HOSTPORT" run "$bin" pull "$tag"
}

# Make sure the tag we are about to use actually exists on the server.
ensure_model_pulled(){
  local tag; tag="$(effective_model)"
  ollama_up || return 1
  ollama_has_model "$tag" && return 0
  warn "model '$tag' is not pulled on this Ollama server."
  if confirm "Pull it now (a few GB)?"; then pull_model "$tag"; else return 1; fi
}

# ---------------------------------------------------------------------------
# inbox + reports
# ---------------------------------------------------------------------------
ensure_inbox(){
  [[ -d "$INBOX" ]] && return 0
  mkdir -p "$INBOX" || return 1
  cat > "$INBOX/README.md" <<EOF
# review_inbox

Drop the C / C++ files you want reviewed in this directory, then run
\`./ccr.sh\` and pick **1) Review the inbox**.

Recognised extensions: .c .cpp .cc .cxx .c++ .h .hpp .hh .hxx .h++ .tcc .ipp .inl
Sub-directories are walked recursively. Anything else is ignored.

Reports are written to \`scan_out/\` (report.html, surface/surface_report.md,
*.sarif). Nothing here is sent anywhere — every lane runs locally.
EOF
  : > "$INBOX/.gitkeep"
  ok "created $INBOX/ (see $INBOX/README.md)"
}

# How many files the scanners will actually pick up under these paths. Uses the
# scanner's own list_sources() so the extension list can never drift from it.
count_sources(){  # count_sources PATH...
  "$PY" - "$@" <<'PY' 2>/dev/null || echo 0
import os, sys
sys.path.insert(0, os.getcwd())
try:
    from local_vuln_scanner import list_sources
except Exception:
    print(0); raise SystemExit(0)
seen = set()
for p in sys.argv[1:]:
    if os.path.exists(p):
        seen.update(list_sources(p))
print(len(seen))
PY
}

inbox_hint(){
  say "Copy the files you want reviewed into: ${B}$PWD/$INBOX${R}"
  if command -v wslpath >/dev/null 2>&1; then
    local w; w="$(wslpath -w "$PWD/$INBOX" 2>/dev/null)"
    [[ -n "$w" ]] && say "From Windows that path is:            ${B}$w${R}"
  fi
}

open_report(){
  local f="$1"
  [[ -f "$f" ]] || { warn "no report at $f yet"; return 1; }
  say "${GRN}report:${R} $f"
  if command -v wslview >/dev/null 2>&1; then wslview "$f" >/dev/null 2>&1 &
  elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$(wslpath -w "$f" 2>/dev/null || printf '%s' "$f")" >/dev/null 2>&1 &
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$f" >/dev/null 2>&1 &
  else say "${DIM}(open it manually)${R}"; fi
  return 0
}

# Print a severity roll-up + the top findings. Several files may be passed as
# FALLBACKS (richest first): the first one that actually carries findings wins,
# so llm_findings.json and the classifier_findings.json it was built from are
# never counted twice.
summarize(){  # summarize LABEL FILE [FALLBACK...]
  local label="$1"; shift
  [[ $# -gt 0 ]] || return 0
  "$PY" - "$label" "$@" <<'PY'
import json, os, sys

label, paths = sys.argv[1], sys.argv[2:]
items = []
for p in paths:
    if not os.path.isfile(p):
        continue
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        continue
    got = d if isinstance(d, list) else (d.get("explanations") or d.get("findings") or [])
    if got:
        items = got
        break

if not items:
    print("  %s: no findings" % label)
    sys.exit(0)

order = ["critical", "high", "medium", "low", "info"]
counts = {}
for e in items:
    counts[str(e.get("severity", "") or "?").lower()] = \
        counts.get(str(e.get("severity", "") or "?").lower(), 0) + 1
bits = ["%s %d" % (s, counts[s]) for s in order if s in counts]
bits += ["%s %d" % (s, n) for s, n in sorted(counts.items()) if s not in order]
print("  %s: %d finding(s)  [%s]" % (label, len(items), ", ".join(bits) or "-"))

rank = {s: i for i, s in enumerate(order)}
items.sort(key=lambda e: rank.get(str(e.get("severity", "") or "").lower(), 9))
for e in items[:8]:
    where = os.path.basename(str(e.get("file", "") or "?"))
    line = e.get("line") or e.get("start_line") or "?"
    sev = str(e.get("severity", "") or "?").lower()
    cwe = str(e.get("cwe", "") or "-")
    title = str(e.get("issue") or e.get("title") or e.get("vulnerability")
                or e.get("explanation") or "").strip().replace("\n", " ")
    print("   - %-22s %-9s %-10s %s" % ("%s:%s" % (where, line), sev, cwe, title[:72]))
if len(items) > 8:
    print("   … %d more (full detail in the report)" % (len(items) - 8))
PY
}

# Merge N classifier-schema JSONs into one (used when several paths are scanned).
merge_findings(){  # merge_findings OUT IN...
  "$PY" - "$@" <<'PY'
import json, os, sys
out, ins = sys.argv[1], sys.argv[2:]
meta, items = {}, []
for p in ins:
    if not os.path.isfile(p):
        continue
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        continue
    if not meta:
        meta = {k: v for k, v in d.items() if k not in ("findings", "explanations")}
    items += d.get("findings") or []
meta["findings"] = items
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
PY
}

# ---------------------------------------------------------------------------
# review lanes
# ---------------------------------------------------------------------------
# fast = model-free pattern scan + explainer (LLM when Ollama is up, else regex)
# deep = attack-surface / contract review of whole files incl. headers (Ollama)
# full = GraphCodeBERT classifier triage + explainer + deep  (torch + a model)
# auto = fast + deep-when-Ollama-is-up   <- the default
lane_label(){
  case "$1" in
    auto) printf 'auto (model-free scan + LLM contract review when Ollama is up)';;
    fast) printf 'fast (model-free scan + explainer)';;
    deep) printf 'deep (attack-surface / contract review only)';;
    full) printf 'full (classifier + explainer + contract review)';;
    *)    printf '%s' "$1";;
  esac
}

profile_args(){  # echo the --profile flag pair, or nothing
  [[ -n "$PROFILE" ]] && printf '%s\n%s\n' "--profile" "$PROFILE"
}

# ---- fast / full: per-path scan, merged, then explained ---------------------
scan_paths(){  # scan_paths SCANNER PATH...   (SCANNER = heuristic|classifier)
  local scanner="$1"; shift
  local -a parts=() prof=()
  local i=0 p d rc=0
  [[ -n "$OUTDIR" ]] || { err "output directory is unset"; return 1; }
  mapfile -t prof < <(profile_args)
  rm -rf "${OUTDIR:?}/.parts"
  for p in "$@"; do
    i=$((i + 1))
    d="$OUTDIR/.parts/$i"
    mkdir -p "$d" || return 1
    if [[ "$scanner" == heuristic ]]; then
      local -a extra=()
      [[ -n "$MIN_SEV" ]] && extra+=(--min-severity "$MIN_SEV")
      run "$PY" ./heuristic_scan.py "$p" -o "$d" ${extra[@]+"${extra[@]}"} || rc=1
    else
      local -a thr=()
      [[ "$THRESHOLD" == "tuned" ]] || thr=(--threshold "$THRESHOLD")
      run "$PY" ./local_vuln_scanner.py "$p" -o "$d" --model "$MODEL_DIR" \
          ${thr[@]+"${thr[@]}"} ${prof[@]+"${prof[@]}"} || rc=1
    fi
    [[ -f "$d/classifier_findings.json" ]] && parts+=("$d/classifier_findings.json")
  done
  [[ ${#parts[@]} -gt 0 ]] || { warn "no findings file produced"; return 1; }
  merge_findings "$OUTDIR/classifier_findings.json" "${parts[@]}"
  rm -rf "$OUTDIR/.parts"
  return $rc
}

explain_findings_lane(){
  local backend=heuristic
  ollama_up && backend=auto
  local -a prof=() extra=()
  mapfile -t prof < <(profile_args)
  [[ -n "$MODEL_TAG" ]] && extra+=(--model "$MODEL_TAG")
  [[ -n "$TOPK" ]]      && extra+=(--top-k "$TOPK")
  run "$PY" ./llm_explain.py "$OUTDIR/classifier_findings.json" \
      --out "$OUTDIR/llm_findings.json" --html "$OUTDIR/report.html" \
      --backend "$backend" --ollama-url "$OLLAMA_URL" \
      ${extra[@]+"${extra[@]}"} ${prof[@]+"${prof[@]}"} || return 1
  run "$PY" ./to_sarif.py "$OUTDIR/llm_findings.json" -o "$OUTDIR/findings.sarif" || true
}

# Delete the artifacts a lane is about to rewrite. Without this, a lane that
# fails (or is not run at all this time) leaves the PREVIOUS run's report and
# findings in place, and the summary below reports them as if they were fresh.
clear_outputs(){  # clear_outputs fast|full|deep
  case "$1" in
    fast|full) rm -f "$OUTDIR/classifier_findings.json" "$OUTDIR/llm_findings.json" \
                     "$OUTDIR/report.html" "$OUTDIR/findings.sarif";;
    deep)      rm -f "$OUTDIR/surface/surface_review.json" "$OUTDIR/surface/surface_findings.json" \
                     "$OUTDIR/surface/surface_report.md" "$OUTDIR/surface/surface_report.html" \
                     "$OUTDIR/surface/surface.sarif";;
  esac
}

lane_fast(){
  clear_outputs fast
  head2 "▶ model-free scan  →  $OUTDIR"
  scan_paths heuristic "$@" || return 1
  explain_findings_lane
}

lane_full(){
  clear_outputs full
  head2 "▶ classifier scan  →  $OUTDIR   (model=$MODEL_DIR threshold=$THRESHOLD)"
  scan_paths classifier "$@" || return 1
  explain_findings_lane
}

lane_deep(){
  local -a prof=() extra=()
  mapfile -t prof < <(profile_args)
  [[ -n "$MODEL_TAG" ]]     && extra+=(--model "$MODEL_TAG")
  [[ "$SAMPLES" -gt 1 ]] 2>/dev/null && extra+=(--samples "$SAMPLES")
  [[ -n "$NUM_CTX" ]]       && extra+=(--num-ctx "$NUM_CTX")
  [[ "$CRITIC" == "0" ]]    && extra+=(--no-critic)
  local sd="$OUTDIR/surface"
  mkdir -p "$sd" || return 1
  clear_outputs deep
  head2 "▶ attack-surface / contract review  →  $sd   (model=$(effective_model))"
  run "$PY" ./surface_review.py "$@" \
      --json "$sd/surface_review.json" \
      --md   "$sd/surface_report.md" \
      --html "$sd/surface_report.html" \
      --pipeline-out "$sd/surface_findings.json" \
      --ollama-url "$OLLAMA_URL" \
      ${extra[@]+"${extra[@]}"} ${prof[@]+"${prof[@]}"} || return 1
  run "$PY" ./to_sarif.py "$sd/surface_findings.json" -o "$sd/surface.sarif" || true
}

do_review(){  # do_review PATH...
  local -a paths=("$@")
  [[ ${#paths[@]} -gt 0 ]] || { err "no paths"; return 1; }

  local p missing=0
  for p in "${paths[@]}"; do
    [[ -e "$p" ]] || { err "path does not exist: $p"; missing=1; }
  done
  [[ "$missing" == 0 ]] || return 1

  local lane="$LANE"
  if [[ "$lane" == auto ]]; then
    if ollama_up; then lane="auto"; else lane="fast"; fi
  fi

  # Preconditions, explained up-front rather than 10 minutes in.
  if [[ "$lane" == full ]]; then
    if [[ "$P_TORCH" != "1" ]]; then
      err "the full lane needs torch (Setup → install the Python env)."; return 1
    fi
    if [[ "$P_MODEL" == absent ]]; then
      err "no classifier at $MODEL_DIR — train one (Train menu) or unpack a transferred one."; return 1
    fi
    [[ "$P_MODEL" == bootstrap ]] && \
      warn "$MODEL_DIR is the 5-step BOOTSTRAP artifact: its scores are meaningless."
  fi
  # Every lane uses the LLM when one is reachable (the fast lane's explainer
  # runs with --backend auto), so the tag is checked once, up front.
  if ollama_up; then
    ensure_model_pulled || warn "continuing without the LLM (regex explanations only)"
  elif [[ "$lane" == deep ]]; then
    err "the contract-review lane needs a running Ollama (Setup → Ollama → start)."; return 1
  fi

  # The contract lane feeds every file to the model in ONE context so cross-file
  # contracts are visible — which is also how it overflows a small context.
  if [[ "$lane" == deep || "$lane" == auto || "$lane" == full ]] && ollama_up; then
    local nsrc; nsrc="$(count_sources "${paths[@]}")"
    if [[ "$nsrc" =~ ^[0-9]+$ ]] && (( nsrc > 6 )); then
      warn "$nsrc source files will be sent to the reviewer together (ctx ${NUM_CTX:-$P_OCTX})."
      warn "For best quality review a handful at a time, or raise Settings → context window."
    fi
  fi

  mkdir -p "$OUTDIR" || return 1
  local t0 t1
  t0="$(date +%s)"
  say ""
  say "${CYN}reviewing:${R} ${paths[*]}"
  if [[ "$lane" != "$LANE" ]]; then
    say "${CYN}lane:${R} $(lane_label "$lane")  ${DIM}(configured: $LANE — Ollama is down)${R}"
  else
    say "${CYN}lane:${R} $(lane_label "$lane")"
  fi

  # want_* records which lanes were SUPPOSED to run, so a lane that failed is
  # reported as a failure instead of silently vanishing from the summary.
  local did_fast=0 did_deep=0 want_fast=0 want_deep=0
  case "$lane" in
    fast)      want_fast=1;;
    deep)      want_deep=1;;
    full|auto|*) want_fast=1; ollama_up && want_deep=1;;
  esac
  case "$lane" in
    fast)      lane_fast "${paths[@]}" && did_fast=1;;
    deep)      lane_deep "${paths[@]}" && did_deep=1;;
    full)      lane_full "${paths[@]}" && did_fast=1
               [[ "$want_deep" == 1 ]] && { lane_deep "${paths[@]}" && did_deep=1; };;
    auto|*)    lane_fast "${paths[@]}" && did_fast=1
               [[ "$want_deep" == 1 ]] && { lane_deep "${paths[@]}" && did_deep=1; };;
  esac
  t1="$(date +%s)"

  head2 "── summary ──  ($((t1 - t0))s)"
  if [[ "$did_fast" == 1 ]]; then
    summarize "code-level findings" "$OUTDIR/llm_findings.json" "$OUTDIR/classifier_findings.json"
  elif [[ "$want_fast" == 1 ]]; then
    err "code-level scan FAILED — no report written (see the error above)"
  fi
  if [[ "$did_deep" == 1 ]]; then
    summarize "contract findings  " "$OUTDIR/surface/surface_findings.json"
  elif [[ "$want_deep" == 1 ]]; then
    err "contract review FAILED — no report written (see the error above)"
    say "  ${DIM}This lane is where header/whole-file analysis comes from; a 'no findings'"
    say "  result above does NOT mean the file is clean. Re-run, or try a larger model:"
    say "  Settings → Ollama model = qwen2.5-coder:14b, context window = 8192.${R}"
  fi
  say ""
  # Only list what THIS run produced (clear_outputs removed the rest).
  local f
  local -a artifacts=()
  [[ "$did_fast" == 1 ]] && artifacts+=("$OUTDIR/report.html" "$OUTDIR/llm_findings.json" "$OUTDIR/findings.sarif")
  [[ "$did_deep" == 1 ]] && artifacts+=("$OUTDIR/surface/surface_report.html" "$OUTDIR/surface/surface_report.md" \
                                        "$OUTDIR/surface/surface_review.json" "$OUTDIR/surface/surface.sarif")
  for f in ${artifacts[@]+"${artifacts[@]}"}; do
    [[ -f "$f" ]] && printf '  %s%s%s\n' "$GRN" "$f" "$R"
  done
  [[ ${#artifacts[@]} -eq 0 ]] && err "no lane completed — see the errors above"
  say ""
  # The contract review is the richer report, so it is what gets offered first.
  if [[ -f "$OUTDIR/surface/surface_report.html" && "$did_deep" == 1 ]]; then
    confirm "Open the contract-review report (HTML) now?" && open_report "$OUTDIR/surface/surface_report.html"
  elif [[ -f "$OUTDIR/report.html" ]]; then
    confirm "Open the HTML report now?" && open_report "$OUTDIR/report.html"
  fi
  return 0
}

view_report(){
  local f="$1"
  [[ -f "$f" ]] || { warn "no report at $f yet"; return 1; }
  if command -v less >/dev/null 2>&1; then less -R "$f"; else cat "$f"; fi
}

action_review_inbox(){
  ensure_inbox
  probe
  if [[ "$P_INBOX_N" == "0" ]]; then
    warn "no C/C++ files found in $INBOX/"
    inbox_hint
    pause; return
  fi
  say "${B}$P_INBOX_N${R} source file(s) in $INBOX/"
  do_review "$INBOX"
  pause
}

action_review_path(){
  local line
  ask line "Path(s) to review (file or dir, space-separated)" "$INBOX"
  local -a paths=()
  # A line that is itself an existing path is taken verbatim (so names with
  # spaces work); otherwise split on whitespace and expand any globs.
  if [[ -e "$line" ]]; then
    paths=("$line")
  else
    local -a words=() hit=()
    read -r -a words <<< "$line"
    local w
    for w in ${words[@]+"${words[@]}"}; do
      if [[ "$w" == *[\*\?\[]* ]]; then
        hit=( $w )                       # unquoted: pathname expansion
        paths+=("${hit[@]}")
      else
        paths+=("$w")
      fi
    done
  fi
  [[ ${#paths[@]} -gt 0 ]] || { err "nothing to review"; pause; return; }
  do_review "${paths[@]}"
  pause
}

action_reports(){
  head2 "Reports under $OUTDIR/"
  local -a found=()
  local f
  for f in "$OUTDIR/surface/surface_report.html" "$OUTDIR/surface/surface_report.md" \
           "$OUTDIR/surface/surface_review.json" "$OUTDIR/surface/surface.sarif" \
           "$OUTDIR/report.html" "$OUTDIR/llm_findings.json" \
           "$OUTDIR/classifier_findings.json" "$OUTDIR/findings.sarif"; do
    if [[ -f "$f" ]]; then
      found+=("$f")
      printf '  %2d) %-46s %s%s%s\n' "${#found[@]}" "$f" "$DIM" "$(date -r "$f" '+%Y-%m-%d %H:%M')" "$R"
    fi
  done
  if [[ ${#found[@]} -eq 0 ]]; then
    warn "no reports yet — run a review first."; pause; return
  fi
  local n
  ask n "Open which # (enter = back)" ""
  [[ -z "$n" ]] && return
  if [[ "$n" =~ ^[0-9]+$ ]] && (( n >= 1 && n <= ${#found[@]} )); then
    local target="${found[n-1]}"
    if [[ "$target" == *.html ]]; then open_report "$target"; else view_report "$target"; fi
  else
    warn "invalid choice"
  fi
  pause
}

# ---------------------------------------------------------------------------
# setup / install
# ---------------------------------------------------------------------------
have_sudo(){ command -v sudo >/dev/null 2>&1 || [[ "${EUID:-$(id -u)}" -eq 0 ]]; }

py_candidates(){  # newest-first list of interpreters that can host the ML stack
  local v
  for v in 3.13 3.12 3.11; do
    command -v "python$v" >/dev/null 2>&1 && printf '%s\n' "python$v"
  done
}

action_doctor(){
  head2 "Doctor"
  probe
  local -a args=()
  [[ "$P_TORCH" == "1" ]] || args+=(--skip-ml)
  args+=(--model "$MODEL_DIR" --ollama-url "$OLLAMA_URL")
  run "$PY" preflight_doctor.py "${args[@]}"
  local rc=$?
  say ""
  hr
  say "  interpreter        $PY  (python $P_PY)"
  say "  venv               $( [[ -x "$VENV/bin/python" ]] && echo "$VENV" || echo "none" )"
  say "  torch              $( [[ "$P_TORCH" == 1 ]] && echo yes || echo no )     transformers: $( [[ "$P_TRANSFORMERS" == 1 ]] && echo yes || echo no )     peft: $( [[ "$P_PEFT" == 1 ]] && echo yes || echo no )"
  say "  tree-sitter        $( [[ "$P_TS" == 1 ]] && echo yes || echo "no (brace fallback in use)" )"
  # preflight only checks that the model dir is complete; it cannot tell a real
  # classifier from the 5-step smoke artifact, so say it explicitly here.
  case "$P_MODEL" in
    bootstrap) say "  classifier         $MODEL_DIR — ${YEL}BOOTSTRAP smoke artifact${R} (5 steps; its scores mean nothing)";;
    trained)   say "  classifier         $MODEL_DIR — trained (threshold ${P_THR:-n/a})";;
    *)         say "  classifier         none at $MODEL_DIR";;
  esac
  say "  classifier data    data/train.jsonl: $P_TRAIN_ROWS rows"
  say "  distillation data  data/surface_sft_train.jsonl: $P_SFT_ROWS rows"
  hr
  say ""
  say "Lanes available right now:"
  say "  fast (model-free)          ${GRN}yes${R}  — always works, pure stdlib"
  if ollama_up; then
    say "  deep (contract review)     ${GRN}yes${R}  — Ollama up, model $(effective_model)"
  else
    say "  deep (contract review)     ${YEL}no${R}   — start Ollama (Setup → Ollama)"
  fi
  if [[ "$P_TORCH" == 1 && "$P_MODEL" == trained ]]; then
    say "  full (classifier)          ${GRN}yes${R}"
  elif [[ "$P_TORCH" == 1 && "$P_MODEL" == bootstrap ]]; then
    say "  full (classifier)          ${YEL}degraded${R} — $MODEL_DIR is the smoke-test artifact"
  else
    say "  full (classifier)          ${YEL}no${R}   — needs torch + a trained model"
  fi
  return $rc
}

action_install_env(){
  head2 "Python environment"
  probe
  say "Current interpreter: $PY (python $P_PY)"
  if [[ "$P_PY311" != "1" ]]; then
    warn "python $P_PY is below 3.11 — the pinned ML deps (numpy==2.3.3) need 3.11+."
    say  "The model-free and contract-review lanes run fine on this interpreter;"
    say  "only the classifier/training lanes need the ML stack."
    local -a cands=()
    mapfile -t cands < <(py_candidates)
    if [[ ${#cands[@]} -gt 0 ]]; then
      say "Found a newer interpreter: ${cands[0]} — the venv will be built with it."
    else
      warn "No python3.11+ found on PATH. Install one first (e.g. apt-get install python3.12-venv)."
    fi
  fi
  say ""
  say "  1) create/repair the venv + install requirements.txt"
  say "  2) install torch for this GPU/arch only          ${DIM}(lib_torch_install.sh)${R}"
  say "  3) install tree-sitter parsers only              ${DIM}(better function boundaries)${R}"
  say "  4) fetch the GraphCodeBERT base                  ${DIM}(needed to TRAIN the classifier)${R}"
  say "  5) full machine setup                            ${DIM}(setup_machine.sh: all of the above)${R}"
  say "  0) back"
  local n; ask n "> pick" "0"
  case "$n" in
    1)
      local base="python3"
      local -a cands=(); mapfile -t cands < <(py_candidates)
      [[ "$P_PY311" != "1" && ${#cands[@]} -gt 0 ]] && base="${cands[0]}"
      if [[ ! -x "$VENV/bin/python" ]]; then
        run "$base" -m venv "$VENV" || { err "venv creation failed"; pause; return; }
      fi
      run "$VENV/bin/python" -m pip install --upgrade pip wheel setuptools || true
      say ""
      say "${DIM}torch is installed separately (option 2) — requirements.txt does not pin it.${R}"
      run "$VENV/bin/python" -m pip install --no-cache-dir -r requirements.txt
      ;;
    2)
      # ccr_install_torch calls a bare `pip`, so the venv must be ACTIVE for the
      # wheel to land in it rather than in the system interpreter.
      if [[ ! -f "$VENV/bin/activate" ]]; then
        err "no venv at $VENV — run option 1 first."
      else
        run bash -c 'source "$1/bin/activate" && source ./lib_torch_install.sh && ccr_install_torch' _ "$VENV"
      fi
      ;;
    3)
      # `--user` is rejected inside a virtualenv, so only pass it outside one.
      local -a user_flag=()
      [[ "$PY" == "$VENV/bin/python" ]] || user_flag=(--user)
      run "$PY" -m pip install ${user_flag[@]+"${user_flag[@]}"} \
          tree_sitter tree_sitter_c tree_sitter_cpp
      ;;
    4)
      if [[ "$P_TRANSFORMERS" != "1" ]]; then
        err "needs transformers (Setup → 3 → 1 first)."
      else
        run env PYBIN="$PY" bash ./fetch_graphcodebert.sh
      fi
      ;;
    5)
      if ! have_sudo && command -v apt-get >/dev/null 2>&1; then
        warn "no sudo on this box: setup_machine.sh will SKIP the apt step and continue."
      fi
      run bash ./setup_machine.sh
      ;;
    0|"") return;;
    *) warn "unknown option";;
  esac
  probe
  pause
}

action_tests(){
  head2 "Test suite"
  local tpy="$PY"
  "$PY" -c "import pytest" >/dev/null 2>&1 || tpy="python3"
  "$tpy" -c "import pytest" >/dev/null 2>&1 || { err "pytest not installed for $tpy"; pause; return; }
  run "$tpy" -m pytest -q
  pause
}

# ---------------------------------------------------------------------------
# Ollama menu
# ---------------------------------------------------------------------------
install_ollama_userspace(){
  local arch asset url
  arch="$(uname -m)"
  case "$arch" in
    x86_64)  asset="ollama-linux-amd64.tgz";;
    aarch64|arm64) asset="ollama-linux-arm64.tgz";;
    *) err "unsupported arch: $arch"; return 1;;
  esac
  url="https://ollama.com/download/$asset"
  say "This installs Ollama into ~/.local (no sudo, no systemd service)."
  say "  source: $url"
  confirm "Download and install now?" || return 1
  mkdir -p "$HOME/.local" "$STATE_DIR"
  local tmp="$STATE_DIR/$asset"
  run curl -fL --progress-bar -o "$tmp" "$url" || { err "download failed"; return 1; }
  run tar -C "$HOME/.local" -xzf "$tmp" || {
    err "extract failed — current releases may ship .tar.zst instead of .tgz."
    say "Manual route: download the release asset from https://github.com/ollama/ollama/releases"
    say "then:  zstd -d ollama-linux-${arch}.tar.zst && tar -C ~/.local -xf ollama-linux-${arch}.tar"
    return 1
  }
  rm -f "$tmp"
  [[ -x "$HOME/.local/bin/ollama" ]] && ok "installed: $HOME/.local/bin/ollama" || warn "binary not where expected"
  case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) warn "add ~/.local/bin to your PATH";; esac
}

menu_ollama(){
  while true; do
    clear 2>/dev/null || true
    head2 "Ollama — local LLM runtime"
    local bin state
    bin="$(ollama_bin 2>/dev/null || echo '(not installed)')"
    ollama_up && state="${GRN}up${R}" || state="${RED}down${R}"
    hr
    printf '  binary   %s\n' "$bin"
    printf '  server   %b @ %s\n' "$state" "$OLLAMA_URL"
    printf '  model    %s%s%s  %s\n' "$B" "$(effective_model)" "$R" \
      "$( ollama_up && { ollama_has_model "$(effective_model)" && echo "${GRN}(pulled)${R}" || echo "${YEL}(not pulled)${R}"; } )"
    hr
    say "  1) start the server        2) stop the server"
    say "  3) list installed models   4) pull $(effective_model)"
    say "  5) pull another model      6) install Ollama into ~/.local"
    say "  0) back"
    local n; ask n "> pick" "0"
    case "$n" in
      1) start_ollama; pause;;
      2) if pkill -f "ollama serve" 2>/dev/null; then ok "stopped"; else warn "no 'ollama serve' process found"; fi; pause;;
      3) head2 "installed models"; ollama_models | sed 's/^/  /' ; pause;;
      4) pull_model "$(effective_model)"; pause;;
      5) local t; ask t "model tag" "qwen2.5-coder:7b"; [[ -n "$t" ]] && pull_model "$t"; pause;;
      6) install_ollama_userspace; pause;;
      0|"") return;;
      *) warn "unknown option"; pause;;
    esac
  done
}

# ---------------------------------------------------------------------------
# train (classifier)
# ---------------------------------------------------------------------------
require_ml(){  # returns 1 and explains when the ML stack is unusable
  probe
  local bad=0
  [[ "$P_TORCH" == "1" ]] || { err "torch is not installed for $PY"; bad=1; }
  [[ -f models/graphcodebert-base/config.json ]] || { err "models/graphcodebert-base is missing (Setup → 2 → 4)"; bad=1; }
  # train.sh / eval_cpp.sh source the venv unconditionally under `set -e`.
  [[ -f "$VENV/bin/activate" ]] || { err "no venv at $VENV — the train/eval wrappers require one"; bad=1; }
  [[ "$bad" == 0 ]] || { say "Fix with: Setup → Install / repair the Python environment"; return 1; }
  return 0
}

menu_train(){
  while true; do
    clear 2>/dev/null || true
    probe
    head2 "Train the classifier (GraphCodeBERT triage model)"
    hr
    printf '  model dir     %s  (%s%s%s)\n' "$MODEL_DIR" "$B" "$P_MODEL" "$R"
    printf '  training data data/train.jsonl: %s rows\n' "$P_TRAIN_ROWS"
    printf '  torch         %s     base model: %s\n' \
      "$( [[ "$P_TORCH" == 1 ]] && echo yes || echo no )" \
      "$( [[ -f models/graphcodebert-base/config.json ]] && echo present || echo missing )"
    hr
    say "  1) smoke train on the bootstrap set   ${DIM}(~1 min; proves plumbing, NOT a usable model)${R}"
    say "  2) real train on BigVul               ${DIM}(one_click: fetch → split → train → eval; needs network+GPU)${R}"
    say "  3) retrain on the existing data/      ${DIM}(train.sh — uses data/train.jsonl as-is)${R}"
    say "  4) evaluate the current model         ${DIM}(held-out data/test.jsonl)${R}"
    say "  5) evaluate on the real C++/PrimeVul sets ${DIM}(eval_cpp.sh)${R}"
    say "  6) pack / unpack a model for transfer"
    say "  0) back"
    local n; ask n "> pick" "0"
    case "$n" in
      1)
        require_ml || { pause; continue; }
        warn "a bootstrap-trained model is a plumbing test, not a usable classifier."
        confirm "Continue?" || { pause; continue; }
        run bash ./bootstrap_data.sh && \
        run env VENV="$VENV" EPOCHS=1 OUT="$MODEL_DIR" bash ./train.sh
        pause;;
      2)
        say "This runs the full online provisioning + training window:"
        say "  ${DIM}DATA=bigvul ./one_click_unlock_fetch_train_relock.sh${R}"
        say "It needs network access, the ML stack, and a GPU; it takes ~1h+."
        [[ "$P_PY311" == "1" ]] || warn "python $P_PY < 3.11: the pinned deps will not install here."
        confirm "Start it?" || { pause; continue; }
        run env DATA=bigvul VENV="$VENV" OUT="$MODEL_DIR" bash ./one_click_unlock_fetch_train_relock.sh
        pause;;
      3)
        require_ml || { pause; continue; }
        [[ "$P_TRAIN_ROWS" -gt 0 ]] || { err "data/train.jsonl is empty/missing — use option 1 or 2 first."; pause; continue; }
        local ep; ask ep "epochs" "3"
        run env VENV="$VENV" EPOCHS="$ep" OUT="$MODEL_DIR" bash ./train.sh
        pause;;
      4)
        require_ml || { pause; continue; }
        [[ -f data/test.jsonl ]] || { err "data/test.jsonl missing"; pause; continue; }
        run "$PY" evaluate_model.py --model "$MODEL_DIR" --test data/test.jsonl
        pause;;
      5)
        require_ml || { pause; continue; }
        run env VENV="$VENV" MODEL="$MODEL_DIR" bash ./eval_cpp.sh
        pause;;
      6)
        if ! command -v zstd >/dev/null 2>&1; then
          err "zstd is not installed — pack/unpack need it (apt-get install zstd)."; pause; continue
        fi
        say "  a) pack $MODEL_DIR  ->  trained_model.tar.zst"
        say "  b) unpack an archive into ."
        local m; ask m "> pick" ""
        case "$m" in
          a|A) run bash ./pack_model.sh "$MODEL_DIR" trained_model.tar.zst;;
          b|B) local arc; ask arc "archive path" "trained_model.tar.zst"
               [[ -f "$arc" ]] && run bash ./unpack_model.sh "$arc" . || err "not found: $arc";;
        esac
        pause;;
      0|"") return;;
      *) warn "unknown option"; pause;;
    esac
  done
}

# ---------------------------------------------------------------------------
# tune (surface-review LoRA distillation)
# ---------------------------------------------------------------------------
menu_tune(){
  while true; do
    clear 2>/dev/null || true
    probe
    local vram=""
    vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    head2 "Tune the reviewer LLM (Opus-review distillation → LoRA)"
    hr
    printf '  SFT data      data/surface_sft_train.jsonl: %s rows\n' "$P_SFT_ROWS"
    printf '  teacher revs  train: %s   eval: %s\n' \
      "$(ls corpus/reviews/train/*.json 2>/dev/null | wc -l)" \
      "$(ls corpus/reviews/eval/*.json 2>/dev/null | wc -l)"
    printf '  stack         torch:%s peft:%s   profile:%s   VRAM:%s MiB\n' \
      "$( [[ "$P_TORCH" == 1 ]] && echo yes || echo no )" \
      "$( [[ "$P_PEFT" == 1 ]] && echo yes || echo no )" "$P_PROFILE" "${vram:-?}"
    printf '  adapter out   ./llm-surface-lora  (%s)\n' \
      "$( [[ -d llm-surface-lora ]] && echo present || echo absent )"
    hr
    say "  ${DIM}Recipe + measured memory budget: TRAINING_SURFACE_LORA.md${R}"
    say "  1) rebuild the SFT dataset from corpus/reviews"
    say "  2) train the LoRA                      ${DIM}(needs ~45GB — DGX Spark; a 24GB card cannot)${R}"
    say "  3) merge the LoRA into a full model"
    say "  4) register the merged model with Ollama"
    say "  5) score a model against the held-out Opus refs ${DIM}(before/after)${R}"
    say "  0) back"
    local n; ask n "> pick" "0"
    case "$n" in
      1)
        [[ -f corpus/sft_manifest.json ]] || { err "corpus/sft_manifest.json missing"; pause; continue; }
        run "$PY" model/build_surface_sft.py --manifest corpus/sft_manifest.json --out-dir data
        pause;;
      2)
        if [[ "$P_TORCH" != "1" || "$P_PEFT" != "1" ]]; then
          err "needs torch + peft (Setup → install the Python env)."; pause; continue
        fi
        [[ "$P_SFT_ROWS" -gt 0 ]] || { err "no SFT rows — run option 1 first."; pause; continue; }
        say "Measured on this corpus (median ~10.9k tokens/example):"
        say "  * bf16 + Liger + LoRA r=32 at max-length 20480 ≈ ${B}45 GB${R} — fits the Spark's 128 GB."
        say "  * a 24 GB card HANGS at 100% GPU under WSL2 instead of raising OOM."
        if [[ "$P_PROFILE" != "spark" ]]; then
          warn "detected profile is '$P_PROFILE', not 'spark' — this run is expected to hang or OOM."
          confirm "I understand and want to start it anyway" || { pause; continue; }
        fi
        local ml base
        ask base "base model" "Qwen/Qwen2.5-Coder-14B-Instruct"
        ask ml "--max-length (20480 keeps 92% of the corpus)" "20480"
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments is fatal with 4-bit; the
        # trainer refuses it. bf16 + Liger is the supported path (no bitsandbytes).
        run env -u PYTORCH_CUDA_ALLOC_CONF "$PY" model/train_surface_sft.py \
            --base "$base" \
            --train data/surface_sft_train.jsonl \
            --val   data/surface_sft_val.jsonl \
            --out   ./llm-surface-lora \
            --no-4bit --liger --max-length "$ml" \
            --epochs 3 --batch 1 --accum 8 --lr 2e-4 --lora-r 32 --lora-alpha 64
        pause;;
      3)
        [[ -d llm-surface-lora ]] || { err "no adapter at ./llm-surface-lora — train it first."; pause; continue; }
        local base; ask base "base model" "Qwen/Qwen2.5-Coder-14B-Instruct"
        run "$PY" model/merge_lora.py --base "$base" --adapter ./llm-surface-lora --out ./llm-surface-merged
        pause;;
      4)
        [[ -d llm-surface-merged ]] || { err "no ./llm-surface-merged — merge it first (option 3)."; pause; continue; }
        local bin; bin="$(ollama_bin)" || { err "ollama binary not found"; pause; continue; }
        local tag; ask tag "new Ollama tag" "qwen2.5-coder-surface:14b"
        mkdir -p "$STATE_DIR"
        # Reuse the recorded ChatML template; only the FROM line changes.
        { printf 'FROM %s/llm-surface-merged\n' "$PWD"
          sed -n '/^TEMPLATE/,$p' model/serving/Modelfile.base; } > "$STATE_DIR/Modelfile"
        say "Modelfile written to $STATE_DIR/Modelfile"
        OLLAMA_HOST="$OLLAMA_HOSTPORT" run "$bin" create "$tag" -f "$STATE_DIR/Modelfile"
        pause;;
      5)
        ollama_up || { err "needs a running Ollama."; pause; continue; }
        local tag; ask tag "model tag to score" "$(effective_model)"
        run "$PY" scripts/py/eval_surface_model.py --model "$tag" \
            --manifest corpus/eval_manifest.prompts.json --out-dir corpus/eval_runs \
            --ollama-url "$OLLAMA_URL"
        pause;;
      0|"") return;;
      *) warn "unknown option"; pause;;
    esac
  done
}

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
menu_settings(){
  while true; do
    clear 2>/dev/null || true
    probe
    head2 "Settings ${DIM}(saved to $CONF)${R}"
    hr
    printf '  1) review lane          %s\n' "$(lane_label "$LANE")"
    printf '  2) hardware profile     %s\n' "${PROFILE:-auto ($P_PROFILE)}"
    printf '  3) Ollama model         %s\n' "$(effective_model)${MODEL_TAG:+ (pinned)}"
    printf '  4) recall samples       %s   %s\n' "$SAMPLES" "${DIM}higher = better recall, slower${R}"
    printf '  5) completeness pass    %s\n' "$( [[ "$CRITIC" == 1 ]] && echo on || echo off )"
    printf '  6) context window       %s\n' "${NUM_CTX:-profile default ($P_OCTX)}"
    printf '  7) explained findings   %s\n' "${TOPK:-profile default}"
    printf '  8) classifier model dir %s   (%s)\n' "$MODEL_DIR" "$P_MODEL"
    printf '  9) classifier threshold %s   %s\n' "$THRESHOLD" "${DIM}or 'tuned' (${P_THR:-n/a})${R}"
    printf ' 10) min severity         %s\n' "${MIN_SEV:-all}"
    printf ' 11) inbox directory      %s\n' "$INBOX"
    printf ' 12) output directory     %s\n' "$OUTDIR"
    hr
    say "  0) back (settings are saved automatically)"
    local n; ask n "> pick" "0"
    case "$n" in
      1) say "  auto | fast | deep | full"
         local l; ask l "lane" "$LANE"
         case "$l" in auto|fast|deep|full) LANE="$l";; *) warn "unknown lane";; esac;;
      2) say "  (empty) = auto-detect | 4090 | spark | laptop | cpu"
         local p; ask p "profile" "${PROFILE:-auto}"
         [[ "$p" == auto ]] && p=""
         case "$p" in ""|4090|spark|laptop|cpu) PROFILE="$p";; *) warn "unknown profile";; esac;;
      3) head2 "installed models"; ollama_models | sed 's/^/  /'
         say "  ${DIM}('-' = use the profile default: $P_OMODEL)${R}"
         local t; ask t "model tag" "${MODEL_TAG:-}"
         [[ "$t" == "-" ]] && t=""
         MODEL_TAG="$t";;
      4) local s; ask s "samples [1-5]" "$SAMPLES"
         [[ "$s" =~ ^[1-5]$ ]] && SAMPLES="$s" || warn "unchanged";;
      5) [[ "$CRITIC" == 1 ]] && CRITIC=0 || CRITIC=1;;
      6) local c; ask c "num_ctx ('-' = profile default $P_OCTX)" "${NUM_CTX:-}"
         [[ "$c" == "-" ]] && c=""
         [[ -z "$c" || "$c" =~ ^[0-9]+$ ]] && NUM_CTX="$c" || warn "unchanged";;
      7) local k; ask k "top-k findings to explain with the LLM ('-' = profile)" "${TOPK:-}"
         [[ "$k" == "-" ]] && k=""
         [[ -z "$k" || "$k" =~ ^[0-9]+$ ]] && TOPK="$k" || warn "unchanged";;
      8) local d; ask d "classifier dir" "$MODEL_DIR"; MODEL_DIR="$d";;
      9) local t; ask t "threshold (0-1, or 'tuned')" "$THRESHOLD"
         [[ "$t" =~ ^(tuned|[01](\.[0-9]+)?|\.[0-9]+)$ ]] && THRESHOLD="$t" || warn "unchanged";;
     10) say "  '-' = every severity | info | low | medium | high | critical"
         local s; ask s "min severity" "${MIN_SEV:-}"
         [[ "$s" == "-" ]] && s=""
         case "$s" in ""|info|low|medium|high|critical) MIN_SEV="$s";; *) warn "unchanged";; esac;;
     11) local d; ask d "inbox dir" "$INBOX"; INBOX="$d"; ensure_inbox;;
     12) local d; ask d "output dir" "$OUTDIR"; OUTDIR="$d";;
      0|"") save_conf; return;;
      *) warn "unknown option"; pause;;
    esac
    save_conf
  done
}

menu_setup(){
  while true; do
    clear 2>/dev/null || true
    head2 "Setup"
    say "  1) Doctor — check everything and say what is missing"
    say "  2) Hardware — measured VRAM and the optimal model for it"
    say "  3) Install / repair the Python environment"
    say "  4) Ollama — server and models"
    say "  5) Run the test suite"
    say "  0) back"
    local n; ask n "> pick" "0"
    case "$n" in
      1) action_doctor; pause;;
      2) action_hardware;;
      3) action_install_env;;
      4) menu_ollama;;
      5) action_tests;;
      0|"") return;;
      *) warn "unknown option"; pause;;
    esac
  done
}

# ---------------------------------------------------------------------------
# main menu
# ---------------------------------------------------------------------------
status_block(){
  local ollama_state cls
  if ollama_up; then ollama_state="${GRN}up${R}"; else ollama_state="${RED}down${R}"; fi
  case "$P_MODEL" in
    trained)   cls="${GRN}trained${R}";;
    bootstrap) cls="${YEL}bootstrap artifact${R}";;
    *)         cls="${DIM}absent${R}";;
  esac
  hr
  printf '  python   %-10s torch %-4s tree-sitter %-4s\n' "$P_PY" \
    "$( [[ "$P_TORCH" == 1 ]] && echo yes || echo no )" \
    "$( [[ "$P_TS" == 1 ]] && echo yes || echo no )"
  printf '  ollama   %b @ %-28s model %s%s%s\n' "$ollama_state" "$OLLAMA_URL" "$B" "$(effective_model)" "$R"
  printf '  profile  %-10s classifier %b\n' "${PROFILE:-$P_PROFILE}" "$cls"
  printf '  inbox    %s/ %s(%s file%s)%s   out %s/\n' "$INBOX" "$DIM" "$P_INBOX_N" \
    "$( [[ "$P_INBOX_N" == 1 ]] || echo s )" "$R" "$OUTDIR"
  printf '  lane     %s\n' "$(lane_label "$LANE")"
  # Only speak up when the machine could run something better than what is set.
  if [[ -n "$P_REC_TAG" && "$P_REC_FITS" == "1" && "$P_REC_TAG" != "$(effective_model)" ]]; then
    printf '  %shint%s     this GPU fits %s%s%s (%s MiB) — Setup → Hardware\n' \
      "$YEL" "$R" "$B" "$P_REC_TAG" "$R" "$P_REC_MIB"
  fi
  hr
}

action_hardware(){
  head2 "Hardware & optimal model"
  # Recompute WITH the installed tag list so the advice knows what is pulled.
  local tags; tags="$(ollama_models 2>/dev/null | paste -sd, -)"
  CCR_INSTALLED_TAGS="$tags" probe
  hr
  printf '  GPU        %s\n' "${P_GPU:-none detected}"
  printf '  VRAM       %s MiB\n' "${P_VRAM:-unknown}"
  printf '  RAM        %s MiB %s\n' "${P_RAM:-unknown}" \
    "$( [[ -r /proc/version ]] && grep -qi microsoft /proc/version 2>/dev/null && \
        echo "${DIM}(WSL sees a slice of host RAM; raise it in .wslconfig)${R}" )"
  printf '  profile    %s   context %s\n' "${PROFILE:-$P_PROFILE}" "${NUM_CTX:-$P_OCTX}"
  hr
  say ""
  say "  ${B}Optimal model for this machine: $P_REC_TAG${R}  (${P_REC_MIB} MiB resident)"
  say "  ${DIM}$P_REC_WHY${R}"
  [[ "$P_REC_FITS" == "1" ]] || warn "nothing in the catalog fits fully — expect host-RAM spill"
  say ""
  say "  What fits at context ${NUM_CTX:-$P_OCTX}:"
  "$PY" - "${NUM_CTX:-$P_OCTX}" "${P_VRAM:-0}" "$tags" <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
import profiles
ctx = int(sys.argv[1]); vram = int(sys.argv[2] or 0)
have = [t for t in (sys.argv[3] or "").split(",") if t]
budget = vram - profiles.VRAM_HEADROOM_MIB if vram else 0
for e in sorted(profiles.MODEL_CATALOG, key=lambda x: -x["rank"]):
    need = profiles.model_footprint_mib(e, ctx)
    fits = "fits " if (vram and need <= budget) else "SPILLS"
    pulled = "pulled" if profiles.tag_present(e["tag"], have) else "not pulled"
    print(f"    {e['tag']:<22} {need:>6} MiB  {fits}  {pulled}")
if vram:
    print(f"    {'':<22} {budget:>6} MiB  usable (VRAM minus "
          f"{profiles.VRAM_HEADROOM_MIB} MiB headroom)")
PY
  say ""
  if [[ -n "$P_REC_TAG" ]] && ! ollama_has_model "$P_REC_TAG"; then
    if confirm "Pull $P_REC_TAG now?"; then pull_model "$P_REC_TAG"; fi
  elif [[ -n "$P_REC_TAG" && "$P_REC_TAG" != "$(effective_model)" ]]; then
    if confirm "Use $P_REC_TAG for reviews (saved to $CONF)?"; then
      MODEL_TAG="$P_REC_TAG"; save_conf; ok "model = $MODEL_TAG"
    fi
  else
    ok "already running the optimal model for this machine"
  fi
  pause
}

main_menu(){
  while true; do
    clear 2>/dev/null || true
    probe
    say "${B}C/C++ Code-Review LLM${R} ${DIM}— local, offline${R}"
    status_block
    say "  ${B}1${R}) Review the inbox        ${DIM}(everything in $INBOX/)${R}"
    say "  ${B}2${R}) Review a specific path"
    say "  ${B}3${R}) Reports                 ${DIM}(open / view the last results)${R}"
    say ""
    say "  ${B}4${R}) Setup                   ${DIM}(doctor, python env, Ollama, tests)${R}"
    say "  ${B}5${R}) Train the classifier"
    say "  ${B}6${R}) Tune the reviewer LLM   ${DIM}(LoRA distillation)${R}"
    say "  ${B}7${R}) Settings"
    say ""
    say "  ${B}q${R}) Quit                    ${DIM}(enter = refresh)${R}"
    local choice="" rc=0
    read -rep $'\n> ' choice; rc=$?
    # rc >128 = interrupted by a signal (Ctrl-C) -> redraw; rc 1 = EOF (Ctrl-D) -> quit.
    if (( rc != 0 )); then
      (( rc > 128 )) && continue
      say ""; exit 0
    fi
    case "$choice" in
      1) action_review_inbox;;
      2) action_review_path;;
      3) action_reports;;
      4) menu_setup;;
      5) menu_train;;
      6) menu_tune;;
      7) menu_settings;;
      q|Q|quit|exit) say "bye."; exit 0;;
      "") ;;   # refresh
      *) warn "unknown option: $choice"; pause;;
    esac
  done
}

# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
case "${1:-}" in
  -h|--help)
    # The header block, up to the first non-comment line.
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
    exit 0;;
  "") ;;
  *) err "unexpected argument: $1 (this script takes none; try --help)"; exit 2;;
esac

mkdir -p "$STATE_DIR" 2>/dev/null || true
load_conf
resolve_py

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  err "need python 3.8+ ($PY is too old or missing)."
  exit 1
fi

ensure_inbox
# Ctrl-C aborts the running action and returns to the menu instead of quitting.
trap 'printf "\n%s(interrupted)%s\n" "$YEL" "$R"' INT
main_menu
