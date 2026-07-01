#!/usr/bin/env bash
# Offline installer for a ccr-airgap bundle (built by build_airgap_bundle.sh).
# Run on the TARGET (airgapped) machine from the extracted bundle root:
#
#   ./install_offline.sh               # full install: venv + wheels + model + smoke
#   ./install_offline.sh verify-only   # integrity + arch/python checks, then stop
#
# Zero network: wheels come from the bundled wheelhouse (pip --no-index), the
# model from model/model.tar.zst, and the smoke test uses the heuristic
# explainer (no Ollama needed).
#
# Flags / env:
#   --skip-ml            force the model-free install even from a full bundle
#   --venv DIR           venv location (default .venv; also env VENV)
#   CCR_INSTALL_DRYRUN=1 same as the verify-only subcommand
#   CCR_ARCH / CCR_PY_VER  override detection (tests / cross-checking a bundle)
#   CCR_FORCE_MODEL=1    re-unpack the model over an existing ./vuln-model
set -euo pipefail

say(){  printf '\n\033[1;36m[airgap-install]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die(){  printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; exit 1; }

MODE="install"
SKIP_ML_FLAG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    verify-only) MODE="verify"; shift;;
    --skip-ml)   SKIP_ML_FLAG=1; shift;;
    --venv)      VENV="$2"; shift 2;;
    -h|--help)
      sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
      exit 0;;
    *) die "unknown argument: $1 (see --help)";;
  esac
done
[[ "${CCR_INSTALL_DRYRUN:-0}" == "1" ]] && MODE="verify"
VENV="${VENV:-.venv}"

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
[[ -f SHA256SUMS ]] || die "SHA256SUMS not found — run this from the extracted bundle root"
[[ -d repo ]] || die "repo/ not found — run this from the extracted bundle root"

# --- 1. integrity: every bundle file must match SHA256SUMS ---------------------
say "Verifying bundle integrity (SHA256SUMS)"
if ! sha256sum -c --quiet SHA256SUMS; then
  die "SHA256SUMS verification FAILED — the bundle is corrupt or was tampered with. Refusing to install."
fi
# sha256sum -c only checks LISTED files; also refuse files ADDED to the shipped
# dirs (wheels/repo/model) that SHA256SUMS does not know about. Install-generated
# artifacts (.venv, hf_cache, vuln-model, .env.locked) live at the root, so
# re-runs stay clean.
UNLISTED="$(python3 - <<'PY'
import os
listed = set()
with open("SHA256SUMS", encoding="utf-8") as f:
    for ln in f:
        ln = ln.rstrip("\n")
        if ln:
            listed.add(ln.split(None, 1)[1].lstrip("*"))
extra = []
for top in ("wheels", "repo", "model"):
    for root, _, names in os.walk(top):
        for n in names:
            p = os.path.join(root, n)
            if os.path.join(".", p) not in listed and p not in listed:
                extra.append(p)
print("\n".join(extra))
PY
)"
if [[ -n "$UNLISTED" ]]; then
  printf '%s\n' "$UNLISTED" | sed 's/^/[err] unlisted file: /' >&2
  die "bundle contains files not covered by SHA256SUMS — refusing to install."
fi
echo "[ok] all bundle files match SHA256SUMS"

# --- 2. bundle mode + arch/python -> wheelhouse selection ----------------------
SKIP_ML="$SKIP_ML_FLAG"
if [[ -f bundle_manifest.json ]]; then
  BUNDLE_SKIP_ML="$(python3 -c 'import json; print(1 if json.load(open("bundle_manifest.json")).get("skip_ml") else 0)')"
  [[ "$BUNDLE_SKIP_ML" == "1" ]] && SKIP_ML=1
fi

ARCH="${CCR_ARCH:-$(uname -m)}"
[[ "$ARCH" == "arm64" ]] && ARCH="aarch64"
PY_VER="${CCR_PY_VER:-$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')}"
PYTAG="cp${PY_VER//./}"

WHEELS="wheels/$ARCH/$PYTAG"
if [[ ! -d "$WHEELS" ]]; then
  echo "[err] no wheelhouse for this machine: $WHEELS (arch=$ARCH python=$PY_VER)" >&2
  echo "[err] available arch/python combos in this bundle:" >&2
  if [[ -d wheels ]]; then
    find wheels -mindepth 2 -maxdepth 2 -type d | sed 's|^wheels/|  - |' >&2
  else
    echo "  - (none: wheels/ is missing entirely)" >&2
  fi
  echo "[err] rebuild with: ./build_airgap_bundle.sh --arch $ARCH --python $PY_VER" >&2
  exit 1
fi

# --- 3. python floor -------------------------------------------------------------
# The pinned ML deps (numpy==2.3.3) need python >= 3.11. The model-free lane
# (--skip-ml) is pure stdlib and runs on 3.10+.
version_ge(){ python3 -c "import sys; sys.exit(0 if tuple(map(int, '$1'.split('.'))) >= ($2) else 1)"; }
if [[ "$SKIP_ML" == "1" ]]; then
  version_ge "$PY_VER" "3, 10" || die "python3 is $PY_VER; even the model-free lane needs >= 3.10."
else
  version_ge "$PY_VER" "3, 11" || die "python3 is $PY_VER but the ML install needs >= 3.11 (numpy pin). Install python3.11+, or use a --skip-ml bundle / pass --skip-ml."
fi

say "Preflight OK: arch=$ARCH python=$PY_VER wheelhouse=$WHEELS skip_ml=$SKIP_ML"
if [[ "$MODE" == "verify" ]]; then
  echo "[ok] verify-only: integrity + arch/python selection all pass."
  exit 0
fi

# --- 4. venv + wheels (no network: --no-index) -------------------------------------
say "Creating venv at $VENV"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [[ "$SKIP_ML" == "1" ]]; then
  say "Installing tree-sitter trio (everything else in the model-free lane is stdlib)"
  if ! pip install --no-index --find-links "$WHEELS" tree_sitter tree_sitter_c tree_sitter_cpp; then
    warn "tree-sitter wheels unavailable — the scanner falls back to the brace parser (--parser brace); still functional."
  fi
else
  say "Installing pinned requirements from the local wheelhouse"
  pip install --no-index --find-links "$WHEELS" -r repo/requirements.txt
  if compgen -G "$WHEELS/torch-*.whl" >/dev/null; then
    say "Installing CPU torch from the wheelhouse"
    pip install --no-index --find-links "$WHEELS" torch
  else
    warn "no torch wheel in $WHEELS — the ML lane will not run (model-free lane unaffected)."
  fi
fi

# --- 5. model -------------------------------------------------------------------------
if [[ "$SKIP_ML" == "1" ]]; then
  # Model-free install: needs no model and therefore no zstd — even from a
  # full bundle (that is --skip-ml's contract). Do not touch model/ at all.
  say "--skip-ml: skipping model unpack (model-free lane needs no model, no zstd)"
elif [[ -f model/model.tar.zst ]]; then
  if [[ -d vuln-model && "${CCR_FORCE_MODEL:-0}" != "1" ]]; then
    warn "./vuln-model already exists — keeping it (set CCR_FORCE_MODEL=1 to re-unpack)."
  else
    command -v zstd >/dev/null 2>&1 || die "zstd not found — unpack_model.sh needs it. Install the zstd package on this machine (it is not bundled) and re-run."
    say "Unpacking model -> ./vuln-model (manifest-verified)"
    rm -rf .model_unpack
    mkdir -p .model_unpack
    bash repo/unpack_model.sh model/model.tar.zst .model_unpack
    MODEL_TOP="$(find .model_unpack -mindepth 2 -maxdepth 3 -name config.json -printf '%h\n' | head -1)"
    [[ -n "$MODEL_TOP" ]] || die "unpacked model contains no config.json"
    rm -rf vuln-model
    mv "$MODEL_TOP" vuln-model
    rm -rf .model_unpack
  fi
else
  warn "bundle carries no model/ — ML lane unavailable (model-free lane still works)."
fi

# --- 6. offline lock (same .env.locked mechanism as one_click, idempotent) -----------
say "Writing .env.locked + venv activation hook (offline by default)"
mkdir -p "$HERE/hf_cache"
cat > .env.locked <<EOF
export HF_HOME="$HERE/hf_cache"
export TRANSFORMERS_CACHE="$HERE/hf_cache"
export HF_DATASETS_CACHE="$HERE/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
EOF
# Relock hook (if-form, absolute path, idempotent). Two invariants:
#   * .env.locked lives at the BUNDLE ROOT ($HERE) — that is where the
#     cat > .env.locked above writes it — not necessarily at $VIRTUAL_ENV/..,
#     so the hook embeds the absolute path; a custom --venv outside the
#     bundle root still gets offline-locked.
#   * it MUST be an `if ...; then ...; fi`: a bare `test -f ... && ...` as the
#     LAST line of activate makes `source activate` return 1 whenever
#     .env.locked is absent (e.g. retired by online_unlock.sh), killing every
#     `set -e` caller — including this installer at its own `source` on re-run.
if ! grep -qF "$HERE/.env.locked" "$VENV/bin/activate"; then
  printf '\n# ccr relock\nif test -f "%s"; then set -a; . "%s"; set +a; fi\n' \
    "$HERE/.env.locked" "$HERE/.env.locked" >> "$VENV/bin/activate"
fi

# --- 7. smoke test ---------------------------------------------------------------------
say "Smoke test"
SMOKE_FAIL=0
SMOKE_OUT="$(mktemp -d)"
trap 'rm -rf "$SMOKE_OUT"' EXIT

if python3 repo/heuristic_scan.py repo/playground -o "$SMOKE_OUT" >/dev/null \
   && python3 repo/llm_explain.py "$SMOKE_OUT/classifier_findings.json" \
        --out "$SMOKE_OUT/llm_findings.json" --backend heuristic >/dev/null \
   && python3 repo/to_sarif.py "$SMOKE_OUT/llm_findings.json" -o "$SMOKE_OUT/findings.sarif" >/dev/null; then
  echo "[PASS] model-free lane (heuristic_scan -> llm_explain --backend heuristic -> to_sarif)"
else
  echo "[FAIL] model-free lane"
  SMOKE_FAIL=1
fi

if python3 -c 'import torch' >/dev/null 2>&1; then
  # Import check + config read only — a full model load is slow and needless here.
  if [[ -f vuln-model/config.json ]]; then
    if python3 -c 'import json, torch, transformers; json.load(open("vuln-model/config.json")); print("[ok] torch", torch.__version__, "| transformers", transformers.__version__)'; then
      echo "[PASS] ML lane imports + model config readable (full scan: bash repo/scan_repo.sh)"
    else
      echo "[FAIL] ML lane import / model config check"
      SMOKE_FAIL=1
    fi
  else
    echo "[SKIP] ML config check (no ./vuln-model in this install)"
  fi
else
  echo "[SKIP] ML lane (torch not installed by this bundle)"
fi

if [[ "$SMOKE_FAIL" == "0" ]]; then
  say "INSTALL PASS — activate with: source $VENV/bin/activate (offline env auto-applied)"
else
  die "INSTALL FAIL — one or more smoke checks failed (see above)."
fi
