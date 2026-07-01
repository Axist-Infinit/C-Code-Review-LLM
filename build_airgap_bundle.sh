#!/usr/bin/env bash
# Build a self-contained airgap bundle ON A CONNECTED MACHINE, carry it to an
# offline box (x86_64 or aarch64 WSL2/Linux), then install with
# install_offline.sh — zero network on the target. See airgap/README_AIRGAP.md.
#
# Bundle layout: <out>/ccr-airgap-<gitrev>/
#   repo/                  tracked working-tree files (git archive HEAD)
#   model/                 model.tar.zst + .manifest.json (via pack_model.sh)
#   wheels/<arch>/cp<ver>/ pinned wheels per arch (+ CPU torch unless --skip-ml)
#   install_offline.sh     the offline installer (run on the target)
#   README_AIRGAP.md       operator doc
#   bundle_manifest.json   version / arches / python / wheel counts / model sha
#   SHA256SUMS             integrity over every file (verified on the target)
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./build_airgap_bundle.sh [options]   (ONLINE machine; needs pip >= 23)

  --model DIR          trained model dir to bundle (default: vuln-model)
  --out DIR            output dir (default: dist/airgap)
  --arch LIST          comma-separated target arches (default: x86_64,aarch64)
  --python LIST        comma-separated target python minor versions
                       (default: 3.10,3.11,3.12,3.13). The pinned ML deps need
                       >= 3.11 (numpy), so pythons below that get a model-free
                       (tree-sitter only) wheelhouse; the installer detects
                       this and degrades to the model-free lane automatically.
  --skip-ml            model-free bundle: wheels limited to the tree-sitter
                       trio (tree_sitter, tree_sitter_c, tree_sitter_cpp);
                       no torch, no ML requirements. The model-free lane
                       (heuristic_scan -> llm_explain --backend heuristic ->
                       to_sarif) needs nothing else beyond stdlib python3.
  --ollama-models DIR  copy an Ollama blob store (e.g. ~/.ollama/models) into
                       the bundle for the offline LLM explainer (optional)

Env: CCR_TORCH_SPEC    override the torch spec for the CPU wheel download
                       (default 'torch'; e.g. CCR_TORCH_SPEC='torch==2.6.*')
EOF
}

MODEL_DIR="vuln-model"
OUT_DIR="dist/airgap"
ARCHES="x86_64,aarch64"
PYVERS="3.10,3.11,3.12,3.13"
ML_PY_FLOOR_MINOR=11   # requirements pins (numpy==2.3.3) need python >= 3.11
SKIP_ML=0
OLLAMA_MODELS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL_DIR="$2"; shift 2;;
    --out)           OUT_DIR="$2"; shift 2;;
    --arch)          ARCHES="$2"; shift 2;;
    --python)        PYVERS="$2"; shift 2;;
    --skip-ml)       SKIP_ML=1; shift;;
    --ollama-models) OLLAMA_MODELS_DIR="$2"; shift 2;;
    -h|--help)       usage; exit 0;;
    *) echo "[err] unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

say(){  printf '\n\033[1;36m[airgap-build]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die(){  printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

python3 -m pip --version >/dev/null 2>&1 || die "python3 -m pip not available; this builder runs on the CONNECTED machine"
IFS=',' read -r -a PYVER_LIST <<< "$PYVERS"
CLEAN_PYVERS=()
for PV in "${PYVER_LIST[@]}"; do
  PV="${PV// /}"
  [[ -n "$PV" ]] || continue
  [[ "$PV" =~ ^3\.[0-9]+$ ]] || die "--python entries must look like 3.12 (got: $PV)"
  CLEAN_PYVERS+=("$PV")
done
[[ ${#CLEAN_PYVERS[@]} -gt 0 ]] || die "--python resolved to an empty list"

VERSION="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d)"
BUNDLE="ccr-airgap-${VERSION}"
STAGE="$OUT_DIR/$BUNDLE"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# --- repo/: tracked files only (no venv/model/scan artifacts) -----------------
say "Staging repo/ (git archive HEAD)"
mkdir -p "$STAGE/repo"
if git rev-parse --verify HEAD >/dev/null 2>&1 && git archive HEAD | tar -x -C "$STAGE/repo"; then
  echo "[ok] repo/ from git archive HEAD"
else
  warn "git archive unavailable; rsync fallback (excluding build artifacts)"
  command -v rsync >/dev/null 2>&1 || die "neither git archive nor rsync available"
  rsync -a \
    --exclude '.git' --exclude 'vuln-model*' --exclude 'dist' \
    --exclude 'scan_out*' --exclude '__pycache__' --exclude '.pytest_cache' \
    ./ "$STAGE/repo/"
fi
# The installer + doc live at the bundle ROOT (working-tree copies, so a not-
# yet-committed airgap/ still ships correctly).
install -m 0755 airgap/install_offline.sh "$STAGE/install_offline.sh"
install -m 0644 airgap/README_AIRGAP.md "$STAGE/README_AIRGAP.md"

# --- model/: packed trained model ---------------------------------------------
if [[ "$SKIP_ML" == "1" && ! -e "$MODEL_DIR" ]]; then
  warn "--skip-ml and no model dir at $MODEL_DIR — bundling WITHOUT a model (model-free lane only)."
else
  [[ -f "$MODEL_DIR/config.json" ]] || die "$MODEL_DIR/config.json missing — --model must point at a trained model dir"
  # The committed vuln-model/ is a 5-step bootstrap smoke artifact, NOT the
  # real classifier. checkpoint-* dirs or a tiny global_step give it away.
  BOOTSTRAP_HINT="$(python3 - "$MODEL_DIR" <<'PY'
import glob, json, os, sys
md = sys.argv[1]
if glob.glob(os.path.join(md, "checkpoint-*")):
    print("checkpoint-* dir present")
    raise SystemExit
steps = []
for p in glob.glob(os.path.join(md, "**", "trainer_state.json"), recursive=True):
    try:
        steps.append(int(json.load(open(p)).get("global_step", 0)))
    except Exception:
        pass
if steps and min(steps) <= 10:
    print("trainer_state.json global_step=%d" % min(steps))
PY
)"
  if [[ -n "$BOOTSTRAP_HINT" ]]; then
    warn "=================================================================="
    warn "MODEL AT '$MODEL_DIR' LOOKS LIKE THE BOOTSTRAP SMOKE ARTIFACT"
    warn "($BOOTSTRAP_HINT). That is the 5-step demo model, NOT the real"
    warn "trained classifier — its scan results are meaningless."
    warn "Pass --model /path/to/the/real/model unless you want the demo."
    warn "=================================================================="
  fi
  say "Packing model $MODEL_DIR -> model/model.tar.zst"
  mkdir -p "$STAGE/model"
  bash ./pack_model.sh "$MODEL_DIR" "$STAGE/model/model.tar.zst"
fi

# --- wheels/<arch>/cp<ver>/ ----------------------------------------------------
# pip does NOT expand an explicit --platform down the manylinux hierarchy (only
# the manylinux2014/2010 legacy aliases), so a single platform can never satisfy
# a mixed wheelhouse — e.g. numpy>=2.3 ships only manylinux_2_28 aarch64 wheels
# while tokenizers ships only manylinux_2_17. Pass the whole ladder (newest
# first) in ONE call so dependency resolution is consistent with the pins.
#
# Full-ML ordering matters: the CPU torch wheel is downloaded FIRST, then the
# requirements are resolved in a single `pip download -r` with --find-links
# into the same wheelhouse, so peft/accelerate's `torch` dependency resolves to
# the already-present local +cpu wheel (local version X.Y.Z+cpu sorts higher
# than PyPI's X.Y.Z) instead of dragging PyPI's CUDA torch plus its
# nvidia_*/triton wheel stack (~3GB) into the bundle. A hard post-check
# (check_wheelhouse_cpu_only) refuses to ship a contaminated wheelhouse.
# Failures are collected and reported at the END.
FAILED=()

platform_ladder() {
  # $1=arch -> repeated --platform flags, newest glibc tag first
  local arch="$1" v
  for v in 2_39 2_35 2_34 2_31 2_28 2_27 2_26 2_24 2_17; do
    printf -- '--platform manylinux_%s_%s ' "$v" "$arch"
  done
  printf -- '--platform manylinux2014_%s ' "$arch"
}

download_wheel() {
  # $1=spec $2=dest $3=arch $4=pyver; remaining args are extra pip flags
  local spec="$1" dest="$2" arch="$3" pyver="$4"; shift 4
  # shellcheck disable=SC2046  # ladder flags are intentionally word-split
  if python3 -m pip download "$spec" -d "$dest" \
       --only-binary=:all: \
       $(platform_ladder "$arch") \
       --python-version "$pyver" --implementation cp \
       "$@"; then
    return 0
  fi
  warn "pip download '$spec' failed (arch $arch, python $pyver, full manylinux ladder)"
  return 1
}

download_requirements() {
  # $1=reqfile $2=dest $3=arch $4=pyver — resolve the WHOLE requirements file
  # in ONE pip call (consistent resolution across the pins); --find-links
  # "$dest" lets the pre-downloaded +cpu torch satisfy the torch dependency of
  # peft/accelerate so pip never reaches for PyPI's CUDA torch.
  local reqfile="$1" dest="$2" arch="$3" pyver="$4"
  # shellcheck disable=SC2046  # ladder flags are intentionally word-split
  if python3 -m pip download -r "$reqfile" -d "$dest" \
       --only-binary=:all: \
       $(platform_ladder "$arch") \
       --python-version "$pyver" --implementation cp \
       --find-links "$dest"; then
    return 0
  fi
  warn "pip download -r '$reqfile' failed (arch $arch, python $pyver, full manylinux ladder)"
  return 1
}

check_wheelhouse_cpu_only() {
  # Hard gate for the full-ML wheelhouse: any nvidia_*/triton*/cuda_* wheel or
  # a torch wheel without '+cpu' means requirement resolution leaked to PyPI's
  # CUDA torch. FAIL the build — do NOT silently delete the offenders, since
  # the rest of the resolution was solved against the CUDA torch and deletion
  # would mask a broken wheelhouse.
  local dest="$1" arch="$2" w base
  local bad=()
  for w in "$dest"/*.whl; do
    [[ -e "$w" ]] || continue
    base="$(basename "$w")"
    case "$base" in
      nvidia_*|triton*|pytorch_triton*|cuda_*) bad+=("$base");;
      torch-*) [[ "$base" == *+cpu* ]] || bad+=("$base (torch wheel without +cpu)");;
    esac
  done
  [[ ${#bad[@]} -eq 0 ]] && return 0
  echo "[err] CUDA-contaminated wheelhouse for $arch: $dest" >&2
  printf '  - %s\n' "${bad[@]}" >&2
  echo "[err] requirement resolution pulled the CUDA torch stack instead of the bundled +cpu torch." >&2
  echo "[err] Refusing to ship it. Check CCR_TORCH_SPEC / the requirement pins and re-run." >&2
  exit 1
}

# Model-free lane: heuristic_scan/llm_explain/to_sarif are pure stdlib; the
# tree-sitter trio only improves function-boundary detection (--parser brace
# is the dependency-free fallback). Nothing else is needed.
mapfile -t TRIO_REQS < <(grep -E '^tree_sitter' "$STAGE/repo/requirements.txt")
[[ ${#TRIO_REQS[@]} -gt 0 ]] || TRIO_REQS=(tree_sitter tree_sitter_c tree_sitter_cpp)
mapfile -t FULL_REQS < <(grep -vE '^[[:space:]]*(#|$)' "$STAGE/repo/requirements.txt" | sed 's/[[:space:]]*#.*$//')

IFS=',' read -r -a ARCH_LIST <<< "$ARCHES"
for ARCH in "${ARCH_LIST[@]}"; do
  ARCH="${ARCH// /}"
  [[ -n "$ARCH" ]] || continue
  for PYVER in "${CLEAN_PYVERS[@]}"; do
    PYTAG="cp${PYVER//./}"
    DEST="$STAGE/wheels/$ARCH/$PYTAG"
    mkdir -p "$DEST"

    # The pinned ML deps need python >= 3.ML_PY_FLOOR_MINOR; below that this
    # wheelhouse carries only the tree-sitter trio and is marked model-free so
    # the installer degrades gracefully instead of failing pip install.
    PY_MINOR="${PYVER#3.}"
    KIND="full"
    if [[ "$SKIP_ML" == "1" ]]; then
      KIND="model-free"
    elif (( PY_MINOR < ML_PY_FLOOR_MINOR )); then
      KIND="model-free"
      warn "python $PYVER is below the ML floor (3.$ML_PY_FLOOR_MINOR needed by the numpy pin):"
      warn "wheels/$ARCH/$PYTAG will be a model-free (tree-sitter only) wheelhouse."
    fi
    echo "$KIND" > "$DEST/WHEELHOUSE_KIND"

    if [[ "$KIND" == "model-free" ]]; then
      say "Wheelhouse for $ARCH / $PYTAG (model-free: ${#TRIO_REQS[@]} requirement(s))"
      for spec in "${TRIO_REQS[@]}"; do
        if ! download_wheel "$spec" "$DEST" "$ARCH" "$PYVER"; then
          warn "FAILED wheel: '$spec' ($ARCH/$PYTAG) — continuing with the rest"
          FAILED+=("$ARCH/$PYTAG: $spec")
        fi
      done
      continue
    fi

    # Full-ML lane, step 1: CPU torch FIRST so the requirements resolution
    # below finds it locally (see the ordering note above).
    TORCH_SPEC="${CCR_TORCH_SPEC:-torch}"
    say "CPU torch for $ARCH / $PYTAG ($TORCH_SPEC from download.pytorch.org/whl/cpu)"
    if ! download_wheel "$TORCH_SPEC" "$DEST" "$ARCH" "$PYVER" --index-url "https://download.pytorch.org/whl/cpu"; then
      warn "FAILED wheel: '$TORCH_SPEC' cpu channel ($ARCH/$PYTAG)"
      FAILED+=("$ARCH/$PYTAG: $TORCH_SPEC (cpu channel)")
    fi

    # Step 2: one pip call for the whole requirements file (consistent
    # resolution); per-spec fallback (also --find-links'd to the local +cpu
    # torch) only if the single call fails, so one missing wheel does not lose
    # the rest of the wheelhouse.
    say "Wheelhouse for $ARCH / $PYTAG (single-resolution pip download -r requirements.txt)"
    if ! download_requirements "$STAGE/repo/requirements.txt" "$DEST" "$ARCH" "$PYVER"; then
      warn "single-call resolution failed — falling back to per-requirement downloads"
      for spec in "${FULL_REQS[@]}"; do
        if ! download_wheel "$spec" "$DEST" "$ARCH" "$PYVER" --find-links "$DEST"; then
          warn "FAILED wheel: '$spec' ($ARCH/$PYTAG) — continuing with the rest"
          FAILED+=("$ARCH/$PYTAG: $spec")
        fi
      done
    fi

    # Step 3: hard gate — no CUDA stack, no non-+cpu torch (fails the build).
    check_wheelhouse_cpu_only "$DEST" "$ARCH"
  done
done

# --- optional Ollama blob store -------------------------------------------------
if [[ -n "$OLLAMA_MODELS_DIR" ]]; then
  [[ -d "$OLLAMA_MODELS_DIR" ]] || die "--ollama-models dir not found: $OLLAMA_MODELS_DIR"
  say "Copying Ollama model blobs from $OLLAMA_MODELS_DIR"
  mkdir -p "$STAGE/ollama/models"
  cp -a "$OLLAMA_MODELS_DIR"/. "$STAGE/ollama/models/"
fi

# --- bundle_manifest.json (written BEFORE SHA256SUMS so it is covered) ----------
say "Writing bundle_manifest.json"
MODEL_SHA=""
[[ -f "$STAGE/model/model.tar.zst" ]] && MODEL_SHA="$(sha256sum "$STAGE/model/model.tar.zst" | awk '{print $1}')"
WHEEL_COUNTS=""
for ARCH in "${ARCH_LIST[@]}"; do
  ARCH="${ARCH// /}"
  [[ -n "$ARCH" ]] || continue
  for PYVER in "${CLEAN_PYVERS[@]}"; do
    PYTAG="cp${PYVER//./}"
    D="$STAGE/wheels/$ARCH/$PYTAG"
    N="$(find "$D" -name '*.whl' 2>/dev/null | wc -l | tr -d ' ')"
    K="$(cat "$D/WHEELHOUSE_KIND" 2>/dev/null || echo full)"
    WHEEL_COUNTS+="${ARCH}/${PYTAG}:${K}=${N} "
  done
done
FAILED_JOINED=""
[[ ${#FAILED[@]} -gt 0 ]] && FAILED_JOINED="$(printf '%s\n' "${FAILED[@]}")"
CCR_MANIFEST_VERSION="$VERSION" CCR_MANIFEST_PYVERS="$PYVERS" \
CCR_MANIFEST_SKIP_ML="$SKIP_ML" CCR_MANIFEST_MODEL_SHA="$MODEL_SHA" \
CCR_MANIFEST_WHEELS="$WHEEL_COUNTS" CCR_MANIFEST_FAILED="$FAILED_JOINED" \
python3 - "$STAGE/bundle_manifest.json" <<'PY'
import json, os, sys, time
wheelhouses = []
arches = set()
for tok in os.environ.get("CCR_MANIFEST_WHEELS", "").split():
    where, _, n = tok.partition("=")
    loc, _, kind = where.partition(":")
    arch, _, pytag = loc.partition("/")
    arches.add(arch)
    wheelhouses.append({"arch": arch, "python": pytag, "kind": kind or "full",
                        "wheels": int(n or 0)})
manifest = {
    "version": os.environ["CCR_MANIFEST_VERSION"],
    "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "arches": sorted(arches),
    "pythons": [p.strip() for p in os.environ["CCR_MANIFEST_PYVERS"].split(",") if p.strip()],
    "skip_ml": os.environ["CCR_MANIFEST_SKIP_ML"] == "1",
    "wheelhouses": wheelhouses,
    "model_sha256": os.environ.get("CCR_MANIFEST_MODEL_SHA") or None,
    "failed_wheels": [l for l in os.environ.get("CCR_MANIFEST_FAILED", "").splitlines() if l],
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)
    f.write("\n")
PY

# --- SHA256SUMS over every file (relative paths; verified by the installer) -----
say "Writing SHA256SUMS"
( cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS )

# --- outer tarball ----------------------------------------------------------------
if command -v zstd >/dev/null 2>&1; then
  TARBALL="$OUT_DIR/${BUNDLE}.tar.zst"
  say "Creating $TARBALL (zstd)"
  tar -I zstd -cf "$TARBALL" -C "$OUT_DIR" "$BUNDLE"
else
  TARBALL="$OUT_DIR/${BUNDLE}.tar"
  say "Creating $TARBALL (no zstd on this machine; plain tar)"
  tar -cf "$TARBALL" -C "$OUT_DIR" "$BUNDLE"
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo >&2
  echo "[err] bundle is INCOMPLETE — required wheel downloads failed:" >&2
  printf '  - %s\n' "${FAILED[@]}" >&2
  echo "[err] fix connectivity / pins (CCR_TORCH_SPEC?) and re-run." >&2
  echo "[err] partial bundle left at: $STAGE and $TARBALL" >&2
  exit 1
fi
say "Bundle ready: $TARBALL"
say "Next: copy it to the offline machine, extract, and run ./install_offline.sh"
