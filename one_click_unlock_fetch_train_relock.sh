#!/usr/bin/env bash
set -euo pipefail
VENV="${VENV:-.venv}"; PY="${PYTHON:-python3}"; USE_GPU="${USE_GPU:-1}"
# BATCH is intentionally NOT defaulted here: when unset, the per-machine profile
# batch (profiles.json) is used at train time; set BATCH to override the profile.
EPOCHS="${EPOCHS:-3}"; OUT="${OUT:-vuln-model}"
HARD_RELOCK="${HARD_RELOCK:-0}"; HF_CACHE="${HF_CACHE:-$PWD/hf_cache}"
# HF_REVISION pins a specific commit sha / tag for the fetched HF models
# (supply-chain integrity). Default empty = current upstream HEAD (documented
# risk; pin a sha for reproducible/audited builds). Exported so child scripts
# (fetch_graphcodebert.sh) and the inline downloader below pick it up.
export HF_REVISION="${HF_REVISION:-}"
# HARD_RELOCK_CONFIRM must be set to "yes" to allow the host-firewall relock,
# which mutates iptables and could otherwise be triggered accidentally.
HARD_RELOCK_CONFIRM="${HARD_RELOCK_CONFIRM:-}"
say(){ printf "\n\033[1;36m[one-click]\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
SUDO="$(command -v sudo || true)"
say "WSL2 host: $(uname -a)"
if command -v apt-get >/dev/null; then $SUDO apt-get update -y && $SUDO apt-get install -y git git-lfs python3 python3-venv python3-pip build-essential; fi
git lfs install || true
[[ -d "$VENV" ]] || $PY -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
# Arch-aware torch install (shared helper). aarch64 -> cu130 NIGHTLY torch-only;
# x86_64 -> channel matching the driver's CUDA. FAILS SOFT so a missing aarch64
# wheel doesn't abort the run before we even reach training. torch only —
# torchvision/torchaudio are unused and torchvision >=0.25 breaks datasets.
# shellcheck source=lib_torch_install.sh
source "$(dirname "$0")/lib_torch_install.sh"
if ! ccr_install_torch; then
  warn "torch install failed for this arch/CUDA; resolve via the per-machine runbook"
  warn "(SETUP_SPARK.md / SETUP_4090.md) and re-run, or run ./install_torch_nightly_for_new_gpu.sh."
  exit 1
fi
pip install --no-cache-dir -r requirements.txt
export HF_HOME="$HF_CACHE"; export TRANSFORMERS_CACHE="$HF_CACHE"; export HF_DATASETS_CACHE="$HF_CACHE"; mkdir -p "$HF_CACHE"
# Dataset: DATA=bigvul (default) fetches BigVul from HF and builds real
# train/val/test splits; DATA=bootstrap writes the tiny smoke-test set instead.
# MIN_TRAIN_ROWS guards the real path: merge_and_split.py HARD-FAILS (no silent
# bootstrap) if BigVul didn't yield at least this many train rows.
DATA="${DATA:-bigvul}"
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-20000}"
if [[ "$DATA" == "bigvul" ]]; then
  say "Fetching BigVul -> data/ingest"
  # Hard-fail on fetch error: a failed fetch must NOT silently degrade to a
  # 12-row bootstrap set and train a worthless model. The operator re-runs the
  # online provisioning window (see SETUP_4090.md / SETUP_SPARK.md).
  python scripts/py/fetch_bigvul_hf.py
  # No --allow-bootstrap here: merge_and_split exits non-zero (caught by -e) if
  # ingest is empty or the train split is below MIN_TRAIN_ROWS.
  python scripts/py/merge_and_split.py --min-train-rows "$MIN_TRAIN_ROWS"
else
  say "DATA=$DATA -> writing tiny bootstrap (smoke-test) dataset"
  bash ./bootstrap_data.sh
fi
bash ./fetch_graphcodebert.sh || true
python - "$HF_REVISION" <<'PY' || true
from huggingface_hub import snapshot_download
import os, sys
revision = (sys.argv[1] if len(sys.argv) > 1 else "") or None
dest=os.path.join("models","graphcodebert-base")
if not os.path.exists(dest) or not os.listdir(dest):
    snapshot_download(repo_id="microsoft/graphcodebert-base", local_dir=dest, revision=revision, local_dir_use_symlinks=False, resume_download=True)
PY
# Per-machine training config from the detected profile (profiles.json):
# 4090 -> batch 64 + --fp16, Spark -> batch 128 + --bf16, cpu -> batch 8 + fp32.
# BATCH (env, default unset) overrides the profile batch; CCR_PRECISION_FLAG
# overrides the precision flag.
eval "$(python profiles.py --training-env)"
TRAIN_BATCH="${BATCH:-$CCR_BATCH}"
PRECISION_FLAG="${CCR_PRECISION_FLAG:-}"
say "Train -> $OUT (profile=$CCR_PROFILE batch=$TRAIN_BATCH precision='${PRECISION_FLAG:-fp32}')"
python model/train_vuln_model.py --base ./models/graphcodebert-base --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl --out "$OUT" --epochs "$EPOCHS" --batch "$TRAIN_BATCH" --class_weight $PRECISION_FLAG

# --- eval: prove the trained model carries signal before relocking ----------
# Set SKIP_EVAL=1 to skip (e.g. CI). ROC-AUC well below ~0.6 means no signal.
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  say "Eval -> held-out test split ($OUT)"
  python evaluate_model.py --model "$OUT" --test data/test.jsonl || warn "eval reported an issue; inspect metrics before trusting scans"
fi

# Relock: offline env flags + optional iptables
cat > .env.locked <<EOF
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
EOF
if ! grep -q HF_HUB_OFFLINE "$VENV/bin/activate"; then echo -e '\n# relock\n test -f "$VIRTUAL_ENV/../.env.locked" && set -a && . "$VIRTUAL_ENV/../.env.locked" && set +a' >> "$VENV/bin/activate"; fi
# Optional host-firewall relock. The previous version ran an UNCONDITIONAL
# `iptables -P OUTPUT DROP`, which severs the SSH session you are running this
# over and has no revert. We now:
#   - require an explicit confirmation (HARD_RELOCK_CONFIRM=yes),
#   - ACCEPT loopback + ESTABLISHED,RELATED *before* dropping new egress, so the
#     current SSH session (an established connection) is never cut,
#   - install scoped rules (not a default-policy flip) into a dedicated chain,
#   - leave a paired revert in online_unlock.sh.
# True air-gap should still be enforced at the OS/hypervisor layer; this is a
# convenience guard, not a security boundary.
if [[ "$HARD_RELOCK" == "1" ]]; then
  if [[ "$HARD_RELOCK_CONFIRM" != "yes" ]]; then
    warn "HARD_RELOCK requested but HARD_RELOCK_CONFIRM!=yes; skipping host-firewall mutation."
    warn "Re-run with HARD_RELOCK=1 HARD_RELOCK_CONFIRM=yes to apply (revert: bash ./online_unlock.sh)."
  elif [[ -z "$SUDO" ]]; then
    warn "HARD_RELOCK requested but sudo is unavailable; skipping host-firewall mutation."
  else
    say "Applying scoped egress lock (loopback + established preserved; revert: bash ./online_unlock.sh)"
    for IPT in iptables ip6tables; do
      command -v "$IPT" >/dev/null 2>&1 || continue
      $SUDO "$IPT" -N CCR_EGRESS_LOCK 2>/dev/null || $SUDO "$IPT" -F CCR_EGRESS_LOCK || true
      $SUDO "$IPT" -A CCR_EGRESS_LOCK -o lo -j ACCEPT || true
      $SUDO "$IPT" -A CCR_EGRESS_LOCK -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT || true
      $SUDO "$IPT" -A CCR_EGRESS_LOCK -j DROP || true
      # Jump to our chain at the head of OUTPUT (idempotent: remove any prior jump first)
      $SUDO "$IPT" -D OUTPUT -j CCR_EGRESS_LOCK 2>/dev/null || true
      $SUDO "$IPT" -I OUTPUT 1 -j CCR_EGRESS_LOCK || true
    done
  fi
fi
say "Done (offline ready)."
