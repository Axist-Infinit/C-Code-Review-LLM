#!/usr/bin/env bash
set -euo pipefail
VENV="${VENV:-.venv}"; PY="${PYTHON:-python3}"; USE_GPU="${USE_GPU:-1}"
EPOCHS="${EPOCHS:-3}"; BATCH="${BATCH:-16}"; OUT="${OUT:-vuln-model}"
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
INDEX_URL=""
if [[ "$USE_GPU" == "1" ]] && command -v nvidia-smi >/dev/null; then
  # match both "CUDA Version: 12.4" and newer "CUDA UMD Version: 13.3" formats
  CUDA_VER=$(nvidia-smi | grep -oE "CUDA[ A-Za-z]* Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" | head -1 || true)
  case "$CUDA_VER" in
    13.*)        INDEX_URL="https://download.pytorch.org/whl/cu130" ;;
    12.8*|12.9*) INDEX_URL="https://download.pytorch.org/whl/cu128" ;;
    12.5*|12.6*|12.7*) INDEX_URL="https://download.pytorch.org/whl/cu126" ;;
    12.4*)       INDEX_URL="https://download.pytorch.org/whl/cu124" ;;
    12.1*|12.2*|12.3*) INDEX_URL="https://download.pytorch.org/whl/cu121" ;;
    11.*)        INDEX_URL="https://download.pytorch.org/whl/cu118" ;;
    *) [[ -n "$CUDA_VER" ]] && INDEX_URL="https://download.pytorch.org/whl/cu128" ;;
  esac
  # DGX Spark (aarch64 Grace Blackwell, sm_121) needs cu130+ aarch64 wheels
  if [[ "$(uname -m)" == "aarch64" && -n "$INDEX_URL" ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cu130"
  fi
fi
# torch only — torchvision/torchaudio are unused and torchvision >=0.25 breaks datasets' torch formatter
if [[ -n "$INDEX_URL" ]]; then pip install --no-cache-dir --index-url "$INDEX_URL" torch; else pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch; fi
pip install --no-cache-dir -r requirements.txt
export HF_HOME="$HF_CACHE"; export TRANSFORMERS_CACHE="$HF_CACHE"; export HF_DATASETS_CACHE="$HF_CACHE"; mkdir -p "$HF_CACHE"
# Dataset: DATA=bigvul (default) fetches BigVul from HF and builds real
# train/val/test splits; DATA=bootstrap writes the tiny smoke-test set instead.
DATA="${DATA:-bigvul}"
if [[ "$DATA" == "bigvul" ]]; then
  say "Fetching BigVul -> data/ingest"
  python scripts/py/fetch_bigvul_hf.py || warn "BigVul fetch failed; merge will fall back to bootstrap data"
  python scripts/py/merge_and_split.py
else
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
say "Train -> $OUT"
python model/train_vuln_model.py --base ./models/graphcodebert-base --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl --out "$OUT" --epochs "$EPOCHS" --batch "$BATCH" --class_weight
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
