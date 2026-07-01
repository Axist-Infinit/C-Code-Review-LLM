#!/usr/bin/env bash
# Re-enable online mode after a relock. MUST be SOURCED to affect your shell:
#
#   source ./online_unlock.sh
#
# (Running `bash ./online_unlock.sh` unsets the env vars only in a throwaway
# child shell.) No set -e/-u here: this file is sourced into the caller's
# shell and must not mutate its options; `unset` never fails anyway.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE 2>/dev/null || true

if [[ "${BASH_SOURCE[0]:-}" == "${0:-}" ]]; then
  echo "[warn] online_unlock.sh is being EXECUTED, not sourced: the HF_*_OFFLINE" >&2
  echo "[warn] unsets above cannot reach your current shell." >&2
  echo "[warn] Run:  source ./online_unlock.sh" >&2
fi

# Stop future venv activations from re-locking: the activate hook installed by
# one_click_unlock_fetch_train_relock.sh sources .env.locked when the file
# exists, so disable the file (rename, not delete: `source ./offline_lockdown.sh`
# or re-running the one-click restores lockdown when wanted).
if [[ -f .env.locked ]]; then
  mv -f .env.locked .env.locked.disabled
  echo "[OK] .env.locked -> .env.locked.disabled (venv activation no longer relocks)."
fi

# Revert the scoped egress lock installed by one_click_unlock_fetch_train_relock.sh
# (HARD_RELOCK=1 HARD_RELOCK_CONFIRM=yes). Removes the OUTPUT jump and flushes/deletes
# the dedicated CCR_EGRESS_LOCK chain. No-op if the chain was never created.
SUDO="$(command -v sudo || true)"
if [[ -n "$SUDO" ]]; then
  for IPT in iptables ip6tables; do
    command -v "$IPT" >/dev/null 2>&1 || continue
    if $SUDO "$IPT" -L CCR_EGRESS_LOCK -n >/dev/null 2>&1; then
      $SUDO "$IPT" -D OUTPUT -j CCR_EGRESS_LOCK 2>/dev/null || true
      $SUDO "$IPT" -F CCR_EGRESS_LOCK 2>/dev/null || true
      $SUDO "$IPT" -X CCR_EGRESS_LOCK 2>/dev/null || true
    fi
  done
fi

echo "[OK] Online mode enabled (offline env vars unset; egress lock reverted if present)."
