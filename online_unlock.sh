#!/usr/bin/env bash
set +u
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true
unset HF_DATASETS_OFFLINE || true
set -u

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
