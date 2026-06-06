#!/usr/bin/env bash
set -euo pipefail
ARCHIVE="${1:-trained_model.tar.zst}"; DEST="${2:-.}"
tar -I zstd -xf "$ARCHIVE" -C "$DEST"; echo "[ok] unpacked to $DEST"
