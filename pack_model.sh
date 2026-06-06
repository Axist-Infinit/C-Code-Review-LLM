#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${1:-vuln-model}"; OUT="${2:-trained_model.tar.zst}"
tar -I 'zstd -19' -cf "$OUT" "$MODEL_DIR"; echo "[ok] $OUT"
