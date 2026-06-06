#!/usr/bin/env bash
set -euo pipefail
BASE_REPO="${1:-bigcode/starcoder2-3b}"
DEST="${2:-./llm-base}"
echo "[INFO] Downloading $BASE_REPO to $DEST ..."
python - <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]; dest = sys.argv[2]
p = snapshot_download(repo_id=repo, local_dir=dest, resume_download=True)
print("[OK] Downloaded to", p)
PY
echo "[OK] LLM base ready at $DEST"
