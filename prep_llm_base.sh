#!/usr/bin/env bash
set -euo pipefail
BASE_REPO="${1:-bigcode/starcoder2-3b}"
DEST="${2:-./llm-base}"
# HF_REVISION pins a specific commit sha / tag for supply-chain integrity.
# Default empty = whatever HEAD the repo currently points at (documented risk:
# an upstream repo can change under you; pin a sha for reproducible/audited builds).
HF_REVISION="${HF_REVISION:-}"
echo "[INFO] Downloading $BASE_REPO (revision='${HF_REVISION:-<default>}') to $DEST ..."
# NOTE: pass args AFTER the heredoc terminator. With `python - <<'PY'` the script
# is read from stdin, so sys.argv[0] is '-' and there are no positional args;
# the args below become sys.argv[1:].
python - "$BASE_REPO" "$DEST" "$HF_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]; dest = sys.argv[2]
revision = sys.argv[3] or None
p = snapshot_download(repo_id=repo, local_dir=dest, revision=revision, resume_download=True)
print("[OK] Downloaded to", p)
PY
echo "[OK] LLM base ready at $DEST"
