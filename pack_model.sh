#!/usr/bin/env bash
# Pack a trained model dir for transfer. Training-only state is excluded —
# checkpoint-*/ (optimizer shards, duplicate weights) and training_args.bin (a
# pickle) — because inference needs only config/tokenizer/safetensors/
# inference.json. Alongside the tar we emit <OUT>.manifest.json (archive
# sha256, per-file sha256 list, config.json fingerprint) which unpack_model.sh
# verifies before extracting.
set -euo pipefail
MODEL_DIR="${1:-vuln-model}"; OUT="${2:-trained_model.tar.zst}"

[[ -d "$MODEL_DIR" ]] || { echo "[err] model dir not found: $MODEL_DIR" >&2; exit 1; }
command -v zstd >/dev/null 2>&1 || { echo "[err] zstd not found (apt-get install zstd)" >&2; exit 1; }

# Tar from the parent dir with the basename so the archive always carries ONE
# relative top-level dir (an absolute --model path would otherwise embed
# absolute member names, which unpack_model.sh rightly refuses).
PARENT="$(cd "$(dirname "$MODEL_DIR")" && pwd)"
BASE="$(basename "$MODEL_DIR")"
tar -I 'zstd -19' --exclude='checkpoint-*' --exclude='training_args.bin' \
    -C "$PARENT" -cf "$OUT" "$BASE"

python3 - "$OUT" "$PARENT" "$BASE" <<'PY'
import hashlib, json, os, sys, time

out, parent, base = sys.argv[1], sys.argv[2], sys.argv[3]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


root = os.path.join(parent, base)
files = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if not d.startswith("checkpoint-")]
    for name in filenames:
        if name == "training_args.bin":
            continue
        p = os.path.join(dirpath, name)
        rel = os.path.join(base, os.path.relpath(p, root))
        files.append({"path": rel.replace(os.sep, "/"), "sha256": sha256(p)})
files.sort(key=lambda e: e["path"])

cfg = os.path.join(root, "config.json")
manifest = {
    "archive": os.path.basename(out),
    "archive_sha256": sha256(out),
    "model_dir": base,
    "config_sha256": sha256(cfg) if os.path.isfile(cfg) else None,
    "files": files,
    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(out + ".manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)
    f.write("\n")
PY

echo "[ok] $OUT (+ ${OUT}.manifest.json; checkpoints/training_args.bin excluded)"
