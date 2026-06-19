#!/usr/bin/env bash
set -euo pipefail
ARCHIVE="${1:-trained_model.tar.zst}"; DEST="${2:-.}"

[[ -f "$ARCHIVE" ]] || { echo "[err] archive not found: $ARCHIVE" >&2; exit 1; }

# --- Path-traversal / symlink guard -------------------------------------------
# A malicious .tar.zst could carry absolute paths ("/etc/..."), parent-dir
# escapes ("../../"), or symlinks/hardlinks that, once extracted, write or point
# outside DEST. Reject any such entry BEFORE extracting anything.
#
# We list with --absolute-names so tar does NOT silently rewrite the member
# names (without it, tar strips a leading '/' or '../' for display and our
# checks would never see them).
echo "[info] vetting archive entries in $ARCHIVE ..."

# 1) Reject absolute paths and '..' components (names-only listing, raw).
while IFS= read -r path_field; do
  [[ -z "$path_field" ]] && continue
  case "$path_field" in
    /*) echo "[err] refusing archive: absolute path entry: $path_field" >&2; exit 1;;
  esac
  case "/$path_field/" in
    */../*) echo "[err] refusing archive: parent-dir ('..') entry: $path_field" >&2; exit 1;;
  esac
done < <(tar -I zstd --absolute-names -tf "$ARCHIVE")

# 2) Reject symlink ('l') and hardlink ('h') entries (verbose listing: the
#    entry type is the first character of each line).
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  case "${line:0:1}" in
    l|h) echo "[err] refusing archive: contains a link entry: $line" >&2; exit 1;;
  esac
done < <(tar -I zstd --absolute-names -tvf "$ARCHIVE")

# --- Safe extraction ----------------------------------------------------------
# Extract into a fresh temp dir first, then move into place. This keeps a
# partially/maliciously crafted archive from clobbering DEST mid-extract.
# Belt-and-suspenders: do NOT pass --absolute-names here, so even if a check
# were bypassed tar still strips leading '/' and refuses '..' members.
mkdir -p "$DEST"
TMP="$(mktemp -d "${DEST%/}/.unpack.XXXXXX")"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

tar -I zstd --no-same-owner -xf "$ARCHIVE" -C "$TMP"

# Move extracted top-level entries into DEST.
shopt -s dotglob nullglob
for item in "$TMP"/*; do
  base="$(basename "$item")"
  target="$DEST/$base"
  [[ -e "$target" ]] && rm -rf "$target"
  mv "$item" "$DEST"/
done
shopt -u dotglob nullglob

echo "[ok] unpacked to $DEST"
