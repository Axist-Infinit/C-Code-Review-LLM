#!/usr/bin/env bash
# review.sh — kept as the familiar entry point; the menu now lives in ccr.sh,
# which covers install / train / tune as well as review.
#
# (The previous standalone menu here could never actually run a surface scan:
# it passed MODELS=/SAMPLES= to surface_scan.sh as expansion-produced
# assignment prefixes — `${SAMPLES:+SAMPLES="$SAMPLES"} ./surface_scan.sh` —
# which bash treats as the COMMAND NAME, so every run died with
# "SAMPLES=1: command not found" before the scanner was reached.)
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ccr.sh" "$@"
