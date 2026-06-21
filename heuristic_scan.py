#!/usr/bin/env python3
"""Model-free vulnerability scan: flag functions containing known-unsafe APIs.

A lightweight, dependency-light alternative to the ML classifier for CI gating
and zero-setup runs — no trained model, no torch, no Ollama. It extracts whole
functions with tree-sitter (brace fallback) and flags any whose body matches the
unsafe-API pattern set (explain_findings.PATTERNS, C and C++), emitting the same
`classifier_findings.json` schema the rest of the pipeline expects. So:

    heuristic_scan.py SRC -o OUT
    llm_explain.py OUT/classifier_findings.json --out OUT/llm_findings.json --backend heuristic
    to_sarif.py OUT/llm_findings.json -o OUT/findings.sarif

runs end-to-end on any machine and produces SARIF for GitHub code scanning.
"""
import argparse
import json
import os

from local_vuln_scanner import list_sources, extract_functions, lang_for_path
from explain_findings import explain_snippet, cwe_for_patterns

# Whole functions, never windowed: one finding per function keeps CI output clean
# (the snippet is only used for pattern matching + explanation, never embedded).
_NO_WINDOW = 10 ** 9


def scan_code(code, lang="c", parser=None):
    """Return findings for functions in `code` that match an unsafe-API pattern.

    Each finding mirrors the classifier schema (start_line/end_line/score/snippet)
    plus the matched pattern ids and CWE. Files with no detectable functions are
    scanned as a single span so globals/macros are still covered.
    """
    spans = extract_functions(code, max_lines=_NO_WINDOW, parser=parser, lang=lang)
    if spans is None:
        lines = code.splitlines()
        spans = [(0, len(lines), lines)] if lines else []
    findings = []
    for a, b, lines in spans:
        snippet = "\n".join(lines)
        _expl, _recs, matched = explain_snippet(snippet)
        if matched:
            findings.append({
                "start_line": a + 1,
                "end_line": b,
                "score": 1.0,  # deterministic regex hit
                "snippet": snippet,
                "matched_patterns": sorted(matched),
                "cwe": cwe_for_patterns(matched),
            })
    return findings


def scan_path(src, parser=None, lang_override=None):
    """Scan a file or directory, returning findings tagged with their file path."""
    all_findings = []
    for path in list_sources(src):
        with open(path, "r", errors="ignore", encoding="utf-8") as fh:
            code = fh.read()
        lang = lang_override or lang_for_path(path)
        for f in scan_code(code, lang=lang, parser=parser):
            all_findings.append({"file": path, **f})
    return all_findings


def main():
    ap = argparse.ArgumentParser(
        description="Model-free heuristic C/C++ vuln scan -> classifier_findings.json")
    ap.add_argument("src", help="Source file or directory")
    ap.add_argument("-o", "--out", default="scan_out")
    ap.add_argument("--parser", choices=["auto", "treesitter", "brace"], default="auto",
                    help="Function-boundary engine (default: tree-sitter if installed)")
    ap.add_argument("--lang", choices=["auto", "c", "cpp"], default="auto",
                    help="tree-sitter grammar: auto (per extension), or force c/cpp")
    args = ap.parse_args()

    lang_override = args.lang if args.lang in ("c", "cpp") else None
    findings = scan_path(args.src, parser=args.parser, lang_override=lang_override)

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "classifier_findings.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"threshold": 1.0, "profile": "heuristic", "findings": findings},
                  f, indent=2)
    print(f"[OK] {len(findings)} heuristic findings -> {out_json}")


if __name__ == "__main__":
    main()
