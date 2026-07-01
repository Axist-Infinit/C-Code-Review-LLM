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

Each finding carries "match_lines": the absolute 1-based lines where patterns
actually matched, so SARIF regions and PR comments can anchor to the call site
instead of the whole function. Top-level lines no function span claims (macros,
globals) are scanned too. Noise controls: `ccr-ignore[: rule1,rule2]` on a line
suppresses matches on that line; --exclude GLOB skips files; --min-severity
drops findings below a severity floor.
"""
import argparse
import fnmatch
import json
import os
import re

from local_vuln_scanner import list_sources, extract_functions, lang_for_path
from explain_findings import (explain_snippet_detailed, cwe_for_patterns,
                              severity_for_patterns)

# Whole functions, never windowed: one finding per function keeps CI output clean
# (the snippet is only used for pattern matching + explanation, never embedded).
_NO_WINDOW = 10 ** 9

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

# Inline suppression: a match on a line containing `ccr-ignore` is dropped;
# `ccr-ignore: rule1,rule2` limits the suppression to those pattern ids.
_IGNORE_RE = re.compile(r"ccr-ignore(?:\s*:\s*([\w\s,-]+))?")


def _suppressed_rules(line):
    """None when the line has no ccr-ignore; empty set = every rule suppressed."""
    m = _IGNORE_RE.search(line)
    if m is None:
        return None
    if m.group(1):
        return {r.strip() for r in m.group(1).split(",") if r.strip()}
    return set()


def _span_finding(a, b, lines):
    """One merged finding for the 0-based [a, b) span of `lines`, or None."""
    snippet = "\n".join(lines)
    _expl, _recs, matched, matches = explain_snippet_detailed(snippet)
    if not matched:
        return None
    kept = []
    for pid, ln in matches:
        rules = _suppressed_rules(lines[ln - 1])
        if rules is not None and (not rules or pid in rules):
            continue
        kept.append((pid, a + ln))  # snippet-relative -> absolute 1-based
    if not kept:
        return None  # every match line was ccr-ignore'd
    matched_ids = {pid for pid, _ in kept}
    return {
        "start_line": a + 1,
        "end_line": b,
        "score": 1.0,  # deterministic regex hit
        "snippet": snippet,
        "matched_patterns": sorted(matched_ids),
        "cwe": cwe_for_patterns(matched_ids),
        "match_lines": sorted({ln for _, ln in kept}),
    }


def _gap_spans(lines, covered):
    """0-based [a, b) spans of contiguous lines no function span claimed,
    trimmed of blank edges — the top-level macros/globals between functions."""
    spans = []
    i, n = 0, len(lines)
    while i < n:
        if i in covered:
            i += 1
            continue
        j = i
        while j < n and j not in covered:
            j += 1
        a, b = i, j
        while a < b and not lines[a].strip():
            a += 1
        while b > a and not lines[b - 1].strip():
            b -= 1
        if a < b:
            spans.append((a, b))
        i = j
    return spans


def scan_code(code, lang="c", parser=None):
    """Return findings for regions of `code` that match an unsafe-API pattern.

    Each finding mirrors the classifier schema (start_line/end_line/score/
    snippet) plus the matched pattern ids, CWE, and the absolute match_lines.
    Function spans are scanned first; the leftover top-level lines (macros,
    globals) are grouped into contiguous spans and scanned with the same
    pattern set, so nothing outside a function is invisible. Files with no
    detectable functions are scanned as a single span.
    """
    spans = extract_functions(code, max_lines=_NO_WINDOW, parser=parser, lang=lang)
    all_lines = code.splitlines()
    if spans is None:
        spans = [(0, len(all_lines), all_lines)] if all_lines else []

    covered = set()
    for a, b, _lines in spans:
        covered.update(range(a, b))

    findings = []
    for a, b, lines in spans:
        f = _span_finding(a, b, lines)
        if f:
            findings.append(f)
    # Residual coverage: uncovered lines can't double-report — they are, by
    # construction, outside every function span scanned above.
    for a, b in _gap_spans(all_lines, covered):
        f = _span_finding(a, b, all_lines[a:b])
        if f:
            findings.append(f)

    findings.sort(key=lambda f: f["start_line"])
    return findings


def scan_path(src, parser=None, lang_override=None, exclude=(), min_severity="info"):
    """Scan a file or directory, returning findings tagged with their file path.

    `exclude` is a list of fnmatch globs applied to each file's path relative
    to `src`; `min_severity` drops findings whose primary severity ranks below
    the floor (default "info" keeps everything).
    """
    min_rank = _SEVERITY_RANK.get(min_severity, 0)
    root = src if os.path.isdir(src) else (os.path.dirname(src) or ".")
    all_findings = []
    for path in list_sources(src):
        rel = os.path.relpath(path, root)
        if any(fnmatch.fnmatch(rel, g) for g in exclude):
            continue
        with open(path, "r", errors="ignore", encoding="utf-8") as fh:
            code = fh.read()
        lang = lang_override or lang_for_path(path)
        for f in scan_code(code, lang=lang, parser=parser):
            if _SEVERITY_RANK.get(severity_for_patterns(f["matched_patterns"]), 0) < min_rank:
                continue
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
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="Skip files whose path relative to SRC matches this "
                         "fnmatch glob (repeatable)")
    ap.add_argument("--min-severity", choices=["info", "low", "medium", "high", "critical"],
                    default="info",
                    help="Drop findings whose primary severity ranks below this floor")
    args = ap.parse_args()

    lang_override = args.lang if args.lang in ("c", "cpp") else None
    findings = scan_path(args.src, parser=args.parser, lang_override=lang_override,
                         exclude=args.exclude, min_severity=args.min_severity)

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "classifier_findings.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"threshold": 1.0, "profile": "heuristic", "findings": findings},
                  f, indent=2)
    print(f"[OK] {len(findings)} heuristic findings -> {out_json}")


if __name__ == "__main__":
    main()
