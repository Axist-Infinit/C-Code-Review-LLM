#!/usr/bin/env python3
"""Run classic static analyzers (cppcheck, flawfinder, clang-tidy) alongside the
ML classifier output and merge everything into one findings list.

Analyzers that are not installed are skipped with a warning — the merge
still works with whatever is available. clang-tidy is the natural C++ companion
to cppcheck/flawfinder (AST-aware bugprone-*/cert-* + the security analyzer).

Usage:
  python ensemble_scan.py playground --ml scan_out/classifier_findings.json \
      -o scan_out/ensemble_findings.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess

from local_vuln_scanner import list_sources  # torch-free file lister

# Security/bug-oriented default check set. AST-matcher checks (bugprone-*/cert-*)
# work without a compilation database; the security analyzer catches insecure C
# APIs (strcpy/system/...). Override with --clang-tidy-checks.
DEFAULT_CLANG_TIDY_CHECKS = "-*,clang-analyzer-security.insecureAPI.*,bugprone-*,cert-*"

# clang-tidy diagnostic line: "<file>:<line>:<col>: <severity>: <msg> [<check>]"
_CLANG_TIDY_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<sev>warning|error):\s+(?P<msg>.*?)\s*(?:\[(?P<check>[^\]]+)\])?$"
)


def run_cppcheck(src):
    if not shutil.which("cppcheck"):
        print("[WARN] cppcheck not installed (apt install cppcheck); skipping")
        return []
    cmd = ["cppcheck", "--enable=warning,portability", "--inline-suppr",
           "--template={file}\t{line}\t{severity}\t{id}\t{message}", "--quiet", src]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    findings = []
    for line in out.stderr.splitlines():
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        f, ln, sev, rid, msg = parts
        try:
            ln = int(ln)
        except ValueError:
            continue
        findings.append({"tool": "cppcheck", "file": f, "start_line": ln, "end_line": ln,
                         "rule": rid, "severity": sev, "issue": msg})
    return findings


def run_flawfinder(src):
    if not shutil.which("flawfinder"):
        print("[WARN] flawfinder not installed (apt install flawfinder); skipping")
        return []
    out = subprocess.run(["flawfinder", "--csv", src],
                         capture_output=True, text=True, timeout=600)
    findings = []
    import csv
    import io
    for row in csv.DictReader(io.StringIO(out.stdout)):
        try:
            ln = int(row.get("Line", 0))
        except ValueError:
            continue
        level = int(row.get("Level", 0) or 0)
        sev = "high" if level >= 4 else "medium" if level >= 2 else "low"
        findings.append({
            "tool": "flawfinder", "file": row.get("File", ""),
            "start_line": ln, "end_line": ln,
            "rule": row.get("CWEs", "") or row.get("Category", ""),
            "severity": sev,
            "issue": f"{row.get('Name','')}: {row.get('Warning','')}",
        })
    return findings


def parse_clang_tidy_output(text):
    """Parse clang-tidy stdout into finding dicts. Pure (no subprocess) so it is
    unit-testable without clang-tidy installed. 'note:' continuation lines and
    source-context lines are ignored; the trailing [check-name] becomes the rule.
    """
    findings = []
    for line in text.splitlines():
        m = _CLANG_TIDY_RE.match(line)
        if not m:
            continue
        sev = "high" if m.group("sev") == "error" else "medium"
        findings.append({
            "tool": "clang-tidy",
            "file": m.group("file"),
            "start_line": int(m.group("line")),
            "end_line": int(m.group("line")),
            "rule": (m.group("check") or "clang-tidy").strip(),
            "severity": sev,
            "issue": m.group("msg").strip(),
        })
    return findings


def run_clang_tidy(src, checks=DEFAULT_CLANG_TIDY_CHECKS, std="c++17"):
    if not shutil.which("clang-tidy"):
        print("[WARN] clang-tidy not installed (apt install clang-tidy); skipping")
        return []
    files = list_sources(src)
    if not files:
        return []
    # `-- -std=...` passes compiler flags directly, so no compile_commands.json
    # is required for a loose-files pass.
    cmd = ["clang-tidy", "--quiet", f"--checks={checks}", *files, "--", f"-std={std}"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return parse_clang_tidy_output(out.stdout)


def load_ml(path):
    if not path or not os.path.exists(path):
        return []
    data = json.load(open(path, "r", encoding="utf-8"))
    entries = data.get("explanations") or data.get("findings") or []
    out = []
    for e in entries:
        out.append({
            "tool": "ml-classifier", "file": e.get("file"),
            "start_line": e.get("start_line"), "end_line": e.get("end_line"),
            "rule": e.get("cwe", ""), "severity": e.get("severity", ""),
            "issue": e.get("issue", "") or "Flagged by vulnerability classifier",
            "score": e.get("score"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Source dir/file to analyze")
    ap.add_argument("--ml", help="ML findings JSON (classifier or llm output) to merge")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--no-clang-tidy", action="store_true",
                    help="Skip clang-tidy even if installed")
    ap.add_argument("--clang-tidy-checks", default=DEFAULT_CLANG_TIDY_CHECKS,
                    help="clang-tidy --checks string (default: security/bugprone/cert)")
    args = ap.parse_args()

    findings = []
    findings += run_cppcheck(args.src)
    findings += run_flawfinder(args.src)
    if not args.no_clang_tidy:
        findings += run_clang_tidy(args.src, checks=args.clang_tidy_checks)
    findings += load_ml(args.ml)

    # group by location so agreement between tools is visible
    by_loc = {}
    for f in findings:
        key = (f["file"], f["start_line"])
        by_loc.setdefault(key, []).append(f)
    for group in by_loc.values():
        tools = sorted({g["tool"] for g in group})
        for g in group:
            g["corroborated_by"] = [t for t in tools if t != g["tool"]]

    findings.sort(key=lambda f: (str(f.get("file")), f.get("start_line") or 0))
    counts = {}
    for f in findings:
        counts[f["tool"]] = counts.get(f["tool"], 0) + 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"tools": counts, "findings": findings}, fh, indent=2)
    print(f"[OK] {len(findings)} findings ({counts}) -> {args.out}")


if __name__ == "__main__":
    main()
