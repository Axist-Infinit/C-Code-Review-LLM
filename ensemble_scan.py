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

from local_vuln_scanner import lang_for_path, list_sources  # torch-free helpers

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
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("[WARN] cppcheck timed out after 600s; skipping")
        return []
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
    try:
        out = subprocess.run(["flawfinder", "--csv", src],
                             capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("[WARN] flawfinder timed out after 600s; skipping")
        return []
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


def split_files_by_lang(files):
    """Group source files for per-language clang-tidy invocations. Headers go
    with the C++ group (the C++ frontend parses C headers too)."""
    groups = {"c": [], "cpp": []}
    for f in files:
        if os.path.splitext(f)[1].lower() == ".h":
            groups["cpp"].append(f)
        else:
            groups[lang_for_path(f, default="c")].append(f)
    return groups


def run_clang_tidy(src, checks=DEFAULT_CLANG_TIDY_CHECKS):
    if not shutil.which("clang-tidy"):
        print("[WARN] clang-tidy not installed (apt install clang-tidy); skipping")
        return []
    files = list_sources(src)
    if not files:
        return []
    groups = split_files_by_lang(files)
    findings = []
    # One invocation per language so .c files are not force-fed -std=c++17
    # (clang rejects that and produces no analysis for them).
    for lang, std in (("c", "c11"), ("cpp", "c++17")):
        group = groups[lang]
        if not group:
            continue
        # `-- -std=...` passes compiler flags directly, so no compile_commands.json
        # is required for a loose-files pass.
        cmd = ["clang-tidy", "--quiet", f"--checks={checks}", *group, "--", f"-std={std}"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print(f"[WARN] clang-tidy ({std}) timed out after 600s; skipping")
            continue
        findings += parse_clang_tidy_output(out.stdout)
    return findings


def load_ml(path):
    if not path or not os.path.exists(path):
        return []
    data = json.load(open(path, "r", encoding="utf-8"))
    entries = data.get("explanations") or data.get("findings") or []
    out = []
    for e in entries:
        entry = {
            "tool": "ml-classifier", "file": e.get("file"),
            "start_line": e.get("start_line"), "end_line": e.get("end_line"),
            "cwe": e.get("cwe", ""),
            # keep "rule" too so consumers keyed on tool findings' shape still work
            "rule": e.get("cwe", ""),
            "severity": e.get("severity", ""),
            "issue": e.get("issue", "") or "Flagged by vulnerability classifier",
            "score": e.get("score"),
            # Carry the structured narrative so the merged file stays usable by
            # to_sarif.py / annotate_pr.py (defaults to "" for classifier-only input).
            "what_code_does": e.get("what_code_does", ""),
            "what_could_go_wrong": e.get("what_could_go_wrong", ""),
            "vulnerability": e.get("vulnerability", ""),
        }
        # Remediation / verdict / anchoring fields must survive the merge so
        # downstream SARIF, --only-vulnerable, and PR annotation keep working.
        for key in ("fix", "is_vulnerable", "matched_patterns", "match_lines",
                    "cross_check"):
            if key in e:
                entry[key] = e[key]
        out.append(entry)
    return out


def dedup_findings(findings):
    """Drop identical (tool, file, start_line, rule) rows — overlapping tool
    invocations (e.g. a header analyzed in several TUs) produce duplicates."""
    seen = set()
    out = []
    for f in findings:
        key = (f.get("tool"), f.get("file"), f.get("start_line"), f.get("rule"))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _contains_line(f, line):
    """True when `line` falls inside finding f's span (or its matched lines)."""
    if line is None:
        return False
    if line in (f.get("match_lines") or []):
        return True
    start = f.get("start_line")
    if start is None:
        return False
    end = f.get("end_line") or start
    return start <= line <= end


def corroborate(findings):
    """Set corroborated_by on every finding: tools whose finding in the same
    file overlaps this one. Containment, not exact-line equality — ML/heuristic
    findings span whole functions while tools report the call line."""
    by_file = {}
    for f in findings:
        by_file.setdefault(f.get("file"), []).append(f)
    for group in by_file.values():
        for f in group:
            others = set()
            for g in group:
                if g is f or g.get("tool") == f.get("tool"):
                    continue
                if _contains_line(f, g.get("start_line")) \
                        or _contains_line(g, f.get("start_line")):
                    others.add(g["tool"])
            f["corroborated_by"] = sorted(others)
    return findings


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

    findings = dedup_findings(findings)
    corroborate(findings)

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
