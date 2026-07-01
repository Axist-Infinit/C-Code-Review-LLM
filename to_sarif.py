#!/usr/bin/env python3
"""Convert pipeline findings (llm_findings.json or classifier_findings.json)
to SARIF 2.1.0 for CI / IDE / GitHub code scanning integration.

Usage:
  python to_sarif.py scan_out/llm_findings.json -o scan_out/findings.sarif
"""
import argparse
import json
import os
import re

SEVERITY_TO_LEVEL = {
    # pipeline severities
    "critical": "error", "high": "error",
    "medium": "warning", "low": "note", "info": "note",
    # words emitted by ensemble tools (cppcheck/clang-tidy) passed through as-is
    "error": "error", "warning": "warning",
    "style": "note", "performance": "note", "portability": "note",
    "information": "note", "informational": "note",
}

_CWE_ID_RE = re.compile(r"^CWE-(\d+)$", re.IGNORECASE)


def _as_line(value):
    """Parse a line value (int or numeric string) to int; None/'' -> None.
    Presence must be judged with `is not None`, never truthiness: 0 is a
    real (out-of-range) value that still needs clamping, not a missing one."""
    if value is None or value == "":
        return None
    return int(value)


def _region(e):
    """Line region for a finding: the matched-pattern lines when present
    (clamped inside the function span), else the function span. Always
    guarantees endLine >= startLine >= 1 (SARIF 2.1.0 requires lines >= 1)."""
    raw_start = _as_line(e.get("start_line"))
    raw_end = _as_line(e.get("end_line"))
    start = max(raw_start if raw_start is not None else 1, 1)
    end = max(raw_end if raw_end is not None else start, start)
    match_lines = [max(int(x), 1) for x in (e.get("match_lines") or [])]
    if match_lines:
        m_start, m_end = min(match_lines), max(match_lines)
        if raw_start is not None:
            # Keep the match region inside the (already >=1) function span.
            m_start = min(max(m_start, start), end)
            m_end = min(max(m_end, start), end)
        start, end = m_start, max(m_end, m_start)
    return {"startLine": start, "endLine": end}


def finding_to_result(e):
    sev = str(e.get("severity", "")).lower()
    level = SEVERITY_TO_LEVEL.get(sev, "warning")
    rule_id = e.get("cwe") or e.get("rule") \
        or (e.get("matched_patterns") or ["ml-classifier"])[0] or "ml-classifier"
    headline = e.get("issue") or e.get("vulnerability") or e.get("explanation") \
        or "Flagged by vulnerability classifier"
    # Build a structured message: headline, then the 3-part narrative when present,
    # then the fix. Code scanning renders this multi-line text on the finding.
    parts = [str(headline)]
    for label, key in (("What the code is doing", "what_code_does"),
                       ("What could go wrong", "what_could_go_wrong"),
                       ("Vulnerability", "vulnerability")):
        val = str(e.get(key, "") or "").strip()
        if val:
            parts.append(f"{label}: {val}")
    # Only add the standalone explanation if it adds something beyond the headline.
    expl = str(e.get("explanation", "") or "").strip()
    if expl and expl != str(headline) and not e.get("what_could_go_wrong"):
        parts.append(expl)
    # The corroboration cross-check note is a trust signal — always surface it,
    # even when the structured narrative is used and the flat explanation is skipped.
    cross = str(e.get("cross_check", "") or "").strip()
    if cross and cross not in "\n\n".join(parts):
        parts.append(cross)
    fix = str(e.get("fix") or "").strip()
    if fix:
        parts.append(f"Fix: {fix}")
    msg = "\n\n".join(parts)
    score = e.get("score")

    # Normalize both separator styles so findings produced on Windows stay
    # valid on POSIX (os.sep alone is a no-op for backslashes there).
    uri = str(e.get("file") or "").replace(os.sep, "/").replace("\\", "/")
    result = {
        "ruleId": str(rule_id),
        "level": level,
        "message": {"text": msg},
        "locations": [{
            "physicalLocation": {
                "artifactLocation": {"uri": uri or "unknown"},
                "region": _region(e),
            }
        }],
    }
    props = {}
    if isinstance(score, (int, float)):
        props["classifierScore"] = round(float(score), 4)
    if e.get("backend"):
        props["explainerBackend"] = e["backend"]
    if e.get("is_vulnerable") is not None:
        props["llmJudgment"] = bool(e.get("is_vulnerable"))
    if props:
        result["properties"] = props
    return result


def rule_metadata(rule_id, entry=None):
    """SARIF reportingDescriptor for a rule id, enriched from the finding that
    introduced it: CWE ids get a MITRE helpUri, and the finding's vulnerability
    or issue text becomes the shortDescription when available."""
    rule = {"id": rule_id, "shortDescription": {"text": rule_id}}
    if entry:
        desc = str(entry.get("vulnerability") or entry.get("issue") or "").strip()
        if desc:
            rule["shortDescription"] = {"text": desc}
    m = _CWE_ID_RE.match(str(rule_id))
    if m:
        rule["helpUri"] = f"https://cwe.mitre.org/data/definitions/{m.group(1)}.html"
    return rule


def build_sarif(entries):
    """Full SARIF 2.1.0 document for a list of findings/explanations."""
    rules = {}
    results = []
    for e in entries:
        r = finding_to_result(e)
        results.append(r)
        rid = r["ruleId"]
        if rid not in rules:
            rules[rid] = rule_metadata(rid, e)
        elif rules[rid]["shortDescription"]["text"] == rid:
            # A later finding may carry the descriptive text the first one lacked.
            rules[rid] = rule_metadata(rid, e)
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "C-Code-Review-LLM",
                "informationUri": "https://github.com/local/c-code-review-llm",
                "version": "0.2.0",
                "rules": list(rules.values()),
            }},
            "results": results,
        }],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", help="llm_findings.json or classifier_findings.json")
    ap.add_argument("-o", "--out", required=True, help="Output .sarif path")
    ap.add_argument("--only-vulnerable", action="store_true",
                    help="Skip findings the LLM judged not vulnerable")
    args = ap.parse_args()

    data = json.load(open(args.inp, "r", encoding="utf-8"))
    entries = data.get("explanations") or data.get("findings") or []
    if args.only_vulnerable:
        entries = [e for e in entries if e.get("is_vulnerable") is not False]

    sarif = build_sarif(entries)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sarif, f, indent=2)
    print(f"[OK] {len(sarif['runs'][0]['results'])} results -> {args.out}")


if __name__ == "__main__":
    main()
