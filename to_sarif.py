#!/usr/bin/env python3
"""Convert pipeline findings (llm_findings.json or classifier_findings.json)
to SARIF 2.1.0 for CI / IDE / GitHub code scanning integration.

Usage:
  python to_sarif.py scan_out/llm_findings.json -o scan_out/findings.sarif
"""
import argparse
import json
import os

SEVERITY_TO_LEVEL = {
    "critical": "error", "high": "error",
    "medium": "warning", "low": "note", "info": "note",
}


def finding_to_result(e):
    sev = str(e.get("severity", "")).lower()
    level = SEVERITY_TO_LEVEL.get(sev, "warning")
    rule_id = e.get("cwe") or (e.get("matched_patterns") or ["ml-classifier"])[0] or "ml-classifier"
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

    result = {
        "ruleId": str(rule_id),
        "level": level,
        "message": {"text": msg},
        "locations": [{
            "physicalLocation": {
                "artifactLocation": {"uri": e.get("file", "").replace(os.sep, "/")},
                "region": {
                    "startLine": int(e.get("start_line") or 1),
                    "endLine": int(e.get("end_line") or e.get("start_line") or 1),
                },
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

    rules = {}
    results = [finding_to_result(e) for e in entries]
    for r in results:
        rid = r["ruleId"]
        if rid not in rules:
            rules[rid] = {"id": rid, "shortDescription": {"text": rid}}

    sarif = {
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
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sarif, f, indent=2)
    print(f"[OK] {len(results)} results -> {args.out}")


if __name__ == "__main__":
    main()
