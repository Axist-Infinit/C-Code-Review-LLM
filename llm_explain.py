#!/usr/bin/env python3
"""LLM explanation layer: enrich classifier findings via a local Ollama model.

The model is selected by hardware profile (4090 -> qwen2.5-coder:14b,
DGX Spark -> qwen2.5-coder:32b). Runs fully offline once the model is pulled.
Falls back to the heuristic regex explainer when Ollama is unavailable.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from profiles import select_profile, add_profile_arg
from explain_findings import explain_snippet, score_to_confidence, write_html

SYSTEM_PROMPT = (
    "You are a C/C++ security code reviewer. You are given a code snippet that "
    "a vulnerability classifier flagged as suspicious. Analyze it and respond "
    "with ONLY a JSON object with these keys:\n"
    '  "is_vulnerable": boolean — your own judgment, disagree with the classifier if warranted\n'
    '  "issue": string — one-line summary, or "none found" if clean\n'
    '  "cwe": string — most relevant CWE id like "CWE-120", or "" if none\n'
    '  "severity": one of "critical","high","medium","low","info"\n'
    '  "explanation": string — why this is (or is not) a real problem, referencing the code\n'
    '  "fix": string — concrete remediation advice, or "" if not vulnerable\n'
    "Be precise. Do not invent issues that are not supported by the code shown."
)


def ollama_available(base_url, model, timeout=5):
    """Return (reachable, model_present)."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout) as r:
            tags = json.load(r)
        names = {m.get("name", "") for m in tags.get("models", [])}
        present = any(n == model or n.split(":")[0] == model.split(":")[0] for n in names)
        return True, present
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False, False


def ollama_chat(base_url, model, snippet, file_path, lines, num_ctx, timeout=600):
    user = (f"File: {file_path} (lines {lines})\n\n"
            f"```c\n{snippet}\n```")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    content = resp.get("message", {}).get("content", "")
    return json.loads(content)


def heuristic_entry(f):
    """Shape a heuristic explanation like an LLM one so outputs are uniform."""
    snippet = f.get("snippet", "")
    expl, recs, matched = explain_snippet(snippet)
    conf = score_to_confidence(f.get("score"), has_pattern=bool(matched))
    return {
        "backend": "heuristic",
        "is_vulnerable": bool(matched),
        "issue": expl[0] if expl else "",
        "cwe": "",
        "severity": "medium" if matched else "info",
        "explanation": " ".join(expl),
        "fix": " ".join(recs[:2]),
        "confidence": conf,
        "matched_patterns": sorted(matched),
    }


def main():
    ap = argparse.ArgumentParser(description="Explain classifier findings via local Ollama LLM")
    ap.add_argument("inp", nargs="?", help="classifier_findings.json")
    ap.add_argument("--inp", dest="inp_flag", help="Alternative to the positional input path")
    ap.add_argument("--out", required=True, help="Output JSON path (llm_findings.json)")
    ap.add_argument("--html", help="Optional HTML report path")
    ap.add_argument("--model", default=None, help="Override the profile's Ollama model tag")
    ap.add_argument("--top-k", type=int, default=None, help="Explain only the top-K findings by score")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--backend", choices=["auto", "ollama", "heuristic"], default="auto",
                    help="auto = Ollama if reachable, else heuristic fallback")
    add_profile_arg(ap)
    args = ap.parse_args()

    inp = args.inp_flag or args.inp
    if not inp:
        ap.error("input file required (positional or --inp)")

    prof_name, prof = select_profile(args.profile)
    model = args.model or prof["ollama_model"]
    top_k = args.top_k if args.top_k is not None else prof["explainer_top_k"]
    num_ctx = prof["ollama_num_ctx"]

    data = json.load(open(inp, "r", encoding="utf-8"))
    findings = sorted(data.get("findings", []), key=lambda x: x.get("score", 0), reverse=True)
    if top_k > 0:
        findings = findings[:top_k]

    backend = args.backend
    if backend in ("auto", "ollama"):
        reachable, present = ollama_available(args.ollama_url, model)
        if not reachable:
            msg = f"Ollama not reachable at {args.ollama_url}"
            if backend == "ollama":
                sys.exit(f"[ERR] {msg}")
            print(f"[WARN] {msg}; falling back to heuristic explainer")
            backend = "heuristic"
        elif not present:
            msg = f"Model {model!r} not found in Ollama (try: ollama pull {model})"
            if backend == "ollama":
                sys.exit(f"[ERR] {msg}")
            print(f"[WARN] {msg}; falling back to heuristic explainer")
            backend = "heuristic"
        else:
            backend = "ollama"
            print(f"[profile] {prof_name}: ollama model={model} ctx={num_ctx} top_k={top_k}")

    out_entries = []
    for i, f in enumerate(findings, 1):
        lines = f"{f.get('start_line')}-{f.get('end_line')}"
        entry = {
            "file": f.get("file"),
            "start_line": f.get("start_line"),
            "end_line": f.get("end_line"),
            "score": f.get("score"),
            "snippet": f.get("snippet", "").splitlines()[:60],
        }
        if backend == "ollama":
            try:
                llm = ollama_chat(args.ollama_url, model, f.get("snippet", ""),
                                  f.get("file"), lines, num_ctx)
                entry.update({
                    "backend": "ollama", "model": model,
                    "is_vulnerable": bool(llm.get("is_vulnerable")),
                    "issue": str(llm.get("issue", "")),
                    "cwe": str(llm.get("cwe", "")),
                    "severity": str(llm.get("severity", "")),
                    "explanation": str(llm.get("explanation", "")),
                    "fix": str(llm.get("fix", "")),
                    "confidence": score_to_confidence(f.get("score"), has_pattern=True),
                })
                print(f"  [{i}/{len(findings)}] {f.get('file')}:{lines} -> "
                      f"{entry['severity'] or 'n/a'} {entry['cwe']}")
            except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as e:
                print(f"  [{i}/{len(findings)}] LLM failed ({e}); heuristic fallback for this finding")
                entry.update(heuristic_entry(f))
        else:
            entry.update(heuristic_entry(f))
        out_entries.append(entry)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"backend": backend, "model": model if backend == "ollama" else None,
                   "profile": prof_name, "explanations": out_entries}, fh, indent=2)
    print(f"[OK] {len(out_entries)} explanations -> {args.out}")

    if args.html:
        # adapt to the heuristic HTML renderer's expected fields
        html_items = []
        for e in out_entries:
            html_items.append({
                **e,
                "explanations": [x for x in (
                    f"{e.get('issue','')}" + (f" [{e['cwe']}]" if e.get("cwe") else ""),
                    e.get("explanation", ""),
                ) if x],
                "recommendations": [e["fix"]] if e.get("fix") else [],
            })
        write_html(html_items, args.html)


if __name__ == "__main__":
    main()
