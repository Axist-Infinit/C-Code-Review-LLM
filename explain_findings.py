#!/usr/bin/env python3
import argparse, os, json, re, html

# --------------------------------------------------------------------
# Pattern definitions: each ties a concrete API pattern to a specific
# explanation + recommendations.
# --------------------------------------------------------------------

# Each pattern carries a CWE id so heuristic findings get a meaningful SARIF rule
# id (not just the API name). Order is priority order: cwe_for_patterns() returns
# the first matched pattern's CWE, so keep the most specific/severe items first.
PATTERNS = [
    {
        "id": "gets",
        "regex": re.compile(r"\bgets\s*\("),
        "cwe": "CWE-242",  # use of inherently dangerous function
        "why": "Use of gets() is unsafe and leads to buffer overflow.",
        "recs": [
            "Replace gets() with fgets() and pass the destination buffer size.",
            "Ensure all input functions have explicit maximum lengths."
        ],
    },
    {
        "id": "system",
        "regex": re.compile(r"\bsystem\s*\("),
        "cwe": "CWE-78",  # OS command injection
        "why": "Passing attacker-influenced data to system() invites OS command injection.",
        "recs": [
            "Avoid the shell: use execve()/posix_spawn() with an explicit argument vector.",
            "If a shell is unavoidable, strictly allow-list and escape every argument."
        ],
    },
    {
        "id": "popen",
        "regex": re.compile(r"\bpopen\s*\("),
        "cwe": "CWE-78",  # OS command injection
        "why": "popen() runs its argument through /bin/sh; untrusted input enables command injection.",
        "recs": [
            "Replace popen() with a direct exec of the target binary and an argument vector.",
            "Never interpolate untrusted data into a popen() command string."
        ],
    },
    {
        "id": "strcpy",
        "regex": re.compile(r"\bstrcpy\s*\("),
        "cwe": "CWE-120",  # classic buffer overflow
        "why": "strcpy() does not check destination size and can overflow the target buffer.",
        "recs": [
            "Replace strcpy() with strncpy()/strlcpy() and pass the destination size.",
            "Validate that the source string length is less than the destination buffer."
        ],
    },
    {
        "id": "strcat",
        "regex": re.compile(r"\bstrcat\s*\("),
        "cwe": "CWE-120",
        "why": "strcat() appends without bounds checks and can overflow the destination buffer.",
        "recs": [
            "Replace strcat() with strncat()/strlcat() and pass the remaining buffer capacity.",
            "Prefer constructing strings via snprintf() or std::string‑like abstractions where possible."
        ],
    },
    {
        "id": "sprintf",
        "regex": re.compile(r"\bsprintf\s*\("),
        "cwe": "CWE-120",
        "why": "sprintf() can overflow fixed‑size buffers if the formatted output is too long.",
        "recs": [
            "Replace sprintf() with snprintf() and cap the output size to the destination buffer.",
            "Validate that attacker‑controlled input is length‑checked before formatting."
        ],
    },
    {
        "id": "scanf_percent_s",
        # FIXED: properly quoted "%s" so the regex is valid Python.
        "regex": re.compile(r'\bscanf\s*\(\s*"%s"'),
        "cwe": "CWE-120",
        "why": 'scanf("%s") with no width specifier can overflow destination buffers.',
        "recs": [
            'Use scanf with a width limit (e.g. scanf("%15s", buf)) or use fgets() with an explicit buffer size.',
            "Avoid reading unbounded strings into fixed‑size buffers."
        ],
    },
    {
        "id": "memcpy",
        "regex": re.compile(r"\bmemcpy\s*\("),
        "cwe": "CWE-120",
        "why": "memcpy() with unchecked length can overflow or over‑read buffers.",
        "recs": [
            "Ensure the length passed to memcpy() is validated against both source and destination buffer sizes.",
            "Prefer higher‑level copy helpers where buffer sizes are tracked explicitly."
        ],
    },
    {
        "id": "memmove",
        "regex": re.compile(r"\bmemmove\s*\("),
        "cwe": "CWE-120",
        "why": "memmove() with an unchecked length can overflow the destination buffer.",
        "recs": [
            "Validate the length against both buffer sizes before calling memmove().",
            "Track buffer capacities explicitly (e.g. std::span) instead of raw pointers + length."
        ],
    },
    {
        "id": "alloca",
        "regex": re.compile(r"\balloca\s*\("),
        "cwe": "CWE-770",  # allocation without limits -> stack exhaustion
        "why": "alloca() allocates on the stack; an attacker-influenced size can exhaust the stack.",
        "recs": [
            "Use a bounded stack buffer or heap allocation (with a checked size) instead of alloca().",
            "Cap and validate any size derived from input before allocating."
        ],
    },
    # --- C++-specific footguns ----------------------------------------------
    {
        "id": "reinterpret_cast",
        "regex": re.compile(r"\breinterpret_cast\s*<"),
        "cwe": "CWE-704",  # incorrect type conversion
        "why": "reinterpret_cast performs an unchecked reinterpretation; misuse causes type confusion and UB.",
        "recs": [
            "Prefer static_cast/dynamic_cast, or std::bit_cast for value reinterpretation.",
            "Verify the source and destination types are layout-compatible before reinterpreting."
        ],
    },
    {
        "id": "const_cast",
        "regex": re.compile(r"\bconst_cast\s*<"),
        "cwe": "CWE-704",
        "why": "const_cast strips constness; writing through it to a truly const object is undefined behavior.",
        "recs": [
            "Redesign the interface so the object is mutable where it must be written.",
            "Never const_cast away const to modify an object that was originally declared const."
        ],
    },
    {
        "id": "printf_format_string",
        # printf called with a single bare identifier as the format -> attacker
        # controls the format string. High-precision: printf(buf) not printf("...", buf).
        "regex": re.compile(r"\bprintf\s*\(\s*[A-Za-z_]\w*\s*\)"),
        "cwe": "CWE-134",  # uncontrolled format string
        "why": "printf() called with a non-literal format string lets input control conversions (%n, %s).",
        "recs": [
            'Use a literal format string: printf("%s", buf) instead of printf(buf).',
            "Never pass attacker-influenced data as the format argument of a *printf function."
        ],
    },
]

GENERIC_REC = [
    "Replace unsafe C APIs (gets/strcpy/strcat/sprintf) with bounded alternatives (fgets/strlcpy/strlcat/snprintf).",
    "Add explicit length checks before copying, concatenating, or formatting data, especially if attacker‑controlled.",
    "Prefer safer input routines (fgets, getline, scanf with width limits) with maximum sizes.",
    "Run classic static analyzers (clang‑tidy, cppcheck, flawfinder, etc.) alongside this classifier for structural issues.",
]

CONF_LEVELS = ["very low", "low", "medium", "high"]


def score_to_confidence(score: float, has_pattern: bool):
    """
    Map raw classifier score (0–1) + presence of a concrete bad API
    to a qualitative confidence level.
    """
    if score is None:
        return {"level": "unknown", "score": None}

    # Base bucket purely from model score
    if score >= 0.80:
        idx = 3  # high
    elif score >= 0.60:
        idx = 2  # medium
    elif score >= 0.50:
        idx = 1  # low
    else:
        idx = 0  # very low / likely noise

    # If we actually matched a dangerous API, bump confidence up one notch
    if has_pattern and idx < len(CONF_LEVELS) - 1:
        idx += 1

    return {"level": CONF_LEVELS[idx], "score": float(score)}


def cwe_for_patterns(matched_ids):
    """Return the most relevant CWE id for a set of matched pattern ids.

    Patterns are scanned in PATTERNS (priority) order, so the first / most
    specific matched pattern's CWE wins. Returns "" when nothing maps.
    """
    matched = set(matched_ids or ())
    for pat in PATTERNS:
        if pat["id"] in matched and pat.get("cwe"):
            return pat["cwe"]
    return ""


def explain_snippet(snippet: str):
    """
    Return (explanations, recommendations, matched_ids)
    explanations/recs are tightly tied to patterns we actually see in the snippet.
    """
    explanations = []
    recs = []
    matched_ids = set()

    for pat in PATTERNS:
        if pat["regex"].search(snippet):
            matched_ids.add(pat["id"])
            explanations.append(pat["why"])
            recs.extend(pat["recs"])

    if not explanations:
        explanations.append(
            "Heuristic review only: no classic unsafe APIs detected in this snippet. "
            "Inspect bounds checks, input validation, and pointer lifetimes manually."
        )
        recs.extend(GENERIC_REC)

    # Dedup while preserving order
    def dedup(seq):
        out, seen = [], set()
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedup(explanations), dedup(recs), matched_ids


def write_html(explanations, html_path: str):
    """
    Render a simple static HTML report for human consumption.
    `explanations` is the list we emit to JSON.
    """
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)

    rows = []
    for item in explanations:
        file_path = item.get("file", "?")
        start = item.get("start_line", "?")
        end = item.get("end_line", "?")
        score = item.get("score")
        conf = item.get("confidence", {})
        conf_level = conf.get("level", "unknown")
        conf_score = conf.get("score")

        snippet_lines = item.get("snippet", [])
        if isinstance(snippet_lines, str):
            snippet_lines = snippet_lines.splitlines()
        code = html.escape("\n".join(snippet_lines))

        expl_list = item.get("explanations", [])
        rec_list = item.get("recommendations", [])

        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else html.escape(str(score))
        conf_score_str = f" ({conf_score:.3f})" if isinstance(conf_score, (int, float)) else ""

        rows.append(f"""
        <section class="finding">
          <h2>{html.escape(file_path)}:{start}-{end}</h2>
          <p><strong>Model score:</strong> {score_str} &nbsp;
             <strong>Confidence:</strong> {html.escape(conf_level)}{conf_score_str}</p>
          <pre><code>{code}</code></pre>
          <h3>Why this is suspicious</h3>
          <ul>
            {''.join(f"<li>{html.escape(e)}</li>" for e in expl_list)}
          </ul>
          <h3>Recommendations</h3>
          <ul>
            {''.join(f"<li>{html.escape(r)}</li>" for r in rec_list)}
          </ul>
        </section>
        """)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>LLM Vulnerability Findings</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; }}
  h1 {{ border-bottom: 1px solid #ccc; padding-bottom: .5rem; }}
  section.finding {{ border: 1px solid #ddd; padding: 1rem; margin-bottom: 1.5rem; border-radius: 6px; }}
  pre {{ background: #f5f5f5; padding: .75rem; overflow-x: auto; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
</style>
</head>
<body>
<h1>LLM Vulnerability Findings</h1>
<p>Total findings: {len(explanations)}</p>
{"".join(rows)}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"[OK] HTML report -> {html_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", help="classifier_findings.json from local_vuln_scanner")
    ap.add_argument("--inp", dest="inp_flag", help="Alternative to the positional input path")
    ap.add_argument("--out", required=True, help="llm_findings_heuristic.json (enriched)")
    ap.add_argument("--html", help="Optional: write an HTML report to this path")
    args = ap.parse_args()

    inp = args.inp_flag or args.inp
    if not inp:
        ap.error("input file required (positional or --inp)")

    data = json.load(open(inp, "r", encoding="utf-8"))
    findings = data.get("findings", [])

    explanations_out = []
    for f in findings:
        raw_snippet = f.get("snippet", "")
        expl, recs, matched_ids = explain_snippet(raw_snippet)
        score = f.get("score", None)
        conf = score_to_confidence(score, has_pattern=bool(matched_ids))

        # store snippet as list of lines (good for JSON + HTML)
        snippet_lines = raw_snippet.splitlines()[:40]

        explanations_out.append({
            "file": f.get("file"),
            "start_line": f.get("start_line"),
            "end_line": f.get("end_line"),
            "score": score,
            "confidence": conf,
            "cwe": cwe_for_patterns(matched_ids),
            "matched_patterns": sorted(matched_ids),
            "explanations": expl,
            "snippet": snippet_lines,
            "recommendations": recs,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"explanations": explanations_out}, f, indent=2)
    print("[OK] JSON explanations ->", args.out)

    if args.html:
        write_html(explanations_out, args.html)


if __name__ == "__main__":
    main()

