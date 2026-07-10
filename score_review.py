#!/usr/bin/env python3
"""Score a local-model security review against a Claude reference — objectively.

Given one or more CANDIDATE reviews (e.g. qwen2.5-coder:14b and :32b) and a
REFERENCE review (Claude Opus, produced on the IDENTICAL prompt + context), plus
the SOURCE files, this computes a quantitative quality score so "how close is the
local model to Claude?" becomes a number you can drive up across prompt/KB/model
changes and fine-tuning rounds.

The metric is weighted toward what actually matters for this product — the two
line-numbered walkthroughs and their grounding:

  grounding  (40)  every walkthrough step / finding must quote the VERBATIM source
                   at the cited file:line. This is fully objective: we have the
                   source, we have the citation, we check the quote. Right code at
                   the wrong line is penalised; hallucinated code scores zero.
  coverage   (25)  are BOTH walkthroughs present (in the structured step form),
                   how many steps vs the reference, and what fraction of the source
                   lines do they actually walk (top-to-bottom thoroughness)?
  recall     (20)  did the candidate catch the reference's findings? (overlap by
                   file+line, backed by CWE + title similarity)
  class.     (15)  on the findings it DID catch, does the CWE + severity agree?

Composite is 0-100. Everything except an optional judge layer is deterministic —
no model, no network — so it is fast, reproducible, and cheap to run every round.

    # one reference, one or more candidates (source auto-resolved from the reviews'
    # own file citations when run from the repo the review ran on):
    python score_review.py scan_out/compare/dhcp_local_14b.json \
        scan_out/compare/dhcp_local_32b.json \
        --ref scan_out/compare/dhcp_claude_ref.json

    # pass --src explicitly (LAST, after the candidates) if the sources live
    # elsewhere or the citations use non-repo-relative paths:
    python score_review.py CAND.json --ref REF.json --src path/to/src/ --json-out score.json
"""
import argparse
import json
import os
import re
import sys

import surface_review as sr
from local_vuln_scanner import list_sources

# --- component weights (sum = 100); grounding-heavy on purpose. -------------
WEIGHTS = {"grounding": 40, "coverage": 25, "recall": 20, "classification": 15}

# A quote "matches" the source at its cited lines when its normalised text is at
# least this similar (difflib ratio). 0.80 tolerates minor reflow/omitted blank
# lines while still rejecting a paraphrase or the wrong block.
GROUND_MATCH = 0.80
# Right code, wrong line still shows the model READ the code — partial credit,
# but docked hard because line numbers are a first-class product requirement.
WRONG_LINE_CREDIT = 0.40
# Two findings are "the same issue" at or above this similarity (see _finding_sim).
# 0.50 requires a "strong" signal (0.60 = same-file line-overlap OR shared anchor
# identifier); CWE/title agreement alone (max 0.40) never fabricates a match.
FINDING_MATCH = 0.50

_STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "for",
         "with", "without", "via", "by", "unchecked", "missing", "possible",
         "potential", "buffer", "issue", "field", "value", "data", "code"}


# --------------------------------------------------------------------------- #
# source resolution + normalisation
# --------------------------------------------------------------------------- #
def _add_source(src, f):
    try:
        lines = open(f, "r", errors="ignore", encoding="utf-8").read().splitlines()
    except OSError:
        return False
    for alias in {f, os.path.normpath(f), os.path.basename(f)}:
        src[alias] = lines
    return True


def load_sources(paths):
    """Map a review's `file` citations to real source lines.

    Reviews cite files by the path used at review time (e.g.
    'playground/dhcp-internal.h'); keep every alias — exact path, normalised
    path, and basename — so a citation that drops the directory still resolves.
    """
    src = {}
    for p in paths:
        for f in list_sources(p):
            _add_source(src, f)
    return src


def auto_sources(reviews):
    """Resolve source files from the `file` citations in the reviews themselves,
    so --src can be omitted when running from the repo the review ran on. Each
    cited path is tried as-is (repo-relative) and by basename in the CWD."""
    src = {}
    wanted = {str(fname) for rev in reviews for _, fname, _, _ in _iter_citations(rev) if fname}
    for fname in wanted:
        if not (os.path.exists(fname) and _add_source(src, fname)):
            base = os.path.basename(fname)
            if os.path.exists(base):
                _add_source(src, base)
    return src


def _resolve(src, fname):
    if not fname:
        return None
    for alias in (fname, os.path.normpath(fname), os.path.basename(fname)):
        if alias in src:
            return src[alias]
    return None


def _norm(s):
    """Collapse whitespace so quote-vs-source matching ignores reflow/indent."""
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _norm_block(text):
    """Normalise a multi-line block to a list of non-empty normalised lines."""
    return [n for n in (_norm(ln) for ln in str(text or "").splitlines()) if n]


# --------------------------------------------------------------------------- #
# grounding — the headline metric
# --------------------------------------------------------------------------- #
def check_citation(src, fname, line_field, code, tol=2):
    """Classify one citation's grounding. Returns (status, credit, detail).

    Line-level, not block-similarity: each non-blank quoted line is matched
    against the source, then we ask whether the matched lines land at the cited
    range (± `tol`, to forgive off-by-one blank-line drift without excusing real
    line-number errors). This is fair to an exact quote whose cited range abuts a
    neighbouring declaration, and it still rewards precise line numbers.

    status ∈ grounded | wrong-line | not-found | out-of-bounds | unresolved-file
            | no-code ;  credit ∈ [0,1] contribution to the grounding score.
    """
    lines = _resolve(src, fname)
    if lines is None:
        return "unresolved-file", 0.0, fname or "(no file)"
    quoted = _norm_block(code)
    if not quoted:
        return "no-code", 0.0, "(empty code field)"
    nums = sr._line_numbers(line_field)
    if not nums:
        return "no-code", 0.0, "(no line cited)"
    lo, hi = nums[0], nums[-1]
    if lo > len(lines):
        return "out-of-bounds", 0.0, f"lines {lo}-{hi} > file has {len(lines)}"
    # Index the source: normalised line text -> the line numbers where it occurs.
    norm_src = {}
    for i, ln in enumerate(lines, 1):
        n = _norm(ln)
        if n:
            norm_src.setdefault(n, []).append(i)
    found = in_range = exact = 0
    for q in quoted:
        pos = norm_src.get(q)
        if not pos:
            continue
        found += 1
        nearest = min(pos, key=lambda p: 0 if lo <= p <= hi else min(abs(p - lo), abs(p - hi)))
        if lo - tol <= nearest <= hi + tol:
            in_range += 1
        if lo <= nearest <= hi:
            exact += 1
    nq = len(quoted)
    if found == 0:
        return "not-found", 0.0, f"quote absent from {os.path.basename(fname or '')}"
    found_frac, range_frac = found / nq, in_range / found
    detail = f"{found}/{nq} quoted lines matched, {in_range} within {lo}-{hi}±{tol}"
    if found_frac >= 0.75 and range_frac >= 0.60:
        return "grounded", found_frac, detail
    if found_frac >= 0.75:
        return "wrong-line", WRONG_LINE_CREDIT * found_frac, detail + " (off cited range)"
    return "not-found", 0.30 * found_frac, detail + " (mostly absent)"


def _iter_citations(review):
    """Every (section, file, line, code) citation in a review."""
    for sect in ("what_the_code_does", "what_could_go_wrong"):
        val = review.get(sect)
        if isinstance(val, list):
            for step in val:
                if isinstance(step, dict):
                    yield sect, step.get("file"), step.get("lines"), step.get("code")
    for f in review.get("findings", []):
        if isinstance(f, dict):
            yield "finding", f.get("file"), f.get("line"), f.get("code")


def score_grounding(review, src):
    cites = list(_iter_citations(review))
    if not cites:
        return 0.0, {"citations": 0, "note": "no grounded citations at all"}
    tally = {"grounded": 0, "wrong-line": 0, "not-found": 0,
             "out-of-bounds": 0, "unresolved-file": 0, "no-code": 0}
    credit = 0.0
    misses = []
    for sect, fname, line, code in cites:
        status, c, detail = check_citation(src, fname, line, code)
        tally[status] += 1
        credit += c
        if status != "grounded":
            misses.append(f"[{sect}] {os.path.basename(str(fname or ''))}:{line} — {status}: {detail}")
    frac = credit / len(cites)
    return frac, {"citations": len(cites), "verbatim_at_line": tally["grounded"],
                  "breakdown": tally, "misses": misses[:20]}


# --------------------------------------------------------------------------- #
# coverage — do both walkthroughs exist and walk the whole file?
# --------------------------------------------------------------------------- #
def _steps(review, key):
    val = review.get(key)
    if isinstance(val, list):
        return [s for s in val if isinstance(s, dict)], "steps"
    if isinstance(val, str) and val.strip():
        return [], "string"   # present but unstructured (old single-paragraph form)
    return [], "absent"


def _covered_lines(steps, src):
    """Set of (basename, lineno) touched by any step's citation."""
    covered = set()
    for s in steps:
        lines = _resolve(src, s.get("file"))
        if lines is None:
            continue
        nums = sr._line_numbers(s.get("lines"))
        if not nums:
            continue
        base = os.path.basename(str(s.get("file") or ""))
        for n in range(nums[0], min(nums[-1], len(lines)) + 1):
            covered.add((base, n))
    return covered


def _significant_lines(src):
    """Non-blank source lines, keyed (basename, lineno) — the walk denominator."""
    seen, out = set(), set()
    for alias, lines in src.items():
        base = os.path.basename(alias)
        if base in seen:
            continue
        seen.add(base)
        for i, ln in enumerate(lines, 1):
            if ln.strip():
                out.add((base, i))
    return out


def score_coverage(review, ref, src):
    does, does_form = _steps(review, "what_the_code_does")
    wrong, wrong_form = _steps(review, "what_could_go_wrong")
    ref_does, _ = _steps(ref, "what_the_code_does")
    ref_wrong, _ = _steps(ref, "what_could_go_wrong")

    # presence: full credit only for the structured step form.
    present = 0.0
    for form in (does_form, wrong_form):
        present += 0.5 if form == "steps" else (0.25 if form == "string" else 0.0)

    # step richness relative to the reference.
    def rich(cand, refc):
        return 1.0 if not refc else min(1.0, len(cand) / max(1, len(refc)))
    richness = (rich(does, ref_does) + rich(wrong, ref_wrong)) / 2

    # line-span: what fraction of significant source lines does the walk touch?
    sig = _significant_lines(src)
    covered = _covered_lines(does + wrong, src)
    span = (len(covered & sig) / len(sig)) if sig else 0.0

    score = 0.40 * present + 0.30 * richness + 0.30 * span
    return score, {
        "what_the_code_does": f"{len(does)} steps ({does_form})",
        "what_could_go_wrong": f"{len(wrong)} steps ({wrong_form})",
        "ref_steps": f"{len(ref_does)}/{len(ref_wrong)}",
        "line_span_walked": f"{len(covered & sig)}/{len(sig)} sig lines ({span:.0%})",
    }


# --------------------------------------------------------------------------- #
# findings — recall vs reference + CWE/severity agreement
# --------------------------------------------------------------------------- #
# Generic anchor words that carry no "same code element" signal — the identifiers
# that actually pin a finding to its anchor are things like `chaddr`, `hlen`,
# `dhcp_option_append`, `dhcppacket`, not these.
_GENERIC_ANCHOR = {"dhcp", "packet", "message", "size", "len", "length", "code",
                   "buf", "buffer", "offset", "option", "options", "field",
                   "value", "data", "type", "flag", "flags", "const", "void",
                   "struct", "char", "enum", "define", "sizeof", "uint8", "uint16",
                   "uint32", "int32", "be32", "be16", "bool", "null"}


def _tokens(f):
    text = " ".join(str(f.get(k, "")) for k in ("title", "bug_class", "anchor"))
    return {t for t in re.findall(r"[a-z0-9_]+", text.lower())
            if t not in _STOP and len(t) > 2}


def _anchor_ids(f):
    """Distinctive code identifiers in a finding's anchor/title — the strongest
    'these two findings are the same issue' signal, robust to different cited
    lines or CWE framing (e.g. the chaddr overflow anchored at the buffer decl
    vs. at the hlen length field)."""
    text = " ".join(str(f.get(k, "")) for k in ("anchor", "title"))
    return {t for t in re.findall(r"[a-z_][a-z0-9_]{3,}", text.lower())
            if t not in _GENERIC_ANCHOR}


def _interval(f):
    nums = sr._line_numbers(f.get("line") or f.get("lines"))
    return (nums[0], nums[-1]) if nums else None


def _cwe(f):
    m = re.search(r"\d+", str(f.get("cwe", "") or ""))
    return m.group(0) if m else ""


def _finding_sim(a, b):
    """Similarity of two findings ∈ [0,1]. Two findings are the SAME issue when,
    in the same file, their line ranges overlap OR they share a distinctive anchor
    identifier; CWE + title-token agreement add confidence and break ties."""
    same_file = os.path.basename(str(a.get("file") or "")) == \
        os.path.basename(str(b.get("file") or "")) and bool(a.get("file"))
    ia, ib = _interval(a), _interval(b)
    overlap = bool(same_file and ia and ib and ia[0] <= ib[1] and ib[0] <= ia[1])
    anchor_hit = bool(same_file and (_anchor_ids(a) & _anchor_ids(b)))
    cwe = bool(_cwe(a)) and _cwe(a) == _cwe(b)
    ta, tb = _tokens(a), _tokens(b)
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return 0.60 * (overlap or anchor_hit) + 0.20 * cwe + 0.20 * jac


def match_findings(ref_findings, cand_findings):
    """Greedy best-match each reference finding to a candidate finding."""
    used, pairs = set(), []
    for i, rf in enumerate(ref_findings):
        best_j, best_s = None, 0.0
        for j, cf in enumerate(cand_findings):
            if j in used:
                continue
            s = _finding_sim(rf, cf)
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None and best_s >= FINDING_MATCH:
            used.add(best_j)
            pairs.append((i, best_j, best_s))
    return pairs


def score_findings(review, ref, src):
    cand_f = [f for f in review.get("findings", []) if isinstance(f, dict)]
    ref_f = [f for f in ref.get("findings", []) if isinstance(f, dict)]
    if not ref_f:
        return 1.0, 0.0, {"note": "reference has no findings", "candidate": len(cand_f)}
    pairs = match_findings(ref_f, cand_f)
    recall = len(pairs) / len(ref_f)
    precision = len(pairs) / len(cand_f) if cand_f else 0.0
    cwe_ok = sev_ok = 0
    for i, j, _ in pairs:
        if _cwe(ref_f[i]) and _cwe(ref_f[i]) == _cwe(cand_f[j]):
            cwe_ok += 1
        if sr._norm_severity(ref_f[i].get("severity")) == sr._norm_severity(cand_f[j].get("severity")):
            sev_ok += 1
    classification = (0.6 * cwe_ok + 0.4 * sev_ok) / len(pairs) if pairs else 0.0
    missed = [ref_f[i].get("title", "?") for i in
              set(range(len(ref_f))) - {p[0] for p in pairs}]
    return recall, classification, {
        "reference_findings": len(ref_f), "candidate_findings": len(cand_f),
        "matched": len(pairs), "recall": round(recall, 3),
        "precision": round(precision, 3),
        "cwe_agree": f"{cwe_ok}/{len(pairs)}", "sev_agree": f"{sev_ok}/{len(pairs)}",
        "missed_reference_findings": missed[:12],
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def score_one(review, ref, src):
    g, g_d = score_grounding(review, src)
    c, c_d = score_coverage(review, ref, src)
    r, cl, f_d = score_findings(review, ref, src)
    composite = (WEIGHTS["grounding"] * g + WEIGHTS["coverage"] * c +
                 WEIGHTS["recall"] * r + WEIGHTS["classification"] * cl)
    return {
        "composite": round(composite, 1),
        "components": {"grounding": round(g, 3), "coverage": round(c, 3),
                       "recall": round(r, 3), "classification": round(cl, 3)},
        "weighted": {"grounding": round(WEIGHTS["grounding"] * g, 1),
                     "coverage": round(WEIGHTS["coverage"] * c, 1),
                     "recall": round(WEIGHTS["recall"] * r, 1),
                     "classification": round(WEIGHTS["classification"] * cl, 1)},
        "grounding_detail": g_d, "coverage_detail": c_d, "findings_detail": f_d,
    }


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _print_report(name, res):
    w = res["weighted"]
    print(f"\n{'=' * 70}\n{name}  —  COMPOSITE {res['composite']}/100\n{'=' * 70}")
    print(f"  grounding      {w['grounding']:>5}/40   "
          f"({res['grounding_detail']['verbatim_at_line']}/"
          f"{res['grounding_detail']['citations']} citations verbatim at cited line)")
    print(f"  coverage       {w['coverage']:>5}/25   "
          f"does={res['coverage_detail']['what_the_code_does']}, "
          f"wrong={res['coverage_detail']['what_could_go_wrong']}, "
          f"{res['coverage_detail']['line_span_walked']}")
    print(f"  recall         {w['recall']:>5}/20   "
          f"matched {res['findings_detail'].get('matched', 0)}/"
          f"{res['findings_detail'].get('reference_findings', 0)} reference findings")
    print(f"  classification {w['classification']:>5}/15   "
          f"CWE {res['findings_detail'].get('cwe_agree', '-')}, "
          f"severity {res['findings_detail'].get('sev_agree', '-')}")
    gb = res["grounding_detail"].get("breakdown", {})
    if gb:
        print("  grounding breakdown: " + ", ".join(f"{k}={v}" for k, v in gb.items() if v))
    misses = res["grounding_detail"].get("misses", [])
    if misses:
        print("  ungrounded citations (first few):")
        for m in misses[:8]:
            print(f"     - {m}")
    missed = res["findings_detail"].get("missed_reference_findings", [])
    if missed:
        print("  reference findings the candidate MISSED:")
        for m in missed[:8]:
            print(f"     - {m}")


def main():
    ap = argparse.ArgumentParser(
        description="Objectively score local review(s) against a Claude reference")
    ap.add_argument("candidates", nargs="+", help="Candidate review JSON file(s)")
    ap.add_argument("--ref", required=True, help="Reference (Claude) review JSON")
    ap.add_argument("--src", nargs="+",
                    help="Source file(s)/dir(s) the review ran on (for grounding). "
                         "Optional: if omitted, files are resolved from the reviews' "
                         "own `file` citations relative to the current directory.")
    ap.add_argument("--json-out", help="Write all scores here for trend tracking")
    args = ap.parse_args()

    ref = _load(args.ref)
    candidates = {path: _load(path) for path in args.candidates}

    if args.src:
        src = load_sources(args.src)
    else:
        src = auto_sources([ref, *candidates.values()])
    if not src:
        sys.exit("[ERR] no source files resolved. Pass --src FILES (the sources the "
                 "review ran on) — grounding needs them.")

    results = {}
    for path, review in candidates.items():
        res = score_one(review, ref, src)
        results[path] = res
        _print_report(os.path.basename(path), res)

    if len(results) > 1:
        print(f"\n{'=' * 70}\nLEADERBOARD (vs {os.path.basename(args.ref)})\n{'=' * 70}")
        for path, res in sorted(results.items(), key=lambda kv: -kv[1]["composite"]):
            c = res["components"]
            print(f"  {res['composite']:>5}/100  {os.path.basename(path):<28} "
                  f"g={c['grounding']:.2f} cov={c['coverage']:.2f} "
                  f"rec={c['recall']:.2f} cls={c['classification']:.2f}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump({"reference": args.ref, "results": results},
                  open(args.json_out, "w", encoding="utf-8"), indent=2)
        print(f"\n[OK] scores -> {args.json_out}")


if __name__ == "__main__":
    main()
