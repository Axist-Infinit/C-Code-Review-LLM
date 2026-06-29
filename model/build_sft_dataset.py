#!/usr/bin/env python3
"""Build explainer SFT data from reviewed pipeline findings.

Turns the explainer's own output (llm_findings.json, or a hand-curated file in
the same shape) into {"code", "analysis"} JSONL that model/train_llm_sft.py
consumes. This is how you close the loop: run the pipeline, have an analyst keep
the good explanations, then fine-tune the local model on your codebase's real
vulnerabilities so it gets better at exactly the bugs you care about.

Pure stdlib so it stays unit-testable and runs in an air-gapped environment.

Usage:
  python model/build_sft_dataset.py scan_out/llm_findings.json \
      --out-dir data --val-frac 0.1 --only-vulnerable
"""
import argparse
import json
import os
import random

# The analysis fields we carry into training, mirroring llm_explain's schema.
# Keep in sync with model/train_llm_sft.py ANALYSIS_FIELDS (a test asserts it).
# Reason-then-judge order (describe before concluding) — see train_llm_sft.py.
ANALYSIS_FIELDS = ("what_code_does", "what_could_go_wrong", "vulnerability",
                   "is_vulnerable", "issue", "cwe", "severity",
                   "explanation", "fix")


def _snippet_text(entry):
    """Findings store the snippet either as a list of lines or a single string."""
    snip = entry.get("snippet", "")
    if isinstance(snip, list):
        return "\n".join(snip)
    return str(snip)


def record_from_entry(entry):
    """Convert one explainer entry to an SFT record, or None if unusable.

    Requires a code snippet and at least one populated analysis field.
    """
    code = _snippet_text(entry).strip()
    if not code:
        return None
    analysis = {k: entry.get(k) for k in ANALYSIS_FIELDS}
    meaningful = ("issue", "explanation", "cwe", "vulnerability", "what_could_go_wrong")
    if not any(str(analysis.get(k) or "").strip() for k in meaningful):
        return None  # nothing meaningful for the model to learn
    return {"code": code, "analysis": analysis}


def build_records(entries, only_vulnerable=False, min_score=None,
                  corroboration=None, min_corroboration=0):
    """Build SFT records from explainer entries, applying optional filters.

    When `corroboration` (from load_corroboration) is supplied, each kept record
    is annotated with how many independent static tools (cppcheck/flawfinder/
    clang-tidy) flagged a line inside the finding's range, and `min_corroboration`
    drops findings below that bar — yielding high-precision positives where the
    ML classifier and static analyzers agree.
    """
    records = []
    for e in entries:
        if only_vulnerable and e.get("is_vulnerable") is False:
            continue
        if min_score is not None:
            score = e.get("score")
            if not isinstance(score, (int, float)) or score < min_score:
                continue
        n_corrob, tools = count_corroboration(e, corroboration)
        if min_corroboration > 0 and n_corrob < min_corroboration:
            continue
        rec = record_from_entry(e)
        if rec is None:
            continue
        if corroboration is not None:
            # Informational metadata for curation; the trainer reads only
            # code/analysis, so this never pollutes the training target.
            rec["corroboration"] = n_corrob
            rec["corroborated_by"] = tools
        records.append(rec)
    return records


def load_corroboration(path):
    """Index an ensemble_scan output by file -> [(line, tool), ...] for the
    independent static tools (everything except the ML classifier itself)."""
    data = json.load(open(path, "r", encoding="utf-8"))
    by_file = {}
    for f in data.get("findings", []):
        tool = f.get("tool")
        line = f.get("start_line")
        if not tool or tool == "ml-classifier" or line is None:
            continue
        try:
            line = int(line)
        except (TypeError, ValueError):
            continue
        by_file.setdefault(f.get("file"), []).append((line, tool))
    return by_file


def count_corroboration(entry, corroboration):
    """Count distinct static tools whose flagged line falls within the entry's
    [start_line, end_line] range. Returns (count, sorted_tools).

    Files are matched exactly first, then by basename (the ML scan and the
    static tools may print paths differently for the same file).
    """
    if not corroboration:
        return 0, []
    path = entry.get("file")
    start = entry.get("start_line")
    end = entry.get("end_line") or start
    if path is None or start is None:
        return 0, []
    candidates = corroboration.get(path)
    if candidates is None:
        base = os.path.basename(path)
        candidates = [lt for k, v in corroboration.items()
                      if os.path.basename(k) == base for lt in v]
    tools = {tool for line, tool in candidates if start <= line <= end}
    return len(tools), sorted(tools)


def split_records(records, val_frac=0.1, seed=42):
    """Deterministically shuffle and split into (train, val).

    Guarantees a non-empty train split; only carves out a val split when there
    are enough records for it to be meaningful.
    """
    if not records:
        return [], []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)
    if len(shuffled) > 1:
        n_val = min(max(n_val, 1), len(shuffled) - 1) if val_frac > 0 else 0
    else:
        n_val = 0
    return shuffled[n_val:], shuffled[:n_val]


def load_entries(path):
    """Load entries from an explainer (explanations) or classifier (findings) file."""
    data = json.load(open(path, "r", encoding="utf-8"))
    return data.get("explanations") or data.get("findings") or []


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Build explainer SFT JSONL from findings")
    ap.add_argument("inputs", nargs="+", help="One or more *_findings.json files")
    ap.add_argument("--out-dir", default="data", help="Where to write sft_train/val.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-vulnerable", action="store_true",
                    help="Keep only entries judged vulnerable")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Drop entries below this classifier score")
    ap.add_argument("--ensemble", default=None,
                    help="ensemble_scan output JSON to corroborate findings against")
    ap.add_argument("--min-corroboration", type=int, default=0,
                    help="Keep only findings flagged within range by >= N independent "
                         "static tools (requires --ensemble)")
    args = ap.parse_args()

    if args.min_corroboration > 0 and not args.ensemble:
        ap.error("--min-corroboration requires --ensemble")

    entries = []
    for path in args.inputs:
        entries.extend(load_entries(path))

    corroboration = load_corroboration(args.ensemble) if args.ensemble else None
    records = build_records(entries, only_vulnerable=args.only_vulnerable,
                            min_score=args.min_score, corroboration=corroboration,
                            min_corroboration=args.min_corroboration)
    if corroboration is not None:
        n_corrob = sum(1 for r in records if r.get("corroboration"))
        print(f"[corroboration] {n_corrob}/{len(records)} kept records are "
              f"corroborated by >=1 static tool")
    if not records:
        raise SystemExit("[ERR] no usable SFT records produced from inputs")

    train, val = split_records(records, val_frac=args.val_frac, seed=args.seed)
    train_path = os.path.join(args.out_dir, "sft_train.jsonl")
    val_path = os.path.join(args.out_dir, "sft_val.jsonl")
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    print(f"[OK] {len(train)} train -> {train_path}")
    print(f"[OK] {len(val)} val   -> {val_path}")


if __name__ == "__main__":
    main()
