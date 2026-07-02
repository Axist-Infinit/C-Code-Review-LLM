#!/usr/bin/env python3
"""Draw a seeded, balanced (50/50 label) subset from an ingest JSONL shard.

This generates the data/bigvul_{train,val,ctest}.jsonl splits referenced by the
reference-model recipe in benchmarks/cpp_eval_findings.md, which previously had
no committed generator. Pure stdlib; deterministic for a given (--seed, input).

Usage:
  python3 scripts/py/make_balanced_subset.py data/ingest/bigvul_train.jsonl \
      data/bigvul_train.jsonl --rows 2400 [--seed 42]
"""
import argparse
import json
import random
import sys


def balanced_sample(rows, n_rows, seed):
    """Return n_rows rows, half label==1 and half label==0, seeded. Pure.

    Sampling order is deterministic: pools keep input order, and one
    random.Random(seed) drives both draws. Raises ValueError when either
    pool is too small.
    """
    if n_rows % 2:
        raise ValueError(f"--rows must be even for a 50/50 split (got {n_rows})")
    half = n_rows // 2
    pos = [r for r in rows if r.get("label") == 1]
    neg = [r for r in rows if r.get("label") == 0]
    if len(pos) < half or len(neg) < half:
        raise ValueError(
            f"not enough rows for a balanced {n_rows}: "
            f"have {len(pos)} vulnerable / {len(neg)} clean, need {half} of each")
    rng = random.Random(seed)
    picked = rng.sample(pos, half) + rng.sample(neg, half)
    rng.shuffle(picked)
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Seeded balanced subset of an ingest JSONL shard")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, required=True,
                    help="total rows to emit (half vulnerable, half clean)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with open(args.inp, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    try:
        picked = balanced_sample(rows, args.rows, args.seed)
    except ValueError as exc:
        print(f"[err] {exc}", file=sys.stderr)
        return 2
    with open(args.out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pos = sum(1 for r in picked if r["label"] == 1)
    print(f"[OK] {args.out}: {len(picked)} rows "
          f"({n_pos} vulnerable / {len(picked) - n_pos} clean, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
