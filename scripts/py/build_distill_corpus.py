#!/usr/bin/env python3
"""Assemble the distillation corpus: real C/C++ code -> blind files + manifest.

The teacher (Claude Opus) reviews each file BLIND; its reviews become both the
eval gold set and the fine-tuning target. This script draws from the PrimeVul
real-world sets (benchmarks/{cpp_eval_real,c_eval_control}.jsonl), then:

  * dedups by normalised code hash,
  * splits TRAIN vs EVAL disjointly by a stable hash of (name+code) — the same
    function can never land in both (no train/eval leakage),
  * stratifies by CWE with the HARD classes the local models miss ranked first
    (UAF/race/int-over-underflow/OOB), balanced 50/50 vulnerable/clean,
  * writes each function to its own .c/.cpp file containing ONLY the code (no CWE
    or label — the reviewer must judge blind), plus a manifest carrying the ground
    truth (cwe/label/cve/source) for later analysis.

The manifest is what scripts drive the Opus-reviewer workflow off of.

    python3 scripts/py/build_distill_corpus.py --out-dir corpus \
        --train-rows 400 --eval-rows 80 --seed 42
"""
import argparse
import collections
import hashlib
import json
import os
import re

SOURCES = ("benchmarks/cpp_eval_real.jsonl", "benchmarks/c_eval_control.jsonl")

# Hard-first CWE priority: the classes every local config missed at temp 0.3
# (see the DHCP baseline) come first so a small pilot slice still covers them.
CWE_PRIORITY = ["CWE-416", "CWE-362", "CWE-191", "CWE-190", "CWE-787", "CWE-125",
                "CWE-476", "CWE-674", "CWE-835", "CWE-20", "CWE-401", "CWE-119",
                "CWE-369", "CWE-617", "CWE-703", "CWE-200", "CWE-399", "CWE-284",
                "CWE-400"]


def _norm_code(code):
    return re.sub(r"\s+", " ", str(code or "")).strip()


def _hash(*parts):
    return hashlib.sha1("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def load_rows():
    rows, seen = [], set()
    for src in SOURCES:
        if not os.path.exists(src):
            continue
        for line in open(src, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            code = r.get("code", "")
            h = _hash(_norm_code(code))
            if h in seen or not _norm_code(code):
                continue
            seen.add(h)
            r["_hash"] = h
            r["_split_key"] = int(_hash(r.get("name", ""), h)[:8], 16)
            rows.append(r)
    return rows


def _cwe_rank(cwe):
    try:
        return CWE_PRIORITY.index(cwe)
    except ValueError:
        return len(CWE_PRIORITY)


def stratify(rows, n_rows, per_cwe_cap):
    """Pick up to n_rows, balanced by label, round-robin across CWEs (hard-first),
    capping any single CWE so no class dominates. Deterministic (rows pre-sorted)."""
    by_cwe = collections.defaultdict(lambda: {0: [], 1: []})
    for r in rows:
        by_cwe[r.get("cwe", "?")][r.get("label")].append(r)
    for cwe in by_cwe:
        for lab in (0, 1):
            by_cwe[cwe][lab].sort(key=lambda r: r["_split_key"])
    order = sorted(by_cwe, key=lambda c: (_cwe_rank(c), c))
    picked, counts = [], collections.Counter()
    want_pos = n_rows // 2
    n_pos = n_neg = 0
    # round-robin: one row per (cwe,label) per lap until we hit the target
    lap = 0
    while len(picked) < n_rows:
        progressed = False
        for cwe in order:
            for lab in (1, 0):
                if lab == 1 and n_pos >= want_pos:
                    continue
                if lab == 0 and n_neg >= n_rows - want_pos:
                    continue
                if counts[cwe] >= per_cwe_cap:
                    continue
                pool = by_cwe[cwe][lab]
                if lap < len(pool):
                    r = pool[lap]
                    picked.append(r)
                    counts[cwe] += 1
                    n_pos += (lab == 1)
                    n_neg += (lab == 0)
                    progressed = True
                    if len(picked) >= n_rows:
                        break
            if len(picked) >= n_rows:
                break
        lap += 1
        if not progressed:
            break
    return picked


def write_split(rows, out_dir, split):
    d = os.path.join(out_dir, split)
    os.makedirs(d, exist_ok=True)
    manifest = []
    for i, r in enumerate(rows):
        ext = "cpp" if r.get("lang") == "cpp" else "c"
        fname = f"{split}_{i:04d}_{r.get('cwe','NA').replace('CWE-','cwe')}.{ext}"
        path = os.path.join(d, fname)
        # BLIND: only the code goes in the file (a neutral banner, no label/CWE).
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"/* sample: {r.get('name','')} */\n{r.get('code','')}\n")
        manifest.append({
            "file": path, "name": r.get("name", ""), "lang": r.get("lang", "c"),
            "cwe_truth": r.get("cwe", ""), "label_truth": r.get("label"),
            "cve": r.get("cve", ""), "source": r.get("source", ""), "hash": r["_hash"],
        })
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Build the blind distillation corpus + manifest")
    ap.add_argument("--out-dir", default="corpus")
    ap.add_argument("--train-rows", type=int, default=400)
    ap.add_argument("--eval-rows", type=int, default=80)
    ap.add_argument("--eval-pct", type=int, default=18, help="hash buckets (0-99) reserved for EVAL")
    ap.add_argument("--per-cwe-cap-train", type=int, default=40)
    ap.add_argument("--per-cwe-cap-eval", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)  # split is hash-based; seed kept for parity
    args = ap.parse_args()

    rows = load_rows()
    print(f"[load] {len(rows)} deduped rows across {len(SOURCES)} source(s)")
    eval_pool = [r for r in rows if (r["_split_key"] % 100) < args.eval_pct]
    train_pool = [r for r in rows if (r["_split_key"] % 100) >= args.eval_pct]
    print(f"[split] disjoint by hash: {len(train_pool)} train-pool / {len(eval_pool)} eval-pool")

    train = stratify(train_pool, args.train_rows, args.per_cwe_cap_train)
    eval_ = stratify(eval_pool, args.eval_rows, args.per_cwe_cap_eval)

    os.makedirs(args.out_dir, exist_ok=True)
    train_manifest = write_split(train, args.out_dir, "train")
    eval_manifest = write_split(eval_, args.out_dir, "eval")
    json.dump(train_manifest, open(os.path.join(args.out_dir, "train_manifest.json"), "w"), indent=2)
    json.dump(eval_manifest, open(os.path.join(args.out_dir, "eval_manifest.json"), "w"), indent=2)

    def dist(ms):
        c = collections.Counter(m["cwe_truth"] for m in ms)
        return dict(c.most_common(10))
    print(f"[OK] TRAIN {len(train_manifest)} files -> {args.out_dir}/train/  "
          f"({sum(m['label_truth']==1 for m in train_manifest)} vuln)")
    print(f"     top CWEs: {dist(train_manifest)}")
    print(f"[OK] EVAL  {len(eval_manifest)} files -> {args.out_dir}/eval/   "
          f"({sum(m['label_truth']==1 for m in eval_manifest)} vuln)")
    print(f"     top CWEs: {dist(eval_manifest)}")
    # leakage assertion: train/eval code hashes must be disjoint.
    th = {m["hash"] for m in train_manifest}
    eh = {m["hash"] for m in eval_manifest}
    assert not (th & eh), "LEAKAGE: shared code hash between train and eval"
    print(f"[OK] no leakage: {len(th & eh)} shared hashes between train/eval")


if __name__ == "__main__":
    main()
