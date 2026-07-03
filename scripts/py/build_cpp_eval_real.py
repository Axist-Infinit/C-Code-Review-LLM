#!/usr/bin/env python3
"""Build benchmarks/cpp_eval_real.jsonl — REAL C++ CVE functions for the ML eval.

The synthetic benchmarks/cpp_eval.jsonl proved the BigVul-trained classifier
collapses on short snippets (a length/distribution effect, not a language gap —
see benchmarks/cpp_eval_findings.md). This builder produces the recommended
follow-up: full real-world C++ functions from PrimeVul (paired config), so C++
generalization is measured on the distribution the model was built for.

Source (pinned): colin/PrimeVul @ 4fd7158 — HF mirror of the official
DLVulDet/PrimeVul dataset (MIT; Ding et al., "Vulnerability Detection with Code
Language Models: How Far Are We?", arXiv:2403.18624, ICSE 2025). In the paired
config adjacent rows are a vuln/fixed pair from the same fix commit, which is
exactly the func_before/func_after hard-negative convention the classifier was
trained with on BigVul.

Leakage control (ON by default; --no-leak-check for offline smoke runs):
  * drop any pair whose fix commit_id appears anywhere in bstee615/bigvul
    (pinned revision) — same-commit functions would be train/test leakage;
  * drop any row whose whitespace-normalized function text hashes into the
    BigVul func_before/func_after corpus (catches same function via a
    different commit / repo mirror).

Output rows: {name, code, label, cwe, lang: "cpp", category, source, cve} —
the same shape benchmarks/cpp_eval.jsonl uses, so eval_cpp.sh / evaluate_model
--group-by category work unchanged. A .provenance.json (dataset id, revision,
license, citation, filter stats, sha256) is written next to the jsonl.

Heavy deps (datasets) are imported lazily inside main-path helpers so this
module stays importable in the torch-free test environment.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

PRIMEVUL_ID = "colin/PrimeVul"
PRIMEVUL_CONFIG = "paired"
PRIMEVUL_REVISION = "4fd7158322872d711e90f091dbd8673ef32cb1be"
PRIMEVUL_LICENSE = "MIT (Copyright (c) 2024 DLVulDet)"
PRIMEVUL_CITATION = (
    "Ding, Y., Fu, Y., Ibrahim, O., Sitawarin, C., Chen, X., Alomair, B., "
    "Wagner, D., Ray, B., Chen, Y. \"Vulnerability Detection with Code "
    "Language Models: How Far Are We?\" arXiv:2403.18624 (ICSE 2025).")

BIGVUL_ID = "bstee615/bigvul"
BIGVUL_REVISION = "4d8647a9b52fdd2ae42343c60fd6c7c48bbff25e"

# Unambiguous C++ extensions. `.h` is C/C++-ambiguous and `file_name` is the
# literal string 'None' on ~1/3 of paired rows — both are excluded.
CPP_EXTS = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}
# Unambiguous C. Used by --lang c to build the same-distribution C control set
# that separates "C++ transfer gap" from "PrimeVul is harder than BigVul".
C_EXTS = {".c"}
EXTS_BY_LANG = {"cpp": CPP_EXTS, "c": C_EXTS}

DEFAULT_OUT = os.path.join("benchmarks", "cpp_eval_real.jsonl")
DEFAULT_OUT_BY_LANG = {
    "cpp": DEFAULT_OUT,
    "c": os.path.join("benchmarks", "c_eval_control.jsonl"),
}
DEFAULT_MAX_ROWS = 1200
DEFAULT_MAX_BYTES = 4_000_000
DEFAULT_MAX_FUNC_CHARS = 30_000


# --- pure helpers (stdlib only, unit-testable without datasets) --------------

def norm_ws(code):
    """Collapse ALL whitespace runs to single spaces.

    Stronger than merge_and_split.norm_code on purpose: leakage matching must
    survive reformatting between dataset exports, not just trailing blanks.
    """
    return " ".join(str(code or "").split())


def fingerprint(code):
    """sha256 of the whitespace-normalized function text."""
    return hashlib.sha256(norm_ws(code).encode("utf-8")).hexdigest()


def is_cpp_filename(file_name, exts=None):
    """True when file_name has an unambiguous extension from `exts`
    (default: the C++ set).

    PrimeVul serializes missing metadata as the literal string 'None'.
    """
    s = str(file_name or "").strip()
    if not s or s == "None":
        return False
    return os.path.splitext(s)[1].lower() in (CPP_EXTS if exts is None else exts)


def norm_cwe(raw):
    """First 'CWE-N' in the row's cwe field, else ''.

    The mirror stores the list as its string repr (e.g. "['CWE-843']"), so a
    regex scan over str() handles both real lists and strings.
    """
    m = re.search(r"CWE-\d+", str(raw or ""))
    return m.group(0) if m else ""


def norm_target(raw):
    """PrimeVul target as int (serialized as the strings '0'/'1')."""
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def pair_up(rows):
    """(vuln_row, fixed_row) pairs from a paired-config split.

    Adjacent rows are a pair sharing commit_id with targets (1, 0); anything
    that violates that invariant is skipped defensively and counted.
    Returns (pairs, n_malformed). Pure.
    """
    pairs, malformed = [], 0
    for i in range(0, len(rows) - 1, 2):
        a, b = rows[i], rows[i + 1]
        if (a.get("commit_id") == b.get("commit_id")
                and norm_target(a.get("target")) == 1
                and norm_target(b.get("target")) == 0):
            pairs.append((a, b))
        else:
            malformed += 1
    return pairs, malformed


def to_row(raw, split, pos, label, revision=PRIMEVUL_REVISION, lang="cpp"):
    """Map one PrimeVul record to the cpp_eval_real.jsonl schema. Pure."""
    cwe = norm_cwe(raw.get("cwe"))
    row = {
        "name": f"primevul_{lang}_{split}_{pos:04d}_{'vuln' if label else 'fixed'}"
                if lang != "cpp" else
                f"primevul_{split}_{pos:04d}_{'vuln' if label else 'fixed'}",
        "code": raw.get("func") or "",
        "label": label,
        "cwe": cwe,
        "lang": lang,
        "category": cwe or "real",
        "source": f"{PRIMEVUL_ID}@{revision}",
    }
    cve = str(raw.get("cve") or "").strip()
    if cve and cve != "None":
        row["cve"] = cve
    return row


def row_bytes(row):
    """Serialized size of one output line (JSON + newline)."""
    return len(json.dumps(row, ensure_ascii=False).encode("utf-8")) + 1


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def select_rows(pairs, singles, max_rows, max_bytes, seed=42):
    """Greedy cap-respecting selection. Pure.

    Complete vuln/fixed pairs go first (they keep the set balanced by
    construction); leftover singletons top the set up only while neither class
    would exceed 62% (a buffer inside the 35-65% schema-test bound). A pair or
    singleton that would bust a cap is skipped, not a loop break, so smaller
    candidates later in the shuffle can still fill the budget.
    """
    rng = random.Random(seed)
    pairs = list(pairs)
    singles = list(singles)
    rng.shuffle(pairs)
    rng.shuffle(singles)
    out, used = [], 0
    for a, b in pairs:
        need = row_bytes(a) + row_bytes(b)
        if len(out) + 2 <= max_rows and used + need <= max_bytes:
            out.extend((a, b))
            used += need
    n_pos = sum(r["label"] for r in out)
    n_neg = len(out) - n_pos
    for r in singles:
        need = row_bytes(r)
        if len(out) + 1 > max_rows or used + need > max_bytes:
            continue
        pos = n_pos + (1 if r["label"] else 0)
        neg = n_neg + (0 if r["label"] else 1)
        if max(pos, neg) / (pos + neg) > 0.62:
            continue
        out.append(r)
        used += need
        n_pos, n_neg = pos, neg
    out.sort(key=lambda r: r["name"])
    return out, used


# --- dataset access (lazy heavy imports) --------------------------------------

def fetch_primevul_splits(revision):
    """{split: [raw rows]} for the pinned paired config."""
    from datasets import load_dataset
    print(f"[INFO] Loading {PRIMEVUL_ID} ({PRIMEVUL_CONFIG}) @ {revision}")
    ds = load_dataset(PRIMEVUL_ID, PRIMEVUL_CONFIG, revision=revision)
    return {split: list(ds[split]) for split in ds.keys()}


def build_bigvul_leak_index(revision):
    """(commit_ids, func_fingerprints) over ALL bstee615/bigvul splits.

    The task minimum is the train split (that is what the classifier saw), but
    hashing every split is cheap and also keeps the eval disjoint from the
    BigVul val/test sets this repo benchmarks against.
    """
    from datasets import load_dataset
    print(f"[INFO] Loading {BIGVUL_ID} @ {revision} for the leakage index")
    ds = load_dataset(BIGVUL_ID, revision=revision)
    commits, hashes = set(), set()
    for split in ds.keys():
        for row in ds[split]:
            cid = str(row.get("commit_id") or "").strip()
            if cid:
                commits.add(cid)
            for key in ("func_before", "func_after"):
                code = row.get(key)
                if code and str(code).strip():
                    hashes.add(fingerprint(code))
    print(f"[INFO] BigVul leak index: {len(commits)} commit_ids, "
          f"{len(hashes)} function hashes")
    return commits, hashes


# --- build ---------------------------------------------------------------------

def build(args):
    splits = fetch_primevul_splits(args.revision)
    total_raw = sum(len(v) for v in splits.values())

    leak_commits, leak_hashes = set(), set()
    if args.leak_check:
        leak_commits, leak_hashes = build_bigvul_leak_index(args.bigvul_revision)
    else:
        print("[WARN] --no-leak-check: skipping the BigVul overlap filter; "
              "do NOT benchmark a BigVul-trained model against this output.")

    stats = {"raw_rows": total_raw, "malformed_pairs": 0, "cpp_pairs": 0,
             "dropped_commit_overlap": 0, "dropped_func_hash": 0,
             "dropped_dup": 0, "dropped_too_long": 0}
    seen = set()
    pairs, singles = [], []
    for split, rows in sorted(splits.items()):
        split_pairs, malformed = pair_up(rows)
        stats["malformed_pairs"] += malformed
        for pos, (raw_v, raw_f) in enumerate(split_pairs):
            exts = EXTS_BY_LANG[getattr(args, "lang", "cpp")]
            if not (is_cpp_filename(raw_v.get("file_name"), exts)
                    and is_cpp_filename(raw_f.get("file_name"), exts)):
                continue
            stats["cpp_pairs"] += 1
            if str(raw_v.get("commit_id") or "") in leak_commits:
                stats["dropped_commit_overlap"] += 2
                continue
            keep = []
            for raw, label in ((raw_v, 1), (raw_f, 0)):
                row = to_row(raw, split, pos, label, args.revision,
                             lang=getattr(args, "lang", "cpp"))
                if not row["code"].strip():
                    continue
                if len(row["code"]) > args.max_func_chars:
                    stats["dropped_too_long"] += 1
                    continue
                h = fingerprint(row["code"])
                if h in leak_hashes:
                    stats["dropped_func_hash"] += 1
                    continue
                if h in seen:
                    stats["dropped_dup"] += 1
                    continue
                seen.add(h)
                row["_project"] = str(raw.get("project") or "?")
                keep.append(row)
            if len(keep) == 2:
                pairs.append((keep[0], keep[1]))
            else:
                singles.extend(keep)

    out_rows, used_bytes = select_rows(pairs, singles, args.max_rows,
                                       args.max_bytes, args.seed)
    if not out_rows:
        raise RuntimeError("no C++ rows survived filtering — check the pinned "
                           "revisions / filters")
    return out_rows, used_bytes, stats


def write_outputs(out_rows, stats, args):
    # The working "_project" tag rode along for per-project provenance (the
    # C++ pairs are TensorFlow-heavy); strip it so the committed rows keep the
    # exact cpp_eval schema. Popping only shrinks rows, so the byte cap that
    # select_rows enforced still holds.
    per_project = {}
    for row in out_rows:
        proj = row.pop("_project", "?")
        per_project[proj] = per_project.get(proj, 0) + 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_pos = sum(r["label"] for r in out_rows)
    n_neg = len(out_rows) - n_pos
    cves = {r["cve"] for r in out_rows if r.get("cve")}
    size = os.path.getsize(args.out)
    prov = {
        "dataset": PRIMEVUL_ID,
        "config": PRIMEVUL_CONFIG,
        "revision": args.revision,
        "license": PRIMEVUL_LICENSE,
        "citation": PRIMEVUL_CITATION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out": args.out,
        "rows": len(out_rows),
        "n_vuln": n_pos,
        "n_clean": n_neg,
        "unique_cves": len(cves),
        "bytes": size,
        "sha256": file_sha256(args.out),
        "caps": {"max_rows": args.max_rows, "max_bytes": args.max_bytes,
                 "max_func_chars": args.max_func_chars, "seed": args.seed},
        "leak_check": {
            "enabled": args.leak_check,
            "against": f"{BIGVUL_ID}@{args.bigvul_revision}" if args.leak_check else None,
            "strategy": "fix-commit_id intersection (pair-level) + sha256 of "
                        "whitespace-normalized function text vs BigVul "
                        "func_before/func_after (row-level), all splits",
        },
        "filter_stats": stats,
        "per_project": dict(sorted(per_project.items(),
                                   key=lambda kv: -kv[1])),
    }
    prov_path = args.provenance
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")
    print(f"[OK] Wrote {len(out_rows)} rows ({n_pos} vuln / {n_neg} clean, "
          f"{size} bytes) -> {args.out}")
    print(f"[OK] Provenance -> {prov_path}")
    print(f"[prov] sha256={prov['sha256']}")
    print(f"[stats] {json.dumps(stats)}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the real-C++ eval set from PrimeVul (paired)")
    ap.add_argument("--lang", choices=("cpp", "c"), default="cpp",
                    help="Language slice: cpp (the eval set) or c (the "
                         "same-distribution control that separates a C++ "
                         "transfer gap from PrimeVul just being harder).")
    ap.add_argument("--out", default=None,
                    help=f"Output JSONL (default per --lang: "
                         f"{DEFAULT_OUT_BY_LANG['cpp']} / "
                         f"{DEFAULT_OUT_BY_LANG['c']})")
    ap.add_argument("--provenance", default=None,
                    help="Provenance JSON path (default: <out> with "
                         ".provenance.json instead of .jsonl)")
    ap.add_argument("--revision", default=PRIMEVUL_REVISION,
                    help="PrimeVul mirror revision to pin the fetch to.")
    ap.add_argument("--bigvul-revision", default=BIGVUL_REVISION,
                    help="bstee615/bigvul revision for the leakage index.")
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--max-func-chars", type=int, default=DEFAULT_MAX_FUNC_CHARS,
                    help="Drop single functions longer than this (keeps one "
                         "pathological row from eating the byte budget).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-leak-check", dest="leak_check", action="store_false",
                    help="Skip the BigVul overlap filter (offline runs only).")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = DEFAULT_OUT_BY_LANG[args.lang]
    if args.provenance is None:
        base = args.out[:-6] if args.out.endswith(".jsonl") else args.out
        args.provenance = base + ".provenance.json"
    return args


def main(argv=None):
    args = parse_args(argv)
    out_rows, _used, stats = build(args)
    write_outputs(out_rows, stats, args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail LOUD (non-zero) — a silent partial build must never masquerade
        # as the real eval set (same policy as fetch_bigvul_hf.py).
        print("[FATAL] cpp_eval_real build failed:", e, file=sys.stderr)
        sys.exit(1)
