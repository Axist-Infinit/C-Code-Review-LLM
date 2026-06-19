#!/usr/bin/env python3
"""Merge ingested JSONL shards into deduplicated train/val/test splits.

Two operating modes, selected by the caller:

* REAL (default; the air-gapped BigVul path): NO silent bootstrap fallback.
  If there are no ingest files, no usable rows, or the resulting train split is
  smaller than ``--min-train-rows`` (default tens of thousands), the script
  HARD-FAILS with a non-zero exit so training never proceeds on a 12-row toy
  set masquerading as BigVul. This is the bug this module fixes: previously a
  failed BigVul fetch silently wrote a 12-row bootstrap set and training ran
  against it, producing a worthless "model".

* BOOTSTRAP (``--allow-bootstrap`` or ``CCR_ALLOW_BOOTSTRAP=1``): the legacy
  smoke-test behavior — write a tiny disjoint dataset when ingest is missing.
  Only ``bootstrap_data.sh`` / ``DATA=bootstrap`` opt into this.

The merge/split/guard logic is factored into pure stdlib functions so it can be
unit-tested without torch / datasets / network.
"""
import os, json, sys, hashlib, random, glob, argparse

random.seed(42)

INGEST_DIR = "data/ingest"
DATA_DIR = "data"
# A real BigVul-derived train split is ~100k+ functions. We require at least
# tens of thousands of train rows before letting a long training run proceed;
# anything smaller means the fetch/merge produced a bootstrap-sized set.
DEFAULT_MIN_TRAIN_ROWS = 20000

# The tiny disjoint smoke-test set (train/val/test are NON-overlapping samples,
# never copies). A model trained on this is a plumbing test, not a classifier.
_BOOTSTRAP = {
    "train": [
        {"code": "void f(){char b[8]; gets(b);}", "label": 1},
        {"code": "int add(int a,int b){return a+b;}", "label": 0},
        {"code": "void g(char* s){char d[16]; strcpy(d,s);}", "label": 1},
    ],
    "val": [
        {"code": "void v(char*s){char d[16];strcpy(d,s);}", "label": 1},
        {"code": "int mul(int a,int b){return a*b;}", "label": 0},
    ],
    "test": [
        {"code": "int main(){return 0;}", "label": 0},
        {"code": "void t(char*x){char y[8];strcat(y,x);}", "label": 1},
    ],
}


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def norm_code(s: str) -> str:
    return "\n".join([ln.rstrip() for ln in s.splitlines() if ln.strip() != ""])


def _code_hash(code: str) -> str:
    return hashlib.sha1(norm_code(code).encode("utf-8")).hexdigest()


def split_of(path):
    """Map an ingest filename to its source split, if recognizable."""
    base = os.path.basename(path).lower()
    if "train" in base:
        return "train"
    if "valid" in base or "val" in base:
        return "val"
    if "test" in base:
        return "test"
    return None


def merge_rows(files):
    """Dedup-merge ingest files into (rows, src_splits).

    Pure function: ``files`` is a list of paths. ``rows`` is the deduplicated
    list of {"code","label"} dicts; ``src_splits`` maps a code hash to its
    source split name when the filename declared one. Stdlib only.
    """
    seen = set()
    rows = []
    src_splits = {}
    for p in files:
        sp = split_of(p)
        for r in iter_jsonl(p):
            code = r.get("code")
            if not code:
                continue
            h = _code_hash(code)
            if h in seen:
                continue
            seen.add(h)
            try:
                lab = int(r.get("label", 0))
            except Exception:
                lab = 0
            rows.append({"code": code, "label": lab})
            if sp:
                src_splits[h] = sp
    return rows, src_splits


def split_rows(rows, src_splits):
    """Split merged rows into (train, val, test, preserved_flag).

    Preserves the source dataset's curated splits when EVERY row declared one
    (e.g. bigvul_train/validation/test) — re-shuffling curated splits would leak
    near-duplicate before/after pairs across train/test. Otherwise falls back to
    a deterministic 80/10/10 shuffle. Pure function (uses the module's seeded
    ``random``); does no I/O.
    """
    if rows and len(src_splits) == len(rows):
        buckets = {"train": [], "val": [], "test": []}
        for r in rows:
            buckets[src_splits[_code_hash(r["code"])]].append(r)
        train, val, test = buckets["train"], buckets["val"], buckets["test"]
        for chunk in (train, val, test):
            random.shuffle(chunk)
        return train, val, test, True
    rows = list(rows)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:], False


def write_splits(train, val, test, data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    for split, chunk in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(data_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_bootstrap(data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    for split, chunk in _BOOTSTRAP.items():
        with open(os.path.join(data_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r) + "\n")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", default=INGEST_DIR)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--min-train-rows", type=int, default=DEFAULT_MIN_TRAIN_ROWS,
                    help="Real-mode guard: fail if the train split is smaller.")
    # Bootstrap fallback is OFF by default. Either flag or CCR_ALLOW_BOOTSTRAP=1
    # opts into the tiny smoke-test set when ingest is missing/empty.
    ap.add_argument("--allow-bootstrap", action="store_true",
                    default=os.environ.get("CCR_ALLOW_BOOTSTRAP", "") == "1",
                    help="Permit the tiny bootstrap fallback (smoke-test only).")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.data_dir, exist_ok=True)

    files = []
    if os.path.isdir(args.ingest_dir):
        files.extend(sorted(glob.glob(os.path.join(args.ingest_dir, "*.jsonl"))))

    rows, src_splits = ([], {})
    if files:
        rows, src_splits = merge_rows(files)

    # --- no usable ingest data ---------------------------------------------
    if not rows:
        if args.allow_bootstrap:
            print("[WARN] No usable ingest rows; writing tiny bootstrap dataset (smoke-test only).")
            write_bootstrap(args.data_dir)
            print("[OK] Wrote bootstrap data to", args.data_dir)
            return 0
        print("[FATAL] No usable ingest rows under", args.ingest_dir,
              "- the BigVul fetch did not produce data.", file=sys.stderr)
        print("[FATAL] Refusing to silently write a bootstrap set. Provision BigVul "
              "during the online window (see SETUP_4090.md / SETUP_SPARK.md), or run "
              "with DATA=bootstrap / --allow-bootstrap for a smoke-test only.",
              file=sys.stderr)
        return 2

    train, val, test, preserved = split_rows(rows, src_splits)
    write_splits(train, val, test, args.data_dir)
    print(("[OK] Preserved source dataset splits; " if preserved else "[OK] ") +
          f"merge complete: train={len(train)} val={len(val)} test={len(test)}")

    # --- guard: real run must have a real-sized train split ----------------
    if not args.allow_bootstrap and len(train) < args.min_train_rows:
        print(f"[FATAL] Train split has only {len(train)} rows (< {args.min_train_rows} "
              "required). The BigVul fetch likely failed or was truncated; refusing to "
              "train on a bootstrap-sized set. Re-provision BigVul, or pass "
              "--allow-bootstrap for a smoke-test only.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
