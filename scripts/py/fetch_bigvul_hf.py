import os, json, sys, argparse, hashlib

def try_get(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def norm_label(v):
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("1","true","yes","vul","vulnerable"): return 1
    if s in ("0","false","no","nonvul","clean","safe"): return 0
    try: return int(s)
    except Exception: return 0

def norm_lang(v):
    """Normalize a dataset language tag; BigVul is C/C++ so the default is 'c'.

    'cpp' (not 'c++') matches the value benchmarks/cpp_eval.jsonl already uses,
    keeping --group-by lang buckets consistent across datasets.
    """
    s = str(v or "").strip().lower()
    if not s:
        return "c"
    if s in ("c++", "cpp", "cxx"):
        return "cpp"
    return s

def rows_from_record(row):
    """Map one BigVul record to output rows (main + optional patched twin). Pure.

    Every row is tagged with source ('func_before' = the pre-fix function,
    'func_after' = the patched hard negative) and lang, so downstream eval can
    slice metrics by them (e.g. unpaired-negative analysis, per-language F1).
    """
    # bstee615/bigvul schema: func_before is the pre-fix function
    # (vulnerable when vul=1), func_after is the patched version.
    code = try_get(row, ("func_before","func","code","functionSource","function","source"))
    if not code:
        return []
    label = norm_label(try_get(row, ("vul","target","label","is_vulnerable")))
    cwe = try_get(row, ("CWE ID","cwe")) or ""
    lang = norm_lang(try_get(row, ("lang","language")))
    rows = [{"code": code, "label": label, "cwe": cwe,
             "source": "func_before", "lang": lang}]
    # the patched function is a hard negative for vulnerable rows
    after = row.get("func_after")
    if label == 1 and after and after.strip() and after != code:
        rows.append({"code": after, "label": 0, "cwe": "",
                     "source": "func_after", "lang": lang})
    return rows

def file_sha256(path, chunk=1 << 20):
    """Streaming sha256 so provenance of multi-hundred-MB shards stays cheap."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Fetch BigVul from HF into data/ingest JSONL shards")
    ap.add_argument("--revision", default=os.environ.get("BIGVUL_REVISION") or None,
                    help="Dataset revision (tag/branch/commit) to pin the fetch to; "
                         "defaults to env BIGVUL_REVISION when set.")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    from datasets import load_dataset
    kw = {"revision": args.revision} if args.revision else {}
    if args.revision:
        print(f"[INFO] Loading bstee615/bigvul @ revision {args.revision}")
    ds = load_dataset("bstee615/bigvul", **kw)
    out_root = "data/ingest"
    os.makedirs(out_root, exist_ok=True)
    total = 0
    outputs = []
    for split in ds.keys():
        outp = os.path.join(out_root, f"bigvul_{split}.jsonl")
        n = 0
        with open(outp, "w", encoding="utf-8") as f:
            for row in ds[split]:
                for rec in rows_from_record(row):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
        print(f"[OK] Wrote {n} rows -> {outp}")
        outputs.append((outp, n))
        total += n
    for outp, n in outputs:
        print(f"[prov] {outp}: rows={n} sha256={file_sha256(outp)}")
    print(f"[OK] BigVul(HF) total rows: {total}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail LOUD (non-zero). A swallowed fetch failure previously let the
        # pipeline silently degrade to a 12-row bootstrap set and train a
        # worthless "model". The caller must re-run the online provisioning
        # window, or explicitly choose DATA=bootstrap for a smoke-test.
        print("[FATAL] BigVul(HF) fetch failed:", e, file=sys.stderr)
        sys.exit(1)
