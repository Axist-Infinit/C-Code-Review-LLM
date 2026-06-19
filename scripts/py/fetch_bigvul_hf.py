import os, json, sys

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

def main():
    from datasets import load_dataset
    ds = load_dataset("bstee615/bigvul")
    out_root = "data/ingest"
    os.makedirs(out_root, exist_ok=True)
    total = 0
    for split in ds.keys():
        outp = os.path.join(out_root, f"bigvul_{split}.jsonl")
        n = 0
        with open(outp, "w", encoding="utf-8") as f:
            for row in ds[split]:
                # bstee615/bigvul schema: func_before is the pre-fix function
                # (vulnerable when vul=1), func_after is the patched version.
                code = try_get(row, ("func_before","func","code","functionSource","function","source"))
                if not code:
                    continue
                label = try_get(row, ("vul","target","label","is_vulnerable"))
                label = norm_label(label)
                cwe = try_get(row, ("CWE ID","cwe")) or ""
                f.write(json.dumps({"code": code, "label": label, "cwe": cwe}, ensure_ascii=False) + "\n")
                n += 1
                # the patched function is a hard negative for vulnerable rows
                after = row.get("func_after")
                if label == 1 and after and after.strip() and after != code:
                    f.write(json.dumps({"code": after, "label": 0, "cwe": ""}, ensure_ascii=False) + "\n")
                    n += 1
        print(f"[OK] Wrote {n} rows -> {outp}")
        total += n
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
