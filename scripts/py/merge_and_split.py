import os, json, sys, hashlib, random, glob

random.seed(42)

ingest_dir = "data/ingest"
os.makedirs("data", exist_ok=True)

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def norm_code(s: str) -> str:
    return "\n".join([ln.rstrip() for ln in s.splitlines() if ln.strip() != ""])

def main():
    files = []
    if os.path.isdir(ingest_dir):
        files.extend(sorted(glob.glob(os.path.join(ingest_dir, "*.jsonl"))))
    if not files:
        print("[WARN] No ingest files found; writing tiny bootstrap dataset.")
        # tiny bootstrap
        os.makedirs("data", exist_ok=True)
        samples = [
            {"code": "void f(){char b[8]; gets(b);}","label":1},
            {"code": "int add(int a,int b){return a+b;}","label":0},
            {"code": "void g(char* s){char d[16]; strcpy(d,s);}","label":1},
            {"code": "int main(){return 0;}","label":0},
        ]
        for split, chunk in (("train", samples[:3]),("val", samples[1:3]),("test", samples[3:])):
            with open(f"data/{split}.jsonl","w",encoding="utf-8") as f:
                for r in chunk: f.write(json.dumps(r)+"\n")
        print("[OK] Wrote bootstrap data to ./data")
        return

    def split_of(path):
        """Map an ingest filename to its source split, if recognizable."""
        base = os.path.basename(path).lower()
        if "train" in base: return "train"
        if "valid" in base or "val" in base: return "val"
        if "test" in base: return "test"
        return None

    seen = set()
    rows = []
    src_splits = {}
    for p in files:
        sp = split_of(p)
        for r in iter_jsonl(p):
            code = r.get("code")
            lab = r.get("label", 0)
            if not code: continue
            h = hashlib.sha1(norm_code(code).encode("utf-8")).hexdigest()
            if h in seen: continue
            seen.add(h)
            try:
                lab = int(lab)
            except Exception:
                lab = 0
            rows.append({"code": code, "label": lab})
            if sp:
                src_splits[h] = sp

    if not rows:
        print("[WARN] Ingest rows empty; writing bootstrap data.")
        samples = [
            {"code": "void f(){char b[8]; gets(b);}","label":1},
            {"code": "int add(int a,int b){return a+b;}","label":0},
            {"code": "void g(char* s){char d[16]; strcpy(d,s);}","label":1},
            {"code": "int main(){return 0;}","label":0},
        ]
        for split, chunk in (("train", samples[:3]),("val", samples[1:3]),("test", samples[3:])):
            with open(f"data/{split}.jsonl","w",encoding="utf-8") as f:
                for r in chunk: f.write(json.dumps(r)+"\n")
        print("[OK] Wrote bootstrap data to ./data")
        return

    # Preserve the source dataset's curated splits when every ingest file
    # declares one (e.g. bigvul_train/validation/test). Re-shuffling curated
    # splits would leak near-duplicate before/after pairs across train/test.
    if len(src_splits) == len(rows):
        buckets = {"train": [], "val": [], "test": []}
        for r in rows:
            h = hashlib.sha1(norm_code(r["code"]).encode("utf-8")).hexdigest()
            buckets[src_splits[h]].append(r)
        train, val, test = buckets["train"], buckets["val"], buckets["test"]
        for chunk in (train, val, test):
            random.shuffle(chunk)
        print("[OK] Preserved source dataset splits")
    else:
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n*0.8)
        n_val = int(n*0.1)
        train = rows[:n_train]
        val = rows[n_train:n_train+n_val]
        test = rows[n_train+n_val:]

    for split, chunk in (("train", train),("val", val),("test", test)):
        with open(f"data/{split}.jsonl","w",encoding="utf-8") as f:
            for r in chunk: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    print(f"[OK] Merge complete: train={len(train)} val={len(val)} test={len(test)}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("[WARN] merge script failed:", e)
        # best-effort bootstrap
        os.makedirs("data", exist_ok=True)
        samples = [
            {"code": "void f(){char b[8]; gets(b);}","label":1},
            {"code": "int add(int a,int b){return a+b;}","label":0},
            {"code": "void g(char* s){char d[16]; strcpy(d,s);}","label":1},
            {"code": "int main(){return 0;}","label":0},
        ]
        for split, chunk in (("train", samples[:3]),("val", samples[1:3]),("test", samples[3:])):
            with open(f"data/{split}.jsonl","w",encoding="utf-8") as f:
                for r in chunk: f.write(json.dumps(r)+"\n")
        print("[OK] Wrote bootstrap data to ./data")
        sys.exit(0)
