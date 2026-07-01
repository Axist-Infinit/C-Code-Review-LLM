#!/usr/bin/env python3
"""Evaluate the trained classifier on a held-out labeled test set.

Reports precision / recall / F1 / ROC-AUC / PR-AUC at the tuned threshold,
plus a small threshold sweep table, so you can see whether the model carries
real signal before trusting scan results.

Usage: python evaluate_model.py --model ./vuln-model --test data/test.jsonl

Gate mode: pass any of --min-pr-auc / --min-f1 / --min-roc-auc to turn the run
into a quality gate — a violated floor prints FAIL lines and exits non-zero so
CI / one_click can block on a degraded model.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

# torch is imported lazily inside score_dataset so that the pure metrics_at
# helper stays importable without the heavy ML stack (e.g. in unit tests / CI).

from profiles import select_profile, add_profile_arg
from local_vuln_scanner import load_model


def score_dataset(tok, model, dev, rows, batch_size=32, max_length=512):
    import torch

    scores = []
    with torch.no_grad():
        for b in range(0, len(rows), batch_size):
            batch = rows[b:b + batch_size]
            enc = tok([r["code"] for r in batch], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc).logits.float()
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return scores


def metrics_at(scores, labels, thr):
    preds = [1 if s >= thr else 0 for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(labels) if labels else 0.0
    return dict(threshold=thr, precision=prec, recall=rec, f1=f1, accuracy=acc,
                tp=tp, fp=fp, fn=fn, tn=tn)


def group_metrics(scores, labels, groups, thr):
    """Per-group confusion metrics at a threshold.

    scores/labels/groups are parallel lists; `groups[i]` is the bucket key for
    sample i (e.g. its vulnerability category or language). Returns an ordered
    dict {group: metrics_at(...)} with a per-group sample count `n`. Pure /
    torch-free so the per-category breakdown is unit-testable.
    """
    buckets = {}
    for s, y, g in zip(scores, labels, groups):
        sc, lb = buckets.setdefault(g, ([], []))
        sc.append(s)
        lb.append(y)
    out = {}
    for g in sorted(buckets, key=str):
        sc, lb = buckets[g]
        m = metrics_at(sc, lb, thr)
        m["n"] = len(sc)
        out[g] = m
    return out


def load_max_length(model_dir, default=512):
    """Tokenizer truncation length persisted by train_vuln_model.py.

    Read from <model_dir>/inference.json ("max_length"); falls back to
    `default` for models trained before the key existed or when the file is
    missing/unreadable. Pure JSON — no torch.
    """
    try:
        with open(os.path.join(model_dir, "inference.json"), "r", encoding="utf-8") as f:
            v = json.load(f).get("max_length")
        return int(v) if v else default
    except Exception:
        return default


def file_sha256(path, chunk=1 << 20):
    """Streaming sha256 of a file (never loads it whole)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dataset_provenance(paths):
    """{path: {"sha256": ..., "rows": ...}} for each existing JSONL path.

    Hash and row count come from one streaming pass; missing/empty paths are
    skipped silently so callers can pass optional files freely.
    """
    out = {}
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        h = hashlib.sha256()
        rows = 0
        last = b""
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
                rows += block.count(b"\n")
                last = block
        if last and not last.endswith(b"\n"):
            rows += 1  # final line lacked a trailing newline
        out[p] = {"sha256": h.hexdigest(), "rows": rows}
    return out


def build_provenance(model_dir=None, extra=None):
    """Machine-readable provenance stamp for benchmark/inference artifacts.

    Pure stdlib (git via subprocess, transformers probed lazily) so both the
    trainer and this evaluator can call it without torch installed. When
    `model_dir` contains model.safetensors its streamed sha256 is included.
    `extra` keys (seed, dataset hashes, ...) are merged last and win.
    """
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           capture_output=True, text=True, timeout=10)
        git_rev = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
    except Exception:
        git_rev = "unknown"
    try:
        import transformers
        tf_ver = getattr(transformers, "__version__", "unknown")
    except Exception:
        tf_ver = "unknown"
    prov = {
        "git_rev": git_rev,
        "python_version": platform.python_version(),
        "transformers_version": tf_ver,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if model_dir:
        weights = os.path.join(model_dir, "model.safetensors")
        if os.path.isfile(weights):
            prov["model_sha256"] = file_sha256(weights)
    if extra:
        prov.update(extra)
    return prov


def check_gate(metrics, floors):
    """Quality-gate check. floors: {"pr_auc": .., "f1": .., "roc_auc": ..}
    (any subset; None values are skipped). `metrics` is a flat dict; a floor
    whose metric is missing/None fails too — an unprovable gate must not pass.
    Returns a list of failure strings, empty when the gate passes. Pure.
    """
    failures = []
    for key in sorted(floors or {}):
        floor = floors[key]
        if floor is None:
            continue
        val = metrics.get(key)
        if val is None:
            failures.append(f"{key}: floor {floor:g} set but metric unavailable")
        elif val < floor:
            failures.append(f"{key}: {val:.4f} < floor {floor:g}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vuln-model")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", default=None, help="Optional: write metrics JSON here")
    ap.add_argument("--group-by", default=None,
                    help="Row field to break metrics down by (e.g. category, lang). "
                         "Useful with benchmarks/cpp_eval.jsonl to see C++ per-CWE results.")
    ap.add_argument("--save-scores", default=None,
                    help="Write per-sample JSONL {name,category,cwe,label,score,pred} "
                         "for diagnosing exactly which cases the model misses.")
    ap.add_argument("--max-length", type=int, default=None,
                    help="Tokenizer truncation length; default: the model's "
                         "inference.json max_length (512 when absent).")
    ap.add_argument("--min-pr-auc", type=float, default=None,
                    help="Gate: exit 1 if PR-AUC is below this floor.")
    ap.add_argument("--min-f1", type=float, default=None,
                    help="Gate: exit 1 if F1 at the tuned threshold is below this floor.")
    ap.add_argument("--min-roc-auc", type=float, default=None,
                    help="Gate: exit 1 if ROC-AUC is below this floor.")
    add_profile_arg(ap)
    args = ap.parse_args()

    prof_name, prof = select_profile(args.profile)
    tok, model, thr, dev = load_model(args.model, prof["classifier_dtype"])
    max_len = args.max_length or load_max_length(args.model)

    rows = []
    with open(args.test, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    labels = [int(r["label"]) for r in rows]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    print(f"[eval] {len(rows)} samples ({n_pos} vulnerable / {n_neg} clean), "
          f"device={dev}, tuned threshold={thr:.2f}, max_length={max_len}")
    if n_pos == 0 or n_neg == 0:
        print("[WARN] Test set is single-class; precision/recall are not meaningful.")

    scores = score_dataset(tok, model, dev, rows, prof["classifier_batch_size"], max_len)

    if args.save_scores:
        with open(args.save_scores, "w", encoding="utf-8") as fh:
            for r, s in zip(rows, scores):
                rec = {k: r[k] for k in ("name", "category", "cwe", "lang", "source") if k in r}
                rec.update({"label": int(r["label"]), "score": float(s),
                            "pred": int(s >= thr)})
                fh.write(json.dumps(rec) + "\n")
        print(f"[OK] per-sample scores -> {args.save_scores}")

    result = {"n": len(rows), "n_pos": n_pos, "n_neg": n_neg,
              "tuned": metrics_at(scores, labels, thr)}
    result["provenance"] = build_provenance(model_dir=args.model, extra={
        "model_dir": args.model,
        "threshold": thr,
        "max_length": max_len,
        "datasets": dataset_provenance([args.test]),
    })

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if n_pos and n_neg:
            result["roc_auc"] = float(roc_auc_score(labels, scores))
            result["pr_auc"] = float(average_precision_score(labels, scores))
    except ImportError:
        pass

    m = result["tuned"]
    print(f"\nAt tuned threshold {thr:.2f}:")
    print(f"  precision={m['precision']:.3f} recall={m['recall']:.3f} "
          f"f1={m['f1']:.3f} accuracy={m['accuracy']:.3f}")
    print(f"  tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
    if "roc_auc" in result:
        print(f"  roc_auc={result['roc_auc']:.3f} pr_auc={result['pr_auc']:.3f}")
        if result["roc_auc"] < 0.6:
            print("  [WARN] ROC-AUC < 0.6 — the model carries little signal; retrain on real data.")

    print("\nThreshold sweep:")
    print("  thr   prec   rec    f1")
    sweep = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = metrics_at(scores, labels, t)
        sweep.append(m)
        print(f"  {t:.1f}   {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}")
    result["sweep"] = sweep

    if args.group_by:
        groups = [str(r.get(args.group_by, "?")) for r in rows]
        grouped = group_metrics(scores, labels, groups, thr)
        result["by_" + args.group_by] = grouped
        print(f"\nBreakdown by {args.group_by} (at threshold {thr:.2f}):")
        print(f"  {'group':<20} n   prec   rec    f1")
        for g, m in grouped.items():
            print(f"  {g:<20} {m['n']:<3} {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] Metrics -> {args.out}")

    # --- quality gate (only active when a floor flag was given) --------------
    floors = {"pr_auc": args.min_pr_auc, "f1": args.min_f1, "roc_auc": args.min_roc_auc}
    if any(v is not None for v in floors.values()):
        flat = {"f1": result["tuned"]["f1"],
                "roc_auc": result.get("roc_auc"),
                "pr_auc": result.get("pr_auc")}
        failures = check_gate(flat, floors)
        if failures:
            for msg in failures:
                print(f"[GATE] FAIL {msg}")
            sys.exit(1)
        print("[GATE] PASS all metric floors met")


if __name__ == "__main__":
    main()
