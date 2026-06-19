#!/usr/bin/env python3
"""Evaluate the trained classifier on a held-out labeled test set.

Reports precision / recall / F1 / ROC-AUC / PR-AUC at the tuned threshold,
plus a small threshold sweep table, so you can see whether the model carries
real signal before trusting scan results.

Usage: python evaluate_model.py --model ./vuln-model --test data/test.jsonl
"""
import argparse
import json

# torch is imported lazily inside score_dataset so that the pure metrics_at
# helper stays importable without the heavy ML stack (e.g. in unit tests / CI).

from profiles import select_profile, add_profile_arg
from local_vuln_scanner import load_model


def score_dataset(tok, model, dev, rows, batch_size=32):
    import torch

    scores = []
    with torch.no_grad():
        for b in range(0, len(rows), batch_size):
            batch = rows[b:b + batch_size]
            enc = tok([r["code"] for r in batch], truncation=True, max_length=512,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vuln-model")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", default=None, help="Optional: write metrics JSON here")
    add_profile_arg(ap)
    args = ap.parse_args()

    prof_name, prof = select_profile(args.profile)
    tok, model, thr, dev = load_model(args.model, prof["classifier_dtype"])

    rows = []
    with open(args.test, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    labels = [int(r["label"]) for r in rows]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    print(f"[eval] {len(rows)} samples ({n_pos} vulnerable / {n_neg} clean), "
          f"device={dev}, tuned threshold={thr:.2f}")
    if n_pos == 0 or n_neg == 0:
        print("[WARN] Test set is single-class; precision/recall are not meaningful.")

    scores = score_dataset(tok, model, dev, rows, prof["classifier_batch_size"])

    result = {"n": len(rows), "n_pos": n_pos, "n_neg": n_neg,
              "tuned": metrics_at(scores, labels, thr)}

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

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[OK] Metrics -> {args.out}")


if __name__ == "__main__":
    main()
