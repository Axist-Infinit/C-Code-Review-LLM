#!/usr/bin/env python3
"""Score an Ollama model's surface reviews against the Opus eval references.

Runs a given Ollama model over every held-out eval file (that has an Opus gold
review), scores each with score_review.py, and reports the mean composite +
component breakdown. Run it once per model tag to get a clean before/after:

    # baseline (un-tuned) vs the distilled model, identical inference config
    python3 scripts/py/eval_surface_model.py --model qwen2.5-coder:14b \
        --manifest corpus/eval_manifest.prompts.json --out-dir corpus/eval_runs
    python3 scripts/py/eval_surface_model.py --model qwen2.5-coder-surface:14b \
        --manifest corpus/eval_manifest.prompts.json --out-dir corpus/eval_runs

Single-pass by default (no critic) so it isolates the model's raw review quality
and stays fast; --critic adds the completeness pass (matching the deployed lane).
"""
import argparse
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import surface_review as sr
import score_review as scorer


def review_one(url, model, src_file, num_ctx, use_kb, use_critic):
    context = sr.build_context([src_file])
    extra = sr.kb_notes_block(sr.match_kb(context, sr.load_kb())) if use_kb else ""
    review = sr.ollama_review(url, model, context, num_ctx, extra)
    if use_critic:
        try:
            crit = sr.ollama_critique(url, model, context, review, num_ctx, extra)
            if sr.critique_acceptable(review, crit):
                review = crit
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            pass
    review["findings"] = sr.canonicalize_cwes(review.get("findings", []))
    return review


def main():
    ap = argparse.ArgumentParser(description="Score an Ollama model's surface reviews vs Opus refs")
    ap.add_argument("--model", required=True, help="Ollama model tag to evaluate")
    ap.add_argument("--manifest", default="corpus/eval_manifest.prompts.json")
    ap.add_argument("--out-dir", default="corpus/eval_runs")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N eval files (0=all)")
    ap.add_argument("--no-kb", action="store_true")
    ap.add_argument("--critic", action="store_true", help="Add the 2nd completeness pass")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    args = ap.parse_args()
    url = sr._normalize_ollama_url(args.ollama_url)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    items = [m for m in manifest if os.path.exists(m.get("review_out", ""))]
    if args.limit:
        items = items[:args.limit]
    if not items:
        sys.exit("[ERR] no eval items with an Opus reference found; generate refs first")

    tag = args.model.replace(":", "_").replace("/", "_")
    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[eval] {args.model}: {len(items)} eval file(s), critic={args.critic}, kb={not args.no_kb}")

    per_file, comps = [], {"grounding": [], "coverage": [], "recall": [], "classification": []}
    composites = []
    for i, m in enumerate(items):
        src, ref_path = m["file"], m["review_out"]
        try:
            cand = review_one(url, args.model, src, args.num_ctx, not args.no_kb, args.critic)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
            print(f"  [{i+1}/{len(items)}] {os.path.basename(src)}: FAILED ({e}); skipping")
            continue
        json.dump(cand, open(os.path.join(run_dir, os.path.basename(ref_path).replace(".claude.json", ".cand.json")), "w"), indent=2)
        ref = json.load(open(ref_path, encoding="utf-8"))
        srcmap = scorer.auto_sources([ref, cand])
        res = scorer.score_one(cand, ref, srcmap)
        composites.append(res["composite"])
        for k in comps:
            comps[k].append(res["components"][k])
        per_file.append({"file": src, "cwe": m.get("cwe_truth"), "composite": res["composite"],
                         "components": res["components"]})
        print(f"  [{i+1}/{len(items)}] {m.get('cwe_truth','?'):9} {os.path.basename(src):28} "
              f"composite={res['composite']:.1f} (g={res['components']['grounding']:.2f} "
              f"cov={res['components']['coverage']:.2f} rec={res['components']['recall']:.2f})")

    n = len(composites)
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
    summary = {"model": args.model, "n_scored": n, "critic": args.critic,
               "mean_composite": round(sum(composites) / n, 2) if n else 0.0,
               "mean_components": {k: mean(v) for k, v in comps.items()},
               "per_file": per_file}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    print(f"\n{'='*60}\n{args.model}  —  MEAN COMPOSITE {summary['mean_composite']}/100  (n={n})")
    mc = summary["mean_components"]
    print(f"  grounding {mc['grounding']:.2f}  coverage {mc['coverage']:.2f}  "
          f"recall {mc['recall']:.2f}  classification {mc['classification']:.2f}")
    print(f"[OK] -> {run_dir}/summary.json")


if __name__ == "__main__":
    main()
