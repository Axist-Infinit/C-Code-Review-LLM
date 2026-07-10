#!/usr/bin/env python3
"""Pre-generate the EXACT surface-review prompt bundle for each corpus file.

Each Opus reviewer (and, later, the fine-tuned local model) must see the same
system prompt + KB hints + line-numbered context the production surface lane
feeds. This writes, per manifest entry, a self-contained bundle:
    <out>/<split>/<id>/system.txt   (SYSTEM_PROMPT + matched KB hints)
    <out>/<split>/<id>/context.txt  (the fenced, line-numbered code)
and rewrites the manifest with `prompt_dir` and `review_out` so the reviewer
workflow just points each subagent at a bundle and an output path.

    python3 scripts/py/emit_surface_prompts.py --manifest corpus/train_manifest.json \
        --split train --out-dir corpus/prompts --review-dir corpus/reviews
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import surface_review as sr


def main():
    ap = argparse.ArgumentParser(description="Emit per-file surface prompt bundles")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True, help="train|eval (namespaces the bundles)")
    ap.add_argument("--out-dir", default="corpus/prompts")
    ap.add_argument("--review-dir", default="corpus/reviews")
    ap.add_argument("--no-kb", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    kb = sr.load_kb()
    enriched = []
    for i, m in enumerate(manifest):
        src = m["file"]
        context = sr.build_context([src])
        extra = "" if args.no_kb else sr.kb_notes_block(sr.match_kb(context, kb))
        sid = f"{args.split}_{i:04d}"
        bundle = os.path.join(args.out_dir, args.split, sid)
        os.makedirs(bundle, exist_ok=True)
        open(os.path.join(bundle, "system.txt"), "w", encoding="utf-8").write(sr.SYSTEM_PROMPT + extra)
        open(os.path.join(bundle, "context.txt"), "w", encoding="utf-8").write(context)
        review_out = os.path.join(args.review_dir, args.split, f"{sid}.claude.json")
        m2 = dict(m, prompt_dir=bundle, review_out=review_out,
                  system_path=os.path.join(bundle, "system.txt"),
                  context_path=os.path.join(bundle, "context.txt"))
        enriched.append(m2)
    os.makedirs(args.review_dir, exist_ok=True)
    out_manifest = args.manifest.replace(".json", ".prompts.json")
    json.dump(enriched, open(out_manifest, "w"), indent=2)
    print(f"[OK] {len(enriched)} bundles -> {args.out_dir}/{args.split}/")
    print(f"[OK] enriched manifest -> {out_manifest}")


if __name__ == "__main__":
    main()
