#!/usr/bin/env python3
"""Build SURFACE-review SFT data — distil Claude's reviews into the local model.

The existing model/build_sft_dataset.py makes finding-granularity records for the
llm_explain lane (one snippet -> one finding's analysis). This builds the harder,
richer target: WHOLE-FILE attack-surface reviews, where the completion is the full
review JSON — including the two line-numbered walkthroughs (what_the_code_does /
what_could_go_wrong) that are the whole point of the surface lane.

The teacher is Claude Opus (produced by compare_review.py / an Opus subagent on
the IDENTICAL surface_review prompt + context). Distilling those reviews teaches
the local qwen model to answer "what is this code doing?" and "what could go
wrong?" the way Claude does — grounded, thorough, top-to-bottom.

Each record is captured in TWO forms so it can train either way:
  * chat form   {system, context, review} — matches Ollama inference EXACTLY
                (system=SYSTEM_PROMPT+KB, user=line-numbered context, assistant=
                review JSON). Use this with a chat-template trainer, then merge the
                LoRA -> GGUF -> Ollama so training and serving agree. RECOMMENDED
                for the Ollama-served surface lane.
  * flat form   {code, analysis} — drop-in for the current model/train_llm_sft.py
                (code = system+context, analysis = the review JSON string, passed
                through format_completion verbatim). Serve via transformers using
                the same format_prompt wrapper.

Input is a manifest pairing each reference review with the sources it ran on:
    [{"review": "scan_out/compare/dhcp_claude_ref.json",
      "src": ["playground/dhcp-internal.h", "playground/dhcp-protocol.h"]}, ...]

    python model/build_surface_sft.py --manifest refs.json --out-dir data
    python model/build_surface_sft.py --review REF.json --src a.h b.h --out-dir data

Pure stdlib + surface_review helpers, so it stays air-gap-safe and unit-testable.
"""
import argparse
import json
import os
import sys

# repo root on path so the top-level modules import regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import surface_review as sr
from build_sft_dataset import split_records, _write_jsonl  # reuse the split/writer

# Metadata keys compare_review.py stamps onto a review; never a training target.
_TRANSIENT = ("_model", "_usage", "_raw", "_source", "_skipped")


def clean_review(review):
    """Drop transient metadata and empty scaffolding so the model learns the
    review, not the plumbing. Keeps the schema keys the renderer expects."""
    out = {k: v for k, v in review.items() if k not in _TRANSIENT}
    return out


def review_is_usable(review):
    """A teacher review must carry the two walkthroughs (the surface lane's point)
    OR at least one finding — otherwise there is nothing worth distilling."""
    does = review.get("what_the_code_does")
    wrong = review.get("what_could_go_wrong")
    has_walk = any(isinstance(v, (list, str)) and v for v in (does, wrong))
    return has_walk or bool(review.get("findings"))


def build_context_and_system(src_paths, use_kb=True):
    """Reconstruct the EXACT (system, context) the surface lane feeds the model."""
    context = sr.build_context(src_paths)
    extra = ""
    if use_kb:
        extra = sr.kb_notes_block(sr.match_kb(context, sr.load_kb()))
    return sr.SYSTEM_PROMPT + extra, context


def record_from_pair(review_path, src_paths, use_kb=True):
    """Build one SFT record (both chat + flat forms) from a reference review."""
    review = clean_review(json.load(open(review_path, "r", encoding="utf-8")))
    if not review_is_usable(review):
        return None
    system, context = build_context_and_system(src_paths, use_kb=use_kb)
    completion = json.dumps(review, ensure_ascii=False)  # compact, like format_completion
    return {
        # chat form — matches Ollama inference exactly
        "system": system,
        "context": context,
        "review": review,
        # flat form — drop-in for the existing train_llm_sft.py {code, analysis}
        "code": f"{system}\n\n{context}",
        "analysis": completion,
        # provenance for auditing / curation
        "source_files": list(src_paths),
        "review_path": review_path,
    }


def load_manifest(args):
    """Return a list of {review, src[]} specs from --manifest or --review/--src."""
    if args.manifest:
        specs = json.load(open(args.manifest, "r", encoding="utf-8"))
        if not isinstance(specs, list):
            sys.exit("[ERR] manifest must be a JSON list of {review, src} objects")
        return specs
    if args.review and args.src:
        return [{"review": args.review, "src": args.src}]
    sys.exit("[ERR] provide --manifest, or both --review and --src")


def main():
    ap = argparse.ArgumentParser(description="Distil Claude surface reviews into SFT data")
    ap.add_argument("--manifest", help="JSON list of {review, src:[...]} pairs")
    ap.add_argument("--review", help="A single reference review JSON")
    ap.add_argument("--src", nargs="+", help="Source file(s) that review ran on")
    ap.add_argument("--out-dir", default="data", help="Where to write surface_sft_{train,val}.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-kb", action="store_true",
                    help="Build the prompt WITHOUT KB hints (match a --no-kb inference lane)")
    args = ap.parse_args()

    specs = load_manifest(args)
    records, skipped = [], 0
    for spec in specs:
        rev = spec.get("review")
        src = spec.get("src") or []
        if not rev or not src:
            print(f"[warn] skipping malformed spec: {spec}")
            skipped += 1
            continue
        rec = record_from_pair(rev, src, use_kb=not args.no_kb)
        if rec is None:
            print(f"[warn] {rev}: no walkthroughs or findings to distil; skipped")
            skipped += 1
            continue
        records.append(rec)

    if not records:
        sys.exit("[ERR] no usable surface-SFT records produced")
    print(f"[data] {len(records)} teacher review(s) captured, {skipped} skipped")

    train, val = split_records(records, val_frac=args.val_frac, seed=args.seed)
    train_path = os.path.join(args.out_dir, "surface_sft_train.jsonl")
    val_path = os.path.join(args.out_dir, "surface_sft_val.jsonl")
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    print(f"[OK] {len(train)} train -> {train_path}")
    print(f"[OK] {len(val)} val   -> {val_path}")
    print("[next] flat form is drop-in for model/train_llm_sft.py:\n"
          f"       python model/train_llm_sft.py --base ./llm-base "
          f"--train {train_path} --val {val_path} --out ./llm-surface-lora\n"
          "       (chat form {system,context,review} is also in each record for a "
          "chat-template trainer that matches Ollama serving.)")


if __name__ == "__main__":
    main()
