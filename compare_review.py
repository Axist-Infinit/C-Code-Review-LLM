#!/usr/bin/env python3
"""Compare a local-model security review against Claude Opus on the SAME snippet.

Runs the identical contract/attack-surface review (surface_review.SYSTEM_PROMPT +
the matched offline KB hints + the same fenced snippet) against BOTH the local
Ollama model and Claude Opus 4.8, then prints them side by side so you can see,
finding-for-finding, how the local model measures up on your own code. The only
variable is the model — same prompt, same hints, same input.

    python compare_review.py path/to/snippet.c          # a file (or a directory)
    python compare_review.py --code 'void f(char*s){char b[8];strcpy(b,s);}'
    echo '<code>' | python compare_review.py -           # read snippet from stdin

The Claude side needs the official SDK and a key:  pip install anthropic  and
export ANTHROPIC_API_KEY=...  (it is the opt-in cloud tier and is the only part
of this project that leaves the machine). The local side uses the same Ollama
setup as surface_review.py. Use --no-claude / --no-local to run just one side.
"""
import argparse
import json
import os
import sys
import urllib.error

import surface_review as sr
from local_vuln_scanner import list_sources
from profiles import select_profile, add_profile_arg

CLAUDE_MODEL = "claude-opus-4-8"  # see the claude-api reference for current IDs


def context_from_code(code, label="snippet"):
    body = f"/* ===== {label} ===== */\n{sr.number_lines(code)}"
    return (f"Files under review: {label}\n\n"
            f"{sr.SNIPPET_BEGIN}\n{body}\n{sr.SNIPPET_END}")


def _strip_fences(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def local_review(context, extra_system, url, model, num_ctx):
    """The local model's review: enumerate-first pass + completeness critic.

    Mirrors surface_review's pipeline so the comparison's local side is the same
    product the standalone tool produces: a thorough walkthrough, then a critic
    pass (degrading gracefully to pass-1 if it fails/truncates), then deterministic
    CWE canonicalization.
    """
    review = sr.ollama_review(url, model, context, num_ctx, extra_system)
    try:
        critiqued = sr.ollama_critique(url, model, context, review, num_ctx, extra_system)
        if sr.critique_acceptable(review, critiqued):
            review = critiqued
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        pass  # keep the complete pass-1 walkthrough
    review["findings"] = sr.canonicalize_cwes(review.get("findings", []))
    return review


def parse_claude_reply(text):
    """Parse Claude's JSON reply and apply the SAME deterministic CWE
    canonicalisation the local side gets, so the printed comparison is
    symmetric (docstring promise: the only variable is the model)."""
    try:
        review = json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:
        review = {"subsystem": "(unparsed)", "findings": [],
                  "summary": f"Claude reply was not valid JSON ({e}).", "_raw": text}
    review["findings"] = sr.canonicalize_cwes(review.get("findings", []))
    return review


def claude_review(context, extra_system, model, max_tokens=32000):
    """Claude Opus's review of the SAME prompt. Streams; parses the JSON reply."""
    try:
        import anthropic
    except ImportError:
        sys.exit("[ERR] the Claude side needs the SDK: pip install anthropic")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("[ERR] set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) for the Claude side")
    client = anthropic.Anthropic()  # resolves credentials from the environment
    # Stream because a thorough review with adaptive thinking can be a long
    # response; get_final_message() reassembles it without HTTP-timeout risk.
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},   # Opus 4.8: adaptive only (no budget_tokens)
        system=sr.SYSTEM_PROMPT + extra_system,
        messages=[{"role": "user", "content": context}],
    ) as stream:
        msg = stream.get_final_message()
    text = next((b.text for b in msg.content if b.type == "text"), "")
    review = parse_claude_reply(text)
    review["_model"] = msg.model
    review["_usage"] = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
    return review


_SEV = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _summary_rows(review):
    rows = []
    for f in sorted(review.get("findings", []), key=lambda x: _SEV.get(str(x.get("severity", "")).lower(), 9)):
        cve = f.get("cve_analog") or ""
        rows.append(f"[{str(f.get('severity', '')).upper():8s}] {f.get('cwe', ''):9s} "
                    f"{(cve or '-'):16s} {f.get('title', '')[:52]}")
    return rows


def print_comparison(local, claude, out_dir="scan_out/compare"):
    print("\n" + "=" * 78)
    print("COMPARISON  ·  local model  vs  Claude Opus")
    print("=" * 78)
    for name, rev in (("LOCAL", local), ("CLAUDE", claude)):
        if rev is None:
            continue
        sub = rev.get("subsystem", "?")
        extra = ""
        if "_usage" in rev:
            extra = f"  ({rev.get('_model', CLAUDE_MODEL)}, {rev['_usage']['output']} out-tok)"
        rows = _summary_rows(rev)
        print(f"\n### {name}: {len(rows)} findings — subsystem: {sub}{extra}")
        for i, r in enumerate(rows, 1):
            print(f"  {i:2d}. {r}")
    if local is not None and claude is not None:
        ln, cn = len(local.get("findings", [])), len(claude.get("findings", []))
        print(f"\n  -> local {ln} findings vs Claude {cn}. "
              f"Full reports written to {out_dir}/.")


def main():
    ap = argparse.ArgumentParser(
        description="Compare a local-model vs Claude Opus security review of the same snippet")
    ap.add_argument("paths", nargs="*", help="Source file(s)/dir(s), or '-' for stdin")
    ap.add_argument("--code", help="Inline code snippet to review")
    ap.add_argument("--out", default="scan_out/compare", help="Output directory")
    ap.add_argument("--model", default=None, help="Override the local Ollama model")
    ap.add_argument("--claude-model", default=CLAUDE_MODEL, help="Claude model id")
    ap.add_argument("--num-ctx", type=int, default=16384, help="Local model context window")
    ap.add_argument("--no-local", action="store_true", help="Skip the local model side")
    ap.add_argument("--no-claude", action="store_true", help="Skip the Claude side")
    ap.add_argument("--no-kb", action="store_true", help="Skip the offline retrieval hints")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    add_profile_arg(ap)
    args = ap.parse_args()

    # Build the one shared review context.
    if args.code:
        context, label_paths = context_from_code(args.code), ["snippet"]
    elif args.paths == ["-"]:
        context, label_paths = context_from_code(sys.stdin.read(), "stdin"), ["stdin"]
    elif args.paths:
        files = []
        for p in args.paths:
            files.extend(list_sources(p))
        if not files:
            sys.exit("[ERR] no C/C++ sources found in the given paths")
        context, label_paths = sr.build_context(files), files
    else:
        ap.error("provide a file/dir, --code, or '-' for stdin")

    extra_system = ""
    if not args.no_kb:
        matched = sr.match_kb(context, sr.load_kb())
        extra_system = sr.kb_notes_block(matched)
        if matched:
            print(f"[kb] {len(matched)} hint(s) injected (same for both models): "
                  f"{', '.join(e['id'] for e in matched)}")

    prof_name, prof = select_profile(args.profile)
    local_model = args.model or prof["ollama_model"]
    os.makedirs(args.out, exist_ok=True)

    local = claude = None
    if not args.no_local:
        print(f"[local] reviewing on {local_model} ...")
        local = local_review(context, extra_system, args.ollama_url, local_model, args.num_ctx)
        json.dump(local, open(os.path.join(args.out, "local.json"), "w"), indent=2)
        open(os.path.join(args.out, "local.md"), "w").write(sr.render_markdown(local, label_paths))
    if not args.no_claude:
        print(f"[claude] reviewing on {args.claude_model} ...")
        claude = claude_review(context, extra_system, args.claude_model)
        json.dump(claude, open(os.path.join(args.out, "claude.json"), "w"), indent=2)
        open(os.path.join(args.out, "claude.md"), "w").write(sr.render_markdown(claude, label_paths))

    print_comparison(local, claude, args.out)


if __name__ == "__main__":
    main()
