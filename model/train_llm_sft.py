#!/usr/bin/env python3
"""Supervised fine-tuning (LoRA) for the local explainer model.

Teaches a causal LM (e.g. qwen2.5-coder) to turn a flagged C/C++ snippet into
the same structured security analysis the pipeline expects. Training data is
JSONL with {"code": ..., "analysis": ...} records — build it from your own
reviewed findings with model/build_sft_dataset.py.

Key properties:
  * Completion-only loss masking — the prompt tokens are masked (-100) so the
    model is trained ONLY on the analysis it should produce, not on echoing the
    input code. This is the standard, correct objective for instruction SFT.
  * Profile-aware precision — fp16 on the 4090 (Ada), bf16 on the DGX Spark
    (Grace-Blackwell), full precision on CPU — resolved from profiles.json.
  * Dataset validation — fails fast with a clear message on empty / malformed
    data instead of a cryptic error deep inside the Trainer.

The pure helpers below (prompt/completion formatting, label masking, validation)
import only the stdlib so they stay unit-testable without torch/transformers.
"""
import argparse
import json
import os
import sys

# Prompt / completion contract. The completion (everything from
# COMPLETION_HEADER onward) is what the model is trained to generate; the prompt
# prefix is masked out of the loss. Keep this in sync with build_sft_dataset.py.
PROMPT_HEADER = "### Code:\n"
COMPLETION_HEADER = "\n\n### Security Analysis:\n"

# Fields of the structured analysis, mirroring llm_explain's output schema so a
# fine-tuned model emits exactly what the pipeline already consumes.
# Keep in sync with model/build_sft_dataset.py ANALYSIS_FIELDS (a test asserts it).
# Reason-then-judge order: dict order == JSON emission order (py3.7+), so the model
# is trained to DESCRIBE (what/risk/vuln) before it CONCLUDES (verdict/cwe/fix).
ANALYSIS_FIELDS = ("what_code_does", "what_could_go_wrong", "vulnerability",
                   "is_vulnerable", "issue", "cwe", "severity",
                   "explanation", "fix")


def format_prompt(code):
    """Prompt prefix shown to the model (masked out of the training loss)."""
    return f"{PROMPT_HEADER}{code}{COMPLETION_HEADER}"


def format_completion(analysis):
    """The target text the model must learn to produce.

    `analysis` may be a plain string or a dict of the structured fields; a dict
    is serialized to compact JSON so the model learns the machine-readable form.
    """
    if isinstance(analysis, dict):
        return json.dumps({k: analysis.get(k) for k in ANALYSIS_FIELDS},
                          ensure_ascii=False)
    return str(analysis)


def build_example_text(code, analysis):
    """Full prompt+completion string for a record."""
    return format_prompt(code) + format_completion(analysis)


def validate_records(records):
    """Validate a list of SFT records, raising ValueError on the first problem.

    Returns the number of valid records. Catches the failure modes that would
    otherwise surface as opaque Trainer/tokenizer errors: empty dataset, missing
    'code'/'analysis', non-string code, empty analysis.
    """
    if not records:
        raise ValueError("SFT dataset is empty — nothing to train on.")
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise ValueError(f"record {i}: expected a JSON object, got {type(r).__name__}")
        if "code" not in r or "analysis" not in r:
            raise ValueError(f"record {i}: must contain both 'code' and 'analysis'")
        if not isinstance(r["code"], str) or not r["code"].strip():
            raise ValueError(f"record {i}: 'code' must be a non-empty string")
        completion = format_completion(r["analysis"]).strip()
        if not completion or completion in ("{}", "null"):
            raise ValueError(f"record {i}: 'analysis' is empty")
        # With the 9-field schema an all-empty analysis serializes to a non-empty
        # JSON object full of nulls (e.g. {"what_code_does": null, ...}), which the
        # checks above do NOT catch. Reject records whose analysis has no usable
        # text content (booleans alone are not a learnable target).
        if isinstance(r["analysis"], dict):
            has_content = any(
                v is not None and not isinstance(v, bool) and str(v).strip()
                for v in r["analysis"].values()
            )
            if not has_content:
                raise ValueError(f"record {i}: 'analysis' has no usable content")
    return len(records)


def mask_labels(prompt_ids, full_ids):
    """Build training labels for completion-only loss.

    labels = full_ids with the leading prompt tokens replaced by -100 so the
    cross-entropy loss is computed only over the completion. Tolerant of minor
    tokenization drift: masks min(len(prompt_ids), len(full_ids)) tokens.
    """
    labels = list(full_ids)
    n = min(len(prompt_ids), len(full_ids))
    for i in range(n):
        labels[i] = -100
    return labels


def tokenize_example(tok, code, analysis, max_length=2048, add_eos=True):
    """Tokenize one record into {input_ids, attention_mask, labels}.

    `tok` is any callable tokenizer exposing __call__(text, add_special_tokens=)
    and an `eos_token` attribute (real HF tokenizer or a test fake). The prompt
    is tokenized separately to find the mask boundary, then the full sequence
    (prompt + completion + EOS) is tokenized and truncated.
    """
    prompt = format_prompt(code)
    completion = format_completion(analysis)
    eos = (getattr(tok, "eos_token", "") or "") if add_eos else ""
    full_text = prompt + completion + eos

    prompt_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=True)["input_ids"]
    if max_length and len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    labels = mask_labels(prompt_ids, full_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def has_trainable_tokens(labels):
    """True if at least one label is unmasked (the example contributes loss)."""
    return any(l != -100 for l in labels)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="LoRA SFT for the explainer model")
    ap.add_argument("--base", required=True, help="Base model dir or HF id")
    ap.add_argument("--train", required=True, help="Training JSONL ({code, analysis})")
    ap.add_argument("--val", required=True, help="Validation JSONL")
    ap.add_argument("--out", required=True, help="Output adapter dir")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Force fp16 (else from profile)")
    ap.add_argument("--bf16", action="store_true", help="Force bf16 (else from profile)")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--profile", default=None,
                    help="Hardware profile for precision defaults (4090|spark|cpu)")
    return ap.parse_args(argv)


def build_base_args(args, use_fp16, use_bf16):
    """TrainingArguments kwargs shared by the eval-key spelling attempts.

    Pure dict assembly (no transformers import) so checkpoint hygiene and
    seeding are unit-testable without the ML stack.
    """
    return dict(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        report_to=[],
        save_strategy="epoch",
        save_total_limit=2,  # per-epoch adapter checkpoints must not pile up
        seed=args.seed,
        fp16=use_fp16,
        bf16=use_bf16,
    )


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def resolve_precision(args):
    """Resolve (fp16, bf16) from explicit flags, else the hardware profile.

    Explicit --fp16/--bf16 win. Otherwise the profile's classifier_dtype maps to
    the right mixed-precision mode. Returns a (fp16, bf16) bool tuple; never both.
    """
    if args.fp16 and args.bf16:
        sys.exit("[ERR] pass at most one of --fp16 / --bf16")
    if args.fp16:
        return True, False
    if args.bf16:
        return False, True
    # repo root on path so `profiles` (top-level module) imports regardless of cwd
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from profiles import select_profile, precision_flags_for_dtype

    name, prof = select_profile(args.profile)
    flags = precision_flags_for_dtype(prof.get("classifier_dtype", "float32"))
    print(f"[profile] {name}: precision fp16={flags['fp16']} bf16={flags['bf16']}")
    return flags["fp16"], flags["bf16"]


def main():
    args = parse_args()
    import torch
    import transformers as tf
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, TaskType
    print("[INFO] transformers:", tf.__version__)

    set_seed = getattr(tf, "set_seed", None)
    if set_seed:
        set_seed(args.seed)  # seeds python/numpy/torch in one call

    use_fp16, use_bf16 = resolve_precision(args)

    train_records = _load_jsonl(args.train)
    val_records = _load_jsonl(args.val)
    n_train = validate_records(train_records)
    n_val = validate_records(val_records)
    print(f"[data] {n_train} train / {n_val} val records validated")

    is_local = os.path.isdir(args.base) or os.path.isfile(os.path.join(args.base, "config.json"))
    tk = AutoTokenizer.from_pretrained(args.base, use_fast=True, local_files_only=is_local)
    if tk.pad_token is None and tk.eos_token is not None:
        tk.pad_token = tk.eos_token

    dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    if dtype is not None and not torch.cuda.is_available():
        dtype = None  # mixed precision only makes sense on GPU
    model = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=is_local, torch_dtype=dtype)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
                      lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def encode(records):
        out, skipped = [], 0
        for r in records:
            ex = tokenize_example(tk, r["code"], r["analysis"], max_length=args.max_length)
            if not has_trainable_tokens(ex["labels"]):
                skipped += 1  # prompt alone exceeded max_length -> no completion to learn
                continue
            out.append(ex)
        if skipped:
            print(f"[warn] skipped {skipped} record(s) whose prompt filled the whole "
                  f"--max-length {args.max_length} window (no completion tokens left)")
        return out

    train_enc = encode(train_records)
    val_enc = encode(val_records)
    if not train_enc:
        sys.exit("[ERR] no trainable examples after encoding; raise --max-length "
                 "or shorten code snippets")

    pad_id = tk.pad_token_id if tk.pad_token_id is not None else 0

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)  # padding never contributes loss
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    base_args = build_base_args(args, use_fp16, use_bf16)
    # HF renamed evaluation_strategy -> eval_strategy; try modern spelling first.
    for eval_key in ("eval_strategy", "evaluation_strategy", None):
        try:
            kw = dict(base_args)
            if eval_key:
                kw[eval_key] = "epoch"
            targs = TrainingArguments(**kw)
            break
        except TypeError:
            continue

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_enc, eval_dataset=val_enc,
        tokenizer=tk, data_collator=collate,
    )
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tk.save_pretrained(args.out)
    print("[OK] Saved SFT adapter to", args.out)


if __name__ == "__main__":
    main()
