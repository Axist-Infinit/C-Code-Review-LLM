#!/usr/bin/env python3
"""QLoRA fine-tune the local model to produce Claude-quality SURFACE reviews.

This distils the whole-file attack-surface review (the two line-numbered
walkthroughs + findings) from Claude Opus teacher reviews into the local qwen
model. It differs from model/train_llm_sft.py on three axes that matter here:

  * CHAT TEMPLATE, not a "### Code:" wrapper. Each example is the real
    [system, user, assistant] turn set — system = the surface SYSTEM_PROMPT (+KB),
    user = the line-numbered context, assistant = the review JSON — tokenised with
    the model's own chat template. This matches how Ollama serves the surface lane
    (ChatML), so after LoRA->GGUF->Ollama, training and serving agree.
  * 4-bit QLoRA. A 14B in fp16 (~28 GB) will not fit a 24 GB card; loading the
    base in nf4 (bitsandbytes) + LoRA + gradient checkpointing does. This is what
    makes fine-tuning the deployable 14B feasible on a single 4090.
  * Completion-only masking on the ASSISTANT turn — the system+user prefix is
    masked (-100), so loss is computed only over the review the model must learn
    to generate.

Data is the chat-form records from model/build_surface_sft.py:
    {"system": ..., "context": ..., "review": {...}}  (review may be a dict or str)

    python model/train_surface_sft.py --base Qwen/Qwen2.5-Coder-14B-Instruct \
        --train data/surface_sft_train.jsonl --val data/surface_sft_val.jsonl \
        --out ./llm-surface-lora --max-length 8192

The pure helpers (message assembly, mask boundary, validation) import only the
stdlib so they stay unit-testable without torch/transformers/bitsandbytes.
"""
import argparse
import json
import os
import sys

# qwen2.5 (Qwen2 arch) linear projections — LoRA on all of them for max capacity.
QWEN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]


def _completion_text(review):
    """The assistant turn: the review JSON as a compact string (dict -> JSON)."""
    if isinstance(review, str):
        return review
    return json.dumps(review, ensure_ascii=False)


def build_messages(rec):
    """[system, user, assistant] messages for one record. `system` optional."""
    msgs = []
    if rec.get("system"):
        msgs.append({"role": "system", "content": rec["system"]})
    msgs.append({"role": "user", "content": rec.get("context", "")})
    msgs.append({"role": "assistant", "content": _completion_text(rec.get("review"))})
    return msgs


def validate_records(records):
    """Fail fast with a clear message on empty/malformed data. Returns the count."""
    if not records:
        raise ValueError("surface-SFT dataset is empty — nothing to train on.")
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise ValueError(f"record {i}: expected an object, got {type(r).__name__}")
        if not str(r.get("context", "")).strip():
            raise ValueError(f"record {i}: empty 'context'")
        comp = _completion_text(r.get("review")).strip()
        if not comp or comp in ("{}", "null", '""'):
            raise ValueError(f"record {i}: empty 'review' target")
    return len(records)


def mask_labels(prompt_len, full_ids):
    """Completion-only labels: mask the first `prompt_len` (system+user) tokens."""
    labels = list(full_ids)
    for i in range(min(prompt_len, len(full_ids))):
        labels[i] = -100
    return labels


def tokenize_example(tok, rec, max_length=8192):
    """Tokenise one record with the model's chat template; mask the prompt.

    prompt = chat_template([system,user], add_generation_prompt=True)
    full   = chat_template([system,user,assistant])
    labels = full with the prompt span masked.

    Returns (features, full_len). We NEVER truncate the completion — a cut review
    JSON teaches the model to emit invalid/partial output. The caller drops
    examples whose full_len exceeds max_length instead (see encode()).
    """
    msgs = build_messages(rec)
    prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True,
                                         tokenize=True)
    full_ids = tok.apply_chat_template(msgs, add_generation_prompt=False,
                                       tokenize=True)
    full_len = len(full_ids)
    labels = mask_labels(len(prompt_ids), full_ids)
    return ({"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
             "labels": labels}, full_len)


def has_trainable_tokens(labels):
    return any(l != -100 for l in labels)


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="QLoRA SFT for the surface-review model")
    ap.add_argument("--base", required=True, help="Base model dir or HF id (Instruct)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-4bit", action="store_true",
                    help="Load the base in bf16 instead of nf4 4-bit (needs the VRAM)")
    ap.add_argument("--liger", action="store_true",
                    help="Use Liger fused linear+cross-entropy (qwen2). At vocab "
                         "152k the plain CE materialises an (L x 152064) fp32 logit "
                         "tensor (~7.5 GB at L=12288); the fused kernel chunks it. "
                         "This is what makes long-context training affordable.")
    return ap.parse_args(argv)


def check_alloc_conf(alloc_conf, four_bit):
    """bitsandbytes + expandable_segments is a known-bad combination.

    The expandable-segments allocator remaps virtual address ranges under the
    hood; bitsandbytes' 4-bit kernels hold raw device pointers into those
    ranges, so the first `dequantize_4bit` in the forward dies with
    `RuntimeError: CUDA driver error: device not ready`. Fail loudly here
    rather than let it look like a mysterious driver/GPU fault.
    """
    if four_bit and "expandable_segments" in (alloc_conf or ""):
        raise SystemExit(
            "[ERR] PYTORCH_CUDA_ALLOC_CONF contains 'expandable_segments' while "
            "loading in 4-bit.\n"
            "      This breaks bitsandbytes -> 'CUDA driver error: device not ready'.\n"
            "      Unset it (env -u PYTORCH_CUDA_ALLOC_CONF ...) or pass --no-4bit.")


def main():
    args = parse_args()
    import torch
    import transformers as tf
    from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                              TrainingArguments, BitsAndBytesConfig)
    from peft import (LoraConfig, get_peft_model, TaskType,
                     prepare_model_for_kbit_training)
    print("[INFO] transformers", tf.__version__, "| torch", torch.__version__)
    if getattr(tf, "set_seed", None):
        tf.set_seed(args.seed)

    train_records = _load_jsonl(args.train)
    val_records = _load_jsonl(args.val)
    print(f"[data] {validate_records(train_records)} train / "
          f"{validate_records(val_records)} val records validated")

    is_local = os.path.isdir(args.base)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, local_files_only=is_local)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    four_bit = not args.no_4bit and torch.cuda.is_available()
    check_alloc_conf(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), four_bit)

    # Liger patches the qwen2 modeling classes, so it must run BEFORE from_pretrained.
    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2(fused_linear_cross_entropy=True,
                                    cross_entropy=False)
        print("[INFO] Liger fused linear+cross-entropy enabled")

    quant = None
    if four_bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    # Force the whole 4-bit model onto GPU 0. device_map="auto" will silently
    # offload trailing layers to CPU when VRAM looks tight (it does at long
    # context), and CPU compute at 8k tokens hangs training for hours. The 4-bit
    # 14B is ~9 GB and fits a 24 GB card whole, so pin it.
    model = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=is_local, quantization_config=quant,
        torch_dtype=compute_dtype, device_map={"": 0} if quant else None)
    model.config.use_cache = False
    # use_reentrant=False: non-reentrant checkpointing is both more memory-efficient
    # (it doesn't retain full layer-boundary activations) and more stable — reentrant
    # GC on the 14B at 12k context overflows 24 GB and aborts at step 0.
    gc_kwargs = {"use_reentrant": False}
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs=gc_kwargs)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        # prepare_model_for_kbit_training() does this for the 4-bit path. On the
        # bf16 path we must do it ourselves: with a frozen base + LoRA, the
        # checkpointed segment's inputs carry no grad_fn, so backward dies with
        # "element 0 of tensors does not require grad". Re-enabling grads on the
        # embedding output reconnects the graph.
        model.enable_input_require_grads()

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
                      lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none", target_modules=QWEN_LORA_TARGETS)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def encode(records):
        out, over, empty = [], 0, 0
        for r in records:
            ex, full_len = tokenize_example(tok, r, max_length=args.max_length)
            if full_len > args.max_length:
                over += 1  # would truncate the review completion -> drop it whole
                continue
            if not has_trainable_tokens(ex["labels"]):
                empty += 1
                continue
            out.append(ex)
        if over:
            print(f"[warn] dropped {over} record(s) longer than --max-length "
                  f"{args.max_length} (would truncate the review); raise --max-length to keep them")
        if empty:
            print(f"[warn] dropped {empty} record(s) with no completion tokens")
        return out

    train_enc, val_enc = encode(train_records), encode(val_records)
    if not train_enc:
        sys.exit("[ERR] no trainable examples after encoding; raise --max-length")
    print(f"[encode] {len(train_enc)} train / {len(val_enc)} val encoded")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        ii, am, lb = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad_id] * pad)
            am.append(b["attention_mask"] + [0] * pad)
            lb.append(b["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am),
                "labels": torch.tensor(lb)}

    base_kw = dict(
        output_dir=args.out, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, logging_steps=5,
        save_strategy="epoch", save_total_limit=2, report_to=[], seed=args.seed,
        bf16=bf16_ok, fp16=not bf16_ok,
        # Gradient checkpointing is enabled ONCE, on the model (via
        # prepare_model_for_kbit_training / gradient_checkpointing_enable below).
        # Enabling it AGAIN here makes the Trainer re-call gradient_checkpointing_enable
        # with different kwargs, clobbering the input-require-grads hook -> a fatal
        # abort at step 0. So leave it False in TrainingArguments.
        gradient_checkpointing=False,
        # LoRA optimizer states are tiny (~1 GB for 137M params), so plain
        # adamw_torch is fine and avoids paged-optimizer CUDA-unified-memory paging
        # that can hang under WSL2.
        optim="adamw_torch",
        warmup_ratio=0.03, lr_scheduler_type="cosine")
    for eval_key in ("eval_strategy", "evaluation_strategy", None):
        try:
            kw = dict(base_kw)
            if eval_key:
                kw[eval_key] = "epoch"
            targs = TrainingArguments(**kw)
            break
        except TypeError:
            continue

    trainer = Trainer(model=model, args=targs, train_dataset=train_enc,
                      eval_dataset=val_enc, data_collator=collate)
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("[OK] Saved surface-SFT LoRA adapter to", args.out)


if __name__ == "__main__":
    main()
