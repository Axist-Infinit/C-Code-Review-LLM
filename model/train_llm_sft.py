#!/usr/bin/env python3
import argparse, os, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType
import transformers as tf

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    return ap.parse_args()

def build_prompt(ex):
    code = ex["code"]
    analysis = ex.get("analysis", "")
    ex["text"] = f"### Code:\n{code}\n\n### Security Analysis:\n{analysis}"
    return ex

def main():
    args = parse_args()
    print("[INFO] transformers:", tf.__version__)
    is_local = os.path.isdir(args.base) or os.path.isfile(os.path.join(args.base, "config.json"))
    tk = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True,
                                       local_files_only=is_local)
    if tk.pad_token is None and tk.eos_token is not None:
        tk.pad_token = tk.eos_token

    dtype = torch.float16 if args.fp16 and torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, local_files_only=is_local,
        torch_dtype=dtype
    )
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
                      lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none")
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files={"train": args.train, "validation": args.val})
    ds = ds.map(build_prompt)
    cols = ds["train"].column_names

    def tok_fn(ex):
        return tk(ex["text"], truncation=True, max_length=2048)
    ds = ds.map(tok_fn, batched=True, remove_columns=cols)

    collator = DataCollatorForLanguageModeling(tk, mlm=False)

    base_args = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        report_to=[],
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=bool(dtype is torch.float16),
    )
    try:
        targs = TrainingArguments(**base_args)
    except TypeError:
        base_args.pop("save_strategy", None)
        base_args.pop("evaluation_strategy", None)
        targs = TrainingArguments(**base_args)

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tk, data_collator=collator
    )
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tk.save_pretrained(args.out)
    print("[OK] Saved SFT model to", args.out)

if __name__ == "__main__":
    main()
