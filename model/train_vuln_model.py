#!/usr/bin/env python3
import argparse, os, json, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
import transformers as tf

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--base", default="microsoft/graphcodebert-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    return ap.parse_args()

def _to_int(v):
    try: return int(v)
    except Exception:
        if isinstance(v, torch.Tensor): return int(v.item())
        return int(v)

def build_class_weights(ds_train):
    labels=[]
    for x in ds_train:
        v = x.get("labels", x.get("label"))
        if v is not None: labels.append(_to_int(v))
    if not labels: return None
    n0 = sum(1 for z in labels if z==0); n1 = sum(1 for z in labels if z==1)
    if n0==0 or n1==0: return None
    tot = n0+n1
    return torch.tensor([tot/(2*n0), tot/(2*n1)], dtype=torch.float)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()
    preds = (probs >= 0.5).astype("int64")
    out = {}
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
        out["accuracy"] = float(accuracy_score(labels, preds))
        p,r,f1,_ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        out.update(precision=float(p), recall=float(r), f1=float(f1))
        try: out["roc_auc"] = float(roc_auc_score(labels, probs))
        except: pass
        try: out["pr_auc"] = float(average_precision_score(labels, probs))
        except: pass
    except Exception:
        out["accuracy"] = float((preds==labels).mean())
    return out

def best_threshold_f1(scores, labels):
    best_t, best_f1 = 0.5, -1.0
    for i in range(1,100):
        t=i/100
        pred=[1 if s>=t else 0 for s in scores]
        tp=sum(1 for p,y in zip(pred,labels) if p==1 and y==1)
        fp=sum(1 for p,y in zip(pred,labels) if p==1 and y==0)
        fn=sum(1 for p,y in zip(pred,labels) if p==0 and y==1)
        prec= tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1  = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        if f1>best_f1: best_f1,best_t=f1,t
    return best_t,best_f1

def build_training_args(args):
    base = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=50,
        seed=args.seed,
    )
    eval_extra = dict(save_strategy="epoch", load_best_model_at_end=True,
                      metric_for_best_model="eval_loss", greater_is_better=False)
    # transformers >=4.46 renamed evaluation_strategy -> eval_strategy; try both
    # spellings WITH eval enabled before ever falling back to a no-eval config.
    attempts = [
        dict(fp16=args.fp16, report_to=[], eval_strategy="epoch", **eval_extra),
        dict(fp16=args.fp16, report_to=[], evaluation_strategy="epoch", **eval_extra),
        dict(fp16=args.fp16, report_to=[]),
        dict(fp16=args.fp16),
        dict(),
    ]
    for i, extra in enumerate(attempts):
        try:
            kw = base.copy(); kw.update(extra)
            ta = TrainingArguments(**kw)
            if i >= 2:
                print("[WARN] TrainingArguments accepted only a no-eval config; "
                      "per-epoch evaluation and best-model selection are DISABLED.")
            return ta
        except TypeError:
            continue
    return TrainingArguments(**base)

def main():
    args = parse_args()
    print("[INFO] transformers version in use:", tf.__version__)
    print("[INFO] module:", getattr(tf, "__file__", "unknown"))
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    ds = load_dataset("json", data_files={"train":args.train,"validation":args.val,"test":args.test})
    def tok_fn(ex): return tok(ex["code"], truncation=True, max_length=512)
    ds = ds.map(tok_fn, batched=True).rename_column("label","labels").with_format(
        "torch", columns=["input_ids","attention_mask","labels"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)
    class_weights = build_class_weights(ds["train"]) if args.class_weight else None
    collator = DataCollatorWithPadding(tokenizer=tok)

    train_args = build_training_args(args)

    class WeightedTrainer(Trainer):
        def __init__(self,*a,class_weights=None,**kw):
            super().__init__(*a,**kw); self.class_weights=class_weights
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels=inputs.get("labels")
            outputs=model(**{k:v for k,v in inputs.items() if k!="labels"})
            logits=outputs.get("logits")
            loss_f=torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device)) if self.class_weights is not None else torch.nn.CrossEntropyLoss()
            loss=loss_f(logits.view(-1,2), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(model=model, args=train_args, train_dataset=ds["train"],
                              eval_dataset=ds["validation"], tokenizer=tok,
                              data_collator=collator, compute_metrics=compute_metrics,
                              class_weights=class_weights)
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out); tok.save_pretrained(args.out)

    val_pred = trainer.predict(ds["validation"])
    val_logits = val_pred.predictions[0] if isinstance(val_pred.predictions, tuple) else val_pred.predictions
    val_scores = torch.softmax(torch.tensor(val_logits), dim=1)[:,1].numpy().tolist()
    val_labels = [int(x) for x in ds["validation"]["labels"]]
    t,f1 = best_threshold_f1(val_scores, val_labels)
    with open(os.path.join(args.out, "inference.json"), "w") as f:
        json.dump({"threshold": t, "notes": "Max F1 on validation"}, f, indent=2)
    print(f"[OK] Saved model to {args.out}; threshold={t:.2f} (F1={f1:.3f})")

if __name__ == "__main__":
    main()
