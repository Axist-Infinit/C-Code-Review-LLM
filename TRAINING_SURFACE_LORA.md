# Surface-Review Distillation — Training Notes & DGX Spark Runbook

Distilling Claude Opus attack-surface reviews into the local `qwen2.5-coder`
model. This is the **surface-review LoRA** lane, not the classifier lane — for
machine setup see **[SETUP_SPARK.md](SETUP_SPARK.md)** / **[SETUP_4090.md](SETUP_4090.md)**.

**Status (2026-07-09): data is DONE and staged. Training is BLOCKED on hardware —
it does not fit a 24 GB RTX 4090. Run it on the DGX Spark.**

---

## 1. What exists right now

| Artifact | Path | State |
|---|---|---|
| Teacher (Opus) reviews — train | `corpus/reviews/train/*.json` | **325** |
| Teacher (Opus) reviews — eval (held out) | `corpus/reviews/eval/*.json` | **80** (all done) |
| SFT train set (chat form) | `data/surface_sft_train.jsonl` | **306** |
| SFT val set | `data/surface_sft_val.jsonl` | **19** |
| Blind eval prompt bundles | `corpus/prompts/eval/` | 80 |
| Trainer | `model/train_surface_sft.py` | fixed, see §4 |
| LoRA merge | `model/merge_lora.py` | ready |
| Scorer | `score_review.py` | ready |
| Eval driver | `scripts/py/eval_surface_model.py` | ready |
| Base Ollama Modelfile (ChatML) | `model/serving/Modelfile.base` | ready |

**Leakage:** the SFT set is built from `train` reviews only. The 80 `eval`
reviews are held out and never enter training. The corpus split is hash-disjoint
(`_split_key % 100 < eval_pct`) over PrimeVul functions.

**Baseline to beat** (DHCP-headers anchor, vs. an Opus gold reference on the
identical prompt, scored by `score_review.py`):

| config | composite /100 |
|---|---|
| `qwen2.5-coder:14b` | **81.1** |
| `qwen2.5-coder:32b` | **86.1** |

Grounding and coverage are already solved at baseline. **The gap is finding
*recall*** — three systemic contracts defeat every local config at temp 0.3:
callback-reentrancy UAF (CWE-416), signed/unsigned size-cast mixing, and
option-overload re-parse. That is what the fine-tune is meant to buy.

---

## 2. Teacher-review token lengths (real Qwen tokenizer, n=306)

This single table drives the whole recipe. `full` = system + context + review.

```
full  tokens: median=10865   p75=14258   p90=19155   max=37869
review-only : median= 6945               p90=11699   max=20580
```

Keep-rate by `--max-length` (the trainer **drops** over-length examples rather
than truncating — a cut review JSON teaches the model to emit invalid output):

| `--max-length` | examples kept |
|---|---|
| 4096 | 0 / 306 (0%) |
| 8192 | 44 / 306 (14%) |
| 12288 | 200 / 306 (65%) |
| 16384 | 250 / 306 (82%) |
| **20480** | **281 / 306 (92%)** |
| 24576 | 295 / 306 (96%) |
| 32768 | 303 / 306 (99%) |

Opus writes long, thorough walkthroughs — **the length *is* the signal.** Training
at 8192 would throw away 86% of the corpus and bias the student toward short,
shallow reviews, destroying the north-star metric. Qwen2.5's native max position
is **32768**, so `--max-length 32768` is the ceiling (3 examples exceed even that).

**Target: `--max-length 20480` (92%), or 32768 (99%) if memory allows.**

---

## 3. Why the RTX 4090 (24 GB) cannot do this — root cause, corrected

Earlier notes in this repo blamed **WSL2 WDDM TDR** (the GPU watchdog). **That was
wrong.** A full Windows reboot with `TdrDelay=60` active changed nothing. Measured
after the reboot:

| test | result |
|---|---|
| Raw sustained CUDA (200× 8192³ bf16 matmul) | ✅ works — TDR never fires |
| `bnb.dequantize_4bit` / `matmul_4bit` in isolation | ✅ works |
| 14B nf4 forward-only inference, L=128 | ✅ 0.4 s, 10.0 GB |
| 14B QLoRA fwd+bwd, LoRA + grad-ckpt, L=128 | ✅ works |
| 14B QLoRA fwd+bwd, L=256 | ✅ 1.3 s, **peak 20.6 GB** |
| 14B QLoRA fwd+bwd+optim.step, **L=512** | ❌ **hangs forever at 100% GPU** |

The two real causes:

### (a) `expandable_segments` breaks bitsandbytes
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (which every prior attempt
set) makes the very first `dequantize_4bit` in the forward die with
`RuntimeError: CUDA driver error: device not ready`. The expandable-segments
allocator remaps virtual address ranges; bnb's 4-bit kernels hold raw device
pointers into them. **Never combine the two.** `train_surface_sft.py` now
refuses this combination at startup (`check_alloc_conf`).

### (b) It simply does not fit — and WSL2 hides the OOM as a hang
The 14B nf4 + LoRA(r=32) + AdamW is **13.6–13.7 GB resident before a single
activation**, leaving ~10 GB. Peak is already **20.6 GB at L=256**. At L=512 it
crosses 24 GB — and on WSL2 that does **not** raise `torch.cuda.OutOfMemoryError`.
WDDM silently spills into shared host RAM over PCIe, so the process pins at
100% GPU util with **zero forward progress, forever**. Every "mysterious hang" in
this project was that.

Confirmed **not** a bitsandbytes version bug: `0.45.5` and `0.49.2` hang
identically at L=512.

> **Tip:** to make WSL2 raise a real OOM instead of hanging, set NVIDIA Control
> Panel → *Manage 3D settings* → **CUDA – Sysmem Fallback Policy** →
> *Prefer No Sysmem Fallback*. Worth doing before ever debugging CUDA on WSL2 again.

**Conclusion:** a 14B QLoRA at the sequence lengths this corpus requires
(median ~10.9k) needs far more than 24 GB. The 4090 tops out near L≈256–512.
Do not try to make it fit by cutting `--max-length` — that guts the dataset (§2).

---

## 4. DGX Spark recipe (the plan)

The Spark's **128 GB unified memory** dissolves the whole problem, and lets us
**drop bitsandbytes entirely** — which also sidesteps the entire bug class above.

### Key decisions

1. **bf16, not 4-bit.** Pass `--no-4bit`. 14B bf16 weights ≈ 29.5 GB — trivial in
   128 GB. bitsandbytes on aarch64 + sm_121 is unreliable; don't depend on it.
2. **Liger fused linear+cross-entropy** (`--liger`). At vocab **152 064**, the
   plain HF CE materialises an `(L × 152064)` fp32 logit tensor: **~12.5 GB at
   L=20480**, on top of a same-sized bf16 copy. The fused kernel chunks it. This
   is the difference between long context being cheap and being impossible.
3. **Gradient checkpointing, `use_reentrant=False`**, enabled exactly once.
4. **`--max-length 20480`** (92% keep) to start; try 32768 (99%) — memory allows.

### Rough memory budget, 14B bf16 LoRA r=32, L=20480, Liger + grad-ckpt

| item | ≈ |
|---|---|
| weights (bf16) | 29.5 GB |
| LoRA grads + AdamW fp32 states (~137 M trainable) | ~2 GB |
| checkpointed activations (48 layers × L × 5120 × 2 B) | ~10 GB |
| fused CE (chunked; replaces a ~12.5 GB fp32 logit tensor) | ~1 GB |
| **total** | **≈ 45 GB of 128 GB** |

Comfortable. At L=32768 it's ≈ 55 GB — still fine.

### Commands (on the Spark)

Per **[SETUP_SPARK.md](SETUP_SPARK.md)**: you need a **cu130** aarch64 torch
(nightly channel or the NGC PyTorch **SBSA** container), otherwise the first CUDA
op throws *"no kernel image is available for execution on the device"*. Verify:

```bash
./env_doctor_cuda.sh     # expect: capability 12.1 (sm_121) ... [doctor] PASS
```

```bash
pip install liger-kernel peft accelerate transformers   # no bitsandbytes needed

# NOTE: no PYTORCH_CUDA_ALLOC_CONF. Do not set expandable_segments.
python3 model/train_surface_sft.py \
    --base Qwen/Qwen2.5-Coder-14B-Instruct \
    --train data/surface_sft_train.jsonl \
    --val   data/surface_sft_val.jsonl \
    --out   ./llm-surface-lora \
    --no-4bit --liger \
    --max-length 20480 \
    --epochs 3 --batch 1 --accum 8 --lr 2e-4 \
    --lora-r 32 --lora-alpha 64
```

Then merge and serve:

```bash
python3 model/merge_lora.py \
    --base Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter ./llm-surface-lora --out ./llm-surface-merged

# Modelfile: copy model/serving/Modelfile.base, swap FROM -> ./llm-surface-merged
ollama create qwen2.5-coder-surface:14b --experimental -q q4_K_M -f Modelfile
```

Score before/after against the held-out refs (the whole point):

```bash
python3 scripts/py/eval_surface_model.py --model qwen2.5-coder:14b          # baseline
python3 scripts/py/eval_surface_model.py --model qwen2.5-coder-surface:14b  # tuned
```

Success = tuned 14B beats **81.1** on the DHCP anchor and lifts mean composite on
the 80 held-out eval refs, driven by **finding recall**, not grounding.

### Worth considering on the Spark: distil into the 32B instead

The `32b` baseline already scores **86.1 vs 81.1** — it starts 5 points closer to
Claude, and the `spark` profile already serves `qwen2.5-coder:32b` at
`num_ctx 16384`. In bf16 it's ~65 GB of weights; at L=16384 the total lands
around ~80–90 GB — plausible in 128 GB but tight. The Spark's unified LPDDR5X is
bandwidth-limited (~273 GB/s, verify), so expect it to be **slow, not impossible**.
Recommendation: **land the 14B run first** (it's the deployable target and the
baseline is measured), then attempt the 32B as an upgrade.

---

## 5. Gotchas that cost real time here

- **Never** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with 4-bit.
  The trainer now hard-refuses it.
- **`device_map="auto"` silently CPU-offloads** trailing layers when VRAM looks
  tight, then "trains" at CPU speed — a 6-hour hang at step 0. Use `{"": 0}`.
- **Enable gradient checkpointing exactly once.** `prepare_model_for_kbit_training`
  *and* `TrainingArguments(gradient_checkpointing=True)` together clobber the
  input-require-grads hook → fatal abort at step 0. TrainingArguments stays `False`.
- **bf16 path needs `model.enable_input_require_grads()`** explicitly (the 4-bit
  path gets it free from `prepare_model_for_kbit_training`). Without it, grad-ckpt
  + frozen base + LoRA → *"element 0 of tensors does not require grad"*. Fixed.
- **`optim="adamw_torch"`**, not `paged_adamw_8bit` (the paged optimizer's
  unified-memory paging can hang under WSL2).
- **Never truncate the review completion** — drop the example instead. A cut JSON
  teaches invalid output. `encode()` does this.
- **Never `pkill -f train_surface_sft`** from a shell whose own command contains
  that string — it kills its own shell. Kill via
  `nvidia-smi --query-compute-apps=pid`.
- On WSL2, **a CUDA "hang" at 100% util is usually an OOM** spilling to sysmem.
  Disable the sysmem fallback policy to get a real exception.

---

## 6. Open items

- [ ] Run the fine-tune on the Spark (§4). Everything upstream is staged.
- [ ] Score tuned vs. baseline on the 80 held-out refs + the DHCP anchor.
- [ ] Optional: generate the last 75 train reviews (325/400; 306 SFT rows is
      already plenty — eval-first ordering means all 80 eval refs are complete).
- [ ] `topic_consolidate` over-merges findings (union path scored 83.9, *below*
      the 86 raw pool). Worth a separate fix; unrelated to training.
