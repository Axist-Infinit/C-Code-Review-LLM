# Training Resources to Research — Surface-Review Distillation

A research backlog for making the local surface-review model better. Curated for
**this** project's specifics: distilling Claude-Opus C/C++ attack-surface reviews
into `qwen2.5-coder`, where the measured gap is **finding *recall* on systemic
contracts** (callback-reentrancy UAF / CWE-416, signed↔unsigned size-cast mixing,
option-overload re-parse), *not* grounding or coverage.

> Priority tiers below. URLs are best-effort — verify the exact repo/paper before
> relying on any one. When unsure, search the **name** given.

Related in-repo docs: **[TRAINING_SURFACE_LORA.md](TRAINING_SURFACE_LORA.md)**
(the runbook), **[SETUP_SPARK.md](SETUP_SPARK.md)** (machine setup).

---

## TIER 0 — Do these first (highest leverage, lowest effort)

### Training frameworks purpose-built for a single big-memory box
Our hand-rolled `train_surface_sft.py` works, but these are battle-tested for
Qwen2.5 QLoRA/bf16 + long context and would remove whole classes of the bugs this
project hit (grad-ckpt hooks, fused CE, sequence packing):

- **Unsloth** — github.com/unslothai/unsloth — fastest single-GPU LoRA/QLoRA,
  bundles its own tested kernels + fused CE + long-context. First thing to try on
  the Spark; sidesteps the bitsandbytes/kernel fragility entirely.
- **Axolotl** — github.com/axolotl-ai-cloud/axolotl — YAML-config SFT, first-class
  Qwen2.5, sample packing, Liger integration, chat-template masking. Great for
  reproducible multi-run sweeps.
- **LLaMA-Factory** — github.com/hiyouga/LLaMA-Factory — broad model support, GUI,
  easy LoRA→merge→export→GGUF flow that matches our Ollama serving path.
- **TRL `SFTTrainer`** — github.com/huggingface/trl — if staying close to HF; gives
  packing + completion-only collators we currently implement by hand.
- **Liger Kernel** — github.com/linkedin/Liger-Kernel — already wired via our
  `--liger` flag; read their Qwen2 notes for the fused-CE memory math.

### Sequence packing
At median ~10.9k tokens and batch 1, we waste compute on padding. **Sample/sequence
packing** (multipack) concatenates examples to a fixed block with a block-diagonal
attention mask — big throughput win on the Spark. All three frameworks above do it.
Research: "multipack sampler", FlashAttention varlen / `flash_attn_varlen_func`.

---

## TIER 1 — Expand the teacher corpus (directly attacks the recall gap)

We have 325 train + 80 eval Opus reviews over blind PrimeVul functions. More and
**harder** examples in the three failing CWE classes is the most direct lever.

### Real-world C/C++ vulnerability datasets (mine for hard cases, then have Opus review)
- **PrimeVul** — (current source) paper: *"Vulnerability Detection with Code
  Language Models: How Far Are We?"* — pairs, de-duped, realistic labels.
- **DiverseVul** — github.com/wagner-group/diversevul — 18k+ vuln functions, 150
  CWEs, curated from security fixes. Best breadth for CWE coverage.
- **BigVul** — github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset — classic
  C/C++ CVE function dataset (already referenced by the classifier lane).
- **CVEfixes** — github.com/secureIT-project/CVEfixes — CVE→fix commit pairs with
  full before/after + CWE + metadata; excellent for *contract-diff* training
  (what changed to fix it) which targets systemic-contract reasoning.
- **Devign** — the FFmpeg/QEMU function dataset (paper: *Devign*, NeurIPS'19);
  hand-labeled, good hard negatives.
- **ReVeal** — from *"Deep Learning based Vulnerability Detection: Are We There
  Yet?"* — Chromium/Debian; realistic class imbalance.
- **D2A** — github.com/ibm/D2A — differential analysis over commit pairs (Infer
  static-analysis labels); noisy but large.
- **Draper VDISC** — ~1.2M functions, static-analysis labels; volume for weak
  supervision / pretraining-style signal.
- **CrossVul** — cross-project CVE dataset spanning many languages/CWEs.
- **Juliet Test Suite / SARD** — samate.nist.gov — *synthetic* but exhaustively
  labeled per-CWE with good/bad variants. Perfect for **targeted CWE-416 / signed-
  unsigned / integer-overflow** coverage where real data is thin. Use to
  deliberately fill the three failing classes.

> Workflow to add: pull functions from these → run the blind-corpus builder
> (`scripts/py/build_distill_corpus.py`, hash-disjoint split) → fan out Opus
> reviewers on the exact SYSTEM_PROMPT → land in `corpus/reviews/train/`. Keep the
> eval set frozen to preserve the 81.1/86.1 baselines' comparability.

### Curriculum idea
Stratify new teacher data toward the hard CWEs (416/362/190/191/787/125) and toward
*multi-function* trust-boundary cases (callback registration + later invocation),
since single-function context is exactly what hides reentrancy/UAF contracts.

---

## TIER 2 — Technique: better distillation & grounding

- **Knowledge distillation for reasoning traces** — have Opus emit not just the
  final review JSON but the *reasoning* that found each systemic contract; train on
  the trace → answer. Search: "distilling step-by-step" (Hsieh et al.),
  "reasoning distillation", "rationale-augmented SFT".
- **Rejection sampling / best-of-n teacher filtering** — generate multiple Opus
  reviews per file, keep only those that pass `score_review.py` grounding + a
  self-consistency check. Raises teacher-label quality. Search: "rejection
  sampling fine-tuning (RFT)", "STaR", "RAFT".
- **Line-number / citation faithfulness** — our north star needs verbatim snippets
  at correct lines. Research constrained decoding / copy mechanisms and
  citation-grounded generation; also a post-hoc verifier that rejects any finding
  whose quoted code ≠ the cited line (we already have the check in `score_review.py`
  — turn it into a training-time reward or a rejection filter).
- **Preference tuning after SFT (DPO/ORPO/KTO)** — once SFT lands, build preference
  pairs (Opus review ≻ local review, or higher-recall ≻ lower-recall) and DPO to
  push recall further. Search: "DPO", "ORPO", "KTO"; TRL implements all three.
- **LLM-as-judge for eval** — beyond `score_review.py`, an Opus judge scoring
  recall of systemic contracts. Search: "LLM-as-a-judge", "MT-Bench methodology".

---

## TIER 3 — Security knowledge to inject (KB / RAG / system prompt)

The model reviews better when the contract vocabulary is in front of it. We already
have `surface_kb.json`; expand it from authoritative sources:

- **CWE** — cwe.mitre.org — especially the "Research Concepts" view and the chains/
  relationships for 416, 362, 190/191, 787/125. Feed definitions + examples into KB.
- **CERT C / C++ Secure Coding Standard** — wiki.sei.cmu.edu/confluence/display/c —
  rule-level contracts (INT30-C signed overflow, INT31-C truncation, EXP34-C null
  deref, etc.) map almost 1:1 to the failing classes. Highest-signal KB source.
- **MITRE CWE Top 25 / "on the cusp"** — prioritize KB coverage by prevalence.
- **OWASP** (for the web/protocol-facing surfaces), **MISRA C** (safety contracts).
- **Linux kernel `Documentation/` + coding style** — real callback/refcount/RCU
  contracts (kref, RCU, workqueue reentrancy) — the source of the CWE-416 pattern
  we keep missing.

---

## TIER 4 — Evaluation & benchmarks

- **SecurityEval** — github.com/s2e-lab/SecurityEval — CWE-labeled prompts for
  security of generated code; adaptable to review eval.
- **CyberSecEval** (Meta/PurpleLlama) — insecure-code + detection benchmarks.
- **CVE-Bench / vuln-detection leaderboards** — for external comparability.
- Keep our **held-out 80 eval refs + DHCP anchor** as the primary internal metric;
  add per-CWE recall breakdown so improvements on 416/362/190 are visible, not
  averaged away.

---

## TIER 5 — Base-model alternatives to distill into

Current: `qwen2.5-coder:14b` (baseline 81.1) and `:32b` (86.1). Worth benchmarking
as students on the Spark's 128 GB:

- **Qwen2.5-Coder-32B-Instruct** — starts 5 pts closer to Claude; top priority
  upgrade (Spark can train it in bf16).
- **DeepSeek-Coder-V2 (Lite/236B-MoE)** — strong C/C++; the Lite fits, the MoE is
  Spark-scale.
- **StarCoder2-15B**, **CodeLlama-34B** — older but well-understood LoRA targets.
- **Reasoning-tuned coders** (QwQ / DeepSeek-R1-distill-Qwen) — if reasoning traces
  help recall (Tier 2), a reasoning base may distill the systemic-contract logic
  better. Benchmark before committing.

---

## Concrete next-experiment shortlist (if you want a plan, not a library)

1. Train the staged 306-example SFT on the Spark with **Unsloth**, bf16,
   `max_length 20480`, 3 epochs. Score vs. 81.1. *(This is the pending run.)*
2. Add **Juliet CWE-416 / integer-overflow** subsets → Opus-review them → retrain;
   measure per-CWE recall delta on the eval set.
3. Add **CVEfixes** before/after pairs as contract-diff examples → retrain.
4. If SFT plateaus, build preference pairs and **DPO** for recall.
5. Fold **CERT C** rules into `surface_kb.json`; re-measure.
