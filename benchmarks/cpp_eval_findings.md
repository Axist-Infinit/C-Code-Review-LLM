# C++ generalization of the BigVul-trained classifier — findings

**TL;DR:** A real BigVul-trained GraphCodeBERT classifies *long, real-world*
functions well (ROC-AUC **0.908** on held-out C/C++) but collapses to ~random on
*short, synthetic* methods — and a control proves this is a **snippet-length /
distribution-shift effect, not a C-vs-C++ language gap**. The ML classifier is
for function-level triage on real codebases; short snippets are the heuristic
scanner's and the static-analysis ensemble's job (which catch all of these).

## Reference model (provenance)

These numbers come from a **small CPU-trained reference model**, trained in-session
to answer the C→C++ transfer question — *not* the production model.

| | |
|---|---|
| Base | `microsoft/graphcodebert-base` |
| Train data | balanced BigVul subset: 2,400 train / 500 val (1:1 vuln:clean, `func_after` as hard negatives) |
| Config | seq_len 256, 2 epochs, batch 8, class-weighted, CPU |
| Tuned threshold | 0.46 (max-F1 on val, F1 0.850) |

The production model (134k functions, 512 seq, 3 epochs, RTX 4090) reports
ROC-AUC 0.959 (`bigvul_test_metrics.json`). Absolute numbers here are lower by
construction; the **relative comparison across test sets on the same model** is
the result.

## Results

| Test set | what it is | n | ROC-AUC | P / R / F1 @0.46 |
|---|---|---|---|---|
| **C baseline** (`c_baseline_metrics.json`) | held-out **real** BigVul functions | 800 | **0.908** | 0.854 / 0.818 / 0.835 |
| **C++ eval** (`cpp_eval_metrics.json`) | **short synthetic** C++ methods | 18 | **0.457** | 0.000 / 0.000 / 0.000 |
| **Control: short C** (`cpp_eval_control_short_c_scores.jsonl`) | **short synthetic** plain-C funcs | 6 | — | all clean (no separation) |

On the C++ set every snippet scored in a tight **0.024–0.105** band (vuln mean
0.045 ≈ clean mean 0.041): the model confidently calls them all "clean" and
cannot rank vuln above clean. The short-C control scored the **same** 0.025–0.031
band with no separation.

## Interpretation

The decisive evidence is the control: **short plain-C snippets fail identically
to the C++ snippets.** So the collapse is driven by *snippet length / synthetic
style* (out-of-distribution for a model trained on long real CVE functions), not
by the language being C++. The C++ eval set as designed is OOD for this model
class. This empirically confirms the project's own note that "textbook toy
snippets score low; the heuristic regex explainer and the static-analysis
ensemble cover those."

For comparison, on the same short snippets the **model-free heuristic scanner**
flags all of them with correct CWEs (CWE-120/134/78/704) — the right tool for
short-snippet / CI gating.

## Recommendations

1. **Measure ML C++ generalization with real functions, not snippets.** Build a
   `cpp_eval_real.jsonl` from full C++ CVE functions (e.g. C++ rows of MegaVul /
   CVEfixes), matched in length/realism to BigVul. Re-run with `--group-by cwe`.
2. **Keep the division of labour.** ML classifier → function-level triage on real
   codebases; `heuristic_scan.py` + `ensemble_scan.py` → short-snippet / CI gating.
3. **Re-run on the production model.** `./eval_cpp.sh` with `./vuln-model` will
   show whether the full model behaves the same (expected: yes, since this is a
   distribution effect, not a capacity one).

## Reproduce

```bash
# 1. balanced BigVul splits (scripts/py/fetch_bigvul_hf.py writes data/ingest/*)
# 2. train the reference model (CPU-friendly):
python model/train_vuln_model.py --train data/bigvul_train.jsonl \
    --val data/bigvul_val.jsonl --test data/bigvul_ctest.jsonl \
    --out vuln-model-smalltrain --epochs 2 --batch 8 --max-length 256 --class_weight
# 3. evaluate both:
python evaluate_model.py --model vuln-model-smalltrain --test data/bigvul_ctest.jsonl
python evaluate_model.py --model vuln-model-smalltrain --test benchmarks/cpp_eval.jsonl \
    --group-by category --save-scores benchmarks/cpp_eval_scores.jsonl
```

## Update 2026-07-01 — real-C++ eval set built (recommendation 1 done)

`benchmarks/cpp_eval_real.jsonl` now exists: **822 real C++ CVE functions
(411 vulnerable / 411 fixed-clean, 361 CVEs, 3.1 MB)** drawn from the paired
config of **PrimeVul** (HF mirror `colin/PrimeVul` pinned @ `4fd7158`, mirror of
the official `DLVulDet/PrimeVul` release). Rows are vuln/fixed pairs from the
same fix commit — the same `func_before`/`func_after` hard-negative convention
the classifier was trained with — restricted to unambiguous C++ files
(`.cpp/.cc/.cxx/.hpp/.hh/.hxx`; `.h` and metadata-less rows excluded).

**Leakage control** (builder default, recorded in
`benchmarks/cpp_eval_real.provenance.json`): every pair whose fix `commit_id`
appears anywhere in `bstee615/bigvul` @ `4d8647a` is dropped (12 rows), and
every remaining function is hash-matched by whitespace-normalized text against
all 171,707 BigVul `func_before`/`func_after` bodies (0 additional hits) — so
the set is disjoint from the BigVul training corpus. Rebuild with
`scripts/py/build_cpp_eval_real.py`; schema/provenance are guarded by
`tests/test_cpp_eval_real_dataset.py`.

`./eval_cpp.sh` now scores this set automatically (→
`benchmarks/cpp_eval_real_metrics.json` / `_scores.jsonl`, per-CWE via
`--group-by category`) alongside the synthetic regression set. **Scoring
awaits the production 0.959-AUC model on the GPU box** — the committed
`vuln-model/` is the bootstrap smoke artifact and its numbers would be
meaningless here. Caveat for reading the results: TensorFlow contributes 296 of
822 rows (~36%), so check the per-project provenance before generalizing.

Dataset attribution (MIT, Copyright (c) 2024 DLVulDet — see `NOTICE`):
Ding, Y., Fu, Y., Ibrahim, O., Sitawarin, C., Chen, X., Alomair, B., Wagner,
D., Ray, B., Chen, Y. *"Vulnerability Detection with Code Language Models: How
Far Are We?"* arXiv:2403.18624 (ICSE 2025).

## Update 2026-07-03 — real-C++ scored, and the story changed

The reference model was retrained from scratch with the current pipeline
(`make_balanced_subset.py` seeded splits → `train_vuln_model.py` with
provenance/`max_length` persisted into `inference.json`; git `699b824`,
seed 42, balanced BigVul 2400/500, seq 256, 2 epochs, class-weighted,
CPU-only) and scored on four sets. A **PrimeVul C control**
(`benchmarks/c_eval_control.jsonl`, 816 rows built by the same builder with
`--lang c`, same leakage filter — which dropped 1,978 C rows for BigVul
commit overlap, vs 12 for C++) separates "C++ transfer gap" from "PrimeVul
is just harder":

| test set | n | ROC-AUC | PR-AUC | F1 @ tuned 0.32 |
| --- | --- | --- | --- | --- |
| BigVul C held-out (balanced)             | 800 | **0.904** | 0.903 | 0.828 |
| PrimeVul **C** control (real functions)  | 816 | **0.492** | 0.498 | 0.540 |
| PrimeVul **C++** (real functions)        | 822 | **0.492** | 0.494 | 0.602 |
| synthetic C++ snippets (regression set)  | 18  | 0.407 | 0.520 | 0.000 |

(Full metrics with embedded model/dataset provenance:
`c_baseline_metrics.json`, `c_eval_control_metrics.json`,
`cpp_eval_real_metrics.json`, `cpp_eval_metrics.json`.)

**Conclusion: this is NOT a language gap — it is a cross-dataset
generalization gap.** The model holds 0.904 on held-out BigVul C but is at
chance on BigVul-disjoint PrimeVul functions *in both languages*
(0.492 = 0.492). The earlier "snippet-length OOD, not a language gap"
conclusion was right about language but missed the larger problem: the
classifier's measured skill is largely BigVul-specific. This independently
reproduces the PrimeVul paper's core finding (Ding et al., ICSE 2025) that
BigVul-benchmark scores do not transfer to cleaner-labeled, cross-project
data.

Caveats before over-reading a chance-level number from a 2,400-row CPU
reference model: the production model sees 134k rows / 66× more data and may
transfer somewhat better — **re-run `./eval_cpp.sh` plus
`evaluate_model.py --test benchmarks/c_eval_control.jsonl` against the
production model on the GPU box before treating 0.492 as its number.**
PrimeVul's post-fix negatives are also deliberately hard (single-commit
differences), so "chance on PrimeVul" understates value on triage-style
workloads with easier negatives.

What this means for the roadmap:

1. **Training-data diversification is now the top ML-lane priority** — add
   PrimeVul-train / DiverseVul rows (both are BigVul-disjoint after the same
   commit/hash filter) to the 4090 training mix, then re-measure all four
   rows of the table above. The gate flags exist
   (`evaluate_model.py --min-roc-auc/--min-pr-auc`) to enforce a
   cross-dataset floor, not just a BigVul floor.
2. **The heuristic lane is unaffected** — it needs no training distribution
   and already catches the playground/synthetic cases the classifier misses;
   keep it as the robust detection floor in CI.
3. The C control is committed and test-guarded
   (`tests/test_cpp_eval_real_dataset.py`), so this comparison is one
   `eval_cpp.sh` + one `evaluate_model.py` invocation to reproduce on any
   trained model.
