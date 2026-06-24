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
