# C-Code-Review-LLM (v2, refreshed 2026-06)

Local, offline-capable C/C++ vulnerability scanner:
**GraphCodeBERT classifier triage → local Ollama LLM explanations → HTML/JSON/SARIF reports.**

## Quick start

Full per-machine instructions (4090 / DGX Spark): **[SETUP.md](SETUP.md)**

```bash
# 1. Machine setup: venv, torch for this GPU/arch, deps, Ollama explainer model
./setup_machine.sh

# 2. Train the classifier (or transfer one: see SETUP.md option B)
./one_click_unlock_fetch_train_relock.sh

# 3. Scan
SRC=path/to/code ./scan_repo.sh
```

Outputs land in `scan_out/`: `classifier_findings.json`, `llm_findings.json`, `report.html`.

## Hardware profiles

The pipeline auto-detects which machine it is on (`profiles.py`) and sizes
batches / picks the Ollama model accordingly. Profiles live in `profiles.json`:

| profile | detected via | classifier | explainer |
|---|---|---|---|
| `4090` | GPU name contains "4090" (or any discrete CUDA GPU) | batch 64, fp16 | `qwen2.5-coder:14b` |
| `spark` | aarch64 / GB10 | batch 128, bf16 | `qwen2.5-coder:32b` |
| `cpu` | no CUDA | batch 8, fp32 | `qwen2.5-coder:7b` |

Override anytime: `CCR_PROFILE=spark ./scan_repo.sh` or `--profile spark`.
The trained classifier moves between machines via `pack_model.sh` / `unpack_model.sh`.

## Pipeline tools

- `local_vuln_scanner.py SRC -o OUT` — classifier scan. Scores whole functions
  (matches BigVul training granularity); `--granularity window` for 12-line windows.
  Threshold comes from the model's `inference.json` (tuned at train time);
  override with `--threshold`.
- `llm_explain.py FINDINGS --out OUT.json [--html report.html]` — per-finding
  LLM analysis (issue, CWE, severity, fix, own vulnerability judgment) via local
  Ollama. Auto-falls back to the heuristic regex explainer when Ollama is down
  (`--backend heuristic` to force).
- `explain_findings.py` — pure-regex heuristic explainer (no LLM needed).
- `evaluate_model.py --model ./vuln-model --test data/test.jsonl` — held-out
  P/R/F1/AUC + threshold sweep. Run this after training; ROC-AUC < 0.6 means
  the model has no signal.
- `to_sarif.py FINDINGS -o out.sarif` — SARIF 2.1.0 for CI / GitHub code scanning.
- `ensemble_scan.py SRC --ml FINDINGS -o out.json` — merge with cppcheck /
  flawfinder results (skipped gracefully if not installed); marks findings
  corroborated by multiple tools.
- `smoke_test.sh` — end-to-end pipeline test.

## Benchmark (BigVul held-out test, 32,710 functions, 3.1% vulnerable)

GraphCodeBERT fine-tuned 3 epochs on 134k BigVul functions (RTX 4090, fp16,
class-weighted). Full metrics: `benchmarks/bigvul_test_metrics.json`.

| | |
|---|---|
| ROC-AUC | **0.959** |
| PR-AUC | 0.505 |
| P / R / F1 @ tuned 0.91 | 0.50 / 0.70 / 0.585 |
| P / R / F1 @ 0.5 | 0.41 / 0.80 / 0.540 |

The tuned (max-F1) threshold is precision-oriented. For triage scanning,
`--threshold 0.5` trades precision for recall — sensible when the LLM
explainer filters findings downstream. Note: BigVul-trained models key on
real-world CVE-style functions; textbook toy snippets score low (the heuristic
regex explainer and the cppcheck/flawfinder ensemble cover those).

## Training

`DATA=bigvul` (default) fetches BigVul from HF and builds deduplicated
80/10/10 splits. `DATA=bootstrap` writes a tiny disjoint smoke-test set —
a model trained on it is a plumbing test, **not** a usable classifier.

Environment knobs: `USE_GPU=1|0`, `EPOCHS`, `BATCH`, `OUT`, `HARD_RELOCK=1`
(iptables egress drop after training), `CCR_PROFILE`.

## Offline operation

`offline_lockdown.sh` sets HF offline env vars; the one-click relocks the venv
after training. Ollama needs no network after `ollama pull`. The heuristic
explainer is the zero-dependency fallback.
