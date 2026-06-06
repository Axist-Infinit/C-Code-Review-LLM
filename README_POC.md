# C-Code-Review-LLM (v2, refreshed 2026-06)

Local, offline-capable C/C++ vulnerability scanner:
**GraphCodeBERT classifier triage → local Ollama LLM explanations → HTML/JSON/SARIF reports.**

## Quick start

```bash
# 1. Install + fetch BigVul + train classifier + relock offline (run on GPU box)
./one_click_unlock_fetch_train_relock.sh

# 2. Pull the explainer model for your machine (one-time, needs network)
ollama pull qwen2.5-coder:14b    # RTX 4090
ollama pull qwen2.5-coder:32b    # DGX Spark

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
