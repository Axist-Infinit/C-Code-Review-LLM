# C-Code-Review-LLM (v2, refreshed 2026-06)

Local, offline-capable C/C++ vulnerability scanner:
**GraphCodeBERT classifier triage → local Ollama LLM explanations → HTML/JSON/SARIF reports.**

## Quick start

Setup hub: **[SETUP.md](SETUP.md)** · per-machine runbooks:
**[SETUP_4090.md](SETUP_4090.md)** (x86_64 / sm_89) ·
**[SETUP_SPARK.md](SETUP_SPARK.md)** (aarch64 Grace-Blackwell / sm_121)

```bash
# 1. Machine setup: venv, torch for this GPU/arch, deps, Ollama explainer model
./setup_machine.sh
./env_doctor_cuda.sh           # on real hardware: assert CUDA + matmul + sm_89/sm_121 kernels

# 2. Train the classifier (or transfer one: see SETUP.md option B)
./preflight_doctor.sh || true  # model/threshold/profile/Ollama preflight
DATA=bigvul ./one_click_unlock_fetch_train_relock.sh   # fetch -> merge(guarded) -> train -> eval -> relock

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
  Function boundaries use **tree-sitter when installed** (`--parser auto`, the
  default) and fall back to a dependency-free brace matcher otherwise. tree-sitter
  correctly captures macro-heavy / K&R / nested definitions (signature + params)
  that the brace heuristic truncates; force with `--parser treesitter|brace`.
  **C++ is fully supported** via the tree-sitter-cpp grammar — class member
  functions, constructors/destructors, operator overloads, out-of-line method
  definitions, function templates (header kept), and namespaces. The grammar is
  chosen per file extension (`--lang auto`), or force `--lang c|cpp`.
  Threshold comes from the model's `inference.json` (tuned at train time);
  override with `--threshold`.
- `llm_explain.py FINDINGS --out OUT.json [--html report.html]` — per-finding
  LLM analysis (issue, CWE, severity, fix, own vulnerability judgment) via local
  Ollama. Findings are explained **in parallel** (`--workers`, default from the
  profile's `max_workers`; pair with `OLLAMA_NUM_PARALLEL` on the server). Auto-
  falls back to the heuristic regex explainer when Ollama is down, and per-finding
  on any single failed call (`--backend heuristic` to force).
- `explain_findings.py` — pure-regex heuristic explainer (no LLM needed). Each
  pattern carries a CWE id, so heuristic findings get real SARIF rule ids. Covers
  classic C footguns (gets/strcpy/strcat/sprintf/scanf/memcpy/memmove/alloca) and
  **C++-specific issues**: `system`/`popen` (CWE-78), `reinterpret_cast`/
  `const_cast` (CWE-704), and non-literal `printf` format strings (CWE-134).
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

`DATA=bigvul` (default) fetches BigVul from HF and builds deduplicated splits
(source splits preserved when present, else 80/10/10). The merge step
**HARD-FAILS** (no silent bootstrap) if the fetch is empty or the train split is
below `MIN_TRAIN_ROWS` (default 20000), so a failed online window can never
quietly train a worthless model. `DATA=bootstrap` writes a tiny disjoint
smoke-test set on purpose — a model trained on it is a plumbing test, **not** a
usable classifier.

**Per-machine training config is wired in:** batch size + precision come from
the detected profile (`profiles.json` via `profiles.py --training-env`):
`4090 → --batch 64 --fp16`, `spark → --batch 128 --bf16`, `cpu → --batch 8`.
Override with `BATCH=` / `CCR_PRECISION_FLAG=` / `CCR_PROFILE=`.

Environment knobs: `USE_GPU=1|0`, `EPOCHS`, `BATCH` (overrides profile),
`MIN_TRAIN_ROWS`, `OUT`, `SKIP_EVAL=1`, `DATA=bigvul|bootstrap`, `HARD_RELOCK=1`
(scoped egress lock after training — see below), `CCR_PROFILE`, `HF_REVISION`
(pin a commit sha for the fetched HF models; empty = upstream HEAD).

### Fine-tuning the explainer (optional, closes the loop)

The explainer model can be specialized on **your** codebase's reviewed findings
so it gets sharper on the bugs you actually see:

```bash
# 1. Run the pipeline, then keep the explanations an analyst agrees with.
# 2. Turn those reviewed findings into SFT data ({code, analysis} JSONL):
python model/build_sft_dataset.py scan_out/llm_findings.json \
    --out-dir data --val-frac 0.1 --only-vulnerable

# 3. LoRA fine-tune the local base model on them:
python model/train_llm_sft.py --base ./llm-base \
    --train data/sft_train.jsonl --val data/sft_val.jsonl --out ./llm-explainer-lora
```

`train_llm_sft.py` uses **completion-only loss masking** (the prompt is masked
out, so the model is trained only on the analysis it should produce, not on
echoing the input), validates the dataset up front (clear error on empty /
malformed records), and picks precision from the hardware profile (`4090 → fp16`,
`spark → bf16`, `cpu → fp32`) unless you pass `--fp16/--bf16`.

## Offline operation

`offline_lockdown.sh` sets HF offline env vars; the one-click relocks the venv
after training. Ollama needs no network after `ollama pull`. The heuristic
explainer is the zero-dependency fallback.

### `HARD_RELOCK` (host firewall — opt-in, reversible)

`HARD_RELOCK=1` no longer flips the host's default `OUTPUT` policy to `DROP`
(that severed the SSH session you ran it over and had no revert). It now:

- requires an explicit `HARD_RELOCK_CONFIRM=yes`,
- installs a **scoped** ruleset in a dedicated `CCR_EGRESS_LOCK` chain that
  `ACCEPT`s loopback and `ESTABLISHED,RELATED` traffic **before** dropping new
  egress, so existing connections (incl. your SSH session) are never cut,
- ships a paired revert: `bash ./online_unlock.sh` removes the chain and unsets
  the offline env vars.

```bash
HARD_RELOCK=1 HARD_RELOCK_CONFIRM=yes ./one_click_unlock_fetch_train_relock.sh
bash ./online_unlock.sh   # revert the egress lock + go back online
```

This is a convenience guard, not a security boundary — enforce a true air-gap at
the OS/hypervisor/network layer.

### Supply-chain pinning (`HF_REVISION`)

The model fetch scripts (`prep_llm_base.sh`, `fetch_graphcodebert.sh`, the
one-click) accept `HF_REVISION=<commit-sha-or-tag>` to pin the exact HF revision
downloaded. Default empty pulls the current upstream HEAD — documented risk: the
upstream repo can change under you, so pin a sha for reproducible/audited builds.
Loading enforces `*.safetensors` (legacy `*.bin` pickle weights are refused).
