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
  LLM analysis via local Ollama. Each finding is reported in a structured
  **3-part review format** so a human can act on it without reading the model's
  raw reasoning:
  - **What the code is doing** (`what_code_does`) — the relevant operation,
  - **What could go wrong** (`what_could_go_wrong`) — the failure mode, how it is
    triggered, and the impact,
  - **Vulnerability** (`vulnerability`) — the identified vulnerability class/name,

  alongside `is_vulnerable`, `issue`, `cwe`, `severity`, and a concrete `fix`.
  These fields render in the HTML report, the SARIF message, and inline PR
  comments. The reviewer prompt explicitly walks a security checklist (bounds,
  integer overflow, use-after-free/double-free, injection, format strings, weak
  crypto/PRNG, TOCTOU) and traces taint to the sink before flagging, to cut false
  positives. If the model omits the structured fields they are backfilled from the
  regex heuristic. Findings are explained **in parallel** (`--workers`, default
  from the profile's `max_workers`; pair with `OLLAMA_NUM_PARALLEL` on the
  server). Auto-falls back to the heuristic regex explainer when Ollama is down,
  and per-finding on any single failed call (`--backend heuristic` to force).
- `heuristic_scan.py SRC -o OUT` — **model-free** scan: extracts functions with
  tree-sitter (C/C++) and flags any whose body matches the unsafe-API/CWE pattern
  set, emitting the same `classifier_findings.json` schema. No model, torch, or
  Ollama — this is the zero-setup path used by the GitHub Action (see below).
- `explain_findings.py` — pure-regex heuristic explainer (no LLM needed). Each
  pattern carries a CWE id, so heuristic findings get real SARIF rule ids. Covers
  classic C footguns (gets/strcpy/strcat/sprintf/scanf/memcpy/memmove/alloca) and
  **C++-specific issues**: `system`/`popen` (CWE-78), `reinterpret_cast`/
  `const_cast` (CWE-704), and non-literal `printf` format strings (CWE-134).
- `evaluate_model.py --model ./vuln-model --test data/test.jsonl` — held-out
  P/R/F1/AUC + threshold sweep. Run this after training; ROC-AUC < 0.6 means
  the model has no signal. Add `--group-by category` (or `lang`) for a per-bucket
  breakdown — e.g. against `benchmarks/cpp_eval.jsonl` to measure how the
  C-trained classifier actually scores C++ per vulnerability class:
  `python evaluate_model.py --model ./vuln-model --test benchmarks/cpp_eval.jsonl --group-by category`.
- `to_sarif.py FINDINGS -o out.sarif` — SARIF 2.1.0 for CI / GitHub code scanning.
- `ensemble_scan.py SRC --ml FINDINGS -o out.json` — merge with cppcheck /
  flawfinder / **clang-tidy** results (each skipped gracefully if not installed);
  marks findings corroborated by multiple tools. clang-tidy is the AST-aware C++
  companion (`bugprone-*`/`cert-*` + the security analyzer); tune with
  `--clang-tidy-checks` or disable with `--no-clang-tidy`.
- `smoke_test.sh` — end-to-end pipeline test.

### Attack-surface / contract review (headers, whole modules)

The classifier and heuristic scanners score function **bodies** — they are blind
to a declaration-only header, where the struct layouts, size macros, and function
**signatures** define the attack surface and the contracts the implementation must
uphold. `surface_review.py` reviews those contracts instead: it asks the local
Ollama model to identify the subsystem, establish the trust boundary, enumerate
every field / macro / signature, and emit a ranked multi-finding threat model
(JSON + Markdown), reviewing all input files **together** so cross-file contracts
are visible.

Every review answers two questions as **thorough, line-numbered walkthroughs** —
**"What is this code doing?"** and **"What could go wrong?"** — each an ordered
sequence of `file:line` → code snippet → detailed explanation. Source is fed to
the model with line-number prefixes so citations are grounded (the verbatim code
snippet is the exact anchor; line numbers reset per file, so every citation is
file-qualified). Every finding likewise carries `file`, `line`, and a verbatim
`code` snippet, and CWEs are corrected conservatively against per-topic
allow-lists (a model CWE already acceptable for the topic is kept; only
mismatches are overridden — an unsigned underflow can't ship as CWE-190
instead of CWE-787, and correct model CWEs survive on unfamiliar code).

```bash
# Single-model, two-pass (enumerate -> completeness critic), offline retrieval hints on
python surface_review.py playground/dhcp-internal.h playground/dhcp-protocol.h \
    --json scan_out/surface/review.json --md scan_out/surface/report.md

# Union finisher: sample several models/temperatures, pool, deterministically consolidate
python surface_review.py SRC --models qwen2.5-coder:14b,qwen2.5-coder:32b --samples 2
```

- **Offline retrieval hints** (`surface_kb.json`) inject trigger-matched facts
  (known CVEs, RFC/protocol semantics, correct CWEs) into the prompt — the single
  biggest quality lever for the knowledge-bound findings. `--no-kb` to disable.
- **Union finisher** (`--models a,b --samples N`) pools findings across samples
  (different models catch different issues) and consolidates them. Consolidation
  is **deterministic by default** (`topic_consolidate`: merge by security-topic,
  canonicalise CWE, restore known CVE) — reliable and VRAM-free; `--llm-consolidate`
  opts into an LLM consolidation pass when a critic model fits in memory.
- Reviews headers the body-scanners skip; pooling reached ~9/10 of a Claude
  reference threat model on the systemd DHCP headers, fully offline.

### Compare the local model against Claude Opus (`compare_review.py`)

To see how the local model measures up on **your** code, `compare_review.py` runs
the *identical* review prompt (same system prompt, same KB hints, same fenced
snippet) against both the local Ollama model and Claude Opus 4.8, and prints them
finding-for-finding. The only variable is the model.

```bash
python compare_review.py path/to/snippet.c        # a file or directory
python compare_review.py --code 'void f(char*s){char b[8];strcpy(b,s);}'
echo '<code>' | python compare_review.py -          # snippet from stdin
```

The Claude side is the **opt-in cloud tier** — the only part of this project that
leaves the machine. It needs the official SDK and a key: `pip install anthropic`
(optional, not in `requirements.txt`) and `export ANTHROPIC_API_KEY=...`. Use
`--no-claude` / `--no-local` to run a single side; both reports land in
`scan_out/compare/`.

## CI integration (GitHub Action)

A composite action (`action.yml`) runs the **model-free** path —
`heuristic_scan.py → llm_explain.py --backend heuristic → to_sarif.py` — and
emits SARIF for the repository's **Code scanning** tab. No model, GPU, or network
is required, so it runs on any stock GitHub runner with zero setup.

```yaml
# .github/workflows/code-scan.yml  (template in ci/code-scan.yml)
permissions:
  contents: read
  security-events: write
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with: { python-version: "3.12" }
  - id: scan
    uses: axist-infinit/c-code-review-llm@main
    with:
      source: "."
      only-vulnerable: "true"
  - uses: github/codeql-action/upload-sarif@v3
    with:
      sarif_file: ${{ steps.scan.outputs.sarif-file }}
```

Inputs: `source` (path), `output` (SARIF path), `lang` (`auto|c|cpp`),
`only-vulnerable`, plus `annotate` / `github-token` (below). For full
ML-classifier results in CI, run `local_vuln_scanner.py` with a provisioned
`vuln-model` instead of `heuristic_scan.py` (heavier: needs torch + the model
artifact).

### Inline PR review comments

Set `annotate: "true"` (and pass `github-token: ${{ github.token }}` with
`pull-requests: write`) to also post findings as **inline review comments** on
pull requests — pinned to the exact line, with severity, CWE, and fix. Findings
on lines outside the PR diff are rolled into the review summary. On non-PR (push)
builds the step skips quietly. This is `annotate_pr.py`, which also runs
standalone:

```bash
python annotate_pr.py scan_out/llm_findings.json --repo OWNER/REPO --pr 123
python annotate_pr.py scan_out/llm_findings.json --dry-run   # preview, no API call
```

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
regex explainer and the cppcheck/flawfinder/clang-tidy ensemble cover those).

### C++ held-out evaluation

The classifier is trained on BigVul (overwhelmingly C). `benchmarks/cpp_eval.jsonl`
is a curated, balanced set of 18 C++ methods (9 vulnerable / 9 clean across 9
CWE categories) to measure how it actually generalizes to C++. Produce a
per-category report with `./eval_cpp.sh` (writes `benchmarks/cpp_eval_metrics.json`),
or directly:

```bash
python evaluate_model.py --model ./vuln-model \
    --test benchmarks/cpp_eval.jsonl --group-by category \
    --out benchmarks/cpp_eval_metrics.json
```

Every case is a single function that the cpp grammar extracts as one unit, so the
scores reflect the same whole-function granularity the scanner uses. Weak
per-category numbers are the signal to add C++ training data (see below).

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

**Corroboration-driven curation (highest-precision positives).** Run the
ensemble, then keep only findings that the ML classifier *and* the independent
static tools agree on — these make the cleanest training signal:

```bash
python ensemble_scan.py SRC --ml scan_out/llm_findings.json -o scan_out/ensemble.json
python model/build_sft_dataset.py scan_out/llm_findings.json \
    --ensemble scan_out/ensemble.json --min-corroboration 1 --only-vulnerable \
    --out-dir data
```

A finding is corroborated when a cppcheck/flawfinder/clang-tidy diagnostic lands
inside its line range; each kept record is tagged with `corroboration` (count)
and `corroborated_by` (tools) for auditability.

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
- ships a paired revert: `source ./online_unlock.sh` removes the chain, unsets
  the offline env vars in the CURRENT shell, and retires `.env.locked` so the
  venv stops re-locking on every activation (with `bash` instead of `source`,
  the env-var unsets die with the child shell).

```bash
HARD_RELOCK=1 HARD_RELOCK_CONFIRM=yes ./one_click_unlock_fetch_train_relock.sh
source ./online_unlock.sh   # revert the egress lock + go back online
```

This is a convenience guard, not a security boundary — enforce a true air-gap at
the OS/hypervisor/network layer.

### Supply-chain pinning (`HF_REVISION`)

The model fetch scripts (`prep_llm_base.sh`, `fetch_graphcodebert.sh`, the
one-click) accept `HF_REVISION=<commit-sha-or-tag>` to pin the exact HF revision
downloaded. Default empty pulls the current upstream HEAD — documented risk: the
upstream repo can change under you, so pin a sha for reproducible/audited builds.
Loading enforces `*.safetensors` (legacy `*.bin` pickle weights are refused).
