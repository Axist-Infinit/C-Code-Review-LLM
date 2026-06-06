# Setup Guide

Two supported machines, one setup script. The pipeline auto-detects which
machine it's on and configures itself (`profiles.py` / `profiles.json`).

| | RTX 4090 (WSL2, x86_64) | DGX Spark (aarch64 Grace Blackwell) |
|---|---|---|
| torch wheels | cu12x/cu130 (auto-selected) | cu130 aarch64 (auto-selected) |
| classifier | batch 64, fp16 | batch 128, bf16 |
| explainer | `qwen2.5-coder:14b` | `qwen2.5-coder:32b` |

## 1. Clone and set up (both machines)

```bash
git clone https://github.com/Axist-Infinit/C-Code-Review-LLM.git
cd C-Code-Review-LLM
./setup_machine.sh
```

This installs system deps, creates `.venv`, installs the right torch wheel for
the detected GPU/arch, installs pinned python deps, fetches the GraphCodeBERT
base, and pulls the profile's Ollama explainer model (skip the ~9–20 GB pull
with `PULL_OLLAMA=0`).

**Ollama** must be installed separately if missing: <https://ollama.com/download>
(native packages exist for both x86_64 Linux/WSL2 and DGX Spark). Without it,
explanations fall back to the built-in regex heuristics.

## 2. Get a trained classifier

**Option A — train where you are** (~30 min on the 4090, fetches BigVul ~2 GB):

```bash
./one_click_unlock_fetch_train_relock.sh   # DATA=bootstrap for a quick smoke-test model
mv vuln-model-bigvul vuln-model            # if trained with OUT=vuln-model-bigvul
```

**Option B — transfer from the other machine** (weights are arch-independent):

```bash
# on the machine that has the model:
./pack_model.sh vuln-model vuln-model-bigvul.tar.zst
scp vuln-model-bigvul.tar.zst user@other-machine:~/C-Code-Review-LLM/

# on the receiving machine:
./unpack_model.sh vuln-model-bigvul.tar.zst
```

## 3. Scan

```bash
SRC=path/to/code ./scan_repo.sh        # classifier -> LLM explainer -> HTML report
./smoke_test.sh                        # verify the whole pipeline
```

Outputs in `scan_out/`: `classifier_findings.json`, `llm_findings.json`,
`report.html`. Add SARIF for CI: `python to_sarif.py scan_out/llm_findings.json -o scan_out/findings.sarif`.

The scanner uses the model's tuned max-F1 threshold (0.91, precision-oriented).
`scan_repo.sh` overrides it to 0.5 for triage (higher recall — the LLM explainer
filters false positives downstream). Set `THRESHOLD=` to change, or
`THRESHOLD=tuned` to use the model's own.

## 4. Switching machines

Nothing to configure — detection is automatic. To force a profile:

```bash
CCR_PROFILE=spark ./scan_repo.sh       # or: --profile spark on any python tool
```

## Troubleshooting

- **torch sees no GPU**: re-run `./setup_machine.sh` after driver updates; it
  re-resolves the wheel index from `nvidia-smi`'s reported CUDA version.
- **`ImportError: VideoReader` from datasets**: torchvision is installed in the
  venv; remove it (`pip uninstall torchvision torchaudio`) — the pipeline needs
  torch only.
- **Explainer says "falling back to heuristic"**: Ollama isn't running
  (`ollama serve`) or the profile's model isn't pulled (`ollama pull <model>`).
- **Everything scores ~0.5 and every file has findings**: you're running the
  bootstrap smoke-test model; train on BigVul (Option A) or transfer the real
  model (Option B). Verify with `python evaluate_model.py` — ROC-AUC should be
  ≥0.9 on the BigVul test split.
