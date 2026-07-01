# Setup Guide

Two supported machines, one setup script. The pipeline auto-detects which
machine it's on (`profiles.py` / `profiles.json`) and wires the per-machine
training config (batch size + precision) into the trainer.

**Per-machine runbooks (accurate torch sources + flags):**
- RTX 4090 (x86_64 / WSL2) → **[SETUP_4090.md](SETUP_4090.md)**
- NVIDIA DGX Spark (aarch64 Grace-Blackwell) → **[SETUP_SPARK.md](SETUP_SPARK.md)**

| | RTX 4090 (WSL2, x86_64, sm_89) | DGX Spark (aarch64 Grace-Blackwell, sm_121) |
|---|---|---|
| torch source | **stable** wheel from the cu12x/cu13x channel matching the driver | **cu130 NIGHTLY** aarch64 wheel **or** NGC PyTorch SBSA container |
| classifier (wired into training) | batch **64**, **fp16** (`--batch 64 --fp16`) | batch **128**, **bf16** (`--batch 128 --bf16`) |
| explainer | `qwen2.5-coder:14b` | `qwen2.5-coder:32b` |

> The torch source is **not** "auto-selected and assumed to exist". On aarch64
> there is no guaranteed stable cu130 wheel, so the installer uses the nightly
> channel (or you use the NGC SBSA container) and **fails soft** instead of
> aborting. Each runbook lists the exact channel and tells you to record the
> verified `torch.__version__` + source you actually installed.

## 1. Clone and set up (both machines)

```bash
git clone https://github.com/Axist-Infinit/C-Code-Review-LLM.git
cd C-Code-Review-LLM
./setup_machine.sh
```

Installs system deps, creates `.venv`, installs the right torch wheel for the
detected GPU/arch (**torch only** — torchvision ≥0.25 breaks `datasets`),
installs pinned python deps, fetches the GraphCodeBERT base, and pulls the
profile's Ollama explainer model (skip the large pull with `PULL_OLLAMA=0`).

If the torch wheel can't be installed for your arch/CUDA, setup **continues**
(it no longer aborts) and points you to the runbook. Resolve torch, then assert
it on the real GPU:

```bash
./env_doctor_cuda.sh    # asserts cuda available + tiny CUDA matmul finite + sm_89/sm_121 kernels
```

`env_doctor_cuda.sh` requires a real GPU and exits non-zero on a CPU box — that
is intentional; it is a post-install hardware assertion, not a dev check.

## 2. Get a trained classifier

**Option A — train where you are** (real BigVul; the per-machine profile picks
the batch + precision automatically):

```bash
./preflight_doctor.sh || true                  # model/threshold/profile/Ollama preflight
DATA=bigvul ./one_click_unlock_fetch_train_relock.sh
```

This runs **fetch → merge (guarded) → train → eval → relock** in one command.
It HARD-FAILS (no silent bootstrap) if BigVul doesn't yield a real-sized train
split. See *Air-gapped provisioning* below.

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
./preflight_doctor.sh                           # verify model + threshold + Ollama first
SRC=path/to/code ./scan_repo.sh                 # classifier -> LLM explainer -> HTML report
./smoke_test.sh                                 # verify the whole pipeline
```

Outputs in `scan_out/`: `classifier_findings.json`, `llm_findings.json`,
`report.html`. SARIF for CI:
`python to_sarif.py scan_out/llm_findings.json -o scan_out/findings.sarif`.

The scanner uses the model's tuned max-F1 threshold; `scan_repo.sh` overrides it
to 0.5 for triage (the LLM explainer filters false positives downstream). Set
`THRESHOLD=` to change, or `THRESHOLD=tuned` to use the model's own.

## 4. Per-machine training config (how the wiring works)

`profiles.json` declares each machine's `classifier_batch_size` and
`classifier_dtype`. `profiles.py` maps that to trainer flags
(`profiles.py --training-env` emits `CCR_BATCH` / `CCR_PRECISION_FLAG`), and
`train.sh` / `train_classifier.sh` / the one-click read it:

| profile | batch | dtype | trainer flags |
|---|---|---|---|
| `4090` | 64 | float16 | `--batch 64 --fp16` |
| `spark` | 128 | bfloat16 | `--batch 128 --bf16` |
| `cpu` | 8 | float32 | `--batch 8` (no mixed precision) |

Override per run: `BATCH=32 ./train.sh`, `CCR_PRECISION_FLAG=--bf16 ./train.sh`,
or force a profile with `CCR_PROFILE=spark`. (Before this wiring the trainer
silently defaulted to fp32/batch16 regardless of the machine.)

## 5. Air-gapped provisioning ("provision online, then lock")

The pipeline runs offline, but the **first** build needs a short online window
to pull artifacts. Do this, then relock.

### Online window — pull everything (per machine, or once + transfer)

1. **torch** for this arch — see the per-machine runbook (4090: stable cu12x/13x
   wheel; Spark: nightly cu130 aarch64 **or** NGC SBSA container).
2. **Python deps**: `pip install -r requirements.txt` (pinned set; verified 2026-06).
3. **GraphCodeBERT base**: `./fetch_graphcodebert.sh` (pin a commit with
   `HF_REVISION=<sha>` for reproducible/audited builds).
4. **BigVul dataset**: `python scripts/py/fetch_bigvul_hf.py` → `data/ingest/`.
   This is the step that must succeed; the merge guard refuses to proceed
   without a real-sized train split.
5. **Ollama explainer model**: `ollama pull <profile model>`
   (4090 → `qwen2.5-coder:14b`, Spark → `qwen2.5-coder:32b`).

One command does steps 2–4 + train + eval:
`DATA=bigvul ./one_click_unlock_fetch_train_relock.sh`.

### Lock

```bash
source ./offline_lockdown.sh        # sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE
# the one-click also appends the offline env to .venv/bin/activate (.env.locked)
```

Optional, reversible host-egress lock (convenience, **not** a security
boundary — enforce a real air-gap at the OS/hypervisor layer):

```bash
HARD_RELOCK=1 HARD_RELOCK_CONFIRM=yes ./one_click_unlock_fetch_train_relock.sh
source ./online_unlock.sh           # revert the lock + go back online (source, not bash)
```

### The hard-fail guard (no silent bootstrap)

On `DATA=bigvul` (default), if the BigVul fetch fails / is empty / yields fewer
than `MIN_TRAIN_ROWS` (default 20000) train rows, `merge_and_split.py` exits
non-zero and training never starts. To run a deliberate smoke-test on the tiny
disjoint set, opt in explicitly: `DATA=bootstrap ./one_click_…` (or
`--allow-bootstrap` / `CCR_ALLOW_BOOTSTRAP=1`). A bootstrap-trained model is a
plumbing test, **not** a usable classifier.

## 6. Preflight doctor

Run before any long train/scan to catch the usual failures cheaply:

```bash
./preflight_doctor.sh                 # model dir + tuned threshold + profile + Ollama
./preflight_doctor.sh --require-ollama # also fail (not warn) if Ollama is down
MODEL=./vuln-model-bigvul ./preflight_doctor.sh
```

Exit 0 = ready (warnings allowed); non-zero = a required check failed.

## 7. Switching machines

Nothing to configure — detection is automatic. Force a profile:

```bash
CCR_PROFILE=spark ./scan_repo.sh      # or: --profile spark on any python tool
```

## Troubleshooting (general; see the per-machine runbooks for arch specifics)

- **torch sees no GPU** — re-run `./env_doctor_cuda.sh`; reinstall torch from the
  channel matching `nvidia-smi` (4090) or the cu130 nightly (Spark).
- **`no kernel image … for execution on the device`** — the wheel lacks kernels
  for your compute capability. 4090 needs sm_89 (any cu12.x/13.x wheel); Spark
  needs sm_121 (cu130 only). Reinstall via `./install_torch_nightly_for_new_gpu.sh`.
- **`ImportError: VideoReader` from datasets** — torchvision is installed; remove
  it (`pip uninstall torchvision torchaudio`). The pipeline needs torch only.
- **Explainer says "falling back to heuristic"** — Ollama isn't running
  (`ollama serve`) or the profile's model isn't pulled. `./preflight_doctor.sh`
  reports this.
- **Training silently used a 12-row set** — fixed: the real path now hard-fails.
  If you see this on an old checkout, update and re-run `DATA=bigvul …`.
- **Everything scores ~0.5 / every file flagged** — bootstrap model; train on
  BigVul. Verify with `python evaluate_model.py` (ROC-AUC ≥0.9 on BigVul test).
