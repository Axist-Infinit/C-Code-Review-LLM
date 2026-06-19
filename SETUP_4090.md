# Setup & Runbook — RTX 4090 (x86_64 / WSL2)

Authoritative per-machine guide for the **RTX 4090 workstation**
(x86_64, Ada Lovelace, compute capability **sm_89**, 24 GB VRAM, WSL2 or native
Linux). For the DGX Spark see **[SETUP_SPARK.md](SETUP_SPARK.md)**; for the
overview and the machine-agnostic flags see **[SETUP.md](SETUP.md)**.

| | RTX 4090 |
|---|---|
| arch / capability | x86_64 / sm_89 (Ada) |
| profile (`profiles.json`) | `4090` — batch **64**, **fp16** |
| training flags wired | `--batch 64 --fp16` (from the profile, via `profiles.py --training-env`) |
| torch source | **stable** wheel from the CUDA channel matching your driver (see below) |
| explainer | `qwen2.5-coder:14b`, num_ctx 8192 |

## torch on the 4090 (verified source per arch)

`setup_machine.sh` resolves the wheel channel from the CUDA version
`nvidia-smi` reports (logic lives in `lib_torch_install.sh`):

| driver CUDA (nvidia-smi) | torch wheel channel (`--index-url`) |
|---|---|
| 13.x | `https://download.pytorch.org/whl/cu130` |
| 12.8 / 12.9 | `…/whl/cu128` |
| 12.5–12.7 | `…/whl/cu126` |
| 12.4 | `…/whl/cu124` |
| 12.1–12.3 | `…/whl/cu121` |
| 11.x | `…/whl/cu118` |

**Verified working set (2026-06):** `torch 2.6.x` from the **stable** `cu126`
channel on CUDA 12.6 (Ada sm_89 kernels are in every cu12.x stable wheel since
torch 2.1). Record the exact version you install:

```bash
python -c "import torch,platform;print(platform.machine(), torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

If your driver reports a CUDA newer than any stable channel ships (e.g. a fresh
13.x driver before the stable cu130 wheel lands), use the nightly:

```bash
./install_torch_nightly_for_new_gpu.sh    # arch-aware: x86_64 -> nightly cu13x/cu12x, torch only
```

> torch **only** — torchvision/torchaudio are unused and torchvision ≥0.25
> breaks the `datasets` torch formatter. The install scripts never install them.

## 1. One-time setup

```bash
git clone https://github.com/Axist-Infinit/C-Code-Review-LLM.git
cd C-Code-Review-LLM
./setup_machine.sh           # system deps -> venv -> torch (right channel) -> deps
                             # -> GraphCodeBERT base -> Ollama explainer pull
```

`setup_machine.sh` no longer aborts the whole run if the torch wheel install
fails: it warns, continues installing the rest, and points you here. Resolve
torch (table above) and then run the on-hardware assertion:

```bash
./env_doctor_cuda.sh         # asserts: cuda available + tiny matmul finite + sm_89 kernels
```

Expected: `[doctor] device 0: NVIDIA GeForce RTX 4090  capability 8.9 (sm_89)`
then `[doctor] PASS`. A `[WARN] … wheel arch list … does NOT include sm_89`
means the wheel lacks native Ada kernels — reinstall from the correct channel.

## 2. Get a trained classifier

**Train here** (real BigVul; ~30 min on the 4090 at fp16/batch 64):

```bash
./preflight_doctor.sh || true                 # checks profile/Ollama before the long run
DATA=bigvul ./one_click_unlock_fetch_train_relock.sh
```

The one-click HARD-FAILS if the BigVul fetch yields fewer than
`MIN_TRAIN_ROWS` (default 20000) train rows — it will **not** silently train on
the 12-row bootstrap set. See **[SETUP.md](SETUP.md) → Air-gapped provisioning**.

**Or transfer** a model trained elsewhere (weights are arch-independent):

```bash
./unpack_model.sh vuln-model-bigvul.tar.zst   # -> ./vuln-model
```

## 3. Scan

```bash
./preflight_doctor.sh                          # model + threshold + Ollama ready?
SRC=path/to/code ./scan_repo.sh                # classifier -> LLM explainer -> report
```

Detection is automatic; force with `CCR_PROFILE=4090 ./scan_repo.sh`.

## Troubleshooting (4090)

- **`torch.cuda.is_available()` is False** — CPU wheel got installed, or the WSL2
  NVIDIA driver isn't visible. Re-run `./env_doctor_cuda.sh`; reinstall torch from
  the channel matching `nvidia-smi`.
- **`no kernel image is available for execution on the device`** — the wheel
  lacks sm_89 kernels. Reinstall a cu12.x/cu13.x wheel (or
  `./install_torch_nightly_for_new_gpu.sh`); confirm with `env_doctor_cuda.sh`.
- **OOM at batch 64** — drop it: `BATCH=32 ./train.sh` (overrides the profile).
- **Everything scores ~0.5 / every file flagged** — you trained on the bootstrap
  set. Re-run `DATA=bigvul ./one_click_unlock_fetch_train_relock.sh`; verify with
  `python evaluate_model.py --model ./vuln-model --test data/test.jsonl`
  (ROC-AUC should be ≥0.9).
