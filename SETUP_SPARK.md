# Setup & Runbook — NVIDIA DGX Spark (aarch64 Grace-Blackwell)

Authoritative per-machine guide for the **DGX Spark** (aarch64, Grace-Blackwell
**GB10** superchip, compute capability **sm_121**, 128 GB unified memory). For
the RTX 4090 see **[SETUP_4090.md](SETUP_4090.md)**; for the overview and shared
flags see **[SETUP.md](SETUP.md)**.

| | DGX Spark |
|---|---|
| arch / capability | aarch64 / sm_121 (Grace-Blackwell GB10) |
| profile (`profiles.json`) | `spark` — batch **128**, **bf16** |
| training flags wired | `--batch 128 --bf16` (from the profile, via `profiles.py --training-env`) |
| torch source | **cu130 NIGHTLY** wheel (aarch64) **or** NGC PyTorch SBSA container |
| explainer | `qwen2.5-coder:32b`, num_ctx 16384 |

## torch on DGX Spark — read this first

The Spark is **aarch64** with a Blackwell GPU at **sm_121**. Two facts drive the
install:

1. **sm_121 kernels only ship in cu130 builds.** A cu128/cu126 wheel will load
   but throw *"no kernel image is available for execution on the device"* on the
   first CUDA op.
2. **There is no guaranteed *stable* cu130 aarch64 wheel** on
   `download.pytorch.org`. The supported sources are:
   - the **cu130 NIGHTLY** channel:
     `https://download.pytorch.org/whl/nightly/cu130` (aarch64 wheels), **or**
   - the **NVIDIA NGC PyTorch container, SBSA (aarch64) tag** — the most
     reliable path on a fresh Spark, ships a torch already built for sm_121.

`lib_torch_install.sh` therefore resolves aarch64 to the **nightly cu130**
channel and installs with `--pre`, **torch only**. It **fails soft**: if the
nightly wheel is unavailable, `setup_machine.sh` warns and continues (it no
longer aborts the whole setup under `set -e`) and points you here.

```bash
./install_torch_nightly_for_new_gpu.sh    # arch-aware: aarch64 -> nightly cu130, torch only
```

If the nightly install fails, fall back to the NGC SBSA container (run the repo
inside it), then re-run the assertion below from within the container.

**Verified source to record (fill in after your online window):**

```bash
python -c "import torch,platform;print(platform.machine(), torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
# expect: aarch64  2.<x>.<y>.devYYYYMMDD+cu130  13.0  [..., 'sm_121']
```

Record that exact `torch.__version__` + the channel (`nightly/cu130` or the NGC
container tag) in your deployment notes — do **not** rely on "auto-selected".

> torch **only** — torchvision/torchaudio are unused and torchvision ≥0.25
> breaks the `datasets` torch formatter.

## 1. One-time setup

```bash
git clone https://github.com/Axist-Infinit/C-Code-Review-LLM.git
cd C-Code-Review-LLM
./setup_machine.sh           # installs the nightly cu130 aarch64 torch (fails soft)
```

Then assert kernels actually exist for this GPU **on the Spark itself**:

```bash
./env_doctor_cuda.sh
```

Expected: `capability 12.1 (sm_121)` then a finite matmul and `[doctor] PASS`.
A `[WARN] … wheel arch list … does NOT include sm_121` means you installed a
non-cu130 wheel (or an old nightly) — reinstall via
`./install_torch_nightly_for_new_gpu.sh` or switch to the NGC SBSA container.

## 2. Get a trained classifier

The Spark's 128 GB unified memory comfortably runs **batch 128 / bf16**
(the `spark` profile wires this automatically):

```bash
./preflight_doctor.sh || true
DATA=bigvul ./one_click_unlock_fetch_train_relock.sh   # profile -> --batch 128 --bf16
```

Hard-fails if BigVul yields < `MIN_TRAIN_ROWS` (default 20000) train rows — no
silent bootstrap. Or transfer an arch-independent model:
`./unpack_model.sh vuln-model-bigvul.tar.zst`.

## 3. Scan

```bash
./preflight_doctor.sh
SRC=path/to/code ./scan_repo.sh
```

Detection is automatic (aarch64 / GB10 → `spark`); force with
`CCR_PROFILE=spark ./scan_repo.sh`.

## Troubleshooting (Spark)

- **`no kernel image is available for execution on the device`** — you do NOT
  have a cu130 wheel. Reinstall: `./install_torch_nightly_for_new_gpu.sh`, or use
  the NGC SBSA container. Verify with `./env_doctor_cuda.sh`.
- **`setup_machine.sh` printed a torch-install warning and continued** — the
  nightly cu130 aarch64 wheel wasn't fetchable in your window. Use the NGC SBSA
  container path; the rest of the venv/deps are already installed.
- **Ollama 32b model is large** — skip the pull with `PULL_OLLAMA=0
  ./setup_machine.sh` and `ollama pull qwen2.5-coder:32b` during your online
  window; or use a smaller model via `CCR_PROFILE=4090` (14b).
- **bf16 vs fp16** — the Spark profile uses **bf16** (Blackwell native, wider
  range, no fp16 overflow). Don't pass `--fp16` here; the profile already
  resolves to `--bf16`.
