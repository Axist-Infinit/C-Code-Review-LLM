# Airgapped install — build online, install offline

Goal: scan C/C++ with the already-trained classifier on a machine with **zero
network** (x86_64 or aarch64 Linux/WSL2). You build a bundle once on a
connected machine, carry it over (USB, etc.), and install from local files
only.

## 1. Build the bundle (CONNECTED machine)

Needs: network, `git`, `python3` with `pip >= 23` (older pips mishandle
cross-platform `--platform` downloads), and ideally `zstd`.

```bash
# full bundle: both arches, python 3.10–3.13 wheels, ML lane + model-free lane
./build_airgap_bundle.sh --model /path/to/real-model

# common variations
./build_airgap_bundle.sh --arch x86_64 --python 3.11,3.12   # one arch, two pythons
./build_airgap_bundle.sh --skip-ml                          # model-free lane only
CCR_TORCH_SPEC='torch==2.6.*' ./build_airgap_bundle.sh      # pin the CPU torch
./build_airgap_bundle.sh --ollama-models ~/.ollama/models   # bundle LLM blobs too
```

The default `--python 3.10,3.11,3.12,3.13` covers any reasonably current
distro. The pinned ML deps need python **>= 3.11** (the numpy pin), so pythons
below that get a **model-free** wheelhouse (tree-sitter trio only), and the
installer on such a machine automatically degrades to the model-free lane with
a notice — it never fails halfway through a pip install. Each wheelhouse
carries a `WHEELHOUSE_KIND` marker (`full` / `model-free`) so the installer
knows which lane it can offer.

Output: `dist/airgap/ccr-airgap-<gitrev>.tar[.zst]` containing

| path | contents |
| --- | --- |
| `repo/` | the tracked source tree (`git archive HEAD`) |
| `model/model.tar.zst` (+ `.manifest.json`) | the trained model, checkpoints/pickles stripped, sha256-manifested |
| `wheels/<arch>/cp<ver>/` | pinned wheels per arch (+ CPU torch unless `--skip-ml`) |
| `install_offline.sh`, `README_AIRGAP.md` | the offline installer + this doc |
| `bundle_manifest.json`, `SHA256SUMS` | build metadata + integrity hashes |

If any required wheel fails to download the builder keeps collecting the rest,
prints the exact failed packages at the end, and exits non-zero — do not ship
an incomplete bundle.

### Include the REAL trained model (not the bootstrap artifact)

The `vuln-model/` committed in this repo is a **5-step bootstrap smoke
artifact**, not the real 0.959 classifier. The builder detects this and warns
loudly when the `--model` dir contains a `checkpoint-*` directory or a
`trainer_state.json` with `global_step <= 10` (a real BigVul run takes
thousands of steps). If you see that warning, point `--model` at the directory
your training run actually wrote (the `OUT=` of
`one_click_unlock_fetch_train_relock.sh`), or transfer one from the training
machine first:

```bash
# on the training box
./pack_model.sh /path/to/real-model real_model.tar.zst
# then build with the unpacked result
./unpack_model.sh real_model.tar.zst ./staging && ./build_airgap_bundle.sh --model staging/real-model
```

Weights are arch-independent safetensors — one model serves both x86_64 and
aarch64 targets.

## 2. Transfer

Copy the single tarball to the offline machine (USB stick etc.) and extract:

```bash
tar -I zstd -xf ccr-airgap-<rev>.tar.zst    # or: tar -xf ccr-airgap-<rev>.tar
cd ccr-airgap-<rev>
```

## 3. Install (OFFLINE machine)

Needs: `python3` (**>= 3.11** for the ML lane; 3.10 is enough for `--skip-ml`
bundles), `python3-venv`, `zstd` (to unpack the model), `sha256sum`. No
network.

```bash
./install_offline.sh verify-only   # optional: integrity + arch/python check only
./install_offline.sh               # venv + wheels + model + smoke test
```

The installer:

1. verifies every file against `SHA256SUMS` (hard fail on mismatch),
2. picks `wheels/<arch>/cp<pyver>` for this machine (clear error listing the
   available combos when the bundle was built for something else),
3. creates `.venv` and installs from the wheelhouse with `pip --no-index`,
4. unpacks the model to `./vuln-model` (manifest-verified),
5. writes `.env.locked` (`HF_HUB_OFFLINE=1` etc.) and hooks it into venv
   activation, so every shell that activates the venv is offline-locked,
6. smoke-tests the model-free lane end to end and, when torch was installed,
   import-checks torch/transformers + reads the model config. `PASS`/`FAIL`
   summary; non-zero exit on failure.

## 4. Run scans

From the bundle root:

```bash
source .venv/bin/activate          # offline env vars auto-applied

# ML lane (classifier + explainer; CPU works, CUDA used if present)
python3 repo/local_vuln_scanner.py <src-dir> -o scan_out --model ./vuln-model --threshold 0.5
python3 repo/llm_explain.py scan_out/classifier_findings.json --out scan_out/llm_findings.json --html scan_out/report.html

# model-free lane (no torch, no model, pure python)
python3 repo/heuristic_scan.py <src-dir> -o scan_out_h
python3 repo/llm_explain.py scan_out_h/classifier_findings.json --out scan_out_h/llm_findings.json --backend heuristic
python3 repo/to_sarif.py scan_out_h/llm_findings.json -o scan_out_h/findings.sarif
```

(The wrapper `repo/scan_repo.sh` also works when run from inside `repo/`:
`cd repo && VENV=../.venv MODEL=../vuln-model SRC=<src-dir> bash ./scan_repo.sh`.)

The heuristic explainer (`--backend heuristic`) needs **zero extra
dependencies** — no Ollama, no GPU, no network. It is what the smoke test and
CI use.

## 5. Optional: offline Ollama for LLM explanations

The richer explainer lane (`llm_explain.py` default backend) talks to a local
Ollama at `localhost:11434`. To have it offline:

1. On a connected machine, download the Ollama release tarball for the target
   arch from `https://github.com/ollama/ollama/releases` (e.g.
   `ollama-linux-amd64.tgz` / `ollama-linux-arm64.tgz`) and `ollama pull` the
   profile model (see `profiles.json`, e.g. `qwen2.5-coder:14b`).
2. Copy the binary tarball plus the blob store `~/.ollama/models` — or build
   the bundle with `--ollama-models ~/.ollama/models` so it lands in
   `ollama/models/` inside the bundle.
3. On the offline box: extract the binary, then

   ```bash
   export OLLAMA_MODELS=/path/to/bundle/ollama/models
   ollama serve &
   ```

Ollama never needs the network after the pull; without it, everything falls
back to the heuristic backend automatically.

## Troubleshooting

| symptom | cause | fix |
| --- | --- | --- |
| `no wheelhouse for this machine: wheels/aarch64/cp312` + a combo list | bundle built for other arch/python | rebuild with `--arch <uname -m>` / `--python <X.Y>`; the error prints what the bundle does contain |
| `python3 is 3.10 but the ML install needs >= 3.11` | full wheelhouse selected on a python below the numpy-pin floor (only possible with a custom-built bundle) | install python3.11+ (`deadsnakes`/distro backport, offline .debs), or pass `--skip-ml` |
| `wheelhouse ... is model-free` notice during a full install | this machine's python is below the ML floor — default bundles ship a tree-sitter-only wheelhouse for it | expected: the model-free lane installs and works; for the ML lane install python3.11+ and re-run (its full wheelhouse is already in the bundle) |
| `SHA256SUMS verification FAILED` | corrupt/tampered transfer | re-copy the tarball; compare `sha256sum` of the tarball on both machines |
| `archive sha256 mismatch` from unpack_model.sh | model tar corrupt | rebuild/re-copy; the manifest travels next to the tar |
| `zstd not found` | zstd not installed on target | install the OS `zstd` package (offline .deb) — it is intentionally not bundled |
| model-free smoke passes, ML smoke `[SKIP]` | `--skip-ml` bundle or torch wheel failed at build | rebuild without `--skip-ml`, check the builder's failure summary |
| scans warn the model looks untrained | bundle carries the bootstrap artifact | rebuild with `--model` pointing at the real classifier (see section 1) |
