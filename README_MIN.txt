C-Code-Review-LLM — Minimal README (offline-capable)

Prereqs (Ubuntu/WSL Ubuntu):
  - git, git-lfs, python3 (>= 3.11 for the ML lane), python3-venv, python3-pip, curl, zstd
  - Optional: NVIDIA GPU + driver + CUDA runtime (script handles CUDA wheels)

Quick start (ONLINE machine):
  1) unzip this bundle and cd into the folder
  2) run:  bash ./one_click_unlock_fetch_train_relock.sh
     - installs venv + deps (arch-aware torch)
     - fetches GraphCodeBERT + BigVul(HF)
     - merges/splits datasets
     - trains classifier (GPU if available) + evaluates it
     - locks offline (.env.locked sourced on every venv activation)
     NOTE: it does NOT run a scan — scan afterwards with:
  3) run:  bash ./scan_repo.sh          (or: bash ./run_demo.sh for playground/)

Outputs:
  - ./vuln-model/                    (trained classifier; local only)
  - ./data/{train,val,test}.jsonl    (merged dataset)
  after a scan (scan_repo.sh / run_demo.sh, default OUT=scan_out):
  - ./scan_out/classifier_findings.json
  - ./scan_out/llm_findings.json
  - ./scan_out/report.html

Re-train (later):
  # unlock -> fetch/merge -> train -> eval -> lock
  bash ./one_click_unlock_fetch_train_relock.sh

Go back online (unlock persists; must be SOURCED, not executed):
  source ./online_unlock.sh

Fully offline inference:
  source ./offline_lockdown.sh
  source ./.venv/bin/activate
  python ./local_vuln_scanner.py playground -o scan_out --model ./vuln-model
  python ./llm_explain.py scan_out/classifier_findings.json --out scan_out/llm_findings.json --backend heuristic

Fully offline INSTALL (airgapped target, no network at all):
  build a bundle on a connected machine with ./build_airgap_bundle.sh, carry it
  over, run ./install_offline.sh — see airgap/README_AIRGAP.md.

Notes:
  - If unzip strips executable bits, run: chmod +x *.sh scripts/*.sh airgap/*.sh
