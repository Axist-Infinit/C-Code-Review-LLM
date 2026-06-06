C-Code-Review-LLM — Minimal README (offline-capable)

Prereqs (Ubuntu/WSL Ubuntu):
  - git, git-lfs, python3, python3-venv, python3-pip, curl
  - Optional: NVIDIA GPU + driver + CUDA runtime (script handles CUDA wheels)

Quick start:
  1) unzip this bundle and cd into the folder
  2) run:  bash ./one_click_unlock_fetch_train_relock.sh
     - installs venv + deps
     - fetches GraphCodeBERT + BigVul(HF)
     - merges/splits datasets
     - trains classifier (GPU if available)
     - locks offline
     - scans playground/vuln_demo.c and writes findings

Outputs:
  - ./vuln-model/                 (trained classifier; local only)
  - ./data/{train,val,test}.jsonl (merged dataset)
  - ./scan_out/classifier_findings.json
  - ./scan_out/explained/llm_findings.json

Re-train (later):
  # unlock -> fetch/merge -> train -> lock -> scan
  bash ./one_click_unlock_fetch_train_relock.sh

Fully offline inference:
  source ./offline_lockdown.sh
  source ./.venv/bin/activate
  python ./local_vuln_scanner.py playground -o scan_out --model ./vuln-model
  python ./explain_findings.py --inp scan_out/classifier_findings.json --out scan_out/explained/llm_findings.json

Notes:
  - Optional datasets (MegaVul, CVEfixes, FuncVul) are cloned best‑effort; they won’t break the run.
  - If unzip strips executable bits, run: chmod +x *.sh scripts/*.sh
