#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import torch, json
info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
if torch.cuda.is_available():
    info["num_gpus"] = torch.cuda.device_count()
    info["name"] = torch.cuda.get_device_name(0)
    info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
print(json.dumps(info, indent=2))
PY
