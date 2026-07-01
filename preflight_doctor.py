#!/usr/bin/env python3
"""Preflight 'doctor' — run BEFORE a long train/scan job to catch the usual
failure modes cheaply instead of 30 minutes in:

  * the classifier model dir exists and looks complete (config + safetensors)
  * a tuned decision threshold is present (inference.json)
  * the hardware profile resolves (and its training/precision config is sane)
  * Ollama is reachable and the profile's explainer model is pulled

The CHECK LOGIC is a pure function (``evaluate_preflight``) over a facts dict so
it is unit-testable without torch / a real model dir / a running Ollama. The I/O
(stat the model dir, hit the Ollama HTTP endpoint) is gathered separately in
``gather_facts`` and only runs when the module is executed.

Exit codes: 0 = all required checks pass (warnings allowed); 1 = a required
check failed. ``--require-ollama`` promotes the Ollama check from warn to fail.
"""
import argparse
import json
import os
import shutil
import sys

# Reuse the SAME reachability probe the explainer uses, so preflight and the
# real run agree on what "reachable" means. Import is torch-free.
from llm_explain import ollama_available
from profiles import select_profile, training_config, add_profile_arg


REQUIRED = "required"
WARN = "warn"


def evaluate_preflight(facts, require_ollama=False, ml_expected=True):
    """Pure decision logic. ``facts`` keys:
        model_dir, model_dir_exists, has_config, has_weights,
        threshold (float|None), profile_name, profile_ok (bool),
        ollama_reachable (bool), ollama_model_present (bool), ollama_model (str),
        python_version (tuple|None), zstd_present (bool), zstd_path (str|None),
        offline_env (dict var -> bool)

    Returns (ok: bool, checks: list[dict]). Each check:
        {name, level, passed, detail}. ``ok`` is False iff any REQUIRED check
        failed. Ollama checks are WARN unless ``require_ollama`` is set; the
        python>=3.11 floor is REQUIRED when ``ml_expected`` (the pinned ML deps
        need it) and report-only otherwise (model-free lane runs on 3.10).
    """
    checks = []

    def add(name, level, passed, detail):
        checks.append({"name": name, "level": level, "passed": bool(passed), "detail": detail})

    md = facts.get("model_dir", "?")
    add("model dir exists", REQUIRED, facts.get("model_dir_exists"),
        f"{md} {'present' if facts.get('model_dir_exists') else 'MISSING (train or unpack a model first)'}")
    add("model config present", REQUIRED, facts.get("has_config"),
        f"config.json {'found' if facts.get('has_config') else 'MISSING in ' + md}")
    add("model weights present", REQUIRED, facts.get("has_weights"),
        "*.safetensors found" if facts.get("has_weights")
        else f"no *.safetensors in {md} (legacy *.bin is refused for supply-chain safety)")

    thr = facts.get("threshold")
    thr_ok = isinstance(thr, (int, float)) and 0.0 < float(thr) < 1.0
    add("tuned threshold present", REQUIRED, thr_ok,
        f"inference.json threshold={thr}" if thr_ok
        else "no usable tuned threshold in inference.json (re-run training to tune it)")

    add("hardware profile resolves", REQUIRED, facts.get("profile_ok"),
        f"profile={facts.get('profile_name')}" if facts.get("profile_ok")
        else "could not resolve a hardware profile")

    pv = facts.get("python_version")
    pv_str = ".".join(str(x) for x in pv) if pv else "unknown"
    pv_ok = bool(pv) and tuple(pv) >= (3, 11)
    add("python >= 3.11", REQUIRED if ml_expected else WARN, pv_ok,
        f"python {pv_str}" if pv_ok
        else f"python {pv_str} — the pinned ML deps (numpy==2.3.3) need 3.11+"
             + ("" if ml_expected else " (report-only: model-free lane runs on 3.10)"))

    add("zstd available", REQUIRED if ml_expected else WARN, facts.get("zstd_present"),
        f"zstd at {facts.get('zstd_path')}" if facts.get("zstd_present")
        else "zstd binary missing — pack/unpack_model.sh need it (apt-get install zstd)"
             + ("" if ml_expected else " (report-only: no model transfer on the model-free lane)"))

    offline = facts.get("offline_env") or {}
    set_vars = sorted(k for k, v in offline.items() if v)
    add("offline env state", WARN, True,
        ("offline: " + ", ".join(set_vars) + " set") if set_vars
        else "no HF_*_OFFLINE vars set (this shell is online-capable)")

    ollama_level = REQUIRED if require_ollama else WARN
    add("ollama reachable", ollama_level, facts.get("ollama_reachable"),
        "reachable" if facts.get("ollama_reachable")
        else "Ollama not reachable (explainer falls back to regex heuristics)")
    add("ollama model pulled", ollama_level,
        facts.get("ollama_reachable") and facts.get("ollama_model_present"),
        f"{facts.get('ollama_model')} present" if facts.get("ollama_model_present")
        else f"model {facts.get('ollama_model')!r} not pulled (ollama pull {facts.get('ollama_model')})")

    ok = all(c["passed"] for c in checks if c["level"] == REQUIRED)
    return ok, checks


def gather_facts(model_dir, profile_name, ollama_url):
    facts = {"model_dir": model_dir}
    facts["model_dir_exists"] = os.path.isdir(model_dir)
    facts["has_config"] = os.path.isfile(os.path.join(model_dir, "config.json"))
    facts["has_weights"] = bool(
        facts["model_dir_exists"]
        and any(n.endswith(".safetensors") for n in os.listdir(model_dir))
    ) if facts["model_dir_exists"] else False

    thr = None
    inf = os.path.join(model_dir, "inference.json")
    if os.path.isfile(inf):
        try:
            with open(inf, "r", encoding="utf-8") as f:
                thr = json.load(f).get("threshold")
        except Exception:
            thr = None
    facts["threshold"] = thr

    facts["python_version"] = tuple(sys.version_info[:2])
    facts["zstd_path"] = shutil.which("zstd")
    facts["zstd_present"] = bool(facts["zstd_path"])
    facts["offline_env"] = {
        k: os.environ.get(k, "") not in ("", "0")
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }

    profile = None
    try:
        profile, prof = select_profile(name=profile_name)
        cfg = training_config(name=profile_name)
        facts["profile_ok"] = bool(cfg["batch_size"] > 0)
        facts["ollama_model"] = prof.get("ollama_model")
    except Exception:
        facts["profile_ok"] = False
        facts["ollama_model"] = None
    facts["profile_name"] = profile

    reachable, present = (False, False)
    if facts.get("ollama_model"):
        reachable, present = ollama_available(ollama_url, facts["ollama_model"])
    facts["ollama_reachable"] = reachable
    facts["ollama_model_present"] = present
    return facts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Preflight doctor for train/scan runs.")
    ap.add_argument("--model", default=os.environ.get("MODEL", "./vuln-model"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--require-ollama", action="store_true",
                    help="Treat an unreachable Ollama / missing model as a failure, not a warning.")
    ap.add_argument("--skip-ml", action="store_true",
                    help="Model-free lane only: the python>=3.11 floor becomes report-only.")
    add_profile_arg(ap)
    args = ap.parse_args(argv)

    facts = gather_facts(args.model, args.profile, args.ollama_url)
    ok, checks = evaluate_preflight(facts, require_ollama=args.require_ollama,
                                    ml_expected=not args.skip_ml)
    for c in checks:
        if c["passed"]:
            mark, color = "PASS", "32"
        elif c["level"] == REQUIRED:
            mark, color = "FAIL", "31"
        else:
            mark, color = "WARN", "33"
        sys.stdout.write(f"\033[1;{color}m[{mark}]\033[0m {c['name']}: {c['detail']}\n")
    print("[preflight] OK" if ok else "[preflight] FAILED — fix the [FAIL] items above before a long run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
