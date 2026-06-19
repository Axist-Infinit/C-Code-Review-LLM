"""Torch-free unit tests for the profile -> training-config decision logic
(profiles.py). This is the wiring that makes per-machine training real:
4090 -> fp16/batch64, Spark -> bf16/batch128, cpu -> fp32/batch8.
"""
import json
import os

import profiles


def test_precision_flags_fp16():
    f = profiles.precision_flags_for_dtype("float16")
    assert f == {"fp16": True, "bf16": False}
    assert profiles.precision_flags_for_dtype("fp16") == {"fp16": True, "bf16": False}


def test_precision_flags_bf16():
    f = profiles.precision_flags_for_dtype("bfloat16")
    assert f == {"fp16": False, "bf16": True}
    assert profiles.precision_flags_for_dtype("bf16") == {"fp16": False, "bf16": True}


def test_precision_flags_fp32_and_unknown_are_full_precision():
    assert profiles.precision_flags_for_dtype("float32") == {"fp16": False, "bf16": False}
    # unknown/None must not raise and must default to full precision (safe)
    assert profiles.precision_flags_for_dtype("???") == {"fp16": False, "bf16": False}
    assert profiles.precision_flags_for_dtype(None) == {"fp16": False, "bf16": False}


def test_fp16_and_bf16_are_never_both_set():
    for dtype in ("float16", "bfloat16", "float32", "fp16", "bf16", "fp32", "weird"):
        f = profiles.precision_flags_for_dtype(dtype)
        assert not (f["fp16"] and f["bf16"]), dtype


def test_training_config_4090():
    cfg = profiles.training_config(name="4090")
    assert cfg["profile"] == "4090"
    assert cfg["batch_size"] == 64
    assert cfg["fp16"] is True
    assert cfg["bf16"] is False
    assert cfg["dtype"] == "float16"


def test_training_config_spark():
    cfg = profiles.training_config(name="spark")
    assert cfg["profile"] == "spark"
    assert cfg["batch_size"] == 128
    assert cfg["fp16"] is False
    assert cfg["bf16"] is True
    assert cfg["dtype"] == "bfloat16"


def test_training_config_cpu_is_full_precision():
    cfg = profiles.training_config(name="cpu")
    assert cfg["profile"] == "cpu"
    assert cfg["batch_size"] == 8
    assert cfg["fp16"] is False
    assert cfg["bf16"] is False


def test_training_config_matches_profiles_json_batch_and_dtype():
    """The training config must faithfully reflect profiles.json — guard against
    drift between the JSON and the mapping."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "profiles.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for name, prof in raw.items():
        cfg = profiles.training_config(name=name)
        assert cfg["batch_size"] == prof["classifier_batch_size"]
        assert cfg["dtype"] == prof["classifier_dtype"]


def test_emit_training_env_shell_assignments(capsys):
    profiles._emit_training_env(name="spark")
    out = capsys.readouterr().out
    assert "CCR_BATCH=128" in out
    assert "CCR_PRECISION_FLAG=--bf16" in out
    assert "CCR_PROFILE=spark" in out

    profiles._emit_training_env(name="4090")
    out = capsys.readouterr().out
    assert "CCR_BATCH=64" in out
    assert "CCR_PRECISION_FLAG=--fp16" in out

    profiles._emit_training_env(name="cpu")
    out = capsys.readouterr().out
    assert "CCR_BATCH=8" in out
    # cpu -> no precision flag (empty)
    assert "CCR_PRECISION_FLAG=\n" in out or out.rstrip().endswith("CCR_PRECISION_FLAG=")
