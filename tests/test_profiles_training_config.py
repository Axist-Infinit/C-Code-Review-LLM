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


def test_training_config_laptop():
    cfg = profiles.training_config(name="laptop")
    assert cfg["profile"] == "laptop"
    assert cfg["batch_size"] == 16
    assert cfg["fp16"] is True and cfg["bf16"] is False
    # 12-16GB tier: the 14b q4 (9GB) + an 8k KV cache (1.5GB) measured 100% on
    # GPU on a 12227MiB card, so this tier serves the 14b, not the 7b.
    _, prof = profiles.select_profile(name="laptop")
    assert prof["ollama_model"] == "qwen2.5-coder:14b"
    assert prof["ollama_num_ctx"] == 8192


def test_training_config_laptop_small_keeps_the_7b():
    # Below ~11.5GB the 14b cannot stay resident; spilling layers to host RAM
    # does not fail, it just crawls — which is worse than a smaller model.
    _, prof = profiles.select_profile(name="laptop-small")
    assert prof["ollama_model"] == "qwen2.5-coder:7b"
    assert prof["ollama_num_ctx"] == 8192


def test_detect_profile_routes_12gb_gpu_to_laptop(monkeypatch):
    # A discrete CUDA GPU with <16GB VRAM must not get the 4090 profile (whose
    # 16k-context KV cache won't fit) — it gets the laptop tier.
    monkeypatch.setattr(profiles, "_gpu_name", lambda: "NVIDIA GeForce RTX 5070 Ti Laptop GPU")
    monkeypatch.setattr(profiles, "_gpu_mem_mib", lambda: 12227)
    assert profiles.detect_profile() == "laptop"


def test_detect_profile_routes_small_gpu_to_laptop_small(monkeypatch):
    monkeypatch.setattr(profiles, "_gpu_name", lambda: "NVIDIA GeForce RTX 3070")
    monkeypatch.setattr(profiles, "_gpu_mem_mib", lambda: 8192)
    assert profiles.detect_profile() == "laptop-small"


def test_detect_profile_vram_tier_boundary(monkeypatch):
    monkeypatch.setattr(profiles, "_gpu_name", lambda: "NVIDIA GeForce RTX 3080")
    monkeypatch.setattr(profiles, "_gpu_mem_mib", lambda: profiles._SMALL_GPU_VRAM_MIB - 1)
    assert profiles.detect_profile() == "laptop-small"
    monkeypatch.setattr(profiles, "_gpu_mem_mib", lambda: profiles._SMALL_GPU_VRAM_MIB)
    assert profiles.detect_profile() == "laptop"


def test_detect_profile_keeps_big_gpu_on_4090(monkeypatch):
    monkeypatch.setattr(profiles, "_gpu_name", lambda: "NVIDIA RTX A6000")
    monkeypatch.setattr(profiles, "_gpu_mem_mib", lambda: 49140)
    assert profiles.detect_profile() == "4090"


def test_detect_profile_unmeasurable_vram_defaults_to_4090(monkeypatch):
    # If VRAM can't be read, keep the historical discrete-GPU default (4090).
    monkeypatch.setattr(profiles, "_gpu_name", lambda: "NVIDIA GeForce RTX 3080")
    monkeypatch.setattr(profiles, "_gpu_mem_mib", lambda: None)
    assert profiles.detect_profile() == "4090"


def test_named_4090_still_wins_regardless_of_vram(monkeypatch):
    # An explicit 4090 in the name is matched before the VRAM branch.
    monkeypatch.setattr(profiles, "_gpu_name", lambda: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(profiles, "_gpu_mem_mib", lambda: 24564)
    assert profiles.detect_profile() == "4090"


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


# --- hardware-aware model recommendation ------------------------------------
# What decides whether a model runs well is residency: weights + KV cache must
# fit in VRAM, or llama.cpp offloads layers to host RAM and it crawls. These
# guard the arithmetic that drives ccr.sh's "optimal model" advice.

def test_footprint_matches_measured_residency():
    e14 = [m for m in profiles.MODEL_CATALOG if m["tag"].endswith(":14b")][0]
    # Measured on an RTX 5070 Ti Laptop: `ollama ps` reported 10GB for the 14b
    # at ctx 8192; the model must predict that within a few hundred MiB.
    assert 10400 <= profiles.model_footprint_mib(e14, 8192) <= 11100
    # Doubling the context adds exactly one KV cache's worth, not a fixed slab.
    kv8 = profiles.model_footprint_mib(e14, 8192) - e14["weights_mib"]
    kv16 = profiles.model_footprint_mib(e14, 16384) - e14["weights_mib"]
    assert abs(kv16 - 2 * kv8) <= 1


def test_recommend_picks_largest_fully_resident_model():
    assert profiles.recommend_model(vram_mib=12227, num_ctx=8192)["tag"] == "qwen2.5-coder:14b"
    assert profiles.recommend_model(vram_mib=24564, num_ctx=16384)["tag"] == "qwen2.5-coder:14b"
    assert profiles.recommend_model(vram_mib=49140, num_ctx=16384)["tag"] == "qwen2.5-coder:32b"
    assert profiles.recommend_model(vram_mib=8192, num_ctx=8192)["tag"] == "qwen2.5-coder:7b"


def test_recommend_downgrades_when_the_context_grows():
    # The SAME 12GB card holds the 14b at 8192 but not at 16384 — the KV cache,
    # not the weights, is what pushes it over.
    small_ctx = profiles.recommend_model(vram_mib=12227, num_ctx=8192)
    big_ctx = profiles.recommend_model(vram_mib=12227, num_ctx=16384)
    assert small_ctx["tag"] == "qwen2.5-coder:14b"
    assert big_ctx["tag"] == "qwen2.5-coder:7b"


def test_recommend_without_a_gpu_reports_that_it_does_not_fit():
    rec = profiles.recommend_model(vram_mib=None, num_ctx=8192)
    assert rec["tag"] == "qwen2.5-coder:7b" and rec["fits"] is False
    assert "no CUDA GPU" in rec["reason"]


def test_recommend_prefers_a_pulled_model_and_names_the_upgrade():
    rec = profiles.recommend_model(vram_mib=12227, num_ctx=8192,
                                   installed=["qwen2.5-coder:7b"])
    assert rec["tag"] == "qwen2.5-coder:7b"          # do not name an absent model
    assert rec["alternatives"] == ["qwen2.5-coder:14b"]
    assert "not pulled" in rec["reason"]


def test_tag_present_never_matches_a_different_size():
    have = ["qwen2.5-coder:7b", "qwen2.5-coder:14b"]
    assert profiles.tag_present("qwen2.5-coder:14b", have) is True
    # Verified regression: base-name matching made every size look pulled.
    assert profiles.tag_present("qwen2.5-coder:32b", have) is False
    # A tag with no ':size' may still resolve to any pulled variant.
    assert profiles.tag_present("qwen2.5-coder", have) is True


def test_detect_hardware_reports_the_fields_ccr_needs():
    hw = profiles.detect_hardware()
    assert set(hw) == {"gpu", "vram_mib", "ram_mib", "arch"}
    assert hw["arch"]
