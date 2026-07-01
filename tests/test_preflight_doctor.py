"""Torch-free unit tests for the preflight doctor decision logic
(preflight_doctor.evaluate_preflight). No torch / model dir / Ollama needed."""
from preflight_doctor import evaluate_preflight, REQUIRED, WARN


def _facts(**over):
    f = dict(
        model_dir="./vuln-model", model_dir_exists=True, has_config=True,
        has_weights=True, threshold=0.91, profile_name="4090", profile_ok=True,
        ollama_reachable=True, ollama_model_present=True, ollama_model="qwen2.5-coder:14b",
        python_version=(3, 12), zstd_present=True, zstd_path="/usr/bin/zstd",
        offline_env={},
    )
    f.update(over)
    return f


def test_all_good_passes():
    ok, checks = evaluate_preflight(_facts())
    assert ok is True
    assert all(c["passed"] for c in checks)


def test_missing_model_dir_fails():
    ok, _ = evaluate_preflight(_facts(model_dir_exists=False, has_config=False, has_weights=False))
    assert ok is False


def test_missing_weights_fails():
    ok, checks = evaluate_preflight(_facts(has_weights=False))
    assert ok is False
    weights = next(c for c in checks if c["name"] == "model weights present")
    assert weights["level"] == REQUIRED and weights["passed"] is False


def test_no_threshold_fails():
    assert evaluate_preflight(_facts(threshold=None))[0] is False
    # out-of-range threshold also fails
    assert evaluate_preflight(_facts(threshold=0.0))[0] is False
    assert evaluate_preflight(_facts(threshold=1.0))[0] is False
    assert evaluate_preflight(_facts(threshold=0.5))[0] is True


def test_unresolved_profile_fails():
    assert evaluate_preflight(_facts(profile_ok=False))[0] is False


def test_ollama_down_is_warning_by_default():
    ok, checks = evaluate_preflight(_facts(ollama_reachable=False, ollama_model_present=False))
    assert ok is True  # warn, not fail
    reach = next(c for c in checks if c["name"] == "ollama reachable")
    assert reach["level"] == WARN and reach["passed"] is False


def test_ollama_down_fails_when_required():
    ok, checks = evaluate_preflight(
        _facts(ollama_reachable=False, ollama_model_present=False), require_ollama=True)
    assert ok is False
    reach = next(c for c in checks if c["name"] == "ollama reachable")
    assert reach["level"] == REQUIRED


def test_ollama_reachable_but_model_missing_warns():
    ok, checks = evaluate_preflight(_facts(ollama_model_present=False))
    assert ok is True
    pulled = next(c for c in checks if c["name"] == "ollama model pulled")
    assert pulled["passed"] is False


def test_old_python_fails_when_ml_expected():
    ok, checks = evaluate_preflight(_facts(python_version=(3, 10)))
    assert ok is False
    py = next(c for c in checks if c["name"] == "python >= 3.11")
    assert py["level"] == REQUIRED and py["passed"] is False
    assert "3.10" in py["detail"]


def test_old_python_is_report_only_without_ml():
    ok, checks = evaluate_preflight(_facts(python_version=(3, 10)), ml_expected=False)
    assert ok is True  # warn, not fail
    py = next(c for c in checks if c["name"] == "python >= 3.11")
    assert py["level"] == WARN and py["passed"] is False


def test_unknown_python_version_fails_when_ml_expected():
    assert evaluate_preflight(_facts(python_version=None))[0] is False


def test_new_python_passes():
    for pv in ((3, 11), (3, 12), (3, 13)):
        ok, checks = evaluate_preflight(_facts(python_version=pv))
        assert ok is True, pv


def test_missing_zstd_fails():
    ok, checks = evaluate_preflight(_facts(zstd_present=False, zstd_path=None))
    assert ok is False
    z = next(c for c in checks if c["name"] == "zstd available")
    assert z["level"] == REQUIRED and z["passed"] is False


def test_missing_zstd_report_only_without_ml():
    # model-free lane never packs/unpacks a model tar, so zstd must not block
    ok, checks = evaluate_preflight(_facts(zstd_present=False, zstd_path=None),
                                    ml_expected=False)
    z = next(c for c in checks if c["name"] == "zstd available")
    assert z["level"] == WARN and z["passed"] is False
    assert ok is True


def test_offline_env_state_is_report_only():
    # never a failure, whichever way the vars point — but the detail must say
    # which offline vars are set
    ok, checks = evaluate_preflight(_facts(offline_env={
        "HF_HUB_OFFLINE": True, "TRANSFORMERS_OFFLINE": True, "HF_DATASETS_OFFLINE": False}))
    assert ok is True
    st = next(c for c in checks if c["name"] == "offline env state")
    assert st["passed"] is True and st["level"] == WARN
    assert "HF_HUB_OFFLINE" in st["detail"] and "HF_DATASETS_OFFLINE" not in st["detail"]

    ok, checks = evaluate_preflight(_facts(offline_env={}))
    assert ok is True
    st = next(c for c in checks if c["name"] == "offline env state")
    assert st["passed"] is True
    assert "online-capable" in st["detail"]
