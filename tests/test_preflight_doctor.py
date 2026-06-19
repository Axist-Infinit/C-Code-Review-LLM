"""Torch-free unit tests for the preflight doctor decision logic
(preflight_doctor.evaluate_preflight). No torch / model dir / Ollama needed."""
from preflight_doctor import evaluate_preflight, REQUIRED, WARN


def _facts(**over):
    f = dict(
        model_dir="./vuln-model", model_dir_exists=True, has_config=True,
        has_weights=True, threshold=0.91, profile_name="4090", profile_ok=True,
        ollama_reachable=True, ollama_model_present=True, ollama_model="qwen2.5-coder:14b",
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
