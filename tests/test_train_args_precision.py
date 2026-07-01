"""Torch-free unit test for the precision wiring in
model/train_vuln_model.build_training_args.

model/train_vuln_model.py imports torch + transformers at module top, which are
not installed in CI. We inject minimal fakes into sys.modules so we can import
and exercise the *pure* arg-assembly logic (the fp16/bf16 mutual-exclusion guard
and that the chosen precision flag reaches TrainingArguments) without the heavy
ML stack or a GPU.
"""
import importlib.util
import os
import sys
import types

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MOD_PATH = os.path.join(_REPO, "model", "train_vuln_model.py")


class _FakeTrainingArguments:
    """Captures the kwargs it was constructed with; rejects unknown keys so the
    attempts-list fallback (eval_strategy vs evaluation_strategy) is exercised
    the same way HF's real class would by raising TypeError."""
    _ALLOWED = {
        "output_dir", "learning_rate", "per_device_train_batch_size",
        "per_device_eval_batch_size", "num_train_epochs", "weight_decay",
        "logging_steps", "seed", "report_to", "eval_strategy", "save_strategy",
        "load_best_model_at_end", "metric_for_best_model", "greater_is_better",
        "fp16", "bf16", "save_total_limit",
        # note: NO "evaluation_strategy" -> the legacy-spelling attempt fails,
        # so the modern eval_strategy attempt (index 0) wins.
    }

    def __init__(self, **kw):
        bad = set(kw) - self._ALLOWED
        if bad:
            raise TypeError(f"unexpected kwargs: {bad}")
        self.kw = kw


def _load_module():
    # minimal fakes for the heavy top-level imports
    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = lambda *a, **k: None
    fake_torch.float = "float"
    fake_torch.manual_seed = lambda *a, **k: None
    fake_torch.Tensor = type("Tensor", (), {})
    fake_torch.nn = types.SimpleNamespace(CrossEntropyLoss=object)
    fake_torch.softmax = lambda *a, **k: None

    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "fake"
    fake_tf.__file__ = "fake"
    fake_tf.AutoTokenizer = object
    fake_tf.AutoModelForSequenceClassification = object
    fake_tf.Trainer = object
    fake_tf.TrainingArguments = _FakeTrainingArguments
    fake_tf.DataCollatorWithPadding = object

    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = lambda *a, **k: None

    saved = {k: sys.modules.get(k) for k in ("torch", "transformers", "datasets")}
    sys.modules["torch"] = fake_torch
    sys.modules["transformers"] = fake_tf
    sys.modules["datasets"] = fake_ds
    try:
        spec = importlib.util.spec_from_file_location("train_vuln_model_fake", _MOD_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


tvm = _load_module()


def _args(**over):
    base = dict(out="o", lr=5e-5, batch=64, epochs=3, weight_decay=0.0, seed=42,
                fp16=False, bf16=False)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_fp16_flag_reaches_training_arguments():
    ta = tvm.build_training_args(_args(fp16=True))
    assert ta.kw["fp16"] is True
    assert ta.kw["bf16"] is False
    assert ta.kw["per_device_train_batch_size"] == 64
    # the modern eval_strategy attempt should have won
    assert ta.kw.get("eval_strategy") == "epoch"


def test_bf16_flag_reaches_training_arguments():
    ta = tvm.build_training_args(_args(bf16=True, batch=128))
    assert ta.kw["bf16"] is True
    assert ta.kw["fp16"] is False
    assert ta.kw["per_device_train_batch_size"] == 128


def test_full_precision_when_neither_flag():
    ta = tvm.build_training_args(_args())
    assert ta.kw["fp16"] is False
    assert ta.kw["bf16"] is False


def test_fp16_and_bf16_together_is_rejected():
    with pytest.raises(SystemExit):
        tvm.build_training_args(_args(fp16=True, bf16=True))


# --- checkpoint hygiene: save_total_limit in every attempt --------------------

@pytest.mark.parametrize("over", [dict(), dict(fp16=True), dict(bf16=True)])
def test_save_total_limit_always_set(over):
    ta = tvm.build_training_args(_args(**over))
    assert ta.kw["save_total_limit"] == 2


class _MinimalTrainingArguments:
    """Rejects report_to and both eval-strategy spellings, forcing the no-eval
    fallback attempts — save_total_limit must survive even there (it lives in
    the base kwargs shared by every attempt)."""
    _ALLOWED = {
        "output_dir", "learning_rate", "per_device_train_batch_size",
        "per_device_eval_batch_size", "num_train_epochs", "weight_decay",
        "logging_steps", "seed", "save_total_limit", "fp16", "bf16",
    }

    def __init__(self, **kw):
        bad = set(kw) - self._ALLOWED
        if bad:
            raise TypeError(f"unexpected kwargs: {bad}")
        self.kw = kw


def test_save_total_limit_survives_no_eval_fallback(monkeypatch):
    monkeypatch.setattr(tvm, "TrainingArguments", _MinimalTrainingArguments)
    ta = tvm.build_training_args(_args(fp16=True))
    assert ta.kw["save_total_limit"] == 2
    assert ta.kw["fp16"] is True and ta.kw["bf16"] is False  # precision kept
    assert "eval_strategy" not in ta.kw and "report_to" not in ta.kw


# --- opt-in resume helpers ----------------------------------------------------

def test_find_resume_checkpoint_none_cases(tmp_path):
    assert tvm.find_resume_checkpoint(None) is None
    assert tvm.find_resume_checkpoint(str(tmp_path / "missing")) is None
    # empty dir: either transformers is absent (lazy import fails) or
    # get_last_checkpoint finds no checkpoint-* subdir — both mean None
    assert tvm.find_resume_checkpoint(str(tmp_path)) is None


def _fake_checkpoint_detection(monkeypatch, tmp_path, name="checkpoint-42"):
    fake_tu = types.ModuleType("transformers.trainer_utils")
    fake_tu.get_last_checkpoint = lambda d: os.path.join(d, name)
    fake_tf = types.ModuleType("transformers")
    fake_tf.trainer_utils = fake_tu
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "transformers.trainer_utils", fake_tu)
    return os.path.join(str(tmp_path), name)


def test_find_resume_checkpoint_uses_transformers_helper(tmp_path, monkeypatch):
    expect = _fake_checkpoint_detection(monkeypatch, tmp_path)
    assert tvm.find_resume_checkpoint(str(tmp_path)) == expect


def test_resume_is_opt_in_stale_checkpoint_ignored_with_notice(tmp_path, monkeypatch, capsys):
    """Regression (finding 16): a stale/bootstrap checkpoint sitting in the
    reused default out dir must NOT be auto-resumed. Without --resume the run
    trains fresh (resume_from_checkpoint=None) and prints a notice naming the
    ignored checkpoint so operators discover --resume."""
    expect = _fake_checkpoint_detection(monkeypatch, tmp_path, name="checkpoint-5")
    got = tvm.resolve_resume(str(tmp_path), resume_requested=False)
    assert got is None  # fresh training despite the existing checkpoint
    out = capsys.readouterr().out
    assert expect in out            # the ignored checkpoint is named
    assert "--resume" in out        # ...and the opt-in flag is advertised


def test_resolve_resume_opt_in_uses_checkpoint(tmp_path, monkeypatch):
    expect = _fake_checkpoint_detection(monkeypatch, tmp_path)
    assert tvm.resolve_resume(str(tmp_path), resume_requested=True) == expect


def test_resolve_resume_opt_in_without_checkpoint_trains_fresh(tmp_path, capsys):
    # --resume on an empty/missing out dir must not crash; it trains fresh
    assert tvm.resolve_resume(str(tmp_path / "missing"), resume_requested=True) is None
    assert "training fresh" in capsys.readouterr().out.lower()


def test_resolve_resume_no_checkpoint_no_notice(tmp_path, capsys):
    assert tvm.resolve_resume(str(tmp_path), resume_requested=False) is None
    assert capsys.readouterr().out == ""  # nothing detected -> stay quiet


# --- inference.json payload (C1 contract) -------------------------------------

def test_inference_payload_contract():
    p = tvm.build_inference_payload(0.44, 512)
    assert p == {"threshold": 0.44, "max_length": 512,
                 "notes": "Max F1 on validation"}
    p2 = tvm.build_inference_payload(0.5, 256, provenance={"git_rev": "abc"})
    assert p2["max_length"] == 256
    assert p2["provenance"] == {"git_rev": "abc"}
    assert list(p2)[:3] == ["threshold", "max_length", "notes"]


def test_parse_args_defaults_resume_off():
    base = ["--train", "t", "--val", "v", "--test", "x", "--out", "o"]
    a = tvm.parse_args(base)
    assert a.resume is False   # regression (finding 16): resume is OPT-IN
    assert a.max_length == 512
    assert a.seed == 42
    assert tvm.parse_args(base + ["--resume"]).resume is True


def test_parse_args_no_resume_is_accepted_noop():
    # back-compat: older wrappers passing --no-resume must not crash, and it
    # must not turn resume on
    base = ["--train", "t", "--val", "v", "--test", "x", "--out", "o"]
    a = tvm.parse_args(base + ["--no-resume"])
    assert a.resume is False
