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
        "fp16", "bf16",
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
