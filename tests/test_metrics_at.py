"""Unit tests for evaluate_model.metrics_at (pure function, torch-free).

evaluate_model imports torch lazily inside score_dataset, so importing the
module here does not require torch.
"""
import math

from evaluate_model import metrics_at, group_metrics


def test_perfect_separation():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1, 1, 0, 0]
    m = metrics_at(scores, labels, 0.5)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["tn"] == 2
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["threshold"] == 0.5


def test_threshold_boundary_is_inclusive():
    # score == thr predicts positive (s >= thr)
    m = metrics_at([0.5], [1], 0.5)
    assert m["tp"] == 1
    assert m["fn"] == 0


def test_all_negative_predictions():
    scores = [0.1, 0.2, 0.3]
    labels = [1, 0, 1]
    m = metrics_at(scores, labels, 0.9)
    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 2
    assert m["tn"] == 1
    assert m["precision"] == 0.0  # no positive predictions -> guarded to 0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_mixed_confusion_matrix():
    #          score  label  thr=0.5 -> pred
    # 0.7  1  -> tp
    # 0.6  0  -> fp
    # 0.4  1  -> fn
    # 0.3  0  -> tn
    scores = [0.7, 0.6, 0.4, 0.3]
    labels = [1, 0, 1, 0]
    m = metrics_at(scores, labels, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert math.isclose(m["precision"], 0.5)
    assert math.isclose(m["recall"], 0.5)
    assert math.isclose(m["f1"], 0.5)
    assert math.isclose(m["accuracy"], 0.5)


def test_empty_inputs_do_not_divide_by_zero():
    m = metrics_at([], [], 0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["accuracy"] == 0.0
    assert m["tp"] == m["fp"] == m["fn"] == m["tn"] == 0


def test_group_metrics_splits_by_key():
    #            score label group
    scores = [0.9, 0.1, 0.8, 0.2]
    labels = [1,   0,   1,   1]
    groups = ["bof", "bof", "fmt", "fmt"]
    g = group_metrics(scores, labels, groups, 0.5)
    assert set(g) == {"bof", "fmt"}
    # bof: perfect separation
    assert g["bof"]["n"] == 2
    assert g["bof"]["tp"] == 1 and g["bof"]["tn"] == 1
    assert g["bof"]["f1"] == 1.0
    # fmt: one caught (0.8>=0.5, label 1), one missed (0.2<0.5, label 1) -> recall .5
    assert g["fmt"]["n"] == 2
    assert g["fmt"]["tp"] == 1 and g["fmt"]["fn"] == 1
    assert g["fmt"]["recall"] == 0.5


def test_group_metrics_keys_are_sorted():
    g = group_metrics([0.9, 0.9, 0.9], [1, 1, 1], ["z", "a", "m"], 0.5)
    assert list(g.keys()) == ["a", "m", "z"]


def test_group_metrics_empty():
    assert group_metrics([], [], [], 0.5) == {}
