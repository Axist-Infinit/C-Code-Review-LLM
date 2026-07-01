"""Unit tests for evaluate_model's pure helpers (torch-free).

evaluate_model imports torch lazily inside score_dataset, so importing the
module here does not require torch.
"""
import hashlib
import json
import math
import os
import re

from evaluate_model import (metrics_at, group_metrics, check_gate, load_max_length,
                            build_provenance, dataset_provenance, file_sha256)


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


def test_group_metrics_by_source_and_lang():
    # 'source' (func_before/func_after) and 'lang' (c/cpp) are ordinary group
    # keys — the fetch tags them, generic grouping must slice on them.
    scores = [0.9, 0.2, 0.8, 0.1]
    labels = [1, 0, 1, 0]
    by_src = group_metrics(scores, labels, ["func_before", "func_after",
                                            "func_before", "func_after"], 0.5)
    assert set(by_src) == {"func_before", "func_after"}
    assert by_src["func_before"]["tp"] == 2
    assert by_src["func_after"]["tn"] == 2 and by_src["func_after"]["fp"] == 0
    by_lang = group_metrics(scores, labels, ["c", "cpp", "c", "cpp"], 0.5)
    assert by_lang["c"]["n"] == 2 and by_lang["cpp"]["n"] == 2


# --- quality gate -------------------------------------------------------------

def test_check_gate_passes_when_floors_met():
    metrics = {"f1": 0.6, "roc_auc": 0.95, "pr_auc": 0.5}
    assert check_gate(metrics, {"f1": 0.5, "roc_auc": 0.9, "pr_auc": 0.4}) == []


def test_check_gate_reports_each_violated_floor():
    metrics = {"f1": 0.3, "roc_auc": 0.95, "pr_auc": 0.1}
    fails = check_gate(metrics, {"f1": 0.5, "roc_auc": 0.9, "pr_auc": 0.4})
    assert len(fails) == 2
    assert any(f.startswith("f1:") for f in fails)
    assert any(f.startswith("pr_auc:") for f in fails)


def test_check_gate_none_floors_are_skipped():
    assert check_gate({"f1": 0.0}, {"f1": None, "pr_auc": None}) == []
    assert check_gate({"f1": 0.0}, {}) == []
    assert check_gate({"f1": 0.0}, None) == []


def test_check_gate_missing_metric_with_floor_fails():
    # sklearn absent -> no roc_auc in results; a floor on it must NOT pass
    fails = check_gate({"f1": 0.9}, {"roc_auc": 0.9})
    assert len(fails) == 1 and "unavailable" in fails[0]


def test_check_gate_boundary_is_inclusive():
    assert check_gate({"pr_auc": 0.4}, {"pr_auc": 0.4}) == []


# --- max_length from inference.json --------------------------------------------

def test_load_max_length_reads_inference_json(tmp_path):
    (tmp_path / "inference.json").write_text(
        json.dumps({"threshold": 0.44, "max_length": 256}))
    assert load_max_length(str(tmp_path)) == 256


def test_load_max_length_defaults(tmp_path):
    # pre-contract inference.json (no key) and missing file both -> default
    (tmp_path / "inference.json").write_text(json.dumps({"threshold": 0.44}))
    assert load_max_length(str(tmp_path)) == 512
    assert load_max_length(str(tmp_path / "nope")) == 512
    assert load_max_length(str(tmp_path / "nope"), default=256) == 256


def test_load_max_length_tolerates_garbage(tmp_path):
    (tmp_path / "inference.json").write_text("not json {")
    assert load_max_length(str(tmp_path)) == 512


# --- provenance ----------------------------------------------------------------

def test_file_sha256_streams_correctly(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"x" * (3 << 20) + b"tail")
    assert file_sha256(str(p)) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_dataset_provenance_rows_and_hash(tmp_path):
    p = tmp_path / "d.jsonl"
    body = b'{"code": "a", "label": 1}\n{"code": "b", "label": 0}\n'
    p.write_bytes(body)
    out = dataset_provenance([str(p), str(tmp_path / "missing.jsonl"), None])
    assert list(out) == [str(p)]  # missing/None skipped silently
    assert out[str(p)]["rows"] == 2
    assert out[str(p)]["sha256"] == hashlib.sha256(body).hexdigest()


def test_dataset_provenance_counts_final_unterminated_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": 2}')  # no trailing newline
    assert dataset_provenance([str(p)])[str(p)]["rows"] == 2


def test_build_provenance_core_keys():
    prov = build_provenance()
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", prov["git_rev"])
    assert prov["python_version"].count(".") == 2
    assert isinstance(prov["transformers_version"], str)
    # ISO-8601 UTC timestamp
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", prov["created_utc"])
    assert "model_sha256" not in prov  # no model_dir given


def test_build_provenance_model_hash_and_extra(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights-bytes")
    prov = build_provenance(model_dir=str(tmp_path),
                            extra={"seed": 42, "git_rev": "override"})
    assert prov["model_sha256"] == hashlib.sha256(b"weights-bytes").hexdigest()
    assert prov["seed"] == 42
    assert prov["git_rev"] == "override"  # extra wins on collision


def test_build_provenance_skips_missing_weights(tmp_path):
    assert "model_sha256" not in build_provenance(model_dir=str(tmp_path))
