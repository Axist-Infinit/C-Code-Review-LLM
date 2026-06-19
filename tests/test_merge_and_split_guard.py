"""Torch-free unit tests for the merge_and_split HARD-FAIL guard logic.

The key fix under test: a failed/empty/too-small BigVul ingest must NOT silently
write a 12-row bootstrap set on the real (DATA=bigvul) path. It must hard-fail
(non-zero exit) so training never runs on a toy set masquerading as BigVul.
Pure stdlib — no torch / datasets / network.
"""
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_PATH = os.path.join(os.path.dirname(_HERE), "scripts", "py", "merge_and_split.py")


def _load_mod():
    spec = importlib.util.spec_from_file_location("merge_and_split", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mas = _load_mod()


def _write_ingest(tmp_path, name, rows):
    d = tmp_path / "ingest"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(d)


# --- pure helpers -----------------------------------------------------------

def test_merge_rows_dedups_by_normalized_code(tmp_path):
    ing = _write_ingest(tmp_path, "shard.jsonl", [
        {"code": "int a(){return 0;}", "label": 0},
        {"code": "int a(){return 0;}   ", "label": 0},   # trailing ws -> dup
        {"code": "void b(){gets(0);}", "label": 1},
        {"code": "", "label": 1},                        # empty code dropped
    ])
    files = [os.path.join(ing, "shard.jsonl")]
    rows, src = mas.merge_rows(files)
    assert len(rows) == 2  # dup + empty removed
    assert src == {}       # filename "shard" declares no split


def test_merge_rows_preserves_declared_splits(tmp_path):
    _write_ingest(tmp_path, "bigvul_train.jsonl", [{"code": "int t(){return 1;}", "label": 0}])
    _write_ingest(tmp_path, "bigvul_test.jsonl", [{"code": "int q(){return 2;}", "label": 1}])
    ing = str(tmp_path / "ingest")
    files = sorted(os.path.join(ing, n) for n in os.listdir(ing))
    rows, src = mas.merge_rows(files)
    assert len(rows) == 2
    assert set(src.values()) == {"train", "test"}


def test_split_rows_preserves_when_all_declared(tmp_path):
    rows = [{"code": f"int f{i}(){{return {i};}}", "label": i % 2} for i in range(10)]
    src = {}
    for i, r in enumerate(rows):
        h = mas._code_hash(r["code"])
        src[h] = "train" if i < 8 else ("val" if i == 8 else "test")
    train, val, test, preserved = mas.split_rows(rows, src)
    assert preserved is True
    assert len(train) == 8 and len(val) == 1 and len(test) == 1


def test_split_rows_80_10_10_when_no_declared_splits():
    rows = [{"code": f"int f{i}(){{return {i};}}", "label": i % 2} for i in range(100)]
    train, val, test, preserved = mas.split_rows(rows, {})
    assert preserved is False
    assert len(train) == 80 and len(val) == 10 and len(test) == 10


# --- the hard-fail guard (via main()) ---------------------------------------

def test_real_mode_empty_ingest_hard_fails(tmp_path):
    ing = str(tmp_path / "ingest")
    os.makedirs(ing, exist_ok=True)
    rc = mas.main(["--ingest-dir", ing, "--data-dir", str(tmp_path / "data"),
                   "--min-train-rows", "100"])
    assert rc == 2
    # no bootstrap written
    assert not os.path.exists(str(tmp_path / "data" / "train.jsonl"))


def test_real_mode_too_small_train_hard_fails(tmp_path):
    ing = _write_ingest(tmp_path, "shard.jsonl",
                        [{"code": f"int f{i}(){{return {i};}}", "label": i % 2} for i in range(20)])
    rc = mas.main(["--ingest-dir", ing, "--data-dir", str(tmp_path / "data"),
                   "--min-train-rows", "1000"])
    assert rc == 3


def test_real_mode_enough_rows_succeeds(tmp_path):
    rows = [{"code": f"int fn{i}(){{return {i};}}", "label": i % 2} for i in range(200)]
    ing = _write_ingest(tmp_path, "shard.jsonl", rows)
    data = str(tmp_path / "data")
    rc = mas.main(["--ingest-dir", ing, "--data-dir", data, "--min-train-rows", "100"])
    assert rc == 0
    with open(os.path.join(data, "train.jsonl"), encoding="utf-8") as f:
        n_train = sum(1 for _ in f)
    assert n_train >= 100


def test_bootstrap_mode_empty_ingest_writes_bootstrap(tmp_path):
    ing = str(tmp_path / "ingest")
    os.makedirs(ing, exist_ok=True)
    data = str(tmp_path / "data")
    rc = mas.main(["--ingest-dir", ing, "--data-dir", data, "--allow-bootstrap"])
    assert rc == 0
    assert os.path.exists(os.path.join(data, "train.jsonl"))


def test_bootstrap_mode_does_not_enforce_min_rows(tmp_path):
    ing = _write_ingest(tmp_path, "shard.jsonl",
                        [{"code": f"int f{i}(){{return {i};}}", "label": i % 2} for i in range(20)])
    data = str(tmp_path / "data")
    rc = mas.main(["--ingest-dir", ing, "--data-dir", data,
                   "--allow-bootstrap", "--min-train-rows", "100000"])
    assert rc == 0  # guard is skipped in bootstrap mode


def test_default_min_train_rows_is_tens_of_thousands():
    assert mas.DEFAULT_MIN_TRAIN_ROWS >= 10000
