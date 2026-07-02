import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_PATH = os.path.join(os.path.dirname(_HERE), "scripts", "py",
                         "make_balanced_subset.py")
_spec = importlib.util.spec_from_file_location("make_balanced_subset", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
balanced_sample, main = _mod.balanced_sample, _mod.main


def _rows(n_pos, n_neg):
    rows = [{"code": f"vuln {i}", "label": 1} for i in range(n_pos)]
    rows += [{"code": f"clean {i}", "label": 0} for i in range(n_neg)]
    return rows


def test_balanced_and_deterministic():
    rows = _rows(50, 200)
    a = balanced_sample(rows, 40, seed=7)
    b = balanced_sample(rows, 40, seed=7)
    assert a == b
    assert sum(r["label"] for r in a) == 20 and len(a) == 40


def test_different_seed_different_sample():
    rows = _rows(50, 200)
    assert balanced_sample(rows, 40, seed=1) != balanced_sample(rows, 40, seed=2)


def test_insufficient_pool_raises():
    with pytest.raises(ValueError, match="not enough rows"):
        balanced_sample(_rows(5, 200), 40, seed=7)


def test_odd_rows_rejected():
    with pytest.raises(ValueError, match="even"):
        balanced_sample(_rows(50, 50), 41, seed=7)


def test_cli_roundtrip(tmp_path):
    inp = tmp_path / "in.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in _rows(30, 30)) + "\n")
    out = tmp_path / "out.jsonl"
    assert main([str(inp), str(out), "--rows", "20", "--seed", "3"]) == 0
    got = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(got) == 20 and sum(r["label"] for r in got) == 10


def test_cli_insufficient_exits_2(tmp_path):
    inp = tmp_path / "in.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in _rows(2, 30)) + "\n")
    assert main([str(inp), str(tmp_path / "out.jsonl"), "--rows", "20"]) == 2
