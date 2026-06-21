"""Validate the curated C++ eval set (benchmarks/cpp_eval.jsonl).

Guards the dataset that measures the C-trained classifier on C++: schema, label
balance, and — crucially — that every case extracts as exactly one function with
the cpp grammar, so eval granularity matches what the scanner actually scores.
"""
import json
import os
import re

import pytest

from local_vuln_scanner import extract_functions, treesitter_available

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "benchmarks", "cpp_eval.jsonl")


def _load():
    rows = []
    with open(DATASET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_dataset_exists_and_is_balanced():
    rows = _load()
    assert len(rows) >= 12
    pos = sum(r["label"] for r in rows)
    neg = len(rows) - pos
    assert pos == neg, f"expected a balanced set, got {pos} vuln / {neg} clean"


def test_every_row_has_valid_schema():
    for r in _load():
        assert r["label"] in (0, 1)
        assert r["lang"] == "cpp"
        assert r["code"].strip(), f"{r['name']}: empty code"
        assert r["category"], f"{r['name']}: missing category"
        assert re.fullmatch(r"CWE-\d+", r["cwe"]), f"{r['name']}: bad cwe {r['cwe']!r}"
        assert r["name"]


def test_names_are_unique():
    names = [r["name"] for r in _load()]
    assert len(names) == len(set(names))


def test_each_category_has_a_vuln_and_clean_pair():
    rows = _load()
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], set()).add(r["label"])
    for cat, labels in by_cat.items():
        assert labels == {0, 1}, f"category {cat} is not a vuln/clean pair"


@pytest.mark.skipif(not treesitter_available("cpp"), reason="cpp grammar not installed")
def test_every_case_extracts_as_one_function():
    # If a case isn't a single function, the classifier would score the wrong
    # unit and the eval would be meaningless.
    for r in _load():
        res = extract_functions(r["code"], lang="cpp")
        n = 0 if res is None else len(res)
        assert n == 1, f"{r['name']} extracted {n} functions, expected 1"
