"""Validate the REAL C++ eval set (benchmarks/cpp_eval_real.jsonl).

Guards the PrimeVul-derived dataset that measures the BigVul-trained classifier
on real-world C++ functions: schema, label balance, size caps, intra-set
dedup, and that the committed provenance sidecar actually describes the
committed jsonl (row count + sha256). Pure stdlib — no torch / datasets /
network; the builder module itself must import without `datasets` installed
(its heavy imports are lazy, same pattern as fetch_bigvul_hf.py).
"""
import hashlib
import importlib.util
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "benchmarks", "cpp_eval_real.jsonl")
PROVENANCE = os.path.join(REPO, "benchmarks", "cpp_eval_real.provenance.json")
BUILDER = os.path.join(REPO, "scripts", "py", "build_cpp_eval_real.py")

MAX_ROWS = 1200
MAX_BYTES = 4 * 1024 * 1024  # the "~4MB" cap (builder targets 4_000_000)


def _load_builder():
    # Import-safety is part of the contract: the module must load in the
    # torch-free CI env where `datasets` is not installed.
    pre = "datasets" in sys.modules
    spec = importlib.util.spec_from_file_location("build_cpp_eval_real", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not pre:
        assert "datasets" not in sys.modules, \
            "builder must lazy-import datasets (heavy deps at module scope)"
    return mod


builder = _load_builder()


def _load():
    rows = []
    with open(DATASET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_builder_module_is_import_safe():
    assert callable(builder.main)
    assert callable(builder.fingerprint)


def test_dataset_exists_and_is_valid_jsonl():
    rows = _load()
    assert len(rows) >= 300, "real C++ eval set is suspiciously small"
    assert len(rows) <= MAX_ROWS


def test_every_row_has_valid_schema():
    for r in _load():
        assert r["label"] in (0, 1), f"{r.get('name')}: bad label {r.get('label')!r}"
        assert r["lang"] == "cpp", f"{r['name']}: lang {r['lang']!r}"
        assert r["code"].strip(), f"{r['name']}: empty code"
        assert r["category"], f"{r['name']}: missing category"
        assert r["cwe"] == "" or re.fullmatch(r"CWE-\d+", r["cwe"]), \
            f"{r['name']}: bad cwe {r['cwe']!r}"
        assert r["category"] == (r["cwe"] or "real"), \
            f"{r['name']}: category {r['category']!r} not cwe-derived"
        assert re.fullmatch(r".+@[0-9a-f]{40}", r["source"]), \
            f"{r['name']}: source {r['source']!r} is not <dataset>@<revision>"
        assert r["name"]
        if "cve" in r:
            assert r["cve"].startswith("CVE-"), f"{r['name']}: bad cve {r['cve']!r}"


def test_names_are_unique():
    names = [r["name"] for r in _load()]
    assert len(names) == len(set(names))


def test_labels_are_roughly_balanced():
    rows = _load()
    share = sum(r["label"] for r in rows) / len(rows)
    assert 0.35 <= share <= 0.65, f"vulnerable share {share:.2f} outside 35-65%"


def test_no_duplicate_normalized_functions():
    # Same normalization the builder dedupes/leak-checks with: a duplicate here
    # means the eval double-counts a function (or the dedup regressed).
    hashes = [builder.fingerprint(r["code"]) for r in _load()]
    assert len(hashes) == len(set(hashes))


def test_size_caps_respected():
    assert os.path.getsize(DATASET) <= MAX_BYTES
    for r in _load():
        assert len(r["code"]) <= builder.DEFAULT_MAX_FUNC_CHARS, \
            f"{r['name']}: function exceeds the per-row size cap"


def test_provenance_matches_committed_dataset():
    with open(PROVENANCE, encoding="utf-8") as fh:
        prov = json.load(fh)
    rows = _load()
    assert prov["rows"] == len(rows)
    assert prov["n_vuln"] == sum(r["label"] for r in rows)
    assert prov["n_clean"] == len(rows) - prov["n_vuln"]
    h = hashlib.sha256()
    with open(DATASET, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    assert prov["sha256"] == h.hexdigest(), \
        "provenance sha256 stale — rebuild or re-stamp after editing the jsonl"
    assert prov["dataset"] == builder.PRIMEVUL_ID
    assert prov["revision"] == builder.PRIMEVUL_REVISION
    assert "MIT" in prov["license"]
    assert prov["citation"].strip()
    # rows must declare the same pinned source the provenance does
    src = f"{prov['dataset']}@{prov['revision']}"
    assert all(r["source"] == src for r in rows)


def test_leak_check_was_enabled_for_committed_artifact():
    # The committed eval set must never be built with --no-leak-check: it is
    # specifically meant to be disjoint from the BigVul training corpus.
    with open(PROVENANCE, encoding="utf-8") as fh:
        prov = json.load(fh)
    assert prov["leak_check"]["enabled"] is True
    assert prov["leak_check"]["against"].startswith("bstee615/bigvul@")
