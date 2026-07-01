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
_FETCH_PATH = os.path.join(os.path.dirname(_HERE), "scripts", "py", "fetch_bigvul_hf.py")


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mas = _load_by_path("merge_and_split", _MOD_PATH)
# import-safe without `datasets` (load_dataset is imported lazily in main)
fetch = _load_by_path("fetch_bigvul_hf", _FETCH_PATH)


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
    data = str(tmp_path / "data")
    rc = mas.main(["--ingest-dir", ing, "--data-dir", data,
                   "--min-train-rows", "1000"])
    assert rc == 3
    # the guard runs before write_splits: a rejected run leaves no split files
    for split in ("train", "val", "test"):
        assert not os.path.exists(os.path.join(data, f"{split}.jsonl"))


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


# --- additive tag passthrough (source/lang/cwe survive the merge) ------------

def test_merge_rows_preserves_tags(tmp_path):
    ing = _write_ingest(tmp_path, "shard.jsonl", [
        {"code": "void g(){gets(0);}", "label": 1, "cwe": "CWE-787",
         "source": "func_before", "lang": "c"},
        {"code": "int ok(){return 0;}", "label": 0},
    ])
    rows, _ = mas.merge_rows([os.path.join(ing, "shard.jsonl")])
    tagged = next(r for r in rows if r["label"] == 1)
    assert tagged["source"] == "func_before"
    assert tagged["lang"] == "c"
    assert tagged["cwe"] == "CWE-787"
    plain = next(r for r in rows if r["label"] == 0)
    assert "source" not in plain and "lang" not in plain  # keys never invented


# --- CCR_REQUIRE_REAL_DATA env guard ------------------------------------------

def test_real_data_guard_pure_helper():
    assert mas.real_data_guard(5, environ={}) is None                # env off
    msg = mas.real_data_guard(5, environ={"CCR_REQUIRE_REAL_DATA": "1"})
    assert msg and "5" in msg and "1000" in msg                      # default floor
    assert mas.real_data_guard(1500, environ={"CCR_REQUIRE_REAL_DATA": "1"}) is None
    assert mas.real_data_guard(
        1500, environ={"CCR_REQUIRE_REAL_DATA": "1", "CCR_MIN_TRAIN_ROWS": "2000"})
    # unparsable floor falls back to the 1000 default
    assert mas.real_data_guard(
        1500, environ={"CCR_REQUIRE_REAL_DATA": "1", "CCR_MIN_TRAIN_ROWS": "junk"}) is None


def test_env_guard_blocks_small_train_even_in_bootstrap_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("CCR_REQUIRE_REAL_DATA", "1")
    ing = _write_ingest(tmp_path, "shard.jsonl",
                        [{"code": f"int f{i}(){{return {i};}}", "label": i % 2}
                         for i in range(20)])
    rc = mas.main(["--ingest-dir", ing, "--data-dir", str(tmp_path / "data"),
                   "--allow-bootstrap", "--min-train-rows", "1"])
    assert rc == 4  # 16 train rows < the 1000-row real-data floor


def test_env_guard_rc4_leaves_no_split_files(tmp_path, monkeypatch):
    """Regression (finding 17): the exact confirmed scenario —
    CCR_REQUIRE_REAL_DATA=1, a 20-row ingest shard, --allow-bootstrap and
    --min-train-rows 1 — must exit 4 WITHOUT leaving train/val/test.jsonl
    behind, or a later train.sh silently trains on the rejected splits."""
    monkeypatch.setenv("CCR_REQUIRE_REAL_DATA", "1")
    ing = _write_ingest(tmp_path, "shard.jsonl",
                        [{"code": f"int f{i}(){{return {i};}}", "label": i % 2}
                         for i in range(20)])
    data = str(tmp_path / "data")
    rc = mas.main(["--ingest-dir", ing, "--data-dir", data,
                   "--allow-bootstrap", "--min-train-rows", "1"])
    assert rc == 4
    for split in ("train", "val", "test"):
        assert not os.path.exists(os.path.join(data, f"{split}.jsonl"))


def test_env_guard_blocks_bootstrap_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CCR_REQUIRE_REAL_DATA", "1")
    ing = str(tmp_path / "ingest")
    os.makedirs(ing, exist_ok=True)
    data = str(tmp_path / "data")
    rc = mas.main(["--ingest-dir", ing, "--data-dir", data, "--allow-bootstrap"])
    assert rc == 4
    assert not os.path.exists(os.path.join(data, "train.jsonl"))  # nothing written


def test_env_guard_passes_with_enough_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("CCR_REQUIRE_REAL_DATA", "1")
    monkeypatch.setenv("CCR_MIN_TRAIN_ROWS", "100")
    rows = [{"code": f"int fn{i}(){{return {i};}}", "label": i % 2} for i in range(200)]
    ing = _write_ingest(tmp_path, "shard.jsonl", rows)
    rc = mas.main(["--ingest-dir", ing, "--data-dir", str(tmp_path / "data"),
                   "--min-train-rows", "100"])
    assert rc == 0


def test_env_unset_keeps_default_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("CCR_REQUIRE_REAL_DATA", raising=False)
    ing = str(tmp_path / "ingest")
    os.makedirs(ing, exist_ok=True)
    data = str(tmp_path / "data")
    rc = mas.main(["--ingest-dir", ing, "--data-dir", data, "--allow-bootstrap"])
    assert rc == 0
    assert os.path.exists(os.path.join(data, "train.jsonl"))


# --- fetch_bigvul_hf row shaping (pure, no `datasets` needed) -----------------

def test_rows_from_record_tags_source_and_lang():
    row = {"func_before": "void f(){gets(0);}", "func_after": "void f(){fgets(0,0,0);}",
           "vul": "1", "CWE ID": "CWE-787"}
    out = fetch.rows_from_record(row)
    assert len(out) == 2
    main_row, twin = out
    assert main_row == {"code": "void f(){gets(0);}", "label": 1, "cwe": "CWE-787",
                        "source": "func_before", "lang": "c"}
    assert twin == {"code": "void f(){fgets(0,0,0);}", "label": 0, "cwe": "",
                    "source": "func_after", "lang": "c"}


def test_rows_from_record_clean_row_has_no_twin():
    row = {"func_before": "int ok(){return 0;}", "func_after": "int ok(){return 0;} ",
           "vul": "0"}
    out = fetch.rows_from_record(row)
    assert len(out) == 1
    assert out[0]["label"] == 0 and out[0]["source"] == "func_before"


def test_rows_from_record_skips_empty_code_and_identical_after():
    assert fetch.rows_from_record({"vul": "1"}) == []
    same = "void f(){gets(0);}"
    out = fetch.rows_from_record({"func_before": same, "func_after": same, "vul": "1"})
    assert len(out) == 1  # unchanged func_after is not a hard negative


def test_rows_from_record_uses_dataset_lang_when_present():
    row = {"func_before": "int x;", "vul": "0", "lang": "C++"}
    assert fetch.rows_from_record(row)[0]["lang"] == "cpp"


def test_norm_lang_mapping():
    assert fetch.norm_lang(None) == "c"
    assert fetch.norm_lang("") == "c"
    assert fetch.norm_lang("C") == "c"
    assert fetch.norm_lang("C++") == "cpp"
    assert fetch.norm_lang("CPP") == "cpp"
    assert fetch.norm_lang("Rust") == "rust"


def test_fetch_revision_env_and_flag(monkeypatch):
    monkeypatch.delenv("BIGVUL_REVISION", raising=False)
    assert fetch.parse_args([]).revision is None
    monkeypatch.setenv("BIGVUL_REVISION", "v1.2")
    assert fetch.parse_args([]).revision == "v1.2"
    assert fetch.parse_args(["--revision", "abc123"]).revision == "abc123"


def test_fetch_file_sha256(tmp_path):
    import hashlib
    p = tmp_path / "shard.jsonl"
    p.write_bytes(b'{"code": "x"}\n' * 1000)
    assert fetch.file_sha256(str(p)) == hashlib.sha256(p.read_bytes()).hexdigest()
