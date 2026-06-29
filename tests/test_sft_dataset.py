"""Tests for the SFT data builder and the trainer's pure (torch-free) helpers.

model/train_llm_sft.py imports torch only inside main(), so its formatting /
masking / validation helpers import cleanly here.
"""
import json

import pytest

from model.build_sft_dataset import (
    record_from_entry,
    build_records,
    split_records,
    load_entries,
    load_corroboration,
    count_corroboration,
)
from model.train_llm_sft import (
    format_prompt,
    format_completion,
    validate_records,
    mask_labels,
    tokenize_example,
    has_trainable_tokens,
    COMPLETION_HEADER,
)


# --- data builder -----------------------------------------------------------

def _entry(**over):
    base = {"file": "a.c", "start_line": 1, "end_line": 3, "score": 0.9,
            "is_vulnerable": True, "issue": "strcpy overflow", "cwe": "CWE-120",
            "severity": "high", "explanation": "unbounded copy", "fix": "use strlcpy",
            "snippet": ["void f(char*s){", "  char b[8]; strcpy(b,s);", "}"]}
    base.update(over)
    return base


def test_record_from_entry_joins_snippet_lines():
    rec = record_from_entry(_entry())
    assert rec["code"].startswith("void f(")
    assert "strcpy" in rec["code"]
    assert rec["analysis"]["cwe"] == "CWE-120"


def test_record_from_entry_rejects_empty_code():
    assert record_from_entry(_entry(snippet=[])) is None


def test_record_from_entry_rejects_contentless_analysis():
    assert record_from_entry(_entry(issue="", explanation="", cwe="")) is None


def test_build_records_only_vulnerable_filter():
    entries = [_entry(), _entry(is_vulnerable=False, issue="clean")]
    assert len(build_records(entries, only_vulnerable=True)) == 1
    assert len(build_records(entries, only_vulnerable=False)) == 2


def test_build_records_min_score_filter():
    entries = [_entry(score=0.95), _entry(score=0.4)]
    assert len(build_records(entries, min_score=0.5)) == 1


def test_split_records_is_deterministic_and_disjoint():
    recs = [{"code": f"c{i}", "analysis": {"issue": str(i)}} for i in range(10)]
    tr1, va1 = split_records(recs, val_frac=0.2, seed=7)
    tr2, va2 = split_records(recs, val_frac=0.2, seed=7)
    assert (tr1, va1) == (tr2, va2)          # deterministic
    assert len(va1) == 2 and len(tr1) == 8
    codes = {r["code"] for r in tr1} | {r["code"] for r in va1}
    assert codes == {r["code"] for r in recs}  # partition covers everything, no overlap


def test_split_single_record_keeps_train_nonempty():
    tr, va = split_records([{"code": "x", "analysis": {"issue": "y"}}], val_frac=0.5)
    assert len(tr) == 1 and len(va) == 0


def test_load_entries_accepts_both_shapes(tmp_path):
    p1 = tmp_path / "ex.json"
    p1.write_text(json.dumps({"explanations": [_entry()]}))
    p2 = tmp_path / "fi.json"
    p2.write_text(json.dumps({"findings": [_entry()]}))
    assert len(load_entries(str(p1))) == 1
    assert len(load_entries(str(p2))) == 1


# --- corroboration from ensemble output -------------------------------------

ENSEMBLE = {
    "tools": {"clang-tidy": 2, "flawfinder": 1, "ml-classifier": 1},
    "findings": [
        {"tool": "clang-tidy", "file": "src/a.c", "start_line": 12, "rule": "x"},
        {"tool": "flawfinder", "file": "src/a.c", "start_line": 13, "rule": "y"},
        {"tool": "clang-tidy", "file": "src/a.c", "start_line": 99, "rule": "z"},
        {"tool": "ml-classifier", "file": "src/a.c", "start_line": 12},  # must be ignored
    ],
}


def test_load_corroboration_ignores_ml_classifier(tmp_path):
    p = tmp_path / "ens.json"
    p.write_text(json.dumps(ENSEMBLE))
    corrob = load_corroboration(str(p))
    lines_tools = corrob["src/a.c"]
    assert (12, "clang-tidy") in lines_tools
    assert (13, "flawfinder") in lines_tools
    assert all(tool != "ml-classifier" for _ln, tool in lines_tools)


def test_count_corroboration_within_range_counts_distinct_tools():
    corrob = {"src/a.c": [(12, "clang-tidy"), (13, "flawfinder"), (99, "clang-tidy")]}
    entry = {"file": "src/a.c", "start_line": 10, "end_line": 20}
    n, tools = count_corroboration(entry, corrob)
    assert n == 2                                  # clang-tidy + flawfinder in [10,20]
    assert tools == ["clang-tidy", "flawfinder"]


def test_count_corroboration_out_of_range_is_zero():
    corrob = {"src/a.c": [(99, "clang-tidy")]}
    entry = {"file": "src/a.c", "start_line": 10, "end_line": 20}
    assert count_corroboration(entry, corrob) == (0, [])


def test_count_corroboration_matches_by_basename():
    corrob = {"/abs/build/a.c": [(15, "cppcheck")]}
    entry = {"file": "src/a.c", "start_line": 10, "end_line": 20}  # different dir, same base
    n, tools = count_corroboration(entry, corrob)
    assert n == 1 and tools == ["cppcheck"]


def test_count_corroboration_none_or_missing_fields():
    assert count_corroboration({"file": "a.c", "start_line": 1}, None) == (0, [])
    assert count_corroboration({"start_line": 1}, {"a.c": [(1, "t")]}) == (0, [])


def test_build_records_min_corroboration_filters_and_annotates():
    entries = [
        {**_entry(), "file": "src/a.c", "start_line": 10, "end_line": 20},   # corroborated
        {**_entry(), "file": "src/b.c", "start_line": 5, "end_line": 8},     # not corroborated
    ]
    corrob = {"src/a.c": [(12, "clang-tidy"), (13, "flawfinder")]}
    kept = build_records(entries, corroboration=corrob, min_corroboration=2)
    assert len(kept) == 1
    assert kept[0]["corroboration"] == 2
    assert kept[0]["corroborated_by"] == ["clang-tidy", "flawfinder"]


def test_build_records_corroboration_annotates_even_at_zero_bar():
    entries = [{**_entry(), "file": "src/b.c", "start_line": 5, "end_line": 8}]
    kept = build_records(entries, corroboration={"src/a.c": [(1, "t")]},
                         min_corroboration=0)
    assert len(kept) == 1
    assert kept[0]["corroboration"] == 0
    assert kept[0]["corroborated_by"] == []


# --- trainer pure helpers ---------------------------------------------------

def test_format_prompt_and_completion_roundtrip():
    prompt = format_prompt("int x;")
    assert prompt.endswith(COMPLETION_HEADER)
    comp = format_completion({"is_vulnerable": True, "issue": "i", "cwe": "CWE-1",
                              "severity": "high", "explanation": "e", "fix": "f"})
    parsed = json.loads(comp)
    assert parsed["cwe"] == "CWE-1" and parsed["is_vulnerable"] is True


def test_analysis_fields_in_sync_and_have_new_format():
    # The builder and trainer must agree on the schema exactly (no drift).
    from model.build_sft_dataset import ANALYSIS_FIELDS as BUILD_FIELDS
    from model.train_llm_sft import ANALYSIS_FIELDS as TRAIN_FIELDS
    assert BUILD_FIELDS == TRAIN_FIELDS
    for f in ("what_code_does", "what_could_go_wrong", "vulnerability"):
        assert f in TRAIN_FIELDS
    # reason-then-judge: the narrative fields precede the boolean verdict
    assert TRAIN_FIELDS.index("what_code_does") < TRAIN_FIELDS.index("is_vulnerable")


def test_validate_records_happy_path():
    recs = [{"code": "int x;", "analysis": {"issue": "i", "explanation": "e"}}]
    assert validate_records(recs) == 1


def test_validate_records_rejects_all_null_nine_field_analysis():
    # An analysis with every field null/empty serializes to a non-empty JSON of
    # nulls; it must still be rejected as having nothing to learn.
    null_analysis = {k: None for k in
                     ("what_code_does", "what_could_go_wrong", "vulnerability",
                      "is_vulnerable", "issue", "cwe", "severity", "explanation", "fix")}
    with pytest.raises(ValueError, match="no usable content"):
        validate_records([{"code": "int x;", "analysis": null_analysis}])


def test_validate_records_rejects_bool_only_analysis():
    # is_vulnerable alone (no text) is not a learnable target.
    with pytest.raises(ValueError, match="no usable content"):
        validate_records([{"code": "int x;", "analysis": {"is_vulnerable": True}}])


def test_validate_records_keeps_record_with_only_new_fields():
    rec = [{"code": "gets(b);", "analysis": {"what_could_go_wrong": "overflow",
                                             "vulnerability": "Stack buffer overflow"}}]
    assert validate_records(rec) == 1


@pytest.mark.parametrize("bad,msg", [
    ([], "empty"),
    ([{"code": "x"}], "must contain both"),
    ([{"code": "", "analysis": {"issue": "i"}}], "non-empty string"),
    ([{"code": "x", "analysis": ""}], "is empty"),
])
def test_validate_records_rejects_bad_input(bad, msg):
    with pytest.raises(ValueError, match=msg):
        validate_records(bad)


def test_mask_labels_masks_only_the_prompt():
    prompt_ids = [1, 2, 3]
    full_ids = [1, 2, 3, 4, 5]
    labels = mask_labels(prompt_ids, full_ids)
    assert labels == [-100, -100, -100, 4, 5]


class _FakeTok:
    """Whitespace tokenizer: ids are the tokens themselves (sufficient to check
    lengths and the prompt/completion mask boundary)."""
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


def test_tokenize_example_masks_prompt_and_keeps_completion():
    ex = tokenize_example(_FakeTok(), "int x;",
                          {"is_vulnerable": True, "issue": "i", "cwe": "",
                           "severity": "low", "explanation": "e", "fix": ""})
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    assert has_trainable_tokens(ex["labels"])           # completion is learnable
    # the leading prompt tokens are masked, later (completion) tokens are not
    assert ex["labels"][0] == -100
    assert any(l != -100 for l in ex["labels"])


def test_tokenize_example_all_masked_when_prompt_exceeds_max_length():
    ex = tokenize_example(_FakeTok(), "a b c d e f g h",
                          {"issue": "i", "explanation": "e"}, max_length=3)
    assert not has_trainable_tokens(ex["labels"])       # nothing to train on
