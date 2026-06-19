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


# --- trainer pure helpers ---------------------------------------------------

def test_format_prompt_and_completion_roundtrip():
    prompt = format_prompt("int x;")
    assert prompt.endswith(COMPLETION_HEADER)
    comp = format_completion({"is_vulnerable": True, "issue": "i", "cwe": "CWE-1",
                              "severity": "high", "explanation": "e", "fix": "f"})
    parsed = json.loads(comp)
    assert parsed["cwe"] == "CWE-1" and parsed["is_vulnerable"] is True


def test_validate_records_happy_path():
    recs = [{"code": "int x;", "analysis": {"issue": "i", "explanation": "e"}}]
    assert validate_records(recs) == 1


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
