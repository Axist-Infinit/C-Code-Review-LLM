"""Unit tests for llm_explain prompt-injection hardening (torch-free).

llm_explain only imports explain_findings / profiles (no torch), so it is safe
to import here.
"""
from llm_explain import (
    apply_corroboration_floor,
    explain_finding,
    SNIPPET_BEGIN,
    SNIPPET_END,
    SYSTEM_PROMPT,
    CORROBORATION_SCORE_FLOOR,
)


def test_system_prompt_has_untrusted_data_instruction():
    sp = SYSTEM_PROMPT.lower()
    assert "untrusted" in sp
    assert "ignore" in sp
    assert SNIPPET_BEGIN in SYSTEM_PROMPT
    assert SNIPPET_END in SYSTEM_PROMPT


def test_system_prompt_requests_three_part_format():
    # The reviewer must be asked for the structured narrative fields.
    for key in ("what_code_does", "what_could_go_wrong", "vulnerability"):
        assert key in SYSTEM_PROMPT


def test_explain_finding_ollama_carries_structured_fields():
    # A fake Ollama returning the structured keys flows straight through.
    def fake_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": True, "issue": "overflow", "cwe": "CWE-120",
                "severity": "high", "what_code_does": "copies a string",
                "what_could_go_wrong": "can overflow the buffer",
                "vulnerability": "Stack buffer overflow",
                "explanation": "unbounded copy", "fix": "use strlcpy"}
    f = {"file": "a.c", "start_line": 1, "end_line": 2, "score": 0.9,
         "snippet": "strcpy(d, s);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=fake_chat)
    assert e["what_code_does"] == "copies a string"
    assert e["what_could_go_wrong"] == "can overflow the buffer"
    assert e["vulnerability"] == "Stack buffer overflow"


def test_explain_finding_survives_non_dict_llm_response():
    # A syntactically valid but non-object JSON reply must fall back to the
    # heuristic for this finding instead of throwing and aborting the run.
    def list_chat(url, model, snippet, file, lines, num_ctx):
        return ["not", "an", "object"]
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "gets(buf);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=list_chat)
    assert e.get("llm_error")
    assert e["backend"] == "heuristic"
    assert "overflow" in e["vulnerability"].lower()


def test_explain_finding_handles_null_snippet():
    # snippet key present but null must not crash (.splitlines on None).
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.5, "snippet": None}
    e = explain_finding(f, backend="heuristic", model=None, ollama_url=None, num_ctx=0)
    assert e["snippet"] == []


def test_explain_finding_backfills_structured_fields_when_model_omits_them():
    # An older/terse model that omits the new keys still yields them, backfilled
    # from the regex heuristic on the snippet.
    def terse_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": True, "issue": "bad", "cwe": "CWE-120",
                "severity": "high", "explanation": "", "fix": ""}
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "gets(buf);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=terse_chat)
    assert "gets" in e["what_code_does"].lower()
    assert e["what_could_go_wrong"]
    assert "overflow" in e["vulnerability"].lower()


def test_delimiters_are_distinct_and_nonempty():
    assert SNIPPET_BEGIN and SNIPPET_END and SNIPPET_BEGIN != SNIPPET_END


def test_corroborated_finding_cannot_be_downgraded_to_not_vulnerable():
    # classifier high + heuristic matched strcpy, but LLM says safe (injection).
    entry = {"is_vulnerable": False, "severity": "info", "explanation": "looks fine"}
    out = apply_corroboration_floor(entry, score=0.95, matched={"strcpy"})
    assert out["is_vulnerable"] is True
    assert out["severity"] == "medium"
    assert out["downgrade_blocked"] is True
    assert "cross-check" in out["explanation"].lower()


def test_corroborated_finding_severity_floor_to_medium():
    entry = {"is_vulnerable": True, "severity": "low", "explanation": ""}
    out = apply_corroboration_floor(entry, score=0.90, matched={"gets"})
    assert out["severity"] == "medium"
    assert out["downgrade_blocked"] is True


def test_high_severity_is_preserved():
    entry = {"is_vulnerable": True, "severity": "critical", "explanation": "rce"}
    out = apply_corroboration_floor(entry, score=0.99, matched={"system"})
    # critical outranks medium -> untouched, no override note.
    assert out["severity"] == "critical"
    assert "downgrade_blocked" not in out


def test_no_floor_when_no_pattern_matched():
    # heuristic found nothing -> LLM judgment is authoritative, even if score high.
    entry = {"is_vulnerable": False, "severity": "info", "explanation": "clean"}
    out = apply_corroboration_floor(entry, score=0.99, matched=set())
    assert out["is_vulnerable"] is False
    assert out["severity"] == "info"
    assert "downgrade_blocked" not in out


def test_no_floor_when_score_below_floor():
    entry = {"is_vulnerable": False, "severity": "info", "explanation": "clean"}
    out = apply_corroboration_floor(entry, score=CORROBORATION_SCORE_FLOOR - 0.01,
                                    matched={"strcpy"})
    assert out["is_vulnerable"] is False
    assert "downgrade_blocked" not in out


def test_none_score_does_not_crash():
    entry = {"is_vulnerable": False, "severity": "info", "explanation": ""}
    out = apply_corroboration_floor(entry, score=None, matched={"strcpy"})
    assert out["is_vulnerable"] is False
