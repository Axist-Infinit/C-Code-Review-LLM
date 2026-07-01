"""Unit tests for llm_explain prompt-injection hardening (torch-free).

llm_explain only imports explain_findings / profiles (no torch), so it is safe
to import here.
"""
import io
import json

import pytest

from llm_explain import (
    apply_corroboration_floor,
    explain_finding,
    language_hint,
    normalize_cwe,
    normalize_severity,
    ollama_chat,
    parse_chat_response,
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


# --- ollama_chat response-shape hardening ------------------------------------

def _chat_body(content):
    return {"message": {"content": content}}


def test_parse_chat_response_rejects_list_body():
    with pytest.raises(ValueError):
        parse_chat_response(["not", "an", "object"])


def test_parse_chat_response_rejects_null_message():
    with pytest.raises(ValueError):
        parse_chat_response({"message": None})


def test_parse_chat_response_rejects_null_content():
    with pytest.raises(ValueError):
        parse_chat_response({"message": {"content": None}})


def test_parse_chat_response_accepts_valid_body():
    out = parse_chat_response(_chat_body('{"is_vulnerable": true}'))
    assert out == {"is_vulnerable": True}


class _FakeHTTPResponse(io.BytesIO):
    """Minimal stand-in for the urlopen response context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("body", [
    ["a", "list"],
    {"message": None},
    {"message": {"content": None}},
])
def test_malformed_chat_body_falls_back_per_finding(monkeypatch, body):
    # End-to-end through the REAL ollama_chat: a malformed /api/chat body must
    # raise ValueError (caught -> heuristic fallback), never AttributeError/
    # TypeError which would abort the whole run.
    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "gets(buf);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="http://x",
                        num_ctx=2048)  # default chat_fn = ollama_chat
    assert e.get("llm_error")
    assert e["backend"] == "heuristic"
    assert e["is_vulnerable"] is True  # heuristic still caught gets()


# --- allow-list validation of LLM severity / cwe ------------------------------

def test_normalize_severity_accepts_allow_list_with_case_and_space():
    assert normalize_severity(" High ") == "high"
    assert normalize_severity("CRITICAL") == "critical"
    assert normalize_severity("info") == "info"


def test_normalize_severity_rejects_garbage():
    assert normalize_severity("URGENT!!") is None
    assert normalize_severity("") is None
    assert normalize_severity(None) is None


def test_normalize_cwe_canonicalizes_common_forms():
    for raw in ("CWE-120", "cwe-120", "CWE 120", "cwe_120", "120", " CWE-120 "):
        assert normalize_cwe(raw) == "CWE-120"


def test_normalize_cwe_rejects_garbage():
    assert normalize_cwe("buffer overflow") is None
    assert normalize_cwe("CWE-123456") is None  # > 5 digits
    assert normalize_cwe("") is None
    assert normalize_cwe(None) is None


def test_normalize_cwe_strips_leading_zeros():
    # Zero-padded ids are common LLM output; they must canonicalize to the
    # same ruleId the heuristic lane emits, never pass through verbatim.
    assert normalize_cwe("CWE-078") == "CWE-78"
    assert normalize_cwe("cwe 078") == "CWE-78"
    assert normalize_cwe("cwe_007") == "CWE-7"
    assert normalize_cwe("00120") == "CWE-120"


def test_normalize_cwe_rejects_zero():
    # CWE 0 does not exist; it must be rejected like other invalid forms so
    # the caller falls back to the heuristic cwe.
    assert normalize_cwe("CWE-0") is None
    assert normalize_cwe("CWE-000") is None
    assert normalize_cwe("0") is None


def test_normalize_cwe_valid_id_untouched():
    assert normalize_cwe("CWE-120") == "CWE-120"


def test_llm_zero_padded_cwe_canonicalized_no_fallback():
    # Confirmed failure: LLM replies "CWE-078" for a system() finding; the
    # entry must carry the canonical "CWE-78" (same ruleId as the heuristic
    # lane) with no fallback marker — this is normalization, not fallback.
    def padded_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": True, "issue": "command injection",
                "cwe": "CWE-078", "severity": "high", "explanation": "x",
                "fix": "y"}
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "system(cmd);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=padded_chat)
    assert e["backend"] == "ollama"
    assert e["cwe"] == "CWE-78"
    assert "llm_field_fallbacks" not in e


def test_llm_cwe_zero_falls_back_to_heuristic_with_marker():
    # Confirmed failure: nonexistent "CWE-0" used to pass the allow-list with
    # no observability marker. It must now fall back to the heuristic cwe and
    # record the substitution.
    def zero_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": True, "issue": "command injection",
                "cwe": "CWE-0", "severity": "high", "explanation": "x",
                "fix": "y"}
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "system(cmd);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=zero_chat)
    assert e["backend"] == "ollama"
    assert e["cwe"] == "CWE-78"  # heuristic cwe for system()
    assert "cwe" in e["llm_field_fallbacks"]


def test_llm_garbage_severity_and_cwe_fall_back_to_heuristic():
    def garbage_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": True, "issue": "overflow", "cwe": "not-a-cwe",
                "severity": "URGENT!!", "explanation": "x", "fix": "y"}
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "strcpy(d, s);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=garbage_chat)
    assert e["backend"] == "ollama"           # not a full fallback
    assert e["severity"] == "high"            # heuristic severity for strcpy
    assert e["cwe"] == "CWE-120"              # heuristic cwe for strcpy
    assert set(e["llm_field_fallbacks"]) == {"severity", "cwe"}


def test_llm_valid_fields_normalized_without_fallback():
    def sloppy_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": True, "issue": "overflow", "cwe": "cwe 120",
                "severity": " High ", "explanation": "x", "fix": "y"}
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "strcpy(d, s);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=sloppy_chat)
    assert e["severity"] == "high"
    assert e["cwe"] == "CWE-120"
    assert "llm_field_fallbacks" not in e


def test_llm_garbage_fields_without_pattern_match_degrade_safely():
    def garbage_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": False, "issue": "none", "cwe": "???",
                "severity": "whatever", "explanation": "", "fix": ""}
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.5,
         "snippet": "int add(int a, int b) { return a + b; }"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=garbage_chat)
    assert e["severity"] == "info"   # nothing matched -> info
    assert e["cwe"] == ""            # nothing matched -> no cwe
    assert set(e["llm_field_fallbacks"]) == {"severity", "cwe"}


def test_corroboration_floor_still_applies_after_validation():
    # Injected LLM hands back invalid vocabulary AND a downgrade; validation
    # must not weaken the floor.
    def lying_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": False, "issue": "none", "cwe": "n/a",
                "severity": "nothing-burger", "explanation": "fine", "fix": ""}
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.95,
         "snippet": "gets(buf);"}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=lying_chat)
    assert e["is_vulnerable"] is True
    assert e["cwe"] == "CWE-242"  # gets() heuristic cwe
    # gets() is critical in the heuristic table, well above the medium floor.
    assert e["severity"] == "critical"


# --- C3: match_lines pass-through ---------------------------------------------

def test_match_lines_passed_through_heuristic_backend():
    f = {"file": "a.c", "start_line": 10, "end_line": 20, "score": 0.9,
         "snippet": "strcpy(d, s);", "match_lines": [12, 17]}
    e = explain_finding(f, backend="heuristic", model=None, ollama_url=None, num_ctx=0)
    assert e["match_lines"] == [12, 17]


def test_match_lines_passed_through_ollama_backend():
    def fake_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": True, "issue": "x", "cwe": "CWE-120",
                "severity": "high", "explanation": "", "fix": ""}
    f = {"file": "a.c", "start_line": 10, "end_line": 20, "score": 0.9,
         "snippet": "strcpy(d, s);", "match_lines": [12, 17]}
    e = explain_finding(f, backend="ollama", model="m", ollama_url="x",
                        num_ctx=2048, chat_fn=fake_chat)
    assert e["match_lines"] == [12, 17]


def test_match_lines_absent_when_not_on_input():
    f = {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "strcpy(d, s);"}
    e = explain_finding(f, backend="heuristic", model=None, ollama_url=None, num_ctx=0)
    assert "match_lines" not in e


# --- C++ awareness -------------------------------------------------------------

def test_system_prompt_mentions_cpp_specific_classes():
    sp = SYSTEM_PROMPT.lower()
    for phrase in ("use-after-move", "iterator", "string_view", "c_str",
                   "delete[]", "exception-safety"):
        assert phrase in sp, phrase


def test_language_hint_from_extension():
    assert language_hint("src/a.cpp") == "C++"
    assert language_hint("b.cc") == "C++"
    assert language_hint("include/x.hpp") == "C++"
    assert language_hint("plain.c") == "C"
    assert language_hint("README.txt") == "C"
    assert language_hint(None) == "C"


def test_user_message_carries_language_hint(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(json.dumps(_chat_body("{}")).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ollama_chat("http://x", "m", "int f();", "widget.cpp", "1-1", 2048)
    user = captured["payload"]["messages"][1]["content"]
    assert "Language: C++" in user
    assert SNIPPET_BEGIN in user and SNIPPET_END in user

    ollama_chat("http://x", "m", "int f();", "widget.c", "1-1", 2048)
    user = captured["payload"]["messages"][1]["content"]
    assert "Language: C" in user
    assert "Language: C++" not in user
