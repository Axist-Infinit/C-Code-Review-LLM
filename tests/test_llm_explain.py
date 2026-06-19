"""Unit tests for llm_explain prompt-injection hardening (torch-free).

llm_explain only imports explain_findings / profiles (no torch), so it is safe
to import here.
"""
from llm_explain import (
    apply_corroboration_floor,
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
