"""Tests for the heuristic pattern set, including the C++ additions and the
CWE wiring that now feeds SARIF rule ids.
"""
import to_sarif
from explain_findings import explain_snippet, cwe_for_patterns
from llm_explain import heuristic_entry


def _matched(snippet):
    _expl, _recs, ids = explain_snippet(snippet)
    return ids


def test_c_patterns_still_match():
    assert "strcpy" in _matched("strcpy(d, s);")
    assert "gets" in _matched("gets(buf);")
    assert "memcpy" in _matched("memcpy(d, s, n);")


def test_cpp_command_injection_patterns():
    assert "system" in _matched('system(user_cmd);')
    assert "popen" in _matched('FILE *p = popen(cmd, "r");')


def test_cpp_cast_patterns():
    assert "reinterpret_cast" in _matched("auto *p = reinterpret_cast<Foo*>(raw);")
    assert "const_cast" in _matched("auto *q = const_cast<Foo*>(cp);")


def test_cpp_memory_and_alloc_patterns():
    assert "memmove" in _matched("memmove(d, s, n);")
    assert "alloca" in _matched("char *p = (char*)alloca(n);")


def test_format_string_pattern_is_high_precision():
    # bare identifier as the format string -> flagged
    assert "printf_format_string" in _matched("printf(user_input);")
    # a literal format string is fine -> not flagged
    assert "printf_format_string" not in _matched('printf("%s", user_input);')


def test_std_qualified_call_still_matches_via_word_boundary():
    # std::strcpy / ::strcpy still trip the strcpy pattern (\b matches at ':').
    assert "strcpy" in _matched("std::strcpy(d, s);")


def test_cwe_for_patterns_priority_and_mapping():
    assert cwe_for_patterns({"strcpy"}) == "CWE-120"
    assert cwe_for_patterns({"system"}) == "CWE-78"
    assert cwe_for_patterns({"reinterpret_cast"}) == "CWE-704"
    assert cwe_for_patterns({"alloca"}) == "CWE-770"
    assert cwe_for_patterns({"printf_format_string"}) == "CWE-134"
    assert cwe_for_patterns(set()) == ""
    # when several match, the higher-priority (earlier) pattern's CWE wins
    assert cwe_for_patterns({"strcpy", "system"}) == "CWE-78"  # system precedes strcpy


def test_heuristic_entry_now_carries_cwe():
    entry = heuristic_entry({"snippet": "strcpy(d, s);", "score": 0.9})
    assert entry["backend"] == "heuristic"
    assert entry["cwe"] == "CWE-120"


def test_heuristic_cwe_flows_into_sarif_rule_id():
    entry = heuristic_entry({"snippet": "system(cmd);", "score": 0.8,
                             "file": "x.cpp", "start_line": 1, "end_line": 1})
    entry.update({"file": "x.cpp", "start_line": 1, "end_line": 1})
    result = to_sarif.finding_to_result(entry)
    assert result["ruleId"] == "CWE-78"          # CWE drives the SARIF rule id
