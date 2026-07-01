"""Unit tests for the SARIF 2.1.0 mapping in to_sarif.py (torch-free)."""
import json
import os

from to_sarif import build_sarif, finding_to_result, rule_metadata, SEVERITY_TO_LEVEL


def _region(result):
    return result["locations"][0]["physicalLocation"]["region"]


def _uri(result):
    return result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]


def test_severity_to_level_mapping():
    assert SEVERITY_TO_LEVEL["critical"] == "error"
    assert SEVERITY_TO_LEVEL["high"] == "error"
    assert SEVERITY_TO_LEVEL["medium"] == "warning"
    assert SEVERITY_TO_LEVEL["low"] == "note"
    assert SEVERITY_TO_LEVEL["info"] == "note"


def test_finding_basic_fields():
    e = {
        "file": "src/parser.c",
        "start_line": 10,
        "end_line": 20,
        "severity": "high",
        "cwe": "CWE-120",
        "issue": "strcpy into fixed buffer",
        "fix": "Use strlcpy.",
        "score": 0.93,
        "is_vulnerable": True,
        "backend": "ollama",
    }
    r = finding_to_result(e)
    assert r["ruleId"] == "CWE-120"
    assert r["level"] == "error"
    assert "strcpy into fixed buffer" in r["message"]["text"]
    assert "Fix: Use strlcpy." in r["message"]["text"]
    loc = r["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "src/parser.c"
    assert loc["region"]["startLine"] == 10
    assert loc["region"]["endLine"] == 20
    assert r["properties"]["classifierScore"] == 0.93
    assert r["properties"]["explainerBackend"] == "ollama"
    assert r["properties"]["llmJudgment"] is True


def test_unknown_severity_defaults_to_warning():
    r = finding_to_result({"file": "a.c", "start_line": 1})
    assert r["level"] == "warning"


def test_ruleid_falls_back_to_matched_pattern_then_classifier():
    r = finding_to_result({"file": "a.c", "matched_patterns": ["strcpy"]})
    assert r["ruleId"] == "strcpy"
    r2 = finding_to_result({"file": "a.c"})
    assert r2["ruleId"] == "ml-classifier"


def test_endline_defaults_to_startline():
    r = finding_to_result({"file": "a.c", "start_line": 7})
    region = r["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] == 7
    assert region["endLine"] == 7


def test_missing_start_line_defaults_to_one():
    r = finding_to_result({"file": "a.c"})
    region = r["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] == 1
    assert region["endLine"] == 1


def test_no_properties_when_none_present():
    r = finding_to_result({"file": "a.c", "start_line": 1})
    assert "properties" not in r


def test_tool_severity_words_map_to_sarif_levels():
    # words emitted by cppcheck/clang-tidy findings passed through the ensemble
    assert SEVERITY_TO_LEVEL["error"] == "error"
    assert SEVERITY_TO_LEVEL["warning"] == "warning"
    for word in ("style", "performance", "portability", "information", "informational"):
        assert SEVERITY_TO_LEVEL[word] == "note"
    r = finding_to_result({"file": "a.c", "start_line": 1, "severity": "error"})
    assert r["level"] == "error"
    r2 = finding_to_result({"file": "a.c", "start_line": 1, "severity": "style"})
    assert r2["level"] == "note"


def test_ruleid_uses_rule_key_between_cwe_and_patterns():
    # ensemble tool findings carry "rule", not "cwe"
    r = finding_to_result({"file": "a.c", "rule": "cert-err34-c", "start_line": 1})
    assert r["ruleId"] == "cert-err34-c"
    # cwe still wins over rule; rule wins over matched_patterns
    r2 = finding_to_result({"file": "a.c", "cwe": "CWE-120", "rule": "cert-err34-c",
                            "matched_patterns": ["strcpy"]})
    assert r2["ruleId"] == "CWE-120"
    r3 = finding_to_result({"file": "a.c", "rule": "cert-err34-c",
                            "matched_patterns": ["strcpy"]})
    assert r3["ruleId"] == "cert-err34-c"


def test_endline_clamped_to_startline():
    r = finding_to_result({"file": "a.c", "start_line": 10, "end_line": 5})
    region = _region(r)
    assert region["startLine"] == 10
    assert region["endLine"] == 10


def test_backslash_paths_normalized_on_posix():
    r = finding_to_result({"file": "src\\win\\a.c", "start_line": 1})
    assert _uri(r) == "src/win/a.c"


def test_missing_or_empty_file_uses_unknown_uri():
    assert _uri(finding_to_result({"start_line": 3})) == "unknown"
    assert _uri(finding_to_result({"file": "", "start_line": 3})) == "unknown"


def test_match_lines_tighten_region_within_function_span():
    r = finding_to_result({"file": "a.c", "start_line": 5, "end_line": 12,
                           "match_lines": [7, 9]})
    region = _region(r)
    assert region["startLine"] == 7
    assert region["endLine"] == 9


def test_match_lines_clamped_inside_function_span():
    r = finding_to_result({"file": "a.c", "start_line": 5, "end_line": 12,
                           "match_lines": [2, 20]})
    region = _region(r)
    assert region["startLine"] == 5
    assert region["endLine"] == 12


def test_start_line_zero_clamped_to_one():
    # SARIF 2.1.0 requires region lines >= 1; a 0 must not slip through
    # the falsy-value gate (docstring guarantees endLine >= startLine >= 1).
    r = finding_to_result({"file": "inc/a.h", "start_line": 0, "end_line": 0})
    region = _region(r)
    assert region["startLine"] == 1
    assert region["endLine"] == 1


def test_match_lines_zero_with_zero_span_clamped_to_one():
    # Contract C4 chain: surface_review parses "line": "0" into
    # {start_line: 0, end_line: 0, match_lines: [0]}; previously this
    # emitted the schema-invalid region {startLine: 0, endLine: 0}.
    r = finding_to_result({"file": "inc/a.h", "start_line": 0, "end_line": 0,
                           "match_lines": [0]})
    region = _region(r)
    assert region["startLine"] == 1
    assert region["endLine"] == 1


def test_match_lines_clamped_into_span_even_when_start_line_zero():
    # A falsy start_line (0) must not disable the span-containment clamp:
    # previously match_lines [99] escaped the 1..9 span as region 99..99.
    r = finding_to_result({"file": "a.c", "start_line": 0, "end_line": 9,
                           "match_lines": [99]})
    region = _region(r)
    assert region["startLine"] == 9
    assert region["endLine"] == 9


def test_match_lines_zero_alone_clamped_to_one():
    # No span at all: each match line is still clamped to >= 1.
    r = finding_to_result({"file": "a.c", "match_lines": [0]})
    region = _region(r)
    assert region["startLine"] == 1
    assert region["endLine"] == 1


def test_match_lines_partially_outside_span_clamped():
    # Only the out-of-span endpoint moves; the in-span one is kept.
    r = finding_to_result({"file": "a.c", "start_line": 5, "end_line": 12,
                           "match_lines": [3, 8]})
    region = _region(r)
    assert region["startLine"] == 5
    assert region["endLine"] == 8


def test_match_lines_used_alone_when_no_span():
    r = finding_to_result({"file": "a.c", "match_lines": [7]})
    region = _region(r)
    assert region["startLine"] == 7
    assert region["endLine"] == 7


def test_rule_metadata_enriches_cwe_rules():
    rule = rule_metadata("CWE-120", {"vulnerability": "Buffer overflow via strcpy",
                                     "issue": "strcpy into fixed buffer"})
    assert rule["id"] == "CWE-120"
    assert rule["helpUri"] == "https://cwe.mitre.org/data/definitions/120.html"
    assert rule["shortDescription"]["text"] == "Buffer overflow via strcpy"
    # non-CWE ids get no helpUri; missing text falls back to the id
    plain = rule_metadata("cert-err34-c", {})
    assert "helpUri" not in plain
    assert plain["shortDescription"]["text"] == "cert-err34-c"


def test_build_sarif_document_shape_and_rules():
    entries = [
        {"file": "a.c", "start_line": 1, "cwe": "CWE-120", "severity": "high",
         "issue": "strcpy into fixed buffer"},
        {"file": "b.c", "start_line": 2, "rule": "cert-err34-c", "severity": "warning",
         "issue": "atoi return unchecked"},
        {"file": "a.c", "start_line": 9, "cwe": "CWE-120", "severity": "high",
         "issue": "another strcpy"},
    ]
    sarif = build_sarif(entries)
    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert len(run["results"]) == 3
    rules = {r["id"]: r for r in run["tool"]["driver"]["rules"]}
    assert set(rules) == {"CWE-120", "cert-err34-c"}
    assert rules["CWE-120"]["helpUri"].endswith("/120.html")
    assert rules["CWE-120"]["shortDescription"]["text"] == "strcpy into fixed buffer"
    result_rule_ids = {r["ruleId"] for r in run["results"]}
    assert result_rule_ids == set(rules)


def test_fixture_findings_map_to_sarif():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "fixtures", "sample_findings.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data["explanations"]
    results = [finding_to_result(e) for e in entries]
    assert len(results) == 2
    assert results[0]["ruleId"] == "CWE-120"
    assert results[0]["level"] == "error"
    # second entry has empty cwe -> falls back to matched_patterns (empty) -> ml-classifier
    assert results[1]["ruleId"] == "ml-classifier"
    assert results[1]["level"] == "note"
