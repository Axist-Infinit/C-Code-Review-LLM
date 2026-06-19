"""Unit tests for the SARIF 2.1.0 mapping in to_sarif.py (torch-free)."""
import json
import os

from to_sarif import finding_to_result, SEVERITY_TO_LEVEL


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
