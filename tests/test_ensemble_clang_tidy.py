"""Tests for the clang-tidy integration in ensemble_scan.

parse_clang_tidy_output is pure, so it is tested directly; run_clang_tidy is
checked only for graceful no-op when the binary is absent (no clang-tidy needed
in CI).
"""
import ensemble_scan
from ensemble_scan import parse_clang_tidy_output, run_clang_tidy

SAMPLE = """\
/src/a.cpp:12:5: warning: function 'strcpy' is insecure; use 'strncpy' [clang-analyzer-security.insecureAPI.strcpy]
    std::strcpy(name, user);
    ^
/src/a.cpp:20:9: error: use of undeclared identifier 'foo' [clang-diagnostic-error]
/src/a.cpp:5:1: note: expanded from macro 'X'
random line that is not a diagnostic
/src/b.cpp:3:1: warning: redundant cast [bugprone-redundant-cast]
"""


def test_parses_warnings_and_errors_skips_notes_and_noise():
    findings = parse_clang_tidy_output(SAMPLE)
    assert len(findings) == 3                       # 2 warnings + 1 error; note/noise dropped
    assert all(f["tool"] == "clang-tidy" for f in findings)
    files = [f["file"] for f in findings]
    assert files == ["/src/a.cpp", "/src/a.cpp", "/src/b.cpp"]


def test_severity_and_rule_extraction():
    findings = parse_clang_tidy_output(SAMPLE)
    by_line = {f["start_line"]: f for f in findings}
    assert by_line[12]["severity"] == "medium"      # warning -> medium
    assert by_line[12]["rule"] == "clang-analyzer-security.insecureAPI.strcpy"
    assert by_line[12]["start_line"] == by_line[12]["end_line"] == 12
    assert by_line[20]["severity"] == "high"        # error -> high
    assert by_line[20]["rule"] == "clang-diagnostic-error"
    assert "insecure" in by_line[12]["issue"]
    assert "[" not in by_line[12]["issue"]           # check name stripped from message


def test_diagnostic_without_check_falls_back_to_clang_tidy_rule():
    findings = parse_clang_tidy_output("/x.cpp:1:1: warning: something odd\n")
    assert len(findings) == 1
    assert findings[0]["rule"] == "clang-tidy"


def test_empty_output_yields_no_findings():
    assert parse_clang_tidy_output("") == []


def test_run_clang_tidy_skips_gracefully_when_not_installed(monkeypatch):
    monkeypatch.setattr(ensemble_scan.shutil, "which", lambda _name: None)
    assert run_clang_tidy("anything") == []
