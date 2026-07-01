"""Tests for the clang-tidy integration and merge logic in ensemble_scan.

parse_clang_tidy_output / split_files_by_lang / corroborate / dedup_findings are
pure, so they are tested directly; run_clang_tidy is checked for graceful no-op
when the binary is absent and for per-language invocation via a mocked
subprocess (no clang-tidy needed in CI).
"""
import json
import subprocess

import ensemble_scan
from ensemble_scan import (
    corroborate,
    dedup_findings,
    load_ml,
    parse_clang_tidy_output,
    run_clang_tidy,
    split_files_by_lang,
)

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


def test_split_files_by_lang_puts_headers_with_cpp():
    groups = split_files_by_lang(["a.c", "b.cpp", "c.h", "d.cc", "e.hpp"])
    assert groups["c"] == ["a.c"]
    assert groups["cpp"] == ["b.cpp", "c.h", "d.cc", "e.hpp"]


class _FakeProc:
    stdout = ""
    stderr = ""


def test_run_clang_tidy_invokes_per_language_with_matching_std(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(ensemble_scan.shutil, "which", lambda _n: "/usr/bin/clang-tidy")
    monkeypatch.setattr(ensemble_scan, "list_sources", lambda _s: ["a.c", "b.cpp", "c.h"])
    monkeypatch.setattr(ensemble_scan.subprocess, "run", fake_run)
    assert run_clang_tidy("src") == []
    assert len(calls) == 2
    c_cmd = next(c for c in calls if "-std=c11" in c)
    cpp_cmd = next(c for c in calls if "-std=c++17" in c)
    assert "a.c" in c_cmd and "b.cpp" not in c_cmd and "c.h" not in c_cmd
    assert "b.cpp" in cpp_cmd and "c.h" in cpp_cmd and "a.c" not in cpp_cmd


def test_run_clang_tidy_skips_c_invocation_when_no_c_files(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(ensemble_scan.shutil, "which", lambda _n: "/usr/bin/clang-tidy")
    monkeypatch.setattr(ensemble_scan, "list_sources", lambda _s: ["b.cpp"])
    monkeypatch.setattr(ensemble_scan.subprocess, "run", fake_run)
    run_clang_tidy("src")
    assert len(calls) == 1
    assert "-std=c++17" in calls[0]


def test_tool_timeouts_are_caught_and_skipped(monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 600)

    monkeypatch.setattr(ensemble_scan.shutil, "which", lambda _n: "/usr/bin/tool")
    monkeypatch.setattr(ensemble_scan, "list_sources", lambda _s: ["a.c", "b.cpp"])
    monkeypatch.setattr(ensemble_scan.subprocess, "run", boom)
    assert run_clang_tidy("src") == []
    assert ensemble_scan.run_cppcheck("src") == []
    assert ensemble_scan.run_flawfinder("src") == []


def test_load_ml_preserves_cwe_fix_verdict_and_anchors(tmp_path):
    p = tmp_path / "llm_findings.json"
    p.write_text(json.dumps({"explanations": [{
        "file": "a.c", "start_line": 5, "end_line": 12, "score": 0.9,
        "cwe": "CWE-120", "severity": "high", "issue": "strcpy overflow",
        "fix": "Use strlcpy.", "is_vulnerable": True,
        "matched_patterns": ["strcpy"], "match_lines": [7],
        "cross_check": "corroborated by cppcheck",
        "what_code_does": "copies", "what_could_go_wrong": "overflow",
        "vulnerability": "buffer overflow",
    }]}))
    out = load_ml(str(p))
    assert len(out) == 1
    e = out[0]
    assert e["cwe"] == "CWE-120"          # no longer renamed away
    assert e["rule"] == "CWE-120"         # kept for tool-finding parity
    assert e["fix"] == "Use strlcpy."
    assert e["is_vulnerable"] is True
    assert e["matched_patterns"] == ["strcpy"]
    assert e["match_lines"] == [7]
    assert e["cross_check"] == "corroborated by cppcheck"
    assert e["what_could_go_wrong"] == "overflow"


def test_load_ml_omits_absent_optional_keys(tmp_path):
    p = tmp_path / "classifier_findings.json"
    p.write_text(json.dumps({"findings": [
        {"file": "a.c", "start_line": 1, "end_line": 3, "score": 0.7},
    ]}))
    e = load_ml(str(p))[0]
    for key in ("fix", "is_vulnerable", "matched_patterns", "match_lines", "cross_check"):
        assert key not in e


def test_corroborate_uses_containment_not_exact_line():
    ml = {"tool": "ml-classifier", "file": "a.c", "start_line": 5, "end_line": 12}
    tool = {"tool": "cppcheck", "file": "a.c", "start_line": 7, "end_line": 7}
    other_file = {"tool": "flawfinder", "file": "b.c", "start_line": 7, "end_line": 7}
    outside = {"tool": "clang-tidy", "file": "a.c", "start_line": 99, "end_line": 99}
    corroborate([ml, tool, other_file, outside])
    assert ml["corroborated_by"] == ["cppcheck"]        # 7 in [5,12]; 99 is not
    assert tool["corroborated_by"] == ["ml-classifier"]  # vice versa
    assert other_file["corroborated_by"] == []           # different file
    assert outside["corroborated_by"] == []


def test_corroborate_same_tool_never_corroborates_itself():
    a = {"tool": "cppcheck", "file": "a.c", "start_line": 3, "end_line": 3}
    b = {"tool": "cppcheck", "file": "a.c", "start_line": 3, "end_line": 3}
    corroborate([a, b])
    assert a["corroborated_by"] == [] and b["corroborated_by"] == []


def test_dedup_drops_identical_tool_file_line_rule_rows():
    row = {"tool": "clang-tidy", "file": "a.c", "start_line": 3,
           "rule": "cert-err34-c", "severity": "medium", "issue": "x"}
    kept = dedup_findings([row, dict(row), {**row, "start_line": 4}])
    assert len(kept) == 2
    assert kept[0] is row                                # first occurrence wins
