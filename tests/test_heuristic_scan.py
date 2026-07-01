"""Tests for the model-free heuristic scanner (no torch / model required)."""
import json

import heuristic_scan
from heuristic_scan import scan_code, scan_path


def test_flags_function_with_unsafe_api():
    code = (
        "void copy(char *s) {\n"
        "    char d[8];\n"
        "    strcpy(d, s);\n"
        "}\n"
    )
    findings = scan_code(code, lang="c")
    assert len(findings) == 1
    f = findings[0]
    assert f["start_line"] == 1 and f["end_line"] == 4
    assert "strcpy" in f["matched_patterns"]
    assert f["cwe"] == "CWE-120"


def test_clean_function_is_not_flagged():
    code = "int add(int a, int b) {\n    return a + b;\n}\n"
    assert scan_code(code, lang="c") == []


def test_only_flags_the_function_that_matches():
    code = (
        "int safe(int x) {\n    return x + 1;\n}\n"
        "\n"
        "void danger(char *s) {\n    char b[4];\n    strcpy(b, s);\n}\n"
    )
    findings = scan_code(code, lang="c")
    assert len(findings) == 1
    assert findings[0]["start_line"] == 5      # only danger(), not safe()


def test_cpp_method_is_flagged_via_cpp_grammar():
    code = (
        "class S {\n"
        "public:\n"
        "    int run(const std::string &c) { return system(c.c_str()); }\n"
        "};\n"
    )
    findings = scan_code(code, lang="cpp")
    assert len(findings) == 1
    assert "system" in findings[0]["matched_patterns"]
    assert findings[0]["cwe"] == "CWE-78"


def test_file_without_functions_is_scanned_as_one_span():
    # no function body -> whole-file span still catches a top-level unsafe call
    code = "#define COPY(d, s) strcpy(d, s)\n"
    findings = scan_code(code, lang="c")
    assert len(findings) == 1
    assert "strcpy" in findings[0]["matched_patterns"]


def test_scan_path_over_directory(tmp_path):
    (tmp_path / "a.c").write_text("void f(char*s){char b[4];strcpy(b,s);}\n")
    (tmp_path / "b.cpp").write_text(
        "class T { public: int r(const char*c){ return system(c);} };\n")
    (tmp_path / "clean.c").write_text("int g(void){return 0;}\n")
    findings = scan_path(str(tmp_path))
    files = {f["file"].split("/")[-1] for f in findings}
    assert files == {"a.c", "b.cpp"}            # clean.c produced no findings
    assert all("score" in f and "cwe" in f for f in findings)


def test_main_writes_classifier_schema(tmp_path, monkeypatch):
    src = tmp_path / "x.c"
    src.write_text("void f(char*s){char b[4];strcpy(b,s);}\n")
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["heuristic_scan.py", str(src), "-o", str(out)])
    heuristic_scan.main()
    data = json.loads((out / "classifier_findings.json").read_text())
    assert data["profile"] == "heuristic"
    assert isinstance(data["findings"], list) and len(data["findings"]) == 1
    assert data["findings"][0]["file"] == str(src)


# --- match_lines: absolute anchors for SARIF/PR comments ----------------------

def test_findings_carry_absolute_sorted_match_lines():
    code = (
        "int safe(int x) {\n    return x + 1;\n}\n"
        "\n"
        "void danger(char *s) {\n    char b[4];\n    strcpy(b, s);\n    gets(b);\n}\n"
    )
    findings = scan_code(code, lang="c")
    assert len(findings) == 1
    f = findings[0]
    assert f["match_lines"] == [7, 8]           # absolute 1-based, sorted
    assert f["start_line"] <= f["match_lines"][0] <= f["end_line"]


def test_commented_out_call_is_not_flagged():
    code = "void f(char *s) {\n    /* strcpy(b, s); */\n    return;\n}\n"
    assert scan_code(code, lang="c") == []


# --- residual coverage: top-level lines outside every function span ----------

def test_toplevel_macro_between_functions_is_scanned():
    code = ("#define COPY(d, s) strcpy(d, s)\n"
            "int add(int a, int b) {\n    return a + b;\n}\n")
    findings = scan_code(code, lang="c")
    assert len(findings) == 1
    f = findings[0]
    assert f["start_line"] == 1 and f["match_lines"] == [1]
    assert "strcpy" in f["matched_patterns"]


def test_gap_scan_does_not_double_report_function_bodies():
    code = ("#define N 8\n"
            "void danger(char *s) {\n    char b[N];\n    strcpy(b, s);\n}\n")
    findings = scan_code(code, lang="c")
    assert len(findings) == 1                   # the function only; macro is clean
    assert findings[0]["start_line"] == 2
    assert findings[0]["match_lines"] == [4]


# --- inline suppression -------------------------------------------------------

def test_ccr_ignore_suppresses_the_whole_line():
    code = "void f(char *s) {\n    char b[4];\n    strcpy(b, s); /* ccr-ignore */\n}\n"
    assert scan_code(code, lang="c") == []


def test_ccr_ignore_rule_list_only_suppresses_named_rules():
    code = ("void f(char *s) {\n"
            "    char b[4];\n"
            "    strcpy(b, s); // ccr-ignore: memcpy\n"   # wrong rule -> kept
            "}\n")
    findings = scan_code(code, lang="c")
    assert len(findings) == 1
    assert findings[0]["matched_patterns"] == ["strcpy"]


def test_ccr_ignore_drops_finding_only_when_all_lines_suppressed():
    code = ("void f(char *s) {\n"
            "    char b[4];\n"
            "    strcpy(b, s); // ccr-ignore: strcpy\n"
            "    gets(b);\n"
            "}\n")
    findings = scan_code(code, lang="c")
    assert len(findings) == 1
    f = findings[0]
    assert "strcpy" not in f["matched_patterns"] and "gets" in f["matched_patterns"]
    assert f["match_lines"] == [4]
    assert f["cwe"] == "CWE-242"                # recomputed from surviving rules


def test_ccr_ignore_suppression_survives_explain_findings_cli(tmp_path, monkeypatch):
    """Confirmed bug: the explain_findings CLI re-derived patterns from the
    snippet and resurrected the suppressed 'system' rule, flipping the finding
    to CWE-78/high. Suppression must hold end-to-end: heuristic_scan output AND
    the downstream re-explanation both report strtok only (CWE-676)."""
    import explain_findings
    src = tmp_path / "t.c"
    src.write_text('void f(char *cmd, char *line) {\n'
                   '    system(cmd); /* ccr-ignore: system */\n'
                   '    char *t = strtok(line, ",");\n'
                   '}\n')
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["heuristic_scan.py", str(src), "-o", str(out)])
    heuristic_scan.main()
    data = json.loads((out / "classifier_findings.json").read_text())
    assert len(data["findings"]) == 1
    f = data["findings"][0]
    assert f["matched_patterns"] == ["strtok_nonreentrant"]
    assert f["cwe"] == "CWE-676"
    assert f["match_lines"] == [3]

    outp = tmp_path / "llm_findings_heuristic.json"
    monkeypatch.setattr("sys.argv", ["explain_findings.py",
                                     str(out / "classifier_findings.json"),
                                     "--out", str(outp)])
    explain_findings.main()
    entry = json.loads(outp.read_text())["explanations"][0]
    assert entry["matched_patterns"] == ["strtok_nonreentrant"]  # not resurrected
    assert entry["cwe"] == "CWE-676"            # not CWE-78
    assert entry["severity"] == "low"           # not high
    assert entry["match_lines"] == [3]


# --- file excludes and severity floor ------------------------------------------

def test_exclude_glob_filters_by_relative_path(tmp_path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "x.c").write_text("void f(char*s){char b[4];strcpy(b,s);}\n")
    (tmp_path / "main.c").write_text("void g(char*s){char b[4];strcpy(b,s);}\n")
    findings = scan_path(str(tmp_path), exclude=["vendor/*"])
    assert {f["file"].split("/")[-1] for f in findings} == {"main.c"}


def test_min_severity_floor_drops_low_findings(tmp_path):
    (tmp_path / "low.c").write_text('void f(char *l){char *t = strtok(l, ",");}\n')
    (tmp_path / "high.c").write_text("void g(char*s){char b[4];strcpy(b,s);}\n")
    everything = scan_path(str(tmp_path))
    assert {f["file"].split("/")[-1] for f in everything} == {"low.c", "high.c"}
    floored = scan_path(str(tmp_path), min_severity="medium")
    assert {f["file"].split("/")[-1] for f in floored} == {"high.c"}


def test_main_cli_wires_noise_controls(tmp_path, monkeypatch):
    (tmp_path / "keep.c").write_text("void f(char*s){char b[4];strcpy(b,s);}\n")
    (tmp_path / "skip.c").write_text("void g(char*s){char b[4];strcpy(b,s);}\n")
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["heuristic_scan.py", str(tmp_path), "-o", str(out),
                                     "--exclude", "skip.c", "--min-severity", "low"])
    heuristic_scan.main()
    data = json.loads((out / "classifier_findings.json").read_text())
    files = {f["file"].split("/")[-1] for f in data["findings"]}
    assert files == {"keep.c"}
    assert data["findings"][0]["match_lines"] == [1]
