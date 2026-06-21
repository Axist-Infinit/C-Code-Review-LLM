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
