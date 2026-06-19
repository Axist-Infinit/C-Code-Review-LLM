"""Unit tests for local_vuln_scanner.extract_functions.

These exercise the signature-boundary walk-back (no torch required: the heavy
ML imports in local_vuln_scanner are lazy, so the module imports clean here).
"""
import os

from local_vuln_scanner import extract_functions, _signature_start


def _spans(code):
    """Return list of (start, end) spans (0-based, [start, end))."""
    res = extract_functions(code)
    return None if res is None else [(a, b) for (a, b, _lines) in res]


def test_simple_same_line_brace():
    code = "static void foo(int x) {\n    bar(x);\n}\n"
    assert _spans(code) == [(0, 3)]


def test_multiline_signature_starts_at_return_type():
    # return type on its own line; must capture from line 0, not idx-2.
    code = "int\nmain(int argc, char **argv)\n{\n    return 0;\n}\n"
    spans = _spans(code)
    assert spans == [(0, 5)]
    lines = extract_functions(code)[0][2]
    assert lines[0] == "int"


def test_struct_is_not_treated_as_function():
    # A struct decl ends in '};' and has no ')'; only the function is captured.
    code = (
        "struct point {\n"
        "    int x;\n"
        "    int y;\n"
        "};\n"
        "\n"
        "int dist(struct point p) {\n"
        "    return p.x + p.y;\n"
        "}\n"
    )
    spans = _spans(code)
    assert spans == [(5, 8)]


def test_initializer_then_function_no_overcapture():
    # The function signature must start at its own line, not be pulled back
    # into the preceding array initializer.
    code = (
        "static const char *names[] = {\n"
        '    "a", "b", "c"\n'
        "};\n"
        "\n"
        "void process(void)\n"
        "{\n"
        "    work();\n"
        "}\n"
    )
    spans = _spans(code)
    assert spans == [(4, 8)]
    first_line = extract_functions(code)[0][2][0]
    assert first_line == "void process(void)"


def test_comment_above_signature_is_excluded():
    code = "// does a thing\nvoid doit(void)\n{\n    x();\n}\n"
    spans = _spans(code)
    assert spans == [(1, 5)]


def test_two_functions():
    code = (
        "int a(void) {\n"
        "    return 1;\n"
        "}\n"
        "\n"
        "int b(void) {\n"
        "    return 2;\n"
        "}\n"
    )
    assert _spans(code) == [(0, 3), (4, 7)]


def test_no_functions_returns_none():
    code = "#define FOO 1\nextern int g;\nint x;\n"
    assert extract_functions(code) is None


def test_forward_declaration_not_captured():
    # prototype ends with ');' -> not a definition
    code = "int f(int);\nint g(void) {\n    return 0;\n}\n"
    assert _spans(code) == [(1, 4)]


def test_signature_start_helper():
    clean = ["", "int", "main(int argc)", "{"]
    # brace is on line index 3; signature begins at line 1 ("int")
    assert _signature_start(clean, 3) == 1


def test_fixture_file_extraction():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "fixtures", "sample.c"), encoding="utf-8") as fh:
        code = fh.read()
    res = extract_functions(code)
    assert res is not None
    # struct + array initializer must be excluded; only add() and greet().
    first_lines = [lines[0] for (_a, _b, lines) in res]
    assert any(fl == "int" for fl in first_lines)            # add's return type line
    assert any("greet" in fl for fl in first_lines)          # greet captured
    assert not any("struct point" in fl for fl in first_lines)
