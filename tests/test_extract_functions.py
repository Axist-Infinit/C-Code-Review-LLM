"""Unit tests for local_vuln_scanner.extract_functions.

These exercise the signature-boundary walk-back and the container-aware brace
matcher (no torch required: the heavy ML imports in local_vuln_scanner are
lazy, so the module imports clean here).
"""
import json
import os

from local_vuln_scanner import (
    extract_functions,
    list_sources,
    load_inference_config,
    load_threshold,
    mask_comments_and_strings,
    _signature_start,
)


def _spans(code):
    """Return list of (start, end) spans (0-based, [start, end))."""
    res = extract_functions(code)
    return None if res is None else [(a, b) for (a, b, _lines) in res]


def _bspans(code):
    """Spans from the brace matcher explicitly (same engine everywhere,
    even when a tree-sitter grammar happens to be installed)."""
    res = extract_functions(code, parser="brace")
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


# --- container-aware brace matching (classes, namespaces, extern "C") --------

def test_class_methods_extracted_from_cpp_fixture():
    # playground/vuln_demo.cpp: class Session's three methods must come out as
    # separate spans with correct line numbers, plus the template fn and main.
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "playground", "vuln_demo.cpp")
    with open(path, encoding="utf-8") as fh:
        code = fh.read()
    res = extract_functions(code, parser="brace")
    spans = [(a, b) for (a, b, _l) in res]
    assert spans == [(9, 12), (13, 16), (17, 20), (25, 29), (30, 37)]
    bodies = ["\n".join(l) for (_a, _b, l) in res]
    assert "strcpy" in bodies[0]        # ctor, 1-based lines 10-12
    assert "printf(fmt)" in bodies[1]   # log(), lines 14-16
    assert "system" in bodies[2]        # shell(), lines 18-20


def test_class_method_visible_alongside_free_function():
    # Regression: the class opener used to swallow the method entirely, so the
    # strcpy into char[8] was invisible whenever a file-scope function existed.
    code = (
        "class B {\n"
        "public:\n"
        "    void fill(const char *s) {\n"
        "        strcpy(d_, s);\n"
        "    }\n"
        "private:\n"
        "    char d_[8];\n"
        "};\n"
        "int top(void) {\n"
        "    return 0;\n"
        "}\n"
    )
    assert _bspans(code) == [(2, 5), (8, 11)]


def test_namespace_and_extern_c_nesting():
    code = (
        "namespace outer {\n"
        'extern "C" {\n'
        "int f(void) {\n"
        "    return 1;\n"
        "}\n"
        "}\n"
        "namespace inner {\n"
        "void g(void) {\n"
        "    do_it();\n"
        "}\n"
        "}\n"
        "}\n"
    )
    assert _bspans(code) == [(2, 5), (7, 10)]


def test_namespace_after_function_not_captured_whole():
    # A ')' within the look-back above 'namespace a {' used to turn the whole
    # namespace into one bogus span.
    code = (
        "int f(void) { return g(); }\n"
        "namespace a {\n"
        "int h(void) { return 1; }\n"
        "}\n"
    )
    assert _bspans(code) == [(0, 1), (2, 3)]


def test_extern_c_single_declaration_is_a_function():
    # extern "C" WITHOUT a brace-block is not a container.
    code = 'extern "C" int f(void) {\n    return 0;\n}\n'
    assert _bspans(code) == [(0, 3)]


def test_struct_data_definition_still_excluded():
    # Data definitions must not become spans even though 'struct' now opens a
    # transparent container and the table entries carry parens.
    code = (
        "struct cmd { const char *name; int (*fn)(void); };\n"
        "static struct cmd table[] = {\n"
        '    { "alpha", (int (*)(void))run_alpha },\n'
        '    { "beta",  run_beta  },\n'
        "};\n"
        "int run_beta(void) {\n"
        "    return 2;\n"
        "}\n"
    )
    assert _bspans(code) == [(5, 8)]


def test_function_returning_struct_is_still_a_function():
    code = "struct point mk(int x) {\n    struct point p; return p;\n}\n"
    assert _bspans(code) == [(0, 3)]


def test_single_line_paren_initializer_not_a_span():
    # ')' AFTER the '{' on the same line used to create a false function span.
    code = (
        "static int arr[] = { (int)F(1),\n"
        "                     (int)F(2) };\n"
        "void ok(void) {\n"
        "    work();\n"
        "}\n"
    )
    assert _bspans(code) == [(2, 5)]


def test_single_line_paren_initializer_alone_yields_none():
    code = "static int arr[] = { (int)F(1), (int)F(2) };\n"
    assert extract_functions(code, parser="brace") is None


def test_kr_function_still_captured():
    # K&R param declarations sit between the ')' and the '{'.
    code = "int foo(a, b)\nint a;\nint b;\n{\n    return a + b;\n}\n"
    assert _bspans(code) == [(3, 6)]


# --- data blocks / brace init after a ')' must not become spans ---------------
# Regression tests for the bounded look-back: a previous statement's ')' used
# to qualify the next construct's '{' as a function body.

def test_enum_after_function_not_a_span():
    code = (
        "void f(void) { do_thing(); }\n"
        "enum Color { RED, GREEN, BLUE };\n"
    )
    assert _bspans(code) == [(0, 1)]


def test_multiline_enum_after_function_not_a_span():
    code = (
        "void f(void) { do_thing(); }\n"
        "enum flags {\n"
        "    F_A = 1,\n"
        "    F_B = 2,\n"
        "};\n"
    )
    assert _bspans(code) == [(0, 1)]


def test_prototype_then_enum_yields_none():
    code = (
        "void f(void);\n"
        "enum Color { RED, GREEN, BLUE };\n"
    )
    assert extract_functions(code, parser="brace") is None


def test_enum_class_after_function_not_a_span():
    code = (
        "void f(void) { do_thing(); }\n"
        "enum class Color : uint8_t { RED, GREEN };\n"
    )
    assert _bspans(code) == [(0, 1)]


def test_static_struct_data_after_function_not_a_span():
    code = (
        "void f(void) { do_thing(); }\n"
        "static struct { int a; } gcfg = { 1 };\n"
    )
    assert _bspans(code) == [(0, 1)]


def test_typedef_struct_after_function_not_a_span():
    code = (
        "int f(void) { return g(); }\n"
        "typedef struct {\n"
        "    int a;\n"
        "} cfg_t;\n"
    )
    assert _bspans(code) == [(0, 1)]


def test_cpp_brace_init_after_function_not_a_span():
    # brace initialization without '=' used to pass the no-'='-after-')' check.
    code = (
        "void f() { g(); }\n"
        "std::atomic<int> counter{0};\n"
    )
    assert _bspans(code) == [(0, 1)]


def test_in_class_brace_init_member_not_a_span():
    code = (
        "class A {\n"
        "    void f() { work(); }\n"
        "    int limit{100};\n"
        "};\n"
    )
    assert _bspans(code) == [(1, 2)]


def test_ctor_init_list_yields_exactly_one_span():
    # x_{0} / y_{1} used to be emitted as duplicate one-line spans on top of
    # the real ctor span.
    code = (
        "class A {\n"
        "    A() : x_{0}, y_{1} {\n"
        "        init();\n"
        "    }\n"
        "};\n"
    )
    assert _bspans(code) == [(1, 4)]


# --- mask_comments_and_strings ------------------------------------------------

def test_mask_preserves_length_and_line_structure():
    code = (
        "int a; // trailing { comment\n"
        'char *s = "br{ace\\"s";\n'
        "/* multi\n"
        "   line { */ int b;\n"
        "char c = '{';\n"
    )
    masked = mask_comments_and_strings(code)
    assert len(masked) == len(code)
    assert [len(l) for l in masked.splitlines()] == [len(l) for l in code.splitlines()]
    # no masked region leaks a brace
    assert masked.count("{") == 0
    # delimiters survive in place; comment/string text is spaces
    assert masked.splitlines()[0].startswith("int a; //")
    assert masked.splitlines()[2] == "/*      "
    assert "*/ int b;" in masked.splitlines()[3]
    assert masked.splitlines()[4] == "char c = ' ';"


def test_mask_line_comment_text_blanked():
    assert mask_comments_and_strings("x; // hi\n") == "x; //   \n"


def test_mask_string_contents_and_escaped_quotes():
    # contents (incl. the escaped quote) become spaces; quotes remain
    assert mask_comments_and_strings('f("a\\"b");') == 'f("    ");'


def test_mask_block_comment_spans_lines():
    assert mask_comments_and_strings("a /* x\ny */ b") == "a /*  \n  */ b"


def test_mask_code_outside_regions_untouched():
    code = "int main(void) { return 0; }\n"
    assert mask_comments_and_strings(code) == code


def test_digit_separator_is_not_a_char_literal():
    # C++14 digit separator: the apostrophe in 10'000 used to open char-literal
    # state, masking the ')' and '{' and truncating the function span.
    code = (
        "int check(long n) {\n"
        "    if (n > 10'000) {\n"
        "        overflow();\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    )
    masked = mask_comments_and_strings(code)
    assert len(masked) == len(code)
    assert masked.splitlines()[1] == "    if (n > 10'000) {"
    assert _bspans(code) == [(0, 6)]


def test_multiple_digit_separators_and_char_literal_coexist():
    code = "long x = 1'000'000; char c = '{'; int f(void) {\n    return 0;\n}\n"
    masked = mask_comments_and_strings(code)
    assert "1'000'000" in masked
    assert masked.count("{") == 1  # only the function brace survives
    assert _bspans(code) == [(0, 3)]


def test_wide_char_literal_still_masked():
    # 'L' before the quote is an encoding prefix, not a digit separator.
    assert mask_comments_and_strings("wchar_t c = L'{';\n") == "wchar_t c = L' ';\n"


def test_raw_string_multiline_masked_and_span_intact():
    # a raw string with per-line unbalanced '}' used to pop the enclosing
    # function's brace and truncate its span.
    code = (
        "const char* schema() {\n"
        '    return R"({\n'
        '  "a": { "b": 1 }\n'
        "}\n"
        '})";\n'
        "}\n"
        "int after(int x) {\n"
        "    return x;\n"
        "}\n"
    )
    masked = mask_comments_and_strings(code)
    assert len(masked) == len(code)
    assert [len(l) for l in masked.splitlines()] == [len(l) for l in code.splitlines()]
    # every brace inside the raw literal is masked; only code braces survive
    assert masked.count("{") == 2
    assert masked.count("}") == 2
    assert _bspans(code) == [(0, 6), (6, 9)]


def test_raw_string_with_delimiter():
    # ')"' inside the literal must not close it: the delimiter is 'ab'.
    code = 'const char *p = R"ab( }} )" )ab";\nint f(void) {\n    return 0;\n}\n'
    masked = mask_comments_and_strings(code)
    assert len(masked) == len(code)
    assert masked.count("{") == 1
    assert masked.count("}") == 1
    assert _bspans(code) == [(1, 4)]


def test_identifier_ending_in_R_is_not_a_raw_prefix():
    masked = mask_comments_and_strings('call(VAR"{text}");\n')
    assert masked == 'call(VAR"      ");\n'


# --- inference config (C1 contract) -------------------------------------------

def test_load_inference_config_defaults_when_missing(tmp_path):
    cfg = load_inference_config(str(tmp_path))
    assert cfg["threshold"] is None
    assert cfg["max_length"] == 512


def test_load_inference_config_reads_explicit_values(tmp_path):
    (tmp_path / "inference.json").write_text(json.dumps({
        "threshold": 0.37, "max_length": 256, "notes": "tuned",
        "provenance": {"dataset": "bigvul"},
    }))
    cfg = load_inference_config(str(tmp_path))
    assert cfg["threshold"] == 0.37
    assert cfg["max_length"] == 256
    assert cfg["notes"] == "tuned"                     # extra keys pass through
    assert cfg["provenance"] == {"dataset": "bigvul"}


def test_load_inference_config_partial_and_garbage(tmp_path):
    (tmp_path / "inference.json").write_text('{"threshold": 0.5}')
    assert load_inference_config(str(tmp_path))["max_length"] == 512
    (tmp_path / "inference.json").write_text("not json")
    assert load_inference_config(str(tmp_path)) == {"threshold": None,
                                                    "max_length": 512}
    (tmp_path / "inference.json").write_text('{"max_length": -3}')
    assert load_inference_config(str(tmp_path))["max_length"] == 512


def test_load_threshold_still_works(tmp_path):
    (tmp_path / "inference.json").write_text('{"threshold": 0.44, "notes": "n"}')
    assert load_threshold(str(tmp_path)) == 0.44
    (tmp_path / "inference.json").write_text("garbage")
    (tmp_path / "threshold.txt").write_text("0.61\n")
    assert load_threshold(str(tmp_path)) == 0.61


# --- source discovery -----------------------------------------------------------

def test_template_impl_extensions_discovered(tmp_path):
    for name in ("a.tcc", "b.ipp", "c.inl", "d.c", "skip.txt"):
        (tmp_path / name).write_text("int x;\n")
    found = [os.path.basename(p) for p in list_sources(str(tmp_path))]
    assert found == ["a.tcc", "b.ipp", "c.inl", "d.c"]
