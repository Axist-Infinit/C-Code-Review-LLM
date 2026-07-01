"""Tests for the tree-sitter function-extraction backend and parser dispatch.

These skip cleanly when tree-sitter is not installed (it is an optional dep), so
the suite stays green on a minimal environment while exercising the real parser
wherever it is available (CI installs it).
"""
import pytest

from local_vuln_scanner import (
    extract_functions,
    treesitter_available,
    _extract_functions_brace,
    _extract_functions_treesitter,
)

ts = pytest.mark.skipif(not treesitter_available(),
                        reason="tree-sitter not installed")


def _spans(res):
    return None if res is None else [(a, b) for (a, b, _l) in res]


@ts
def test_treesitter_matches_brace_on_simple_cases():
    code = (
        "struct point { int x; };\n"
        "int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n"
        "static void greet(const char *who) {\n"
        "    char buf[16];\n"
        "    strcpy(buf, who);\n"
        "}\n"
    )
    # add() at rows 1-3, greet() at rows 4-7; struct excluded.
    assert _spans(_extract_functions_treesitter(code)) == [(1, 4), (4, 8)]


@ts
def test_treesitter_returns_none_when_no_functions():
    code = "#define FOO 1\nextern int g;\nint x;\n"
    assert _extract_functions_treesitter(code) is None


@ts
def test_treesitter_handles_macro_heavy_definition():
    # Attribute macros + a macro-wrapped return type confuse the brace heuristic's
    # "preceded by ')'" guess far more than a real parser. tree-sitter still finds
    # the function body and captures the whole definition.
    code = (
        "#define API __attribute__((visibility(\"default\")))\n"
        "API int process(struct ctx *c, size_t n)\n"
        "{\n"
        "    char buf[64];\n"
        "    memcpy(buf, c->data, n);\n"
        "    return 0;\n"
        "}\n"
    )
    res = _extract_functions_treesitter(code)
    assert res is not None
    # exactly one function captured, body includes the memcpy line
    assert len(res) == 1
    a, b, lines = res[0]
    assert any("memcpy" in ln for ln in lines)


@ts
def test_treesitter_windows_oversized_function():
    body = "\n".join(f"    x += {i};" for i in range(300))
    code = f"int big(void) {{\n{body}\n}}\n"
    res = _extract_functions_treesitter(code, max_lines=50)
    # one 302-line function must be split into several <=60-line windows
    assert len(res) > 1
    assert all((b - a) <= 60 for (a, b, _l) in res)


def test_dispatch_brace_mode_forces_brace():
    code = "int add(int a, int b) {\n    return a + b;\n}\n"
    assert _spans(extract_functions(code, parser="brace")) == _spans(_extract_functions_brace(code))


def test_dispatch_auto_agrees_with_brace_on_fixture_shape():
    # auto picks tree-sitter when present, brace otherwise; either way the public
    # contract (which functions, in order) is identical for clean code.
    code = (
        "int a(void) {\n    return 1;\n}\n"
        "\n"
        "int b(void) {\n    return 2;\n}\n"
    )
    assert _spans(extract_functions(code, parser="auto")) == [(0, 3), (4, 7)]


def test_dispatch_treesitter_mode_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr("local_vuln_scanner.treesitter_available", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="not installed"):
        extract_functions("int f(void){return 0;}", parser="treesitter")


def test_parser_engine_banner_respects_cpp_only_grammar(monkeypatch):
    # The banner used to consult only the default C grammar, reporting "brace"
    # even when the cpp grammar was installed and would be used per-file.
    import local_vuln_scanner as lvs
    monkeypatch.setattr(lvs, "treesitter_available", lambda lang="c": lang == "cpp")
    assert lvs._parser_engine("auto", "cpp") == "tree-sitter"
    assert lvs._parser_engine("auto", "auto") == "tree-sitter"   # any grammar counts
    assert lvs._parser_engine("auto", "c") == "brace"
    assert lvs._parser_engine("brace", "cpp") == "brace"


def test_parser_engine_banner_brace_when_no_grammars(monkeypatch):
    import local_vuln_scanner as lvs
    monkeypatch.setattr(lvs, "treesitter_available", lambda lang="c": False)
    assert lvs._parser_engine("auto", "auto") == "brace"
    assert lvs._parser_engine("treesitter", "cpp") == "brace"
