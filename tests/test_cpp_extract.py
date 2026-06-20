"""Tests for C++ function/method extraction via the tree-sitter-cpp grammar.

Skips cleanly when the C++ grammar is not installed. The C grammar cannot model
classes/templates/namespaces, so these specifically exercise the cpp path.
"""
import os

import pytest

from local_vuln_scanner import (
    extract_functions,
    _extract_functions_treesitter,
    treesitter_available,
    lang_for_path,
)

cpp = pytest.mark.skipif(not treesitter_available("cpp"),
                         reason="tree-sitter-cpp not installed")

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.cpp")


def _first_lines(res):
    return [lines[0].strip() for (_a, _b, lines) in res]


@cpp
def test_cpp_fixture_captures_all_function_kinds():
    code = open(FIXTURE, encoding="utf-8").read()
    res = _extract_functions_treesitter(code, lang="cpp")
    assert res is not None
    firsts = _first_lines(res)
    # 7 definitions: template fn, ctor, dtor, inline method, out-of-line method,
    # operator overload, free function. The two declaration-only members are not.
    assert len(res) == 7
    assert any(fl.startswith("template") for fl in firsts)        # template header kept
    assert any(fl.startswith("Buffer(const char") for fl in firsts)  # constructor
    assert any(fl.startswith("~Buffer()") for fl in firsts)       # destructor
    assert any("int size()" in fl for fl in firsts)               # inline member
    assert any("Buffer::copy" in fl for fl in firsts)             # out-of-line method
    assert any("operator=" in fl for fl in firsts)                # operator overload
    assert any(fl.startswith("int run(") for fl in firsts)        # free function


@cpp
def test_template_function_keeps_its_header():
    code = "template <typename T>\nT max2(T a, T b) {\n    return a > b ? a : b;\n}\n"
    res = _extract_functions_treesitter(code, lang="cpp")
    assert len(res) == 1
    a, b, lines = res[0]
    assert a == 0 and lines[0].startswith("template")  # header travels with the body


@cpp
def test_class_template_descends_to_member_methods():
    # A *class* template must not be captured whole; its member function is.
    code = (
        "template <typename T>\n"
        "class Box {\n"
        "public:\n"
        "    T get() const { return v_; }\n"
        "private:\n"
        "    T v_;\n"
        "};\n"
    )
    res = _extract_functions_treesitter(code, lang="cpp")
    assert res is not None
    firsts = _first_lines(res)
    assert any("T get()" in fl for fl in firsts)
    assert not any(fl.startswith("class Box") for fl in firsts)


@cpp
def test_lambda_is_not_a_separate_function():
    code = "int f() {\n    auto g = [](int x){ return x + 1; };\n    return g(1);\n}\n"
    res = _extract_functions_treesitter(code, lang="cpp")
    assert len(res) == 1                        # only f(); the lambda is inside it
    assert res[0][0] == 0


@cpp
def test_namespaced_function_is_captured():
    code = "namespace a {\nnamespace b {\nint deep() { return 0; }\n}\n}\n"
    res = _extract_functions_treesitter(code, lang="cpp")
    assert len(res) == 1
    assert "int deep()" in res[0][2][0]


@cpp
def test_extract_functions_dispatch_cpp_lang():
    code = open(FIXTURE, encoding="utf-8").read()
    res = extract_functions(code, parser="auto", lang="cpp")
    assert res is not None and len(res) == 7


def test_lang_for_path_extension_mapping():
    assert lang_for_path("a.cpp") == "cpp"
    assert lang_for_path("a.cc") == "cpp"
    assert lang_for_path("a.cxx") == "cpp"
    assert lang_for_path("a.hpp") == "cpp"
    assert lang_for_path("a.c") == "c"
    assert lang_for_path("a.weird") == "c"      # unknown -> conservative default


@pytest.mark.skipif(not treesitter_available("cpp"),
                    reason="cpp grammar present changes .h default")
def test_h_header_prefers_cpp_when_available():
    assert lang_for_path("a.h") == "cpp"
