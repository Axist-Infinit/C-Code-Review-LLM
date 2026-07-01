"""Tests for the heuristic pattern set, including the C++ additions and the
CWE wiring that now feeds SARIF rule ids.
"""
import to_sarif
from explain_findings import (explain_snippet, cwe_for_patterns, PATTERNS,
                              structured_explanation, severity_for_patterns,
                              primary_vuln)
from llm_explain import heuristic_entry


def _matched(snippet):
    _expl, _recs, ids = explain_snippet(snippet)
    return ids


def test_c_patterns_still_match():
    assert "strcpy" in _matched("strcpy(d, s);")
    assert "gets" in _matched("gets(buf);")
    assert "memcpy" in _matched("memcpy(d, s, n);")


def test_cpp_command_injection_patterns():
    assert "system" in _matched('system(user_cmd);')
    assert "popen" in _matched('FILE *p = popen(cmd, "r");')


def test_cpp_cast_patterns():
    assert "reinterpret_cast" in _matched("auto *p = reinterpret_cast<Foo*>(raw);")
    assert "const_cast" in _matched("auto *q = const_cast<Foo*>(cp);")


def test_cpp_memory_and_alloc_patterns():
    assert "memmove" in _matched("memmove(d, s, n);")
    assert "alloca" in _matched("char *p = (char*)alloca(n);")


def test_format_string_pattern_is_high_precision():
    # bare identifier as the format string -> flagged
    assert "printf_format_string" in _matched("printf(user_input);")
    # a literal format string is fine -> not flagged
    assert "printf_format_string" not in _matched('printf("%s", user_input);')


def test_std_qualified_call_still_matches_via_word_boundary():
    # std::strcpy / ::strcpy still trip the strcpy pattern (\b matches at ':').
    assert "strcpy" in _matched("std::strcpy(d, s);")


def test_cwe_for_patterns_priority_and_mapping():
    assert cwe_for_patterns({"strcpy"}) == "CWE-120"
    assert cwe_for_patterns({"system"}) == "CWE-78"
    assert cwe_for_patterns({"reinterpret_cast"}) == "CWE-704"
    assert cwe_for_patterns({"alloca"}) == "CWE-770"
    assert cwe_for_patterns({"printf_format_string"}) == "CWE-134"
    assert cwe_for_patterns(set()) == ""
    # when several match, the higher-priority (earlier) pattern's CWE wins
    assert cwe_for_patterns({"strcpy", "system"}) == "CWE-78"  # system precedes strcpy


def test_heuristic_entry_now_carries_cwe():
    entry = heuristic_entry({"snippet": "strcpy(d, s);", "score": 0.9})
    assert entry["backend"] == "heuristic"
    assert entry["cwe"] == "CWE-120"


def test_heuristic_cwe_flows_into_sarif_rule_id():
    entry = heuristic_entry({"snippet": "system(cmd);", "score": 0.8,
                             "file": "x.cpp", "start_line": 1, "end_line": 1})
    entry.update({"file": "x.cpp", "start_line": 1, "end_line": 1})
    result = to_sarif.finding_to_result(entry)
    assert result["ruleId"] == "CWE-78"          # CWE drives the SARIF rule id


# --- structured 3-part description fields -----------------------------------

def test_every_pattern_carries_structured_description_fields():
    # Each pattern must supply the fields the structured report renders.
    for pat in PATTERNS:
        for key in ("what", "risk", "vuln", "severity", "cwe"):
            assert str(pat.get(key, "")).strip(), f"{pat['id']} missing {key}"


def test_structured_explanation_populates_three_fields():
    s = structured_explanation({"strcpy"})
    assert "strcpy" in s["what_code_does"].lower()
    assert s["what_could_go_wrong"]              # the failure mode
    assert "overflow" in s["vulnerability"].lower()


def test_structured_explanation_generic_when_no_pattern():
    s = structured_explanation(set())
    assert s["vulnerability"] == ""
    assert s["what_code_does"] and s["what_could_go_wrong"]


def test_structured_explanation_joins_multiple_with_separators():
    s = structured_explanation({"gets", "sprintf"})
    # distinct vuln classes are separated, not run together
    assert ";" in s["vulnerability"]


def test_severity_for_patterns_takes_highest():
    # gets is critical, memcpy is medium -> highest wins
    assert severity_for_patterns({"gets", "memcpy"}) == "critical"
    assert severity_for_patterns(set(), default="info") == "info"


def test_primary_vuln_is_highest_priority_single_name():
    # system precedes strcpy in PATTERNS order -> its vuln name is primary
    assert primary_vuln({"strcpy", "system"}) == "OS command injection"
    assert primary_vuln(set()) == ""


def test_heuristic_entry_has_structured_fields():
    entry = heuristic_entry({"snippet": "gets(buf);", "score": 0.9})
    assert entry["what_code_does"] and entry["what_could_go_wrong"]
    assert "overflow" in entry["vulnerability"].lower()
    assert entry["severity"] == "critical"       # gets is critical
    # the headline is a single concise vuln name, not the run-on join
    assert ";" not in entry["issue"]


def test_structured_fields_flow_into_sarif_message():
    entry = heuristic_entry({"snippet": "strcpy(d, s);", "score": 0.9})
    entry.update({"file": "x.c", "start_line": 1, "end_line": 1})
    msg = to_sarif.finding_to_result(entry)["message"]["text"]
    assert "What the code is doing" in msg
    assert "What could go wrong" in msg


# --- expanded vulnerability-class coverage (model-free heuristic) ------------

def test_new_patterns_detect_key_vuln_classes():
    cases = {
        # integer overflow in allocation size
        "p = malloc(n * sizeof(long));": "alloc_mul_overflow",
        "p = realloc(p, count * sz);": "realloc_mul_overflow",
        # off-by-one allocation
        "p = malloc(strlen(s));": "malloc_strlen_no_nul",
        # uncontrolled format string beyond printf
        "fprintf(stderr, msg);": "fprintf_format_string",
        # weak crypto / randomness
        "h = MD5(data, len);": "weak_crypto_primitive",
        "t = rand();": "insecure_prng",
        # timing side channel on a secret
        "if (strcmp(user_mac, expected_hmac) == 0) {}": "insecure_secret_comparison",
        # command exec via PATH search
        "execvp(prog, argv);": "exec_path_search",
        # insecure temp file + world-writable perms
        "char *n = mktemp(tmpl);": "insecure_temp_file",
        "chmod(path, 0777);": "world_writable_perms",
        # unbounded wide/legacy copies
        "wcscpy(dst, src);": "wcscpy_wcscat",
        # size taken from unvalidated input
        "buf = malloc(atoi(argv[1]));": "alloc_size_from_atoi",
    }
    for snippet, expected_id in cases.items():
        assert expected_id in _matched(snippet), f"{expected_id} missed on {snippet!r}"


def test_new_patterns_do_not_flag_safe_siblings():
    # the curated set must stay quiet on these correct usages
    safe = [
        'fprintf(stderr, "%s", msg);',     # literal format
        "p = calloc(n, sizeof(*p));",       # overflow-checked allocator
        "RAND_bytes(key, 32);",             # CSPRNG
        "h = SHA256(data, len);",           # strong hash
        "chmod(path, 0600);",               # owner-only perms
        "mkstemp(tmpl);",                   # atomic temp file
    ]
    for s in safe:
        assert _matched(s) == set(), f"unexpected match on safe line {s!r}: {_matched(s)}"


def test_pattern_cwe_coverage_expanded():
    cwes = {p["cwe"] for p in PATTERNS}
    # new classes the heuristic now covers beyond the original set
    for cwe in ("CWE-190", "CWE-327", "CWE-330", "CWE-208", "CWE-134",
                "CWE-426", "CWE-732", "CWE-377", "CWE-367", "CWE-798"):
        assert cwe in cwes, f"missing {cwe}"


# --- regression guards for the issues caught by adversarial review ----------

def test_no_pattern_has_catastrophic_backtracking():
    # A pathological single line (many commas / repeats) must scan in well under a
    # second across the whole pattern set — guards against the memcpy ReDoS class.
    import time
    pathological = [
        "memcpy(" + "a," * 2000 + "x)",
        "malloc(" + "a," * 2000 + "x)",
        "chmod(" + "a," * 2000 + "0777)",
        "strcmp(" + "a," * 2000 + "password)",
    ]
    for line in pathological:
        t = time.perf_counter()
        explain_snippet(line)
        dt = time.perf_counter() - t
        assert dt < 0.5, f"slow scan ({dt:.2f}s) on {line[:30]}... — possible ReDoS"


def test_secret_comparison_no_false_positive_on_literals_and_sizeof():
    assert "insecure_secret_comparison" not in _matched('strncmp(hdr, "password:", 9);')
    assert "insecure_secret_comparison" not in _matched("memcmp(a, b, sizeof(digest));")
    assert "insecure_secret_comparison" in _matched("memcmp(user_hmac, want_hmac, 32);")


def test_world_writable_no_fp_and_catches_special_bits():
    assert "world_writable_perms" not in _matched("chmod(files[02], 0644);")  # index, safe mode
    assert "world_writable_perms" not in _matched("mkdir(path, 0755);")
    assert "world_writable_perms" in _matched("chmod(path, 0777);")
    assert "world_writable_perms" in _matched("chmod(path, 04777);")          # setuid+world
    assert "world_writable_perms" in _matched("mknod(path, 0666, dev);")      # mode not last arg


def test_namespaced_calls_do_not_false_positive():
    assert "access_toctou" not in _matched("fs::access(p);")
    assert "access_toctou" in _matched("if (access(path, R_OK)) open(path);")
    assert "unix_crypt_weak_hash" not in _matched("obj.crypt(x);")


def test_primary_pattern_drives_coherent_cwe_severity_vuln():
    # When a medium (CWE-120) and a high (CWE-190) pattern both match, all three
    # derived fields come from the more severe one (no cross-pattern disagreement).
    ids = {"memcpy", "alloc_mul_overflow"}
    assert cwe_for_patterns(ids) == "CWE-190"
    assert severity_for_patterns(ids) == "high"
    assert "overflow" in primary_vuln(ids).lower()


# --- C++ / lifetime pattern additions ----------------------------------------

def test_patterns_count_pins_the_full_rule_set():
    # 41 original rules + 11 C++/lifetime additions. Bump when adding rules.
    assert len(PATTERNS) == 52


def test_new_delete_mismatch():
    assert "new_delete_mismatch" in _matched("int *p = new int[8];\ndelete p;")
    assert "new_delete_mismatch" in _matched("Foo *p = new Foo();\ndelete[] p;")
    # correctly paired forms stay quiet
    assert "new_delete_mismatch" not in _matched("int *p = new int[8];\ndelete[] p;")
    assert "new_delete_mismatch" not in _matched("Foo *p = new Foo();\ndelete p;")


def test_use_after_move():
    assert "use_after_move" in _matched("auto b = std::move(a);\nuse(a);")
    # reassignment right after the move re-initialises the object
    assert "use_after_move" not in _matched("auto b = std::move(a);\na = fresh();\nuse(a);")
    # move as the last use is the idiomatic safe case
    assert "use_after_move" not in _matched("auto b = std::move(a);\nreturn b;")


def test_dangling_cstr():
    assert "dangling_cstr" in _matched("const char *p = s.c_str();")
    assert "dangling_cstr" in _matched("std::string tmp = build();\nreturn tmp.c_str();")
    # immediate use as a call argument is fine
    assert "dangling_cstr" not in _matched('printf("%s", s.c_str());')
    # returning a member's c_str() (no local std::string decl) stays quiet
    assert "dangling_cstr" not in _matched("return member_.c_str();")


def test_string_view_temporary():
    assert "string_view_temporary" in _matched('std::string_view sv = std::string("tmp");')
    assert "string_view_temporary" in _matched("std::string_view sv = make_name().substr(0, 3);")
    assert "string_view_temporary" not in _matched("std::string_view sv = name;")


def test_vprintf_family_format_string():
    assert "vprintf_format" in _matched("vfprintf(log_fp, fmt, ap);")
    assert "vprintf_format" in _matched("vsnprintf(buf, sizeof(buf), fmt, ap);")
    assert "vprintf_format" in _matched("vprintf(fmt, ap);")
    assert "vprintf_format" not in _matched('vfprintf(log_fp, "%s\\n", ap);')


def test_strncpy_without_nul_termination():
    assert "strncpy_no_nul" in _matched("strncpy(dst, src, n);")
    assert "strncpy_no_nul" not in _matched("strncpy(dst, src, n);\ndst[n - 1] = '\\0';")
    assert "strncpy_no_nul" not in _matched(
        "strncpy(buf, src, sizeof(buf));\nbuf[sizeof(buf) - 1] = 0;")


def test_free_then_use_and_double_free():
    assert "free_then_use" in _matched("free(p);\nreturn p->next;")
    assert "free_then_use" in _matched("free(p);\nlog_close(q);\nn = p[0];")
    # reassignment between free and use resets the pointer
    assert "free_then_use" not in _matched("free(p);\np = malloc(8);\nuse(p->x);")
    assert "double_free" in _matched("free(p);\nfree(p);")
    assert "double_free" not in _matched("free(p);\np = other;\nfree(p);")


def test_strtok_nonreentrant():
    assert "strtok_nonreentrant" in _matched('char *t = strtok(line, ",");')
    assert "strtok_nonreentrant" not in _matched('char *t = strtok_r(line, ",", &save);')


def test_iterator_invalidation_in_range_for():
    bad = "for (auto &x : items) {\n    if (x.stale) items.erase(items.begin());\n}"
    assert "iterator_invalidation" in _matched(bad)
    # mutating a DIFFERENT container, or the same one after the loop, is fine
    ok = "for (auto &x : items) {\n    out.push_back(x);\n}\nitems.push_back(extra);"
    assert "iterator_invalidation" not in _matched(ok)


def test_malloc_no_null_check():
    assert "malloc_no_null_check" in _matched("char *p = malloc(len);\nmemcpy(p, src, len);")
    assert "malloc_no_null_check" not in _matched(
        "char *p = malloc(len);\nif (!p) return -1;\nmemcpy(p, src, len);")
    assert "malloc_no_null_check" not in _matched(
        "char *p = malloc(len);\nif (p == NULL) return -1;\np->x = 1;")
    # sizeof(*p) inspects the type, not the pointer value
    assert "malloc_no_null_check" not in _matched("p = calloc(n, sizeof(*p));")


# --- memcpy/memmove bounded-by-destination exclusion --------------------------

def test_memcpy_bounded_by_dst_sizeof_not_flagged():
    assert "memcpy" not in _matched("memcpy(buf, src, sizeof(buf));")
    assert "memcpy" not in _matched("memcpy(buf, src, sizeof buf);")
    assert "memmove" not in _matched("memmove(dst, src, sizeof(dst));")


def test_memcpy_still_flags_unbounded_lengths():
    assert "memcpy" in _matched("memcpy(buf, src, n);")
    # sizeof of the SOURCE says nothing about the destination's capacity
    assert "memcpy" in _matched("memcpy(buf, src, sizeof(src));")
    assert "memmove" in _matched("memmove(dst, src, len);")


# --- comment/string masking before matching -----------------------------------

def test_comments_never_match():
    assert _matched("/* old code: strcpy(d, s); */") == set()
    assert _matched("// gets(buf);") == set()


def test_string_literal_contents_do_not_match():
    assert "system" not in _matched('puts("do not call system(rm) here");')
    assert "strcpy" not in _matched('log("strcpy(a, b) was removed");')


def test_patterns_that_need_string_contents_still_see_them():
    assert "scanf_percent_s" in _matched('scanf("%s", buf);')
    assert "scanf_family_unbounded" in _matched('sscanf(line, "%d %s", &n, word);')
    assert "hardcoded_credentials" in _matched('const char *password = "hunter2";')
    # ...but never inside comments
    assert "hardcoded_credentials" not in _matched('// password = "hunter2"')


def test_mask_fallback_is_same_shape_and_masks_comments_and_strings():
    # The local fallback must behave even when the shared helper in
    # local_vuln_scanner is unavailable: same shape, comments/strings blanked.
    from explain_findings import _mask_fallback, _mask_code
    src = 'int a; /* strcpy(d,s) */ puts("system(x)"); // gets(b)\nfoo();'
    masked = _mask_fallback(src)
    assert len(masked) == len(src) and masked.count("\n") == src.count("\n")
    assert "strcpy" not in masked and "system(" not in masked and "gets" not in masked
    assert "foo();" in masked
    # the strings-kept variant still masks comments but preserves literals
    keep = _mask_code(src, mask_strings=False)
    assert "system(x)" in keep and "strcpy" not in keep and "gets" not in keep


def test_detailed_scan_reports_snippet_relative_match_lines():
    from explain_findings import explain_snippet_detailed
    snippet = "void f(char *s) {\n    char d[8];\n    strcpy(d, s);\n    gets(d);\n}"
    _e, _r, ids, matches = explain_snippet_detailed(snippet)
    assert ids == {"strcpy", "gets"}
    assert ("strcpy", 3) in matches and ("gets", 4) in matches


def test_cli_passes_match_lines_through(tmp_path, monkeypatch):
    import json
    import explain_findings
    inp = tmp_path / "cf.json"
    inp.write_text(json.dumps({"findings": [
        {"file": "a.c", "start_line": 1, "end_line": 3, "score": 1.0,
         "snippet": "char b[4];\nstrcpy(b, s);", "match_lines": [2]}]}))
    outp = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv",
                        ["explain_findings.py", str(inp), "--out", str(outp)])
    explain_findings.main()
    data = json.loads(outp.read_text())
    assert data["explanations"][0]["match_lines"] == [2]


# --- ccr-ignore suppression honored end-to-end in explain paths ---------------
# Confirmed bug: explain paths re-matched patterns on the snippet and
# resurrected rules heuristic_scan had suppressed, flipping CWE/severity.

def test_ccr_ignore_suppressed_rule_not_resurrected_by_explain():
    # exact confirmed scenario: suppressed system + live strtok
    snip = ('void f(char *cmd, char *line) {\n'
            '    system(cmd); /* ccr-ignore: system */\n'
            '    char *t = strtok(line, ",");\n'
            '}')
    ids = _matched(snip)
    assert ids == {"strtok_nonreentrant"}          # system stays suppressed
    assert cwe_for_patterns(ids) == "CWE-676"      # not CWE-78
    assert severity_for_patterns(ids) == "low"     # not high


def test_ccr_ignore_bare_marker_suppresses_every_rule_on_the_line():
    assert _matched("strcpy(d, s); /* ccr-ignore */") == set()


def test_ccr_ignore_with_wrong_rule_name_keeps_the_match():
    # same semantics as heuristic_scan: a rule list only suppresses those ids
    assert "strcpy" in _matched("strcpy(d, s); // ccr-ignore: memcpy")


# --- malloc_no_null_check: member-pointer null checks count -------------------
# Confirmed bug: '!s->buf' / 'if (s->buf)' captured the base object 's' while
# the trigger captured the member 'buf', so real checks were not recognized.

def test_malloc_null_check_on_member_pointer_is_recognized():
    assert "malloc_no_null_check" not in _matched(
        "s->buf = malloc(n);\nif (!s->buf) return -1;\ns->buf[0] = 1;")
    assert "malloc_no_null_check" not in _matched(
        "s->buf = malloc(n);\nif (s->buf) { s->buf[0] = 1; }")
    assert "malloc_no_null_check" not in _matched(
        "s->buf = malloc(n);\nif (s->buf == NULL) return -1;\ns->buf[0] = 1;")
    # a genuinely unchecked member allocation still flags
    assert "malloc_no_null_check" in _matched("s->buf = malloc(n);\ns->buf[0] = 1;")


# --- strncpy_no_nul: trigger must not match inside string literals ------------
# Confirmed bug: needs_strings ran the TRIGGER on literal-kept text, so a log
# string merely mentioning strncpy(...) fired the rule.

def test_strncpy_mentioned_in_string_literal_is_not_flagged():
    assert "strncpy_no_nul" not in _matched('puts("call strncpy(buf, src, n) first");')
    # a real strncpy without a NUL store still flags (confirm still sees '\0')
    assert "strncpy_no_nul" in _matched("strncpy(dst, src, n);")
    assert "strncpy_no_nul" not in _matched("strncpy(dst, src, n);\ndst[n - 1] = '\\0';")


# --- raw string literals in the local fallback masker --------------------------
# Confirmed bug: _mask_code had no R"delim(...)delim" support, so a raw string
# with an embedded quote left the masker in string state, masking real code
# after the literal (false-negative regression).

def test_mask_code_lexes_raw_string_literal_as_one_unit():
    from explain_findings import _mask_code
    src = 'void f(char*s){ auto q = R"(")"; strcpy(d, s); }'
    masked = _mask_code(src, mask_strings=True)
    assert len(masked) == len(src)
    assert "strcpy(d, s)" in masked                # code after the literal survives
    # strings-kept variant must also skip the literal without desyncing
    keep = _mask_code(src, mask_strings=False)
    assert len(keep) == len(src) and "strcpy(d, s)" in keep


def test_mask_code_raw_string_with_delimiter_spanning_lines():
    from explain_findings import _mask_code
    src = 'const char *j = R"json({\n  "a": "b"\n})json";\nstrcpy(d, s);'
    masked = _mask_code(src, mask_strings=True)
    assert len(masked) == len(src) and masked.count("\n") == src.count("\n")
    assert "strcpy" in masked
    assert '"a"' not in masked                     # raw body contents are masked


def test_mask_code_raw_prefix_requires_token_boundary():
    from explain_findings import _mask_code
    # FOUR"(y)" is an identifier followed by an ordinary string, not a raw literal
    masked = _mask_code('x = FOUR"(y)"; strcpy(d, s);', mask_strings=True)
    assert "strcpy" in masked and "(y)" not in masked
    # encoding-prefixed raw literals are recognized
    assert "strcpy" in _mask_code('auto q = u8R"(")"; strcpy(d, s);', mask_strings=True)


def test_code_after_raw_string_is_still_scanned_via_fallback(monkeypatch):
    import explain_findings
    # force the local fallback masker (the shared helper is fixed separately)
    monkeypatch.setattr(explain_findings, "_mask_all_impl",
                        explain_findings._mask_fallback)
    assert "strcpy" in _matched('void f(char*s){ auto q = R"(")"; strcpy(d, s); }')


def test_vuln_demo_style_snippets_map_to_expected_cwes():
    # The ground-truth constructs from playground/vuln_demo.cpp, as explicit
    # snippets so this holds regardless of the function extractor in use.
    assert cwe_for_patterns(_matched("strcpy(name_, n);")) == "CWE-120"
    assert cwe_for_patterns(_matched("printf(fmt);")) == "CWE-134"
    assert cwe_for_patterns(_matched("return system(cmd.c_str());")) == "CWE-78"
    assert cwe_for_patterns(_matched("auto *h = reinterpret_cast<Header*>(buf);")) == "CWE-704"
