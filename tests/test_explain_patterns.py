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
