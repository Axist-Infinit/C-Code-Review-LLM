"""Tests for surface_review.py's deterministic logic (no model / network needed).

Covers the pure helpers that make the union finisher reliable: identifier
extraction, finding dedup, the topic-based consolidator (conservative CWE
canonicalisation + CVE restore), anchor cleanup, the KB retrieval matcher,
pooling, the critic acceptance gate, the pipeline-schema adapter, and the
fault-isolated main() sampling loop (Ollama monkeypatched).
"""
import json
import os
import sys
import urllib.error

import pytest

import compare_review as cr
import surface_review as sr
import to_sarif

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- identifier extraction -------------------------------------------------

def test_ident_skips_type_keywords():
    assert sr._ident("be32_t ciaddr") == "ciaddr"
    assert sr._ident("uint8_t hlen") == "hlen"
    assert sr._ident("dhcp_option_append(DHCPMessage *m, size_t size)") == "dhcp_option_append"
    assert sr._ident("#define DHCP_CLIENT_DONT_DESTROY(client)") == "dhcp_client_dont_destroy"


# --- dedup_findings --------------------------------------------------------

def test_dedup_collapses_same_anchor_and_cwe_keeping_richest():
    findings = [
        {"anchor": "dhcp_option_append(size, offset)", "cwe": "CWE-787", "failure_mode": "short"},
        {"anchor": "dhcp_option_append(DHCPMessage *m, size_t size)", "cwe": "CWE-787",
         "failure_mode": "a much longer and more detailed failure description"},
        {"anchor": "DHCP_CLIENT_DONT_DESTROY(client)", "cwe": "CWE-416", "failure_mode": "uaf"},
    ]
    out = sr.dedup_findings(findings)
    assert len(out) == 2  # the two appends collapse, the UAF stays distinct
    append = [f for f in out if "append" in f["anchor"]][0]
    assert "longer and more detailed" in append["failure_mode"]


# --- topic_consolidate -----------------------------------------------------

def test_topic_consolidate_canonicalises_cwe_and_restores_cve():
    f = {"title": "Unsigned Underflow in dhcp_option_append",
         "anchor": "dhcp_option_append(...)", "bug_class": "Heap buffer overflow",
         "cwe": "CWE-190", "failure_mode": "size - offset underflows", "cve_analog": ""}
    out = sr.topic_consolidate([f])
    assert len(out) == 1
    assert out[0]["cwe"] == "CWE-787"            # 190 -> canonical 787
    assert out[0]["cve_analog"] == "CVE-2018-15688"  # restored


def test_topic_consolidate_fixes_uaf_and_sign_cast_cwes():
    uaf = {"title": "Callback re-entrancy", "anchor": "DHCP_CLIENT_DONT_DESTROY",
           "bug_class": "Use-after-free", "cwe": "CWE-401", "failure_mode": "freed during callback"}
    sign = {"title": "Sign-Conversion Hygiene", "anchor": "#define DHCP_IP_SIZE",
            "bug_class": "sign", "cwe": "CWE-190", "failure_mode": "int32_t cast of size"}
    out = {f["title"]: f for f in sr.topic_consolidate([uaf, sign])}
    assert out["Callback re-entrancy"]["cwe"] == "CWE-416"
    assert out["Sign-Conversion Hygiene"]["cwe"] == "CWE-195"


def test_topic_consolidate_merges_duplicate_topics():
    # Two phrasings of hlen/chaddr and two of magic/endianness collapse to one each.
    findings = [
        {"title": "HW addr overflow", "anchor": "chaddr[16]", "bug_class": "", "cwe": "CWE-787",
         "failure_mode": "hlen exceeds chaddr"},
        {"title": "hlen overflow", "anchor": "uint8_t hlen", "bug_class": "", "cwe": "CWE-120",
         "failure_mode": "hardware address length unclamped"},
        {"title": "Magic cookie", "anchor": "be32_t magic", "bug_class": "", "cwe": "CWE-20",
         "failure_mode": "magic not validated"},
        {"title": "Endianness", "anchor": "be32_t xid", "bug_class": "", "cwe": "CWE-697",
         "failure_mode": "byte-swap missing"},
    ]
    out = sr.topic_consolidate(findings)
    assert len(out) == 2  # one length-vs-buffer, one validation-gate


# --- clean_anchors ---------------------------------------------------------

def test_clean_anchors_dedups_and_finding_wins():
    # Both ciaddr entries reduce to the same identifier (type keywords skipped).
    anchors = [
        {"anchor": "be32_t ciaddr", "disposition": "dismissed", "reason": "addr"},
        {"anchor": "ciaddr field", "disposition": "finding", "reason": ""},
        {"anchor": "options[0]", "disposition": "finding", "reason": ""},
    ]
    out = sr.clean_anchors(anchors)
    by = {sr._ident(a["anchor"]): a for a in out}
    assert len(out) == 2                      # both ciaddr entries collapse
    assert by["ciaddr"]["disposition"] == "finding"  # finding beats dismissed


# --- KB retrieval ----------------------------------------------------------

def test_match_kb_triggers_on_relevant_tokens_only():
    kb = sr.load_kb()  # ships with the repo
    assert kb, "surface_kb.json should load with entries"
    hits = {e["id"] for e in sr.match_kb("int dhcp_option_append(DHCPMessage *m);", kb)}
    assert "dhcp-option-append-cve" in hits
    fqdn = {e["id"] for e in sr.match_kb("#define DHCP_MAX_FQDN_LENGTH 255", kb)}
    assert "dhcp-fqdn-dns-labels" in fqdn
    assert sr.match_kb("int add(int a, int b) { return a + b; }", kb) == []


def test_kb_notes_block_empty_when_no_match():
    assert sr.kb_notes_block([]) == ""
    block = sr.kb_notes_block([{"id": "x", "note": "a fact"}])
    assert "a fact" in block and "REFERENCE NOTES" in block


# --- pool_samples ----------------------------------------------------------

def test_number_lines_prefixes_each_line():
    assert sr.number_lines("a();\nb();") == "   1| a();\n   2| b();"


def test_render_walkthrough_and_per_finding_code():
    rv = {"subsystem": "X",
          "what_the_code_does": [
              {"file": "snip.c", "lines": "1-2", "code": "char d[8];\nstrcpy(d, s);",
               "explanation": "declares an 8-byte buffer then copies into it."}],
          "what_could_go_wrong": [
              {"file": "snip.c", "lines": "2", "code": "strcpy(d, s);",
               "explanation": "an over-long s overflows d."}],
          "reviewed_anchors": [],
          "findings": [{"title": "T", "anchor": "strcpy", "file": "snip.c", "line": "2",
                        "code": "strcpy(d, s);", "cwe": "CWE-120", "severity": "high",
                        "failure_mode": "overflow"}]}
    md = sr.render_markdown(rv, ["snip.c"])
    assert "## What is this code doing?" in md and "**snip.c:1-2**" in md  # file-attributed
    assert "declares an 8-byte buffer" in md
    assert "## What could go wrong?" in md and "an over-long s overflows d" in md
    assert "snip.c:2" in md                          # per-finding location citation
    assert "```c\nstrcpy(d, s);\n```" in md          # quoted code snippet


def test_canonicalize_cwes_fixes_direction_without_merging():
    findings = [
        {"title": "append underflow", "anchor": "dhcp_option_append", "cwe": "CWE-190",
         "failure_mode": "size - offset underflows", "cve_analog": ""},
        {"title": "uaf", "anchor": "DHCP_CLIENT_DONT_DESTROY", "cwe": "CWE-401",
         "failure_mode": "freed during callback"},
    ]
    out = sr.canonicalize_cwes(findings)
    assert len(out) == 2                       # no merging
    assert out[0]["cwe"] == "CWE-787" and out[0]["cve_analog"] == "CVE-2018-15688"
    assert out[1]["cwe"] == "CWE-416"


def test_overload_param_in_append_signature_is_not_mis_bucketed():
    # The dhcp_option_append signature contains a `uint8_t overload` parameter;
    # it must canonicalise to the append-overflow CWE (787), not overload (674).
    append = {"title": "Buffer Overflow in dhcp_option_append",
              "anchor": "int dhcp_option_append(DHCPMessage *m, size_t size, size_t *offset, "
                        "uint8_t overload, uint8_t code, size_t optlen, const void *optval)",
              "bug_class": "Heap buffer overflow (unsigned underflow)", "cwe": "CWE-190",
              "failure_mode": "optlen exceeds remaining buffer space"}
    overload = {"title": "Overload re-parse", "anchor": "enum { DHCP_OVERLOAD_FILE = 1 }",
                "bug_class": "recursion", "cwe": "CWE-674", "failure_mode": "sname re-parsed twice"}
    out = sr.canonicalize_cwes([append, overload])
    assert out[0]["cwe"] == "CWE-787"          # append, not 674
    assert out[1]["cwe"] == "CWE-674"          # genuine overload still 674


def test_render_walkthrough_tolerates_plain_string():
    md = sr.render_markdown({"what_the_code_does": "a plain paragraph.",
                             "reviewed_anchors": [], "findings": []}, ["x"])
    assert "a plain paragraph." in md


def test_pool_samples_carries_narratives():
    walkthrough = [{"lines": "3", "code": "x();", "explanation": "does X"}]
    s = {"subsystem": "DHCP", "what_the_code_does": walkthrough,
         "what_could_go_wrong": [{"lines": "5", "code": "y();", "explanation": "risk"}],
         "reviewed_anchors": [], "findings": [], "audit_checklist": []}
    pooled, _ = sr.pool_samples([("m#1", s)])
    assert pooled["what_the_code_does"] == walkthrough
    assert pooled["what_could_go_wrong"][0]["lines"] == "5"


def test_pool_samples_dedups_findings_and_reports_max_single():
    s1 = {"subsystem": "DHCP", "reviewed_anchors": [
              {"anchor": "options[0]", "disposition": "finding"}],
          "findings": [{"anchor": "dhcp_option_append(a)", "cwe": "CWE-787", "failure_mode": "x"}],
          "audit_checklist": ["check a"]}
    s2 = {"reviewed_anchors": [{"anchor": "uint8_t options[0]", "disposition": "dismissed"}],
          "findings": [
              {"anchor": "dhcp_option_append(DHCPMessage *m)", "cwe": "CWE-787", "failure_mode": "y"},
              {"anchor": "log_dhcp_client_errno", "cwe": "CWE-476", "failure_mode": "null"}],
          "audit_checklist": ["check b"]}
    pooled, max_single = sr.pool_samples([("m#1", s1), ("m#2", s2)])
    # the two appends dedup; the null-deref stays -> 2 distinct findings
    assert len(pooled["findings"]) == 2
    assert max_single == 2  # s2 had the most findings (2)
    assert pooled["subsystem"] == "DHCP"


# --- conservative canonicalisation: verified mis-bucket regressions ----------

def test_option_space_text_does_not_route_to_overload_reparse():
    f = {"title": "Unsigned underflow in dhcp_option_parse",
         "anchor": "int dhcp_option_parse(DHCPMessage *message, size_t len, ...)",
         "bug_class": "Out-of-bounds read", "cwe": "CWE-191",
         "failure_mode": "option space computed as (len - 240) underflows for short packets"}
    out = sr.canonicalize_cwes([f])
    assert out[0]["cwe"] == "CWE-191"      # acceptable for the parse topic; kept
    assert out[0]["cwe"] != "CWE-674"      # never the overload-reparse canonical


def test_parse_topic_accepts_125_and_overrides_wrong_direction():
    kept = {"title": "Over-read in option parsing", "anchor": "dhcp_option_parse",
            "cwe": "CWE-125", "failure_mode": "advances by len without bounds check"}
    wrong = {"title": "Overflow in option parsing", "anchor": "dhcp_option_parse",
             "cwe": "CWE-787", "failure_mode": "advances by len without bounds check"}
    out = sr.canonicalize_cwes([kept, wrong])
    assert out[0]["cwe"] == "CWE-125"      # already acceptable: untouched
    assert out[1]["cwe"] == "CWE-125"      # a write CWE on the read path: corrected


def test_log_macro_format_string_keeps_cwe_134():
    f = {"title": "Format string in logging macro",
         "anchor": "log_dhcp_client(client, fmt, ...)",
         "bug_class": "Format string", "cwe": "CWE-134",
         "failure_mode": "attacker-controlled fmt reaches the log_dhcp logging "
                         "macro as the format argument"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-134"


def test_log_macro_null_deref_still_gets_476():
    f = {"title": "Crash in log_dhcp_client_errno", "anchor": "log_dhcp_client_errno",
         "cwe": "CWE-20",
         "failure_mode": "the macro dereferences client->xid while client may be NULL"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-476"


def test_bare_checksum_text_keeps_model_cwe():
    f = {"title": "Weak checksum", "anchor": "crc16",
         "cwe": "CWE-328", "failure_mode": "the checksum is not cryptographic; forgeable"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-328"


def test_checksum_overread_still_gets_125():
    f = {"title": "Checksum over-read",
         "anchor": "uint16_t dhcp_packet_checksum(uint8_t *buf, size_t len)",
         "cwe": "CWE-120",
         "failure_mode": "a caller passing a length larger than the buffer makes "
                         "the checksum loop read past the end"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-125"


def test_bare_int32_text_keeps_model_cwe():
    f = {"title": "int32_t counter wrap", "anchor": "int32_t seq",
         "cwe": "CWE-190", "failure_mode": "sequence counter wraps past INT32_MAX"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-190"


def test_int32_size_macro_cast_still_gets_195():
    f = {"title": "Size macro sign issue", "anchor": "#define DHCP_MESSAGE_SIZE",
         "cwe": "CWE-190", "failure_mode": "the int32_t cast makes size comparisons signed"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-195"


def test_acceptable_cwe_in_topic_set_is_preserved():
    f = {"title": "hlen drives copy into chaddr", "anchor": "uint8_t hlen",
         "cwe": "CWE-787", "failure_mode": "hlen up to 255 copies into 16-byte chaddr"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-787"  # in {120, 787}: kept


def test_missing_cwe_gets_topic_canonical():
    f = {"title": "hlen vs chaddr", "anchor": "uint8_t hlen",
         "cwe": "", "failure_mode": "hlen exceeds chaddr"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-120"


def test_no_topic_finding_is_untouched():
    f = {"title": "Race on shared config", "anchor": "config_reload",
         "cwe": "CWE-362", "failure_mode": "concurrent reload tears the struct"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-362"


def test_dereference_prose_alone_does_not_hijack_to_null_deref():
    # Verified regression: 'dereference' is near-universal in memory-safety
    # failure_mode prose; without null/NULL/nullptr context the finding must
    # keep its own topic and CWE instead of being rewritten to CWE-476.
    oob = {"title": "Out-of-bounds read via ihl", "anchor": "struct iphdr",
           "cwe": "CWE-125",
           "failure_mode": "a crafted ihl makes the parser dereference memory "
                           "past the ip header"}
    dangling = {"title": "Dangling pointer use of freed buffer", "anchor": "buf",
                "cwe": "CWE-416",
                "failure_mode": "the dangling pointer is dereferenced after "
                                "the buffer is freed"}
    out = sr.canonicalize_cwes([oob, dangling])
    assert out[0]["cwe"] == "CWE-125"      # packed-offset topic, not null-deref
    assert out[1]["cwe"] == "CWE-416"      # no topic: untouched, not 476
    # and the two DISTINCT findings must not merge into one null-deref bucket
    assert len(sr.topic_consolidate([oob, dangling])) == 2


def test_deref_with_null_context_still_routes_to_476():
    # The tightened token still catches genuine null-deref prose that lacks
    # the literal 'null pointer' / '->xid' phrasings.
    f = {"title": "Crash in log macro", "anchor": "log_dhcp_client_errno",
         "cwe": "CWE-20",
         "failure_mode": "the macro dereferences the client handle while it "
                         "may be NULL"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-476"


def test_append_mention_with_other_bug_class_keeps_cwe_and_gets_no_cve():
    # Verified regression: a null-deref finding that merely mentions
    # dhcp_option_append was rewritten to CWE-787 with CVE-2018-15688 attached.
    nullf = {"title": "NULL pointer dereference in dhcp_option_append",
             "anchor": "dhcp_option_append", "cwe": "CWE-476",
             "failure_mode": "message may be NULL when appended"}
    out = sr.canonicalize_cwes([nullf])
    assert out[0]["cwe"] == "CWE-476"          # not rebadged to CWE-787
    assert not out[0].get("cve_analog")        # no fabricated CVE
    # multi-sample mode: it must not merge with the real overflow finding
    real = {"title": "Unsigned Underflow in dhcp_option_append",
            "anchor": "dhcp_option_append(...)", "bug_class": "Heap buffer overflow",
            "cwe": "CWE-190", "failure_mode": "size - offset underflows",
            "cve_analog": ""}
    merged = sr.topic_consolidate([nullf, real])
    assert len(merged) == 2
    by_cwe = {f["cwe"]: f for f in merged}
    assert by_cwe["CWE-787"]["cve_analog"] == "CVE-2018-15688"  # legit case intact
    assert not by_cwe["CWE-476"].get("cve_analog")


def test_append_cve_attached_only_on_actual_override():
    # The CVE analog is attached only when the CWE was actually overridden
    # into CWE-787; an already-acceptable CWE keeps its (empty) cve_analog.
    kept = {"title": "Heap overflow in dhcp_option_append",
            "anchor": "dhcp_option_append", "cwe": "CWE-787",
            "failure_mode": "optlen exceeds the remaining buffer space",
            "cve_analog": ""}
    out = sr.canonicalize_cwes([kept])
    assert out[0]["cwe"] == "CWE-787"
    assert out[0]["cve_analog"] == ""


def test_parse_topic_keeps_kb_mandated_cwe_190_for_offset_wrap():
    # Verified regression: surface_kb.json's c-tlv-length-parse note, case (c),
    # instructs CWE-190 for 'offset + len' wrap — the parse-overread accept set
    # must not clobber it to CWE-125.
    f = {"title": "Integer overflow in option parsing", "anchor": "dhcp_option_parse",
         "cwe": "CWE-190",
         "failure_mode": "offset + len wraps past the end pointer during "
                         "option parsing"}
    assert sr.canonicalize_cwes([f])[0]["cwe"] == "CWE-190"


# --- generic KB entries ------------------------------------------------------

def test_kb_generic_entries_trigger_on_cpp_demo():
    kb = sr.load_kb()
    with open(os.path.join(ROOT, "playground", "vuln_demo.cpp"), encoding="utf-8") as fh:
        code = fh.read()
    hits = {e["id"] for e in sr.match_kb(code, kb)}
    assert {"c-unbounded-string-copy", "c-command-injection",
            "cpp-reinterpret-cast", "cpp-cstr-lifetime", "c-format-string"} <= hits


def test_kb_generic_entries_trigger_on_c_demo():
    kb = sr.load_kb()
    with open(os.path.join(ROOT, "playground", "vuln_demo.c"), encoding="utf-8") as fh:
        code = fh.read()
    hits = {e["id"] for e in sr.match_kb(code, kb)}
    assert "c-unbounded-string-copy" in hits


def test_kb_dhcp_entries_still_trigger_on_headers():
    kb = sr.load_kb()
    code = ""
    for name in ("dhcp-internal.h", "dhcp-protocol.h"):
        with open(os.path.join(ROOT, "playground", name), encoding="utf-8") as fh:
            code += fh.read()
    hits = {e["id"] for e in sr.match_kb(code, kb)}
    assert {"dhcp-option-append-cve", "dhcp-option-parse-underflow", "dhcp-overload",
            "dhcp-fqdn-dns-labels", "dhcppacket-ihl-offsets", "dhcp-endianness-magic",
            "bootp-hlen-chaddr"} <= hits


# --- critic acceptance gate --------------------------------------------------

def test_critique_that_strips_hallucination_is_accepted_with_coverage():
    draft = {"reviewed_anchors": [{"anchor": "uint8_t hlen", "disposition": "finding"},
                                  {"anchor": "chaddr[16]", "disposition": "finding"}],
             "findings": [{"anchor": "uint8_t hlen"},
                          {"anchor": "invented_internal_subtraction"}]}
    critique = {"reviewed_anchors": [{"anchor": "uint8_t hlen", "disposition": "finding"},
                                     {"anchor": "chaddr[16]", "disposition": "dismissed"}],
                "findings": [{"anchor": "uint8_t hlen"}]}
    assert sr.critique_acceptable(draft, critique)


def test_critique_with_fewer_anchors_is_rejected():
    draft = {"reviewed_anchors": [{"anchor": "uint8_t hlen"}, {"anchor": "chaddr[16]"}],
             "findings": [{"anchor": "uint8_t hlen"}, {"anchor": "chaddr[16]"}]}
    critique = {"reviewed_anchors": [{"anchor": "uint8_t hlen"}],
                "findings": [{"anchor": "uint8_t hlen"}]}
    assert not sr.critique_acceptable(draft, critique)
    # never accept fewer anchors, even when the finding count grows
    grown = {"reviewed_anchors": [{"anchor": "uint8_t hlen"}],
             "findings": [{"anchor": "a"}, {"anchor": "b"}, {"anchor": "c"}]}
    assert not sr.critique_acceptable(draft, grown)


def test_critique_that_adds_findings_is_accepted():
    draft = {"reviewed_anchors": [{"anchor": "uint8_t hlen"}],
             "findings": [{"anchor": "uint8_t hlen"}]}
    critique = {"reviewed_anchors": [{"anchor": "uint8_t hlen"},
                                     {"anchor": "options[0]"}],
                "findings": [{"anchor": "uint8_t hlen"}, {"anchor": "options[0]"}]}
    assert sr.critique_acceptable(draft, critique)


def test_empty_critique_is_rejected():
    assert not sr.critique_acceptable({"findings": [], "reviewed_anchors": []}, {})


def test_anchorless_draft_falls_back_to_count_gate():
    # Verified regression: a draft with NO reviewed_anchors (missing or empty)
    # made empty-vs-empty coverage vacuously true, so a truncated critic reply
    # {"findings": []} wiped all pass-1 findings. Must fall back to the count
    # gate instead.
    draft = {"findings": [{"anchor": "a"}, {"anchor": "b"}, {"anchor": "c"}]}
    assert not sr.critique_acceptable(draft, {"findings": []})
    assert not sr.critique_acceptable({"reviewed_anchors": [], **draft},
                                      {"findings": []})
    # the count gate still accepts a critique that keeps (or adds) findings
    kept = {"findings": [{"anchor": "a"}, {"anchor": "b"}, {"anchor": "c"}]}
    assert sr.critique_acceptable(draft, kept)


def test_non_dict_critique_is_rejected():
    # format:"json" guarantees valid JSON, not a JSON object.
    assert not sr.critique_acceptable({"findings": [{"anchor": "a"}]}, [])
    assert not sr.critique_acceptable({"findings": [{"anchor": "a"}]},
                                      [{"anchor": "a"}])


# --- pipeline-schema adapter (C4) --------------------------------------------

def _surface_review_fixture():
    return {"findings": [
        {"title": "hlen overflow", "file": "playground/dhcp-protocol.h", "line": "18",
         "cwe": "CWE-120", "severity": "High", "bug_class": "Buffer copy without size check",
         "where_it_lives": "implementation (file not provided)",
         "invariant": "clamp hlen to sizeof(chaddr)",
         "failure_mode": "hlen up to 255 drives a copy into 16-byte chaddr",
         "what_to_confirm": "check the memcpy bound in the client", "cve_analog": ""},
        {"title": "range finding", "file": "a.h", "line": "12-15",
         "cwe": "CWE-125", "severity": "moderate"},
        {"title": "list finding", "file": "a.h", "line": "12, 14",
         "cwe": "", "severity": "bogus"},
        {"title": "junk line", "file": "a.h", "line": "twelve", "severity": ""},
    ]}


def test_findings_to_pipeline_maps_schema():
    out = sr.findings_to_pipeline(_surface_review_fixture())
    first = out[0]
    assert first["file"] == "playground/dhcp-protocol.h"
    assert first["start_line"] == 18 and first["end_line"] == 18
    assert first["match_lines"] == [18]
    assert first["score"] == 1.0
    assert first["issue"] == "hlen overflow"
    assert first["cwe"] == "CWE-120" and first["bug_class"] == "Buffer copy without size check"
    assert first["severity"] == "high"
    assert "clamp hlen" in first["what_code_does"]
    assert "implementation (file not provided)" in first["what_code_does"]
    assert first["what_could_go_wrong"].startswith("hlen up to 255")
    assert first["vulnerability"] == "Buffer copy without size check"
    assert first["fix"] == "check the memcpy bound in the client"
    assert first["explainer_backend"] == "surface"


def test_findings_to_pipeline_line_parsing_variants():
    out = sr.findings_to_pipeline(_surface_review_fixture())
    rng, lst, junk = out[1], out[2], out[3]
    assert (rng["start_line"], rng["end_line"], rng["match_lines"]) == (12, 15, [12, 15])
    assert rng["severity"] == "medium"           # 'moderate' normalised
    assert (lst["start_line"], lst["end_line"], lst["match_lines"]) == (12, 14, [12, 14])
    assert lst["severity"] == "medium"           # unknown value defaults to medium
    assert (junk["start_line"], junk["end_line"]) == (1, 1)
    assert "match_lines" not in junk             # unparsable line: no fake anchors


def test_findings_to_pipeline_clamps_line_zero_to_one():
    # Verified regression: line "0" (0-based model output) yielded
    # {start_line: 0, end_line: 0, match_lines: [0]}, violating the absolute
    # 1-based match_lines contract and producing SARIF region startLine 0.
    review = {"findings": [
        {"title": "zero", "file": "inc/a.h", "line": "0", "severity": "high"},
        {"title": "zero range", "file": "inc/a.h", "line": "0-0", "severity": "low"},
        {"title": "zero mixed", "file": "inc/a.h", "line": "0-7", "severity": "low"},
    ]}
    out = sr.findings_to_pipeline(review)
    zero, zrange, zmixed = out
    for entry in (zero, zrange):
        assert entry["start_line"] == 1 and entry["end_line"] == 1
        assert entry["match_lines"] == [1]
    assert (zmixed["start_line"], zmixed["end_line"]) == (1, 7)
    assert zmixed["match_lines"] == [1, 7]
    region = to_sarif.finding_to_result(zero)["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] >= 1 and region["endLine"] >= region["startLine"]


def test_pipeline_findings_are_accepted_by_to_sarif():
    entries = sr.findings_to_pipeline(_surface_review_fixture())
    res = to_sarif.finding_to_result(entries[0])
    region = res["locations"][0]["physicalLocation"]["region"]
    assert region == {"startLine": 18, "endLine": 18}
    assert res["ruleId"] == "CWE-120"
    assert res["level"] == "error"               # high -> error
    assert "hlen overflow" in res["message"]["text"]
    assert "What could go wrong" in res["message"]["text"]
    junk = to_sarif.finding_to_result(entries[3])
    assert junk["locations"][0]["physicalLocation"]["region"]["startLine"] == 1


# --- main(): per-sample fault isolation + output wiring (Ollama mocked) ------

_GOOD_REVIEW = {
    "subsystem": "demo", "provenance": "test", "trust_boundary": "argv",
    "summary": "one real finding",
    "reviewed_anchors": [
        {"anchor": "be32_t ciaddr", "disposition": "dismissed", "reason": "addr"},
        {"anchor": "ciaddr field", "disposition": "finding", "reason": ""},
        {"anchor": "strcpy call", "disposition": "finding", "reason": ""},
    ],
    "findings": [{"title": "unbounded copy", "anchor": "strcpy call",
                  "file": "x.c", "line": "3", "cwe": "CWE-120", "severity": "high",
                  "bug_class": "buffer copy", "failure_mode": "overflow",
                  "invariant": "bound the copy", "what_to_confirm": "check dst size",
                  "where_it_lives": "x.c", "cve_analog": ""}],
    "audit_checklist": ["check strcpy"],
}


def _run_main(monkeypatch, tmp_path, argv_extra, fake_review):
    src = tmp_path / "x.c"
    src.write_text("void f(char *s) { char b[8]; strcpy(b, s); }\n")
    out_json = tmp_path / "review.json"
    out_pipe = tmp_path / "pipeline.json"
    monkeypatch.setattr(sr, "ollama_review", fake_review)
    monkeypatch.setattr(sys, "argv",
                        ["surface_review.py", str(src), "--no-critic", "--no-kb",
                         "--profile", "cpu", "--json", str(out_json),
                         "--pipeline-out", str(out_pipe), "--md", str(tmp_path / "r.md")]
                        + argv_extra)
    sr.main()
    return out_json, out_pipe


def test_main_skips_failed_samples_and_pools_the_rest(monkeypatch, tmp_path, capsys):
    def fake(url, model, context, num_ctx, extra_system="", temperature=0.3, timeout=1800):
        if model == "bad":
            raise urllib.error.URLError("connection refused")
        return json.loads(json.dumps(_GOOD_REVIEW))
    out_json, out_pipe = _run_main(monkeypatch, tmp_path,
                                   ["--models", "bad,good", "--samples", "2"], fake)
    printed = capsys.readouterr().out
    assert "[sample bad#1] FAILED" in printed and "skipping" in printed
    review = json.loads(out_json.read_text())
    assert len(review["findings"]) == 1
    assert review["findings"][0]["cwe"] == "CWE-120"


def test_main_skips_sample_whose_reply_is_non_dict_json(monkeypatch, tmp_path, capsys):
    # Verified regression: Ollama's format:"json" guarantees valid JSON, not a
    # JSON OBJECT; a reply parsing to [] escaped the per-sample try/except and
    # crashed the whole multi-model run with AttributeError. It must be warned
    # about and skipped like any other failed sample.
    def fake(url, model, context, num_ctx, extra_system="", temperature=0.3, timeout=1800):
        if model == "weird":
            return []          # valid JSON, but not a JSON object
        return json.loads(json.dumps(_GOOD_REVIEW))
    out_json, _ = _run_main(monkeypatch, tmp_path, ["--models", "weird,good"], fake)
    printed = capsys.readouterr().out
    assert "[sample weird#1] FAILED" in printed and "skipping" in printed
    review = json.loads(out_json.read_text())
    assert len(review["findings"]) == 1
    assert review["findings"][0]["cwe"] == "CWE-120"


def test_main_exits_nonzero_only_when_all_samples_fail(monkeypatch, tmp_path):
    def fake(url, model, context, num_ctx, extra_system="", temperature=0.3, timeout=1800):
        raise urllib.error.URLError("connection refused")
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, tmp_path, ["--models", "bad1,bad2"], fake)
    assert "all" in str(exc.value)


def test_main_dedupes_anchors_and_writes_pipeline_findings(monkeypatch, tmp_path):
    def fake(url, model, context, num_ctx, extra_system="", temperature=0.3, timeout=1800):
        return json.loads(json.dumps(_GOOD_REVIEW))
    out_json, out_pipe = _run_main(monkeypatch, tmp_path, [], fake)
    review = json.loads(out_json.read_text())
    # clean_anchors is wired into the single-run path: the two ciaddr entries
    # collapse and the 'finding' disposition wins.
    ciaddr = [a for a in review["reviewed_anchors"] if "ciaddr" in a["anchor"]]
    assert len(ciaddr) == 1 and ciaddr[0]["disposition"] == "finding"
    pipe = json.loads(out_pipe.read_text())
    assert pipe["findings"][0]["issue"] == "unbounded copy"
    assert pipe["findings"][0]["start_line"] == 3
    assert pipe["findings"][0]["explainer_backend"] == "surface"
    assert to_sarif.finding_to_result(pipe["findings"][0])["ruleId"] == "CWE-120"


# --- compare_review symmetry --------------------------------------------------

def test_parse_claude_reply_gets_same_canonicalisation_as_local():
    reply = json.dumps({"subsystem": "dhcp", "findings": [
        {"title": "append underflow", "anchor": "dhcp_option_append", "cwe": "CWE-190",
         "failure_mode": "size - offset underflows", "cve_analog": ""},
        {"title": "hlen copy", "anchor": "uint8_t hlen", "cwe": "CWE-787",
         "failure_mode": "hlen exceeds chaddr"}]})
    rev = cr.parse_claude_reply(reply)
    assert rev["findings"][0]["cwe"] == "CWE-787"
    assert rev["findings"][0]["cve_analog"] == "CVE-2018-15688"
    assert rev["findings"][1]["cwe"] == "CWE-787"   # acceptable CWE preserved


def test_parse_claude_reply_handles_fences_and_bad_json():
    fenced = cr.parse_claude_reply("```json\n{\"subsystem\": \"x\", \"findings\": []}\n```")
    assert fenced["subsystem"] == "x" and fenced["findings"] == []
    bad = cr.parse_claude_reply("this is not json")
    assert bad["findings"] == [] and bad["_raw"] == "this is not json"


def test_print_comparison_honors_out_dir(capsys):
    local = {"subsystem": "x", "findings": [{"title": "t", "severity": "high",
                                             "cwe": "CWE-120"}]}
    claude = {"subsystem": "x", "findings": []}
    cr.print_comparison(local, claude, "custom/outdir")
    printed = capsys.readouterr().out
    assert "custom/outdir/" in printed
    assert "scan_out/compare" not in printed
