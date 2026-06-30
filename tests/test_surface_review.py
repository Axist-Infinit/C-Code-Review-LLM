"""Tests for surface_review.py's deterministic logic (no model / network needed).

Covers the pure helpers that make the union finisher reliable: identifier
extraction, finding dedup, the topic-based consolidator (CWE canonicalisation +
CVE restore), anchor cleanup, the KB retrieval matcher, and pooling.
"""
import surface_review as sr


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
