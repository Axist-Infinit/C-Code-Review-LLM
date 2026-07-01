"""Tests for the PR inline-annotation logic (pure helpers; no network)."""
import json

from annotate_pr import (
    load_findings,
    finding_body,
    diff_addressable_lines,
    match_file,
    commentable_line,
    build_review,
    review_summary,
    resolve_pr_number,
)

PATCH = """\
@@ -1,3 +1,5 @@
 int safe(void) {
+    char buf[8];
+    strcpy(buf, src);
     return 0;
 }
@@ -20,2 +22,3 @@ void other(void) {
+    gets(name);
 }
"""


def test_diff_addressable_lines_includes_added_and_context():
    lines = diff_addressable_lines(PATCH)
    # hunk 1 starts at new line 1: line1 ctx, 2 added, 3 added, 4 ctx, 5 ctx
    assert {1, 2, 3, 4, 5}.issubset(lines)
    # hunk 2 starts at new line 22: 22 added, 23 ctx
    assert {22, 23}.issubset(lines)


def test_diff_addressable_lines_empty_patch():
    assert diff_addressable_lines("") == set()
    assert diff_addressable_lines(None) == set()


def test_diff_no_newline_marker_is_not_a_phantom_line():
    patch = ("@@ -1,3 +1,3 @@\n"
             " int a;\n"
             " int b;\n"
             "+int c;\n"
             "\\ No newline at end of file\n")
    assert diff_addressable_lines(patch) == {1, 2, 3}


def test_diff_no_newline_marker_mid_patch_keeps_following_lines_aligned():
    # old side lacked a trailing newline: the marker follows the removed line,
    # and the added line after it must still be new-side line 2 (not 3)
    patch = ("@@ -1,2 +1,2 @@\n"
             " int keep;\n"
             "-old last\n"
             "\\ No newline at end of file\n"
             "+new last\n")
    assert diff_addressable_lines(patch) == {1, 2}


def test_finding_body_has_severity_cwe_and_fix():
    body = finding_body({"severity": "high", "cwe": "CWE-120",
                         "issue": "buffer overflow", "explanation": "unbounded copy",
                         "fix": "use strlcpy"})
    assert "CWE-120" in body
    assert "buffer overflow" in body
    assert "**Fix:** use strlcpy" in body
    assert "C-Code-Review-LLM" in body


def test_finding_body_renders_structured_narrative():
    body = finding_body({"severity": "critical", "cwe": "CWE-242",
                         "issue": "Stack buffer overflow",
                         "what_code_does": "reads input with gets()",
                         "what_could_go_wrong": "input longer than the buffer overflows it",
                         "vulnerability": "Stack buffer overflow",
                         "fix": "use fgets()"})
    assert "**What the code is doing:** reads input with gets()" in body
    assert "**What could go wrong:** input longer than the buffer overflows it" in body
    assert "**Vulnerability:** Stack buffer overflow" in body
    assert "**Fix:** use fgets()" in body


def test_match_file_exact_and_basename():
    addr = {"src/a.c": {3, 4}}
    key, lines = match_file("src/a.c", addr)
    assert key == "src/a.c" and lines == {3, 4}
    # leading ./ and different dir but same basename still resolves
    key2, lines2 = match_file("./build/a.c", addr)
    assert key2 == "src/a.c" and lines2 == {3, 4}


def test_commentable_line_picks_first_in_range():
    entry = {"start_line": 2, "end_line": 6}
    assert commentable_line(entry, {5, 6}) == 5
    assert commentable_line(entry, {99}) is None
    assert commentable_line({"start_line": None}, {1}) is None


def test_commentable_line_prefers_match_lines_over_range_start():
    entry = {"start_line": 2, "end_line": 10, "match_lines": [7, 9]}
    # 5 is the first in-range diff line, but the matched call is on 7
    assert commentable_line(entry, {5, 7, 9}) == 7
    # no match line in the diff -> fall back to the range scan
    assert commentable_line(entry, {5}) == 5
    # neither match lines nor range lines in the diff -> None
    assert commentable_line(entry, {99}) is None


def test_build_review_splits_inline_and_overflow():
    findings = [
        {"file": "src/a.c", "start_line": 2, "end_line": 3, "is_vulnerable": True,
         "issue": "strcpy", "cwe": "CWE-120", "severity": "high"},
        {"file": "src/a.c", "start_line": 50, "end_line": 50, "is_vulnerable": True,
         "issue": "off-diff", "cwe": "", "severity": "medium"},
    ]
    addressable = {"src/a.c": {2, 3}}
    comments, overflow = build_review(findings, addressable)
    assert len(comments) == 1 and len(overflow) == 1
    assert comments[0]["path"] == "src/a.c"
    assert comments[0]["line"] in (2, 3)
    assert comments[0]["side"] == "RIGHT"
    assert "strcpy" in comments[0]["body"]


def test_build_review_merges_findings_on_same_anchor_line():
    findings = [
        {"file": "src/a.c", "start_line": 2, "end_line": 2, "issue": "strcpy",
         "cwe": "CWE-120", "severity": "high"},
        {"file": "src/a.c", "start_line": 2, "end_line": 2, "issue": "gets",
         "cwe": "CWE-242", "severity": "critical"},
    ]
    comments, overflow = build_review(findings, {"src/a.c": {2}})
    assert overflow == []
    assert len(comments) == 1
    body = comments[0]["body"]
    assert "strcpy" in body and "gets" in body
    assert body.count("C-Code-Review-LLM") == 1        # one signature, not two


def test_build_review_anchors_on_match_line():
    findings = [{"file": "src/a.c", "start_line": 2, "end_line": 9,
                 "match_lines": [7], "issue": "system", "cwe": "CWE-78",
                 "severity": "high"}]
    comments, overflow = build_review(findings, {"src/a.c": {2, 3, 7}})
    assert overflow == []
    assert comments[0]["line"] == 7


def test_review_summary_lists_overflow():
    findings = [{"file": "a.c", "start_line": 5, "issue": "x", "cwe": "CWE-78"}]
    summary = review_summary(findings, n_inline=0, overflow=findings)
    assert "1 finding(s)" in summary
    assert "a.c:5" in summary
    assert "CWE-78" in summary


def test_load_findings_filters_clean(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"explanations": [
        {"file": "a.c", "is_vulnerable": True, "issue": "bad"},
        {"file": "b.c", "is_vulnerable": False, "issue": "fine"},
    ]}))
    assert len(load_findings(str(p), only_vulnerable=True)) == 1
    assert len(load_findings(str(p), only_vulnerable=False)) == 2


def test_resolve_pr_number_from_ref(monkeypatch):
    monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    assert resolve_pr_number() == 42
    assert resolve_pr_number("7") == 7        # explicit wins


def test_resolve_pr_number_from_event_payload(tmp_path, monkeypatch):
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps({"pull_request": {"number": 99}}))
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
    assert resolve_pr_number() == 99
