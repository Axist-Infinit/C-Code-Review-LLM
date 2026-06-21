#!/usr/bin/env python3
"""Post scan findings as inline review comments on a pull request.

Complements the SARIF / code-scanning path: reads the explainer output
(llm_findings.json) and posts ONE pull-request review whose comments are pinned
to the exact lines, each with severity, CWE, explanation, and fix. GitHub only
accepts inline comments on lines that are part of the PR diff, so findings whose
line falls outside the diff are rolled into the review summary instead of being
dropped. Stdlib only (urllib) — no extra dependencies, just a GITHUB_TOKEN.

In GitHub Actions the context comes from the environment:
  GITHUB_TOKEN        - token with pull-requests: write
  GITHUB_REPOSITORY   - "owner/repo"
  GITHUB_REF          - "refs/pull/<n>/merge"  (or pass --pr / GITHUB_EVENT_PATH)
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

SEVERITY_EMOJI = {"critical": "🔴", "high": "🔴", "medium": "🟠",
                  "low": "🟡", "info": "🔵"}
SIGNATURE = "<sub>flagged by C-Code-Review-LLM</sub>"


# --- pure helpers (unit-tested; no network) ---------------------------------

def load_findings(path, only_vulnerable=True):
    """Load explainer/classifier findings, optionally dropping clean verdicts."""
    data = json.load(open(path, "r", encoding="utf-8"))
    entries = data.get("explanations") or data.get("findings") or []
    if only_vulnerable:
        entries = [e for e in entries if e.get("is_vulnerable") is not False]
    return entries


def finding_body(entry):
    """Markdown body for one inline review comment."""
    sev = str(entry.get("severity", "")).lower()
    emoji = SEVERITY_EMOJI.get(sev, "⚪")
    cwe = entry.get("cwe") or ""
    title = entry.get("issue") or "Potential vulnerability"
    head = f"{emoji} **{title}**" + (f" ({cwe})" if cwe else "")
    parts = [head]
    if entry.get("explanation"):
        parts.append(str(entry["explanation"]))
    if entry.get("fix"):
        parts.append(f"**Fix:** {entry['fix']}")
    parts.append(SIGNATURE)
    return "\n\n".join(parts)


def diff_addressable_lines(patch):
    """Set of new-file line numbers addressable for review comments in a unified
    diff patch (added + context lines on the RIGHT side; removed lines skipped)."""
    addressable = set()
    if not patch:
        return addressable
    new_ln = None
    for line in patch.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            new_ln = int(m.group(1))
            continue
        if new_ln is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            addressable.add(new_ln)
            new_ln += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass  # removed line: the new side does not advance
        else:
            addressable.add(new_ln)  # context line is part of the hunk
            new_ln += 1
    return addressable


def _norm(path):
    return path[2:] if path and path.startswith("./") else path


def match_file(path, addressable_by_file):
    """Resolve a finding's file against the PR's changed files. Returns
    (repo_relative_path, addressable_lines) — exact match, then suffix/basename."""
    n = _norm(path)
    if n in addressable_by_file:
        return n, addressable_by_file[n]
    base = os.path.basename(n or "")
    for k in addressable_by_file:
        if k == n or (n and k.endswith("/" + n)) or (k and n and n.endswith("/" + k)) \
                or os.path.basename(k) == base:
            return k, addressable_by_file[k]
    return n, set()


def commentable_line(entry, addressable):
    """First line within the finding's [start,end] range that is in the diff."""
    start = entry.get("start_line")
    if start is None:
        return None
    end = entry.get("end_line") or start
    for ln in range(int(start), int(end) + 1):
        if ln in addressable:
            return ln
    return None


def build_review(findings, addressable_by_file):
    """Split findings into inline review comments (on diff lines) and an overflow
    list (outside the diff). Returns (comments, overflow)."""
    comments, overflow = [], []
    for e in findings:
        key, addr = match_file(e.get("file"), addressable_by_file)
        line = commentable_line(e, addr)
        if line is not None:
            comments.append({"path": key, "line": line, "side": "RIGHT",
                             "body": finding_body(e)})
        else:
            overflow.append(e)
    return comments, overflow


def review_summary(findings, n_inline, overflow):
    """Markdown summary posted as the review body."""
    out = [f"## 🛡️ C-Code-Review-LLM — {len(findings)} finding(s)",
           f"{n_inline} shown inline on changed lines."]
    if overflow:
        out.append(f"\n<details><summary>{len(overflow)} finding(s) outside this "
                   f"PR's diff</summary>\n")
        for e in overflow:
            cwe = e.get("cwe") or ""
            out.append(f"- `{e.get('file')}:{e.get('start_line')}` — "
                       f"{e.get('issue', '')}" + (f" ({cwe})" if cwe else ""))
        out.append("\n</details>")
    return "\n".join(out)


# --- GitHub API (thin wrappers) ---------------------------------------------

def _gh_request(method, url, token, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "c-code-review-llm",
    })
    with urllib.request.urlopen(req) as r:
        body = r.read()
    return json.loads(body) if body else {}


def fetch_changed_files(owner, repo, pr, token):
    """Return {repo_relative_path: addressable_lines} for the PR (paginated)."""
    by_file = {}
    page = 1
    while True:
        url = (f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}/files"
               f"?per_page=100&page={page}")
        rows = _gh_request("GET", url, token)
        if not rows:
            break
        for f in rows:
            by_file[f["filename"]] = diff_addressable_lines(f.get("patch", ""))
        if len(rows) < 100:
            break
        page += 1
    return by_file


def post_review(owner, repo, pr, token, comments, summary, event="COMMENT"):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr}/reviews"
    payload = {"event": event, "body": summary, "comments": comments}
    return _gh_request("POST", url, token, payload)


def resolve_pr_number(explicit=None):
    if explicit:
        return int(explicit)
    ref = os.environ.get("GITHUB_REF", "")
    m = re.match(r"refs/pull/(\d+)/", ref)
    if m:
        return int(m.group(1))
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        ev = json.load(open(event_path, "r", encoding="utf-8"))
        num = ev.get("number") or (ev.get("pull_request") or {}).get("number")
        if num:
            return int(num)
    return None


def main():
    ap = argparse.ArgumentParser(description="Post findings as inline PR review comments")
    ap.add_argument("findings", help="llm_findings.json (or classifier_findings.json)")
    ap.add_argument("--pr", default=None, help="PR number (else from GITHUB_REF/event)")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"),
                    help="owner/repo (default: $GITHUB_REPOSITORY)")
    ap.add_argument("--event", default="COMMENT",
                    choices=["COMMENT", "REQUEST_CHANGES", "APPROVE"],
                    help="Review event type (default: COMMENT)")
    ap.add_argument("--all", action="store_true",
                    help="Include findings the explainer judged not vulnerable")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the review and print it; do not call the API")
    args = ap.parse_args()

    findings = load_findings(args.findings, only_vulnerable=not args.all)
    if not findings:
        print("[annotate] no findings to post")
        return

    if args.dry_run:
        # Build against an empty diff map so everything overflows; useful offline.
        comments, overflow = build_review(findings, {})
        print(review_summary(findings, len(comments), overflow))
        print(json.dumps({"comments": comments}, indent=2))
        return

    # No PR in context (e.g. a push build) -> skip quietly instead of failing,
    # so the same workflow step is safe on both push and pull_request events.
    pr = resolve_pr_number(args.pr)
    if not pr:
        print("[annotate] no pull request in context; skipping")
        return
    token = os.environ.get("GITHUB_TOKEN")
    if not token or not args.repo:
        sys.exit("[ERR] GITHUB_TOKEN and --repo/$GITHUB_REPOSITORY are required")
    owner, repo = args.repo.split("/", 1)

    addressable = fetch_changed_files(owner, repo, pr, token)
    comments, overflow = build_review(findings, addressable)
    summary = review_summary(findings, len(comments), overflow)
    try:
        post_review(owner, repo, pr, token, comments, summary, event=args.event)
    except urllib.error.HTTPError as e:
        sys.exit(f"[ERR] GitHub API {e.code}: {e.read().decode(errors='ignore')[:300]}")
    print(f"[OK] posted review on {args.repo}#{pr}: "
          f"{len(comments)} inline, {len(overflow)} in summary")


if __name__ == "__main__":
    main()
