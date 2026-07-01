"""Tests for the parallel explainer path (explain_finding / run_explanations).

A fake chat_fn stands in for the Ollama server so we can exercise the LLM path,
ordering guarantees, worker fan-out, and per-finding fallback deterministically
without a model or network.
"""
import json
import os
import threading

import pytest

import llm_explain
from llm_explain import (checkpoint_path, explain_finding, run_explanations,
                         write_json_atomic)


def _finding(file, snippet, score):
    lines = snippet.splitlines()
    return {"file": file, "start_line": 1, "end_line": len(lines),
            "score": score, "snippet": snippet}


def test_explain_finding_heuristic_backend_flags_strcpy():
    f = _finding("a.c", "void f(char*s){char b[8];strcpy(b,s);}", 0.9)
    entry = explain_finding(f, backend="heuristic", model="m",
                            ollama_url="x", num_ctx=8192)
    assert entry["backend"] == "heuristic"
    assert entry["is_vulnerable"] is True
    assert "strcpy" in entry["matched_patterns"]


def test_explain_finding_ollama_path_uses_chat_fn():
    f = _finding("a.c", "int g(void){return 0;}", 0.7)

    def fake_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": True, "issue": "x", "cwe": "CWE-120",
                "severity": "high", "explanation": "why", "fix": "do y"}

    entry = explain_finding(f, backend="ollama", model="qwen", ollama_url="u",
                            num_ctx=8192, chat_fn=fake_chat)
    assert entry["backend"] == "ollama"
    assert entry["cwe"] == "CWE-120"
    assert entry["severity"] == "high"


def test_ollama_corroboration_floor_blocks_injection_downgrade():
    # classifier high + heuristic sees strcpy, but the (injected) LLM says safe.
    f = _finding("a.c", "void f(char*s){char b[8];strcpy(b,s);}", 0.95)

    def lying_chat(*a, **k):
        return {"is_vulnerable": False, "issue": "none", "cwe": "",
                "severity": "info", "explanation": "looks fine", "fix": ""}

    entry = explain_finding(f, backend="ollama", model="m", ollama_url="u",
                            num_ctx=8192, chat_fn=lying_chat)
    assert entry["is_vulnerable"] is True
    assert entry["severity"] == "medium"
    assert entry.get("downgrade_blocked") is True


def test_ollama_failure_falls_back_to_heuristic_per_finding():
    f = _finding("a.c", "void f(char*s){char b[8];strcpy(b,s);}", 0.9)

    def boom(*a, **k):
        raise OSError("connection refused")

    entry = explain_finding(f, backend="ollama", model="m", ollama_url="u",
                            num_ctx=8192, chat_fn=boom)
    assert entry["backend"] == "heuristic"          # fell back
    assert "connection refused" in entry["llm_error"]
    assert entry["is_vulnerable"] is True            # heuristic still caught strcpy


def test_run_explanations_preserves_order_under_parallelism():
    findings = [_finding(f"f{i}.c", f"int fn{i}(void){{return {i};}}", 0.6)
                for i in range(20)]

    def fake_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": False, "issue": file, "cwe": "", "severity": "low",
                "explanation": "", "fix": ""}

    out = run_explanations(findings, backend="ollama", model="m", ollama_url="u",
                           num_ctx=8192, workers=8, chat_fn=fake_chat)
    assert len(out) == len(findings)
    # output order must match input order despite as_completed scheduling
    assert [e["file"] for e in out] == [f"f{i}.c" for i in range(20)]
    assert [e["issue"] for e in out] == [f"f{i}.c" for i in range(20)]


def test_run_explanations_actually_uses_multiple_threads():
    findings = [_finding(f"f{i}.c", f"int fn{i}(void){{return {i};}}", 0.6)
                for i in range(6)]
    seen = set()
    lock = threading.Lock()
    barrier = threading.Barrier(3, timeout=5)

    def fake_chat(url, model, snippet, file, lines, num_ctx):
        with lock:
            seen.add(threading.get_ident())
        try:
            barrier.wait()  # only completes if >=3 threads run concurrently
        except threading.BrokenBarrierError:  # pragma: no cover
            pytest.fail("explainer did not run with enough concurrency")
        return {"is_vulnerable": False, "issue": "", "cwe": "", "severity": "low",
                "explanation": "", "fix": ""}

    run_explanations(findings, backend="ollama", model="m", ollama_url="u",
                     num_ctx=8192, workers=3, chat_fn=fake_chat)
    assert len(seen) >= 3


def test_progress_callback_invoked_once_per_finding():
    findings = [_finding(f"f{i}.c", "int x(void){return 0;}", 0.6) for i in range(5)]
    calls = []
    lock = threading.Lock()

    def progress(done, total, entry):
        with lock:
            calls.append((done, total))

    def fake_chat(*a, **k):
        return {"is_vulnerable": False, "issue": "", "cwe": "", "severity": "low",
                "explanation": "", "fix": ""}

    run_explanations(findings, backend="ollama", model="m", ollama_url="u",
                     num_ctx=8192, workers=4, chat_fn=fake_chat, progress=progress)
    assert len(calls) == 5
    assert sorted(d for d, _ in calls) == [1, 2, 3, 4, 5]


# --- incremental JSONL checkpoint + atomic final write --------------------------

def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_crash_mid_run_preserves_completed_entries_in_checkpoint(tmp_path):
    findings = [_finding(f"f{i}.c", f"int fn{i}(void){{return {i};}}", 0.6)
                for i in range(5)]
    out = str(tmp_path / "llm_findings.json")
    cp = checkpoint_path(out)
    calls = {"n": 0}

    def crashing_chat(url, model, snippet, file, lines, num_ctx):
        calls["n"] += 1
        if calls["n"] > 2:
            # RuntimeError is NOT in explain_finding's fallback tuple, so it
            # simulates a hard mid-run crash rather than a per-finding failure.
            raise RuntimeError("simulated crash")
        return {"is_vulnerable": False, "issue": file, "cwe": "", "severity": "low",
                "explanation": "", "fix": ""}

    with pytest.raises(RuntimeError):
        run_explanations(findings, backend="ollama", model="m", ollama_url="u",
                         num_ctx=8192, workers=1, chat_fn=crashing_chat,
                         checkpoint=cp)
    rows = _read_jsonl(cp)
    assert len(rows) == 2                      # the two completed entries survived
    assert [r["index"] for r in rows] == [0, 1]
    assert [r["entry"]["file"] for r in rows] == ["f0.c", "f1.c"]
    assert not os.path.exists(out)             # crash -> no (possibly torn) final JSON


def test_checkpoint_contains_every_entry_and_is_truncated_across_runs(tmp_path):
    findings = [_finding(f"f{i}.c", f"int fn{i}(void){{return {i};}}", 0.6)
                for i in range(6)]
    cp = str(tmp_path / "out.json.partial.jsonl")
    with open(cp, "w", encoding="utf-8") as fh:
        fh.write('{"stale": true}\n')  # leftover from a previous crashed run

    def fake_chat(url, model, snippet, file, lines, num_ctx):
        return {"is_vulnerable": False, "issue": file, "cwe": "", "severity": "low",
                "explanation": "", "fix": ""}

    run_explanations(findings, backend="ollama", model="m", ollama_url="u",
                     num_ctx=8192, workers=3, chat_fn=fake_chat, checkpoint=cp)
    rows = _read_jsonl(cp)
    assert len(rows) == 6                              # stale line gone
    assert sorted(r["index"] for r in rows) == list(range(6))
    assert {r["entry"]["file"] for r in rows} == {f"f{i}.c" for i in range(6)}


def test_checkpoint_written_for_heuristic_backend_too(tmp_path):
    findings = [_finding("a.c", "strcpy(d, s);", 0.9)]
    cp = str(tmp_path / "h.json.partial.jsonl")
    run_explanations(findings, backend="heuristic", model=None, ollama_url=None,
                     num_ctx=0, checkpoint=cp)
    rows = _read_jsonl(cp)
    assert len(rows) == 1
    assert rows[0]["entry"]["backend"] == "heuristic"


def test_write_json_atomic_replaces_and_leaves_no_temp(tmp_path):
    path = str(tmp_path / "nested" / "out.json")
    write_json_atomic(path, {"a": 1})
    write_json_atomic(path, {"a": 2})  # overwrite via os.replace
    assert json.load(open(path, encoding="utf-8")) == {"a": 2}
    assert os.listdir(os.path.dirname(path)) == ["out.json"]  # no .tmp leftover


def test_main_writes_final_json_and_removes_checkpoint(tmp_path, monkeypatch):
    inp = tmp_path / "classifier_findings.json"
    inp.write_text(json.dumps({"findings": [
        {"file": "a.c", "start_line": 1, "end_line": 1, "score": 0.9,
         "snippet": "gets(buf);", "match_lines": [1]},
        {"file": "b.c", "start_line": 2, "end_line": 2, "score": 0.4,
         "snippet": "int y;"},
    ]}), encoding="utf-8")
    out = tmp_path / "llm_findings.json"
    monkeypatch.setattr("sys.argv", [
        "llm_explain.py", str(inp), "--out", str(out),
        "--backend", "heuristic", "--profile", "cpu", "--top-k", "0",
    ])
    llm_explain.main()
    data = json.load(open(out, encoding="utf-8"))
    assert data["backend"] == "heuristic"
    assert len(data["explanations"]) == 2
    # C3: match_lines passed through to the output entry (sorted by score,
    # so the gets() finding is first).
    assert data["explanations"][0]["match_lines"] == [1]
    assert not os.path.exists(checkpoint_path(str(out)))  # cleaned on success
