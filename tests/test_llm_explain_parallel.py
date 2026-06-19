"""Tests for the parallel explainer path (explain_finding / run_explanations).

A fake chat_fn stands in for the Ollama server so we can exercise the LLM path,
ordering guarantees, worker fan-out, and per-finding fallback deterministically
without a model or network.
"""
import threading

import pytest

from llm_explain import explain_finding, run_explanations


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
