"""End-to-end pipeline tests: classifier findings -> explain -> SARIF.

Two layers:
  * in-process, exercising run_explanations (heuristic + injected Ollama) wired
    straight into the SARIF converter, asserting on real content; and
  * a subprocess run of the actual llm_explain.py and to_sarif.py CLIs over the
    heuristic backend, so the wired-together entry points are covered without
    needing torch, a GPU, or an Ollama server.
"""
import json
import os
import subprocess
import sys

import pytest

import to_sarif
from llm_explain import run_explanations

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A small classifier-output document with one real bug and one benign function.
CLASSIFIER_FINDINGS = {
    "threshold": 0.5,
    "profile": "cpu",
    "findings": [
        {"file": "src/parser.c", "start_line": 10, "end_line": 14, "score": 0.96,
         "snippet": "void copy(char *s){\n  char dst[8];\n  strcpy(dst, s);\n}"},
        {"file": "src/util.c", "start_line": 3, "end_line": 5, "score": 0.55,
         "snippet": "int add(int a, int b){\n  return a + b;\n}"},
    ],
}


def _build_sarif(entries, only_vulnerable=False):
    """Mirror to_sarif.main's assembly so we can validate the doc in-process."""
    if only_vulnerable:
        entries = [e for e in entries if e.get("is_vulnerable") is not False]
    results = [to_sarif.finding_to_result(e) for e in entries]
    rules = {}
    for r in results:
        rules.setdefault(r["ruleId"], {"id": r["ruleId"],
                                       "shortDescription": {"text": r["ruleId"]}})
    return {
        "version": "2.1.0",
        "runs": [{"tool": {"driver": {"name": "C-Code-Review-LLM",
                                      "rules": list(rules.values())}},
                  "results": results}],
    }


def test_inprocess_heuristic_to_sarif():
    entries = run_explanations(CLASSIFIER_FINDINGS["findings"], backend="heuristic",
                               model=None, ollama_url="x", num_ctx=4096)
    assert len(entries) == 2
    strcpy_entry = next(e for e in entries if e["file"] == "src/parser.c")
    assert strcpy_entry["is_vulnerable"] is True
    assert "strcpy" in strcpy_entry["matched_patterns"]

    sarif = _build_sarif(entries)
    results = sarif["runs"][0]["results"]
    assert len(results) == 2
    # the high-severity strcpy finding maps to a SARIF error at the right location
    parser_res = next(r for r in results
                      if r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
                      == "src/parser.c")
    assert parser_res["level"] in ("error", "warning")
    region = parser_res["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] == 10 and region["endLine"] == 14


def test_inprocess_ollama_path_to_sarif_uses_cwe_ruleid():
    def fake_chat(url, model, snippet, file, lines, num_ctx):
        if "strcpy" in snippet:
            return {"is_vulnerable": True, "issue": "buffer overflow", "cwe": "CWE-120",
                    "severity": "critical", "explanation": "unbounded copy",
                    "fix": "use strlcpy"}
        return {"is_vulnerable": False, "issue": "none found", "cwe": "",
                "severity": "info", "explanation": "benign", "fix": ""}

    entries = run_explanations(CLASSIFIER_FINDINGS["findings"], backend="ollama",
                               model="qwen", ollama_url="u", num_ctx=8192,
                               workers=4, chat_fn=fake_chat)
    sarif = _build_sarif(entries, only_vulnerable=True)
    results = sarif["runs"][0]["results"]
    assert len(results) == 1                                  # benign one filtered out
    assert results[0]["ruleId"] == "CWE-120"                  # CWE drives the ruleId
    assert results[0]["level"] == "error"                     # critical -> error
    assert "use strlcpy" in results[0]["message"]["text"]     # fix carried into message
    rule_ids = {r["id"] for r in sarif["runs"][0]["tool"]["driver"]["rules"]}
    assert rule_ids == {"CWE-120"}


def test_cli_pipeline_explain_then_sarif(tmp_path):
    findings_path = tmp_path / "classifier_findings.json"
    findings_path.write_text(json.dumps(CLASSIFIER_FINDINGS))
    llm_out = tmp_path / "llm_findings.json"
    sarif_out = tmp_path / "findings.sarif"

    env = dict(os.environ, CCR_PROFILE="cpu")
    r1 = subprocess.run(
        [sys.executable, "llm_explain.py", str(findings_path),
         "--out", str(llm_out), "--backend", "heuristic"],
        cwd=REPO, env=env, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    assert llm_out.exists()
    explained = json.loads(llm_out.read_text())
    assert len(explained["explanations"]) == 2

    r2 = subprocess.run(
        [sys.executable, "to_sarif.py", str(llm_out), "-o", str(sarif_out),
         "--only-vulnerable"],
        cwd=REPO, env=env, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    sarif = json.loads(sarif_out.read_text())
    assert sarif["version"] == "2.1.0"
    runs = sarif["runs"]
    assert runs and runs[0]["tool"]["driver"]["name"] == "C-Code-Review-LLM"
    # strcpy function is vulnerable -> survives --only-vulnerable
    uris = [res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
            for res in runs[0]["results"]]
    assert "src/parser.c" in uris
