"""Guard the GitHub Action definition: valid YAML, composite, and every script
it invokes actually exists. Keeps action.yml from drifting away from the code.
"""
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTION = os.path.join(REPO, "action.yml")
WORKFLOW = os.path.join(REPO, "ci", "code-scan.yml")

yaml = pytest.importorskip("yaml", reason="PyYAML not installed")


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_action_is_composite_with_expected_inputs():
    spec = _load(ACTION)
    assert spec["runs"]["using"] == "composite"
    assert set(spec["inputs"]) >= {"source", "output", "lang", "only-vulnerable"}
    assert spec["outputs"]["sarif-file"]["value"]


def test_action_invokes_only_scripts_that_exist():
    steps = _load(ACTION)["runs"]["steps"]
    run_blocks = "\n".join(s.get("run", "") for s in steps)
    for script in ("heuristic_scan.py", "llm_explain.py", "to_sarif.py"):
        assert script in run_blocks, f"action.yml never runs {script}"
        assert os.path.exists(os.path.join(REPO, script)), f"missing {script}"


def test_example_workflow_requests_sarif_upload_permissions():
    wf = _load(WORKFLOW)
    # workflows put 'on' as a bare key; YAML may parse it as the boolean True
    assert ("on" in wf) or (True in wf)
    assert wf["permissions"]["security-events"] == "write"
    steps = wf["jobs"]["scan"]["steps"]
    assert any("upload-sarif" in str(s.get("uses", "")) for s in steps)
