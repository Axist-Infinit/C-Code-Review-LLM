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
    assert set(spec["inputs"]) >= {"source", "output", "lang", "only-vulnerable",
                                   "annotate", "github-token"}
    assert spec["outputs"]["sarif-file"]["value"]


def test_action_invokes_only_scripts_that_exist():
    steps = _load(ACTION)["runs"]["steps"]
    run_blocks = "\n".join(s.get("run", "") for s in steps)
    for script in ("heuristic_scan.py", "llm_explain.py", "to_sarif.py", "annotate_pr.py"):
        assert script in run_blocks, f"action.yml never runs {script}"
        assert os.path.exists(os.path.join(REPO, script)), f"missing {script}"


def test_annotate_step_is_guarded_by_input():
    steps = _load(ACTION)["runs"]["steps"]
    annotate = [s for s in steps if "annotate_pr.py" in s.get("run", "")]
    assert len(annotate) == 1
    assert annotate[0]["if"] == "${{ inputs.annotate == 'true' }}"


def test_tree_sitter_install_is_best_effort():
    """The action claims 'no network required': the pip install must not fail
    the run when the runner is offline (the brace-parser fallback covers it)."""
    steps = _load(ACTION)["runs"]["steps"]
    pip_steps = [s for s in steps if "pip install" in s.get("run", "")]
    assert len(pip_steps) == 1
    run = pip_steps[0]["run"]
    assert "||" in run, "pip install step has no failure fallback"
    assert "::notice::" in run, "fallback should surface a workflow notice"


def test_example_workflow_requests_sarif_upload_permissions():
    wf = _load(WORKFLOW)
    # workflows put 'on' as a bare key; YAML may parse it as the boolean True
    assert ("on" in wf) or (True in wf)
    assert wf["permissions"]["security-events"] == "write"
    steps = wf["jobs"]["scan"]["steps"]
    assert any("upload-sarif" in str(s.get("uses", "")) for s in steps)
