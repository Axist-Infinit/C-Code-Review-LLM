"""Tests for ccr.sh — the single interactive control menu.

The menu is the one entry point a human is expected to use (install → train →
tune → review), so these guard the things that silently break a TUI:

  * it starts, renders, and exits cleanly;
  * `--help` works and describes the lifecycle;
  * an unknown argument is refused instead of being ignored;
  * review.sh still works as the historical entry point;
  * no shell script re-introduces the expansion-produced assignment prefix that
    made the previous menu's surface-scan action a no-op.

Nothing here runs a scan, a model, or the network.
"""
import os
import re
import shutil
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CCR = os.path.join(_REPO, "ccr.sh")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _run(args, stdin="", timeout=180):
    return subprocess.run(
        ["bash", _CCR] + args,
        input=stdin, capture_output=True, text=True, cwd=_REPO, timeout=timeout,
    )


def test_ccr_is_executable_and_parses():
    assert os.path.isfile(_CCR)
    assert os.access(_CCR, os.X_OK), "ccr.sh must be chmod +x"
    res = subprocess.run(["bash", "-n", _CCR], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_help_describes_the_lifecycle():
    res = _run(["--help"])
    assert res.returncode == 0, res.stderr
    for stage in ("REVIEW", "SETUP", "TRAIN", "TUNE"):
        assert stage in res.stdout
    # The header block must stop at the first non-comment line.
    assert "set -uo pipefail" not in res.stdout


def test_unknown_argument_is_refused():
    res = _run(["--scan-everything"])
    assert res.returncode == 2
    assert "unexpected argument" in (res.stdout + res.stderr)


def test_menu_renders_and_quits():
    res = _run([], stdin="q\n")
    assert res.returncode == 0, res.stderr
    out = res.stdout
    for entry in ("Review the inbox", "Setup", "Train the classifier",
                  "Tune the reviewer LLM", "Settings"):
        assert entry in out, f"missing menu entry: {entry}"


def test_eof_exits_cleanly():
    # Ctrl-D at the prompt must not spin or traceback.
    res = _run([], stdin="")
    assert res.returncode == 0, res.stderr


def test_review_sh_delegates_to_the_menu():
    src = open(os.path.join(_REPO, "review.sh"), encoding="utf-8").read()
    assert "ccr.sh" in src and "exec" in src


# --- regression guard -------------------------------------------------------
# `FOO=1 ${X:+BAR=2} cmd` does NOT export BAR: bash recognises assignment
# prefixes BEFORE expansion, so the expanded word becomes the command name and
# the run dies with "BAR=2: command not found". The old review.sh shipped this
# in its surface-review action, so the action could never run at all.
_ASSIGN_PREFIX_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:[+-][A-Za-z_][A-Za-z0-9_]*=")


def _shell_scripts():
    out = []
    for dirpath, dirnames, filenames in os.walk(_REPO):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "hf_cache", "dist")]
        out += [os.path.join(dirpath, f) for f in filenames if f.endswith(".sh")]
    return sorted(out)


def test_no_expansion_produced_assignment_prefixes():
    offenders = []
    for path in _shell_scripts():
        with open(path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                if line.lstrip().startswith("#"):
                    continue
                if _ASSIGN_PREFIX_RE.search(line):
                    offenders.append(f"{os.path.relpath(path, _REPO)}:{n}: {line.strip()}")
    assert not offenders, (
        "expansion-produced assignment prefixes never reach the child process; "
        "export in a subshell or pass a flag instead:\n  " + "\n  ".join(offenders)
    )
