"""Assert every shell script in the repo parses with `bash -n`.

This guards the hardware-portability / train-flow scripts (setup_machine.sh,
one_click_unlock_fetch_train_relock.sh, install_torch_nightly_for_new_gpu.sh,
env_doctor_cuda.sh, lib_torch_install.sh, train*.sh, ...) against syntax drift.
No GPU/network/training is executed — `bash -n` only parses.
"""
import glob
import os
import shutil
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _scripts():
    paths = sorted(glob.glob(os.path.join(_REPO, "*.sh")))
    paths += sorted(glob.glob(os.path.join(_REPO, "airgap", "*.sh")))
    paths += sorted(glob.glob(os.path.join(_REPO, "scripts", "**", "*.sh"), recursive=True))
    return paths


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", _scripts(), ids=lambda p: os.path.basename(p))
def test_bash_syntax(script):
    res = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
    assert res.returncode == 0, f"{script} failed bash -n:\n{res.stderr}"


def test_found_some_scripts():
    # sanity: make sure the glob actually matched the key scripts
    names = {os.path.basename(p) for p in _scripts()}
    for required in ("setup_machine.sh", "one_click_unlock_fetch_train_relock.sh",
                     "install_torch_nightly_for_new_gpu.sh", "env_doctor_cuda.sh",
                     "lib_torch_install.sh", "train.sh", "train_classifier.sh",
                     "build_airgap_bundle.sh", "install_offline.sh"):
        assert required in names, f"missing {required}"
