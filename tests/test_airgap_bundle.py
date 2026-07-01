"""Torch-free tests for the airgap tooling.

Covers:
  * bash -n on build_airgap_bundle.sh / airgap/install_offline.sh (+ --help)
  * install_offline.sh pure verification paths against a fabricated mini-bundle
    (SHA256SUMS tamper detection, arch/python wheelhouse selection with a
    listing of available combos, python floor for ML vs --skip-ml)
  * a full --skip-ml install from a fabricated FULL bundle: the model unpack
    (and its zstd requirement) must be skipped entirely
  * the relock hook written into <venv>/bin/activate by both
    one_click_unlock_fetch_train_relock.sh and install_offline.sh: it must be
    the if-form (so `source activate` still exits 0 after online_unlock.sh
    retires .env.locked) and, for the installer, reference the ABSOLUTE
    bundle-root .env.locked so a custom --venv still gets offline-locked
  * build_airgap_bundle.sh full-ML wheelhouse assembly against a stub `pip`
    module (PYTHONPATH shim): CPU torch downloaded FIRST, requirements
    resolved in ONE `pip download -r ... --find-links` call (per-spec fallback
    only on failure), and a hard FAIL — not a silent delete — when the
    wheelhouse is contaminated with nvidia_*/triton*/non-+cpu-torch wheels
  * pack_model.sh checkpoint/pickle exclusion + manifest emission and
    unpack_model.sh manifest verification (skipped when zstd is absent)
  * lib_torch_install.sh writing torch.lock after a successful install
    (exercised with a stub pip on PATH — no real install)

No network, no torch, no real venv creation: the full-install tests pre-seed a
fake venv (the installer skips `python3 -m venv` when the dir exists) and the
builder tests shadow pip with a PYTHONPATH stub that fabricates wheel files.
"""
import hashlib
import json
import os
import shutil
import stat
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(REPO, "build_airgap_bundle.sh")
INSTALL = os.path.join(REPO, "airgap", "install_offline.sh")
ONE_CLICK = os.path.join(REPO, "one_click_unlock_fetch_train_relock.sh")
PACK = os.path.join(REPO, "pack_model.sh")
UNPACK = os.path.join(REPO, "unpack_model.sh")
TORCH_LIB = os.path.join(REPO, "lib_torch_install.sh")

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
needs_sha256sum = pytest.mark.skipif(shutil.which("sha256sum") is None,
                                     reason="sha256sum not available")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# --------------------------------------------------------------------------
# syntax / CLI surface
# --------------------------------------------------------------------------

@needs_bash
@pytest.mark.parametrize("script", [BUILD, INSTALL], ids=os.path.basename)
def test_bash_syntax(script):
    assert os.path.isfile(script), f"missing {script}"
    res = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
    assert res.returncode == 0, f"{script} failed bash -n:\n{res.stderr}"


@needs_bash
def test_build_help_exits_zero_without_side_effects(tmp_path):
    res = subprocess.run(["bash", BUILD, "--help"], capture_output=True, text=True,
                         cwd=tmp_path)
    assert res.returncode == 0
    assert "Usage" in res.stdout
    assert list(tmp_path.iterdir()) == []  # --help must not stage anything


@needs_bash
def test_build_rejects_unknown_flag(tmp_path):
    res = subprocess.run(["bash", BUILD, "--bogus"], capture_output=True, text=True,
                         cwd=tmp_path)
    assert res.returncode != 0
    assert "unknown argument" in (res.stdout + res.stderr)


@needs_bash
def test_install_help_exits_zero():
    res = subprocess.run(["bash", INSTALL, "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "verify-only" in res.stdout


# --------------------------------------------------------------------------
# install_offline.sh verification paths (fabricated mini-bundle)
# --------------------------------------------------------------------------

def _make_bundle(tmp_path, skip_ml=False, arch="x86_64", pytag="cp312", kind=None):
    """Fabricate a minimal bundle dir with a valid SHA256SUMS."""
    b = tmp_path / "bundle"
    (b / "repo").mkdir(parents=True)
    (b / "repo" / "requirements.txt").write_text("tree_sitter>=0.22\n")
    wheel_dir = b / "wheels" / arch / pytag
    wheel_dir.mkdir(parents=True)
    (wheel_dir / "dummy-1.0-py3-none-any.whl").write_bytes(b"not a real wheel")
    if kind is not None:
        (wheel_dir / "WHEELHOUSE_KIND").write_text(kind + "\n")
    (b / "bundle_manifest.json").write_text(json.dumps(
        {"version": "test", "skip_ml": skip_ml, "python": "3.12",
         "arches": [arch], "wheel_counts": {arch: 1}}) + "\n")
    shutil.copy(INSTALL, b / "install_offline.sh")
    os.chmod(b / "install_offline.sh",
             os.stat(b / "install_offline.sh").st_mode | stat.S_IXUSR)
    _write_sums(b)
    return b


def _write_sums(bundle):
    lines = []
    for root, _dirs, files in os.walk(bundle):
        for name in files:
            if name == "SHA256SUMS":
                continue
            p = os.path.join(root, name)
            rel = "./" + os.path.relpath(p, bundle)
            lines.append(f"{_sha256(p)}  {rel}")
    (bundle / "SHA256SUMS").write_text("\n".join(sorted(lines)) + "\n")


def _verify(bundle, *args, env_extra=None, subcommand="verify-only"):
    env = dict(os.environ)
    env.setdefault("CCR_ARCH", "x86_64")
    env.setdefault("CCR_PY_VER", "3.12")
    env.update(env_extra or {})
    cmd = ["bash", str(bundle / "install_offline.sh")]
    if subcommand:
        cmd.append(subcommand)
    cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


@needs_bash
@needs_sha256sum
def test_verify_only_passes_on_intact_bundle(tmp_path):
    b = _make_bundle(tmp_path)
    res = _verify(b)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "verify-only" in res.stdout
    assert "SHA256SUMS" in res.stdout


@needs_bash
@needs_sha256sum
def test_tampered_bundle_fails_hash_check(tmp_path):
    b = _make_bundle(tmp_path)
    (b / "repo" / "requirements.txt").write_text("evil==6.6.6\n")  # post-SUMS tamper
    res = _verify(b)
    assert res.returncode != 0
    out = res.stdout + res.stderr
    assert "SHA256SUMS" in out and "FAILED" in out


@needs_bash
@needs_sha256sum
def test_missing_arch_combo_lists_available(tmp_path):
    b = _make_bundle(tmp_path, arch="x86_64", pytag="cp312")
    res = _verify(b, env_extra={"CCR_ARCH": "aarch64"})
    assert res.returncode != 0
    out = res.stdout + res.stderr
    assert "x86_64/cp312" in out          # the combos the bundle DOES have
    assert "aarch64" in out               # ... and what this machine wanted


@needs_bash
@needs_sha256sum
def test_ml_bundle_rejects_python_310(tmp_path):
    b = _make_bundle(tmp_path, pytag="cp310")
    res = _verify(b, env_extra={"CCR_PY_VER": "3.10"})
    assert res.returncode != 0
    assert "3.11" in (res.stdout + res.stderr)


@needs_bash
@needs_sha256sum
def test_skip_ml_bundle_accepts_python_310(tmp_path):
    b = _make_bundle(tmp_path, skip_ml=True, pytag="cp310")
    res = _verify(b, env_extra={"CCR_PY_VER": "3.10"})
    assert res.returncode == 0, res.stdout + res.stderr


@needs_bash
@needs_sha256sum
def test_skip_ml_flag_overrides_full_bundle_floor(tmp_path):
    b = _make_bundle(tmp_path, skip_ml=False, pytag="cp310")
    res = _verify(b, "--skip-ml", env_extra={"CCR_PY_VER": "3.10"})
    assert res.returncode == 0, res.stdout + res.stderr


@needs_bash
@needs_sha256sum
def test_model_free_wheelhouse_auto_degrades_full_install_on_310(tmp_path):
    # A default multi-python build marks the cp310 wheelhouse model-free; a
    # 3.10 machine installing WITHOUT --skip-ml must degrade to the model-free
    # lane (with a notice) instead of dying at the python floor.
    b = _make_bundle(tmp_path, skip_ml=False, pytag="cp310", kind="model-free")
    res = _verify(b, env_extra={"CCR_PY_VER": "3.10"})
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "model-free" in out.lower()
    assert "skip_ml=1" in out


@needs_bash
@needs_sha256sum
def test_full_wheelhouse_kind_keeps_ml_floor(tmp_path):
    # An explicit 'full' marker must behave exactly like the unmarked case:
    # 3.10 + full ML wheelhouse is still refused at the floor.
    b = _make_bundle(tmp_path, skip_ml=False, pytag="cp310", kind="full")
    res = _verify(b, env_extra={"CCR_PY_VER": "3.10"})
    assert res.returncode != 0
    assert "3.11" in (res.stdout + res.stderr)


@needs_bash
@needs_sha256sum
def test_dryrun_env_stops_before_venv(tmp_path):
    b = _make_bundle(tmp_path)
    res = _verify(b, subcommand=None, env_extra={"CCR_INSTALL_DRYRUN": "1"})
    assert res.returncode == 0, res.stdout + res.stderr
    assert not (b / ".venv").exists()


@needs_bash
def test_install_refuses_outside_bundle_root(tmp_path):
    # No SHA256SUMS/repo/ here: must fail with a pointer, not plough on.
    empty = tmp_path / "empty"
    empty.mkdir()
    shutil.copy(INSTALL, empty / "install_offline.sh")
    res = subprocess.run(["bash", str(empty / "install_offline.sh"), "verify-only"],
                         capture_output=True, text=True)
    assert res.returncode != 0
    assert "bundle root" in (res.stdout + res.stderr)


# --------------------------------------------------------------------------
# pack_model.sh / unpack_model.sh manifest round-trip
# --------------------------------------------------------------------------

def _zstd_env(tmp_path):
    """Env for the pack/unpack subprocesses: real zstd when installed, else a
    passthrough stub on PATH. tar -I only needs SOME filter binary — the
    exclusion/manifest/verification logic under test is compression-agnostic."""
    env = dict(os.environ)
    if shutil.which("zstd") is None:
        stub_dir = tmp_path / "zstd-stub"
        stub_dir.mkdir(exist_ok=True)
        stub = stub_dir / "zstd"
        stub.write_text("#!/usr/bin/env bash\nexec cat\n")  # ignores -19/-d flags
        os.chmod(stub, 0o755)
        env["PATH"] = f"{stub_dir}:{env['PATH']}"
    return env


def _fake_model(tmp_path):
    model = tmp_path / "mymodel"
    (model / "checkpoint-5").mkdir(parents=True)
    (model / "config.json").write_text('{"architectures": ["RobertaForSequenceClassification"]}')
    (model / "model.safetensors").write_bytes(b"\x00" * 64)
    (model / "inference.json").write_text('{"threshold": 0.5}')
    (model / "training_args.bin").write_bytes(b"pickle-ish")
    (model / "checkpoint-5" / "optimizer.pt").write_bytes(b"\x00" * 128)
    (model / "checkpoint-5" / "trainer_state.json").write_text('{"global_step": 5}')
    return model


@needs_bash
def test_pack_model_excludes_training_state_and_writes_manifest(tmp_path):
    env = _zstd_env(tmp_path)
    model = _fake_model(tmp_path)
    out = tmp_path / "m.tar.zst"
    res = subprocess.run(["bash", PACK, str(model), str(out)],
                         capture_output=True, text=True, cwd=tmp_path, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert out.is_file()

    listing = subprocess.run(["tar", "-I", "zstd", "-tf", str(out)],
                             capture_output=True, text=True, env=env).stdout
    assert "mymodel/config.json" in listing
    assert "mymodel/model.safetensors" in listing
    assert "checkpoint-5" not in listing
    assert "training_args.bin" not in listing

    manifest = json.loads((tmp_path / "m.tar.zst.manifest.json").read_text())
    assert manifest["archive_sha256"] == _sha256(out)
    assert manifest["config_sha256"] == _sha256(model / "config.json")
    paths = {e["path"] for e in manifest["files"]}
    assert "mymodel/config.json" in paths
    assert not any("checkpoint-5" in p or p.endswith("training_args.bin") for p in paths)


@needs_bash
def test_unpack_verifies_manifest_and_rejects_corrupt_archive(tmp_path):
    env = _zstd_env(tmp_path)
    model = _fake_model(tmp_path)
    out = tmp_path / "m.tar.zst"
    subprocess.run(["bash", PACK, str(model), str(out)], check=True,
                   capture_output=True, cwd=tmp_path, env=env)

    # corrupt the archive AFTER packing: manifest verification must refuse it
    data = out.read_bytes()
    out.write_bytes(data + b"tamper")
    dest = tmp_path / "dest_bad"
    res = subprocess.run(["bash", UNPACK, str(out), str(dest)],
                         capture_output=True, text=True, env=env)
    assert res.returncode != 0
    assert "sha256 mismatch" in (res.stdout + res.stderr)
    assert not (dest / "mymodel" / "config.json").exists()

    # restore -> verification passes and the model extracts
    out.write_bytes(data)
    dest = tmp_path / "dest_ok"
    res = subprocess.run(["bash", UNPACK, str(out), str(dest)],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "sha256 verified" in res.stdout
    assert (dest / "mymodel" / "config.json").is_file()
    assert not (dest / "mymodel" / "training_args.bin").exists()


@needs_bash
def test_unpack_without_manifest_still_works(tmp_path):
    env = _zstd_env(tmp_path)
    model = _fake_model(tmp_path)
    out = tmp_path / "m.tar.zst"
    subprocess.run(["bash", PACK, str(model), str(out)], check=True,
                   capture_output=True, cwd=tmp_path, env=env)
    (tmp_path / "m.tar.zst.manifest.json").unlink()  # older-bundle scenario
    dest = tmp_path / "dest"
    res = subprocess.run(["bash", UNPACK, str(out), str(dest)],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (dest / "mymodel" / "config.json").is_file()


# --------------------------------------------------------------------------
# lib_torch_install.sh -> torch.lock
# --------------------------------------------------------------------------

@needs_bash
def test_torch_lock_appended_after_successful_install(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "pip"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == show ]]; then\n"
        "  echo 'Name: torch'; echo 'Version: 9.9.9'\n"
        "fi\n"
        "exit 0\n")
    os.chmod(stub, 0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["USE_GPU"] = "0"  # forces the CPU path regardless of host GPUs
    res = subprocess.run(
        ["bash", "-c", f'source "{TORCH_LIB}" && ccr_install_torch'],
        capture_output=True, text=True, cwd=tmp_path, env=env)
    assert res.returncode == 0, res.stdout + res.stderr

    lock = tmp_path / "torch.lock"
    assert lock.is_file(), "torch.lock not written next to the install"
    line = lock.read_text().strip()
    assert "torch==9.9.9" in line
    assert "download.pytorch.org/whl/cpu" in line
    assert line.split()[0] == os.uname().machine


@needs_bash
def test_torch_lock_not_written_on_failed_install(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "pip"
    stub.write_text("#!/usr/bin/env bash\nif [[ \"${1:-}\" == install ]]; then exit 1; fi\nexit 0\n")
    os.chmod(stub, 0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["USE_GPU"] = "0"
    res = subprocess.run(
        ["bash", "-c", f'source "{TORCH_LIB}" && ccr_install_torch'],
        capture_output=True, text=True, cwd=tmp_path, env=env)
    assert res.returncode != 0
    assert not (tmp_path / "torch.lock").exists()


# --------------------------------------------------------------------------
# relock hook: if-form, set -e survival, absolute path (install_offline)
# --------------------------------------------------------------------------

def _fake_venv(parent, name="venv"):
    """Minimal venv double: install_offline.sh skips `python3 -m venv` when the
    dir exists, and the relock hook only needs VIRTUAL_ENV + an activate file."""
    venv = parent / name
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(f'export VIRTUAL_ENV="{venv}"\n')
    return venv


def _source_probe(activate):
    """Source activate under `set -euo pipefail` in a clean shell; report both
    survival and whether the offline lock got applied."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_")}
    return subprocess.run(
        ["bash", "-c",
         f'set -euo pipefail; source "{activate}"; '
         'echo "SURVIVED LOCK=${HF_HUB_OFFLINE:-unset}"'],
        capture_output=True, text=True, env=env)


def _one_click_hook_append_line():
    """The exact single-line hook-append command from the one-click script."""
    with open(ONE_CLICK) as f:
        lines = [ln for ln in f if '>> "$VENV/bin/activate"' in ln]
    assert len(lines) == 1, "expected exactly one hook-append line in one_click"
    return lines[0]


@needs_bash
def test_one_click_relock_hook_survives_env_locked_retirement(tmp_path):
    """Regression (confirmed): the hook appended as the LAST line of activate
    was 'test -f ... && set -a && . ... && set +a'; once online_unlock.sh
    renamed .env.locked away, `source activate` returned 1 and every set -e
    script (one_click itself at its own `source`, run_demo.sh, scan_repo.sh)
    died silently with rc=1. The if-form must always exit 0."""
    venv = _fake_venv(tmp_path)
    append = _one_click_hook_append_line()
    env = dict(os.environ)
    env["VENV"] = str(venv)
    res = subprocess.run(["bash", "-c", append], capture_output=True, text=True,
                         cwd=tmp_path, env=env)
    assert res.returncode == 0, res.stderr
    activate = venv / "bin" / "activate"
    assert ".env.locked" in activate.read_text()

    # .env.locked ABSENT (retired by `source ./online_unlock.sh`): a set -e
    # caller sourcing activate must survive — this was the confirmed failure.
    res = _source_probe(activate)
    assert res.returncode == 0, (
        f"activate killed a set -e caller after .env.locked retirement:\n{res.stderr}")
    assert "SURVIVED LOCK=unset" in res.stdout

    # .env.locked PRESENT (relocked): the hook must still apply the lock.
    (tmp_path / ".env.locked").write_text("export HF_HUB_OFFLINE=1\n")
    res = _source_probe(activate)
    assert res.returncode == 0, res.stderr
    assert "SURVIVED LOCK=1" in res.stdout


@needs_bash
def test_one_click_relock_hook_is_idempotent_and_cures_old_venvs(tmp_path):
    venv = _fake_venv(tmp_path)
    activate = venv / "bin" / "activate"
    # A venv from before the fix: broken &&-hook already appended.
    activate.write_text(
        activate.read_text()
        + '\n# relock\n test -f "$VIRTUAL_ENV/../.env.locked" && set -a && '
          '. "$VIRTUAL_ENV/../.env.locked" && set +a\n')
    append = _one_click_hook_append_line()
    env = dict(os.environ)
    env["VENV"] = str(venv)
    for _ in range(2):  # run twice: fixed hook appended exactly once
        subprocess.run(["bash", "-c", append], check=True, cwd=tmp_path, env=env)
    assert activate.read_text().count('if test -f "$VIRTUAL_ENV/../.env.locked"') == 1
    # With the fixed hook appended after the old one, activate ends with the
    # if-form and set -e callers survive even though .env.locked is absent.
    res = _source_probe(activate)
    assert res.returncode == 0, res.stderr
    assert "SURVIVED" in res.stdout


# --------------------------------------------------------------------------
# install_offline.sh full --skip-ml install (fabricated FULL bundle)
# --------------------------------------------------------------------------

def _make_installable_bundle(tmp_path):
    """A FULL bundle (model/model.tar.zst present) whose repo/ carries stub
    smoke-lane scripts, so a --skip-ml install can run end to end without the
    real pipeline, torch, network — or zstd."""
    b = _make_bundle(tmp_path)  # x86_64/cp312, manifest skip_ml=false
    (b / "repo" / "heuristic_scan.py").write_text(
        "import json, os, sys\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "json.dump([], open(os.path.join(out, 'classifier_findings.json'), 'w'))\n")
    (b / "repo" / "llm_explain.py").write_text(
        "import sys\nopen(sys.argv[sys.argv.index('--out') + 1], 'w').write('[]')\n")
    (b / "repo" / "to_sarif.py").write_text(
        "import sys\nopen(sys.argv[sys.argv.index('-o') + 1], 'w').write('{}')\n")
    # Garbage archive on purpose: a --skip-ml install must never unpack it
    # (and would also die on this zstd-less machine if it tried).
    (b / "model").mkdir()
    (b / "model" / "model.tar.zst").write_bytes(b"\x28\xb5\x2f\xfdgarbage")
    _write_sums(b)
    return b


def _run_install(bundle, venv, *args):
    env = dict(os.environ)
    env["CCR_ARCH"] = "x86_64"
    env["CCR_PY_VER"] = "3.12"
    return subprocess.run(
        ["bash", str(bundle / "install_offline.sh"), "--skip-ml",
         "--venv", str(venv), *args],
        capture_output=True, text=True, env=env)


@needs_bash
@needs_sha256sum
def test_skip_ml_install_skips_model_unpack_and_needs_no_zstd(tmp_path):
    """Regression (confirmed): './install_offline.sh --skip-ml' from a full
    bundle reached the model section and died at 'zstd not found' on a target
    without zstd — impossible to remedy offline. --skip-ml must skip the model
    unpack entirely (the model-free lane needs no model)."""
    b = _make_installable_bundle(tmp_path)
    venv = _fake_venv(tmp_path / "opt")  # custom --venv OUTSIDE the bundle root
    res = _run_install(b, venv)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "INSTALL PASS" in out
    assert "zstd not found" not in out
    assert "skipping model unpack" in out
    assert not (b / "vuln-model").exists()
    assert (b / ".env.locked").is_file()


@needs_bash
@needs_sha256sum
def test_install_hook_uses_absolute_bundle_root_and_custom_venv_gets_locked(tmp_path):
    """Regression (confirmed): the hook referenced $VIRTUAL_ENV/../.env.locked
    while .env.locked is written to the bundle root, so a custom --venv outside
    the bundle was never offline-locked (and its activate exited 1). The hook
    must embed the absolute bundle-root path."""
    b = _make_installable_bundle(tmp_path)
    venv = _fake_venv(tmp_path / "opt")
    assert _run_install(b, venv).returncode == 0
    activate = venv / "bin" / "activate"
    hook = activate.read_text()
    assert f"{b}/.env.locked" in hook          # absolute bundle-root path
    assert "$VIRTUAL_ENV/../.env.locked" not in hook
    # activating the custom venv applies the offline lock and exits 0
    res = _source_probe(activate)
    assert res.returncode == 0, res.stderr
    assert "SURVIVED LOCK=1" in res.stdout


@needs_bash
@needs_sha256sum
def test_install_rerun_survives_env_locked_retirement(tmp_path):
    """Regression (confirmed): after online_unlock.sh renamed .env.locked to
    .env.locked.disabled, re-running the installer died silently (rc=1, no
    diagnostics) at `source "$VENV/bin/activate"` because the &&-form hook was
    the last line of activate. The if-form hook must let re-runs proceed."""
    b = _make_installable_bundle(tmp_path)
    venv = _fake_venv(tmp_path / "opt")
    assert _run_install(b, venv).returncode == 0
    activate = venv / "bin" / "activate"

    # retire the lock the way online_unlock.sh does
    (b / ".env.locked").rename(b / ".env.locked.disabled")
    res = _source_probe(activate)
    assert res.returncode == 0, f"activate killed a set -e caller:\n{res.stderr}"
    assert "SURVIVED LOCK=unset" in res.stdout

    # re-run: must not die at `source activate`, must not duplicate the hook
    res = _run_install(b, venv)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "INSTALL PASS" in res.stdout + res.stderr
    assert activate.read_text().count("# ccr relock") == 1


# --------------------------------------------------------------------------
# build_airgap_bundle.sh wheelhouse assembly (stub pip via PYTHONPATH)
# --------------------------------------------------------------------------

_PIP_STUB = '''\
import os, re, sys

def log(argv):
    p = os.environ.get("PIP_STUB_LOG")
    if p:
        with open(p, "a") as f:
            f.write(" ".join(argv) + "\\n")

def arg_after(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None

def touch(dest, name):
    open(os.path.join(dest, name), "wb").write(b"stub-wheel")

argv = sys.argv[1:]
log(argv)
if argv[:1] == ["--version"]:
    print("pip 25.0 (stub)")
    sys.exit(0)
if argv[:1] == ["download"]:
    dest = arg_after(argv, "-d")
    index = arg_after(argv, "--index-url") or ""
    reqfile = arg_after(argv, "-r")
    if "download.pytorch.org/whl/cpu" in index:
        touch(dest, "torch-2.9.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl")
        sys.exit(0)
    if reqfile:
        mode = os.environ.get("PIP_STUB_REQS_MODE", "ok")
        if mode == "fail":
            print("stub: -r resolution forced to fail", file=sys.stderr)
            sys.exit(1)
        touch(dest, "transformers-4.57.1-py3-none-any.whl")
        touch(dest, "numpy-2.3.3-cp312-cp312-manylinux_2_28_x86_64.whl")
        if mode == "cuda":
            # what a PyPI-resolved torch dependency drags in
            touch(dest, "torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl")
            touch(dest, "nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl")
            touch(dest, "triton-3.1.0-cp312-cp312-manylinux_2_28_x86_64.whl")
        sys.exit(0)
    name = re.split(r"[<>=!~\\[ ]", argv[1])[0].replace("-", "_")
    touch(dest, name + "-1.0-py3-none-any.whl")
    sys.exit(0)
sys.exit(0)
'''


def _run_build(tmp_path, mode, *extra_args):
    stub = tmp_path / "pystub"
    (stub / "pip").mkdir(parents=True)
    (stub / "pip" / "__init__.py").write_text("")
    (stub / "pip" / "__main__.py").write_text(_PIP_STUB)
    model = _fake_model(tmp_path)
    out = tmp_path / "dist"
    log = tmp_path / "pip_calls.log"
    env = _zstd_env(tmp_path)
    env["PYTHONPATH"] = f"{stub}{os.pathsep}" + env.get("PYTHONPATH", "")
    env["PIP_STUB_LOG"] = str(log)
    env["PIP_STUB_REQS_MODE"] = mode
    res = subprocess.run(
        ["bash", BUILD, "--model", str(model), "--out", str(out),
         "--arch", "x86_64", "--python", "3.12", *extra_args],
        capture_output=True, text=True, env=env)
    return res, out, log


def _download_calls(log):
    if not log.exists():
        return []
    return [ln for ln in log.read_text().splitlines() if ln.startswith("download ")]


def _wheelhouse(out):
    import glob as _glob
    dirs = _glob.glob(str(out / "ccr-airgap-*" / "wheels" / "x86_64" / "cp312"))
    assert dirs, "wheelhouse dir not staged"
    return dirs[0]


@needs_bash
def test_build_downloads_cpu_torch_first_then_single_resolution(tmp_path):
    """Regression (confirmed by a real build): per-spec `pip download` let
    peft/accelerate resolve torch against PyPI, pulling the CUDA torch plus
    ~18 nvidia_*/triton wheels (3.4G) alongside the +cpu torch. The builder
    must download CPU torch FIRST, then resolve requirements in ONE
    `pip download -r` call with --find-links into the same wheelhouse."""
    res, out, log = _run_build(tmp_path, "ok")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Bundle ready" in res.stdout

    calls = _download_calls(log)
    torch_calls = [c for c in calls if "download.pytorch.org/whl/cpu" in c]
    req_calls = [c for c in calls if " -r " in c]
    assert len(torch_calls) == 1 and len(req_calls) == 1
    assert len(calls) == 2, f"expected exactly torch + one -r call, got:\n{calls}"
    assert calls.index(torch_calls[0]) < calls.index(req_calls[0]), \
        "CPU torch must be downloaded BEFORE the requirements resolution"

    dest = _wheelhouse(out)
    assert f"--find-links {dest}" in req_calls[0], \
        "-r resolution must --find-links the wheelhouse holding the +cpu torch"
    names = sorted(os.listdir(dest))
    assert any(n.startswith("torch-") and "+cpu" in n for n in names)
    assert not any(n.startswith(("nvidia_", "triton")) for n in names)


@needs_bash
def test_build_fails_hard_on_cuda_contaminated_wheelhouse(tmp_path):
    """If the resolution still leaks to PyPI's CUDA torch, the build must FAIL
    with a clear error — not silently delete the offending wheels (deletion
    could mask a resolution solved against the CUDA torch)."""
    res, out, _log = _run_build(tmp_path, "cuda")
    assert res.returncode != 0
    err = res.stdout + res.stderr
    assert "CUDA-contaminated" in err
    assert "nvidia_cublas_cu12" in err
    assert "torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl" in err
    assert "Bundle ready" not in res.stdout
    # the evidence is left in place, not deleted
    names = os.listdir(_wheelhouse(out))
    assert any(n.startswith("nvidia_") for n in names)


@needs_bash
def test_build_falls_back_to_per_spec_with_find_links(tmp_path):
    """When the single -r resolution fails, the per-requirement fallback loop
    must still pass --find-links so torch keeps resolving to the +cpu wheel."""
    res, out, log = _run_build(tmp_path, "fail")
    assert res.returncode == 0, res.stdout + res.stderr
    calls = _download_calls(log)
    assert calls and "download.pytorch.org/whl/cpu" in calls[0], \
        "CPU torch must still be the first download"
    spec_calls = [c for c in calls if " -r " not in c
                  and "download.pytorch.org" not in c]
    assert len(spec_calls) >= 5, "per-spec fallback loop did not run"
    dest = _wheelhouse(out)
    assert all(f"--find-links {dest}" in c for c in spec_calls), \
        "fallback downloads must --find-links the local +cpu torch"


@needs_bash
def test_build_skip_ml_wheel_path_unchanged(tmp_path):
    """--skip-ml keeps the old shape: per-spec downloads of the tree-sitter
    trio only — no torch call, no -r resolution, no CUDA gate interference."""
    res, _out, log = _run_build(tmp_path, "ok", "--skip-ml")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Bundle ready" in res.stdout
    calls = _download_calls(log)
    assert calls, "no downloads logged"
    assert all("download.pytorch.org" not in c and " -r " not in c for c in calls)
    assert all("tree_sitter" in c.split()[1] for c in calls)


@needs_bash
def test_build_multi_python_splits_full_and_model_free_at_the_floor(tmp_path):
    """--python 3.10,3.12: the cp310 wheelhouse (below the numpy 3.11 floor)
    gets only the tree-sitter trio and a model-free marker; cp312 keeps the
    full torch-first + single -r shape and a full marker."""
    res, out, log = _run_build(tmp_path, "ok", "--python", "3.10,3.12")
    assert res.returncode == 0, res.stdout + res.stderr

    calls = _download_calls(log)
    calls_310 = [c for c in calls if "--python-version 3.10" in c]
    calls_312 = [c for c in calls if "--python-version 3.12" in c]
    assert calls_310 and calls_312
    # cp310: trio only — no torch channel, no -r resolution
    assert all("download.pytorch.org" not in c and " -r " not in c
               for c in calls_310)
    assert all("tree_sitter" in c.split()[1] for c in calls_310)
    # cp312: torch first, then the single -r resolution
    assert any("download.pytorch.org/whl/cpu" in c for c in calls_312)
    assert any(" -r " in c for c in calls_312)

    import glob as _glob
    stage = next(p for p in _glob.glob(str(out / "ccr-airgap-*"))
                 if os.path.isdir(p))
    kind_310 = open(os.path.join(stage, "wheels", "x86_64", "cp310",
                                 "WHEELHOUSE_KIND")).read().strip()
    kind_312 = open(os.path.join(stage, "wheels", "x86_64", "cp312",
                                 "WHEELHOUSE_KIND")).read().strip()
    assert kind_310 == "model-free" and kind_312 == "full"

    manifest = json.loads(open(os.path.join(stage, "bundle_manifest.json")).read())
    kinds = {(w["arch"], w["python"]): w["kind"] for w in manifest["wheelhouses"]}
    assert kinds[("x86_64", "cp310")] == "model-free"
    assert kinds[("x86_64", "cp312")] == "full"
    assert manifest["pythons"] == ["3.10", "3.12"]
