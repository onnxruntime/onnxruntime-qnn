# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Local tests for resolve_tool_versions.sh.

The script is dual-mode bash; these tests drive it as an executable via
subprocess, feeding synthetic sdk.yaml / VERSION_NUMBER files and environment
overrides to assert its precedence rules and exit codes. Run locally with:

    pytest qcom/scripts/linux/tests -v

(These are not CI-wired, mirroring qcom/scripts/all/tests.)
"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "resolve_tool_versions.sh"

# Every version source the script reads. We strip all of them from the inherited
# environment so a stray value on the developer's box can't leak into a test.
_VERSION_ENV_KEYS = (
    "QAIRT_SDK_ROOT",
    "QNN_SDK_ROOT",
    "SNPE_ROOT",
    "ORT_PREBUILT_ROOT",
)


def run(args, *, env=None, cwd=None):
    """Run resolve_tool_versions.sh with a scrubbed environment plus `env` overrides."""
    clean = os.environ.copy()
    for key in _VERSION_ENV_KEYS:
        clean.pop(key, None)
    if env:
        clean.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=clean,
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def write_sdk_yaml(root: Path, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sdk.yaml").write_text(body)
    return root


# ---------------------------------------------------------------------------
# sdk.yaml parsing variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body,expected",
    [
        ("version: 2.35.0\n", "2.35.0"),
        ('version: "2.48.40"\n', "2.48.40"),
        ("version: '2.31.0'\n", "2.31.0"),
        ("Version: 2.30.0\n", "2.30.0"),                       # capitalized key
        ("  version:   2.29.0  \n", "2.29.0"),                 # leading/trailing ws
        ("sdk_version: 9.9.9\nversion: 2.28.0\n", "2.28.0"),   # decoy first line
        ('sdk_version: 9.9.9\n  Version:  "2.48.40"\n', "2.48.40"),  # combined
    ],
)
def test_sdk_yaml_variants(tmp_path, body, expected):
    write_sdk_yaml(tmp_path, body)
    r = run(["qairt"], env={"QAIRT_SDK_ROOT": str(tmp_path)})
    assert r.returncode == 0
    assert r.stdout == f"{expected}\n"


def test_qairt_root_precedence(tmp_path):
    # QAIRT_SDK_ROOT wins over QNN_SDK_ROOT wins over SNPE_ROOT.
    qairt = write_sdk_yaml(tmp_path / "qairt", "version: 1.1.1\n")
    qnn = write_sdk_yaml(tmp_path / "qnn", "version: 2.2.2\n")
    snpe = write_sdk_yaml(tmp_path / "snpe", "version: 3.3.3\n")

    r = run(["qairt"], env={"QAIRT_SDK_ROOT": str(qairt), "QNN_SDK_ROOT": str(qnn), "SNPE_ROOT": str(snpe)})
    assert r.stdout == "1.1.1\n"

    r = run(["qairt"], env={"QNN_SDK_ROOT": str(qnn), "SNPE_ROOT": str(snpe)})
    assert r.stdout == "2.2.2\n"

    r = run(["qairt"], env={"SNPE_ROOT": str(snpe)})
    assert r.stdout == "3.3.3\n"


def test_sdk_root_without_yaml_is_skipped(tmp_path):
    # A root that exists but has no sdk.yaml falls through to the next root.
    empty = tmp_path / "empty"
    empty.mkdir()
    good = write_sdk_yaml(tmp_path / "good", "version: 4.4.4\n")
    r = run(["qairt"], env={"QAIRT_SDK_ROOT": str(empty), "QNN_SDK_ROOT": str(good)})
    assert r.returncode == 0
    assert r.stdout == "4.4.4\n"


# ---------------------------------------------------------------------------
# ORT sourcing: $ORT_PREBUILT_ROOT (the ORT RUNTIME version). repo-root
# VERSION_NUMBER is the plugin/wheel version and is intentionally NOT read.
# ---------------------------------------------------------------------------
def test_ort_prebuilt_version_number(tmp_path):
    (tmp_path / "VERSION_NUMBER").write_text("1.20.0\n")
    r = run(["ort"], env={"ORT_PREBUILT_ROOT": str(tmp_path)})
    assert r.returncode == 0
    assert r.stdout == "1.20.0\n"


def test_ort_prebuilt_version_fallback_name(tmp_path):
    # Falls back to VERSION when VERSION_NUMBER is absent.
    (tmp_path / "VERSION").write_text("1.19.2\n")
    r = run(["ort"], env={"ORT_PREBUILT_ROOT": str(tmp_path)})
    assert r.returncode == 0
    assert r.stdout == "1.19.2\n"


def test_ort_ignores_repo_root_version_number():
    # Running inside the repo (default cwd) with no env and no ORT_PREBUILT_ROOT
    # must be undeterminable: the repo-root VERSION_NUMBER (plugin/wheel version)
    # is deliberately NOT a source for the ORT runtime version.
    r = run(["ort"])
    assert r.returncode == 3
    assert r.stdout == ""


# ---------------------------------------------------------------------------
# Undeterminable -> exit 3, no stdout
# ---------------------------------------------------------------------------
def test_qairt_undeterminable(tmp_path):
    r = run(["qairt"], cwd=str(tmp_path))
    assert r.returncode == 3
    assert r.stdout == ""


def test_ort_undeterminable(tmp_path):
    # No env, no ORT_PREBUILT_ROOT -> nothing to resolve.
    r = run(["ort"], cwd=str(tmp_path))
    assert r.returncode == 3
    assert r.stdout == ""


def test_both_exit3_if_either_undeterminable(tmp_path):
    # ORT resolvable via prebuilt root, QAIRT not -> both fails with 3.
    (tmp_path / "VERSION_NUMBER").write_text("1.20.0\n")
    r = run(["both"], env={"ORT_PREBUILT_ROOT": str(tmp_path)}, cwd=str(tmp_path))
    assert r.returncode == 3


# ---------------------------------------------------------------------------
# Usage / help
# ---------------------------------------------------------------------------
def test_unknown_arg_is_usage_error():
    r = run(["bogus"])
    assert r.returncode == 2


def test_help_exits_zero():
    r = run(["--help"])
    assert r.returncode == 0
