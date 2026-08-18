# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Local tests for resolve_tool_versions.sh.

The script is dual-mode bash; these tests drive it as an executable via
subprocess, feeding a synthetic `--bin-dir` tree (CMakeCache.txt + sdk.yaml /
VERSION_NUMBER files) to assert its precedence rules and exit codes. Run
locally with:

    pytest qcom/scripts/linux/tests -v

(These are not CI-wired, mirroring qcom/scripts/all/tests.)
"""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "resolve_tool_versions.sh"


def run(args, *, cwd=None):
    """Run resolve_tool_versions.sh with the given CLI args."""
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        check=False,
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def write_cmake_cache(bin_dir: Path, **entries: str) -> Path:
    """Write a minimal CMakeCache.txt with the given `<var>:UNINITIALIZED=<value>` entries."""
    lines = [f"{k}:UNINITIALIZED={v}" for k, v in entries.items()]
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "CMakeCache.txt").write_text("\n".join(lines) + "\n")
    return bin_dir


def write_sdk_yaml(root: Path, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sdk.yaml").write_text(body)
    return root


# ---------------------------------------------------------------------------
# sdk.yaml parsing variants (via onnxruntime_QNN_HOME in CMakeCache.txt)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body,expected",
    [
        ("version: 2.35.0\n", "2.35.0"),
        ('version: "2.48.40"\n', "2.48.40"),
        ("version: '2.31.0'\n", "2.31.0"),
        ("Version: 2.30.0\n", "2.30.0"),  # capitalized key
        ("  version:   2.29.0  \n", "2.29.0"),  # leading/trailing ws
        ("sdk_version: 9.9.9\nversion: 2.28.0\n", "2.28.0"),  # decoy first line
        ('sdk_version: 9.9.9\n  Version:  "2.48.40"\n', "2.48.40"),  # combined
    ],
)
def test_sdk_yaml_variants(tmp_path, body, expected):
    bin_dir = tmp_path / "bin"
    qnn_home = write_sdk_yaml(tmp_path / "qairt_sdk", body)
    write_cmake_cache(bin_dir, onnxruntime_QNN_HOME=str(qnn_home))

    r = run([f"--bin-dir={bin_dir}", "qairt"])
    assert r.returncode == 0
    assert r.stdout == f"{expected}\n"


def test_qairt_missing_cache_entry_is_undeterminable(tmp_path):
    # CMakeCache.txt exists but has no onnxruntime_QNN_HOME entry.
    bin_dir = tmp_path / "bin"
    write_cmake_cache(bin_dir, some_other_var="whatever")

    r = run([f"--bin-dir={bin_dir}", "qairt"])
    assert r.returncode == 3
    assert r.stdout == ""


def test_qairt_home_without_sdk_yaml_is_undeterminable(tmp_path):
    # onnxruntime_QNN_HOME points at a real dir, but it has no sdk.yaml.
    bin_dir = tmp_path / "bin"
    qnn_home = tmp_path / "qairt_sdk"
    qnn_home.mkdir()
    write_cmake_cache(bin_dir, onnxruntime_QNN_HOME=str(qnn_home))

    r = run([f"--bin-dir={bin_dir}", "qairt"])
    assert r.returncode == 3
    assert r.stdout == ""


def test_qairt_missing_cmake_cache_is_undeterminable(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()  # no CMakeCache.txt at all

    r = run([f"--bin-dir={bin_dir}", "qairt"])
    assert r.returncode == 3
    assert r.stdout == ""


# ---------------------------------------------------------------------------
# ORT sourcing: onnxruntime_ORT_HOME (prebuilt) precedes the FetchContent
# fallback under <bin_dir>/_deps/ort_core-src.
# ---------------------------------------------------------------------------
def test_ort_prebuilt_version_number(tmp_path):
    bin_dir = tmp_path / "bin"
    ort_home = tmp_path / "ort_prebuilt"
    ort_home.mkdir()
    (ort_home / "VERSION_NUMBER").write_text("1.20.0\n")
    write_cmake_cache(bin_dir, onnxruntime_ORT_HOME=str(ort_home))

    r = run([f"--bin-dir={bin_dir}", "ort"])
    assert r.returncode == 0
    assert r.stdout == "1.20.0\n"


def test_ort_prebuilt_version_fallback_name(tmp_path):
    # Falls back to VERSION when VERSION_NUMBER is absent.
    bin_dir = tmp_path / "bin"
    ort_home = tmp_path / "ort_prebuilt"
    ort_home.mkdir()
    (ort_home / "VERSION").write_text("1.19.2\n")
    write_cmake_cache(bin_dir, onnxruntime_ORT_HOME=str(ort_home))

    r = run([f"--bin-dir={bin_dir}", "ort"])
    assert r.returncode == 0
    assert r.stdout == "1.19.2\n"


def test_ort_fetchcontent_fallback(tmp_path):
    # No onnxruntime_ORT_HOME entry (the path every current CI build takes) ->
    # fall back to the FetchContent-populated ORT source tree.
    bin_dir = tmp_path / "bin"
    write_cmake_cache(bin_dir, some_other_var="whatever")
    fc_dir = bin_dir / "_deps" / "ort_core-src"
    fc_dir.mkdir(parents=True)
    (fc_dir / "VERSION_NUMBER").write_text("1.27.0\n")

    r = run([f"--bin-dir={bin_dir}", "ort"])
    assert r.returncode == 0
    assert r.stdout == "1.27.0\n"


def test_ort_neither_source_is_undeterminable(tmp_path):
    bin_dir = tmp_path / "bin"
    write_cmake_cache(bin_dir, some_other_var="whatever")

    r = run([f"--bin-dir={bin_dir}", "ort"])
    assert r.returncode == 3
    assert r.stdout == ""


# ---------------------------------------------------------------------------
# Undeterminable -> exit 3, no stdout
# ---------------------------------------------------------------------------
def test_both_exit3_if_either_undeterminable(tmp_path):
    # ORT resolvable via FetchContent fallback, QAIRT not -> both fails with 3.
    bin_dir = tmp_path / "bin"
    write_cmake_cache(bin_dir, some_other_var="whatever")
    fc_dir = bin_dir / "_deps" / "ort_core-src"
    fc_dir.mkdir(parents=True)
    (fc_dir / "VERSION_NUMBER").write_text("1.27.0\n")

    r = run([f"--bin-dir={bin_dir}", "both"])
    assert r.returncode == 3


def test_both_succeeds_when_both_resolvable(tmp_path):
    bin_dir = tmp_path / "bin"
    qnn_home = write_sdk_yaml(tmp_path / "qairt_sdk", "version: 2.48.40\n")
    write_cmake_cache(bin_dir, onnxruntime_QNN_HOME=str(qnn_home))
    fc_dir = bin_dir / "_deps" / "ort_core-src"
    fc_dir.mkdir(parents=True)
    (fc_dir / "VERSION_NUMBER").write_text("1.27.0\n")

    r = run([f"--bin-dir={bin_dir}", "both"])
    assert r.returncode == 0
    assert r.stdout == "qairt=2.48.40\nort=1.27.0\n"


# ---------------------------------------------------------------------------
# Usage / help
# ---------------------------------------------------------------------------
def test_missing_bin_dir_is_usage_error():
    r = run(["qairt"])
    assert r.returncode == 2


def test_unknown_arg_is_usage_error(tmp_path):
    r = run([f"--bin-dir={tmp_path}", "bogus"])
    assert r.returncode == 2


def test_help_exits_zero():
    r = run(["--help"])
    assert r.returncode == 0
