# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from model_test_filter import QNN_MODEL_TEST_EXCLUSIONS, filter_model_test_suite

_LINUX_RUN_TESTS = Path(__file__).resolve().parents[2] / "linux" / "run_tests.sh"


def _case(root: Path, name: str) -> None:
    case = root / name
    case.mkdir()
    (case / "model.onnx").write_text(name)


def _cleanup_trap() -> str:
    for line in _LINUX_RUN_TESTS.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("trap ") and "filtered_test_root" in line:
            return line.strip()
    raise AssertionError("Filtered model-test cleanup trap not found in run_tests.sh")


def test_filter_excludes_cpu_roialign_cases(tmp_path: Path) -> None:
    source = tmp_path / "node"
    source.mkdir()
    _case(source, "test_add")
    for name in QNN_MODEL_TEST_EXCLUSIONS["cpu"]["node"]:
        _case(source, name)
    (source / "LICENSE").write_text("fixture")

    destination = tmp_path / "filtered-node"
    skipped = filter_model_test_suite(source, destination, "node", "cpu")

    assert skipped == dict(QNN_MODEL_TEST_EXCLUSIONS["cpu"]["node"])
    assert (destination / "test_add" / "model.onnx").read_text() == "test_add"
    assert (destination / "LICENSE").read_text() == "fixture"
    for name in skipped:
        assert not (destination / name).exists()


def test_filter_excludes_only_htp_mode_max_roialign_case(tmp_path: Path) -> None:
    source = tmp_path / "node"
    source.mkdir()
    for name in QNN_MODEL_TEST_EXCLUSIONS["cpu"]["node"]:
        _case(source, name)

    skipped = filter_model_test_suite(source, tmp_path / "filtered-node", "node", "htp")

    assert skipped == dict(QNN_MODEL_TEST_EXCLUSIONS["htp"]["node"])
    assert (tmp_path / "filtered-node" / "test_roialign_aligned_false").is_dir()
    assert (tmp_path / "filtered-node" / "test_roialign_aligned_true").is_dir()


def test_filter_leaves_unconfigured_backend_intact(tmp_path: Path) -> None:
    source = tmp_path / "node"
    source.mkdir()
    _case(source, "test_roialign_mode_max")

    skipped = filter_model_test_suite(source, tmp_path / "filtered-node", "node", "gpu")

    assert skipped == {}
    assert (tmp_path / "filtered-node" / "test_roialign_mode_max").is_dir()


def test_filter_rejects_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "node"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        filter_model_test_suite(source, destination, "node", "cpu")


@pytest.mark.skipif(os.name == "nt", reason="This test exercises the Linux Bash test harness.")
@pytest.mark.parametrize("early_exit", [False, True], ids=["return", "exit"])
def test_linux_model_test_cleanup_trap_removes_temporary_suite(tmp_path: Path, early_exit: bool) -> None:
    filtered_test_root = tmp_path / "filtered-node"
    filtered_test_root.mkdir()
    trap = _cleanup_trap()
    quoted_root = shlex.quote(str(filtered_test_root))

    if early_exit:
        script = f"filtered_test_root={quoted_root}; {trap}; exit 37"
        expected_return_code = 37
    else:
        script = (
            f"function run_model_test() {{ local filtered_test_root={quoted_root}; {trap}; return; }}; run_model_test"
        )
        expected_return_code = 0

    result = subprocess.run(["bash", "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == expected_return_code, result.stderr
    assert not filtered_test_root.exists()
