# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Check ARM64 shadow task planning and isolation without building ORT."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

QCOM_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("task_name", "mode"),
    [
        ("_build_ort_linux_aarch64_manylinux_2_34_internal_symbols", "build"),
        ("test_ort_linux_aarch64_manylinux_2_34_internal_symbols", "test"),
    ],
)
def test_arm64_internal_symbol_task_passes_required_flags(tmp_path: Path, task_name: str, mode: str) -> None:
    # Task availability is selected at import time. Use a subprocess so simulating
    # ARM64 does not change the task registry used by the other Linux tests.
    script = """
import sys
from pathlib import Path

import ep_build.util as util

util.is_host_linux = lambda: True
util.is_host_arm64 = lambda: True
util.is_host_x86_64 = lambda: False
util.is_host_windows = lambda: False
util.is_host_mac = lambda: False

from build_and_test import TaskLibrary
from ep_build.plan import Plan

task_name, mode, temp_dir = sys.argv[1:]
library = TaskLibrary(
    python_executable=Path(sys.executable),
    venv_path=Path(temp_dir) / "venv",
    config="Release",
    target_py_version=None,
    ort_prebuilt_root=None,
    qairt_sdk_root=None,
    docker_ccache_root=None,
    build_nuget=False,
    build_archive=False,
    build_aar=False,
)
plan = Plan()
getattr(library, task_name)(plan)
tasks = []
plan.for_each(lambda step_id, task: tasks.append(task))
assert len(tasks) == 1
command = tasks[0].executables_and_args[0]
assert f"--mode={mode}" in command
assert "--enable-qnn-internal-test-symbols" in command
assert "--build-dir-suffix=internal-symbols" in command
assert "--target-arch=aarch64_manylinux_2_34" in command
if mode == "build":
    assert "--qnn-arch-abi=aarch64-oe-linux-gcc11.2" in command
    assert "--no-use-cache" in command
"""
    result = subprocess.run(
        [sys.executable, "-c", script, task_name, mode, str(tmp_path)],
        cwd=QCOM_ROOT,
        env=os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_arm64_docker_cleanup_preserves_shipping_wheels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(QCOM_ROOT))
    from ep_build.task import NoOpTask  # noqa: PLC0415
    from ep_build.tasks import build  # noqa: PLC0415

    shipping_dist = tmp_path / "build/linux-aarch64_manylinux_2_34/Release/dist"
    internal_dist = tmp_path / "build/linux-aarch64_manylinux_2_34-internal-symbols/Release/dist"
    for dist in (shipping_dist, internal_dist):
        dist.mkdir(parents=True)
        (dist / "test.whl").touch()

    monkeypatch.setattr(build, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build, "DockerBuildAndTestTask", lambda *args, **kwargs: NoOpTask())
    task = build.BuildEpDockerTask(
        None,
        "aarch64_manylinux_2_34",
        "Release",
        None,
        None,
        None,
        build_dir_suffix="internal-symbols",
    )
    task.run_task()

    assert (shipping_dist / "test.whl").is_file()
    assert not (internal_dist / "test.whl").exists()
