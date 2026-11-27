# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Check CustomOp package overrides across the Ubuntu Docker task boundary."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

QCOM_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_OVERRIDE = "ORT_QNN_BUILD_UDO_OP_PACKAGES"


@pytest.mark.parametrize(
    "task_name",
    [
        "build_ort_linux_x86_64_ubuntu_22_04",
        "build_ort_linux_x86_64_ubuntu_22_04_internal_symbols",
    ],
)
@pytest.mark.parametrize("override", ["OFF", "ON", None])
def test_ubuntu_docker_preserves_custom_op_override(tmp_path: Path, task_name: str, override: str | None) -> None:
    script = """
import sys
from pathlib import Path

from build_and_test import TaskLibrary
from ep_build.plan import Plan
from ep_build.tasks.docker import DockerRunTask

task_name, temp_dir, override = sys.argv[1:]
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
commands = []

def visit(task):
    if isinstance(task, DockerRunTask):
        args = task.executables_and_args
        commands.extend(args() if callable(args) else args)
    for child in getattr(task, "tasks", []):
        visit(child)

plan.for_each(lambda step_id, task: visit(task))
assert len(commands) == 1
command = commands[0]
forwarded = [
    command[i + 1] for i, arg in enumerate(command[:-1])
    if arg == "--env" and command[i + 1].startswith("ORT_QNN_BUILD_UDO_OP_PACKAGES=")
]
expected = [] if override == "unset" else [f"ORT_QNN_BUILD_UDO_OP_PACKAGES={override}"]
assert forwarded == expected, (forwarded, expected)
"""
    env = os.environ.copy()
    env.pop(PACKAGE_OVERRIDE, None)
    if override is not None:
        env[PACKAGE_OVERRIDE] = override
    env["PYTHONPATH"] = os.pathsep.join([str(QCOM_ROOT), *sys.path])
    result = subprocess.run(
        [sys.executable, "-c", script, task_name, str(tmp_path), override or "unset"],
        cwd=QCOM_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
