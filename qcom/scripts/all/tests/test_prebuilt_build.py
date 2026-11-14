# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Verify that Linux CI build tasks preserve prebuilt ORT options through Docker."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_and_test import TaskLibrary
from ep_build import tools
from ep_build.plan import Plan
from ep_build.tasks import docker


@pytest.mark.parametrize(
    "task_name",
    [
        "build_ort_linux_aarch64_manylinux_2_34",
        "build_ort_linux_aarch64_manylinux_2_34_internal_symbols",
        "build_ort_linux_x86_64_ubuntu_22_04",
    ],
)
@pytest.mark.parametrize("mode", ["source", "download", "explicit"])
def test_linux_ci_build_preserves_ort_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_name: str, mode: str
) -> None:
    monkeypatch.setattr(docker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tools, "get_package_dir", lambda: tmp_path / "packages")
    monkeypatch.setattr(tools, "get_tools_dir", lambda: tmp_path / "tools")
    monkeypatch.setattr(tools, "get_uv_cache_dir", lambda: tmp_path / "uv")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    root = tmp_path / "prebuilt ort" if mode == "explicit" else None
    library = TaskLibrary(
        python_executable=Path(sys.executable),
        venv_path=tmp_path / "venv",
        config="Release",
        target_py_version="3.12",
        use_ort_prebuilt=mode != "source",
        ort_prebuilt_root=root,
        qairt_sdk_root=None,
        docker_ccache_root=None,
        build_nuget=False,
        build_archive=False,
        build_aar=False,
    )
    plan = Plan()
    getattr(library, task_name)(plan)
    commands = []
    plan.for_each(lambda _step_id, task: commands.extend(task.tasks[-1].executables_and_args()))
    assert len(commands) == 1
    command = commands[0]
    assert command[:2] == ["docker", "run"]
    assert ("--use-ort-prebuilt" in command) == (mode != "source")
    assert "--target-py-version=3.12" in command
    assert str(docker.DOCKER_REPO_ROOT / "qcom/build_and_test.py") in command
    if root is not None:
        assert f"{root.resolve()}:{Path('/ort_caches/ort_prebuilt')}" in command
        assert "--ort-prebuilt=/ort_caches/ort_prebuilt" in command
    else:
        assert not any(arg.startswith("--ort-prebuilt=") for arg in command)
