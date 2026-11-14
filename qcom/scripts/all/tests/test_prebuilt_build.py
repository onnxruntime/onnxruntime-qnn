# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Verify that Linux CI build tasks preserve prebuilt ORT options through Docker."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import build_and_test
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
    library = build_and_test.TaskLibrary(
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


@pytest.fixture
def planned_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[bool, Path | None]]:
    calls = []

    def planner(*args):
        calls.append((args[5], args[6]))
        return Plan()

    monkeypatch.setattr(build_and_test, "initialize_logging", lambda: None)
    monkeypatch.setattr(build_and_test, "get_qairt_sdk_root", lambda _root: None)
    monkeypatch.setattr(build_and_test, "get_docker_ccache_root", lambda _root: None)
    monkeypatch.setattr(build_and_test, "plan_from_task_list", planner)
    monkeypatch.setattr(Plan, "print", lambda _plan: None)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv"))
    return calls


@pytest.mark.parametrize(
    ("environment", "explicit", "download", "expected_prebuilt", "expected_root"),
    [
        (None, False, False, False, None),
        ("", False, False, False, None),
        ("environment", False, False, True, "environment"),
        ("environment", True, False, True, "explicit"),
        (None, True, False, True, "explicit"),
        (None, False, True, True, None),
        ("environment", False, True, True, "environment"),
    ],
)
def test_cli_ort_prebuilt_root_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planned_cli: list[tuple[bool, Path | None]],
    environment: str | None,
    explicit: bool,
    download: bool,
    expected_prebuilt: bool,
    expected_root: str | None,
) -> None:
    roots = {name: tmp_path / f"{name} ORT" for name in ("environment", "explicit")}
    for root in roots.values():
        root.mkdir()
    if environment is None:
        monkeypatch.delenv("ORT_PREBUILT_ROOT", raising=False)
    else:
        monkeypatch.setenv("ORT_PREBUILT_ROOT", str(roots[environment]) if environment else "")
    argv = ["build_and_test.py", "--only", "--dry-run", "build_ort_windows_x86_64"]
    if explicit:
        argv += ["--ort-prebuilt", str(roots["explicit"])]
    if download:
        argv.append("--use-ort-prebuilt")
    monkeypatch.setattr(sys, "argv", argv)

    build_and_test.build_and_test()

    assert planned_cli == [(expected_prebuilt, roots[expected_root] if expected_root else None)]


def test_cli_missing_environment_ort_root_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planned_cli: list[tuple[bool, Path | None]]
) -> None:
    monkeypatch.setenv("ORT_PREBUILT_ROOT", str(tmp_path / "missing ORT"))
    monkeypatch.setattr(sys, "argv", ["build_and_test.py", "--only", "--dry-run", "build_ort_windows_x86_64"])

    with pytest.raises(FileNotFoundError, match="ORT Prebuilt root"):
        build_and_test.build_and_test()

    assert not planned_cli
