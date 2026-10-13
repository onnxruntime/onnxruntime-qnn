# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Exercise build.sh mode transitions without downloading SDKs or compiling ORT."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

QCOM_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(QCOM_ROOT))

from build_and_test import TaskLibrary  # noqa: E402
from ep_build.plan import Plan  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "build.sh"
MODE_FLAGS = {
    "normal": [],
    "internal": ["--enable-qnn-internal-test-symbols"],
    "coverage": ["--enable-coverage"],
}


@pytest.fixture
def build_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    scripts = tmp_path / "qcom/scripts/linux"
    scripts.mkdir(parents=True)
    shutil.copy2(SCRIPT.parent / "common.sh", scripts / "common.sh")
    (scripts / "tools.sh").write_text(
        'get_tools_dir() { echo "${REPO_ROOT}/fake-tools"; }\n'
        "get_cmake_bindir() { get_tools_dir; }\n"
        "get_ninja_bindir() { get_tools_dir; }\n"
        "get_llvm_contentdir() { get_tools_dir; }\n"
        "get_hexagon_sdk_contentdir() { get_tools_dir; }\n"
    )
    (scripts / "run_tests.sh").touch()
    all_scripts = tmp_path / "qcom/scripts/all"
    all_scripts.mkdir()
    (all_scripts / "python_test_files.txt").touch()
    (all_scripts / "fetch_cmake_deps.py").touch()
    tools = tmp_path / "fake-tools"
    tools.mkdir()
    (tools / "ctest").touch()
    (tools / "python").symlink_to(sys.executable)
    builder = tmp_path / "tools/ci_build/build.py"
    builder.parent.mkdir(parents=True)
    builder.write_text(
        """import argparse
import json
import os
import shutil
import sys
from pathlib import Path

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--build_dir", type=Path)
parser.add_argument("--config")
parser.add_argument("--update", action="store_true")
parser.add_argument("--build", action="store_true")
args, _ = parser.parse_known_args()
config_dir = args.build_dir / args.config
marker = args.build_dir / f"qnn-build-mode-{args.config}.txt"
with (args.build_dir / "calls.jsonl").open("a") as log:
    log.write(json.dumps({"update": args.update, "marker_exists": marker.exists()}) + "\\n")

# Model configure changing persistent settings before compilation can fail.
configured = config_dir / "configured-mode.json"
if args.update:
    definitions = dict(
        arg.split("=", 1)
        for arg in sys.argv[1:]
        if arg.startswith(("ENABLE_COVERAGE:BOOL=", "onnxruntime_QNN_ENABLE_INTERNAL_TEST_SYMBOLS:BOOL="))
    )
    configured.write_text(json.dumps(definitions))
    (config_dir / "build.ninja").touch()
if os.environ["FAKE_BUILD_FAIL"] == "1":
    sys.exit(17)
if args.build:
    shutil.copyfile(configured, config_dir / "built-mode.json")
"""
    )
    return tmp_path


def run_build(
    repo: Path, mode: str, *, fail: bool = False, build_dir_suffix: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "PATH": f"{repo / 'fake-tools'}{os.pathsep}{os.environ['PATH']}",
        "ORT_BUILD_PRUNE_PACKAGES": "0",
        "ORT_BUILD_XTRACE": "",
        "ORT_NIGHTLY_BUILD": "0",
        "ORT_VERSION_SUFFIX": "",
        "FAKE_BUILD_FAIL": "1" if fail else "0",
    }
    args = [
        "bash",
        str(SCRIPT),
        "--target-platform=linux",
        "--target-arch=x86_64",
        "--config=RelWithDebInfo",
        "--mode=build",
        "--no-use-cache",
        f"--qairt-sdk-root={repo / 'fake-qairt'}",
        *MODE_FLAGS[mode],
    ]
    if build_dir_suffix is not None:
        args.append(f"--build-dir-suffix={build_dir_suffix}")

    return subprocess.run(
        args,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def assert_built_mode(repo: Path, mode: str) -> None:
    root = repo / "build/linux-x86_64"
    coverage = "ON" if mode == "coverage" else "OFF"
    symbols = "OFF" if mode == "normal" else "ON"
    expected = {
        "ENABLE_COVERAGE:BOOL": coverage,
        "onnxruntime_QNN_ENABLE_INTERNAL_TEST_SYMBOLS:BOOL": symbols,
    }
    for name in ("configured-mode.json", "built-mode.json"):
        assert json.loads((root / "RelWithDebInfo" / name).read_text()) == expected
    assert (root / "qnn-build-mode-RelWithDebInfo.txt").read_text().strip() == (
        f"coverage={coverage};internal_symbols={symbols}"
    )


@pytest.mark.parametrize("mode", MODE_FLAGS)
def test_successful_build_reuses_unchanged_mode(build_repo: Path, mode: str) -> None:
    for _ in range(2):
        result = run_build(build_repo, mode)
        assert result.returncode == 0, result.stderr
        assert_built_mode(build_repo, mode)

    calls = (build_repo / "build/linux-x86_64/calls.jsonl").read_text().splitlines()
    assert [json.loads(call) for call in calls] == [
        {"update": True, "marker_exists": False},
        {"update": False, "marker_exists": True},
    ]


def test_successful_mode_switches_reconfigure_and_replace_marker(build_repo: Path) -> None:
    for mode in ("normal", "coverage", "internal", "normal"):
        result = run_build(build_repo, mode)
        assert result.returncode == 0, result.stderr
        assert_built_mode(build_repo, mode)

    calls = (build_repo / "build/linux-x86_64/calls.jsonl").read_text().splitlines()
    assert len(calls) == 4
    assert all(json.loads(call) == {"update": True, "marker_exists": False} for call in calls)


@pytest.mark.parametrize("initial_mode", ["normal", "coverage"])
@pytest.mark.parametrize("retry_original_mode", [False, True], ids=["retry-new-mode", "restore-original-mode"])
def test_failed_switch_forces_reconfigure_on_retry(
    build_repo: Path, initial_mode: str, retry_original_mode: bool
) -> None:
    result = run_build(build_repo, initial_mode)
    assert result.returncode == 0, result.stderr

    result = run_build(build_repo, "internal", fail=True)
    assert result.returncode == 17, result.stderr
    root = build_repo / "build/linux-x86_64"
    marker_after_failure = (root / "qnn-build-mode-RelWithDebInfo.txt").exists()
    assert json.loads((root / "RelWithDebInfo/configured-mode.json").read_text()) == {
        "ENABLE_COVERAGE:BOOL": "OFF",
        "onnxruntime_QNN_ENABLE_INTERNAL_TEST_SYMBOLS:BOOL": "ON",
    }

    retry_mode = initial_mode if retry_original_mode else "internal"
    result = run_build(build_repo, retry_mode)
    assert result.returncode == 0, result.stderr
    assert_built_mode(build_repo, retry_mode)
    assert not marker_after_failure
    calls = (root / "calls.jsonl").read_text().splitlines()
    assert len(calls) == 3
    assert all(json.loads(call) == {"update": True, "marker_exists": False} for call in calls)


def test_internal_symbol_public_task_passes_required_build_flags(tmp_path: Path) -> None:
    task_library = TaskLibrary(
        python_executable=Path(sys.executable),
        venv_path=tmp_path / "venv",
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
    task_library.build_ort_linux_x86_64_internal_symbols(plan)

    planned_tasks = []
    plan.for_each(lambda step_id, task: planned_tasks.append((step_id, task)))
    assert len(planned_tasks) == 1
    step_id, task = planned_tasks[0]
    assert step_id == "build_ort_linux_x86_64_internal_symbols"
    command = task.executables_and_args[0]
    assert "--enable-qnn-internal-test-symbols" in command
    assert "--build-dir-suffix=internal-symbols" in command


def test_build_dir_suffix_uses_isolated_directory(build_repo: Path) -> None:
    result = run_build(build_repo, "internal", build_dir_suffix="internal-symbols")
    assert result.returncode == 0, result.stderr

    root = build_repo / "build/linux-x86_64-internal-symbols"
    assert (root / "RelWithDebInfo/built-mode.json").is_file()
    assert not (build_repo / "build/linux-x86_64").exists()


@pytest.mark.parametrize("suffix", ["../escape", "bad suffix", "/absolute"])
def test_build_dir_suffix_rejects_unsafe_values(build_repo: Path, suffix: str) -> None:
    result = run_build(build_repo, "internal", build_dir_suffix=suffix)
    assert result.returncode != 0
    assert f"Invalid --build-dir-suffix: {suffix}" in result.stderr
    assert not (build_repo / "build").exists()
