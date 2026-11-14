# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Run platform prebuilt-mode entry points with fake tools instead of compiling."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
BASH = shutil.which("bash")
if BASH is None and sys.platform == "win32":
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.is_file():
        BASH = str(git_bash)
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


@pytest.fixture
def bash_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    scripts = tmp_path / "qcom/scripts/linux"
    scripts.mkdir(parents=True)
    for name in ("common.sh", "build.sh", "run_tests.sh"):
        shutil.copy2(REPO_ROOT / "qcom/scripts/linux" / name, scripts / name)
    (scripts / "tools.sh").write_text(
        'get_tools_dir() { echo "${REPO_ROOT}/fake-tools"; }\n'
        "get_cmake_bindir() { get_tools_dir; }\n"
        "get_ninja_bindir() { get_tools_dir; }\n"
        "get_llvm_contentdir() { get_tools_dir; }\n"
        "get_hexagon_sdk_contentdir() { get_tools_dir; }\n"
        'get_ort_x64_prebuilt_root() { echo "Unexpected ORT download" >&2; return 31; }\n'
        'get_onnx_models_dir() { echo "Unexpected model setup" >&2; return 32; }\n',
        newline="\n",
    )
    all_scripts = tmp_path / "qcom/scripts/all"
    all_scripts.mkdir()
    (all_scripts / "python_test_files.txt").touch()
    (all_scripts / "fetch_cmake_deps.py").touch()
    tools = tmp_path / "fake-tools"
    tools.mkdir()
    (tools / "ctest").write_text('#!/usr/bin/env bash\nexit "${FAKE_CTEST_STATUS:-0}"\n', newline="\n")
    (tools / "python").write_text(
        f'#!/usr/bin/env bash\nexec {shlex.quote(Path(sys.executable).as_posix())} "$@"\n', newline="\n"
    )
    for name in ("ctest", "python"):
        (tools / name).chmod(0o755)
    builder = tmp_path / "tools/ci_build/build.py"
    builder.parent.mkdir(parents=True)
    builder.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['FAKE_BUILD_COMMAND']).write_text(json.dumps(sys.argv[1:]))\n"
        "build_dir = Path(sys.argv[sys.argv.index('--build_dir') + 1])\n"
        "config = sys.argv[sys.argv.index('--config') + 1]\n"
        "(build_dir / config / 'build.ninja').touch()\n"
    )
    return tmp_path


def bash_command(script: Path, *args: str) -> list[str]:
    if sys.platform == "win32":
        # Git Bash reports a Windows repo path, while the Linux script expects a
        # POSIX path when assembling PATH. Use the fixture's working directory.
        return [
            BASH,
            "-c",
            'git() { if [[ "$*" == "rev-parse --show-toplevel" ]]; then pwd; else command git "$@"; fi; }; export -f git; exec bash "$@"',
            "bash",
            script.as_posix(),
            *args,
        ]
    return [BASH, script.as_posix(), *args]


@pytest.mark.skipif(BASH is None, reason="Bash is unavailable")
def test_linux_explicit_ort_home_enables_prebuilt_and_source_mode_clears_it(bash_repo: Path) -> None:
    root = bash_repo / "prebuilt ORT"
    root.mkdir()
    command_file = bash_repo / "command.json"
    env = os.environ | {
        "PATH": f"{bash_repo / 'fake-tools'}{os.pathsep}{os.environ['PATH']}",
        "FAKE_BUILD_COMMAND": str(command_file),
        "ORT_BUILD_PRUNE_PACKAGES": "0",
        "ORT_BUILD_XTRACE": "",
        "ORT_NIGHTLY_BUILD": "0",
        "ORT_VERSION_SUFFIX": "",
    }
    command = bash_command(
        bash_repo / "qcom/scripts/linux/build.sh",
        "--target-platform=linux",
        "--target-arch=x86_64",
        "--mode=build",
        "--no-use-cache",
        f"--qairt-sdk-root={bash_repo.as_posix()}",
    )
    result = subprocess.run(
        [*command, f"--ort-home={root.as_posix()}"],
        check=False,
        cwd=bash_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(command_file.read_text())
    assert Path(args[args.index("--ort_home") + 1]) == root
    marker = bash_repo / "build/linux-x86_64/qnn-build-mode-Release.txt"
    assert Path(marker.read_text().strip().split(";ort_home=", 1)[1]) == root

    result = subprocess.run(command, check=False, cwd=bash_repo, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(command_file.read_text())
    assert "--ort_home" not in args
    assert "onnxruntime_ORT_HOME:PATH=" in args
    assert ";ort_home=" not in marker.read_text()


@pytest.mark.skipif(BASH is None, reason="Bash is unavailable")
@pytest.mark.parametrize(("ctest_status", "expected_status"), [(0, 0), (17, 1)])
def test_linux_skip_models_preserves_ctest_status_without_model_setup(
    bash_repo: Path, ctest_status: int, expected_status: int
) -> None:
    binary_dir = bash_repo / "build/test"
    binary_dir.mkdir(parents=True)
    shutil.copy2(bash_repo / "qcom/scripts/linux/run_tests.sh", binary_dir)
    shutil.copy2(bash_repo / "fake-tools/ctest", binary_dir)
    (binary_dir / "CTestTestfile.cmake").write_text(f"# Build directory: {binary_dir.as_posix()}\n")
    result = subprocess.run(
        bash_command(binary_dir / "run_tests.sh", "--skip-model-tests"),
        check=False,
        cwd=bash_repo,
        env=os.environ | {"FAKE_CTEST_STATUS": str(ctest_status), "ORT_BUILD_XTRACE": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert "Skipping ONNX model tests" in result.stderr
    assert "Unexpected model setup" not in result.stderr


@pytest.fixture
def windows_repo(tmp_path: Path) -> Path:
    scripts = tmp_path / "qcom/scripts/windows"
    scripts.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "qcom/scripts/windows/build.ps1", scripts)
    (scripts / "tools.ps1").write_text(
        'function Get-QairtSdkVersion { param($QairtSdkRoot) "test" }\n'
        'function Get-DefaultCMakeGenerator { param($Arch) "Ninja" }\n'
        "function Get-CCacheBinDir { $env:FAKE_WINDOWS_TOOLS }\n"
        "function Get-CMakeBinDir { $env:FAKE_WINDOWS_TOOLS }\n"
        "function Get-NinjaBinDir { $env:FAKE_WINDOWS_TOOLS }\n"
        "function Get-OnnxModelsRoot { $env:FAKE_WINDOWS_TOOLS }\n"
        'function Get-OrtARM64PrebuiltRoot { throw "Unexpected ARM64 ORT download" }\n'
        'function Get-OrtX64PrebuiltRoot { throw "Unexpected x64 ORT download" }\n'
    )
    (scripts / "utils.ps1").write_text(
        "function Enter-PyVenv { param($PyVEnv) }\n"
        "function Enter-MsvcEnv { param($TargetArch) }\n"
        "function Test-UpdateNeeded { param($BuildDir, $Config, $TargetPyVersion, $QairtSdkRoot, $CMakeGenerator, $Update) $false }\n"
    )
    (scripts / "run_tests.ps1").write_text(
        "param($Config, $OnnxModelsRoot, [switch]$SkipModelTests)\n"
        "Set-Content -LiteralPath $env:FAKE_WINDOWS_TEST_LOG -Value $SkipModelTests.IsPresent\n"
    )
    all_scripts = tmp_path / "qcom/scripts/all"
    all_scripts.mkdir()
    (all_scripts / "python_test_files.txt").touch()
    tools = tmp_path / "fake-tools"
    tools.mkdir()
    (tools / "ctest.exe").touch()
    (tmp_path / "prebuilt ORT").mkdir()
    return tmp_path


def run_windows(repo: Path, arch: str, *, explicit: bool, automatic: bool = False) -> subprocess.CompletedProcess[str]:
    script = """
$ErrorActionPreference = 'Stop'
$params = @{ Arch = $env:FAKE_WINDOWS_ARCH; Mode = 'test'; PyVEnv = $env:FAKE_WINDOWS_TOOLS; QairtSdkRoot = $env:FAKE_WINDOWS_TOOLS }
if ($env:FAKE_WINDOWS_EXPLICIT -eq '1') { $params.OrtPrebuiltRoot = $env:FAKE_WINDOWS_ORT }
if ($env:FAKE_WINDOWS_AUTOMATIC -eq '1') { $params.UseOrtPrebuilt = $true }
. $env:FAKE_WINDOWS_SCRIPT @params
@{ prebuilt = $UseOrtPrebuilt; args = @($QnnArgs); mode = $OrtBuildMode } | ConvertTo-Json | Set-Content -LiteralPath $env:FAKE_WINDOWS_COMMAND
"""
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-Command", script],
        check=False,
        cwd=repo,
        env=os.environ
        | {
            "FAKE_WINDOWS_ARCH": arch,
            "FAKE_WINDOWS_SCRIPT": str(repo / "qcom/scripts/windows/build.ps1"),
            "FAKE_WINDOWS_TOOLS": str(repo / "fake-tools"),
            "FAKE_WINDOWS_ORT": str(repo / "prebuilt ORT"),
            "FAKE_WINDOWS_EXPLICIT": "1" if explicit else "0",
            "FAKE_WINDOWS_AUTOMATIC": "1" if automatic else "0",
            "FAKE_WINDOWS_COMMAND": str(repo / "command.json"),
            "FAKE_WINDOWS_TEST_LOG": str(repo / "tests.txt"),
            "ORT_BUILD_PRUNE_PACKAGES": "0",
            "ORT_NIGHTLY_BUILD": "0",
            "ORT_VERSION_SUFFIX": "",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.skipif(POWERSHELL is None or sys.platform != "win32", reason="Windows PowerShell is unavailable")
@pytest.mark.parametrize("arch", ["aarch64", "arm64", "x86_64"])
def test_windows_explicit_ort_root_enables_prebuilt_and_skips_models(windows_repo: Path, arch: str) -> None:
    result = run_windows(windows_repo, arch, explicit=True)
    assert result.returncode == 0, result.stdout + result.stderr
    command = json.loads((windows_repo / "command.json").read_text(encoding="utf-8-sig"))
    assert command["prebuilt"]
    assert Path(command["args"][command["args"].index("--ort_home") + 1]) == windows_repo / "prebuilt ORT"
    assert (windows_repo / "tests.txt").read_text(encoding="utf-8-sig").strip() == "True"


@pytest.mark.skipif(POWERSHELL is None or sys.platform != "win32", reason="Windows PowerShell is unavailable")
def test_windows_source_mode_clears_prebuilt_root(windows_repo: Path) -> None:
    result = run_windows(windows_repo, "x86_64", explicit=False)
    assert result.returncode == 0, result.stdout + result.stderr
    command = json.loads((windows_repo / "command.json").read_text(encoding="utf-8-sig"))
    assert not command["prebuilt"]
    assert "--ort_home" not in command["args"]
    assert "onnxruntime_ORT_HOME:PATH=" in command["args"]
    assert (windows_repo / "tests.txt").read_text(encoding="utf-8-sig").strip() == "False"


@pytest.mark.skipif(POWERSHELL is None or sys.platform != "win32", reason="Windows PowerShell is unavailable")
def test_windows_automatic_prebuilt_rejects_arm64ec(windows_repo: Path) -> None:
    result = run_windows(windows_repo, "arm64ec", explicit=False, automatic=True)
    assert result.returncode != 0
    assert "Unknown arch arm64ec for ORT prebuilt" in result.stderr
    assert "Unexpected ARM64 ORT download" not in result.stderr
