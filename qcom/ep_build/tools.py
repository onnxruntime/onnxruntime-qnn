# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import logging
import os
from pathlib import Path
from typing import Literal

from .tasks.build import TargetPyVersionT
from .util import (
    REPO_ROOT,
    is_host_arm64,
    is_host_mac,
    is_host_windows,
    process_output,
    run_with_venv,
    run_with_venv_and_get_output,
)

PACKAGE_MANAGER = REPO_ROOT / "qcom" / "scripts" / "all" / "package_manager.py"

ORT_BUILD_UV_CACHE_PATH: Path = Path(
    os.environ.get("ORT_BUILD_UV_CACHE_PATH", str((Path("~") / ".ort-uv-cache").expanduser()))
)

_UV_EXE: Path | None = None
_UV_CACHE_PRUNED: bool = False


def _uv_package_name() -> str:
    if is_host_windows():
        return "uv_windows_arm64" if is_host_arm64() else "uv_windows_x86_64"
    if is_host_mac():
        return "uv_macos_arm64" if is_host_arm64() else "uv_macos_x86_64"
    return "uv_linux_aarch64" if is_host_arm64() else "uv_linux_x86_64"


def is_managed_uv_ready() -> bool:
    """Return True if the managed uv binary has already been set up for this process."""
    return _UV_EXE is not None


def get_uv_executable(bootstrap_venv: Path) -> Path:
    """Return the path to the managed uv binary, installing it via PackageManager if needed.

    On first call also prepends uv's bin directory to PATH and sets UV_CACHE_DIR.
    """
    global _UV_EXE
    if _UV_EXE is not None:
        return _UV_EXE

    bin_dir = get_package_bin_dir(bootstrap_venv, _uv_package_name())
    suffix = ".exe" if is_host_windows() else ""
    uv_exe = bin_dir / f"uv{suffix}"

    ORT_BUILD_UV_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("UV_CACHE_DIR", str(ORT_BUILD_UV_CACHE_PATH))
    path_sep = ";" if is_host_windows() else ":"
    os.environ["PATH"] = str(bin_dir) + path_sep + os.environ.get("PATH", "")

    _UV_EXE = uv_exe
    return uv_exe


def setup_managed_uv(bootstrap_venv: Path) -> None:
    """Configure managed uv for this process: install it, set PATH/UV_CACHE_DIR, and prune cache.

    Idempotent; the cache prune runs only once per process execution.
    """
    global _UV_CACHE_PRUNED
    uv_exe = get_uv_executable(bootstrap_venv)
    if not _UV_CACHE_PRUNED:
        run_with_venv(None, [str(uv_exe), "cache", "prune"], check=False)
        _UV_CACHE_PRUNED = True


def get_model_zoo_root(package_manager_venv: Path | None) -> Path:
    return get_package_content_dir(package_manager_venv, "model_zoo")


def get_onnx_models_root(package_manager_venv: Path | None) -> Path:
    return get_package_content_dir(package_manager_venv, "onnx_models")


def get_qualcomm_device_cloud_sdk_root(package_manager_venv: Path | None) -> Path:
    return get_package_content_dir(package_manager_venv, "qualcomm_device_cloud_sdk")


def get_package_bin_dir(package_manager_venv: Path | None, package: str) -> Path:
    install_package(package_manager_venv, package)
    return Path(_package_action(package_manager_venv, package, "print-bin-dir"))


def get_package_content_dir(package_manager_venv: Path | None, package: str) -> Path:
    install_package(package_manager_venv, package)
    return Path(_package_action(package_manager_venv, package, "print-content-dir"))


PythonExecutableArchT = Literal["arm64", "x86_64"]


def get_python_executable(
    package_manager_venv: Path | None,
    arch: PythonExecutableArchT,
    version: TargetPyVersionT,
) -> Path:
    if not is_host_windows():
        return Path(f"python{version}")

    py_package = f"python_{version.replace('.', '')}_windows_{arch}"

    def _query_launcher():
        py_vsn = version if arch == "x86_64" else f"{version}-{arch}"
        get_exe_cmd = ["py", f"-{py_vsn}", "-c", "import sys; print(sys.executable)"]
        get_exe_result = run_with_venv(None, get_exe_cmd, check=False, capture_output=True)
        return (get_exe_result.returncode, process_output(get_exe_result))

    rc, py_exe = _query_launcher()
    # See https://docs.python.org/3/using/windows.html#return-codes
    # 101 --> Failed to launch Python
    # 103 --> Unable to locate the requested version
    if rc == 0:
        return Path(py_exe)
    elif rc == 101:
        logging.info("Looks like this python version has been clobbered")
        repair_package(package_manager_venv, py_package)
    elif rc == 103:
        logging.info("This version has not been installed")
        install_package(package_manager_venv, py_package)
    else:
        raise RuntimeError(f"Unknown return code {rc} from py launcher")

    # We've hopefully installed/repaired Python at this point
    rc, py_exe = _query_launcher()
    if rc != 0:
        raise RuntimeError("Could not find Python executable, even after attempting to install/repair it.")
    return Path(py_exe)


def get_package_dir() -> Path:
    return Path(
        os.environ.get(
            "ORT_BUILD_PACKAGE_CACHE_PATH",
            str((Path("~") / ".ort-package-cache").expanduser()),
        )
    )


def get_tools_dir() -> Path:
    tools_name = "Tools" if is_host_windows() else "tools"
    tools_dir = Path(os.getenv("ORT_BUILD_TOOLS_PATH", REPO_ROOT / "build" / tools_name))
    tools_dir.mkdir(parents=True, exist_ok=True)
    return tools_dir


def install_package(package_manager_venv: Path | None, package: str) -> None:
    _package_action(package_manager_venv, package, "install")


def repair_package(package_manager_venv: Path | None, package: str) -> None:
    _package_action(package_manager_venv, package, "repair")


def _package_action(
    package_manager_venv: Path | None,
    package: str,
    action: str,
) -> str:
    return run_with_venv_and_get_output(
        package_manager_venv,
        ["python", str(PACKAGE_MANAGER), f"--{action}", f"--package={package}", f"--package-root={get_tools_dir()}"],
    )
