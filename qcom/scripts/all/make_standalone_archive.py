#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from archive_tests import ALWAYS_REJECT_RE, should_archive
from package_manager import PackageManager

QCOM_ROOT = Path(__file__).parent.parent.parent
REPO_ROOT = QCOM_ROOT.parent
EXTRA_CONTENT_ROOT = Path(__file__).parent / "standalone_archive"


def _add_package(archive: zipfile.ZipFile, package_name: str) -> Path:
    pm = PackageManager(package_name)
    pkg_path = pm.fetch()
    relative_path = pkg_path.relative_to(pm.cache.cache_dir)
    archive_path = Path("build") / "package-cache" / relative_path
    logging.debug(f"Adding package {archive_path}")
    archive.write(pkg_path, archive_path)
    return archive_path


def _add_py_installer(
    archive: zipfile.ZipFile, target_platform: str, bootstrap_py_version: str
) -> dict[str, str | list[str]]:
    target_os, _ = target_platform.split("-")

    # This Python is used to run build_and_test.py, which needs to be able to install all
    # of ONNX Runtime's Python dependencies. For the time being, this means it has to be
    # x86_64.
    package_name = f"python_{bootstrap_py_version.replace('.', '')}_{target_os}_x86_64"
    archive_path = _add_package(archive, package_name)

    pm = PackageManager(package_name)
    pkg_config = pm.get_config()

    install_args = [pm.format(a) for a in pkg_config.get("install_args", [])]

    return {
        "installer_path": str(archive_path),
        "install_args": install_args,
        "launcher_version": f"{bootstrap_py_version}",
    }


def _add_venv_wheels(archive: zipfile.ZipFile, target_platform: str, target_py_version: str | None) -> None:
    if target_py_version is None:
        logging.info("Not adding venv wheels because Python was not targeted")
        return
    make_wheels = REPO_ROOT / "qcom" / "scripts" / "windows" / "build_venv_wheels.ps1"
    venv_path = REPO_ROOT / "build" / target_platform / f"venv-{target_py_version}"
    wheels_dir = REPO_ROOT / "build" / target_platform / f"wheels-{target_py_version}"
    make_wheels_cmd = f"powershell.exe {make_wheels} -VenvPath {venv_path} -WheelDir {wheels_dir}"
    logging.info(f"Exporting wheels of all packages found in {venv_path}")
    subprocess.run(make_wheels_cmd, check=True)

    logging.debug(f"Copying wheels to {wheels_dir.relative_to(REPO_ROOT)}")
    for filename in wheels_dir.glob("**/*"):
        archive.write(filename, filename.relative_to(REPO_ROOT))


def main(
    target_platform: str,
    target_py_version: str,
    bootstrap_py_version: str,
    bnt_args: list[str],
    include_packages: list[str],
) -> None:
    build_root = REPO_ROOT / "build"
    base_archive_path = build_root / f"onnxruntime-tests-{target_platform}.zip"
    full_archive_path = build_root / f"onnxruntime-tests-{target_platform}-standalone.zip"

    bootstrap_info: dict[str, Any] = {"build_and_test": {"args": bnt_args}}
    with tempfile.TemporaryDirectory(prefix="MakeStandaloneArchive") as tmpdir:
        tmp_archive_path = Path(tmpdir) / full_archive_path.name

        shutil.copyfile(base_archive_path, tmp_archive_path)

        with zipfile.ZipFile(tmp_archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            _add_venv_wheels(archive, target_platform, target_py_version)
            _add_package(archive, f"cmake_{target_platform.replace('-', '_')}")
            for pkg_name in include_packages:
                _add_package(archive, pkg_name)

            archive_dirs = [
                (QCOM_ROOT, REPO_ROOT),
                (EXTRA_CONTENT_ROOT, EXTRA_CONTENT_ROOT),
                (REPO_ROOT / "tools" / "ci_build" / "github" / "windows" / "python", REPO_ROOT),
            ]
            individual_files = [
                REPO_ROOT / "requirements-dev.txt",
                REPO_ROOT / "requirements-lintrunner.txt",
                REPO_ROOT / ".lintrunner.toml",
            ]
            for content_dir, relative_to_dir in archive_dirs:
                logging.debug(f"Adding {relative_to_dir}")
                for filename in content_dir.glob("**/*"):
                    if should_archive(filename, reject=ALWAYS_REJECT_RE):
                        archive.write(filename, filename.relative_to(relative_to_dir))
            for filename in individual_files:
                logging.debug(f"Adding {filename.name}")
                archive.write(filename, filename.name)
            bootstrap_info["python"] = _add_py_installer(archive, target_platform, bootstrap_py_version)
            archive.writestr("bootstrap.json", json.dumps(bootstrap_info, indent=2))

        shutil.move(tmp_archive_path, full_archive_path)

    logging.info(f"Standalone archive ready in {full_archive_path}")


if __name__ == "__main__":
    log_format = "[%(asctime)s] [make_standalone_archive.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build-and-test-args",
        required=True,
        type=lambda x: x.split(","),
        help="Comma-separated arguments to pass to build_and_test.py when the standalone's run script is invoked.",
    )

    parser.add_argument(
        "--include-packages",
        default=[],
        type=lambda x: x.split(","),
        help="Comma separated list of packages to include in the archive.",
    )

    parser.add_argument(
        "--bootstrap-py-version",
        choices=["3.10", "3.11", "3.12", "3.13"],
        default="3.12",
        help="The version of Python to use to run build_and_test.py.",
    )

    parser.add_argument(
        "--target-py-version",
        choices=["3.10", "3.11", "3.12", "3.13", "None"],
        default="3.12",
        help="The version of Python to use to run build_and_test.py.",
    )

    parser.add_argument(
        "--target-platform",
        help="The platform for which to package tests.",
        choices=[
            "windows-arm64",
            "windows-x86_64",
        ],
        required=True,
    )

    args = parser.parse_args()

    main(
        args.target_platform,
        args.target_py_version,
        args.bootstrap_py_version,
        args.build_and_test_args,
        args.include_packages,
    )
