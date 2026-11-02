#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PREBUILT_CMAKE = REPO_ROOT / "cmake" / "external" / "onnxruntime_prebuilt.cmake"
BUILD_AND_TEST = REPO_ROOT / "qcom" / "build_and_test.py"
ORT_VERSION = "1.24.4"


def _write(path: Path, content: str = "smoke") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_prebuilt_sdk(root: Path) -> list[str]:
    _write(root / "VERSION_NUMBER", ORT_VERSION)
    _write(root / "include" / "onnxruntime_c_api.h")
    _write(root / "include" / "onnxruntime_cxx_api.h")

    if sys.platform == "win32":
        libraries = [
            "onnxruntime.dll",
            "onnxruntime.lib",
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_shared.lib",
        ]
    else:
        libraries = [
            "libonnxruntime.so",
            "libonnxruntime.so.1",
            f"libonnxruntime.so.{ORT_VERSION}",
            "libonnxruntime_providers_shared.so",
        ]

    for library in libraries:
        _write(root / "lib" / library)
    return libraries


class PrebuiltCmakeSmokeTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux-only task graph")
    def test_composite_test_tasks_require_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ort-prebuilt-task-smoke-") as temp_dir:
            sdk = Path(temp_dir) / "sdk"
            _create_prebuilt_sdk(sdk)

            for task in (
                "asan_linux_x86_64",
                "coverage_linux_x86_64",
                "coverage_report_linux_x86_64",
                "diff_coverage_linux_x86_64",
                "file_coverage_excel_linux_x86_64",
            ):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(BUILD_AND_TEST),
                        "--dry-run",
                        "--ort-prebuilt",
                        str(sdk),
                        task,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertNotEqual(result.returncode, 0, task)
                self.assertIn("Prebuilt ORT test tasks require --ort-test-assets", result.stderr, task)

    def test_prebuilt_target_copies_without_source_patch(self) -> None:
        cmake = os.environ.get("CMAKE_EXECUTABLE") or shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake is not available")

        with tempfile.TemporaryDirectory(prefix="ort-prebuilt-smoke-") as temp_dir:
            root = Path(temp_dir)
            sdk = root / "sdk"
            test_assets = root / "test-assets"
            libraries = _create_prebuilt_sdk(sdk)
            _write(test_assets / "VERSION_NUMBER", ORT_VERSION)
            _write(test_assets / "onnxruntime" / "test" / "testdata" / "model.onnx")

            cmake_lists = f'''cmake_minimum_required(VERSION 3.26)
project(ort_prebuilt_smoke NONE)
set(onnxruntime_ORT_HOME "{sdk.as_posix()}")
set(ORT_CORE_VER "{ORT_VERSION}")
set(CMAKE_BUILD_TYPE "Release")
set(Patch_EXECUTABLE "patch")
include("{PREBUILT_CMAKE.as_posix()}")
'''
            _write(root / "CMakeLists.txt", cmake_lists)

            build_dir = root / "build"
            env = os.environ.copy()
            env["ORT_TEST_ASSETS_ROOT"] = str(test_assets)
            configure_command = [cmake, "-S", str(root), "-B", str(build_dir)]
            ninja = os.environ.get("NINJA_EXECUTABLE") or shutil.which("ninja")
            if ninja is not None:
                configure_command.extend(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"])
            subprocess.run(
                configure_command,
                check=True,
                capture_output=True,
                env=env,
                text=True,
                timeout=120,
            )
            subprocess.run(
                [cmake, "--build", str(build_dir), "--target", "ort_core_target"],
                check=True,
                capture_output=True,
                env=env,
                text=True,
                timeout=120,
            )

            for library in libraries:
                self.assertTrue((build_dir / library).is_file(), library)

            generated_build_text = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in build_dir.rglob("*")
                if path.is_file() and path.suffix in {".ninja", ".make"}
            )
            self.assertNotIn("patches/ort_core", generated_build_text.replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
