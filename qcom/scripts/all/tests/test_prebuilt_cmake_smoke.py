#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[4]
PREBUILT_CMAKE = REPO_ROOT / "cmake" / "external" / "onnxruntime_prebuilt.cmake"
ACQUISITION_CMAKE = REPO_ROOT / "cmake" / "external" / "onnxruntime_ort_core.cmake"
BUILD_AND_TEST = REPO_ROOT / "qcom" / "build_and_test.py"
IDENTITY_SCRIPT = REPO_ROOT / "qcom" / "scripts" / "all" / "ort_build_identity.py"
FETCH_DEPS_SCRIPT = REPO_ROOT / "qcom" / "scripts" / "all" / "fetch_cmake_deps.py"
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


def _create_runner(sdk: Path) -> str:
    name = "onnxruntime_plugin_ep_onnx_test" + (".exe" if sys.platform == "win32" else "")
    _write(sdk / "bin" / name, "#!/bin/sh\nexit 0\n")
    (sdk / "bin" / name).chmod(0o755)
    return name


def _configure(root: Path, env: dict[str, str], *definitions: str) -> subprocess.CompletedProcess[str]:
    cmake = os.environ.get("CMAKE_EXECUTABLE") or shutil.which("cmake")
    if cmake is None:
        raise unittest.SkipTest("cmake is not available")
    command = [cmake, "-S", str(root / "cmake"), "-B", str(root / "build"), *definitions]
    ninja = os.environ.get("NINJA_EXECUTABLE") or shutil.which("ninja")
    if ninja is not None:
        command.extend(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"])
    return subprocess.run(command, check=False, capture_output=True, env=env, text=True, timeout=120)


def _create_project(root: Path, sdk: Path, *, copy_target: bool = True) -> None:
    cmake_lists = f'''cmake_minimum_required(VERSION 3.26)
project(ort_prebuilt_smoke NONE)
set(REPO_ROOT "${{PROJECT_SOURCE_DIR}}/..")
set(onnxruntime_ORT_HOME "{sdk.as_posix()}")
set(CMAKE_BUILD_TYPE "Release")
function(onnxruntime_fetchcontent_declare)
  message(FATAL_ERROR "ORT source fetch tripwire")
endfunction()
function(FetchContent_Populate)
  message(FATAL_ERROR "ORT source populate tripwire")
endfunction()
include("{ACQUISITION_CMAKE.as_posix()}")
file(WRITE "${{CMAKE_BINARY_DIR}}/ort-version.txt" "${{ORT_CORE_VER}}")
'''
    if copy_target:
        cmake_lists += f'''include("{PREBUILT_CMAKE.as_posix()}")
file(WRITE "${{CMAKE_BINARY_DIR}}/selected-assets.txt" "${{onnxruntime_ORT_TEST_ASSETS_HOME}}")
'''
    _write(root / "cmake" / "CMakeLists.txt", cmake_lists)
    shutil.copyfile(REPO_ROOT / "requirements.txt", root / "requirements.txt")


class PrebuiltCmakeSmokeTest(unittest.TestCase):
    def test_dependency_preparation_skips_only_replaced_ort_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cached_file = root / "cached.zip"
            _write(cached_file)
            cache = Mock()
            cache.fetch.return_value = cached_file
            package_manager = Mock(DEFAULT_PACKAGE_CACHE_DIR=root, FileCache=Mock(return_value=cache))
            spec = importlib.util.spec_from_file_location("prebuilt_dependency_fetch", FETCH_DEPS_SCRIPT)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            fetch_deps = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"package_manager": package_manager}):
                spec.loader.exec_module(fetch_deps)
            dependencies = [
                ("ort_core", "https://example.com/ort.zip", "ort-sha1"),
                ("eigen", "https://example.com/eigen.zip", "eigen-sha1"),
            ]
            for ort_home, expected in ((None, ["ort_core", "eigen"]), (root / "sdk", ["eigen"])):
                with self.subTest(prebuilt=ort_home is not None):
                    cache.fetch.reset_mock()
                    with patch.object(fetch_deps, "_parse_deps_txt", return_value=dependencies):
                        fetch_deps.main(root / "cache", root / "mirror", ort_home=ort_home)
                    self.assertEqual([call.args[0] for call in cache.fetch.call_args_list], expected)
                    self.assertEqual((root / "mirror/example.com/ort.zip").exists(), ort_home is None)

    def test_acquisition_version_validation_without_fetch(self) -> None:
        for version, error in (
            (ORT_VERSION, None),
            (None, "Prebuilt ORT version file not found"),
            ("not-a-version", "Invalid prebuilt ORT version"),
            ("1.0.0", "is unsupported"),
        ):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                sdk = root / "sdk"
                sdk.mkdir()
                if version is not None:
                    _write(sdk / "VERSION_NUMBER", version)
                _create_project(root, sdk, copy_target=False)
                result = _configure(root, os.environ.copy())
                output = result.stdout + result.stderr
                self.assertNotIn("tripwire", output)
                if error is None:
                    self.assertEqual(result.returncode, 0, output)
                    self.assertEqual((root / "build" / "ort-version.txt").read_text(), ORT_VERSION)
                else:
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(error, output)

    def test_asset_transitions_stage_without_relink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sdk = root / "sdk"
            libraries = _create_prebuilt_sdk(sdk)
            runner = _create_runner(sdk)
            _create_project(root, sdk)
            env = os.environ.copy()
            env.pop("ORT_TEST_ASSETS_ROOT", None)
            result = _configure(root, env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cmake = os.environ.get("CMAKE_EXECUTABLE") or shutil.which("cmake")

            for name in ("assets-A", "assets-B"):
                assets = root / name
                _write(assets / "VERSION_NUMBER", ORT_VERSION)
                _write(assets / "onnxruntime/test/testdata/model.onnx", name)
                _write(assets / f"onnxruntime/test/testdata/{name}/removed-or-added.onnx", name)
                env["ORT_TEST_ASSETS_ROOT"] = str(assets)
                result = _configure(root, env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((root / "build/selected-assets.txt").read_text(), assets.as_posix())
                result = subprocess.run(
                    [cmake, "--build", str(root / "build"), "--target", "ort_test_assets_target"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((root / "build/testdata/model.onnx").read_text(), name)
                self.assertEqual(
                    sorted(
                        path.relative_to(root / "build/testdata").as_posix()
                        for path in (root / "build/testdata").rglob("*.onnx")
                    ),
                    [f"{name}/removed-or-added.onnx", "model.onnx"],
                )
                self.assertTrue((root / "build" / runner).is_file())
                for library in libraries:
                    self.assertTrue((root / "build" / library).is_file(), library)

                generated_build_text = "\n".join(
                    path.read_text(encoding="utf-8", errors="ignore")
                    for path in (root / "build").rglob("*")
                    if path.is_file() and path.suffix in {".ninja", ".make"}
                )
                self.assertNotIn("patches/ort_core", generated_build_text.replace("\\", "/"))

            # An explicit definition wins over the environment, including an empty path.
            result = _configure(root, env, "-Donnxruntime_ORT_TEST_ASSETS_HOME:PATH=")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "build/selected-assets.txt").read_text(), "")

    def test_task_preflight_requires_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sdk = root / "sdk"
            assets = root / "assets"
            _create_prebuilt_sdk(sdk)
            _write(assets / "VERSION_NUMBER", ORT_VERSION)
            _write(assets / "onnxruntime/test/testdata/model.onnx")
            env = os.environ.copy()
            env["ORT_TEST_ASSETS_ROOT"] = str(assets)
            command = [sys.executable, str(BUILD_AND_TEST), "--dry-run", "--ort-prebuilt", str(sdk), "test"]
            result = subprocess.run(command, check=False, capture_output=True, text=True, env=env, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("matching plugin model-test runner", result.stderr)
            _create_runner(sdk)
            result = subprocess.run(command, check=False, capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_build_identity_missing_version_defers_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sdk = root / "sdk"
            assets = root / "assets"
            sdk.mkdir()
            assets.mkdir()
            command = [sys.executable, str(IDENTITY_SCRIPT), f"--sdk={sdk}", f"--assets={assets}"]

            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            missing_identity = json.loads(result.stdout)
            self.assertEqual(missing_identity["sdk"], {"path": sdk.as_posix(), "version": None})
            self.assertEqual(missing_identity["assets"], {"path": assets.as_posix(), "version": None})

            _write(sdk / "VERSION_NUMBER", ORT_VERSION)
            _write(assets / "VERSION_NUMBER", ORT_VERSION)
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            valid_identity = json.loads(result.stdout)
            self.assertEqual(valid_identity["sdk"]["version"], ORT_VERSION)
            self.assertEqual(valid_identity["assets"]["version"], ORT_VERSION)
            self.assertNotEqual(missing_identity, valid_identity)

    def test_build_identity_tracks_assets_paths_presence_and_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sdk = root / "sdk"
            _create_prebuilt_sdk(sdk)
            identities = []
            for assets, version in (("", ORT_VERSION), ("A", ORT_VERSION), ("B", ORT_VERSION), ("B", "1.26.0")):
                path = root / assets if assets else None
                if path is not None:
                    _write(path / "VERSION_NUMBER", version)
                command = [sys.executable, str(IDENTITY_SCRIPT), f"--sdk={sdk}"]
                if path is not None:
                    command.append(f"--assets={path}")
                result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
                identities.append(result.stdout)
            self.assertEqual(len(set(identities)), 4)

            # Exercise the actual host update policy, including the unchanged-input case.
            build = root / "policy-build"
            _write(build / "Release/build.ninja")
            env = os.environ.copy()
            env["ORT_POLICY_BUILD"] = str(build)
            if sys.platform == "win32":
                pwsh = shutil.which("pwsh") or shutil.which("powershell")
                if pwsh is None:
                    self.skipTest("PowerShell is not available")
                env["ORT_IDENTITIES"] = json.dumps([item.strip() for item in identities])
                env["ORT_POLICY_UTILS"] = str(REPO_ROOT / "qcom/scripts/windows/utils.ps1")
                script = """. $env:ORT_POLICY_UTILS
function Test-TargetPyVersionDiffers { return $false }
function Test-QairtSdkDiffers { return $false }
$checks = @()
foreach ($identity in ($env:ORT_IDENTITIES | ConvertFrom-Json)) {
  $params = @{ BuildDir=$env:ORT_POLICY_BUILD; Config='Release'; QairtSdkRoot='sdk';
               CMakeGenerator='Ninja'; Update=$false; OrtBuildIdentity=$identity }
  $checks += Test-UpdateNeeded @params 6>$null
  Set-Content -LiteralPath "$env:ORT_POLICY_BUILD/ort-build-identity-Release.txt" -Value $identity
  $checks += Test-UpdateNeeded @params 6>$null
}
ConvertTo-Json -InputObject $checks -Compress
"""
                result = subprocess.run(
                    [pwsh, "-NoProfile", "-Command", script],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=120,
                )
                self.assertEqual(json.loads(result.stdout), [True, False] * 4)
            elif sys.platform.startswith("linux"):
                build_script = (REPO_ROOT / "qcom/scripts/linux/build.sh").read_text()
                policy = build_script.split("function update_needed() {", 1)[1].split(
                    "function save_qairt_sdk_path()", 1
                )[0]
                env["ORT_IDENTITIES"] = "".join(identities)
                script = (
                    "function log_debug() { :; }\nfunction update_needed() {"
                    + policy
                    + """
qairt_sdk_file_path="$ORT_POLICY_BUILD/sdk.txt"
qnn_build_mode_file_path="$ORT_POLICY_BUILD/mode.txt"
ort_build_identity_file_path="$ORT_POLICY_BUILD/ort-build-identity-Release.txt"
echo sdk > "$qairt_sdk_file_path"
echo mode > "$qnn_build_mode_file_path"
while IFS= read -r identity; do
  echo "changed=$(update_needed "$ORT_POLICY_BUILD/Release" sdk mode "$identity")"
  printf '%s\n' "$identity" > "$ort_build_identity_file_path"
  echo "same=$(update_needed "$ORT_POLICY_BUILD/Release" sdk mode "$identity")"
done <<< "$ORT_IDENTITIES"
"""
                )
                # Strip the final newline so the here-string does not add an empty input.
                env["ORT_IDENTITIES"] = env["ORT_IDENTITIES"].rstrip("\n")
                result = subprocess.run(
                    ["bash", "-c", script],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=120,
                )
                self.assertEqual(result.stdout.splitlines(), ["changed=1", "same="] * 4)

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
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertNotEqual(result.returncode, 0, task)
                self.assertIn("Prebuilt ORT test tasks require --ort-test-assets", result.stderr, task)


if __name__ == "__main__":
    unittest.main()
