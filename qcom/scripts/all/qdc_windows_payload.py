# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Prepare Windows QDC tests without changing the shared test archive's layout."""

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import yaml
from extract_testdata import MAPPING, extract_to_staging
from package_manager import PACKAGE_CONFIG, FileCache, PackageManager

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_PACKAGE = "python_312_windows_arm64"
DEFAULT_WINDOWS_HTP_ARCH = 81


def validate_windows_results(log_dir: Path) -> None:
    """Require device outcome evidence even when QDC reports transport success."""
    manifests = list(log_dir.rglob("required-reports.results.txt"))
    if len(manifests) != 1:
        raise ValueError("Windows QDC requires exactly one required-reports.results.txt manifest")
    try:
        manifest = json.loads(manifests[0].read_text(encoding="utf-8-sig"))
        required = {"bootstrap.results.xml", "htp-smoke.results.xml"}
        if manifest["test_mode"] == "full":
            config = manifest["build_config"]
            if config not in ("Release", "Debug", "RelWithDebInfo"):
                raise ValueError("Unsupported build configuration")
            ctest_reports = manifest["ctest_reports"]
            if not isinstance(ctest_reports, list) or not ctest_reports:
                raise ValueError("CTest report list is missing or empty")
            required.update(ctest_reports)
            required.add(f"onnxruntime_provider_test.exe.{config}.results.xml")
            required.update(
                f"{suite}_model_tests.results.xml" for suite in ("node", "float32", "qdq", "qdq-with-context-cache")
            )
        elif manifest["test_mode"] != "smoke":
            raise ValueError("Unsupported Windows test mode")
        if any(not isinstance(name, str) or Path(name).name != name for name in required):
            raise ValueError("Expected report names must be filenames")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Windows QDC report manifest: {error}") from error
    reports = list(log_dir.rglob("*.results.xml"))
    names = [report.name for report in reports]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate Windows QDC report names")
    missing = required - set(names)
    if missing:
        raise ValueError(f"Windows QDC reports are missing: {sorted(missing)}")
    bootstrap_passed = False
    for report in reports:
        try:
            root = ElementTree.parse(report).getroot()
            if root.tag not in ("testsuites", "testsuite"):
                raise ValueError("Expected a JUnit testsuite or testsuites root")
            for node in root.iter():
                if node.tag in ("failure", "error") or any(
                    int(node.get(attribute, "0")) != 0 for attribute in ("failures", "errors")
                ):
                    raise ValueError("Reported test failure or error")
            if report.name == "bootstrap.results.xml":
                cases = list(root.iter("testcase"))
                bootstrap_passed = (
                    len(cases) == 1
                    and cases[0].get("name") == "BootstrapAndTests"
                    and (cases[0].get("classname") == "QDCWindows" and cases[0].find("skipped") is None)
                )
                if not bootstrap_passed:
                    raise ValueError("Bootstrap did not report a completed test")
            elif report.name in required:
                cases = list(root.iter("testcase"))
                if report.name == "htp-smoke.results.xml" or report.name.startswith("onnxruntime_provider_test.exe."):
                    cases = [case for case in cases if case.get("classname", "").startswith("QnnHTP")]
                if not any(
                    case.find("skipped") is None
                    and case.get("status", "run") == "run"
                    and case.get("result", "completed") in ("complete", "completed")
                    for case in cases
                ):
                    raise ValueError("No completed, non-skipped test cases")
        except (ElementTree.ParseError, ValueError) as error:
            raise ValueError(f"Invalid or failed Windows QDC report {report}: {error}") from error
    if not bootstrap_passed:
        raise ValueError("Windows QDC bootstrap.results.xml is missing")


def fetch_python_installer() -> tuple[Path, dict[str, str]]:
    """Use the existing persistent cache, verifying SHA-256 on cache hits too."""
    package = yaml.safe_load(PACKAGE_CONFIG.read_text())[PYTHON_PACKAGE]
    installer = FileCache().fetch(
        f"{PYTHON_PACKAGE}-{package['version']}",
        package["url"].format(version=package["version"]),
        expected_sha256=package["sha256"],
    )
    return installer, {"python_version": package["version"], "python_sha256": package["sha256"]}


def prepare_windows_payload(
    archive_path: Path,
    output_dir: Path,
    testdata_archive: Path,
    config: str = "Release",
    test_mode: str = "full",
    models_dir: Path | None = None,
    installer: Path | None = None,
    python_info: dict[str, str] | None = None,
    htp_arch: int = DEFAULT_WINDOWS_HTP_ARCH,
) -> Path:
    """Add bootstrap, installer, models and testdata to a copy of the build artifact.

    Keep every original entry at its original path. Multi-config binaries and their
    testdata remain under <config>/<config>; ctest and run_tests.ps1 stay one level up.
    Optional local inputs allow payload verification without a device or downloads.
    htp_arch selects runtime dependencies and smoke-test expectations, not the QDC device.
    """
    if config not in ("Release", "Debug", "RelWithDebInfo"):
        raise ValueError(f"Unsupported build config: {config}")
    if test_mode not in ("smoke", "full"):
        raise ValueError(f"Unsupported Windows test mode: {test_mode}")
    if type(htp_arch) is not int or htp_arch <= 0:
        raise ValueError(f"Invalid HTP architecture: {htp_arch}")
    if not testdata_archive.is_file():
        raise FileNotFoundError(f"Windows QDC requires testdata: {testdata_archive}")

    test_root = f"build/windows-arm64/{config}"
    with zipfile.ZipFile(archive_path) as original:
        entries = set(original.namelist())
    required = (f"{test_root}/run_tests.ps1", f"{test_root}/ctest.exe", f"{test_root}/CTestTestfile.cmake")
    binary_root = test_root
    if f"{test_root}/{config}/onnxruntime_provider_test.exe" in entries:
        binary_root = f"{test_root}/{config}"
    required += tuple(
        f"{binary_root}/{name}"
        for name in (
            "onnxruntime_provider_test.exe",
            "onnxruntime_plugin_ep_onnx_test.exe",
            "onnxruntime.dll",
            "onnxruntime_providers_qnn.dll",
            "QnnHtp.dll",
            "QnnSystem.dll",
            f"QnnHtpV{htp_arch}Stub.dll",
            f"libQnnHtpV{htp_arch}Skel.so",
            f"libqnnhtpv{htp_arch}.cat",
            "msvcp140.dll",
            "msvcp140_1.dll",
            "vcruntime140.dll",
        )
    )
    missing = set(required) - entries
    if missing:
        raise ValueError(f"Incomplete Windows QDC test archive: {sorted(missing)}")

    if installer is None or python_info is None:
        installer, python_info = fetch_python_installer()
    if hashlib.sha256(installer.read_bytes()).hexdigest() != python_info["python_sha256"]:
        raise ValueError("Python installer SHA-256 mismatch")
    if models_dir is None and test_mode == "full":
        models = PackageManager("onnx_models")
        models.install()
        models_dir = models.get_content_dir()
    if test_mode == "full":
        for suite in ("float32", "qdq", "qdq-with-context-cache"):
            if not (models_dir / "testdata" / suite).is_dir():
                raise ValueError(f"Missing {suite} model suite in {models_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / archive_path.name
    shutil.copyfile(archive_path, output_path)
    with zipfile.ZipFile(output_path, "a", compression=zipfile.ZIP_DEFLATED) as payload:

        def add_tree(source: Path, prefix: str) -> None:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    name = f"{prefix}/{path.relative_to(source).as_posix()}"
                    if name not in entries:
                        payload.write(path, name)
                        entries.add(name)

        payload.write(REPO_ROOT / "qcom/scripts/windows/qdc_bootstrap.ps1", "bootstrap.ps1")
        payload.write(installer, "installers/python-arm64.exe")
        payload.writestr(
            "qdc_windows_config.json",
            json.dumps(
                {
                    **python_info,
                    "build_config": config,
                    "test_mode": test_mode,
                    "expected_htp_arch": htp_arch,
                },
                indent=2,
            ),
        )
        if f"{test_root}/utils.ps1" not in entries:
            payload.write(REPO_ROOT / "qcom/scripts/windows/utils.ps1", f"{test_root}/utils.ps1")

        with tempfile.TemporaryDirectory(prefix="QdcMergeTestdata-", dir=output_dir) as tmp:
            staging = Path(tmp)
            extract_to_staging(testdata_archive, staging)
            for handle, mapping in MAPPING.items():
                source_dir = staging / handle
                if not source_dir.is_dir():
                    raise ValueError(f"Testdata archive is missing {handle}")
                config_dir = binary_root.removeprefix("build/windows-arm64/")
                add_tree(source_dir, mapping.format(plat="windows-arm64", config=config_dir))
        # The model zoo differs from the four arch-neutral testdata handles above.
        if test_mode == "full":
            add_tree(models_dir, f"{test_root}/model_tests/onnx_models")
    return output_path
