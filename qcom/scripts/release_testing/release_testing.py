# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""
Release testing orchestrator for onnxruntime-qnn artifacts.

Runs validation tests for wheel, zip, tgz, and nuget artifacts either from
a local directory or by downloading from Artifactory first. Designed to be
invocable both from CI (GitHub Actions) and from a local dev/host machine.

Usage:
    # Local mode — artifacts already downloaded
    python qcom/scripts/release_testing/release_testing.py --artifact-type wheel --artifact-version 2.6.0 \\
        --source-directory ./release_testing

    # Download mode — fetch from Artifactory first
    python qcom/scripts/release_testing/release_testing.py --artifact-type wheel --artifact-version 2.6.0 \\
        --source-directory ./release_testing --download-from-artifactory
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import re
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
RELEASE_SCRIPTS_DIR = SCRIPT_DIR

# Upleveling infrastructure paths (reused for certs and config)
UPLEVELING_DIR = SCRIPT_DIR.parent / "upleveling"
INI_FILE = UPLEVELING_DIR / "config.ini"
ARTIFACTORY_CERTS_FILE = UPLEVELING_DIR / "certs" / "artifactory-ca.pem"

# Hardcoded defaults
PRODUCT_NAME = "onnxruntime-qnn"
SAMPLE_PATH = REPO_ROOT / "qcom" / "samples" / "test_wheel.py"
MODEL_PATH = (
    REPO_ROOT
    / "cmake"
    / "external"
    / "onnx"
    / "onnx"
    / "backend"
    / "test"
    / "data"
    / "node"
    / "test_abs"
    / "model.onnx"
)
BACKWARD_COMPAT_ORT_VERSION = "1.24.2"

# Storage index servers per artifact type (config.ini sections with direct storage URLs)
ARTIFACT_STORAGE_INDEXES = {
    "wheel": "artifactory-pypi-public-storage",
    "nuget": "artifactory-nuget-public-storage",
    "zip": "artifactory-zip-public-storage",
    "tgz": "artifactory-zip-public-storage",
    "test_package": "artifactory-zip-testproj-storage",
}

ARTIFACT_SUFFIXES = {
    "wheel": ".whl",
    "nuget": ".nupkg",
    "zip": ".zip",
    "tgz": ".tgz",
}

log = logging.getLogger("release_test")


# =============================================================================
# Platform detection
# =============================================================================


def is_host_windows() -> bool:
    return platform.system() == "Windows"


def is_host_linux() -> bool:
    return platform.system() == "Linux"


def is_host_arm64() -> bool:
    # On Windows ARM64 with x64-emulated Python, platform.machine()
    # returns "AMD64"; check PROCESSOR_ARCHITEW6432 for the true arch.
    arch_w6432 = os.environ.get("PROCESSOR_ARCHITEW6432", "").upper()
    if arch_w6432 == "ARM64":
        return True
    return platform.machine().lower() in ("aarch64", "arm64")


def is_host_x86_64() -> bool:
    return platform.machine().lower() in ("x86_64", "amd64")


# =============================================================================
# Artifactory download (follows qnn_ep_uplevel.py pattern)
# =============================================================================


def _get_artifactory_credentials() -> tuple[str, str]:
    user = os.environ.get("ARTIFACTORY_USERNAME", "")
    password = os.environ.get("ARTIFACTORY_PASSWORD", "")
    missing = []
    if not user:
        missing.append("ARTIFACTORY_USERNAME")
    if not password:
        missing.append("ARTIFACTORY_PASSWORD")
    if missing:
        log.error(f"Missing env var(s) required for downloads: {', '.join(missing)}")
        sys.exit(1)
    return user, password


def _get_repository_url(artifact_type: str, version: str) -> str:
    config = ConfigParser()
    config.read([str(INI_FILE)])
    index = ARTIFACT_STORAGE_INDEXES[artifact_type]
    base_url = config.get(index, "repository")
    return f"{base_url}/{PRODUCT_NAME}/{version}"


def download_artifacts(
    artifact_type: str,
    version: str,
    suffix: str,
    dest_dir: Path,
    filename_filter: str | None = None,
) -> list[str]:
    """Download artifacts from Artifactory using HTTP (no JFrog CLI needed)."""
    url = _get_repository_url(artifact_type, version)
    user, password = _get_artifactory_credentials()
    auth = HTTPBasicAuth(user, password)
    verify = str(ARTIFACTORY_CERTS_FILE) if ARTIFACTORY_CERTS_FILE.exists() else True

    log.info(f"Fetching artifact list from: {url}")
    response = requests.get(url, auth=auth, verify=verify)
    if response.status_code != 200:
        log.error(f"Failed to fetch artifact list from {url} (HTTP {response.status_code})")
        sys.exit(1)

    # Parse Artifactory HTML directory listing (same regex as qnn_ep_uplevel.py)
    artifact_list = [
        m.group(1)
        for m in (
            re.search(r'href=["\']([^"\']*' + re.escape(suffix) + r')["\']', line)
            for line in response.text.splitlines()
            if suffix in line
        )
        if m
    ]

    if filename_filter:
        artifact_list = [f for f in artifact_list if re.search(filename_filter, f)]

    if not artifact_list:
        log.error(f"No artifacts with suffix '{suffix}' found at {url}")
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)

    for artifact_file in artifact_list:
        artifact_url = f"{url}/{artifact_file}"
        download_path = dest_dir / artifact_file
        log.info(f"Downloading {artifact_file}")
        r = requests.get(artifact_url, auth=auth, verify=verify)
        if r.status_code != 200:
            log.error(f"Failed to download {artifact_file} (HTTP {r.status_code})")
            sys.exit(1)
        download_path.write_bytes(r.content)
        log.info(f"Saved {download_path}")

    return artifact_list


def list_artifacts(
    artifact_type: str,
    version: str,
    suffix: str,
    filename_filter: str | None = None,
) -> list[str]:
    """List artifact filenames on Artifactory without downloading them."""
    url = _get_repository_url(artifact_type, version)
    user, password = _get_artifactory_credentials()
    auth = HTTPBasicAuth(user, password)
    verify = str(ARTIFACTORY_CERTS_FILE) if ARTIFACTORY_CERTS_FILE.exists() else True

    log.info(f"Listing artifacts at: {url}")
    response = requests.get(url, auth=auth, verify=verify)
    if response.status_code != 200:
        log.error(f"Failed to fetch artifact list from {url} (HTTP {response.status_code})")
        sys.exit(1)

    artifact_list = [
        m.group(1)
        for m in (
            re.search(r'href=["\']([^"\']*' + re.escape(suffix) + r')["\']', line)
            for line in response.text.splitlines()
            if suffix in line
        )
        if m
    ]

    if filename_filter:
        artifact_list = [f for f in artifact_list if re.search(filename_filter, f)]

    return artifact_list


# =============================================================================
# Artifact count verification
# =============================================================================

_WHEEL_ARCHES = ("win_arm64", "win_amd64", "manylinux_2_34_aarch64", "manylinux_2_35_x86_64")
_EXPECTED_ZIP_SUFFIXES = ("win-arm64.zip", "win-arm64x.zip", "win-x64.zip")
_EXPECTED_TGZ_SUFFIXES = ("linux-aarch64.tgz", "linux-x64.tgz")
_ZIP_SKIP_PATTERN = r"(-pdb\.zip|-win-arm64ec\.zip)$"


def verify_artifact_counts(args: argparse.Namespace) -> None:
    """Verify all expected artifacts are present on Artifactory before running tests."""
    python_versions = args.python_versions.split(",")
    failed = False

    # --- Wheels: 4 Python versions x 4 arches = 16 ---
    wheels = list_artifacts("wheel", args.artifact_version, ".whl")
    expected_wheels = [
        f"cp{v.replace('.', '')}-cp{v.replace('.', '')}-{arch}.whl"
        for v in python_versions
        for arch in _WHEEL_ARCHES
    ]
    missing_wheels = [e for e in expected_wheels if not any(e in f for f in wheels)]
    if missing_wheels:
        log.error("Wheels: FAIL — %d missing:\n%s", len(missing_wheels), "\n".join(f"  {m}" for m in missing_wheels))
        failed = True
    else:
        log.info(f"Wheels: PASS — {len(wheels)}/{len(expected_wheels)}")

    # --- NuGet: 1 expected ---
    nupkgs = list_artifacts("nuget", args.artifact_version, ".nupkg")
    if len(nupkgs) != 1:
        log.error(f"NuGet: FAIL — expected 1, found {len(nupkgs)}: {nupkgs}")
        failed = True
    else:
        log.info("NuGet: PASS — 1/1")

    # --- Zip: 3 expected, excluding -pdb.zip and -win-arm64ec.zip ---
    all_zips = list_artifacts("zip", args.artifact_version, ".zip")
    zips = [f for f in all_zips if not re.search(_ZIP_SKIP_PATTERN, f)]
    missing_zips = [s for s in _EXPECTED_ZIP_SUFFIXES if not any(f.endswith(s) for f in zips)]
    if missing_zips:
        log.error(f"Zip: FAIL — missing: {missing_zips}")
        failed = True
    else:
        log.info(f"Zip: PASS — {len(zips)}/3")

    # --- TGZ: 2 expected ---
    tgzs = list_artifacts("tgz", args.artifact_version, ".tgz")
    missing_tgzs = [s for s in _EXPECTED_TGZ_SUFFIXES if not any(f.endswith(s) for f in tgzs)]
    if missing_tgzs:
        log.error(f"TGZ: FAIL — missing: {missing_tgzs}")
        failed = True
    else:
        log.info(f"TGZ: PASS — {len(tgzs)}/2")

    if failed:
        sys.exit(1)
    log.info("Artifact count check PASS: all expected artifacts are present")


def run_script(script_name: str, args: list[str]) -> None:
    """Run a release testing script (.ps1 or .sh)."""
    script = RELEASE_SCRIPTS_DIR / script_name
    if not script.exists():
        log.error(f"Script not found: {script}")
        sys.exit(1)

    if script.suffix == ".ps1":
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script), *args]
    elif script.suffix == ".sh":
        cmd = ["bash", str(script), *args]
    else:
        log.error(f"Unknown script type: {script.suffix}")
        sys.exit(1)

    log.info(f"Running: {script.name} {' '.join(args)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error(f"{script.name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


# =============================================================================
# Per-artifact-type test runners
# =============================================================================


def _wheel_download_filter(args: argparse.Namespace) -> str | None:
    """Return a filename regex that limits wheel downloads to what this job actually needs."""
    if args.only_metadata:
        # Metadata job verifies Authenticode on all Windows wheels (arm64 + amd64).
        return r"(win_arm64|win_amd64)\.whl$"

    if args.only_backward_compat:
        # Backward compat always uses the win_arm64 cp312 wheel.
        return r"cp312-cp312-win_arm64\.whl$"

    # Install+test: filter by host platform and the requested Python version(s).
    if is_host_windows():
        arch_tag = "win_arm64" if is_host_arm64() else "win_amd64"
    elif is_host_linux():
        arch_tag = "manylinux_2_34_aarch64" if is_host_arm64() else "manylinux_2_35_x86_64"
    else:
        return r"\.whl$"  # unknown platform — download everything

    py_alts = "|".join(f"cp{v.replace('.', '')}" for v in args.python_versions.split(","))
    return rf"(?:{py_alts})-{arch_tag}\.whl$"


def test_wheel(args: argparse.Namespace) -> None:
    wheel_dir = Path(args.source_directory) / "wheel"

    if args.download_from_artifactory:
        downloaded = download_artifacts(
            "wheel",
            args.artifact_version,
            ARTIFACT_SUFFIXES["wheel"],
            wheel_dir,
            filename_filter=_wheel_download_filter(args),
        )

        if args.only_metadata:
            log.info(f"Downloaded {len(downloaded)} Windows wheel(s) for metadata verification")

    if args.only_metadata:
        if is_host_windows():
            run_script(
                "verify_windows_metadata.ps1",
                [
                    "-ArtifactType",
                    "wheel",
                    "-SourceDirectory",
                    str(wheel_dir),
                    "-ExpectedVersion",
                    args.artifact_version,
                ],
            )
        else:
            log.info("Metadata verification is Windows-only, skipping")
        return

    if args.only_backward_compat:
        if is_host_windows() and is_host_arm64():
            run_script(
                "verify_backward_compatibility.ps1",
                [
                    "-PythonVersion",
                    "3.12",
                    "-WheelArch",
                    "win_arm64",
                    "-WheelDirectory",
                    str(wheel_dir),
                    "-OnnxruntimeVersion",
                    BACKWARD_COMPAT_ORT_VERSION,
                    "-SamplePath",
                    str(SAMPLE_PATH),
                ],
            )
        else:
            log.info("Backward compatibility test only runs on Windows ARM64, skipping")
        return

    # Default: install + test only
    python_versions = args.python_versions.split(",")

    if is_host_windows():
        wheel_arch = "win_arm64" if is_host_arm64() else "win_amd64"
        for pyver in python_versions:
            run_script(
                "install_and_test_wheel.ps1",
                [
                    "-PythonVersion",
                    pyver,
                    "-WheelArch",
                    wheel_arch,
                    "-WheelDirectory",
                    str(wheel_dir),
                    "-ExpectedVersion",
                    args.artifact_version,
                    "-SamplePath",
                    str(SAMPLE_PATH),
                ],
            )

    elif is_host_linux():
        wheel_arch = "manylinux_2_34_aarch64" if is_host_arm64() else "manylinux_2_35_x86_64"
        for pyver in python_versions:
            run_script(
                "install_and_test_wheel.sh",
                [
                    "--python-version",
                    pyver,
                    "--wheel-arch",
                    wheel_arch,
                    "--wheel-directory",
                    str(wheel_dir),
                    "--expected-version",
                    args.artifact_version,
                    "--sample-path",
                    str(SAMPLE_PATH),
                ],
            )


def test_zip(args: argparse.Namespace) -> None:
    if not is_host_windows():
        log.info("Zip testing is Windows-only, skipping on this platform")
        return

    archive_dir = Path(args.source_directory) / "archive"

    if args.download_from_artifactory:
        if not args.test_package_version:
            log.error("--test-package-version is required for zip testing")
            sys.exit(1)
        download_artifacts(
            "zip",
            args.artifact_version,
            ARTIFACT_SUFFIXES["zip"],
            archive_dir,
        )
        if not args.only_metadata:
            # Download test package (only needed for smoke tests)
            test_pkg_files = download_artifacts(
                "test_package",
                args.test_package_version,
                ".zip",
                archive_dir,
                filename_filter=r"test_package\.zip$",
            )
            for f in test_pkg_files:
                src = archive_dir / f
                dst = archive_dir / "test_package.zip"
                if src != dst:
                    src.rename(dst)

    if args.only_metadata:
        run_script(
            "verify_windows_metadata.ps1",
            [
                "-ArtifactType",
                "zip",
                "-SourceDirectory",
                str(archive_dir),
                "-ExpectedVersion",
                args.artifact_version,
            ],
        )
        return

    # Smoke test — determine platform-specific args
    if is_host_arm64():
        configs = [
            ("win-arm64", "windows-arm64", "QnnHtp.dll"),
            ("win-arm64x", "windows-arm64", "QnnHtp.dll"),
            ("win-arm64x", "windows-x86_64", "QnnHtp.dll"),  # ARM64EC path
        ]
    elif is_host_x86_64():
        configs = [
            ("win-x64", "windows-x86_64", "QnnCpu.dll"),
        ]
    else:
        log.warning(f"Unknown Windows architecture: {platform.machine()}")
        return

    test_package_zip = archive_dir / "test_package.zip"
    for zip_arch, test_bin_arch, backend_dll in configs:
        run_script(
            "smoke_test_zip.ps1",
            [
                "-ZipDirectory",
                str(archive_dir),
                "-TestPackageZip",
                str(test_package_zip),
                "-ModelPath",
                str(MODEL_PATH),
                "-ZipArch",
                zip_arch,
                "-TestBinArch",
                test_bin_arch,
                "-BackendDll",
                backend_dll,
            ],
        )


def test_tgz(args: argparse.Namespace) -> None:
    if not is_host_linux():
        log.info("TGZ testing is Linux-only, skipping on this platform")
        return

    archive_dir = Path(args.source_directory) / "archive"

    if args.download_from_artifactory:
        if not args.test_package_version:
            log.error("--test-package-version is required for tgz testing")
            sys.exit(1)
        download_artifacts(
            "tgz",
            args.artifact_version,
            ARTIFACT_SUFFIXES["tgz"],
            archive_dir,
        )
        # Download test package
        test_pkg_files = download_artifacts(
            "test_package",
            args.test_package_version,
            ".zip",
            archive_dir,
            filename_filter=r"test_package\.zip$",
        )
        for f in test_pkg_files:
            src = archive_dir / f
            dst = archive_dir / "test_package.zip"
            if src != dst:
                src.rename(dst)

    # Smoke test — determine platform-specific args
    if is_host_arm64():
        tgz_arch, test_bin_arch, backend_lib = "linux-aarch64", "linux-arm64", "libQnnHtp.so"
    elif is_host_x86_64():
        tgz_arch, test_bin_arch, backend_lib = "linux-x64", "linux-x86_64", "libQnnCpu.so"
    else:
        log.warning(f"Unknown Linux architecture: {platform.machine()}")
        return

    test_package_zip = archive_dir / "test_package.zip"
    run_script(
        "smoke_test_tgz.sh",
        [
            "--tgz-directory",
            str(archive_dir),
            "--test-package-zip",
            str(test_package_zip),
            "--model-path",
            str(MODEL_PATH),
            "--tgz-arch",
            tgz_arch,
            "--test-bin-arch",
            test_bin_arch,
            "--backend-lib",
            backend_lib,
        ],
    )


def test_nuget(args: argparse.Namespace) -> None:
    if not is_host_windows():
        log.info("NuGet testing is Windows-only, skipping on this platform")
        return

    nuget_dir = Path(args.source_directory) / "nuget"

    if args.download_from_artifactory:
        download_artifacts(
            "nuget",
            args.artifact_version,
            ARTIFACT_SUFFIXES["nuget"],
            nuget_dir,
        )

    if args.only_metadata:
        run_script(
            "verify_windows_metadata.ps1",
            [
                "-ArtifactType",
                "nuget",
                "-SourceDirectory",
                str(nuget_dir),
                "-ExpectedVersion",
                args.artifact_version,
            ],
        )
        return

    # Determine host-appropriate RID and backend
    if is_host_arm64():
        runtime_id, backend_dll = "win-arm64", "QnnHtp.dll"
    elif is_host_x86_64():
        runtime_id, backend_dll = "win-x64", "QnnCpu.dll"
    else:
        log.warning(f"Unknown Windows architecture: {platform.machine()}")
        return

    run_script(
        "test_nuget_package.ps1",
        [
            "-NuGetDirectory",
            str(nuget_dir),
            "-ExpectedVersion",
            args.artifact_version,
            "-RuntimeIdentifiers",
            runtime_id,
            "-ModelPath",
            str(MODEL_PATH),
            "-BackendDll",
            backend_dll,
        ],
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Release testing orchestrator for onnxruntime-qnn artifacts",
    )
    parser.add_argument(
        "--artifact-type",
        required=True,
        choices=["wheel", "zip", "tgz", "nuget", "all"],
        help="Type of artifact to test",
    )
    parser.add_argument(
        "--artifact-version",
        required=True,
        help="Version of the artifact to test (e.g. 2.6.0, 2.5.0rc3)",
    )
    parser.add_argument(
        "--source-directory",
        required=True,
        help="Base directory for artifacts. Sub-directories (wheel/, archive/, nuget/) are created per type.",
    )
    parser.add_argument(
        "--test-package-version",
        default=None,
        help="Version of the test package archive. Required for zip/tgz testing.",
    )
    parser.add_argument(
        "--download-from-artifactory",
        action="store_true",
        help=(
            "Download artifacts from Artifactory before testing. "
            "Requires ARTIFACTORY_USERNAME and ARTIFACTORY_PASSWORD env vars."
        ),
    )
    parser.add_argument(
        "--python-versions",
        default="3.11,3.12,3.13,3.14",
        help="Comma-separated Python versions for wheel testing (default: 3.11,3.12,3.13,3.14)",
    )
    parser.add_argument(
        "--only-count-check",
        action="store_true",
        help="Only verify that all expected artifacts are present on Artifactory.",
    )
    parser.add_argument(
        "--only-backward-compat",
        action="store_true",
        help="Only run the backward compatibility test (wheels only).",
    )
    parser.add_argument(
        "--only-metadata",
        action="store_true",
        help="Only run metadata verification (Authenticode + version checks).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [release_test] [%(levelname)s] %(message)s",
    )

    args = parse_args()

    log.info(f"Platform: {platform.system()} {platform.machine()}")
    log.info(f"Artifact type: {args.artifact_type}")
    log.info(f"Artifact version: {args.artifact_version}")

    if args.only_count_check:
        verify_artifact_counts(args)
        return

    runners = {
        "wheel": test_wheel,
        "zip": test_zip,
        "tgz": test_tgz,
        "nuget": test_nuget,
    }

    if args.artifact_type == "all":
        for name, runner in runners.items():
            log.info(f"=== Testing {name} artifacts ===")
            runner(args)
    else:
        runners[args.artifact_type](args)

    log.info("All tests passed")


if __name__ == "__main__":
    main()
