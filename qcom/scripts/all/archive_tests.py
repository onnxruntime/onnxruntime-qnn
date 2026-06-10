#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

"""Produce the per-arch test archive: binaries + scripts only (testdata moved to a separate archive)."""

import argparse
import logging
import platform
import re
import tarfile
import zipfile
from pathlib import Path

QCOM_ROOT = Path(__file__).parent.parent.parent
REPO_ROOT = QCOM_ROOT.parent

# Top-level files inside Release/ accepted purely by extension or exact name.
_TOP_LEVEL_EXT_RE = re.compile(r"\.(dll|exe|cat|so|so\.[0-9.]+)$")
_TOP_LEVEL_EXACT_NAMES = {
    "CTestTestfile.cmake",
    "run_tests.ps1",
    "run_tests.sh",
    "ctest",
    "ctest.exe",
    "python_test_files.txt",
}
_TOP_LEVEL_PYTHON_TEST_RE = re.compile(r"^onnxruntime_test_python.*\.py$")
_QCOM_SCRIPTS_RE = re.compile(r"^qcom/scripts/")
_QUANTIZATION_RE = re.compile(r"^quantization/")
_PROVIDERS_QNN_RE = re.compile(r".*onnxruntime_providers_qnn[^/]*\.(so|so\.[0-9.]+|dll)$")
# AAR-build APKs consumed by qcom/scripts/linux/appium/tests/test_aar.py. These live under
# java/androidtest/android/app/build/outputs/apk/{debug,androidTest/debug}/ and are only
# produced when the Android build runs with -BuildAar/build_aar=true.
_AAR_APK_RE = re.compile(r"^java/androidtest/.*\.apk$")

# Drop __pycache__ and .pytest_cache unconditionally.
_ALWAYS_REJECT_RE = re.compile(r".*/(__pycache__|\.pytest_cache)/")


class PerArchAcceptRules:
    """Accept-list filter for per-arch test archive contents.

    Two surfaces:
      - top-level (Release/<file>): extension / exact name / exec-bit
      - subpath (under Release/<subdir>/...): full pattern matching
    """

    def accept_top_level(self, name: str) -> bool:
        if _TOP_LEVEL_EXT_RE.search(name):
            return True
        if name in _TOP_LEVEL_EXACT_NAMES:
            return True
        if _TOP_LEVEL_PYTHON_TEST_RE.match(name):
            return True
        return False

    def accept_top_level_with_path(self, path: Path) -> bool:
        """For Linux/Android: extension-less executables (+x) are test runners."""
        if self.accept_top_level(path.name):
            return True
        if not path.is_file():
            return False
        if "." in path.name:
            return False
        try:
            return bool(path.stat().st_mode & 0o111)
        except OSError:
            return False

    def accept_repo_relative(self, rel: Path) -> bool:
        """For files under build/<plat>/Release/<subdir>/... or qcom/scripts/...
        The caller strips build/<plat>/Release/ prefix for subdir paths."""
        s = rel.as_posix()
        if _QCOM_SCRIPTS_RE.match(s):
            return True
        if _QUANTIZATION_RE.match(s):
            return True
        if _PROVIDERS_QNN_RE.match(s):
            return True
        if _AAR_APK_RE.match(s):
            return True
        return False


def _iter_release_files(release_dir: Path):
    for p in release_dir.glob("**/*"):
        if not p.is_file() or (platform.system() == "Windows" and p.is_symlink()):
            continue
        if _ALWAYS_REJECT_RE.search(p.as_posix()):
            continue
        yield p


def _iter_qcom_scripts(repo_root: Path):
    for p in (repo_root / "qcom" / "scripts").glob("**/*"):
        if not p.is_file():
            continue
        if _ALWAYS_REJECT_RE.search(p.as_posix()):
            continue
        yield p


def _select_release_entries(release_dir: Path, rules: PerArchAcceptRules, repo_root: Path = REPO_ROOT):
    """Yield (filesystem_path, arcname_relative_to_REPO_ROOT) for accepted Release/ files."""
    for p in _iter_release_files(release_dir):
        rel_to_release = p.relative_to(release_dir)
        if rel_to_release.parent == Path("."):
            # Top-level under Release/
            if rules.accept_top_level_with_path(p):
                yield p, p.relative_to(repo_root)
        else:
            if rules.accept_repo_relative(rel_to_release):
                yield p, p.relative_to(repo_root)


def archive_linux(target_platform: str, config: str = "Release", repo_root: Path = REPO_ROOT) -> None:
    build_root = repo_root / "build"
    archive_path = build_root / f"onnxruntime-tests-{target_platform}.tar.bz2"
    release_dir = build_root / target_platform / config
    rules = PerArchAcceptRules()

    archive_path.unlink(missing_ok=True)
    logging.info(f"Creating per-arch archive at {archive_path}")
    with tarfile.open(archive_path, "w:bz2") as tf:
        for fs, arc in _select_release_entries(release_dir, rules, repo_root):
            tf.add(fs, str(arc))
        for p in _iter_qcom_scripts(repo_root):
            tf.add(p, str(p.relative_to(repo_root)))


def archive_windows(target_platform: str, config: str = "Release", repo_root: Path = REPO_ROOT) -> None:
    build_root = repo_root / "build"
    archive_path = build_root / f"onnxruntime-tests-{target_platform}.zip"
    release_dir = build_root / target_platform / config
    rules = PerArchAcceptRules()

    # Multi-config generators (VS) nest config again: build/<plat>/Release/Release/
    if (release_dir / config).is_dir():
        nested = release_dir / config
        outer_keep = ["CTestTestfile.cmake", "ctest.exe", "run_tests.ps1"]
        archive_path.unlink(missing_ok=True)
        logging.info(f"Creating per-arch archive at {archive_path}")
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in outer_keep:
                src = release_dir / name
                if src.exists():
                    zf.write(src, str(src.relative_to(repo_root)))
            for fs, arc in _select_release_entries(nested, rules, repo_root):
                # Drop the inner Release/ doubling so layout matches single-config.
                outer_arc = Path("build") / target_platform / config / Path(*arc.parts[3:])
                zf.write(fs, str(outer_arc))
            for p in _iter_qcom_scripts(repo_root):
                zf.write(p, str(p.relative_to(repo_root)))
        return

    archive_path.unlink(missing_ok=True)
    logging.info(f"Creating per-arch archive at {archive_path}")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for fs, arc in _select_release_entries(release_dir, rules, repo_root):
            zf.write(fs, str(arc))
        for p in _iter_qcom_scripts(repo_root):
            zf.write(p, str(p.relative_to(repo_root)))


def archive_android(target_platform: str, config: str, qairt_sdk_root: Path, repo_root: Path = REPO_ROOT) -> None:
    """Like archive_linux but also includes QNN aarch64-android + hexagon libs from QAIRT SDK."""
    build_root = repo_root / "build"
    archive_path = build_root / f"onnxruntime-tests-{target_platform}.zip"
    release_dir = build_root / target_platform / config
    rules = PerArchAcceptRules()

    archive_path.unlink(missing_ok=True)
    logging.info(f"Creating per-arch archive at {archive_path}")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for fs, arc in _select_release_entries(release_dir, rules, repo_root):
            zf.write(fs, str(arc))
        for p in _iter_qcom_scripts(repo_root):
            zf.write(p, str(p.relative_to(repo_root)))
        qairt_accept_re = re.compile(r".*/(aarch64-android|hexagon-v.+)/.*")
        for p in (qairt_sdk_root / "lib").glob("**/*"):
            if p.is_file() and qairt_accept_re.match(p.as_posix()):
                zf.write(p, str(p.relative_to(qairt_sdk_root)))


if __name__ == "__main__":
    log_format = "[%(asctime)s] [archive_tests.py] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Release", choices=["Debug", "Release", "RelWithDebInfo"])
    parser.add_argument(
        "--target-platform",
        required=True,
        choices=[
            "android-aarch64",
            "linux-x86_64",
            "linux-x86_64_ubuntu_22_04",
            "linux-aarch64_manylinux_2_34",
            "linux-aarch64_oe_gcc11_2",
            "windows-arm64",
            "windows-arm64ec",
            "windows-arm64x",
            "windows-x86_64",
        ],
    )
    parser.add_argument("--qairt-sdk-root", type=Path, required=True)
    args = parser.parse_args()

    if args.target_platform.startswith("android-"):
        archive_android(args.target_platform, args.config, args.qairt_sdk_root)
    elif args.target_platform.startswith("linux-"):
        archive_linux(args.target_platform, args.config)
    elif args.target_platform.startswith("windows-"):
        archive_windows(args.target_platform, args.config)
    else:
        raise ValueError(f"Unknown platform {args.target_platform}.")
