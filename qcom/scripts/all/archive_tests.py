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

# Microsoft Visual C++ runtime DLLs that onnxruntime.dll (from upstream MS prebuilt) imports.
# Bundled into the Windows test archive so QA machines don't need vc_redist preinstalled.
# Verified against the v1.26.0 prebuilt PE import tables: x64 onnxruntime.dll imports all four;
# arm64 imports msvcp140/msvcp140_1/vcruntime140 (no vcruntime140_1). The regex bundles whatever
# the redist provides; _MSVC_REDIST_REQUIRED is the per-arch completeness floor the loader needs.
_MSVC_REDIST_FILE_RE = re.compile(r"^(msvcp140|vcruntime140)(_1)?\.dll$", re.IGNORECASE)
# Per-arch minimum the loader needs to bring up onnxruntime.dll on a clean machine. x64 additionally
# imports vcruntime140_1.dll; arm64 (and arm64ec/arm64x, which load the native ARM64 CRT) does not.
# Keys mirror _MSVC_REDIST_ARCH_SUBDIR so the guard's promise ("won't fail to load onnxruntime.dll")
# holds per arch rather than against a lowest-common-denominator floor that under-covers x64.
_MSVC_REDIST_REQUIRED = {
    "windows-x86_64": ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"),
    "windows-arm64": ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll"),
    "windows-arm64ec": ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll"),
    "windows-arm64x": ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll"),
}

# Map per-arch target_platform to the redist subdirectory under VCToolsRedistDir.
# arm64ec and arm64x both load the native ARM64 CRT (arm64ec/arm64 share the ARM64 runtime;
# x64 code in the process is emulated), and Microsoft ships no arm64ec redist folder.
# Keys mirror the argparse --target-platform choices for the windows-* archive targets. The
# transient "windows-arm64-x-slice" build dir (BuildAsX + Arch=arm64) never runs archive mode
# — only the arm64ec slice does, as windows-arm64x — so it is intentionally absent here.
_MSVC_REDIST_ARCH_SUBDIR = {
    "windows-x86_64": "x64",
    "windows-arm64": "arm64",
    "windows-arm64ec": "arm64",
    "windows-arm64x": "arm64",
}


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


def _iter_msvc_redist(vc_redist_dir: Path, target_platform: str):
    """Yield MSVC redistributable DLLs (msvcp140*.dll, vcruntime140*.dll) for the target arch.

    vc_redist_dir is VCToolsRedistDir, e.g.
    'C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Redist/MSVC/14.40.33807'.
    Under it, each arch has a 'Microsoft.VC<NNN>.CRT' folder holding the runtime DLLs.
    """
    arch_subdir = _MSVC_REDIST_ARCH_SUBDIR.get(target_platform)
    if arch_subdir is None:
        logging.warning("No MSVC redist arch mapping for %s; no redist DLLs found.", target_platform)
        return
    arch_root = vc_redist_dir / arch_subdir
    if not arch_root.is_dir():
        logging.warning("MSVC redist arch dir not found: %s; no redist DLLs found.", arch_root)
        return
    crt_dirs = list(arch_root.glob("Microsoft.VC*.CRT"))
    if not crt_dirs:
        logging.warning("No Microsoft.VC*.CRT folder under %s; no redist DLLs found.", arch_root)
        return

    # A side-by-side toolset install can expose multiple CRT version folders. Bundle only the
    # highest so the archive never carries duplicate-named DLL entries. Sort on the numeric VC
    # version (e.g. Microsoft.VC143.CRT -> 143), not lexicographically, so VC9 < VC143.
    def _crt_version(p: Path) -> int:
        m = re.search(r"Microsoft\.VC(\d+)\.CRT", p.name)
        return int(m.group(1)) if m else -1

    crt_dir = max(crt_dirs, key=_crt_version)
    for p in crt_dir.iterdir():
        if p.is_file() and _MSVC_REDIST_FILE_RE.match(p.name):
            yield p


def _add_msvc_redist(zf: zipfile.ZipFile, vc_redist_dir: Path | None, target_platform: str, arc_root: Path) -> None:
    """Write the target arch's MSVC redist DLLs into the zip under arc_root. No-op when unset.

    Fails loudly when vc_redist_dir is supplied but the expected runtime DLLs can't be bundled:
    a silently-incomplete archive fails to load onnxruntime.dll on a clean QA machine, which is
    exactly what this bundling exists to prevent. This mirrors Get-VcRedistDir, which throws.
    """
    if vc_redist_dir is None:
        return
    bundled = set()
    for p in _iter_msvc_redist(vc_redist_dir, target_platform):
        zf.write(p, str(arc_root / p.name))
        bundled.add(p.name.lower())
    required = _MSVC_REDIST_REQUIRED.get(target_platform, ())
    missing = {n.lower() for n in required} - bundled
    if missing:
        raise RuntimeError(
            f"MSVC redist bundling for {target_platform} is incomplete (missing {sorted(missing)}) "
            f"under {vc_redist_dir}. The test archive would fail to load onnxruntime.dll on a clean "
            f"machine. Check the VC redist install and the arch mapping."
        )


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
    if config != "Release":
        archive_path = build_root / f"onnxruntime-tests-{config}-{target_platform}.tar.bz2"
    release_dir = build_root / target_platform / config
    rules = PerArchAcceptRules()

    archive_path.unlink(missing_ok=True)
    logging.info(f"Creating per-arch archive at {archive_path}")
    with tarfile.open(archive_path, "w:bz2") as tf:
        for fs, arc in _select_release_entries(release_dir, rules, repo_root):
            tf.add(fs, str(arc))
        for p in _iter_qcom_scripts(repo_root):
            tf.add(p, str(p.relative_to(repo_root)))


def archive_windows(
    target_platform: str,
    config: str = "Release",
    repo_root: Path = REPO_ROOT,
    vc_redist_dir: Path | None = None,
) -> None:
    build_root = repo_root / "build"
    archive_path = build_root / f"onnxruntime-tests-{target_platform}.zip"
    if config != "Release":
        archive_path = build_root / f"onnxruntime-tests-{config}-{target_platform}.zip"
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
                # arc is repo-relative (build/<plat>/<config>/<config>/...); rebuild it from
                # parts[3:] so entries stay rooted at build/<plat>/<config>/, preserving the
                # doubled <config>/<config>/ dir the VS multi-config generator produces.
                outer_arc = Path("build") / target_platform / config / Path(*arc.parts[3:])
                zf.write(fs, str(outer_arc))
            for p in _iter_qcom_scripts(repo_root):
                zf.write(p, str(p.relative_to(repo_root)))
            # Binaries (incl. onnxruntime.dll) land under build/<plat>/<config>/<config>/ in this
            # layout; the redist must co-locate there or the loader won't find it on a clean machine.
            _add_msvc_redist(zf, vc_redist_dir, target_platform, Path("build") / target_platform / config / config)
        return

    archive_path.unlink(missing_ok=True)
    logging.info(f"Creating per-arch archive at {archive_path}")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for fs, arc in _select_release_entries(release_dir, rules, repo_root):
            zf.write(fs, str(arc))
        for p in _iter_qcom_scripts(repo_root):
            zf.write(p, str(p.relative_to(repo_root)))
        # Single-config (Ninja): binaries live directly under build/<plat>/<config>/.
        _add_msvc_redist(zf, vc_redist_dir, target_platform, Path("build") / target_platform / config)


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
    parser.add_argument(
        "--vc-redist-dir",
        type=Path,
        default=None,
        help=(
            "VCToolsRedistDir on Windows. When set, msvcp140*.dll and vcruntime140*.dll for the "
            "target arch are bundled into the Windows test archive alongside onnxruntime.dll."
        ),
    )
    args = parser.parse_args()

    if args.target_platform.startswith("android-"):
        archive_android(args.target_platform, args.config, args.qairt_sdk_root)
    elif args.target_platform.startswith("linux-"):
        archive_linux(args.target_platform, args.config)
    elif args.target_platform.startswith("windows-"):
        archive_windows(args.target_platform, args.config, vc_redist_dir=args.vc_redist_dir)
    else:
        raise ValueError(f"Unknown platform {args.target_platform}.")
