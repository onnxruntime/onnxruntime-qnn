#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import platform
import re
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from util import get_logger, is_windows

log = get_logger("pkg_assets")


def parse_version_number(source_dir):
    """
    Parse the VERSION_NUMBER file to get the ORT version.

    Args:
        source_dir: Path to source directory containing VERSION_NUMBER file

    Returns:
        str: Version string (e.g., "1.23.0") or None if not found
    """
    version_file = os.path.join(source_dir, "VERSION_NUMBER")
    try:
        with open(version_file) as f:
            version_line = f.readline().strip()
            # Extract version using regex to handle any extra whitespace/content
            version_match = re.match(r"(\d+\.\d+\.\d+)", version_line)
            if version_match:
                return version_match.group(1)
            else:
                log.warning(f"Could not parse version from VERSION_NUMBER: {version_line}")
                return None
    except (OSError, FileNotFoundError) as e:
        log.warning(f"Could not read VERSION_NUMBER file: {e}")
        return None


def get_qnn_asset_file_list():
    """
    Returns the list of QNN asset files to include in the archive package.

    Args:
        is_windows_platform: If None, auto-detect from platform.system()

    Returns:
        list[str]: List of filenames to include
    """
    qnn_assets = {
        "windows": [
            "Genie.dll",
            "HtpPrepare.dll",
            "libQnnHtpV68Skel.so",
            "libqnnhtpv73.cat",
            "libQnnHtpV73Skel.so",
            "libqnnhtpv81.cat",
            "libQnnHtpV81Skel.so",
            "onnxruntime_providers_qnn.dll",
            "QnnCpu.dll",
            "QnnGpu.dll",
            "QnnHtp.dll",
            "QnnHtpNetRunExtensions.dll",
            "QnnHtpPrepare.dll",
            "QnnHtpV68Stub.dll",
            "QnnHtpV73Stub.dll",
            "QnnHtpV81Stub.dll",
            "QnnIr.dll",
            "QnnSaver.dll",
            "QnnSystem.dll",
        ],
        "others": [
            "libGenie.so",
            "libHtpPrepare.so",
            "libonnxruntime_providers_qnn.so",
            "libQnnCpu.so",
            "libQnnGpu.so",
            "libQnnHtp.so",
            "libQnnHtpNetRunExtensions.so",
            "libQnnHtpPrepare.so",
            "libQnnHtpV68Skel.so",
            "libQnnHtpV68Stub.so",
            "libQnnHtpV69Skel.so",
            "libQnnHtpV69Stub.so",
            "libQnnHtpV73Skel.so",
            "libQnnHtpV73Stub.so",
            "libQnnHtpV75Skel.so",
            "libQnnHtpV75Stub.so",
            "libQnnHtpV79Skel.so",
            "libQnnHtpV79Stub.so",
            "libQnnHtpV81Skel.so",
            "libQnnHtpV81Stub.so",
            "libQnnIr.so",
            "libQnnSaver.so",
            "libQnnSystem.so",
        ],
    }
    return qnn_assets["windows"] if is_windows() else qnn_assets["others"]


def _compute_archive_name(source_dir, version_suffix, archive_name_suffix, archive_ext, target_arch=None):
    """
    Build the archive filename using the shared naming rules.

    Format: onnxruntime-qnn[-<version>][<version_suffix>][-<archive_name_suffix>]-<platform>-<arch><ext>

    Args:
        source_dir: Path to source directory
        version_suffix: Optional version suffix for archive filename
        archive_name_suffix: Optional suffix for archive filename
        archive_ext: String for archive extension
        target_arch: Optional explicit target architecture. When set, overrides
            platform.machine() — needed for cross-compile builds where the build
            host arch differs from the target arch (e.g. arm64ec on an x64 host).
    """
    sys_name = platform.system().lower()
    platform_name = "win" if sys_name == "windows" else sys_name
    arch = (target_arch or platform.machine()).lower()
    arch = {"amd64": "x64", "x86_64": "x64"}.get(arch, arch)

    version = parse_version_number(source_dir)

    name = "onnxruntime-qnn"
    if version:
        name += f"-{version}"
    if version_suffix:
        name += f"{version_suffix}"
    if archive_name_suffix:
        name += f"-{archive_name_suffix}"
    name += f"-{platform_name}-{arch}{archive_ext}"
    return name


def _resolve_config_cwd(build_dir, config, use_ninja):
    """Resolve the per-config working directory for asset packaging."""
    config_build_dir = os.path.join(build_dir, config)
    if is_windows() and not use_ninja:
        return os.path.join(config_build_dir, config)
    return config_build_dir


def build_archive_asset(
    source_dir,
    build_dir,
    configs,
    archive_name_suffix=None,
    version_suffix="",
    use_ninja=False,
    target_arch=None,
):
    """
    Build archive asset packages containing QNN EP and dependencies.

    Args:
        source_dir: Path to source directory
        build_dir: Path to build directory
        configs: List of build configurations (e.g., ['RelWithDebInfo'])
        archive_name_suffix: Optional suffix for archive filename
        version_suffix: Optional version suffix for archive filename
        use_ninja: Whether Ninja generator was used
        target_arch: Override for platform.machine() in the archive filename.
            Required for cross-compile builds (e.g. arm64ec on x64) where the
            build host arch differs from the target arch.

    Returns:
        list[Path]: List of created archive file paths
    """
    created_archives = []

    for config in configs:
        log.info(f"Building archive asset for {config} configuration")

        cwd = _resolve_config_cwd(build_dir, config, use_ninja)
        if not os.path.exists(cwd):
            raise FileNotFoundError(f"Build directory not found: {cwd}")

        # Create dist directory
        dist_dir = os.path.join(cwd, "dist")
        os.makedirs(dist_dir, exist_ok=True)

        archive_ext = ".zip" if is_windows() else ".tgz"
        archive_name = _compute_archive_name(
            source_dir, version_suffix, archive_name_suffix, archive_ext, target_arch=target_arch
        )
        archive_path = Path(dist_dir) / archive_name

        # Get list of files to include
        asset_files = get_qnn_asset_file_list()
        asset_files.extend(
            [
                "LICENSE",
                "Qualcomm_LICENSE.pdf",
                "Privacy.md",
                "ThirdPartyNotices.txt",
                "README.md",
                "release-notes.md",
            ]
        )
        doc_md_files = ["QNN-ExecutionProvider.md", "build.md", "development.md"]
        doc_png_files = [
            "qnn_ep_quant_workflow.png",
            "quantization_mixed_precision_1.png",
            "quantization_mixed_precision_2.png",
            "PluginEP-final.png",
            "Q-icon-rgb-blue.png",
            "header.png",
        ]
        asset_files.extend(doc_md_files)
        asset_files.extend(doc_png_files)

        necessary_files_dict = {
            "windows": [
                "onnxruntime_providers_qnn.dll",
            ],
            "others": [
                "libonnxruntime_providers_qnn.so",
            ],
        }
        necessary_files = necessary_files_dict["windows"] if is_windows() else necessary_files_dict["others"]

        # Collect and verify files exist
        missing_files = []
        found_files = []

        for filename in asset_files:
            if filename in doc_md_files:
                file_path = os.path.join(cwd, "docs", "execution_providers", filename)
            elif filename in doc_png_files:
                file_path = os.path.join(cwd, "docs", "images", filename)
            else:
                file_path = os.path.join(cwd, filename)

            if os.path.exists(file_path):
                found_files.append(file_path)
                log.debug(f"Found asset file: {file_path}")
            else:
                missing_files.append(filename)

        if missing_files:
            log.warning(f"Missing asset files in {cwd}:")
            for missing in missing_files:
                if missing in necessary_files:
                    raise FileNotFoundError(f"Required file missing: {missing}")
                log.warning(f"  - {missing}")
            log.warning("Continuing with available files...")

        if not found_files:
            raise FileNotFoundError(f"No asset files found in {cwd}. Expected files: {asset_files}")

        # Create archive file
        log.info(f"Creating archive: {archive_path}")
        if is_windows():
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in found_files:
                    arcname = os.path.relpath(file_path, cwd)
                    zipf.write(file_path, arcname)
                    log.debug(f"Added to zip: {arcname}")
        else:
            with tarfile.open(archive_path, "w:gz") as tgzf:
                for file_path in found_files:
                    arcname = os.path.relpath(file_path, cwd)
                    tgzf.add(file_path, arcname)
                    log.debug(f"Added to tgz: {arcname}")

        log.info(f"Created archive: {archive_path} ({len(found_files)} files)")
        created_archives.append(archive_path)

    return created_archives


def build_pdb_archive_asset(
    source_dir,
    build_dir,
    configs,
    version_suffix="",
    use_ninja=False,
    target_arch=None,
):
    """
    Build a Windows-only archive containing PDB debug symbol files.

    The resulting archive is a sibling of the main asset archive and uses the same
    naming convention with a trailing "-pdb" suffix, e.g.:
        onnxruntime-qnn-<version>[<version_suffix>]-win-<arch>-pdb.zip

    Args:
        source_dir: Path to source directory
        build_dir: Path to build directory
        configs: List of build configurations (e.g., ['RelWithDebInfo'])
        version_suffix: Optional version suffix for archive filename
        use_ninja: Whether Ninja generator was used
        target_arch: Override for platform.machine() in the archive filename.
            Required for cross-compile builds (e.g. arm64ec on x64) where the
            build host arch differs from the target arch.

    Returns:
        list[Path]: List of created archive file paths (empty on non-Windows).
    """
    if not is_windows():
        log.info("Skipping PDB archive: not on Windows")
        return []

    pdb_files = ["onnxruntime_providers_qnn.pdb"]
    created_archives = []

    for config in configs:
        log.info(f"Building PDB archive asset for {config} configuration")

        cwd = _resolve_config_cwd(build_dir, config, use_ninja)
        if not os.path.exists(cwd):
            raise FileNotFoundError(f"Build directory not found: {cwd}")

        dist_dir = os.path.join(cwd, "dist")
        os.makedirs(dist_dir, exist_ok=True)

        base_name = _compute_archive_name(source_dir, version_suffix, None, "", target_arch=target_arch)
        archive_path = Path(dist_dir) / f"{base_name}-pdb.zip"

        found_files = []
        for filename in pdb_files:
            file_path = os.path.join(cwd, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required PDB file missing: {file_path}")
            found_files.append(file_path)

        log.info(f"Creating PDB archive: {archive_path}")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in found_files:
                arcname = os.path.relpath(file_path, cwd)
                zipf.write(file_path, arcname)
                log.debug(f"Added to PDB archive: {arcname}")

        log.info(f"Created PDB archive: {archive_path} ({len(found_files)} files)")
        created_archives.append(archive_path)

    return created_archives


def main():
    """
    Main entry point for standalone execution of pkg_assets.py
    """
    parser = argparse.ArgumentParser(
        description="Build QNN asset archive packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pkg_assets.py --source_dir . --build_dir build --config RelWithDebInfo
  python pkg_assets.py --source_dir . --build_dir build --config Debug --config Release --suffix custom
        """,
    )

    parser.add_argument("--source_dir", required=True, help="Path to source directory containing VERSION_NUMBER file")

    parser.add_argument("--build_dir", required=True, help="Path to build directory containing compiled assets")

    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Build configuration(s) to package (e.g., RelWithDebInfo, Debug). Can be specified multiple times.",
    )

    parser.add_argument("--suffix", help="Optional suffix for archive filename")

    parser.add_argument("--version_suffix", type=str, default="", help="Optional version suffix for archive filename")

    parser.add_argument("--use_ninja", action="store_true", help="Whether Ninja generator was used for build")

    parser.add_argument(
        "--pdb_only",
        action="store_true",
        help="Build a Windows PDB-only archive (onnxruntime-qnn-<version>-win-<arch>-pdb.zip) instead of the main asset archive.",
    )

    parser.add_argument(
        "--target_arch",
        type=str,
        default=None,
        help="Target architecture for the archive filename (overrides platform.machine()). "
        "Required for cross-compile builds such as arm64ec on an x64 host.",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Default to RelWithDebInfo if no configs specified
    if not args.config:
        args.config = ["RelWithDebInfo"]

    try:
        if args.pdb_only:
            created_archives = build_pdb_archive_asset(
                source_dir=args.source_dir,
                build_dir=args.build_dir,
                configs=args.config,
                version_suffix=args.version_suffix,
                use_ninja=args.use_ninja,
                target_arch=args.target_arch,
            )
        else:
            created_archives = build_archive_asset(
                source_dir=args.source_dir,
                build_dir=args.build_dir,
                configs=args.config,
                archive_name_suffix=args.suffix,
                version_suffix=args.version_suffix,
                use_ninja=args.use_ninja,
                target_arch=args.target_arch,
            )

        print(f"Successfully created {len(created_archives)} archive package(s):")
        for archive_path in created_archives:
            print(f"  {archive_path}")

    except Exception as e:
        log.error(f"Failed to create archive packages: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
