#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Merge two architecture-specific wheels into a unified wheel package.

This script takes AMD64 and ARM64EC wheels and combines them into a single
wheel with both library sets in separate subdirectories.
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def extract_wheel(wheel_path, extract_dir):
    """Extract a wheel file to a directory."""
    logging.info(f"  Extracting: {wheel_path}")
    with zipfile.ZipFile(wheel_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir


def get_package_dir(extract_dir):
    """Find the main package directory in extracted wheel."""
    extract_path = Path(extract_dir)
    # Look for onnxruntime_qnn directory
    for item in extract_path.iterdir():
        if item.is_dir() and item.name.startswith("onnxruntime"):
            if not item.name.endswith(".dist-info"):
                return item
    raise ValueError(f"Could not find package directory in {extract_dir}")


def is_library_file(filename):
    """Check if a file is a library file that should be moved."""
    suffixes = {".dll", ".so", ".cat", ".pyd", ".dylib"}
    return Path(filename).suffix.lower() in suffixes


def merge_wheels(amd64_wheel, arm64ec_wheel, output_folder):
    """
    Merge two architecture-specific wheels into a unified wheel.

    Args:
        amd64_wheel: Path to AMD64 wheel file
        arm64ec_wheel: Path to ARM64EC wheel file
        output_folder: Output folder path for output unified wheel
    """
    logging.info("=" * 80)
    logging.info("Merging Wheels into Unified Package")
    logging.info("=" * 80)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract both wheels
        logging.info("[Step 1] Extracting wheels...")
        amd64_dir = temp_path / "amd64"
        extract_wheel(amd64_wheel, amd64_dir)

        arm64ec_dir = temp_path / "arm64ec"
        extract_wheel(arm64ec_wheel, arm64ec_dir)

        # Create unified structure
        logging.info("[Step 2] Creating unified structure...")
        unified_dir = temp_path / "unified"
        unified_dir.mkdir()

        # Get package directories
        amd64_pkg = get_package_dir(amd64_dir)
        arm64ec_pkg = get_package_dir(arm64ec_dir)
        logging.info(f"  AMD64 package: {amd64_pkg.name}")
        logging.info(f"  ARM64EC package: {arm64ec_pkg.name}")

        # Copy AMD64 package as base
        logging.info("[Step 3] Copying base package structure...")
        unified_pkg = unified_dir / amd64_pkg.name
        shutil.copytree(amd64_pkg, unified_pkg)
        logging.info(f"  Created: {unified_pkg.name}/")

        # Create libs directory structure
        libs_dir = unified_pkg / "libs"
        libs_dir.mkdir(exist_ok=True)

        amd64_libs = libs_dir / "amd64"
        arm64ec_libs = libs_dir / "arm64ec"

        # Move AMD64 libraries to libs/amd64/
        logging.info("[Step 4] Organizing AMD64 libraries...")
        amd64_libs.mkdir()
        amd64_lib_count = 0
        for item in unified_pkg.iterdir():
            if item.is_file() and is_library_file(item.name):
                dest = amd64_libs / item.name
                shutil.move(str(item), str(dest))
                amd64_lib_count += 1
                logging.info(f"  Moved: {item.name} -> libs/amd64/")

        # Copy ARM64EC libraries to libs/arm64ec/
        logging.info("[Step 5] Copying ARM64EC libraries...")
        arm64ec_libs.mkdir()
        arm64ec_lib_count = 0
        for item in arm64ec_pkg.iterdir():
            if item.is_file() and is_library_file(item.name):
                dest = arm64ec_libs / item.name
                shutil.copy2(str(item), str(dest))
                arm64ec_lib_count += 1
                logging.info(f"  Copied: {item.name} -> libs/arm64ec/")

        # Copy platform_loader.py to package
        logging.info("[Step 6] Adding platform_loader.py...")
        platform_loader_src = Path(__file__).parent / "platform_loader.py"
        if platform_loader_src.exists():
            shutil.copy2(platform_loader_src, unified_pkg / "platform_loader.py")
            logging.info("  Added: platform_loader.py")
        else:
            logging.info(f"  WARNING: platform_loader.py not found at {platform_loader_src}")
            logging.info("  You may need to create this file manually")

        # Update __init__.py to call platform loader
        logging.info("[Step 7] Updating __init__.py...")
        init_file = unified_pkg / "__init__.py"
        if init_file.exists():
            with open(init_file, encoding="utf-8") as f:
                init_content = f.read()

            # Check if platform loader is already added
            if "platform_loader" not in init_content:
                init_lines = init_content.splitlines(keepends=True)

                # Add platform loader import at the beginning
                loader_code = """# Platform-aware library loading
try:
    from .platform_loader import setup_library_path
    _lib_dir_path = setup_library_path()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to setup platform-specific library path: {e}")
"""
                processed_lines = []
                added = False
                str_to_be_replaced = "os.path.dirname(os.path.abspath(__file__))"
                for line in init_lines:
                    if not added and not line.strip().startswith("#"):
                        processed_lines.append("\n" + loader_code + "\n")
                        added = True
                    new_line = line
                    if str_to_be_replaced in line:
                        new_line = line.replace(str_to_be_replaced, "os.path.abspath(_lib_dir_path)")
                    processed_lines.append(new_line)

                with open(init_file, "w", encoding="utf-8") as f:
                    f.writelines(processed_lines)
                logging.info("  Updated: __init__.py with platform loader")
            else:
                logging.info("  Skipped: __init__.py already has platform loader")
        else:
            logging.info("  WARNING: __init__.py not found")

        # Copy dist-info from AMD64 wheel
        logging.info("[Step 8] Copying dist-info...")
        for item in amd64_dir.iterdir():
            if item.name.endswith(".dist-info"):
                dest = unified_dir / item.name
                shutil.copytree(item, dest)
                logging.info(f"  Copied: {item.name}/")

        # Create the unified wheel
        logging.info("[Step 9] Creating unified wheel...")
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        subprocess.run(["wheel", "pack", unified_dir, "-d", output_folder], check=True)

        # Log summary
        logging.info("=" * 80)
        logging.info("Merge Complete!")
        logging.info("=" * 80)
        logging.info("Library Summary:")
        logging.info(f"  AMD64 libraries: {amd64_lib_count}")
        logging.info(f"  ARM64EC libraries: {arm64ec_lib_count}")
        logging.info(f"  Total: {amd64_lib_count + arm64ec_lib_count}")
        logging.info("Structure:")
        logging.info(f"  {unified_pkg.name}/")
        logging.info("    ├── libs/")
        logging.info(f"    │   ├── amd64/     ({amd64_lib_count} files)")
        logging.info(f"    │   └── arm64ec/    ({arm64ec_lib_count} files)")
        logging.info("    ├── platform_loader.py")
        logging.info("    └── __init__.py (with platform detection)")
        logging.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Merge AMD64 and ARM64EC wheels into a unified package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python merge_wheels.py \\
      --amd64-wheel dist/onnxruntime_qnn-1.0.0-py3-none-win_amd64.whl \\
      --arm64ec-wheel dist/onnxruntime_qnn-1.0.0-py3-none-win_arm64.whl \\
      --output-folder dist
        """,
    )
    parser.add_argument("--amd64-wheel", required=True, help="Path to AMD64 wheel file")
    parser.add_argument("--arm64ec-wheel", required=True, help="Path to ARM64EC wheel file")
    parser.add_argument("--output-folder", required=True, help="Output folder path for output unified wheel")

    args = parser.parse_args()

    # Validate input files exist
    amd64_path = Path(args.amd64_wheel)
    arm64ec_path = Path(args.arm64ec_wheel)

    if not amd64_path.exists():
        raise FileNotFoundError(f"AMD64 wheel not found: {args.amd64_wheel}")
    if not arm64ec_path.exists():
        raise FileNotFoundError(f"ARM64EC wheel not found: {args.arm64ec_wheel}")

    # Perform merge
    merge_wheels(args.amd64_wheel, args.arm64ec_wheel, args.output_folder)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
