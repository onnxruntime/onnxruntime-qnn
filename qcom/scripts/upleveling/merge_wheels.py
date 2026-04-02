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
import subprocess
import tempfile
import zipfile
from pathlib import Path


def is_library_file(filename):
    """Check if a file is a library file that should be moved."""
    suffixes = {".dll", ".so", ".cat", ".pyd", ".dylib"}
    return Path(filename).suffix.lower() in suffixes


def get_package_name_from_zip(zip_file):
    """Find the main package directory name in a wheel zip file."""
    for name in zip_file.namelist():
        parts = name.split("/")
        if len(parts) > 0 and parts[0].startswith("onnxruntime_qnn") and not parts[0].endswith(".dist-info"):
            return parts[0]
    raise ValueError("Could not find package directory in wheel")


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

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    output_wheel = output_path / Path(amd64_wheel).name

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_wheel = Path(temp_dir) / "temp.whl"

        # Open AMD64 wheel and get package name
        logging.info("[Step 1] Opening AMD64 wheel as base...")
        with zipfile.ZipFile(amd64_wheel, "r") as amd64_zip:
            package_name = get_package_name_from_zip(amd64_zip)
            logging.info(f"  Package: {package_name}")

            # Create new wheel with reorganized AMD64 files
            logging.info("[Step 2] Reorganizing AMD64 libraries...")
            amd64_lib_count = 0

            with zipfile.ZipFile(temp_wheel, "w", zipfile.ZIP_DEFLATED) as output_zip:
                # Copy all files, moving libraries to libs/amd64/
                for item in amd64_zip.infolist():
                    data = amd64_zip.read(item.filename)

                    # Check if this is a library file in the package root
                    if item.filename.startswith(f"{package_name}/") and is_library_file(item.filename):
                        parts = item.filename.split("/")
                        if len(parts) == 2:  # package_name/filename
                            # Move to libs/amd64/
                            new_filename = f"{package_name}/libs/amd64/{parts[1]}"
                            output_zip.writestr(new_filename, data)
                            amd64_lib_count += 1
                            logging.info(f"  Moved: {parts[1]} -> libs/amd64/")
                            continue

                    # Copy file as-is
                    output_zip.writestr(item, data)

        # Append ARM64EC libraries
        logging.info("[Step 3] Appending ARM64EC libraries...")
        arm64ec_lib_count = 0

        with zipfile.ZipFile(arm64ec_wheel, "r") as arm64ec_zip, zipfile.ZipFile(temp_wheel, "a") as output_zip:
            for item in arm64ec_zip.infolist():
                # Only add library files from package root
                if item.filename.startswith(f"{package_name}/") and is_library_file(item.filename):
                    parts = item.filename.split("/")
                    if len(parts) == 2:
                        # Add to libs/arm64ec/
                        new_filename = f"{package_name}/libs/arm64ec/{parts[1]}"
                        data = arm64ec_zip.read(item.filename)
                        output_zip.writestr(new_filename, data)
                        arm64ec_lib_count += 1
                        logging.info(f"  Added: {parts[1]} -> libs/arm64ec/")

        # Append platform_loader.py
        logging.info("[Step 4] Appending platform_loader.py...")
        platform_loader_src = Path(__file__).parent / "platform_loader.py"

        with zipfile.ZipFile(temp_wheel, "a") as output_zip:
            if platform_loader_src.exists():
                output_zip.write(platform_loader_src, f"{package_name}/platform_loader.py")
                logging.info("  Added: platform_loader.py")
            else:
                logging.warning(f"  platform_loader.py not found at {platform_loader_src}")

        # Extract and repack with wheel pack for proper compression
        logging.info("[Step 5] Repacking with wheel pack for compression...")
        extract_dir = Path(temp_dir) / "extract"
        extract_dir.mkdir()

        with zipfile.ZipFile(temp_wheel, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        subprocess.run(["wheel", "pack", str(extract_dir), "-d", str(output_path)], check=True)
        logging.info(f"  Created: {Path(amd64_wheel).name}")

    # Log summary
    logging.info("=" * 80)
    logging.info("Merge Complete!")
    logging.info("=" * 80)
    logging.info("Library Summary:")
    logging.info(f"  AMD64 libraries: {amd64_lib_count}")
    logging.info(f"  ARM64EC libraries: {arm64ec_lib_count}")
    logging.info(f"  Total: {amd64_lib_count + arm64ec_lib_count}")
    logging.info("Structure:")
    logging.info(f"  {package_name}/")
    logging.info("    ├── libs/")
    logging.info(f"    │   ├── amd64/     ({amd64_lib_count} files)")
    logging.info(f"    │   └── arm64ec/    ({arm64ec_lib_count} files)")
    logging.info("    ├── platform_loader.py")
    logging.info("    └── __init__.py")
    logging.info(f"Output: {output_wheel}")
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
