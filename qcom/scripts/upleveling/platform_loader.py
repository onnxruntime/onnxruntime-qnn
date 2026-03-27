#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Platform-aware library loader for ONNX Runtime QNN
Automatically selects the correct library path based on runtime architecture
"""

import os
import platform
import sys
from pathlib import Path


def get_runtime_architecture():
    """
    Detect the current runtime architecture.

    Returns:
        str: 'amd64' for native AMD64 execution
             'arm64ec' for ARM64 device running x64 emulated code
             'arm64' for native ARM64 execution

    Raises:
        RuntimeError: If architecture is unsupported
    """
    machine = platform.machine().lower()

    # Check if running on ARM64
    if machine in ("arm64", "aarch64"):
        # Check if this is an x64 emulated process on ARM64
        # On Windows ARM64, x64 emulated processes have PROCESSOR_ARCHITECTURE=AMD64
        processor_arch = os.environ.get("PROCESSOR_ARCHITECTURE", "").upper()
        if processor_arch == "AMD64":
            return "arm64ec"  # ARM64 device running x64 emulated
        return "arm64"  # Native ARM64

    # Native AMD64 execution
    if machine in ("amd64", "x86_64", "x64"):
        return "amd64"

    raise RuntimeError(f"Unsupported architecture: {machine}")


def get_library_path():
    """
    Get the appropriate library path based on runtime architecture.

    Returns:
        Path: Path to the directory containing the correct libraries

    Raises:
        RuntimeError: If architecture is unsupported or libraries not found
    """
    arch = get_runtime_architecture()

    # Get the package installation directory
    package_dir = Path(__file__).parent
    libs_dir = package_dir / "libs"

    # Select the appropriate library subdirectory
    if arch == "amd64" or arch == "arm64ec":
        lib_path = libs_dir / arch
    else:
        raise RuntimeError(f"No libraries available for architecture: {arch}")

    if not lib_path.exists():
        raise RuntimeError(
            f"Library path not found: {lib_path}\nDetected architecture: {arch}\nPackage directory: {package_dir}"
        )

    return lib_path


def setup_library_path():
    """
    Setup the library search path for the current platform.
    This should be called during package initialization.

    Returns:
        Path: The library path that was added to the search path
    """
    lib_path = get_library_path()

    # Add to DLL search path (Windows)
    if sys.platform == "win32":
        # Python 3.8+ has os.add_dll_directory
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib_path))

        # Also add to PATH for older Python versions or as fallback
        os.environ["PATH"] = str(lib_path) + os.pathsep + os.environ.get("PATH", "")

    # Add to LD_LIBRARY_PATH (Linux)
    elif sys.platform.startswith("linux"):
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(lib_path) + os.pathsep + ld_path

    return lib_path


def get_platform_info():
    """
    Get detailed platform information for debugging.

    Returns:
        dict: Platform information including architecture, Python version, etc.
    """
    return {
        "runtime_architecture": get_runtime_architecture(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_version": sys.version,
        "processor_architecture": os.environ.get("PROCESSOR_ARCHITECTURE", "N/A"),
        "library_path": str(get_library_path()),
    }


if __name__ == "__main__":
    # Print platform information when run as a script
    import json

    info = get_platform_info()
    print(json.dumps(info, indent=2))
