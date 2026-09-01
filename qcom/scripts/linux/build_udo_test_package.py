#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Build the QNN UDO ("MyAdd") op-package test fixture consumed by
# onnxruntime/test/providers/qnn/udo_op_test.cc.
#
# This used to be built inline by cmake/onnxruntime_unittests_udo.cmake on every Linux x86_64 ORT
# build, which required downloading ~17 GB of pinned LLVM + Hexagon SDK toolchain for a ~4 MB test
# fixture. It is a test fixture, not a product artifact, so it is built here instead and published
# once per QAIRT SDK version -- see qcom/scripts/artifactory/{publish,download}_tool.py and
# .github/workflows/qualcomm-internal-publish-udo-package.yml.
#
# A prebuilt op package is locked to the QAIRT SDK's HTP/CPU backend ABI, not just its documented
# C API (QnnOpPackage.h is byte-identical across SDK versions that are NOT ABI-compatible). This
# script therefore validates every library it builds against the pinned SDK's own backend
# libraries before staging it -- see validate_package(). Re-run this script (and republish) any
# time qcom/packages.yml:qairt.version changes.
#
# Usage:
#   python3 qcom/scripts/linux/build_udo_test_package.py [--qairt-sdk PATH] [--output-dir DIR] [--keep-workdir]

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
UDO_TEST_DIR = REPO_ROOT / "onnxruntime" / "test" / "providers" / "qnn" / "udo"
PACKAGES_YML = REPO_ROOT / "qcom" / "packages.yml"

sys.path.insert(0, str(REPO_ROOT / "qcom" / "scripts" / "all"))
from package_manager import PackageManager  # noqa: E402

# Must match qcom/packages.yml. This script is now the only place these are duplicated (previously:
# qcom/packages.yml, cmake/onnxruntime_unittests_udo.cmake, and the forked
# onnxruntime/test/providers/qnn/udo/HTP_Makefile). Checked against packages.yml at startup below,
# so a drift here fails loudly instead of silently building against the wrong toolchain.
LLVM_VERSION = "21.1.8"
HEXAGON_SDK_VERSION = "6.5.0.0"
HEXAGON_TOOLS_VERSION = "19.0.07"

INTERFACE_SYMBOL = "MyAddOpPackageInterfaceProvider"

# libQnnCpu.so / libQnnHtp.so both declare these as NEEDED (readelf -d), so they are guaranteed to
# already be loaded in-process wherever the backend library is loaded. Undefined symbols an op
# package imports that these provide (operator new/delete, std::exception, typeinfo, ...) are
# therefore not actually missing at runtime, even though they are not exported by the backend
# library itself -- only genuinely backend-specific symbols (e.g. the QAIRT 2.46->2.49
# hnnx::PackageOpStorageBase ABI break) should be validated against the backend library.
RUNTIME_LIBRARY_SONAMES = ("libc++.so.1", "libc++abi.so.1", "libunwind.so.1")


class BuildError(RuntimeError):
    pass


def _check_pinned_versions() -> None:
    with PACKAGES_YML.open() as f:
        config = yaml.safe_load(f)
    expected = {"llvm_linux_x86_64": LLVM_VERSION, "hexagon_linux_x86_64": HEXAGON_SDK_VERSION}
    for package, version in expected.items():
        actual = config[package]["version"]
        if actual != version:
            raise BuildError(
                f"{PACKAGES_YML} pins {package}=={actual}, but this script hardcodes {version}. "
                "Update the constant at the top of this script to match."
            )


def _get_content_dir(package: str) -> Path:
    manager = PackageManager(package)
    manager.install()
    return manager.get_content_dir()


def _run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    logging.debug("Running: %s", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def _nm_defined_symbols(path: Path) -> set[str]:
    out = _run(["nm", "-D", "--defined-only", str(path)], capture_output=True, text=True).stdout
    return {line.split()[-1] for line in out.splitlines() if line.split()}


def _nm_undefined_mangled_symbols(path: Path) -> set[str]:
    out = _run(["nm", "-D", "--undefined-only", str(path)], capture_output=True, text=True).stdout
    symbols = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "U" and parts[1].startswith("_Z"):
            symbols.add(parts[1])
    return symbols


def _resolve_runtime_libraries() -> list[Path]:
    """Locate this host's copies of RUNTIME_LIBRARY_SONAMES via ldconfig. Best-effort: a library
    that can't be located is simply not added to the allowed-symbols set (validate_package will
    then correctly flag any symbol it alone would have provided as missing)."""
    try:
        out = _run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return []
    resolved = []
    for soname in RUNTIME_LIBRARY_SONAMES:
        for line in out.splitlines():
            name, _, path = line.strip().partition(" => ")
            if name.split(" ")[0] == soname and path:
                resolved.append(Path(path))
                break
    return resolved


def validate_package(so_path: Path, backend_lib_path: Path, runtime_libs: list[Path]) -> None:
    """Verify `so_path` exports its interface symbol and that every mangled C++ symbol it imports
    resolves against the QAIRT SDK's own backend library or the host's C++ runtime libraries. This
    is what catches an SDK ABI break (e.g. the QAIRT 2.46->2.49 PackageOpStorageBase constructor
    signature change) at package-build time instead of at dlopen()."""
    if not backend_lib_path.is_file():
        raise BuildError(f"Cannot validate {so_path}: {backend_lib_path} does not exist.")

    defined = _nm_defined_symbols(so_path)
    if INTERFACE_SYMBOL not in defined:
        raise BuildError(f"{so_path} does not export {INTERFACE_SYMBOL}.")

    undefined = _nm_undefined_mangled_symbols(so_path)
    known_defined = _nm_defined_symbols(backend_lib_path)
    for lib in runtime_libs:
        known_defined |= _nm_defined_symbols(lib)
    missing = sorted(undefined - known_defined)
    if missing:
        raise BuildError(
            f"{so_path} imports {len(missing)} C++ symbol(s) not exported by {backend_lib_path} "
            f"or {[str(p) for p in runtime_libs]}:\n  "
            + "\n  ".join(missing)
            + "\nThe QAIRT SDK's backend ABI does not match what this op package was built against. "
            "A prebuilt op package cannot float across QAIRT SDK versions -- do not publish this artifact."
        )


def build_cpu(sdk: Path, llvm_bin: Path, workdir: Path) -> Path:
    shutil.rmtree(workdir, ignore_errors=True)
    pkg_dir = workdir / "MyAddOpPackage"

    env = dict(os.environ, PYTHONPATH=str(sdk / "lib" / "python"))
    _run(
        [
            sys.executable,
            str(sdk / "bin" / "x86_64-linux-clang" / "qnn-op-package-generator"),
            "-p",
            str(UDO_TEST_DIR / "MyAddOpPackageCpu.xml"),
            "-o",
            str(workdir),
        ],
        env=env,
    )
    shutil.copyfile(UDO_TEST_DIR / "MyAddCPU.cpp", pkg_dir / "src" / "ops" / "MyAdd.cpp")

    _run(
        [
            "make",
            "-C",
            str(pkg_dir),
            f"QNN_SDK_ROOT={sdk}",
            f"CXX={llvm_bin / 'clang++'} -stdlib=libc++ -static-libstdc++ -Wl,--exclude-libs,ALL",
            "all_x86",
        ]
    )
    return pkg_dir / "libs" / "x86_64-linux-clang" / "libMyAddOpPackage.so"


def build_htp(sdk: Path, llvm_bin: Path, hexagon_sdk_root: Path, workdir: Path) -> Path:
    shutil.rmtree(workdir, ignore_errors=True)
    pkg_dir = workdir / "MyAddOpPackage"

    env = dict(os.environ, PYTHONPATH=str(sdk / "lib" / "python"))
    _run(
        [
            sys.executable,
            str(sdk / "bin" / "x86_64-linux-clang" / "qnn-op-package-generator"),
            "-p",
            str(UDO_TEST_DIR / "MyAddOpPackageHtp.xml"),
            "-o",
            str(workdir),
        ],
        env=env,
    )
    shutil.copyfile(UDO_TEST_DIR / "MyAddHTP.cpp", pkg_dir / "src" / "ops" / "MyAdd.cpp")

    # Use the SDK's own HTP Makefile template, unforked: the deltas this repo needs
    # (HEXAGON_SDK_ROOT_V*/HEXAGON_TOOLS_VERSION_*, the X86_CXX toolchain, and X86_LDFLAGS) are all
    # expressible as make command-line overrides, which beat the template's in-file `:=`. This also
    # means a future QAIRT header/flag fix (like the -I$(QNN_INCLUDE)/HTP/core fix PR #772 had to
    # hand-port for QAIRT 2.50) is picked up automatically on the next SDK bump, instead of
    # requiring a re-diff of a forked Makefile.
    #
    # X86_LDFLAGS statically links this pinned LLVM's own libc++.a/libc++abi.a. ORT is not built
    # with this LLVM, and the *runtime* libc++.so.1 available on CI/dev hosts is typically an older
    # distro build (e.g. libc++1-14) that does not export every libc++ symbol this LLVM's headers
    # can pull in (verified: -std=c++17 HTP/core headers reference std::__1::__hash_memory, which
    # libc++1-14 does not export) -- so the op package must not depend on the runtime libc++.so.1
    # for those symbols. `-l:libc++.a -l:libc++abi.a` needs no `-L` for this LLVM's lib dir: the
    # `-stdlib=libc++` flag on X86_CXX makes clang++ search its own installation's lib directory
    # automatically.
    htp_makefile = sdk / "share" / "QNN" / "OpPackageGenerator" / "makefiles" / "HTP" / "Makefile"
    if not htp_makefile.is_file():
        raise BuildError(f"QAIRT SDK is missing the expected HTP Makefile template: {htp_makefile}")

    libnative_release_dir = hexagon_sdk_root / "tools" / "HEXAGON_Tools" / HEXAGON_TOOLS_VERSION / "Tools"
    if not libnative_release_dir.is_dir():
        raise BuildError(f"Hexagon libnative directory not found: {libnative_release_dir}")

    x86_ldflags = (
        f"-Wl,--whole-archive -L{libnative_release_dir}/libnative/lib -lnative -Wl,--no-whole-archive "
        "-lpthread -l:libc++.a -l:libc++abi.a"
    )

    _run(
        [
            "make",
            "-C",
            str(pkg_dir),
            "-f",
            str(htp_makefile),
            f"QNN_SDK_ROOT={sdk}",
            f"HEXAGON_SDK_ROOT={hexagon_sdk_root}",
            f"X86_LIBNATIVE_RELEASE_DIR={libnative_release_dir}",
            f"X86_CXX={llvm_bin / 'clang++'} -stdlib=libc++",
            f"X86_LDFLAGS={x86_ldflags}",
            "htp_x86",
        ]
    )
    return pkg_dir / "build" / "x86_64-linux-clang" / "libQnnMyAddOpPackage.so"


def _read_sdk_metadata(sdk: Path) -> dict:
    sdk_yaml = sdk / "sdk.yaml"
    if not sdk_yaml.is_file():
        return {}
    with sdk_yaml.open() as f:
        data = yaml.safe_load(f) or {}
    return {k: data[k] for k in ("version", "build_id", "qnn_backend_api_version") if k in data}


def _git_head_sha() -> str:
    try:
        return _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [build_udo_test_package] [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Build and validate the QNN UDO test op-package fixture, staged for publishing to Artifactory."
    )
    parser.add_argument(
        "--qairt-sdk",
        type=Path,
        default=None,
        help="Path to a QAIRT SDK root. Defaults to the SDK pinned in qcom/packages.yml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "build" / "qnn-udo-test-package",
        help="Where to stage the built package.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the intermediate qnn-op-package-generator build directories, for debugging.",
    )
    args = parser.parse_args()

    if sys.version_info[:2] not in ((3, 10), (3, 12)):
        raise BuildError(
            "qnn-op-package-generator ships Python-version-tagged SDK extensions for 3.10 and 3.12 "
            f"only; this interpreter is {sys.version_info.major}.{sys.version_info.minor}."
        )

    _check_pinned_versions()

    sdk = args.qairt_sdk if args.qairt_sdk is not None else _get_content_dir("qairt")
    if not (sdk / "include" / "QNN" / "QnnOpPackage.h").is_file():
        raise BuildError(f"{sdk} does not look like a QAIRT SDK root (missing include/QNN/QnnOpPackage.h).")

    llvm_bin = _get_content_dir("llvm_linux_x86_64") / "bin"
    hexagon_sdk_root = _get_content_dir("hexagon_linux_x86_64") / HEXAGON_SDK_VERSION

    # Deliberately a sibling of --output-dir, not nested inside it: --output-dir is uploaded
    # verbatim (see qcom/scripts/artifactory/publish_tool.py), and these intermediate generator
    # trees should never end up in the published artifact even with --keep-workdir.
    workdir_root = REPO_ROOT / "build" / "_qnn_udo_test_package_workdir"

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True)

    try:
        cpu_so = build_cpu(sdk, llvm_bin, workdir_root / "cpu")
        htp_so = build_htp(sdk, llvm_bin, hexagon_sdk_root, workdir_root / "htp")

        runtime_libs = _resolve_runtime_libraries()
        validate_package(cpu_so, sdk / "lib" / "x86_64-linux-clang" / "libQnnCpu.so", runtime_libs)
        validate_package(htp_so, sdk / "lib" / "x86_64-linux-clang" / "libQnnHtp.so", runtime_libs)

        # Only stage into --output-dir once both backends have built and validated, so a failed
        # rebuild never leaves a stale/mismatched pair of .so files behind.
        shutil.copyfile(cpu_so, args.output_dir / "libMyAddOpPackage_cpu.so")
        shutil.copyfile(htp_so, args.output_dir / "libMyAddOpPackage_htp.so")
    finally:
        if not args.keep_workdir:
            shutil.rmtree(workdir_root, ignore_errors=True)

    sdk_metadata = _read_sdk_metadata(sdk)
    manifest = {
        "qairt_sdk_root": str(sdk),
        "qairt_sdk_metadata": sdk_metadata,
        "llvm_version": LLVM_VERSION,
        "hexagon_sdk_version": HEXAGON_SDK_VERSION,
        "hexagon_tools_version": HEXAGON_TOOLS_VERSION,
        "source_git_commit": _git_head_sha(),
        "build_date": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    logging.info(
        "Staged UDO test package at %s (QAIRT SDK metadata: %s)",
        args.output_dir,
        sdk_metadata or "<none found>",
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as e:
        logging.error(str(e))
        sys.exit(1)
