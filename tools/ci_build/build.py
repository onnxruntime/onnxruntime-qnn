#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# Licensed under the MIT License.

import contextlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, os.path.join(REPO_DIR, "tools", "python"))
from build_args import parse_arguments  # noqa: E402
from pkg_assets import build_zip_asset  # noqa: E402
from util import (  # noqa: E402
    generate_android_triplets,
    generate_linux_triplets,
    generate_windows_triplets,
    get_logger,
    is_linux,
    is_macOS,
    is_windows,
    parse_qnn_version_from_sdk_yaml,
    run,
)

log = get_logger("build")


class BaseError(Exception):
    """Base class for errors originating from build.py."""


class BuildError(BaseError):
    """Error from running build steps."""

    def __init__(self, *messages):
        super().__init__("\n".join(messages))


def is_reduced_ops_build(args):
    return args.include_ops_by_config is not None


def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    if command_or_path and command_or_path.strip():
        executable_path = shutil.which(command_or_path)
        if executable_path is None:
            raise BuildError(f"Failed to resolve executable path for '{command_or_path}'.")
        return os.path.abspath(executable_path)
    else:
        return None


def run_subprocess(
    args,
    cwd=None,
    capture_stdout=False,
    dll_path=None,
    shell=False,
    env=None,
    python_path=None,
):
    if env is None:
        env = {}
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            if "PATH" in my_env:
                my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
            else:
                my_env["PATH"] = dll_path
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path

    if python_path:
        if "PYTHONPATH" in my_env:
            my_env["PYTHONPATH"] += os.pathsep + python_path
        else:
            my_env["PYTHONPATH"] = python_path

    my_env.update(env)
    log.info(" ".join(args))
    return run(*args, cwd=cwd, capture_stdout=capture_stdout, shell=shell, env=my_env)


def update_submodules(source_dir):
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=source_dir)
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir)


def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)


def use_dev_mode(args):
    if args.compile_no_warning_as_error:
        return False
    if args.use_qnn:
        return True
    SYSTEM_COLLECTIONURI = os.getenv("SYSTEM_COLLECTIONURI")  # noqa: N806
    if SYSTEM_COLLECTIONURI:
        return False
    return True


def add_default_definition(definition_list, key, default_value):
    for x in definition_list:
        if x.startswith(key + "="):
            return definition_list
    definition_list.append(key + "=" + default_value)


def number_of_parallel_jobs(args):
    return os.cpu_count() if args.parallel == 0 else args.parallel


# See https://learn.microsoft.com/en-us/vcpkg/commands/install
def generate_vcpkg_install_options(build_dir, args):
    # NOTE: each option string should not contain any whitespace.
    vcpkg_install_options = ["--x-feature=tests"]
    vcpkg_install_options.append("--x-feature=qnn-ep")

    overlay_triplets_dir = None

    folder_name_parts = []
    if args.enable_address_sanitizer:
        folder_name_parts.append("asan")
    if args.use_binskim_compliant_compile_flags and not args.android:
        folder_name_parts.append("binskim")
    if args.disable_rtti:
        folder_name_parts.append("nortti")
    if args.disable_exceptions:
        folder_name_parts.append("noexception")
    if args.minimal_build is not None:
        folder_name_parts.append("minimal")
    if len(folder_name_parts) == 0:
        # It's hard to tell whether we must use a custom triplet or not. The official triplets work fine for most common situations. However, if a Windows build has set msvc toolset version via args.msvc_toolset then we need to, because we need to ensure all the source code are compiled by the same MSVC toolset version otherwise we will hit link errors like "error LNK2019: unresolved external symbol __std_mismatch_4 referenced in function ..."
        # So, to be safe we always use a custom triplet.
        folder_name = "default"
    else:
        folder_name = "_".join(folder_name_parts)
    overlay_triplets_dir = (Path(build_dir) / folder_name).absolute()

    vcpkg_install_options.append(f"--overlay-triplets={overlay_triplets_dir}")
    if "AGENT_TEMPDIRECTORY" in os.environ:
        temp_dir = os.environ["AGENT_TEMPDIRECTORY"]
        vcpkg_install_options.append(f"--x-buildtrees-root={temp_dir}")
    elif "RUNNER_TEMP" in os.environ:
        temp_dir = os.environ["RUNNER_TEMP"]
        vcpkg_install_options.append(f"--x-buildtrees-root={temp_dir}")

    # Config asset cache
    if args.use_vcpkg_ms_internal_asset_cache:
        terrapin_cmd_path = shutil.which("TerrapinRetrievalTool")
        if terrapin_cmd_path is None:
            terrapin_cmd_path = "C:\\local\\Terrapin\\TerrapinRetrievalTool.exe"
            if not os.path.exists(terrapin_cmd_path):
                terrapin_cmd_path = None
        if terrapin_cmd_path is not None:
            vcpkg_install_options.append(
                "--x-asset-sources=x-script,"
                + terrapin_cmd_path
                + " -b https://vcpkg.storage.devpackages.microsoft.io/artifacts/ -a true -u Environment -p {url} -s {sha512} -d {dst}\\;x-block-origin"
            )
        else:
            vcpkg_install_options.append(
                "--x-asset-sources=x-azurl,https://vcpkg.storage.devpackages.microsoft.io/artifacts/\\;x-block-origin"
            )

    return vcpkg_install_options


def generate_build_tree(
    cmake_path,
    source_dir,
    build_dir,
    ort_home,
    qnn_home,
    path_to_protoc_exe,
    configs,
    cmake_extra_defines,
    args,
    cmake_extra_args,
):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    cmake_args = [cmake_path, cmake_dir]
    if not use_dev_mode(args):
        cmake_args += ["--compile-no-warning-as-error"]

    types_to_disable = args.disable_types
    # enable/disable float 8 types
    disable_float8_types = args.android or ("float8" in types_to_disable)
    # enable/disable float 4 type
    disable_float4_types = args.android or ("float4" in types_to_disable)
    disable_optional_type = "optional" in types_to_disable
    disable_sparse_tensors = "sparsetensor" in types_to_disable

    cmake_args += [
        "-Donnxruntime_RUN_ONNX_TESTS=" + ("ON" if args.enable_onnx_tests else "OFF"),
        "-Donnxruntime_GENERATE_TEST_REPORTS=ON",
        "-DPython_EXECUTABLE=" + sys.executable,
        "-Donnxruntime_USE_VCPKG=" + ("ON" if args.use_vcpkg else "OFF"),
        "-Donnxruntime_USE_MIMALLOC=" + ("ON" if args.use_mimalloc else "OFF"),
        "-Donnxruntime_BUILD_SHARED_LIB=" + ("ON" if args.build_shared_lib else "OFF"),
        "-Donnxruntime_USE_QNN_INTERFACE=" + ("ON" if args.enable_generic_interface else "OFF"),
        "-Donnxruntime_DISABLE_RTTI=" + ("ON" if args.disable_rtti or args.minimal_build is not None else "OFF"),
        "-Donnxruntime_DISABLE_EXCEPTIONS=" + ("ON" if args.disable_exceptions else "OFF"),
        # Need to use 'is not None' with minimal_build check as it could be an empty list.
        "-Donnxruntime_MINIMAL_BUILD=" + ("ON" if args.minimal_build is not None else "OFF"),
        "-Donnxruntime_EXTENDED_MINIMAL_BUILD="
        + ("ON" if args.minimal_build and "extended" in args.minimal_build else "OFF"),
        "-Donnxruntime_MINIMAL_BUILD_CUSTOM_OPS="
        + ("ON" if (args.minimal_build is not None and ("custom_ops" in args.minimal_build)) else "OFF"),
        "-Donnxruntime_REDUCED_OPS_BUILD=" + ("ON" if is_reduced_ops_build(args) else "OFF"),
        "-Donnxruntime_CLIENT_PACKAGE_BUILD=" + ("ON" if args.client_package_build else "OFF"),
        "-Donnxruntime_BUILD_BENCHMARKS=" + ("ON" if args.build_micro_benchmarks else "OFF"),
        "-Donnxruntime_GCOV_COVERAGE=" + ("ON" if args.code_coverage else "OFF"),
        "-Donnxruntime_ENABLE_MEMORY_PROFILE=" + ("ON" if args.enable_memory_profile else "OFF"),
        "-Donnxruntime_DISABLE_FLOAT8_TYPES=" + ("ON" if disable_float8_types else "OFF"),
        "-Donnxruntime_DISABLE_FLOAT4_TYPES=" + ("ON" if disable_float4_types else "OFF"),
        "-Donnxruntime_DISABLE_SPARSE_TENSORS=" + ("ON" if disable_sparse_tensors else "OFF"),
        "-Donnxruntime_DISABLE_OPTIONAL_TYPE=" + ("ON" if disable_optional_type else "OFF"),
    ]
    if args.minimal_build is not None:
        add_default_definition(cmake_extra_defines, "ONNX_MINIMAL_BUILD", "ON")
    if args.rv64:
        add_default_definition(cmake_extra_defines, "onnxruntime_CROSS_COMPILING", "ON")
        if not args.riscv_toolchain_root:
            raise BuildError("The --riscv_toolchain_root option is required to build for riscv64.")
        if not args.skip_tests and not args.riscv_qemu_path:
            raise BuildError("The --riscv_qemu_path option is required for testing riscv64.")

        cmake_args += [
            "-DRISCV_TOOLCHAIN_ROOT:PATH=" + args.riscv_toolchain_root,
            "-DRISCV_QEMU_PATH:PATH=" + args.riscv_qemu_path,
            "-DCMAKE_TOOLCHAIN_FILE=" + os.path.join(source_dir, "cmake", "riscv64.toolchain.cmake"),
        ]

    if args.use_vcpkg:
        # Setup CMake flags for vcpkg

        # Find VCPKG's toolchain cmake file
        vcpkg_cmd_path = shutil.which("vcpkg")
        vcpkg_toolchain_path = None
        if vcpkg_cmd_path is not None:
            vcpkg_toolchain_path = Path(vcpkg_cmd_path).parent / "scripts" / "buildsystems" / "vcpkg.cmake"
            if not vcpkg_toolchain_path.exists():
                if is_windows():
                    raise BuildError(
                        "Cannot find VCPKG's toolchain cmake file. Please check if your vcpkg command was provided by Visual Studio"
                    )
                # Fallback to the next
                vcpkg_toolchain_path = None
        # Fallback to use the "VCPKG_INSTALLATION_ROOT" env var
        vcpkg_installation_root = os.environ.get("VCPKG_INSTALLATION_ROOT")
        if vcpkg_installation_root is None:
            # Fallback to checkout vcpkg from github
            vcpkg_installation_root = os.path.join(os.path.abspath(build_dir), "vcpkg")
            if not os.path.exists(vcpkg_installation_root):
                run_subprocess(
                    ["git", "clone", "-b", "2025.08.27", "https://github.com/microsoft/vcpkg.git", "--recursive"],
                    cwd=build_dir,
                )
        vcpkg_toolchain_path = Path(vcpkg_installation_root) / "scripts" / "buildsystems" / "vcpkg.cmake"

        if args.android:
            generate_android_triplets(build_dir, configs, args.android_cpp_shared, args.android_api)
        elif is_windows():
            generate_windows_triplets(build_dir, configs, args.msvc_toolset)
        else:
            # Linux, *BSD, AIX or other platforms
            generate_linux_triplets(build_dir, configs, args.use_full_protobuf)
        add_default_definition(cmake_extra_defines, "CMAKE_TOOLCHAIN_FILE", str(vcpkg_toolchain_path))

        # Choose the cmake triplet
        triplet = None
        if args.android:
            if args.android_abi == "armeabi-v7a":
                triplet = "arm-neon-android"
            elif args.android_abi == "arm64-v8a":
                triplet = "arm64-android"
            elif args.android_abi == "x86_64":
                triplet = "x64-android"
            elif args.android_abi == "x86":
                triplet = "x86-android"
            else:
                raise BuildError("Unknown android_abi")
        elif is_windows():
            target_arch = platform.machine()
            if args.arm64:
                target_arch = "ARM64"
            elif args.arm64ec:
                target_arch = "ARM64EC"
            cpu_arch = platform.architecture()[0]
            if target_arch == "AMD64":
                if cpu_arch == "32bit" or args.x86:
                    triplet = "x86-windows-static" if args.enable_msvc_static_runtime else "x86-windows-static-md"
                else:
                    triplet = "x64-windows-static" if args.enable_msvc_static_runtime else "x64-windows-static-md"
            elif target_arch == "ARM64":
                triplet = "arm64-windows-static" if args.enable_msvc_static_runtime else "arm64-windows-static-md"
            elif target_arch == "ARM64EC":
                triplet = "arm64ec-windows-static" if args.enable_msvc_static_runtime else "arm64ec-windows-static-md"
            else:
                raise BuildError("unknown python arch")
        if triplet:
            log.info(f"setting target triplet to {triplet}")
            add_default_definition(cmake_extra_defines, "VCPKG_TARGET_TRIPLET", triplet)

    # By default on Windows we currently support only cross compiling for ARM/ARM64
    if is_windows() and (args.arm64 or args.arm64ec or args.arm) and platform.architecture()[0] != "AMD64":
        # The onnxruntime_CROSS_COMPILING flag is deprecated. Prefer to use CMAKE_CROSSCOMPILING.
        add_default_definition(cmake_extra_defines, "onnxruntime_CROSS_COMPILING", "ON")
    if args.use_cache:
        cmake_args.append("-Donnxruntime_BUILD_CACHE=ON")
        if not (is_windows() and args.cmake_generator != "Ninja"):
            cmake_args.append("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
            cmake_args.append("-DCMAKE_C_COMPILER_LAUNCHER=ccache")

    if is_windows():
        if args.enable_msvc_static_runtime:
            add_default_definition(
                cmake_extra_defines, "CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            )
        # Set flags for 3rd-party libs
        if not args.use_vcpkg:
            if args.enable_msvc_static_runtime:
                add_default_definition(cmake_extra_defines, "ONNX_USE_MSVC_STATIC_RUNTIME", "ON")
                add_default_definition(cmake_extra_defines, "protobuf_MSVC_STATIC_RUNTIME", "ON")
                # The following build option was added in ABSL 20240722.0 and it must be explicitly set
                add_default_definition(cmake_extra_defines, "ABSL_MSVC_STATIC_RUNTIME", "ON")
                add_default_definition(cmake_extra_defines, "gtest_force_shared_crt", "OFF")
            else:
                # CMAKE_MSVC_RUNTIME_LIBRARY is default to MultiThreaded$<$<CONFIG:Debug>:Debug>DLL
                add_default_definition(cmake_extra_defines, "ONNX_USE_MSVC_STATIC_RUNTIME", "OFF")
                add_default_definition(cmake_extra_defines, "protobuf_MSVC_STATIC_RUNTIME", "OFF")
                add_default_definition(cmake_extra_defines, "ABSL_MSVC_STATIC_RUNTIME", "OFF")
                add_default_definition(cmake_extra_defines, "gtest_force_shared_crt", "ON")

    if qnn_home and os.path.exists(qnn_home):
        cmake_args += ["-Donnxruntime_QNN_HOME=" + qnn_home]

    if ort_home and os.path.exists(ort_home):
        cmake_args += ["-Donnxruntime_ORT_HOME=" + ort_home]

    if args.use_full_protobuf or args.gen_doc or args.enable_generic_interface:
        cmake_args += ["-Donnxruntime_USE_FULL_PROTOBUF=ON", "-DProtobuf_USE_STATIC_LIBS=ON"]

    if args.android:
        if not args.android_ndk_path:
            raise BuildError("android_ndk_path required to build for Android")
        if not args.android_sdk_path:
            raise BuildError("android_sdk_path required to build for Android")
        android_toolchain_cmake_path = os.path.join(args.android_ndk_path, "build", "cmake", "android.toolchain.cmake")
        cmake_args += [
            "-DANDROID_PLATFORM=android-" + str(args.android_api),
            "-DANDROID_ABI=" + str(args.android_abi),
            "-DANDROID_MIN_SDK=" + str(args.android_api),
            "-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=false",
        ]
        if args.disable_rtti:
            add_default_definition(cmake_extra_defines, "CMAKE_ANDROID_RTTI", "OFF")
        if args.disable_exceptions:
            add_default_definition(cmake_extra_defines, "CMAKE_ANDROID_EXCEPTIONS", "OFF")
        if not args.use_vcpkg:
            cmake_args.append("-DCMAKE_TOOLCHAIN_FILE=" + android_toolchain_cmake_path)
        else:
            cmake_args.append("-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=" + android_toolchain_cmake_path)

        if args.android_cpp_shared:
            cmake_args += ["-DANDROID_STL=c++_shared"]

    if args.use_qnn:
        if args.qnn_home is None or os.path.exists(args.qnn_home) is False:
            raise BuildError("qnn_home=" + qnn_home + " not valid." + " qnn_home paths must be specified and valid.")
        cmake_args += ["-Donnxruntime_USE_QNN=ON"]

        if args.use_qnn == "static_lib":
            cmake_args += ["-Donnxruntime_BUILD_QNN_EP_STATIC_LIB=ON"]
        if args.android and args.use_qnn != "static_lib":
            raise BuildError("Only support Android + QNN builds with QNN EP built as a static library.")
        if args.use_qnn == "static_lib" and args.enable_generic_interface:
            raise BuildError("Generic ORT interface only supported with QNN EP built as a shared library.")

    if path_to_protoc_exe:
        cmake_args += [f"-DONNX_CUSTOM_PROTOC_EXECUTABLE={path_to_protoc_exe}"]

    if args.cmake_deps_mirror_dir:
        cmake_args += [f"-Donnxruntime_CMAKE_DEPS_MIRROR_DIR={args.cmake_deps_mirror_dir}"]

    if args.fuzz_testing:
        if not (
            args.build_shared_lib
            and is_windows()
            and args.cmake_generator in ("Visual Studio 17 2022", "Visual Studio 18 2026")
            and args.use_full_protobuf
        ):
            raise BuildError("Fuzz test has only be tested with build shared libs option using MSVC on windows")
        cmake_args += [
            "-Donnxruntime_BUILD_UNIT_TESTS=ON",
            "-Donnxruntime_FUZZ_TEST=ON",
            "-Donnxruntime_USE_FULL_PROTOBUF=ON",
        ]

    if is_windows():
        if not args.android:
            if args.use_cache:
                add_default_definition(
                    cmake_extra_defines,
                    "CMAKE_MSVC_DEBUG_INFORMATION_FORMAT",
                    "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>",
                )
            else:
                # Always enable debug info even in release build. The debug information is in separated *.pdb files that
                # can be easily discarded when debug symbols are not needed. We enable it by default because many auditting
                # tools need to use the symbols.
                add_default_definition(cmake_extra_defines, "CMAKE_MSVC_DEBUG_INFORMATION_FORMAT", "ProgramDatabase")

        if number_of_parallel_jobs(args) > 0:
            # https://devblogs.microsoft.com/cppblog/improved-parallelism-in-msbuild/
            # NOTE: this disables /MP if set (according to comments on blog post).
            # By default, MultiProcMaxCount and CL_MPCount value are equal to the number of CPU logical processors.
            # See logic around setting CL_MPCount below
            cmake_args += ["-DCMAKE_VS_GLOBALS=UseMultiToolTask=true;EnforceProcessCountAcrossBuilds=true"]

    cmake_args += [f"-D{define}" for define in cmake_extra_defines]

    cmake_args += cmake_extra_args

    # ADO pipelines will store the pipeline build number
    # (e.g. 191101-2300.1.master) and source version in environment
    # variables. If present, use these values to define the
    # WinML/ORT DLL versions.
    build_number = os.getenv("Build_BuildNumber")  # noqa: SIM112
    source_version = os.getenv("Build_SourceVersion")  # noqa: SIM112
    if build_number and source_version:
        build_matches = re.fullmatch(r"(\d\d)(\d\d)(\d\d)(\d\d)\.(\d+)", build_number)
        if build_matches:
            YY = build_matches.group(2)  # noqa: N806
            MM = build_matches.group(3)  # noqa: N806
            DD = build_matches.group(4)  # noqa: N806

            # Get ORT major and minor number
            with open(os.path.join(source_dir, "VERSION_NUMBER")) as f:
                first_line = f.readline()
                ort_version_matches = re.match(r"(\d+).(\d+)", first_line)
                if not ort_version_matches:
                    raise BuildError("Couldn't read version from VERSION_FILE")
                ort_major = ort_version_matches.group(1)
                ort_minor = ort_version_matches.group(2)
                # Example (BuildNumber: 191101-2300.1.master,
                # SourceVersion: 0bce7ae6755c792eda558e5d27ded701707dc404)
                # MajorPart = 1
                # MinorPart = 0
                # BuildPart = 1911
                # PrivatePart = 123
                # String = 191101-2300.1.master.0bce7ae
                cmake_args += [
                    f"-DVERSION_MAJOR_PART={ort_major}",
                    f"-DVERSION_MINOR_PART={ort_minor}",
                    f"-DVERSION_BUILD_PART={YY}",
                    f"-DVERSION_PRIVATE_PART={MM}{DD}",
                    f"-DVERSION_STRING={ort_major}.{ort_minor}.{build_number}.{source_version[0:7]}",
                ]

    for config in configs:
        cflags = []
        cxxflags = None
        ldflags = None
        if is_windows() and not args.android:
            njobs = number_of_parallel_jobs(args)
            if njobs > 1:
                if args.parallel == 0:
                    cflags += ["/MP"]
                else:
                    cflags += [f"/MP{njobs}"]
        # Setup default values for cflags/cxxflags/ldflags.
        # The values set here are purely for security and compliance purposes. ONNX Runtime should work fine without these flags.
        if (args.use_binskim_compliant_compile_flags or args.enable_address_sanitizer) and not args.android:
            if is_windows():
                cflags += ["/guard:cf", "/DWIN32", "/D_WINDOWS"]

                # Target Windows 10
                cflags += [
                    "/DWINAPI_FAMILY=100",
                    "/DWINVER=0x0A00",
                    "/D_WIN32_WINNT=0x0A00",
                    "/DNTDDI_VERSION=0x0A000000",
                ]
                # The "/profile" flag implies "/DEBUG:FULL /DEBUGTYPE:cv,fixup /OPT:REF /OPT:NOICF /INCREMENTAL:NO /FIXED:NO". We set it for satisfying a Microsoft internal compliance requirement. External users
                # do not need to have it.
                ldflags = ["/profile", "/DYNAMICBASE"]
                # Address Sanitizer libs do not have a Qspectre version. So they two cannot be both enabled.
                if not args.enable_address_sanitizer:
                    cflags += ["/Qspectre"]
                if config == "Release":
                    cflags += ["/O2", "/Ob2", "/DNDEBUG"]
                elif config == "RelWithDebInfo":
                    cflags += ["/O2", "/Ob1", "/DNDEBUG"]
                elif config == "Debug":
                    cflags += ["/Ob0", "/Od", "/RTC1"]
                elif config == "MinSizeRel":
                    cflags += ["/O1", "/Ob1", "/DNDEBUG"]
                if args.enable_address_sanitizer:
                    cflags += ["/fsanitize=address"]
                cxxflags = cflags.copy()
                if not args.disable_exceptions:
                    cxxflags.append("/EHsc")
            elif is_linux() or is_macOS():
                if is_linux():
                    ldflags = ["-Wl,-Bsymbolic-functions", "-Wl,-z,relro", "-Wl,-z,now", "-Wl,-z,noexecstack"]
                else:
                    ldflags = []
                if config == "Release":
                    cflags = [
                        "-DNDEBUG",
                        "-U_FORTIFY_SOURCE",
                        "-D_FORTIFY_SOURCE=2",
                        "-Wp,-D_GLIBCXX_ASSERTIONS",
                        "-fstack-protector-strong",
                        "-O3",
                        "-pipe",
                    ]
                    if is_linux():
                        ldflags += ["-Wl,--strip-all"]
                elif config == "RelWithDebInfo":
                    cflags = [
                        "-DNDEBUG",
                        "-U_FORTIFY_SOURCE",
                        "-D_FORTIFY_SOURCE=2",
                        "-Wp,-D_GLIBCXX_ASSERTIONS",
                        "-fstack-protector-strong",
                        "-O3",
                        "-pipe",
                        "-g",
                    ]
                elif config == "Debug":
                    cflags = ["-g", "-O0"]
                    if args.enable_address_sanitizer:
                        cflags += ["-fsanitize=address"]
                        ldflags += ["-fsanitize=address"]
                elif config == "MinSizeRel":
                    cflags = [
                        "-DNDEBUG",
                        "-U_FORTIFY_SOURCE",
                        "-D_FORTIFY_SOURCE=2",
                        "-Wp,-D_GLIBCXX_ASSERTIONS",
                        "-fstack-protector-strong",
                        "-Os",
                        "-pipe",
                        "-g",
                    ]
                if is_linux() and platform.machine() == "x86_64":
                    # The following flags needs GCC 8 and newer
                    cflags += ["-fstack-clash-protection"]
                    if not args.rv64:
                        cflags += ["-fcf-protection"]
                cxxflags = cflags.copy()
        if cxxflags is None and cflags is not None and len(cflags) != 0:
            cxxflags = cflags.copy()
        config_build_dir = get_config_build_dir(build_dir, config)
        os.makedirs(config_build_dir, exist_ok=True)
        temp_cmake_args = cmake_args.copy()
        if cflags is not None and cxxflags is not None and len(cflags) != 0 and len(cxxflags) != 0:
            temp_cmake_args += [
                "-DCMAKE_C_FLAGS={}".format(" ".join(cflags)),
                "-DCMAKE_CXX_FLAGS={}".format(" ".join(cxxflags)),
            ]
        if ldflags is not None and len(ldflags) != 0:
            temp_cmake_args += [
                "-DCMAKE_EXE_LINKER_FLAGS_INIT={}".format(" ".join(ldflags)),
                "-DCMAKE_MODULE_LINKER_FLAGS_INIT={}".format(" ".join(ldflags)),
                "-DCMAKE_SHARED_LINKER_FLAGS_INIT={}".format(" ".join(ldflags)),
            ]
        env = {}
        if args.use_vcpkg:
            # append VCPKG_INSTALL_OPTIONS
            #
            # VCPKG_INSTALL_OPTIONS is a CMake list. It must be joined by semicolons
            # Therefore, if any of the option string contains a semicolon, it must be escaped
            temp_cmake_args += [
                "-DVCPKG_INSTALL_OPTIONS={}".format(
                    ";".join(generate_vcpkg_install_options(Path(build_dir) / config, args))
                )
            ]

            vcpkg_keep_env_vars = ["TRT_UPLOAD_AUTH_TOKEN"]

            #
            # Workaround for vcpkg failed to find the correct path of Python
            #
            # Since vcpkg does not inherit the environment variables `PATH` from the parent process, CMake will fail to
            # find the Python executable if the Python executable is not in the default location. This usually happens
            # to the Python installed by Anaconda.
            #
            # To minimize the impact of this problem, we set the `Python3_ROOT_DIR` environment variable to the
            # directory of current Python executable.
            #
            # see https://cmake.org/cmake/help/latest/module/FindPython3.html
            #
            env["Python3_ROOT_DIR"] = str(Path(os.path.dirname(sys.executable)).resolve())
            vcpkg_keep_env_vars += ["Python3_ROOT_DIR"]

            env["VCPKG_KEEP_ENV_VARS"] = ";".join(vcpkg_keep_env_vars)

        run_subprocess(
            [*temp_cmake_args, f"-DCMAKE_BUILD_TYPE={config}"],
            cwd=config_build_dir,
            env=env,
        )


def clean_targets(cmake_path, build_dir, configs):
    for config in configs:
        log.info("Cleaning targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path, "--build", build_dir2, "--config", config, "--target", "clean"]

        run_subprocess(cmd_args)


def build_targets(args, cmake_path, build_dir, configs, num_parallel_jobs, targets: list[str] | None):
    for config in configs:
        log.info("Building targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path, "--build", build_dir2, "--config", config]
        if targets:
            log.info(f"Building specified targets: {targets}")
            cmd_args.extend(["--target", *targets])

        build_tool_args = []
        if num_parallel_jobs != 0:
            if is_windows() and args.cmake_generator != "Ninja":
                # https://github.com/Microsoft/checkedc-clang/wiki/Parallel-builds-of-clang-on-Windows suggests
                # not maxing out CL_MPCount
                # Start by having one less than num_parallel_jobs (default is num logical cores),
                # limited to a range of 1..15
                # that gives maxcpucount projects building using up to 15 cl.exe instances each
                build_tool_args += [
                    f"/maxcpucount:{num_parallel_jobs}",
                    # one less than num_parallel_jobs, at least 1, up to 15
                    f"/p:CL_MPCount={min(max(num_parallel_jobs - 1, 1), 15)}",
                    # if nodeReuse is true, msbuild processes will stay around for a bit after the build completes
                    "/nodeReuse:False",
                ]
            elif args.cmake_generator == "Xcode":
                build_tool_args += [
                    "-parallelizeTargets",
                    "-jobs",
                    str(num_parallel_jobs),
                ]
            else:
                build_tool_args += [f"-j{num_parallel_jobs}"]

        if build_tool_args:
            cmd_args += ["--"]
            cmd_args += build_tool_args

        env = {}
        if args.android:
            env["ANDROID_SDK_ROOT"] = args.android_sdk_path
            env["ANDROID_NDK_HOME"] = args.android_ndk_path

        run_subprocess(cmd_args, env=env)


def build_python_wheel(
    source_dir,
    build_dir,
    configs,
    qnn_home,
    wheel_name_suffix,
    nightly_build=False,
    use_ninja=False,
):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows() and not use_ninja:
            cwd = os.path.join(cwd, config)

        args = [sys.executable, os.path.join(source_dir, "setup.py"), "bdist_wheel"]

        # Any combination of the following arguments can be applied
        if nightly_build:
            args.append("--nightly_build")
        if wheel_name_suffix:
            args.append(f"--wheel_name_suffix={wheel_name_suffix}")

        qnn_version = parse_qnn_version_from_sdk_yaml(qnn_home)
        if qnn_version:
            args.append(f"--qnn_version={qnn_version}")

        run_subprocess(args, cwd=cwd)


def build_qnn_ep_helper_assembly(source_dir, config):
    """Build the Qualcomm.ML.OnnxRuntime.QNN helper assembly"""
    helper_project_dir = os.path.join(source_dir, "csharp", "src", "Qualcomm.ML.OnnxRuntime.QNN")
    helper_project_file = os.path.join(helper_project_dir, "Qualcomm.ML.OnnxRuntime.QNN.csproj")

    if not os.path.exists(helper_project_file):
        log.warning(f"QNN EP helper project not found: {helper_project_file}")
        return False

    log.info("Building Qualcomm.ML.OnnxRuntime.QNN helper assembly...")
    try:
        run_subprocess(["dotnet", "build", helper_project_file, "-c", config], cwd=helper_project_dir)
        log.info("Successfully built QNN EP helper assembly")
        return True
    except Exception as e:
        log.error(f"Failed to build QNN EP helper assembly: {e}")
        return False


def build_nuget_package(
    source_dir,
    build_dir,
    configs,
    use_qnn,
    msbuild_extra_options,
    target_arch_name,
):
    if not (is_windows() or is_linux()):
        raise BuildError(
            "Currently csharp builds and nuget package creation is only supported on Windows and Linux platforms."
        )

    csharp_build_dir = os.path.join(source_dir, "csharp")

    # in most cases we don't want/need to include the MAUI mobile targets, as doing so means the mobile workloads
    # must be installed on the machine.
    # they are only included in the Microsoft.ML.OnnxRuntime nuget package
    sln = "OnnxRuntime.DesktopOnly.CSharp.sln"
    have_exclude_mobile_targets_option = "IncludeMobileTargets=false" in msbuild_extra_options

    # derive package name and execution provider based on the build args
    target_name = "/t:CreatePackage"
    execution_provider = "/p:ExecutionProvider=None"
    package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime"
    enable_training_tests = "/p:TrainingEnabledNativeBuild=false"
    target_architecture_name = "/p:TargetArchitecture=" + target_arch_name

    if use_qnn:
        if use_qnn != "shared_lib":
            raise BuildError("Currently NuGet packages with QNN require QNN EP to be built as a shared library.")
        execution_provider = "/p:ExecutionProvider=qnn"
        package_name = "/p:OrtPackageId=Qualcomm.ML.OnnxRuntime.QNN"

        # Build the QNN EP helper assembly for each config
        log.info("Building Qualcomm.ML.OnnxRuntime.QNN helper assembly...")
        for config in configs:
            if not build_qnn_ep_helper_assembly(source_dir, config):
                log.warning("Failed to build QNN EP helper assembly, continuing with package build...")
    elif any("OrtPackageId=" in x for x in msbuild_extra_options):
        pass
    else:
        # we currently only allow building with mobile targets on Windows.
        # it should be possible to allow building with android targets on Linux but that requires updating the
        # csproj to separate the inclusion of ios and android targets.
        if is_windows() and have_exclude_mobile_targets_option is False:
            # use the sln that include the mobile targets
            sln = "OnnxRuntime.CSharp.sln"

    # expand extra_options to add prefix
    extra_options = ["/p:" + option for option in msbuild_extra_options]

    # explicitly exclude mobile targets in this case
    if sln != "OnnxRuntime.CSharp.sln" and have_exclude_mobile_targets_option is False:
        extra_options.append("/p:IncludeMobileTargets=false")

    # we have to use msbuild directly if including Xamarin targets as dotnet only supports MAUI (.net6)
    use_dotnet = sln != "OnnxRuntime.CSharp.sln"

    # build csharp bindings and create nuget package for each config
    for config in configs:
        configuration = "/p:Configuration=" + config
        extra_options += [configuration, "/p:Platform=Any CPU"]
        if use_dotnet:
            cmd_args = ["dotnet", "restore", sln, "--configfile", "NuGet.CSharp.config", *extra_options]
        else:
            cmd_args = ["msbuild", sln, "/t:restore", "/p:RestoreConfigFile=NuGet.CSharp.config", *extra_options]

        # set build directory based on build_dir arg
        native_dir = os.path.normpath(os.path.join(source_dir, build_dir))
        ort_build_dir = "/p:OnnxRuntimeBuildDirectory=" + native_dir

        run_subprocess(cmd_args, cwd=csharp_build_dir)

        cmd_args = ["dotnet"] if use_dotnet else []
        cmd_args += [
            "msbuild",
            sln,
            package_name,
            ort_build_dir,
            enable_training_tests,
            *extra_options,
        ]

        run_subprocess(cmd_args, cwd=csharp_build_dir)

        if is_windows():
            # user needs to make sure nuget is installed and added to the path variable
            nuget_exe = "nuget.exe"
        else:
            # `dotnet pack` is used on Linux
            nuget_exe = "NugetExe_not_set"

        nuget_exe_arg = '/p:NugetExe="' + nuget_exe + '"'

        cmd_args = ["dotnet"] if use_dotnet else []
        cmd_args += [
            "msbuild",
            "OnnxRuntime.CSharp.proj",
            target_name,
            package_name,
            execution_provider,
            ort_build_dir,
            target_architecture_name,
            nuget_exe_arg,
            *extra_options,
        ]

        run_subprocess(cmd_args, cwd=csharp_build_dir)

        log.info(f"nuget package was created in the {config} build output directory.")


def generate_documentation(source_dir, build_dir, configs, validate):
    # Randomly choose one build config
    config = next(iter(configs))
    cwd = get_config_build_dir(build_dir, config)
    if is_windows():
        cwd = os.path.join(cwd, config)

    contrib_op_doc_path = os.path.join(source_dir, "docs", "ContribOperators.md")
    opkernel_doc_path = os.path.join(source_dir, "docs", "OperatorKernels.md")
    shutil.copy(os.path.join(source_dir, "tools", "python", "gen_contrib_doc.py"), cwd)
    shutil.copy(os.path.join(source_dir, "tools", "python", "gen_opkernel_doc.py"), cwd)
    # limit to just com.microsoft (excludes purely internal stuff like com.microsoft.nchwc).
    run_subprocess(
        [sys.executable, "gen_contrib_doc.py", "--output_path", contrib_op_doc_path, "--domains", "com.microsoft"],
        cwd=cwd,
    )
    # we currently limit the documentation created by a build to a subset of EP's.
    # Run get_opkernel_doc.py directly if you need/want documentation from other EPs that are enabled in the build.
    run_subprocess(
        [
            sys.executable,
            "gen_opkernel_doc.py",
            "--output_path",
            opkernel_doc_path,
            "--providers",
            "CPU",
            "CUDA",
            "DML",
        ],
        cwd=cwd,
    )

    if validate:
        try:
            have_diff = False

            def diff_file(path, regenerate_qualifiers=""):
                diff = subprocess.check_output(["git", "diff", "--ignore-blank-lines", path], cwd=source_dir).decode(
                    "utf-8"
                )
                if diff:
                    nonlocal have_diff
                    have_diff = True
                    log.warning(
                        f"The updated document {path} is different from the checked in version. "
                        f"Please regenerate the file{regenerate_qualifiers}, or copy the updated version from the "
                        "CI build's published artifacts if applicable."
                    )
                    log.debug("diff:\n" + diff)  # noqa: G003

            diff_file(opkernel_doc_path, " with CPU, CUDA and DML execution providers enabled")
            diff_file(contrib_op_doc_path)

            if have_diff:
                # Output for the CI to publish the updated md files as an artifact
                print("##vso[task.setvariable variable=DocUpdateNeeded]true")
                raise BuildError("Generated documents have diffs. Check build output for details.")

        except subprocess.CalledProcessError:
            raise BuildError("git diff returned non-zero error code")  # noqa: B904


def main():
    log.debug("Command line arguments:\n  {}".format(" ".join(shlex.quote(arg) for arg in sys.argv[1:])))  # noqa: G001

    args = parse_arguments()

    print(args)

    if os.getenv("ORT_BUILD_WITH_CACHE") == "1":
        args.use_cache = True

    # VCPKG's scripts/toolchains/android.cmake has logic for autodetecting NDK home when the ANDROID_NDK_HOME env is not set, but it is only implemented for Windows
    if args.android and args.use_vcpkg and args.android_ndk_path is not None and os.path.exists(args.android_ndk_path):
        os.environ["ANDROID_NDK_HOME"] = args.android_ndk_path

    if not is_windows() and not is_macOS():
        if not args.allow_running_as_root:
            is_root_user = os.geteuid() == 0
            if is_root_user:
                raise BuildError(
                    "Running as root is not allowed. If you really want to do that, use '--allow_running_as_root'."
                )

    cmake_extra_defines = list(args.cmake_extra_defines)

    if args.code_coverage and not args.android:
        raise BuildError("Using --code_coverage requires --android")

    configs = set(args.config)

    # setup paths and directories
    # cmake_path can be None. For example, if a person only wants to run the tests, he/she doesn't need to have cmake.
    cmake_path = resolve_executable_path(args.cmake_path)
    build_dir = args.build_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))

    qnn_home = ""
    if args.use_qnn:
        qnn_home = args.qnn_home
    ort_home = args.ort_home

    if args.update or args.build:
        for config in configs:
            os.makedirs(get_config_build_dir(build_dir, config), exist_ok=True)

    log.info("Build started")

    if args.update:
        cmake_extra_args = []
        path_to_protoc_exe = None
        if args.path_to_protoc_exe:
            path_to_protoc_exe = Path(args.path_to_protoc_exe)
            if not path_to_protoc_exe.exists():
                raise BuildError("The value to --path_to_protoc_exe is invalid.")
        if not args.skip_submodule_sync:
            update_submodules(source_dir)
        if is_windows():
            cpu_arch = platform.architecture()[0]
            if args.cmake_generator == "Ninja":
                if cpu_arch == "32bit" or args.arm or args.arm64 or args.arm64ec:
                    raise BuildError(
                        "To cross-compile with Ninja, load the toolset "
                        "environment for the target processor (e.g. Cross "
                        "Tools Command Prompt for VS)"
                    )
                cmake_extra_args = ["-G", args.cmake_generator]
            elif args.arm or args.arm64 or args.arm64ec:
                if args.arm:
                    cmake_extra_args = ["-A", "ARM"]
                elif args.arm64:
                    cmake_extra_args = ["-A", "ARM64"]
                    if args.buildasx:
                        cmake_extra_args += ["-D", "BUILD_AS_ARM64X=ARM64"]
                elif args.arm64ec:
                    cmake_extra_args = ["-A", "ARM64EC"]
                    if args.buildasx:
                        cmake_extra_args += ["-D", "BUILD_AS_ARM64X=ARM64EC"]
                cmake_extra_args += ["-G", args.cmake_generator]
                # Cannot test on host build machine for cross-compiled
                # builds (Override any user-defined behavior for test if any)
                if args.test:
                    log.warning(
                        "Cannot test on host build machine for cross-compiled "
                        "ARM(64) builds. Will skip test running after build."
                    )
                    args.test = False
            else:
                target_arch = platform.machine()
                if target_arch == "AMD64":
                    if cpu_arch == "32bit" or args.x86:
                        target_arch = "Win32"
                    else:
                        target_arch = "x64"
                    host_arch = "x64"
                elif target_arch == "ARM64":
                    host_arch = "ARM64"
                else:
                    raise BuildError("unknown python arch")
                if args.msvc_toolset:
                    toolset = "host=" + host_arch + ",version=" + args.msvc_toolset
                else:
                    toolset = "host=" + host_arch
                if args.windows_sdk_version:
                    target_arch += ",version=" + args.windows_sdk_version
                cmake_extra_args = ["-A", target_arch, "-T", toolset, "-G", args.cmake_generator]
            if args.enable_wcos:
                cmake_extra_defines.append("CMAKE_USER_MAKE_RULES_OVERRIDE=wcos_rules_override.cmake")

        elif args.cmake_generator is not None:
            cmake_extra_args += ["-G", args.cmake_generator]

        generate_build_tree(
            cmake_path,
            source_dir,
            build_dir,
            ort_home,
            qnn_home,
            path_to_protoc_exe,
            configs,
            cmake_extra_defines,
            args,
            cmake_extra_args,
        )

    if args.clean:
        clean_targets(cmake_path, build_dir, configs)

    if args.build:
        if args.parallel < 0:
            raise BuildError(f"Invalid parallel job count: {args.parallel}")
        num_parallel_jobs = number_of_parallel_jobs(args)
        build_targets(args, cmake_path, build_dir, configs, num_parallel_jobs, args.targets)

    # if args.test:
    #     if args.enable_onnx_tests:
    #         source_onnx_model_dir = "C:\\local\\models" if is_windows() else "/data/models"
    #         setup_test_data(source_onnx_model_dir, "models", build_dir, configs)

    #     run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs)

    #     # run node.js binding tests
    #     if args.build_nodejs and not args.skip_nodejs_tests:
    #         nodejs_binding_dir = os.path.normpath(os.path.join(source_dir, "js", "node"))
    #         run_nodejs_tests(nodejs_binding_dir)

    # Build packages after running the tests.
    # NOTE: if you have a test that rely on a file which only get copied/generated during packaging step, it could
    # fail unexpectedly. Similar, if your packaging step forgot to copy a file into the package, we don't know it
    # either.
    if args.build:
        # TODO: find asan DLL and copy it to onnxruntime/capi folder when args.enable_address_sanitizer is True and
        #  the target OS is Windows
        if args.build_wheel:
            nightly_build = bool(os.getenv("NIGHTLY_BUILD") == "1")
            build_python_wheel(
                source_dir,
                build_dir,
                configs,
                args.qnn_home,
                args.wheel_name_suffix,
                nightly_build=nightly_build,
                use_ninja=(args.cmake_generator == "Ninja"),
            )

        if args.build_nuget:
            platform_arch = platform.machine()
            if platform_arch == "ARM64" or args.arm64 or args.arm64ec:
                target_arch_name = "arm64"
            elif args.arm:
                target_arch_name = "arm"
            elif platform_arch == "AMD64":
                if args.x86:
                    target_arch_name = "x86"
                else:
                    target_arch_name = "x64"
            else:
                raise BuildError("unknown python arch")
            build_nuget_package(
                source_dir,
                build_dir,
                configs,
                args.use_qnn,
                args.msbuild_extra_options,
                target_arch_name,
            )

        if args.build_zip_asset:
            build_zip_asset(
                source_dir,
                build_dir,
                configs,
                args.zip_asset_name_suffix,
                use_ninja=(args.cmake_generator == "Ninja"),
            )

    if args.gen_doc:
        # special case CI where we create the build config separately to building
        if args.update and not args.build:
            pass
        else:
            # assumes build has occurred for easier use in CI where we don't always build via build.py and need to run
            # documentation generation as a separate task post-build
            generate_documentation(source_dir, build_dir, configs, args.gen_doc == "validate")

    log.info("Build complete")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseError as e:
        log.error(str(e))
        sys.exit(1)
