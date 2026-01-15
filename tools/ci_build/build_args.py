# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.import argparse
import argparse
import os
import platform
import shlex
import sys
import warnings

from util import (
    is_macOS,
    is_windows,
)


def _str_to_bool(s: str) -> bool:
    """Convert string to bool (in argparse context) using match/case."""
    match s.lower():
        case "true":
            return True
        case "false":
            return False
        case _:
            raise ValueError(f"Invalid boolean value: {s!r}. Use 'true' or 'false'.")
    return False


# --- Argument Verification Helpers ---
def _qnn_verify_library_kind(library_kind: str) -> str:
    """Verifies the library kind for the QNN Execution Provider."""
    choices = ["shared_lib", "static_lib"]
    if library_kind not in choices:
        print("\nYou have specified an invalid library kind for QNN EP.")
        print(f"The invalid library kind was: {library_kind}")
        print("Provide a library kind from the following options: ", choices)
        print(f"Example: --use_qnn {choices[0]}")
        sys.exit("Incorrect build configuration")
    return library_kind


# --- Argument Grouping Functions ---


def add_core_build_args(parser: argparse.ArgumentParser) -> None:
    """Adds core build process arguments."""
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["Debug"],
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build.",
    )
    parser.add_argument("--update", action="store_true", help="Update makefiles.")
    parser.add_argument("--build", action="store_true", help="Build.")
    parser.add_argument(
        "--clean", action="store_true", help="Run 'cmake --build --target clean' for the selected config/s."
    )
    parser.add_argument(
        "--parallel",
        nargs="?",
        const="0",
        default="1",
        type=int,
        help="Use parallel build. Optional value specifies max jobs (0=num CPUs).",
    )
    parser.add_argument(
        "--target",
        nargs=1,
        action="extend",
        metavar="TARGET",
        dest="targets",
        help="Build a specific CMake target (e.g., winml_dll).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        action="extend",
        default=[],
        help="Build one or more specific CMake targets.",
    )
    parser.add_argument(
        "--compile_no_warning_as_error",
        action="store_true",
        help="Prevent warnings from being treated as errors during compile. Only works for cmake targets that honor the COMPILE_WARNING_AS_ERROR property",
    )
    parser.add_argument("--build_shared_lib", action="store_true", help="Build a shared library for ONNXRuntime.")
    parser.add_argument(
        "--build_apple_framework", action="store_true", help="Build a macOS/iOS framework for ONNXRuntime."
    )
    parser.add_argument("--enable_lto", action="store_true", help="Enable Link Time Optimization (LTO).")
    parser.add_argument("--use_cache", action="store_true", help="Use ccache in CI")
    parser.add_argument(
        "--use_binskim_compliant_compile_flags",
        action="store_true",
        help="[MS Internal] Use preset compile flags for BinSkim compliance.",
    )


def add_cmake_build_config_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to CMake and general build system configuration."""
    parser.add_argument(
        "--cmake_extra_defines",
        nargs="+",
        action="extend",
        default=[],
        help="Extra CMake definitions (-D<key>=<value>). Provide as <key>=<value>.",
    )
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake executable.")
    parser.add_argument(
        "--cmake_generator",
        choices=[
            "MinGW Makefiles",
            "Ninja",
            "NMake Makefiles",
            "NMake Makefiles JOM",
            "Unix Makefiles",
            "Visual Studio 17 2022",
            "Xcode",
        ],
        default=None,  # Will be set later based on OS and WASM
        help="Specify the generator for CMake.",
    )
    parser.add_argument(
        "--use_vcpkg", action="store_true", help="Use vcpkg for dependencies (requires CMAKE_TOOLCHAIN_FILE)."
    )
    parser.add_argument(
        "--use_vcpkg_ms_internal_asset_cache", action="store_true", help="[MS Internal] Use internal vcpkg asset cache."
    )
    parser.add_argument("--skip_submodule_sync", action="store_true", help="Skip 'git submodule update'.")
    parser.add_argument("--skip_pip_install", action="store_true", help="Skip 'pip install'.")


def add_testing_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to running tests."""
    parser.add_argument("--test", action="store_true", help="Run unit tests.")
    parser.add_argument("--skip_tests", action="store_true", help="Skip all tests.")
    parser.add_argument(
        "--ctest_path",
        default="ctest",
        help="Path to CTest. Empty string uses script to drive tests.",
    )
    parser.add_argument(
        "--enable_onnx_tests",
        action="store_true",
        help="Run onnx_test_runner against test data. Only used in ONNX Runtime's CI pipelines",
    )
    parser.add_argument("--path_to_protoc_exe", help="Path to protoc executable.")
    parser.add_argument("--cmake_deps_mirror_dir", help="Path to the local mirror of cmake dependencies.")
    parser.add_argument("--fuzz_testing", action="store_true", help="Enable Fuzz testing.")
    parser.add_argument(
        "--enable_symbolic_shape_infer_tests",
        action="store_true",
        help="Run symbolic shape inference tests.",
    )
    parser.add_argument("--skip_onnx_tests", action="store_true", help="Explicitly disable ONNX related tests.")
    parser.add_argument("--ctest_timeout", default="10800", help="Timeout provided to CTest --timeout (seconds).")
    parser.add_argument("--enable_transformers_tool_test", action="store_true", help="Enable transformers tool test.")
    parser.add_argument("--build_micro_benchmarks", action="store_true", help="Build ONNXRuntime micro-benchmarks.")
    parser.add_argument("--code_coverage", action="store_true", help="Generate code coverage report (Android only).")


def add_general_profiling_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to general (non-EP specific) profiling."""
    parser.add_argument("--enable_memory_profile", action="store_true", help="Enable memory profiling.")


def add_debugging_sanitizer_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to debugging, sanitizers, and compliance."""
    parser.add_argument(
        "--enable_address_sanitizer", action="store_true", help="Enable Address Sanitizer (ASan) (Linux/macOS/Windows)."
    )


def add_documentation_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to documentation generation."""
    parser.add_argument(
        "--gen_doc",
        nargs="?",
        const="yes",
        type=str,
        help="Generate operator/type docs. Use '--gen_doc validate' to check against /docs.",
    )


def add_cross_compile_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for cross-compiling to non-Windows target CPU architectures."""
    parser.add_argument(
        "--rv64",
        action="store_true",
        help="[cross-compiling] Target RISC-V 64-bit.",
    )
    parser.add_argument(
        "--riscv_toolchain_root",
        type=str,
        default="",
        help="Path to RISC-V toolchain root.",
    )
    parser.add_argument(
        "--riscv_qemu_path",
        type=str,
        default="",
        help="Path to RISC-V qemu executable.",
    )


def add_android_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for Android platform builds."""
    parser.add_argument("--android", action="store_true", help="Build for Android.")
    parser.add_argument(
        "--android_abi",
        default="arm64-v8a",
        choices=["armeabi-v7a", "arm64-v8a", "x86", "x86_64"],
        help="Target Android ABI.",
    )
    parser.add_argument("--android_api", type=int, default=27, help="Android API Level (e.g., 21).")
    parser.add_argument(
        "--android_sdk_path", type=str, default=os.environ.get("ANDROID_HOME", ""), help="Path to Android SDK."
    )
    parser.add_argument(
        "--android_ndk_path", type=str, default=os.environ.get("ANDROID_NDK_HOME", ""), help="Path to Android NDK."
    )
    parser.add_argument(
        "--android_cpp_shared",
        action="store_true",
        help="Link shared libc++ instead of static (default).",
    )
    parser.add_argument(
        "--android_run_emulator", action="store_true", help="Start an Android emulator if needed for tests."
    )


def add_windows_specific_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments specific to Windows builds or Windows cross-compilation."""
    # Build tools / config
    parser.add_argument("--msvc_toolset", help="MSVC toolset version (e.g., 14.11). Must be >=14.40")
    parser.add_argument("--windows_sdk_version", help="Windows SDK version (e.g., 10.0.19041.0).")
    parser.add_argument("--enable_msvc_static_runtime", action="store_true", help="Statically link MSVC runtimes.")
    parser.add_argument("--use_telemetry", action="store_true", help="Enable telemetry (official builds only).")
    parser.add_argument("--caller_framework", type=str, help="Name of the framework calling ONNX Runtime.")

    # Cross-compilation targets hosted on Windows
    parser.add_argument(
        "--x86",
        action="store_true",
        help="[Windows cross-compiling] Target Windows x86.",
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help="[Windows cross-compiling] Target Windows ARM.",
    )
    parser.add_argument(
        "--arm64",
        action="store_true",
        help="[Windows cross-compiling] Target Windows ARM64.",
    )
    parser.add_argument(
        "--arm64ec",
        action="store_true",
        help="[Windows cross-compiling] Target Windows ARM64EC.",
    )
    parser.add_argument(
        "--buildasx",
        action="store_true",
        help="[Windows cross-compiling] Create ARM64X Binary.",
    )

    parser.add_argument(
        "--disable_memleak_checker",
        action="store_true",
        help="Disable memory leak checker (enabled by default in Debug builds).",
    )
    parser.add_argument(
        "--enable_pix_capture", action="store_true", help="Enable Pix support for GPU debugging (requires D3D12)."
    )

    parser.add_argument(
        "--enable_wcos",
        action="store_true",
        help="Build for Windows Core OS. Link to Windows umbrella libraries instead of kernel32.lib.",
    )


def add_linux_specific_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments specific to Linux builds."""
    parser.add_argument(
        "--allow_running_as_root",
        action="store_true",
        help="Allow build script to run as root (disallowed by default).",
    )
    parser.add_argument(
        "--enable_external_custom_op_schemas",
        action="store_true",
        help="Enable loading custom op schemas from external shared libraries (Ubuntu only).",
    )


def add_dependency_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to external dependencies."""
    parser.add_argument("--use_full_protobuf", action="store_true", help="Use the full (non-lite) protobuf library.")
    parser.add_argument("--use_mimalloc", action="store_true", help="Use mimalloc memory allocator.")
    parser.add_argument(
        "--external_graph_transformer_path", type=str, help="Path to external graph transformer directory."
    )


def add_size_reduction_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for reducing the binary size."""
    parser.add_argument(
        "--minimal_build",
        default=None,
        nargs="*",
        type=str.lower,
        help="Create a minimal build supporting only ORT format models. "
        "Options: 'extended' (runtime kernel compilation), 'custom_ops'. "
        "e.g., '--minimal_build extended custom_ops'. RTTI disabled automatically.",
    )
    parser.add_argument(
        "--include_ops_by_config",
        type=str,
        help="Include only ops specified in the config file (see docs/Reduced_Operator_Kernel_build.md).",
    )
    parser.add_argument(
        "--enable_reduced_operator_type_support",
        action="store_true",
        help="Further reduce size by limiting operator data types based on --include_ops_by_config file.",
    )
    parser.add_argument("--disable_contrib_ops", action="store_true", help="Disable contrib operators.")
    parser.add_argument("--disable_ml_ops", action="store_true", help="Disable traditional ML operators.")
    parser.add_argument("--disable_rtti", action="store_true", help="Disable Run-Time Type Information (RTTI).")
    parser.add_argument(
        "--disable_types",
        nargs="+",
        default=[],
        choices=["float4", "float8", "optional", "sparsetensor"],
        help="Disable selected data types.",
    )
    parser.add_argument(
        "--disable_exceptions",
        action="store_true",
        help="Disable exceptions (requires --minimal_build).",
    )


def add_client_package_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for client package build package."""
    parser.add_argument(
        "--client_package_build",
        action="store_true",
        help="Create ORT package with default settings more appropriate for client/on-device workloads.",
    )


def add_python_binding_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for Python bindings."""
    parser.add_argument("--build_wheel", action="store_true", help="Build Python wheel package.")
    parser.add_argument(
        "--wheel_name_suffix",
        help="Suffix for wheel name (used for nightly builds).",
    )
    parser.add_argument("--skip-keras-test", action="store_true", help="Skip Keras-related tests.")


def add_execution_provider_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for enabling various Execution Providers (EPs)."""

    # --- QNN ---
    qnn_group = parser.add_argument_group("QNN Execution Provider (Qualcomm)")
    qnn_group.add_argument(
        "--use_qnn",
        nargs="?",
        const="shared_lib",  # Default linkage if only flag is present
        type=_qnn_verify_library_kind,
        help="Enable QNN EP. Optionally specify 'shared_lib' (default) or 'static_lib'.",
    )
    qnn_group.add_argument("--qnn_home", help="Path to QNN SDK directory.")
    qnn_group.add_argument("--ort_home", help="Path to ORT Prebuilt directory.")


def add_other_feature_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for other miscellaneous features."""
    parser.add_argument("--enable_lazy_tensor", action="store_true", help="Enable ORT backend for PyTorch LazyTensor.")
    parser.add_argument("--use_lock_free_queue", action="store_true", help="Use lock-free task queue for threadpool.")
    parser.add_argument(
        "--enable_generic_interface",
        action="store_true",
        help="Build ORT shared lib with compatible bridge for primary EPs (TRT, OV, QNN, VitisAI), excludes tests.",
    )


def is_cross_compiling(args: argparse.Namespace) -> bool:
    return any(
        [
            # Check existence before accessing for conditionally added args
            getattr(args, "x86", False),
            getattr(args, "arm", False),
            getattr(args, "arm64", False),
            getattr(args, "arm64ec", False),
            args.rv64,  # General cross-compile arg
            args.android,
            # Check existence for macOS/Apple specific args
            getattr(args, "ios", False),
            getattr(args, "visionos", False),
            getattr(args, "tvos", False),
        ]
    )


# --- Main Argument Parsing Function ---
def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for the ONNX Runtime build."""

    class Parser(argparse.ArgumentParser):
        # override argument file line parsing behavior - allow multiple arguments per line and handle quotes
        def convert_arg_line_to_args(self, arg_line: str) -> list[str]:  # Use list[str] for Python 3.9+
            return shlex.split(arg_line)

    parser = Parser(
        description="ONNXRuntime CI build driver.",
        usage="""
        Default behavior is --update --build --test for native architecture builds.
        Default behavior is --update --build for cross-compiled builds.

        The Update phase will update git submodules, and run cmake to generate makefiles.
        The Build phase will build all projects.
        The Test phase will run all unit tests, and optionally the ONNX tests.

        Use the individual flags (--update, --build, --test) to only run specific stages.
        """,
        fromfile_prefix_chars="@",  # Allow args from file (@filename)
    )

    # Add arguments by category
    add_core_build_args(parser)
    add_cmake_build_config_args(parser)
    add_testing_args(parser)
    add_general_profiling_args(parser)
    add_debugging_sanitizer_args(parser)
    add_documentation_args(parser)
    add_cross_compile_args(parser)  # Non-Windows cross-compile args
    add_android_args(parser)
    add_dependency_args(parser)
    add_size_reduction_args(parser)
    add_client_package_args(parser)

    # Language Bindings
    add_python_binding_args(parser)

    # Execution Providers (now includes EP-specific profiling args)
    add_execution_provider_args(parser)

    # Other Features
    add_other_feature_args(parser)

    # Platform specific args (now includes Windows cross-compile targets & specific config/debug args)
    if is_windows():
        add_windows_specific_args(parser)
    else:  # Assuming Linux or other non-Windows, non-macOS Unix-like
        add_linux_specific_args(parser)

    # --- Parse Arguments ---
    args: argparse.Namespace = parser.parse_args()

    # --- Post-processing and Defaults ---

    # Normalize paths
    if args.android_sdk_path:
        args.android_sdk_path = os.path.normpath(args.android_sdk_path)
    if args.android_ndk_path:
        args.android_ndk_path = os.path.normpath(args.android_ndk_path)

    # Set default CMake generator if not specified
    # Check if cmake_generator attribute exists (it might if --use_xcode was used)
    # before checking if it's None.
    if not hasattr(args, "cmake_generator") or args.cmake_generator is None:
        if is_windows():
            args.cmake_generator = "Visual Studio 17 2022"
        # else: Linux/macOS default (usually Makefiles or Ninja) is handled by CMake itself

    # Default behavior (update/build/test) if no action flags are specified
    # Determine if it's a cross-compiled build (approximated by checking common cross-compile flags)
    if not (args.update or args.build or args.test or args.clean or args.gen_doc):
        args.update = True
        args.build = True
        # Only default to running tests for native builds if tests aren't explicitly skipped
        if not is_cross_compiling(args) and not args.skip_tests:
            args.test = True
        elif is_cross_compiling(args):
            print(
                "Cross-compiling build detected: Defaulting to --update --build. Specify --test explicitly to run tests."
            )

    # Validation: Minimal build requires disabling exceptions
    if args.disable_exceptions and args.minimal_build is None:
        parser.error("--disable_exceptions requires --minimal_build to be specified.")
    if is_windows():
        if hasattr(args, "msvc_toolset") and args.msvc_toolset:
            try:
                # Extract major.minor version parts (e.g., "14.36")
                version_parts = args.msvc_toolset.split(".")
                if len(version_parts) >= 2:
                    major = int(version_parts[0])
                    minor = int(version_parts[1])
                    # Check known problematic range based on previous script comments/help text
                    # Refined check: >= 14.36 and <= 14.39
                    # Help text now says >= 14.40 is required, so check < 14.40
                    if major == 14 and minor < 40:
                        # You could make this an error or just a warning
                        # parser.error(f"MSVC toolset version {args.msvc_toolset} is not supported. Use 14.40 or higher.")
                        warnings.warn(
                            f"Specified MSVC toolset version {args.msvc_toolset} might have compatibility issues. Version 14.40 or higher is recommended."
                        )

            except (ValueError, IndexError):
                warnings.warn(
                    f"Could not parse MSVC toolset version: {args.msvc_toolset}. Skipping compatibility check."
                )

    elif is_macOS():
        if getattr(args, "build_apple_framework", False) and not any(
            [
                getattr(args, "ios", False),
                getattr(args, "macos", None),
                getattr(args, "visionos", False),
                getattr(args, "tvos", False),
            ]
        ):
            parser.error("--build_apple_framework requires --ios, --macos, --visionos, or --tvos to be specified.")

        if getattr(args, "macos", None) and not getattr(args, "build_apple_framework", False):
            parser.error("--macos target requires --build_apple_framework.")
    return args
