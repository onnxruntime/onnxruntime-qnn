#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Run onnxruntime_provider_test under AddressSanitizer via asan_filter_leaks.sh.
#
# Owns the ASan test-time concerns (LD_LIBRARY_PATH, ASAN_OPTIONS,
# LSAN_OPTIONS, suppressions file path) so that build.sh / run_tests.sh
# stay ASan-unaware. ONNX model tests are intentionally not invoked here:
# onnxruntime_plugin_ep_onnx_test dlopens an ASan-instrumented .so without
# ASan in the host process's initial library list, which makes ASan refuse
# to initialize.
#
# Prerequisites:
#   - A build compiled with --enable-asan (i.e. -fsanitize=address).
#     Use: python qcom/build_and_test.py asan_linux_x86_64
#
# Usage:
#   bash run_asan.sh \
#       --build-dir=/path/to/build/linux-x86_64 \
#       [--config=Debug]
#
# Note: --config currently accepts Debug only. Upstream tools/ci_build/build.py
# wires `-fsanitize=address` into the Linux build only on the Debug branch, so
# RelWithDebInfo + --enable_address_sanitizer would silently produce a
# non-instrumented binary. The script enforces this with a guard below.

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"
source "${REPO_ROOT}/qcom/scripts/linux/tools.sh"

set_strict_mode

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
build_dir=""
config="Debug"

for arg in "$@"; do
    case "${arg}" in
        --build-dir=*)
            build_dir="${arg#--build-dir=}"
            ;;
        --config=*)
            config="${arg#--config=}"
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --build-dir=<path> [--config=<cfg>]

  --build-dir=<path>    Required. Build root (e.g. build/linux-x86_64).
  --config=<cfg>        Optional. Build configuration subdirectory. Default: Debug.
                        Currently Debug is the only supported value; see the
                        header note for why RelWithDebInfo is not yet available.
EOF
            exit 0
            ;;
        *)
            die "Unknown argument: ${arg}"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate arguments
# ---------------------------------------------------------------------------
if [ -z "${build_dir}" ]; then
    die "--build-dir is required. Run with --help for usage."
fi

if [ "${config}" != "Debug" ]; then
    die "ASan currently only supports --config=Debug (got: ${config})"
fi

build_dir="$(realpath "${build_dir}")"

if [ ! -d "${build_dir}/${config}" ]; then
    die "Build directory not found: ${build_dir}/${config}"
fi

cd "${build_dir}/${config}"

log_info "=== QNN EP ASan Test Runner ==="
log_info "build_dir : ${build_dir}"
log_info "config    : ${config}"

# ---------------------------------------------------------------------------
# Set environment for the ASan run.
#
# The QNN backend libraries (libQnnCpu.so, libQnnHtp.so, ...) sit alongside
# onnxruntime_provider_test in the build directory. ctest would normally
# inject this via per-test ENVIRONMENT properties, but we run the binary
# directly here to feed it through asan_filter_leaks.sh.
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH="${PWD}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ASAN_OPTIONS="halt_on_error=1:detect_leaks=1:print_stats=0"
export LSAN_OPTIONS="exitcode=1:suppressions=${REPO_ROOT}/tools/ci_build/lsan_suppressions.txt:print_suppressions=0"

# ---------------------------------------------------------------------------
# Run the test binary through asan_filter_leaks.sh
# ---------------------------------------------------------------------------
log_info "--- Running onnxruntime_provider_test under ASan ---"
"${REPO_ROOT}/qcom/scripts/linux/asan_filter_leaks.sh" \
    "./onnxruntime_provider_test"
