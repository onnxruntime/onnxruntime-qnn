#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Generate ONNX Runtime QNN EP code coverage report.
#
# Prerequisites:
#   - A build compiled with coverage instrumentation (--enable-coverage flag in build.sh).
#     Use: python qcom/build_and_test.py coverage_linux_x86_64
#   - lcov 1.x (auto-installed via packages.yml)
#   - genhtml (bundled with lcov)
#   - Perl (must be present on the host; raise an error if missing)
#
# Usage:
#   bash generate_coverage.sh \
#       --build-dir=/path/to/build/linux-x86_64 \
#       [--config=Debug|RelWithDebInfo|Release] \
#       [--output-dir=/path/to/report] \
#       [--test-filter="*Qnn*"]

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"
source "${REPO_ROOT}/qcom/scripts/linux/tools.sh"

set_strict_mode

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
build_dir=""
config="RelWithDebInfo"
output_dir=""
test_filter="*Qnn*"

for arg in "$@"; do
    case "${arg}" in
        --build-dir=*)
            build_dir="${arg#--build-dir=}"
            ;;
        --config=*)
            config="${arg#--config=}"
            ;;
        --output-dir=*)
            output_dir="${arg#--output-dir=}"
            ;;
        --test-filter=*)
            test_filter="${arg#--test-filter=}"
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --build-dir=<path> [--config=<cfg>] [--output-dir=<path>] [--test-filter=<str>]

  --build-dir=<path>    Required. Build root (e.g. build/linux-x86_64).
  --config=<cfg>        Optional. Build configuration subdirectory.  Default: RelWithDebInfo
  --output-dir=<path>   Optional. Output directory for HTML report.  Default: <build-dir>/<config>/coverage
  --test-filter=<str>   Optional. GTest filter string.               Default: *Qnn*
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

# Resolve to absolute path so that cd inside subshells does not break relative paths.
build_dir="$(realpath "${build_dir}")"

if [ ! -d "${build_dir}/${config}" ]; then
    die "Build directory not found: ${build_dir}/${config}"
fi

if [ -z "${output_dir}" ]; then
    output_dir="${build_dir}/${config}/coverage"
fi

log_info "=== QNN EP Coverage Report Generator ==="
log_info "build_dir   : ${build_dir}"
log_info "config      : ${config}"
log_info "output_dir  : ${output_dir}"
log_info "test_filter : ${test_filter}"

# ---------------------------------------------------------------------------
# Locate Perl (required by lcov)
# ---------------------------------------------------------------------------
log_info "--- Checking Perl ---"
if ! command -v perl &>/dev/null; then
    die "Perl not found in PATH. lcov requires Perl. Please install perl (e.g. apt install perl)."
fi
log_info "Using Perl: $(command -v perl)  ($(perl --version 2>&1 | head -1 || true))"

# ---------------------------------------------------------------------------
# Locate lcov / genhtml (auto-installed via packages.yml)
# ---------------------------------------------------------------------------
log_info "--- Locating lcov ---"
lcov_bindir="$(get_lcov_bindir)"
export PATH="${lcov_bindir}:${PATH}"

if ! command -v lcov &>/dev/null; then
    die "lcov not found after package install. Check packages.yml entry for lcov_$(get_host_platform)."
fi
if ! command -v genhtml &>/dev/null; then
    die "genhtml not found. It should be bundled with lcov in ${lcov_bindir}."
fi

lcov_version=$(lcov --version 2>&1 | head -1 || true)
log_info "Using lcov   : $(command -v lcov)  (${lcov_version})"
log_info "Using genhtml: $(command -v genhtml)"

# Verify .gcno files exist (compile-time notes from gcov instrumentation)
gcno_count=$(find "${build_dir}" -name '*.gcno' | wc -l)
if [ "${gcno_count}" -eq 0 ]; then
    die "No .gcno files found under ${build_dir}. Was the build compiled with --enable-coverage?"
fi
log_info "Found ${gcno_count} .gcno file(s)."

# ---------------------------------------------------------------------------
# Clear stale counters
# ---------------------------------------------------------------------------
log_info "--- Clearing stale coverage counters ---"
lcov --zerocounters --directory "${build_dir}" 2>&1 || true
rm -f "${build_dir}/${config}/coverage_lcov.info" \
      "${build_dir}/${config}/coverage_lcov_filtered.info"

# ---------------------------------------------------------------------------
# Run tests to generate .gcda runtime data
# ---------------------------------------------------------------------------
log_info "--- Running tests (filter: ${test_filter}) ---"
test_exit=0
(
    cd "${build_dir}/${config}"
    export LD_LIBRARY_PATH="${build_dir}/${config}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    ./onnxruntime_provider_test --gtest_filter="${test_filter}"
) || test_exit=$?

if [ "${test_exit}" -ne 0 ]; then
    log_warn "Tests exited with ${test_exit}; continuing to collect coverage from .gcda written so far."
fi

gcda_count=$(find "${build_dir}" -name '*.gcda' | wc -l)
if [ "${gcda_count}" -eq 0 ]; then
    log_warn "No .gcda files found — coverage data may be sparse."
else
    log_info "Found ${gcda_count} .gcda file(s)."
fi

# ---------------------------------------------------------------------------
# Collect coverage data
# ---------------------------------------------------------------------------
log_info "--- Collecting coverage data ---"
(
    cd "${build_dir}/${config}"
    lcov --capture \
         --directory "${build_dir}" \
         --output-file coverage_lcov.info \
         --rc lcov_branch_coverage=1
)

# ---------------------------------------------------------------------------
# Filter: allowlist QNN EP sources only; strip third-party, test, and deps
# ---------------------------------------------------------------------------
log_info "--- Filtering coverage data ---"
(
    cd "${build_dir}/${config}"
    # Step 1: extract only QNN EP production sources
    lcov --extract coverage_lcov.info \
         "*/onnxruntime/core/providers/qnn/*" \
         --output-file coverage_lcov_filtered.info \
         --rc lcov_branch_coverage=1

    # Step 2: remove anything that slipped through (tests, deps, system headers)
    lcov --remove coverage_lcov_filtered.info \
         '/usr/*' \
         '*/googletest/*' \
         '*/test/*' \
         '*/_deps/*' \
         --output-file coverage_lcov_filtered.info \
         --rc lcov_branch_coverage=1
)

# ---------------------------------------------------------------------------
# Generate HTML report
# ---------------------------------------------------------------------------
log_info "--- Generating HTML report ---"
mkdir -p "${output_dir}"
genhtml "${build_dir}/${config}/coverage_lcov_filtered.info" \
        --output-directory "${output_dir}" \
        --branch-coverage \
        --rc lcov_branch_coverage=1

# ---------------------------------------------------------------------------
# Copy .info files to output_dir
# ---------------------------------------------------------------------------
cp "${build_dir}/${config}/coverage_lcov.info"          "${output_dir}/coverage_lcov.info"
cp "${build_dir}/${config}/coverage_lcov_filtered.info" "${output_dir}/coverage_lcov_filtered.info"

# ---------------------------------------------------------------------------
# Convert filtered .info to Cobertura XML for diff-cover
# ---------------------------------------------------------------------------
log_info "--- Converting to Cobertura XML ---"
if ! command -v lcov_cobertura &>/dev/null; then
    die "lcov_cobertura not found. Activate the project venv or: pip install lcov_cobertura"
fi
lcov_cobertura "${output_dir}/coverage_lcov_filtered.info" \
    --base-dir "${REPO_ROOT}" \
    --output "${output_dir}/coverage.xml"

log_info "=== Coverage report complete ==="
log_info "HTML report  : ${output_dir}/index.html"
log_info "lcov raw     : ${output_dir}/coverage_lcov.info"
log_info "lcov filtered: ${output_dir}/coverage_lcov_filtered.info"
log_info "Cobertura XML: ${output_dir}/coverage.xml"

# ---------------------------------------------------------------------------
# Copy README.md for CI artifact consumers
# ---------------------------------------------------------------------------
cp "${REPO_ROOT}/qcom/scripts/linux/coverage_artifact_README.md" \
   "${output_dir}/README.md"
log_info "README       : ${output_dir}/README.md"

# ---------------------------------------------------------------------------
# Propagate test failure after coverage report has been generated
# ---------------------------------------------------------------------------
if [ "${test_exit}" -ne 0 ]; then
    die "Tests failed with exit code ${test_exit}. Coverage report was still generated at ${output_dir}."
fi
