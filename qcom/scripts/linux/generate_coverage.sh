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
test_filter=""
skip_snapshot=false
skip_accuracy=false

# Three-phase split for COVERAGE. All three phases run the same instrumented
# binary back-to-back; their .gcda counters accumulate, so the single lcov
# --capture at the end sees the union. The phases are ordered by a DATA
# dependency, not preference: component -> snapshot -> accuracy.
#   - component phase (GATING): element/component-level UT + old integration
#     tests + any other Qnn suite. Defined by EXCLUSION so new suites land here
#     automatically. Non-zero exit fails this script.
#   - snapshot phase (NON-gating): re-runs the migrated ops through the builder and
#     compares the emitted graph against goldens (the QnnUnit_*_Snapshot* /
#     QnnUnit_*_SessionSnapshot* suites). It re-exercises the full builder path so
#     it contributes builder coverage. A golden byte-mismatch (graph-structure
#     drift) logs a warning but does NOT fail this script: structure drift is a
#     routing signal for the accuracy tier, not a build failure. Writes a gtest
#     JSON report. No gate consumes it today; a future accuracy-routing gate will
#     read it per-case to decide which accuracy tests to route. It MUST run
#     before accuracy.
#   - accuracy phase (GATING): QnnUnit_*_Accuracy* — the numerical-correctness
#     gate. Non-zero exit fails this script. Today this runs unconditionally
#     (safe baseline: no golden store yet, so every case runs). Once a golden
#     store with version metadata exists, a future accuracy-routing gate (not
#     built yet) replaces this filter with a run-set derived from the snapshot
#     JSON above (skip cases whose snapshot passed at a matching golden version;
#     run the rest).
#
# Note on coverage attribution: accuracy runs the same session-compile builder
# path as the snapshot phase, so it adds ~0 builder coverage (measured on
# clip_op_builder.cc: component+snapshot 94.2% == with-accuracy 94.2%). Its .gcda
# is still captured — that is harmless because snapshot already covers those
# lines. "Accuracy is not a coverage patch" is a migration-completeness criterion
# (don't close coverage gaps with accuracy), not a data-exclusion rule.
#
# gtest filter grammar: a single '-' separates the positive section from the
# negative section; ':'-joined patterns after that '-' are ALL negative (do NOT
# prefix each with its own '-', or they become literal, never-matching patterns).
component_filter="*Qnn*:-QnnUnit_*_Snapshot*:QnnUnit_*_SessionSnapshot*:QnnUnit_*_Accuracy*"
snapshot_filter="QnnUnit_*_Snapshot*:QnnUnit_*_SessionSnapshot*"
# Safe baseline: run every accuracy test. Once the golden-version gate exists it
# replaces this constant with a run-set computed from the snapshot JSON report.
accuracy_filter="QnnUnit_*_Accuracy*"

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
        --skip-snapshot)
            skip_snapshot=true
            ;;
        --skip-accuracy)
            skip_accuracy=true
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --build-dir=<path> [--config=<cfg>] [--output-dir=<path>] [--test-filter=<str>] [--skip-snapshot] [--skip-accuracy]

  --build-dir=<path>    Required. Build root (e.g. build/linux-x86_64).
  --config=<cfg>        Optional. Build configuration subdirectory.  Default: RelWithDebInfo
  --output-dir=<path>   Optional. Output directory for HTML report.  Default: <build-dir>/<config>/coverage
  --test-filter=<str>   Optional. Override the three-phase split with a single GTest
                        filter run (legacy behavior). When set, --skip-snapshot and
                        --skip-accuracy are ignored.
  --skip-snapshot       Optional. Skip the snapshot phase (re-run builder +
                        golden compare).
  --skip-accuracy       Optional. Skip the numerical accuracy phase.

Default (no --test-filter): tests run in three separately-tracked phases whose
.gcda counters accumulate into a single coverage capture —
  component: ${component_filter}
  snapshot : ${snapshot_filter}
  accuracy : ${accuracy_filter}
The phases are ordered by a data dependency (component -> snapshot -> accuracy):
a future accuracy-routing gate will read the snapshot JSON to route accuracy,
so snapshot must precede it. Coverage is captured once after all phases. The component and
accuracy phases GATE (non-zero exit on failure); the snapshot phase is NON-gating
(a golden mismatch only logs a warning — drift is a routing signal, not a build
failure).
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
if [ -n "${test_filter}" ]; then
    log_info "mode        : single-phase (--test-filter override)"
    log_info "test_filter : ${test_filter}"
else
    log_info "mode        : three-phase (component + snapshot + accuracy)"
    log_info "component   : ${component_filter}"
    if [ "${skip_snapshot}" = true ]; then
        log_info "snapshot    : SKIPPED (--skip-snapshot)"
    else
        log_info "snapshot    : ${snapshot_filter}"
    fi
    if [ "${skip_accuracy}" = true ]; then
        log_info "accuracy    : SKIPPED (--skip-accuracy)"
    else
        log_info "accuracy    : ${accuracy_filter}"
    fi
fi

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
#
# .gcda counters accumulate across every invocation of the instrumented binary,
# so running the phases back-to-back yields combined coverage — the single
# lcov --capture below sees all of them. Each phase's exit code is tracked
# separately so we can report which phase failed while still emitting one merged
# report. An optional third arg to run_test_phase requests a gtest JSON report
# (the snapshot phase writes one so a future accuracy-routing gate can route accuracy per-case).
# ---------------------------------------------------------------------------
run_test_phase() {
    local phase_name="$1"
    local filter="$2"
    local json_out="${3:-}"
    log_info "--- Running ${phase_name} tests (filter: ${filter}) ---"
    local rc=0
    (
        cd "${build_dir}/${config}"
        export LD_LIBRARY_PATH="${build_dir}/${config}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        if [ -n "${json_out}" ]; then
            ./onnxruntime_provider_test --gtest_filter="${filter}" --gtest_output="json:${json_out}"
        else
            ./onnxruntime_provider_test --gtest_filter="${filter}"
        fi
    ) || rc=$?
    if [ "${rc}" -ne 0 ]; then
        log_warn "${phase_name} tests exited with ${rc}; continuing to collect coverage from .gcda written so far."
    fi
    return "${rc}"
}

# Snapshot-phase JSON report path. Written today but not yet consumed by anything;
# reserved for a future accuracy-routing gate.
# Holds the QnnUnit_*_Snapshot* / QnnUnit_*_SessionSnapshot* per-case results.
snapshot_json="${build_dir}/${config}/snapshot_results.json"

comp_exit=0
snapshot_exit=0
accuracy_exit=0

if [ -n "${test_filter}" ]; then
    # Legacy single-phase override.
    run_test_phase "filtered" "${test_filter}" || comp_exit=$?
else
    run_test_phase "component" "${component_filter}" || comp_exit=$?
    if [ "${skip_snapshot}" = true ]; then
        log_info "--- Skipping snapshot phase (--skip-snapshot) ---"
    else
        # Snapshot MUST run before accuracy: a future accuracy-routing gate will read
        # this JSON to decide which accuracy cases to route.
        run_test_phase "snapshot" "${snapshot_filter}" "${snapshot_json}" || snapshot_exit=$?
    fi
    if [ "${skip_accuracy}" = true ]; then
        log_info "--- Skipping accuracy phase (--skip-accuracy) ---"
    else
        run_test_phase "accuracy" "${accuracy_filter}" || accuracy_exit=$?
    fi
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
# Propagate test failure after coverage report has been generated.
#
# The component and accuracy phases GATE (non-zero exit fails this script). The
# snapshot phase is NON-gating: a golden byte-mismatch means the graph
# structure drifted, which is allowed to land (goldens may go stale on main; a
# nightly job reconciles them). Drift is a routing signal for accuracy, and
# numerical correctness is enforced by the accuracy phase above.
# ---------------------------------------------------------------------------
if [ "${snapshot_exit}" -ne 0 ]; then
    log_warn "snapshot phase exited ${snapshot_exit} — graph-structure drift detected."
    log_warn "This is NON-gating. Run run_snapshot_accuracy.sh to verify numerical correctness,"
    log_warn "and --update-goldens once the new structure is accepted."
fi

if [ "${comp_exit}" -ne 0 ] && [ "${accuracy_exit}" -ne 0 ]; then
    die "Component (exit ${comp_exit}) and accuracy (exit ${accuracy_exit}) phases failed. Coverage report was still generated at ${output_dir}."
fi
if [ "${comp_exit}" -ne 0 ]; then
    die "Component test phase failed (exit ${comp_exit}). Coverage report was still generated at ${output_dir}."
fi
if [ "${accuracy_exit}" -ne 0 ]; then
    die "Accuracy test phase failed (exit ${accuracy_exit}) — numerical regression. Coverage report was still generated at ${output_dir}."
fi
