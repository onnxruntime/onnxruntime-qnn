#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Platform-agnostic two-pass snapshot+accuracy runner for QNN EP unit tests.
#
# Pass 1: Run all snapshot tests (QnnSnapshot_<Op>_OpBuilder* + QnnSnapshot_<Op>_Session*).
#          Trusted matching cases can skip their paired accuracy rerun.
# Pass 2: Start from full scoped Accuracy; skip only exact paired trustworthy
#          Snapshot matches with compatible QAIRT/ORT golden metadata.
#
# Suite naming is tier-first with an explicit delimiter token:
#   QnnSnapshot_<Op>_OpBuilder[_<Variant>]Test
#   QnnSnapshot_<Op>_Session[_<Variant>]Test
#   QnnAcc_<Op>_Accuracy[_<Variant>]Test
# The op is recovered as the segment(s) between the prefix and the delimiter
# token, so op names may themselves contain underscores (e.g. Gelu_Fusion)
# without ambiguity.
#
# Exit codes:
#   0  — All good (snapshots pass; OR drift detected + accuracy pass)
#   1  — Accuracy regression (drift detected + accuracy tests FAIL)
#   99 — Script usage / setup error
#
# Usage:
#   bash run_snapshot_accuracy.sh \
#       --build-dir=/path/to/build/linux-x86_64 \
#       [--generate-goldens] \
#       [--force-accuracy] \
#       [--filter=Clip,Conv]

REPO_ROOT=$(git rev-parse --show-toplevel)

set -euo pipefail
source "${REPO_ROOT}/qcom/scripts/all/qnn_test_stages.sh"
if [ "${BASH_VERSINFO:-0}" -gt 3 ]; then
    shopt -s inherit_errexit
fi
if [ -n "${ORT_BUILD_XTRACE:-}" ]; then
    set -x
fi

log_info() {
    printf "%s\n" "$*" >&2
}

log_warn() {
    printf "WARNING: %s\n" "$*" >&2
}

log_err() {
    printf "ERROR: %s\n" "$*" >&2
}

die() {
    log_err "$*"
    exit 1
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
build_dir=""
filter_groups=""
force_accuracy=false
generate_goldens=false
routing_result=""
python_bin="${PYTHON_BIN:-python3}"

for arg in "$@"; do
    case "${arg}" in
        --build-dir=*)
            build_dir="${arg#--build-dir=}"
            ;;
        --filter=*)
            filter_groups="${arg#--filter=}"
            ;;
        --emit-routing-result=*)
            routing_result="${arg#--emit-routing-result=}"
            ;;
        --force-accuracy)
            force_accuracy=true
            ;;
        --generate-goldens)
            generate_goldens=true
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") --build-dir=<path> [options]

Two-pass snapshot+accuracy test runner. Runs snapshot tests first; if any
snapshot evidence is missing or untrusted, retains those exact accuracy cases.
Only exact paired matches with compatible versions may skip accuracy.

Options:
  --build-dir=<path>        Required. Build root (e.g. build/linux-x86_64).
  --generate-goldens          Generate golden files from current output, then run accuracy
                            tests to verify the new graph structure is numerically correct.
  --force-accuracy          Always run accuracy tests regardless of snapshot outcome.
  --emit-routing-result=<path>
                            Export observed routing selection as JSON.
  --filter=<group1,group2,...>
                            Scope both passes to these test groups only.
                            Group name = op segment, i.e. QnnSnapshot_<Group>_OpBuilder...Test.
                            Examples: Clip, Conv, GeluFusion (case-sensitive).
EOF
            exit 0
            ;;
        *)
            die "Unknown argument: ${arg}"
            ;;
    esac
done
PYTHON_BIN="${python_bin}"

if [ -n "${routing_result}" ]; then
    if prepared_routing_result="$(realpath -m "${routing_result}")" &&
       rm -f -- "${prepared_routing_result}" "${prepared_routing_result}.tmp"; then
        routing_result="${prepared_routing_result}"
    else
        log_warn "Routing artifact preparation failed; continuing tests without artifact publication."
        routing_result=""
    fi
fi

# ---------------------------------------------------------------------------
# Validate & auto-detect build config
# ---------------------------------------------------------------------------
if [ -z "${build_dir}" ]; then
    die "--build-dir is required. Run with --help for usage."
fi

build_dir="$(realpath "${build_dir}")"

# Auto-detect config subdir by searching for the test binary. Git Bash/MSYS may
# resolve an extensionless executable probe to the .exe file, so prefer the
# explicit Windows filename to keep the selected path unambiguous.
binary_names=(onnxruntime_provider_test.exe onnxruntime_provider_test)
binary=""
for cfg in RelWithDebInfo Release Debug; do
    for name in "${binary_names[@]}"; do
        candidate="${build_dir}/${cfg}/${name}"
        if [ -x "${candidate}" ]; then
            binary="${candidate}"
            config="${cfg}"
            break 2
        fi
    done
done

# Also check if binary is directly in build_dir (user pointed to config dir).
if [ -z "${binary}" ]; then
    for name in "${binary_names[@]}"; do
        if [ -x "${build_dir}/${name}" ]; then
            binary="${build_dir}/${name}"
            config=""
            break
        fi
    done
fi

if [ -z "${binary}" ]; then
    die "onnxruntime_provider_test not found under ${build_dir}. Build a QNN EP internal-symbol test binary first."
fi

bin_dir="$(dirname "${binary}")"

run_provider_test() {
    qnn_run_provider "${binary}" "${bin_dir}" "$@"
}

to_provider_test_path() {
    local path="$1"
    if command -v cygpath &>/dev/null; then
        cygpath -w "${path}"
    else
        printf "%s\n" "${path}"
    fi
}

# Verify that the selected binary contains the snapshot test tier.
snapshot_probe=$(run_provider_test --gtest_list_tests --gtest_filter="QnnSnapshot_*" 2>/dev/null || true)
if [ -z "${snapshot_probe}" ]; then
    die "No QnnSnapshot_* tests found in binary. This requires a QNN EP internal-symbol test build."
fi

log_info "=== QNN EP Two-Pass Snapshot+Accuracy Runner ==="
log_info "binary : ${binary}"
if [ "${generate_goldens}" = true ]; then
    log_info "mode   : generate-goldens"
fi
if [ "${force_accuracy}" = true ]; then
    log_info "mode   : force-accuracy"
fi
if [ -n "${filter_groups}" ]; then
    log_info "filter : ${filter_groups}"
fi

# ---------------------------------------------------------------------------
# Build snapshot filter
# ---------------------------------------------------------------------------
if [ -n "${filter_groups}" ]; then
    # Scope Pass 1 to specified groups only.
    IFS=',' read -ra groups <<< "${filter_groups}"
    snapshot_filter=""
    for g in "${groups[@]}"; do
        if [ -n "${snapshot_filter}" ]; then
            snapshot_filter+=":"
        fi
        snapshot_filter+="QnnSnapshot_${g}_OpBuilder*Test.*:QnnSnapshot_${g}_Session*Test.*"
    done
else
    snapshot_filter="${QNN_SNAPSHOT_SCOPE}"
fi

# ---------------------------------------------------------------------------
# Setup environment
# ---------------------------------------------------------------------------
results_dir="${bin_dir}/snapshot_accuracy_results"
mkdir -p "${results_dir}"

snapshot_json="${results_dir}/snapshot_results.json"
accuracy_json="${results_dir}/accuracy_results.json"
snapshot_list="${results_dir}/snapshot_tests.txt"
rm -f "${snapshot_json}" "${accuracy_json}" "${snapshot_list}"

selection_complete=false
accuracy_list="${results_dir}/routing_accuracy_tests.txt"
accuracy_filter_file="${results_dir}/accuracy_filter.txt"
rm -f "${accuracy_list}" "${accuracy_filter_file}" "${results_dir}/selection.json" \
      "${results_dir}/accuracy_gate_summary.txt"
snapshot_exit=0

export_routing_result() {
    local runner_exit="$1"
    local mode="compare"
    if [ "${generate_goldens}" = true ]; then
        mode="generate_goldens"
    elif [ "${force_accuracy}" = true ]; then
        mode="force_accuracy"
    fi
    local destination="${2:-${routing_result:-${results_dir}/selection.json}}"
    [ "${QNN_ROUTING_UPSTREAM_EXIT:-0}" = 0 ] || runner_exit=1
    qnn_export_verdict "${bin_dir}" "${snapshot_json}" "${snapshot_list}" "${accuracy_list}" \
        "${destination}" "${snapshot_exit}" "${mode}" "${runner_exit}" "${selection_complete}" || return 1
    if [ "${destination}" = "${routing_result}" ] && [ -n "${routing_result}" ] && [ -n "${GITHUB_OUTPUT:-}" ]; then
        printf 'routing_result_path=%s\n' "${routing_result}" >> "${GITHUB_OUTPUT}" || return 1
    fi
}

if [ -n "${routing_result}" ]; then
    # Export after selection, including early success and failure exits. Export
    # failures leave the original test exit code and selection intact.
    trap 'runner_exit=$?; export_routing_result "${runner_exit}" || log_warn "Routing artifact export failed; no result is available for consumption."' EXIT
fi

# Native Windows binaries do not understand Git Bash paths such as /c/work/....
snapshot_json_arg="$(to_provider_test_path "${snapshot_json}")"
accuracy_json_arg="$(to_provider_test_path "${accuracy_json}")"

# ---------------------------------------------------------------------------
# Pass 1: Snapshot tests
# ---------------------------------------------------------------------------
log_info "--- Pass 1: Running snapshot tests ---"

if [ "${generate_goldens}" = true ]; then
    export QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1
    log_info "QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 (writing new golden files)"
fi

snapshot_exit=0
run_provider_test \
    --gtest_filter="${snapshot_filter}" \
    --gtest_output="json:${snapshot_json_arg}" || snapshot_exit=$?

# In update mode, unset so Pass 2 accuracy tests run normally (compare, don't write).
if [ "${generate_goldens}" = true ]; then
    unset QNN_UT_SNAPSHOT_GOLDEN_UPDATE
fi

# ---------------------------------------------------------------------------
# Analyze Pass 1 results
# ---------------------------------------------------------------------------

# Start from the entire registered scoped Accuracy universe, including cases
# with no Snapshot pair. Selection and exported actions share one implementation.
accuracy_scope="${QNN_ACCURACY_SCOPE}"
if [ -n "${filter_groups}" ]; then
    accuracy_scope=""
    for g in "${groups[@]}"; do
        accuracy_scope+="${accuracy_scope:+:}QnnAcc_${g}_Accuracy*Test.*"
    done
fi
inventory_complete=false
if run_provider_test --gtest_list_tests --gtest_filter="${snapshot_filter}" > "${snapshot_list}" &&
   run_provider_test --gtest_list_tests --gtest_filter="${accuracy_scope}" > "${accuracy_list}"; then
    inventory_complete=true
fi
selection_complete="${inventory_complete}"
selection_mode="compare"
if [ "${generate_goldens}" = true ]; then
    selection_mode="generate_goldens"
elif [ "${force_accuracy}" = true ]; then
    selection_mode="force_accuracy"
fi
if [ "${inventory_complete}" = true ] && qnn_export_and_select \
    "${bin_dir}" "${snapshot_json}" "${snapshot_list}" "${accuracy_list}" \
    "${results_dir}/selection.json" "${accuracy_filter_file}" "${snapshot_exit}" "${selection_mode}" \
    "${results_dir}/accuracy_gate_summary.txt"; then
    accuracy_filter="$(cat "${accuracy_filter_file}")"
else
    log_warn "Testcase selection failed; running the full scoped Accuracy universe."
    accuracy_filter="${accuracy_scope}"
    selection_complete=false
fi

# Drift and absent goldens are routing signals. Preserve genuine Snapshot
# failures even when Accuracy passes; an incomplete report cannot clear failure.
snapshot_verdict=0
qnn_snapshot_failure_is_routing_only "${snapshot_json}" "${snapshot_exit}" || snapshot_verdict=${snapshot_exit}

log_info "--- Pass 2: Running accuracy tests ---"
log_info "Filter: ${accuracy_filter}"

accuracy_exit=0
if [ "${accuracy_filter}" = "QnnAcc_ZZZ_NoCasesSelected.None" ]; then
    log_info "All scoped Accuracy cases have trusted matching Snapshot evidence; skipping Accuracy execution."
else
    run_provider_test \
        --gtest_filter="${accuracy_filter}" \
        --gtest_output="json:${accuracy_json_arg}" || accuracy_exit=$?
fi

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
if [ ${accuracy_exit} -eq 0 ]; then
    if [ "${generate_goldens}" = true ]; then
        log_info "=== PASS: Goldens updated and accuracy verified ==="
    else
        log_info "=== PASS: Snapshot drift/missing goldens verified numerically correct ==="
        log_info "Action: Run with --generate-goldens to accept the current graph structure."
    fi
    exit "${snapshot_verdict}"
else
    log_err "=== FAIL: Accuracy regression detected ==="
    log_err "Selected Accuracy testcases FAILED."
    log_err "Results: ${accuracy_json}"
    exit 1
fi
