#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Two-pass snapshot+accuracy test runner for QNN EP unit tests.
#
# Pass 1: Run all snapshot tests (QnnUnit_<Op>_Snapshot* + QnnUnit_<Op>_SessionSnapshot*).
#          If all pass -> done (exit 0). Graph structure unchanged -> accuracy is redundant.
# Pass 2: For any ops whose snapshot tests failed (golden mismatch), run their
#          QnnUnit_<Op>_Accuracy* tests to verify numerical correctness.
#
# Suite naming is op-first: QnnUnit_<Op>_<Tier>[_<Variant>]Test, where <Tier> is
# one of Component/Snapshot/SessionSnapshot/Accuracy. The op is recovered as the
# segment(s) between "QnnUnit_" and the first tier token, so op names may
# themselves contain underscores (e.g. Gelu_Fusion) without ambiguity.
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

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"

set_strict_mode

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
build_dir=""
filter_groups=""
force_accuracy=false
generate_goldens=false

for arg in "$@"; do
    case "${arg}" in
        --build-dir=*)
            build_dir="${arg#--build-dir=}"
            ;;
        --filter=*)
            filter_groups="${arg#--filter=}"
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
fail (golden mismatch), runs accuracy tests for the affected ops only.

Options:
  --build-dir=<path>        Required. Build root (e.g. build/linux-x86_64).
  --generate-goldens          Generate golden files from current output, then run accuracy
                            tests to verify the new graph structure is numerically correct.
  --force-accuracy          Always run accuracy tests regardless of snapshot outcome.
  --filter=<group1,group2,...>
                            Scope both passes to these test groups only.
                            Group name = op segment, i.e. QnnUnit_<Group>_Snapshot...Test.
                            Examples: Clip, Conv, GeluFusion (case-sensitive).
EOF
            exit 0
            ;;
        *)
            die "Unknown argument: ${arg}"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate & auto-detect build config
# ---------------------------------------------------------------------------
if [ -z "${build_dir}" ]; then
    die "--build-dir is required. Run with --help for usage."
fi

build_dir="$(realpath "${build_dir}")"

# Auto-detect config subdir by searching for the test binary.
binary=""
for cfg in RelWithDebInfo Release Debug; do
    candidate="${build_dir}/${cfg}/onnxruntime_provider_test"
    if [ -x "${candidate}" ]; then
        binary="${candidate}"
        config="${cfg}"
        break
    fi
done

# Also check if binary is directly in build_dir (user pointed to config dir).
if [ -z "${binary}" ] && [ -x "${build_dir}/onnxruntime_provider_test" ]; then
    binary="${build_dir}/onnxruntime_provider_test"
    config=""
fi

if [ -z "${binary}" ]; then
    die "onnxruntime_provider_test not found under ${build_dir}. Is this a coverage build?"
fi

bin_dir="$(dirname "${binary}")"

# Verify this is a coverage build: probe for snapshot tests.
snapshot_probe=$("${binary}" --gtest_list_tests --gtest_filter="QnnUnit_*_Snapshot*" 2>/dev/null || true)
if [ -z "${snapshot_probe}" ]; then
    die "No QnnUnit_*_Snapshot* tests found in binary. This is not a coverage build (requires --enable-coverage)."
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
        snapshot_filter+="QnnUnit_${g}_Snapshot*Test.*:QnnUnit_${g}_SessionSnapshot*Test.*"
    done
else
    snapshot_filter="QnnUnit_*_Snapshot*Test.*:QnnUnit_*_SessionSnapshot*Test.*"
fi

# ---------------------------------------------------------------------------
# Setup environment
# ---------------------------------------------------------------------------
results_dir="${bin_dir}/snapshot_accuracy_results"
mkdir -p "${results_dir}"

snapshot_json="${results_dir}/snapshot_results.json"
accuracy_json="${results_dir}/accuracy_results.json"

# ---------------------------------------------------------------------------
# Pass 1: Snapshot tests
# ---------------------------------------------------------------------------
log_info "--- Pass 1: Running snapshot tests ---"

if [ "${generate_goldens}" = true ]; then
    export QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1
    log_info "QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 (writing new golden files)"
fi

snapshot_exit=0
(
    cd "${bin_dir}"
    export LD_LIBRARY_PATH="${bin_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    ./onnxruntime_provider_test \
        --gtest_filter="${snapshot_filter}" \
        --gtest_output="json:${snapshot_json}"
) || snapshot_exit=$?

# In update mode, unset so Pass 2 accuracy tests run normally (compare, don't write).
if [ "${generate_goldens}" = true ]; then
    unset QNN_UT_SNAPSHOT_GOLDEN_UPDATE
fi

# ---------------------------------------------------------------------------
# Analyze Pass 1 results
# ---------------------------------------------------------------------------

# Determine which groups need accuracy testing.
target_ops=""

if [ "${generate_goldens}" = true ] || [ "${force_accuracy}" = true ]; then
    # In update/force mode: run accuracy for all groups that were in scope.
    if [ -n "${filter_groups}" ]; then
        target_ops="${filter_groups}"
    else
        # Derive all groups from the JSON output (all snapshot suites that ran).
        target_ops=$(python3 -c "
import json, sys, re

with open(sys.argv[1]) as f:
    data = json.load(f)

ops = set()
pattern = re.compile(r'^QnnUnit_(.+?)_(?:SessionSnapshot|Snapshot)(?:_\w+)?Test$')
for suite in data.get('testsuites', []):
    m = pattern.match(suite['name'])
    if m:
        ops.add(m.group(1))

print(','.join(sorted(ops)))
" "${snapshot_json}" 2>/dev/null) || true
    fi
    if [ "${generate_goldens}" = true ]; then
        log_info "Goldens updated. Verifying accuracy for: ${target_ops}"
    else
        log_info "Force-accuracy mode. Running accuracy for: ${target_ops}"
    fi
else
    # Normal mode: run accuracy only for groups whose snapshot FAILED (drift):
    # the builder changed but the golden was not updated, so the new graph
    # structure is unverified and must be numerically checked.
    if [ ${snapshot_exit} -eq 0 ]; then
        log_info "All snapshot tests passed. No accuracy tests needed."
        exit 0
    fi

    if [ ! -f "${snapshot_json}" ]; then
        log_err "Snapshot JSON output not found at ${snapshot_json}."
        log_err "Test binary may have crashed (exit code: ${snapshot_exit})."
        exit 99
    fi

    target_ops=$(python3 -c "
import json, sys, re

with open(sys.argv[1]) as f:
    data = json.load(f)

ops = set()
pattern = re.compile(r'^QnnUnit_(.+?)_(?:SessionSnapshot|Snapshot)(?:_\w+)?Test$')
for suite in data.get('testsuites', []):
    if suite.get('failures', 0) > 0 or suite.get('errors', 0) > 0:
        m = pattern.match(suite['name'])
        if m:
            ops.add(m.group(1))

print(','.join(sorted(ops)))
" "${snapshot_json}" 2>/dev/null) || true

    if [ -z "${target_ops}" ]; then
        log_err "Snapshot tests exited ${snapshot_exit} but no group failures could be extracted."
        exit ${snapshot_exit}
    fi
fi

if [ -z "${target_ops}" ]; then
    log_info "No groups to verify. Done."
    exit 0
fi

log_info "Accuracy targets: ${target_ops}"

# ---------------------------------------------------------------------------
# Pass 2: Accuracy tests for target groups
# ---------------------------------------------------------------------------

# Probe whether accuracy tests are compiled in.
accuracy_probe=$("${binary}" --gtest_list_tests --gtest_filter="QnnUnit_*_Accuracy*" 2>/dev/null || true)
if [ -z "${accuracy_probe}" ]; then
    log_warn "No QnnUnit_*_Accuracy* tests found (QNN_EP_ACCURACY_UT not enabled?)."
    log_warn "Skipping Pass 2. Snapshot drift is unverified."
    exit 0
fi

# Build gtest filter from group list.
IFS=',' read -ra op_array <<< "${target_ops}"
accuracy_filter=""
for op in "${op_array[@]}"; do
    if [ -n "${accuracy_filter}" ]; then
        accuracy_filter+=":"
    fi
    accuracy_filter+="QnnUnit_${op}_Accuracy*Test.*"
done

log_info "--- Pass 2: Running accuracy tests ---"
log_info "Filter: ${accuracy_filter}"

accuracy_exit=0
(
    cd "${bin_dir}"
    export LD_LIBRARY_PATH="${bin_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    ./onnxruntime_provider_test \
        --gtest_filter="${accuracy_filter}" \
        --gtest_output="json:${accuracy_json}"
) || accuracy_exit=$?

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
if [ ${accuracy_exit} -eq 0 ]; then
    if [ "${generate_goldens}" = true ]; then
        log_info "=== PASS: Goldens updated and accuracy verified ==="
    else
        log_info "=== PASS: Snapshot drift verified numerically correct ==="
        log_info "Action: Run with --generate-goldens to accept the new graph structure."
    fi
    exit 0
else
    log_err "=== FAIL: Accuracy regression detected ==="
    log_err "Groups (${target_ops}): accuracy tests FAILED."
    log_err "Results: ${accuracy_json}"
    exit 1
fi
