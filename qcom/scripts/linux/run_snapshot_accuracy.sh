#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Two-pass snapshot+accuracy test runner for QNN EP unit tests.
#
# Pass 1: Run all snapshot tests (QnnUnit_<Op>_Snapshot* + QnnUnit_<Op>_SessionSnapshot*).
#          If all pass -> done (exit 0). Graph structure unchanged, so this
#          runner can skip the paired accuracy rerun.
# Pass 2: For any ops whose snapshot tests drifted or could not compare because
#          goldens are absent, run their QnnUnit_<Op>_Accuracy* tests to verify
#          numerical correctness.
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
snapshot tests drift or skip due to missing goldens, runs accuracy tests for the
affected ops only.

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

run_provider_test() {
    (
        cd "${bin_dir}"
        export LD_LIBRARY_PATH="${bin_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        ./onnxruntime_provider_test "$@"
    )
}

# Verify this is a coverage build: probe for snapshot tests.
snapshot_probe=$(run_provider_test --gtest_list_tests --gtest_filter="QnnUnit_*_Snapshot*" 2>/dev/null || true)
if [ -z "${snapshot_probe}" ]; then
    die "No QnnUnit_*_Snapshot* tests found in binary. This is not a coverage build (requires --enable-coverage)."
fi

extract_snapshot_groups() {
    local snapshot_json="$1"
    local mode="${2:-all}"
    python3 - "${snapshot_json}" "${mode}" <<'PY'
import json
import re
import sys

snapshot_json = sys.argv[1]
mode = sys.argv[2]
pattern = re.compile(r"^QnnUnit_(.+?)_(?:SessionSnapshot|Snapshot)(?:_\w+)?Test$")

with open(snapshot_json, encoding="utf-8") as f:
    data = json.load(f)


def contains_marker(value, marker):
    if isinstance(value, dict):
        return any(contains_marker(v, marker) for v in value.values())
    if isinstance(value, list):
        return any(contains_marker(v, marker) for v in value)
    return marker in str(value)


def suite_has_drift(suite):
    return contains_marker(suite, "QNN_SNAPSHOT_DRIFT")


def suite_has_absent_golden(suite):
    return contains_marker(suite, "QNN_GOLDEN_ABSENT")


def suite_has_skipped_test(suite):
    if suite.get("skipped", 0) > 0:
        return True
    for testcase in suite.get("testsuite", []):
        if str(testcase.get("result", "")).upper() == "SKIPPED":
            return True
        if str(testcase.get("status", "")).upper() == "SKIPPED":
            return True
    return False


def suite_is_unverified(suite):
    if suite_has_drift(suite) or suite_has_absent_golden(suite) or suite_has_skipped_test(suite):
        return True
    return suite.get("failures", 0) > 0 or suite.get("errors", 0) > 0


ops = set()
for suite in data.get("testsuites", []):
    match = pattern.match(suite.get("name", ""))
    if not match:
        continue

    include = False
    if mode == "all":
        include = True
    elif mode == "needs_accuracy":
        include = suite_is_unverified(suite)
    else:
        raise ValueError(f"unknown mode: {mode}")

    if include:
        ops.add(match.group(1))

print(",".join(sorted(ops)))
PY
}

extract_snapshot_groups_from_gtest_list() {
    local list_file="$1"
    python3 - "${list_file}" <<'PY'
import re
import sys

list_file = sys.argv[1]
pattern = re.compile(r"^QnnUnit_(.+?)_(?:SessionSnapshot|Snapshot)(?:_\w+)?Test$")

ops = set()
with open(list_file, encoding="utf-8") as f:
    for line in f:
        name = line.strip()
        if not name.endswith("."):
            continue
        match = pattern.match(name[:-1])
        if match:
            ops.add(match.group(1))

print(",".join(sorted(ops)))
PY
}

list_in_scope_snapshot_groups() {
    if [ -n "${filter_groups}" ]; then
        echo "${filter_groups}"
        return 0
    fi

    run_provider_test --gtest_list_tests --gtest_filter="${snapshot_filter}" > "${snapshot_list}"
    extract_snapshot_groups_from_gtest_list "${snapshot_list}"
}

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
snapshot_list="${results_dir}/snapshot_tests.txt"
rm -f "${snapshot_json}" "${accuracy_json}" "${snapshot_list}"

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
    --gtest_output="json:${snapshot_json}" || snapshot_exit=$?

# In update mode, unset so Pass 2 accuracy tests run normally (compare, don't write).
if [ "${generate_goldens}" = true ]; then
    unset QNN_UT_SNAPSHOT_GOLDEN_UPDATE
fi

# ---------------------------------------------------------------------------
# Analyze Pass 1 results
# ---------------------------------------------------------------------------

# Snapshot routing rule:
#   - [QNN_SNAPSHOT_DRIFT] means the snapshot test reached golden comparison and
#     only the QNN graph JSON changed.
#   - [QNN_GOLDEN_ABSENT] means no golden comparison happened for that case.
#   - Both markers leave the snapshot result unverified for that op group, so
#     route only the affected QnnUnit_<Op>_Accuracy* tests.
#   - Other snapshot failures also leave the snapshot result unverified. Route
#     the affected op to accuracy instead of blocking on graph-diff enforcement.
#   - If no snapshot JSON is produced, treat the in-scope snapshot groups as
#     unverified, same as missing goldens, and verify them through accuracy.
#   - The setup failure is an unverified snapshot group without a matching
#     QnnUnit_<Op>_Accuracy* test, because then correctness is not gated.
target_ops=""

if [ "${generate_goldens}" = true ] || [ "${force_accuracy}" = true ]; then
    # In update/force mode: run accuracy for all groups that were in scope.
    if [ -f "${snapshot_json}" ]; then
        target_ops=$(extract_snapshot_groups "${snapshot_json}" all 2>/dev/null) || true
    else
        # No snapshot result was produced. Treat the in-scope snapshot groups as
        # unverified, same as missing goldens, and verify them through accuracy.
        target_ops=$(list_in_scope_snapshot_groups 2>/dev/null) || true
    fi
    if [ "${generate_goldens}" = true ]; then
        log_info "Goldens updated. Verifying accuracy for: ${target_ops}"
    else
        log_info "Force-accuracy mode. Running accuracy for: ${target_ops}"
    fi
else
    if [ ! -f "${snapshot_json}" ]; then
        log_warn "Snapshot JSON output not found at ${snapshot_json}."
        log_warn "Treating in-scope snapshot groups as unverified and routing them to accuracy."
        target_ops=$(list_in_scope_snapshot_groups 2>/dev/null) || true
    else
        # Normal mode: run accuracy for groups whose snapshot is unverified:
        #   - [QNN_SNAPSHOT_DRIFT]: builder output changed vs the checked-in golden.
        #   - [QNN_GOLDEN_ABSENT]: no golden comparison happened.
        #   - any other snapshot failure/error: snapshot could not be used as an
        #     accuracy-skip signal.
        target_ops=$(extract_snapshot_groups "${snapshot_json}" needs_accuracy 2>/dev/null) || true
    fi

    if [ -z "${target_ops}" ]; then
        if [ ${snapshot_exit} -eq 0 ]; then
            log_info "All snapshot tests passed. No accuracy tests needed."
            exit 0
        fi
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

# Build gtest filter from group list and verify every unverified snapshot group
# has a matching accuracy suite. This is the only setup failure for an
# unverified snapshot: without QnnUnit_<Op>_Accuracy*, correctness is not gated.
IFS=',' read -ra op_array <<< "${target_ops}"
missing_accuracy=""
for op in "${op_array[@]}"; do
    accuracy_probe=$(run_provider_test --gtest_list_tests --gtest_filter="QnnUnit_${op}_Accuracy*Test.*" 2>/dev/null || true)
    if [ -z "${accuracy_probe}" ]; then
        if [ -n "${missing_accuracy}" ]; then
            missing_accuracy+=","
        fi
        missing_accuracy+="${op}"
    fi
done
if [ -n "${missing_accuracy}" ]; then
    die "No matching QnnUnit_<Op>_Accuracy* tests found for unverified snapshot groups: ${missing_accuracy}."
fi

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
run_provider_test \
    --gtest_filter="${accuracy_filter}" \
    --gtest_output="json:${accuracy_json}" || accuracy_exit=$?

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
    exit 0
else
    log_err "=== FAIL: Accuracy regression detected ==="
    log_err "Groups (${target_ops}): accuracy tests FAILED."
    log_err "Results: ${accuracy_json}"
    exit 1
fi
