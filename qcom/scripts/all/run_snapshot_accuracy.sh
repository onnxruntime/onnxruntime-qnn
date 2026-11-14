#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Platform-agnostic two-pass snapshot+accuracy runner for QNN EP unit tests.
#
# Pass 1: Run all snapshot tests (QnnSnapshot_<Op>_OpBuilder* + QnnSnapshot_<Op>_Session*).
# Pass 2: Use accuracy_gate.py to route each unverified snapshot case to its
#         paired QnnAcc_* case. Generate/force modes intentionally verify every
#         in-scope op group for golden production and manual diagnosis.
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

set -euo pipefail
if [ "${BASH_VERSINFO:-0}" -gt 3 ]; then
    shopt -s inherit_errexit
fi
if [ -n "${ORT_BUILD_XTRACE:-}" ]; then
    set -x
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

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
python_bin="${PYTHON_BIN:-python3}"
gate_script="${repo_root}/qcom/scripts/all/accuracy_gate.py"
version_resolver="${repo_root}/qcom/scripts/all/resolve_tool_versions.sh"

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
    (
        cd "${bin_dir}"
        export LD_LIBRARY_PATH="${bin_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        "${binary}" "$@"
    )
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

extract_snapshot_groups() {
    local snapshot_json="$1"
    "${python_bin}" - "${snapshot_json}" <<'PY'
import json
import re
import sys

snapshot_json = sys.argv[1]
pattern = re.compile(r"^QnnSnapshot_(.+?)_(?:OpBuilder|Session)(?:_\w+)?Test$")

with open(snapshot_json, encoding="utf-8") as f:
    data = json.load(f)

ops = set()
for suite in data.get("testsuites", []):
    match = pattern.match(suite.get("name", ""))
    if match:
        ops.add(match.group(1))

print(",".join(sorted(ops)))
PY
}

extract_snapshot_groups_from_gtest_list() {
    local list_file="$1"
    "${python_bin}" - "${list_file}" <<'PY'
import re
import sys

list_file = sys.argv[1]
pattern = re.compile(r"^QnnSnapshot_(.+?)_(?:OpBuilder|Session)(?:_\w+)?Test$")

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
        snapshot_filter+="QnnSnapshot_${g}_OpBuilder*Test.*:QnnSnapshot_${g}_Session*Test.*"
    done
else
    snapshot_filter="QnnSnapshot_*_OpBuilder*Test.*:QnnSnapshot_*_Session*Test.*"
fi

# ---------------------------------------------------------------------------
# Setup environment
# ---------------------------------------------------------------------------
results_dir="${bin_dir}/snapshot_accuracy_results"
mkdir -p "${results_dir}"

snapshot_json="${results_dir}/snapshot_results.json"
accuracy_json="${results_dir}/accuracy_results.json"
snapshot_list="${results_dir}/snapshot_tests.txt"
accuracy_list="${results_dir}/accuracy_tests.txt"
accuracy_filter_file="${results_dir}/accuracy_filter.txt"
gate_summary="${results_dir}/accuracy_gate_summary.txt"
rm -f "${snapshot_json}" \
      "${accuracy_json}" \
      "${snapshot_list}" \
      "${accuracy_list}" \
      "${accuracy_filter_file}" \
      "${gate_summary}"

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
if [ "${snapshot_exit}" -ne 0 ]; then
    log_warn "Snapshot tests exited ${snapshot_exit}; unverified cases will route to accuracy."
fi

# In update mode, unset so Pass 2 accuracy tests run normally (compare, don't write).
if [ "${generate_goldens}" = true ]; then
    unset QNN_UT_SNAPSHOT_GOLDEN_UPDATE
fi

# ---------------------------------------------------------------------------
# Normal validation: shared per-case routing contract
# ---------------------------------------------------------------------------
if [ "${generate_goldens}" = false ] && [ "${force_accuracy}" = false ]; then
    accuracy_list_filter="QnnAcc_*"
    if [ -n "${filter_groups}" ]; then
        accuracy_list_filter=""
        IFS=',' read -ra groups <<< "${filter_groups}"
        for group in "${groups[@]}"; do
            if [ -n "${accuracy_list_filter}" ]; then
                accuracy_list_filter+=":"
            fi
            accuracy_list_filter+="QnnAcc_${group}_Accuracy*Test.*"
        done
    fi

    if ! run_provider_test --gtest_list_tests --gtest_filter="${accuracy_list_filter}" > "${accuracy_list}"; then
        die "Failed to list accuracy tests."
    fi

    if [ -z "${QNN_UT_QAIRT_VERSION:-}" ] || [ -z "${QNN_UT_ORT_VERSION:-}" ]; then
        resolved_versions=""
        resolver_exit=0
        resolved_versions="$("${version_resolver}" --bin-dir="${bin_dir}" both)" || resolver_exit=$?
        if [ "${resolver_exit}" -eq 0 ]; then
            while IFS="=" read -r key value; do
                case "${key}" in
                    qairt) export QNN_UT_QAIRT_VERSION="${value}" ;;
                    ort) export QNN_UT_ORT_VERSION="${value}" ;;
                esac
            done <<< "${resolved_versions}"
            if [ -z "${QNN_UT_QAIRT_VERSION:-}" ] || [ -z "${QNN_UT_ORT_VERSION:-}" ]; then
                die "Tool version resolver returned incomplete QAIRT/ORT versions."
            fi
        elif [ "${resolver_exit}" -eq 3 ]; then
            unset QNN_UT_QAIRT_VERSION QNN_UT_ORT_VERSION
            log_warn "QAIRT/ORT versions could not be resolved; running full accuracy fallback."
        else
            die "Tool version resolver failed (exit ${resolver_exit})."
        fi
    fi

    log_info "--- Routing QNN accuracy tests from snapshot results ---"
    "${python_bin}" "${gate_script}" \
        --snapshot-json="${snapshot_json}" \
        --golden-root="${QNN_UT_SNAPSHOT_GOLDEN_DIR:-}" \
        --accuracy-list-file="${accuracy_list}" \
        --emit-filter-file="${accuracy_filter_file}" \
        --emit-summary-file="${gate_summary}"

    accuracy_filter="$(head -1 "${accuracy_filter_file}" 2>/dev/null || true)"
    [ -n "${accuracy_filter}" ] || die "accuracy_gate.py did not emit an accuracy filter."

    log_info "--- Pass 2: Running routed QNN accuracy tests ---"
    log_info "Filter: ${accuracy_filter}"
    accuracy_exit=0
    run_provider_test \
        --gtest_filter="${accuracy_filter}" \
        --gtest_output="json:${accuracy_json_arg}" || accuracy_exit=$?

    if [ "${accuracy_exit}" -ne 0 ]; then
        log_err "=== FAIL: Routed QNN accuracy tests failed ==="
        log_err "Results: ${accuracy_json}"
        exit 1
    fi

    log_info "=== PASS: Snapshot/accuracy routing completed ==="
    exit 0
fi

# ---------------------------------------------------------------------------
# Producer/forced validation: preserve the existing group-level routing
# ---------------------------------------------------------------------------

# Golden generation and forced validation always run accuracy for every
# in-scope snapshot group. The publisher uses this complete numerical evidence
# to decide which regenerated groups are eligible for upload.
target_ops=""

# In update/force mode: run accuracy for all groups that were in scope.
if [ -f "${snapshot_json}" ]; then
    target_ops=$(extract_snapshot_groups "${snapshot_json}" 2>/dev/null) || true
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

if [ -z "${target_ops}" ]; then
    log_info "No groups to verify. Done."
    exit 0
fi

log_info "Accuracy targets: ${target_ops}"

# ---------------------------------------------------------------------------
# Pass 2: Accuracy tests for target groups
# ---------------------------------------------------------------------------

# Build the group filter and verify every in-scope snapshot group has a matching
# accuracy suite; otherwise the producer lacks numerical evidence for that group.
IFS=',' read -ra op_array <<< "${target_ops}"
missing_accuracy=""
for op in "${op_array[@]}"; do
    accuracy_probe=$(run_provider_test --gtest_list_tests --gtest_filter="QnnAcc_${op}_Accuracy*Test.*" 2>/dev/null || true)
    if [ -z "${accuracy_probe}" ]; then
        if [ -n "${missing_accuracy}" ]; then
            missing_accuracy+=","
        fi
        missing_accuracy+="${op}"
    fi
done
if [ -n "${missing_accuracy}" ]; then
    die "No matching QnnAcc_<Op>_Accuracy* tests found for unverified snapshot groups: ${missing_accuracy}."
fi

accuracy_filter=""
for op in "${op_array[@]}"; do
    if [ -n "${accuracy_filter}" ]; then
        accuracy_filter+=":"
    fi
    accuracy_filter+="QnnAcc_${op}_Accuracy*Test.*"
done

log_info "--- Pass 2: Running accuracy tests ---"
log_info "Filter: ${accuracy_filter}"

accuracy_exit=0
run_provider_test \
    --gtest_filter="${accuracy_filter}" \
    --gtest_output="json:${accuracy_json_arg}" || accuracy_exit=$?

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
