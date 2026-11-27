#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Shared provider execution, tier scopes, verdict export and Accuracy selection.
QNN_NON_ROUTED_EXCLUSIONS='QnnSnapshot_*:QnnAcc_*'
QNN_SNAPSHOT_SCOPE='QnnSnapshot_*'
QNN_ACCURACY_SCOPE='QnnAcc_*'

qnn_run_provider() {
    local binary="$1" bin_dir="$2"
    shift 2
    (
        cd "${bin_dir}"
        export LD_LIBRARY_PATH="${bin_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        if [ -n "${QNN_PROVIDER_TIMEOUT_SECONDS:-}" ]; then
            timeout "${QNN_PROVIDER_TIMEOUT_SECONDS}" "${binary}" "$@"
        else
            "${binary}" "$@"
        fi
    )
}

qnn_export_verdict() {
    local bin_dir="$1" snapshot_json="$2" snapshot_list="$3" accuracy_list="$4"
    local output="$5" snapshot_exit="$6" mode="$7" runner_exit="$8" complete="$9"
    local qairt_version ort_version source_commit
    source "${REPO_ROOT}/qcom/scripts/all/resolve_tool_versions.sh"
    qairt_version="$(resolve_qairt_version "${bin_dir}" || true)"
    ort_version="$(resolve_ort_version "${bin_dir}" || true)"
    source_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)" || return 1
    local options=()
    [ "${complete}" != true ] || options+=(--selection-complete)
    "${PYTHON_BIN:-python3}" "${REPO_ROOT}/qcom/scripts/all/export_accuracy_routing.py" \
        --snapshot-json="${snapshot_json}" --snapshot-list="${snapshot_list}" \
        --accuracy-list="${accuracy_list}" \
        --manifest="${QNN_UT_SNAPSHOT_GOLDEN_DIR:-${bin_dir}/no-golden-store}/manifest.json" \
        --source-commit="${source_commit}" --qairt-version="${qairt_version}" --ort-version="${ort_version}" \
        --mode="${mode}" --runner-exit="${runner_exit}" --snapshot-exit="${snapshot_exit}" \
        --output="${output}" "${options[@]}" || return 1
}

qnn_export_and_select() {
    local bin_dir="$1" snapshot_json="$2" snapshot_list="$3" accuracy_list="$4"
    local verdict="$5" filter_file="$6" snapshot_exit="$7" mode="$8" summary="$9"
    qnn_snapshot_failure_is_routing_only "${snapshot_json}" "${snapshot_exit}" || return 1
    qnn_export_verdict "${bin_dir}" "${snapshot_json}" "${snapshot_list}" "${accuracy_list}" \
        "${verdict}" "${snapshot_exit}" "${mode}" 0 true || return 1
    local qairt_version ort_version source_commit
    qairt_version="$(resolve_qairt_version "${bin_dir}" || true)"
    ort_version="$(resolve_ort_version "${bin_dir}" || true)"
    source_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)" || return 1
    if ! "${PYTHON_BIN:-python3}" "${REPO_ROOT}/qcom/scripts/all/select_accuracy_from_routing.py" \
        --verdict="${verdict}" --accuracy-list="${accuracy_list}" --output="${filter_file}" \
        --source-commit="${source_commit}" --qairt-version="${qairt_version}" --ort-version="${ort_version}" \
        --summary="${summary}"; then
        rm -f -- "${verdict}" "${filter_file}" "${summary}"
        return 1
    fi
}

qnn_snapshot_failure_is_routing_only() {
    "${PYTHON_BIN:-python3}" "${REPO_ROOT}/qcom/scripts/all/select_accuracy_from_routing.py" \
        --snapshot-json="$1" --snapshot-exit="$2" --check-snapshot-exit
}
