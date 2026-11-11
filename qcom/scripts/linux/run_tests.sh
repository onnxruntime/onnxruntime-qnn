#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

REPO_ROOT=$(git rev-parse --show-toplevel)

source "${REPO_ROOT}/qcom/scripts/linux/common.sh"
source "${REPO_ROOT}/qcom/scripts/linux/tools.sh"

set_strict_mode

# QNN CPU is not advertised by default; tests use it as the x86 attach point, so opt in.
export ORT_QNN_ENABLE_CPU_BACKEND=1

declare -i errors=0
#
# Run a command and increment ${errors} if it fails.
#
function count_errors() {
    set +e
    "$@"
    rc=$?
    set -e

    if [ ${rc} -ne 0 ]; then
        errors=$(($errors+1))
    fi
}

#
# Run a model test with onnxruntime_plugin_ep_onnx_test
#
function run_model_test() {
    local backend="${1}"
    local suite="${2}"
    local test_path="${3:-testdata/${suite}}"

    local model_log="${build_dir}/${suite}_model_tests.log"
    local model_xml="${build_dir}/${suite}_model_tests.results.xml"

    # Remove old log and XML files
    if [ -f "${model_log}" ]; then
        rm -f "${model_log}"
    fi
    if [ -f "${model_xml}" ]; then
        rm -f "${model_xml}"
    fi

    log_info "-=-=-=- Running onnx/models ${suite} tests with the ABI-stable EP plugin -=-=-=-"

    local runner_test_path="${test_path}"
    local filtered_test_root=""
    local filtered_test_path=""
    if [ "${suite}" = "node" ]; then
        # The upstream plugin runner takes a whole suite directory and has no
        # per-case skip flag. Filter QNN-owned exclusions into a temporary copy.
        # The filter requires a destination that does not yet exist. Keep the
        # temporary parent separate from the destination passed to the filter.
        filtered_test_root=$(mktemp -d -t qnn-model-tests.XXXXXX)
        filtered_test_path="${filtered_test_root}/${suite}"
        # Remove the temporary copy on either a normal return or an early
        # script exit (including a filter failure or interruption).
        trap 'if [ -n "${filtered_test_root}" ]; then rm -rf -- "${filtered_test_root}" || true; fi; trap - RETURN EXIT' RETURN EXIT
        set +e
        "${python_exe}" "${REPO_ROOT}/qcom/scripts/all/model_test_filter.py" \
            --source "${test_path}" \
            --destination "${filtered_test_path}" \
            --suite "${suite}" \
            --backend "${backend}" | tee "${model_log}"
        filter_return_code=${PIPESTATUS[0]}
        set -e
        if [ ${filter_return_code} -ne 0 ]; then
            errors=$(($errors+1))
            return
        fi
        runner_test_path="${filtered_test_path}"
    fi

    set +e
    "${build_dir}/onnxruntime_plugin_ep_onnx_test" \
        -j 1 \
        --plugin_ep_libs "qnn|libonnxruntime_providers_qnn.so" \
        --plugin_eps qnn \
        -i "backend_type|${backend}" \
        "${runner_test_path}" 2>&1 | tee -a "${model_log}"
    test_return_code=$?
    set -e
    if [ -n "${filtered_test_root}" ]; then
        rm -rf -- "${filtered_test_root}" || log_warn "Failed to remove filtered model-test suite: ${filtered_test_root}"
        filtered_test_root=""
        trap - RETURN EXIT
    fi

    if [ -f "${model_log}" ]; then
        "${python_exe}" "${REPO_ROOT}/qcom/scripts/all/model_test_log_to_junit_xml.py" \
            "${model_log}" > "${model_xml}"
    fi

    if [ ${test_return_code} -ne 0 ]; then
        errors=$(($errors+1))
    fi
}

python_exe=python3

for i in "$@"; do
  case $i in
    --python=*)
      python_exe="${i#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: ${i}"
      exit 1
  esac
done

cd "$(dirname ${BASH_SOURCE[0]})"
build_dir="${PWD}"

onnx_models_root="$(get_onnx_models_dir)"

# CTestTestfile.cmake files aren't relocatable. Rewrite it to find the build in this directory.
orig_build_dir=$(sed -n "s@# Build directory: @@p" CTestTestfile.cmake)
sed --in-place=".bak" "s@${orig_build_dir}@${build_dir}@g" CTestTestfile.cmake

log_info "-=-=-=- Running ctests -=-=-=-"
exclude_args=()
count_errors ./ctest --verbose --timeout 10800 --stop-on-failure "${exclude_args[@]}"

# TODO: We will support python wheel in linux
# log_info "-=-=-=- Running Python tests -=-=-=-"
# mapfile -t PYTHON_TEST_FILES < "python_test_files.txt"

# for python_file in "${PYTHON_TEST_FILES[@]}"; do
#     if [ -f "${python_file}" ]; then
#         # TODO: [AISW-164203] ORT test failures on Rubik Pi
#         if [[ "${python_file}" =~ ^(onnxruntime_test_python(_compile_api|_mlops)?.py)$ ]]; then
#             log_warn "Skipping ${python_file} due to known failures."
#         else
#             log_debug "Running ${python_file}..."
#             count_errors "${python_exe}" ${python_file}
#         fi
#     else
#         log_warn "Failed to find ${python_file} - may be OK on platforms which do not support Python."
#     fi
# done

# if [ -d "quantization" ]; then
#     # Quantization tests ran calling unittest directly in MSFT build.py
#     count_errors "${python_exe}" -m unittest discover -s quantization
# else
#     log_warn "Failed to find directory 'quantization' - may be OK on platforms which do not support Python."
# fi

log_info "-=-=-=- Running ONNX model tests -=-=-=-"

cd "${onnx_models_root}"

declare -a model_test_runners=("run_model_test")
for runner in "${model_test_runners[@]}"; do

    # Following tests are not supported on ARM64 Linux Runner
    # TODO: [AISW-163150]
    if [ "$(uname -m)" == "aarch64" ]; then
        rm -rf "${REPO_ROOT}/cmake/external/onnx/onnx/backend/test/data/node/test_strnormalizer_export_monday_casesensintive_lower"
        rm -rf "${REPO_ROOT}/cmake/external/onnx/onnx/backend/test/data/node/test_strnormalizer_export_monday_casesensintive_nochangecase"
        rm -rf "${REPO_ROOT}/cmake/external/onnx/onnx/backend/test/data/node/test_strnormalizer_export_monday_casesensintive_upper"
        rm -rf "${REPO_ROOT}/cmake/external/onnx/onnx/backend/test/data/node/test_strnormalizer_export_monday_empty_output"
    fi
    
    "${runner}" cpu node "${REPO_ROOT}/cmake/external/onnx/onnx/backend/test/data/node"

    #TODO: [AISW-164203] - Known issues with QDQ model suite
    if [ "$(uname -m)" != "aarch64" ]; then
        "${runner}" cpu float32
        "${runner}" htp qdq

        log_debug "Scrubbing old context caches"
        find "testdata/qdq-with-context-cache" -name "*_ctx.onnx" -print -delete
        "${runner}" htp qdq-with-context-cache
    fi

done

exit "${errors}"
