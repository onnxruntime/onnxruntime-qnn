#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

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

# CTestTestfile.cmake files aren't relocatable. Rewrite it to find the build in this directory.

orig_build_dir=$(sed -n "s@# Build directory: @@p" CTestTestfile.cmake)
new_build_dir="${PWD}"

sed --in-place=".bak" "s@${orig_build_dir}@${new_build_dir}@g" CTestTestfile.cmake

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

log_info "-=-=-=- Running ONNX model tests -=-=-=-=-"

cd "${onnx_models_root}"

declare -a model_test_runners=("run_legacy_model_test" "run_model_test")
for runner in "${model_test_runners[@]}"; do
    "${runner}" cpu node "${REPO_ROOT}/cmake/external/onnx/onnx/backend/test/data/node"

    "${runner}" cpu float32

    if [ "$(uname -m)" != "aarch64" ]; then  # TODO: [AISW-164203] ORT test failures on Rubik Pi
        "${runner}" htp qdq
    fi

    log_debug "Scrubbing old context caches"
    find "testdata/qdq-with-context-cache" -name "*_ctx.onnx" -print -delete
    "${runner}" htp qdq-with-context-cache
done

exit "${errors}"
