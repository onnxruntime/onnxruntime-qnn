#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tgz-directory)    tgz_directory="$2";    shift 2 ;;
    --test-package-zip) test_package_zip="$2"; shift 2 ;;
    --model-path)       model_path="$2";       shift 2 ;;
    --tgz-arch)         tgz_arch="$2";         shift 2 ;;
    --test-bin-arch)    test_bin_arch="$2";    shift 2 ;;
    --backend-lib)      backend_lib="$2";      shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

: "${tgz_directory:?--tgz-directory is required}"
: "${test_package_zip:?--test-package-zip is required}"
: "${model_path:?--model-path is required}"
: "${tgz_arch:?--tgz-arch is required}"
: "${test_bin_arch:?--test-bin-arch is required}"
backend_lib="${backend_lib:-libQnnHtp.so}"

# --- Extract the release tgz ---
release_tgz=$(find "$tgz_directory" -maxdepth 1 -name "*${tgz_arch}.tgz" | head -n 1)
if [ -z "$release_tgz" ]; then
    echo ""
    echo "ERROR: No ${tgz_arch} tgz found in $tgz_directory"
    exit 1
fi
release_tgz_name=$(basename "$release_tgz")
release_extract_dir="${tgz_directory}/${release_tgz_name%.tgz}_extracted"
echo ""
echo "Extracting release tgz: $release_tgz_name"
mkdir -p "$release_extract_dir"
tar -xf "$release_tgz" -C "$release_extract_dir"

# --- Extract test_package.zip ---
test_pkgs_dir="$(dirname "$test_package_zip")/test_package"
echo "Extracting test_package.zip"
mkdir -p "$test_pkgs_dir"
python3 -c "import zipfile; zipfile.ZipFile('${test_package_zip}').extractall('${test_pkgs_dir}')"

# --- Locate the test binaries folder in test_package ---
test_bin_dir="${test_pkgs_dir}/${test_bin_arch}"
if [ ! -d "$test_bin_dir" ]; then
    echo "ERROR: ${test_bin_arch} folder not found in $test_pkgs_dir"
    echo "Contents of $test_pkgs_dir:"
    ls -1 "$test_pkgs_dir"
    exit 1
fi

# --- Find onnxruntime_perf_test in the test binaries ---
perf_test=$(find "$test_bin_dir" -name "onnxruntime_perf_test" -not -name "*.exe" | head -n 1)
if [ -z "$perf_test" ]; then
    echo "ERROR: onnxruntime_perf_test not found in $test_bin_dir"
    exit 1
fi

# --- Copy onnxruntime_perf_test to the extracted release folder ---
cp "$perf_test" "$release_extract_dir/"
chmod +x "$release_extract_dir/onnxruntime_perf_test"
echo ""
echo "Copied onnxruntime_perf_test to release test directory"

# --- Copy libonnxruntime.so and libonnxruntime_providers_shared.so from test_package ---
cp "$test_bin_dir/libonnxruntime.so"                  "$release_extract_dir/"
cp "$test_bin_dir/libonnxruntime_providers_shared.so" "$release_extract_dir/"
echo "Copied libonnxruntime.so and libonnxruntime_providers_shared.so to release test directory"

# --- Run the perf test smoke test ---
echo "Running smoke test"
echo ""

cd "$release_extract_dir"
ln -sf libonnxruntime.so libonnxruntime.so.1
export LD_LIBRARY_PATH="$PWD"
export ADSP_LIBRARY_PATH="$PWD"
./onnxruntime_perf_test \
    -I \
    --plugin_ep_libs "QNNExecutionProvider|libonnxruntime_providers_qnn.so" \
    --plugin_eps QNNExecutionProvider \
    -m times \
    -r 1 \
    -p burst \
    -i "backend_path|${backend_lib}" \
    "$model_path"

echo ""
echo "Smoke test PASSED"
