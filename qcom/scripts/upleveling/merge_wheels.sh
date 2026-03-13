#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Script to merge wheels
# Usage: merge_wheels.sh <amd_dir> <arm64ec_dir> <output_dir>

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <amd_dir> <arm64ec_dir> <output_dir>"
    exit 1
fi

AMD_DIR="$1"
ARM64EC_DIR="$2"
OUTPUT_DIR="$3"

if [ ! -d "$AMD_DIR" ]; then
    echo "Error: AMD directory '$AMD_DIR' does not exist"
    exit 1
fi

if [ ! -d "$ARM64EC_DIR" ]; then
    echo "Error: ARM64EC directory '$ARM64EC_DIR' does not exist"
    exit 1
fi

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# Upload files using --netrc-file
find "$AMD_DIR" -name "*.whl" -type f -print0 | while IFS= read -r -d '' file; do
    file_basename=$(basename "$file")

    echo "Merge AMD and ARM64EC wheels with filename $file_basename..."
    # The name of the arm64ec wheel should be the same as the name of the amd64 wheel.
    python ./qcom/scripts/upleveling/merge_wheels.py \
        --amd64-wheel $file \
        --arm64ec-wheel $ARM64EC_DIR/$file_basename \
        --output-folder $OUTPUT_DIR
    echo "Successfully merge $file_basename"
done

echo "All AMD wheels merged successfully"
