#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Script to upload zip files to Artifactory using netrc authentication
# Usage: upload_zip_to_artifactory.sh <input_dir> <netrc_file>

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <netrc_file>"
    exit 1
fi

INPUT_DIR="$1"
NETRC_FILE="$2"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

if [ ! -f "$NETRC_FILE" ]; then
    echo "Error: Netrc file '$NETRC_FILE' does not exist"
    exit 1
fi

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)

# Upload files using --netrc-file
find "$INPUT_DIR" -name "*.zip" -type f -print0 | while IFS= read -r -d '' file; do
    file_basename=$(basename "$file")
    filename_no_ext="${file_basename%.zip}"
    version=$(echo "$filename_no_ext" | cut -d'-' -f3)
    
    echo "Uploading $file_basename (version: $version)..."
    
    curl -T "$file" \
        --cacert "$REPO_ROOT/qcom/scripts/upleveling/certs/artifactory-ca.pem" \
        --netrc-file "$NETRC_FILE" \
        https://re-artifactory.qualcomm.com/artifactory/aisw-zip-test-project/onnxruntime-qnn/"$version"/"$file_basename"
    
    echo "Successfully uploaded $file_basename"
done

echo "All zip files uploaded successfully"
