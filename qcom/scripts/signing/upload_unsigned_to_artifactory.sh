#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Script to upload unsigned signing artifacts to Artifactory
# Usage: upload_unsigned_to_artifactory.sh <zip_file> <filename> <version> <netrc_file>

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <zip_file> <filename> <version> <netrc_file>"
    exit 1
fi

ZIP_FILE="$1"
FILENAME="$2"
VERSION="$3"
NETRC_FILE="$4"

if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: Zip file '$ZIP_FILE' does not exist"
    exit 1
fi

if [ ! -f "$NETRC_FILE" ]; then
    echo "Error: Netrc file '$NETRC_FILE' does not exist"
    exit 1
fi

# Get the repository root directory
REPO_ROOT=$(git rev-parse --show-toplevel)

echo "Uploading $FILENAME"

curl --fail -s -T "$ZIP_FILE" \
    --cacert "$REPO_ROOT/qcom/scripts/upleveling/certs/artifactory-ca.pem" \
    --netrc-file "$NETRC_FILE" \
    https://re-artifactory.qualcomm.com/artifactory/aisw-zip-test-project/onnxruntime-qnn/"${VERSION}"/unsigned/"$FILENAME" > /dev/null

echo "Successfully uploaded $FILENAME"
