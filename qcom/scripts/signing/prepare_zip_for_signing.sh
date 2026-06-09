#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Script to prepare archives for signing
# Usage: ./prepare_zip_for_signing.sh <ZipDirectory> <OutputDirectory>

set -euo pipefail

ZIP_DIRECTORY="$1"
OUTPUT_DIRECTORY="$2"

# Verify the directories exist or create output directory
if [ ! -d "$ZIP_DIRECTORY" ]; then
    echo ""
    echo "ERROR: Directory not found: $ZIP_DIRECTORY" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_DIRECTORY" ]; then
    echo ""
    mkdir -p "$OUTPUT_DIRECTORY"
fi

# Create unsigned/zip subdirectories
UNSIGNED_DIR="$OUTPUT_DIRECTORY/unsigned_libs"
ZIP_DIR="$UNSIGNED_DIR/zip"

if [ ! -d "$ZIP_DIR" ]; then
    mkdir -p "$ZIP_DIR"
fi

OUTPUT_DIRECTORY="$ZIP_DIR"

# Find all zip files (excluding -pdb.zip files)
ZIP_FILES=$(find "$ZIP_DIRECTORY" -type f -name "*.zip" ! -name "*-pdb.zip" | sort)

ZIP_COUNT=$(echo "$ZIP_FILES" | grep -c . || echo 0)

if [ "$ZIP_COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No zip files found matching *.zip" >&2
    exit 1
fi

echo ""
echo "Found $ZIP_COUNT zip file(s)"
echo ""

PROCESSED_ZIPS=0
FAILED_ZIPS=0
FAILED_ZIP_NAMES=""
TOTAL_ZIPS=$ZIP_COUNT

# Record a zip failure: remove any partial output and update counters.
# Usage: record_failure <zip_name> <message> [path...]
record_failure() {
    local zip_name="$1"
    local message="$2"
    shift 2
    echo "  ERROR: $message"
    if [ "$#" -gt 0 ]; then
        rm -rf "$@"
    fi
    FAILED_ZIPS=$((FAILED_ZIPS + 1))
    FAILED_ZIP_NAMES="$FAILED_ZIP_NAMES$zip_name"$'\n'
}

# Convert to array to avoid subshell issues
mapfile -t ZIP_ARRAY <<< "$ZIP_FILES"

# Process each zip file
for zipFile in "${ZIP_ARRAY[@]}"; do
    [ -z "$zipFile" ] && continue

    ZIP_NAME=$(basename "$zipFile")
    ZIP_BASE_NAME="${ZIP_NAME%.zip}"
    TEMP_EXTRACT_DIR="$OUTPUT_DIRECTORY/${ZIP_BASE_NAME}_temp"
    FINAL_EXTRACT_DIR="$OUTPUT_DIRECTORY/$ZIP_BASE_NAME"

    echo "Processing: $ZIP_NAME"

    # Extract the zip to temporary directory using Python
    if ! python3 -m zipfile -e "$zipFile" "$TEMP_EXTRACT_DIR" 2>/dev/null; then
        record_failure "$ZIP_NAME" "Failed to process zip - Could not extract zip" "$TEMP_EXTRACT_DIR" "$FINAL_EXTRACT_DIR"
        continue
    fi

    # Create final extraction directory
    if [ ! -d "$FINAL_EXTRACT_DIR" ]; then
        mkdir -p "$FINAL_EXTRACT_DIR"
    fi

    # DLL path to extract
    DLL_PATH=$(find "$TEMP_EXTRACT_DIR" -name "onnxruntime_providers_qnn.dll" | head -1)

    if [ -z "$DLL_PATH" ]; then
        record_failure "$ZIP_NAME" "DLL not found in extracted zip" "$TEMP_EXTRACT_DIR" "$FINAL_EXTRACT_DIR"
        continue
    fi

    # Copy only the DLL to final directory
    TARGET_DLL_PATH="$FINAL_EXTRACT_DIR/onnxruntime_providers_qnn.dll"
    if ! cp "$DLL_PATH" "$TARGET_DLL_PATH" 2>/dev/null; then
        record_failure "$ZIP_NAME" "Failed to process zip - Could not copy DLL" "$TEMP_EXTRACT_DIR" "$FINAL_EXTRACT_DIR"
        continue
    fi

    # Clean up temporary extraction
    rm -rf "$TEMP_EXTRACT_DIR"

    PROCESSED_ZIPS=$((PROCESSED_ZIPS + 1))
    echo "  Ready for signing"
done

echo ""
echo "=== Preparation Summary ==="
echo "Total zip files: $TOTAL_ZIPS"
echo "Prepared for signing: $PROCESSED_ZIPS"
echo "Failures: $FAILED_ZIPS"
if [ -n "$FAILED_ZIP_NAMES" ]; then
    echo "$FAILED_ZIP_NAMES" | while read -r zip; do
        [ -z "$zip" ] && continue
        echo "  - $zip"
    done
fi
echo ""

if [ "$FAILED_ZIPS" -ne 0 ]; then
    echo "Preparation failed: $FAILED_ZIPS zip file(s) could not be processed"
    echo "=== End of Summary ==="
    exit 1
fi

echo "Preparation successful: $PROCESSED_ZIPS zip file(s) prepared for signing"
echo "=== End of Summary ==="

# Compress unsigned zip libs for signing
echo ""
echo "Compressing unsigned zip libs"
cd "$UNSIGNED_DIR/zip"
python3 << 'PYTHON_EOF'
import os
import shutil
import sys
try:
    shutil.make_archive('../zip', 'zip', '.', '.')
except Exception as error:
    print(f"ERROR: Failed to create zip.zip: {error}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    if [ -f "../zip.zip" ]; then
        echo "Successfully created zip.zip"
    else
        echo "ERROR: zip.zip was not created" >&2
        exit 1
    fi
else
    echo "ERROR: Failed to create zip.zip" >&2
    exit 1
fi
