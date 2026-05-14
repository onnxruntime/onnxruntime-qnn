#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Script to prepare wheels for signing
# Usage: ./prepare_wheels_for_signing.sh <WheelDirectory> <OutputDirectory>

set -euo pipefail

WHEEL_DIRECTORY="$1"
OUTPUT_DIRECTORY="$2"

# Verify the directories exist or create output directory
if [ ! -d "$WHEEL_DIRECTORY" ]; then
    echo ""
    echo "ERROR: Directory not found: $WHEEL_DIRECTORY" >&2
    exit 1
fi

if [ ! -d "$OUTPUT_DIRECTORY" ]; then
    echo ""
    mkdir -p "$OUTPUT_DIRECTORY"
fi

# Create unsigned/wheels subdirectories
UNSIGNED_DIR="$OUTPUT_DIRECTORY/unsigned"
WHEELS_DIR="$UNSIGNED_DIR/wheels"

if [ ! -d "$WHEELS_DIR" ]; then
    mkdir -p "$WHEELS_DIR"
fi

OUTPUT_DIRECTORY="$WHEELS_DIR"

# Find all wheels matching the pattern
WHEELS=$(find "$WHEEL_DIRECTORY" -type f \( -name "*win_amd64.whl" -o -name "*win_arm64.whl" \) | sort)

WHEEL_COUNT=$(echo "$WHEELS" | grep -c . || echo 0)

if [ "$WHEEL_COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No wheels found matching win_amd64.whl or win_arm64.whl" >&2
    exit 1
fi

echo ""
echo "Found $WHEEL_COUNT wheel(s)"
echo ""

PROCESSED_WHEELS=0
FAILED_WHEELS=0
FAILED_WHEEL_NAMES=""
TOTAL_WHEELS=$WHEEL_COUNT

# Record a wheel failure: remove any partial output and update counters.
# Usage: record_failure <wheel_name> <message> [path...]
record_failure() {
    local wheel_name="$1"
    local message="$2"
    shift 2
    echo "  ERROR: $message"
    if [ "$#" -gt 0 ]; then
        rm -rf "$@"
    fi
    FAILED_WHEELS=$((FAILED_WHEELS + 1))
    FAILED_WHEEL_NAMES="$FAILED_WHEEL_NAMES$wheel_name"$'\n'
}

# Process each wheel - use array to avoid subshell issues
mapfile -t WHEEL_ARRAY <<< "$WHEELS"

for wheel in "${WHEEL_ARRAY[@]}"; do
    [ -z "$wheel" ] && continue

    WHEEL_NAME=$(basename "$wheel")
    WHEEL_BASE_NAME="${WHEEL_NAME%.whl}"
    ZIP_PATH="$OUTPUT_DIRECTORY/$WHEEL_BASE_NAME.zip"
    IS_ARM64=false

    echo "Processing: $WHEEL_NAME"

    # Check if ARM64 wheel
    if [[ "$WHEEL_NAME" == *"win_arm64.whl" ]]; then
        IS_ARM64=true
    elif [[ "$WHEEL_NAME" == *"win_amd64.whl" ]]; then
        IS_ARM64=false
    else
        continue
    fi

    # Copy wheel and rename to .zip
    if ! cp "$wheel" "$ZIP_PATH" 2>/dev/null; then
        record_failure "$WHEEL_NAME" "Failed to process wheel - Could not copy wheel"
        continue
    fi

    if [ "$IS_ARM64" = true ]; then
        # ARM64: Extract to temp, copy only DLL, then clean up
        TEMP_EXTRACT_DIR="$OUTPUT_DIRECTORY/${WHEEL_BASE_NAME}_temp"
        FINAL_EXTRACT_DIR="$OUTPUT_DIRECTORY/$WHEEL_BASE_NAME"

        # Extract the zip to temporary directory using Python
        if ! python3 -m zipfile -e "$ZIP_PATH" "$TEMP_EXTRACT_DIR" 2>/dev/null; then
            record_failure "$WHEEL_NAME" "Failed to process wheel - Could not extract zip" "$TEMP_EXTRACT_DIR" "$ZIP_PATH"
            continue
        fi

        # Find the DLL in the temporary extraction
        DLL_PATH=$(find "$TEMP_EXTRACT_DIR" -name "onnxruntime_providers_qnn.dll" | head -1)

        if [ -z "$DLL_PATH" ]; then
            record_failure "$WHEEL_NAME" "DLL not found in extracted wheel" "$TEMP_EXTRACT_DIR" "$ZIP_PATH"
            continue
        fi

        # Create final extraction directory
        if [ ! -d "$FINAL_EXTRACT_DIR" ]; then
            mkdir -p "$FINAL_EXTRACT_DIR"
        fi

        # Copy only the DLL to final directory
        TARGET_DLL_PATH="$FINAL_EXTRACT_DIR/onnxruntime_providers_qnn.dll"
        if ! cp "$DLL_PATH" "$TARGET_DLL_PATH" 2>/dev/null; then
            record_failure "$WHEEL_NAME" "Failed to process wheel - Could not copy DLL" "$TEMP_EXTRACT_DIR" "$FINAL_EXTRACT_DIR" "$ZIP_PATH"
            continue
        fi

        # Clean up temporary extraction
        rm -rf "$TEMP_EXTRACT_DIR"
    else
        # AMD64: Extract to temp, copy DLLs from libs subdirectories, then clean up
        TEMP_EXTRACT_DIR="$OUTPUT_DIRECTORY/${WHEEL_BASE_NAME}_temp"
        FINAL_EXTRACT_DIR="$OUTPUT_DIRECTORY/$WHEEL_BASE_NAME"

        # Extract the zip to temporary directory using Python
        if ! python3 -m zipfile -e "$ZIP_PATH" "$TEMP_EXTRACT_DIR" 2>/dev/null; then
            record_failure "$WHEEL_NAME" "Failed to process wheel - Could not extract zip" "$TEMP_EXTRACT_DIR" "$ZIP_PATH"
            continue
        fi

        # Find the libs directory
        LIBS_DIR=$(find "$TEMP_EXTRACT_DIR" -type d -name "libs" | head -1)

        if [ -z "$LIBS_DIR" ]; then
            record_failure "$WHEEL_NAME" "libs directory not found in extracted wheel" "$TEMP_EXTRACT_DIR" "$ZIP_PATH"
            continue
        fi

        # Create final extraction directory
        if [ ! -d "$FINAL_EXTRACT_DIR" ]; then
            mkdir -p "$FINAL_EXTRACT_DIR"
        fi

        # Copy DLLs from amd64 and arm64ec subdirectories
        DLL_FOUND=false
        COPY_FAILED=false

        for subdir in amd64 arm64ec; do
            SUB_DIR_PATH="$LIBS_DIR/$subdir"
            SOURCE_DLL_PATH="$SUB_DIR_PATH/onnxruntime_providers_qnn.dll"

            if [ -f "$SOURCE_DLL_PATH" ]; then
                TARGET_DIR="$FINAL_EXTRACT_DIR/$subdir"
                if [ ! -d "$TARGET_DIR" ]; then
                    mkdir -p "$TARGET_DIR"
                fi

                TARGET_DLL_PATH="$TARGET_DIR/onnxruntime_providers_qnn.dll"
                if ! cp "$SOURCE_DLL_PATH" "$TARGET_DLL_PATH" 2>/dev/null; then
                    COPY_FAILED=true
                    break
                fi

                DLL_FOUND=true
            fi
        done

        if [ "$COPY_FAILED" = true ]; then
            record_failure "$WHEEL_NAME" "Failed to process wheel - Could not copy DLL" "$TEMP_EXTRACT_DIR" "$FINAL_EXTRACT_DIR" "$ZIP_PATH"
            continue
        fi

        if [ "$DLL_FOUND" = false ]; then
            record_failure "$WHEEL_NAME" "No DLLs found in libs subdirectories" "$TEMP_EXTRACT_DIR" "$FINAL_EXTRACT_DIR" "$ZIP_PATH"
            continue
        fi

        # Clean up temporary extraction
        rm -rf "$TEMP_EXTRACT_DIR"
    fi

    PROCESSED_WHEELS=$((PROCESSED_WHEELS + 1))
    echo "  Ready for signing"

    # Delete the zip file
    rm -f "$ZIP_PATH"
done

echo ""
echo "=== Preparation Summary ==="
echo "Total wheels: $TOTAL_WHEELS"
echo "Prepared for signing: $PROCESSED_WHEELS"
echo "Failures: $FAILED_WHEELS"
if [ -n "$FAILED_WHEEL_NAMES" ]; then
    echo "$FAILED_WHEEL_NAMES" | while read -r wheel; do
        [ -z "$wheel" ] && continue
        echo "  - $wheel"
    done
fi
echo ""

if [ "$FAILED_WHEELS" -ne 0 ]; then
    echo "Preparation failed: $FAILED_WHEELS wheel(s) could not be processed"
    echo "=== End of Summary ==="
    exit 1
fi

echo "Preparation successful: $PROCESSED_WHEELS wheel(s) prepared for signing"
echo "=== End of Summary ==="

# Compress unsigned wheels for signing
echo ""
echo "Compressing unsigned wheel libs"
cd "$UNSIGNED_DIR/wheels"
python3 << 'PYTHON_EOF'
import os
import shutil
import sys
try:
    shutil.make_archive('../wheels', 'zip', '.', '.')
except Exception as error:
    print(f"ERROR: Failed to create wheels.zip: {error}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    if [ -f "../wheels.zip" ]; then
        echo "Successfully created wheels.zip"
    else
        echo "ERROR: wheels.zip was not created" >&2
        exit 1
    fi
else
    echo "ERROR: Failed to create wheels.zip" >&2
    exit 1
fi
