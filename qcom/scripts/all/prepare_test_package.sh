#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() {
    echo "Usage: $0 <archives-directory> <output-directory> <version>"
    exit 1
}

[[ $# -lt 3 ]] && usage

ARCHIVES_DIR="$1"
OUTPUT_DIR="$2"
VERSION="$3"

# Verify the archives directory exists
if [[ ! -d "$ARCHIVES_DIR" ]]; then
    echo ""
    echo "Directory not found: $ARCHIVES_DIR"
    exit 1
fi

# Create output directory (test_package subfolder inside it)
TEST_PACKAGE_DIR="$OUTPUT_DIR/test_package"
if [[ -d "$TEST_PACKAGE_DIR" ]]; then
    echo ""
    echo "Folder already exists: $TEST_PACKAGE_DIR - continuing"
else
    mkdir -p "$TEST_PACKAGE_DIR"
    echo ""
    echo "Created folder: $TEST_PACKAGE_DIR"
fi

# Detect architecture from archive filename.
# Prints the arch string, or empty string to signal skip.
get_arch() {
    local name="$1"
    local lower
    lower="$(echo "$name" | tr '[:upper:]' '[:lower:]')"

    # Android is a known non-target platform
    if echo "$lower" | grep -qE "android"; then
        echo ""; return
    fi

    if echo "$lower" | grep -qE "arm64ec"; then
        echo "windows-arm64ec"; return
    fi
    if echo "$lower" | grep -qE "arm64"; then
        if echo "$lower" | grep -qE "win"; then
            echo "windows-arm64"; return
        fi
        if echo "$lower" | grep -qE "linux"; then
            if echo "$lower" | grep -qE "manylinux"; then
                echo "linux-arm64"; return
            fi
            echo ""; return  # excluded variant (e.g. oe_gcc11)
        fi
        echo ""; return
    fi
    # aarch64 is Linux-only naming
    if echo "$lower" | grep -qE "aarch64"; then
        if echo "$lower" | grep -qE "linux"; then
            if echo "$lower" | grep -qE "manylinux"; then
                echo "linux-arm64"; return
            fi
            echo ""; return  # excluded variant (e.g. oe_gcc11)
        fi
        echo ""; return
    fi
    if echo "$lower" | grep -qE "x86_64|x64|amd64"; then
        if echo "$lower" | grep -qE "win"; then
            echo "windows-x86_64"; return
        fi
        if echo "$lower" | grep -qE "linux"; then
            if echo "$lower" | grep -qE "ubuntu"; then
                echo "linux-x86_64"; return
            fi
            echo ""; return  # skip generic linux variant
        fi
        echo ""; return
    fi
    echo ""
}

WINDOWS_FILES=(
    "ep_weight_sharing_ctx_gen.exe"
    "msvcp140.dll"
    "msvcp140_1.dll"
    "onnxruntime.dll"
    "onnxruntime_perf_test.exe"
    "onnxruntime_plugin_ep_onnx_test.exe"
    "onnxruntime_provider_test.exe"
    "onnxruntime_providers_qnn.dll"
    "onnxruntime_providers_shared.dll"
    "vcruntime140.dll"
    "vcruntime140_1.dll"
)

LINUX_FILES=(
    "ep_weight_sharing_ctx_gen"
    "libonnxruntime.so"
    "libonnxruntime_providers_qnn.so"
    "libonnxruntime_providers_shared.so"
    "onnxruntime_perf_test"
    "onnxruntime_plugin_ep_onnx_test"
    "onnxruntime_provider_test"
)

# Collect archives (.zip, .tar, .tar.gz, .tar.bz2, .tgz)
mapfile -t ARCHIVES < <(find "$ARCHIVES_DIR" -maxdepth 1 -type f \( \
    -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" \
    -o -name "*.tar.bz2" -o -name "*.tgz" \) | sort)

TOTAL=${#ARCHIVES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo ""
    echo "No archive files found in: $ARCHIVES_DIR"
    exit 1
fi

echo ""
echo "Found $TOTAL archive(s)"
echo ""

TOTAL_ARCHIVES=$TOTAL
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
FAILED_NAMES=()
SKIPPED_NAMES=()

for ARCHIVE in "${ARCHIVES[@]}"; do
    BASENAME="$(basename "$ARCHIVE")"
    ARCH="$(get_arch "$BASENAME")"

    if [[ -z "$ARCH" ]]; then
        echo "SKIP: $BASENAME"
        echo ""
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_NAMES+=("$BASENAME")
        continue
    fi

    echo "Processing: $BASENAME  =>  $ARCH"

    ARCH_DIR="$TEST_PACKAGE_DIR/$ARCH"
    EXTRACT_TMP="$TEST_PACKAGE_DIR/${ARCH}_extract_tmp"

    mkdir -p "$ARCH_DIR"
    rm -rf "$EXTRACT_TMP"
    mkdir -p "$EXTRACT_TMP"

    # Extract archive
    if ! (
        case "$BASENAME" in
            *.zip)      python3 -m zipfile -e "$ARCHIVE" "$EXTRACT_TMP" ;;
            *.tar.gz)   tar -xzf "$ARCHIVE" -C "$EXTRACT_TMP" ;;
            *.tar.bz2)  python3 -W ignore::DeprecationWarning -m tarfile -e "$ARCHIVE" "$EXTRACT_TMP" ;;
            *.tgz)      tar -xzf "$ARCHIVE" -C "$EXTRACT_TMP" ;;
            *.tar)      tar -xf  "$ARCHIVE" -C "$EXTRACT_TMP" ;;
        esac
    ); then
        echo "  ERROR: Failed to extract archive"
        rm -rf "$EXTRACT_TMP"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_NAMES+=("$BASENAME")
        echo ""
        continue
    fi

    # Choose file list
    if [[ "$ARCH" == windows-* ]]; then
        EXPECTED=("${WINDOWS_FILES[@]}")
    else
        EXPECTED=("${LINUX_FILES[@]}")
    fi

    ALL_FOUND=true
    MISSING=()

    for PATTERN in "${EXPECTED[@]}"; do
        mapfile -t MATCHES < <(find "$EXTRACT_TMP" \( -type f -o -type l \) -name "$PATTERN" 2>/dev/null | head -1)

        if [[ ${#MATCHES[@]} -eq 0 || -z "${MATCHES[0]}" ]]; then
            echo "  MISSING: $PATTERN"
            ALL_FOUND=false
            MISSING+=("$PATTERN")
        else
            SRC="${MATCHES[0]}"
            DEST="$ARCH_DIR/$(basename "$SRC")"
            cp -P "$SRC" "$DEST"
            echo "  OK: $(basename "$SRC")"
        fi
    done

    rm -rf "$EXTRACT_TMP"

    if $ALL_FOUND; then
        echo "  All files collected for $ARCH"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        MISSING_STR="${MISSING[*]}"
        echo "  ERROR: Missing files for $ARCH"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_NAMES+=("$BASENAME (missing: ${MISSING_STR// /, })")
    fi

    echo ""
done

echo "=== Test Package Summary ==="
echo "Total archives found     : $TOTAL_ARCHIVES"
echo "Succeeded                : $SUCCESS_COUNT"
echo "Failed                   : $FAILED_COUNT"
if [[ ${#FAILED_NAMES[@]} -gt 0 ]]; then
    for NAME in "${FAILED_NAMES[@]}"; do
        echo "  - $NAME"
    done
fi
echo "Skipped                  : $SKIPPED_COUNT"
if [[ ${#SKIPPED_NAMES[@]} -gt 0 ]]; then
    for NAME in "${SKIPPED_NAMES[@]}"; do
        echo "  - $NAME"
    done
fi
echo "=== End of Summary ==="

if [[ $FAILED_COUNT -ne 0 ]]; then
    exit 1
fi

# Compress contents of test_package
ZIP_NAME="onnxruntime-qnn-${VERSION}-test_package.zip"
ZIP_PATH="$TEST_PACKAGE_DIR/$ZIP_NAME"
echo ""
echo "Compressing test_package contents"
if (cd "$TEST_PACKAGE_DIR" && python3 - "$ZIP_PATH" <<'PYEOF'
import os
import sys
import zipfile
zip_path = os.path.abspath(sys.argv[1])
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        for fname in sorted(files):
            fp = os.path.join(root, fname)
            if os.path.abspath(fp) != zip_path:
                zf.write(fp)
PYEOF
); then
    echo "Created $ZIP_NAME"
else
    echo "ERROR: Failed to create zip"
    exit 1
fi

# Remove everything in test_package except the zip
find "$TEST_PACKAGE_DIR" -mindepth 1 ! -name "$ZIP_NAME" -delete
