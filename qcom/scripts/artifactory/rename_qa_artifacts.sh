#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# Renames build artifacts into the canonical QA naming convention before
# uploading to QA Artifactory.
#
# Usage:
#   rename_qa_artifacts.sh <build_dir> <version_number_file> \
#       <skip_arm64_debug> <skip_arm64_relwithdebinfo> <skip_arm64x>
#
# Arguments:
#   build_dir              Path to the build/ directory (e.g. $GITHUB_WORKSPACE/build)
#   version_number_file    Path to the VERSION_NUMBER file
#   skip_arm64_debug       "true" or "false"
#   skip_arm64_relwithdebinfo  "true" or "false"
#   skip_arm64x            "true" or "false"

set -euo pipefail

BUILD_DIR="${1:?build_dir is required}"
VERSION_NUMBER_FILE="${2:?version_number_file is required}"
SKIP_ARM64_DEBUG="${3:?skip_arm64_debug is required}"
SKIP_ARM64_RELWITHDEBINFO="${4:?skip_arm64_relwithdebinfo is required}"
SKIP_ARM64X="${5:?skip_arm64x is required}"

# Read version
VERSION=$(cat "$VERSION_NUMBER_FILE")
echo "Version: ${VERSION}"

# Export for the caller (GitHub Actions) to pick up
echo "QA_VERSION=${VERSION}" >> "${GITHUB_ENV:-/dev/null}"

# ---------------------------------------------------------------------------
# Rename test archives: onnxruntime-tests-{os}-{arch}.ext
#                     → onnxruntime_qnn-{ver}-{os}-{arch}.ext
# e.g. onnxruntime-tests-windows-arm64.zip → onnxruntime_qnn-2.2.0-windows-arm64.zip
# ---------------------------------------------------------------------------
for f in "${BUILD_DIR}"/onnxruntime-tests-*; do
    [ -f "$f" ] || continue
    filename=$(basename "$f")
    suffix="${filename#onnxruntime-tests-}"
    mv "$f" "${BUILD_DIR}/onnxruntime_qnn-${VERSION}-${suffix}"
    echo "Renamed: ${filename} → onnxruntime_qnn-${VERSION}-${suffix}"
done

# ---------------------------------------------------------------------------
# Rename AAR into android/ subfolder so it uploads separately from test_archives
# ---------------------------------------------------------------------------
mkdir -p "${BUILD_DIR}/android"
AAR_SRC="${BUILD_DIR}/android-aarch64/Release/java/build/android-qnn-ep/outputs/aar/onnxruntime-android-qnn.aar"
AAR_DST="${BUILD_DIR}/android/onnxruntime_qnn-${VERSION}-android.aar"
mv "$AAR_SRC" "$AAR_DST"
echo "Renamed AAR: $(basename "$AAR_SRC") → $(basename "$AAR_DST")"

# ---------------------------------------------------------------------------
# Rename Debug zip
# ---------------------------------------------------------------------------
if [ "${SKIP_ARM64_DEBUG}" == "false" ]; then
    DEBUG_ZIP=$(find "${BUILD_DIR}/windows-arm64/Debug/dist" -name "onnxruntime-qnn*.zip" 2>/dev/null | head -1)
    if [ -n "$DEBUG_ZIP" ]; then
        mv "$DEBUG_ZIP" "${BUILD_DIR}/onnxruntime_qnn-${VERSION}-windows-arm64-debug.zip"
        echo "Renamed Debug zip → onnxruntime_qnn-${VERSION}-windows-arm64-debug.zip"
    fi
fi

# ---------------------------------------------------------------------------
# Rename RelWithDebInfo zip
# ---------------------------------------------------------------------------
if [ "${SKIP_ARM64_RELWITHDEBINFO}" == "false" ]; then
    RWDI_ZIP=$(find "${BUILD_DIR}/windows-arm64/RelWithDebInfo/dist" -name "onnxruntime-qnn*.zip" 2>/dev/null | head -1)
    if [ -n "$RWDI_ZIP" ]; then
        mv "$RWDI_ZIP" "${BUILD_DIR}/onnxruntime_qnn-${VERSION}-windows-arm64-relwithdebinfo.zip"
        echo "Renamed RelWithDebInfo zip → onnxruntime_qnn-${VERSION}-windows-arm64-relwithdebinfo.zip"
    fi
fi

# ---------------------------------------------------------------------------
# Rename ARM64 NuGet:
#   Qualcomm.ML.OnnxRuntime.QNN.{ver}.nupkg → Qualcomm.ML.OnnxRuntime.QNN.{ver}.nupkg
# ---------------------------------------------------------------------------
ARM64_NUGET=$(find "${BUILD_DIR}/windows-arm64/Release/dist" -name "Qualcomm.ML.OnnxRuntime.QNN.*.nupkg" 2>/dev/null | head -1)
if [ -n "$ARM64_NUGET" ]; then
    NUGET_VER=$(basename "$ARM64_NUGET" | sed 's/Qualcomm\.ML\.OnnxRuntime\.QNN\.\(.*\)\.nupkg/\1/')
    NEW_NAME="$(dirname "$ARM64_NUGET")/Qualcomm.ML.OnnxRuntime.QNN.arm64.${NUGET_VER}.nupkg"
    mv "$ARM64_NUGET" "$NEW_NAME"
    echo "Renamed ARM64 NuGet → Qualcomm.ML.OnnxRuntime.QNN.${NUGET_VER}.nupkg"
fi

# ---------------------------------------------------------------------------
# Rename ARM64x NuGet:
#   Qualcomm.ML.OnnxRuntime.QNN.{ver}.nupkg → Qualcomm.ML.OnnxRuntime.QNN.ARM64x.{ver}.nupkg
# ---------------------------------------------------------------------------
if [ "${SKIP_ARM64X}" == "false" ]; then
    ARM64X_NUGET=$(find "${BUILD_DIR}/windows-arm64x/Release/Release/dist" -name "Qualcomm.ML.OnnxRuntime.QNN.*.nupkg" 2>/dev/null | head -1)
    if [ -n "$ARM64X_NUGET" ]; then
        NUGET_VER=$(basename "$ARM64X_NUGET" | sed 's/Qualcomm\.ML\.OnnxRuntime\.QNN\.\(.*\)\.nupkg/\1/')
        NEW_NAME="$(dirname "$ARM64X_NUGET")/Qualcomm.ML.OnnxRuntime.QNN.ARM64x.${NUGET_VER}.nupkg"
        mv "$ARM64X_NUGET" "$NEW_NAME"
        echo "Renamed ARM64x NuGet → Qualcomm.ML.OnnxRuntime.QNN.ARM64x.${NUGET_VER}.nupkg"
    fi
fi
