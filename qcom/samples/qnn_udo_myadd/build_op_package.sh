#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# build_op_package.sh -- Build MyAdd QNN op package(s).
#
# Usage:
#   ./build_op_package.sh cpu          # build CPU x86 op package
#   ./build_op_package.sh htp          # build HTP x86 op package
#   ./build_op_package.sh all          # build both
#
# Required environment variables:
#   QNN_SDK_ROOT      – path to QAIRT SDK root (e.g. .../qairt/<version>)
#   LLVM_TOOL_DIR     – path to LLVM bin dir   (e.g. .../LLVM-21.1.8-Linux-X64)
#   HEXAGON_SDK_ROOT  – (HTP only) path to Hexagon SDK version dir (e.g. .../6.5.0.0)
#
# Outputs (under this sample's artifacts/ directory):
#   artifacts/libMyAddOpPackage_cpu.so   (CPU target)
#   artifacts/libMyAddOpPackage_htp.so   (HTP target)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OP_PACKAGE_DIR="${REPO_ROOT}/onnxruntime/test/providers/qnn/udo"
ARTIFACT_DIR="${SCRIPT_DIR}/artifacts"
BUILD_DIR="${ARTIFACT_DIR}/build"

build_cpu() {
    QNN_SDK_ROOT="${QNN_SDK_ROOT:?QNN_SDK_ROOT must be set}"
    LLVM_TOOL_DIR="${LLVM_TOOL_DIR:?LLVM_TOOL_DIR must be set}"
    echo ">>> Building CPU x86 op package..."
    mkdir -p "${ARTIFACT_DIR}"
    local cpu_build="${BUILD_DIR}/cpu"
    rm -rf "${cpu_build}"

    # Step 1: generate skeleton
    PYTHONPATH="${QNN_SDK_ROOT}/lib/python" \
    python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-op-package-generator" \
        -p "${OP_PACKAGE_DIR}/MyAddOpPackageCpu.xml" \
        -o "${cpu_build}"

    # Step 2: copy pre-implemented kernel
    cp "${OP_PACKAGE_DIR}/MyAddCPU.cpp" \
       "${cpu_build}/MyAddOpPackage/src/ops/MyAdd.cpp"

    # Step 3: build
    QNN_SDK_ROOT="${QNN_SDK_ROOT}" \
    PATH="${LLVM_TOOL_DIR}/bin:${PATH}" \
    make -C "${cpu_build}/MyAddOpPackage" \
        "CXX=${LLVM_TOOL_DIR}/bin/clang++ -stdlib=libc++ -static-libstdc++ -Wl,--exclude-libs,ALL" \
        all_x86

    # Step 4: copy output
    cp "${cpu_build}/MyAddOpPackage/libs/x86_64-linux-clang/libMyAddOpPackage.so" \
       "${ARTIFACT_DIR}/libMyAddOpPackage_cpu.so"
    echo ">>> CPU package: ${ARTIFACT_DIR}/libMyAddOpPackage_cpu.so"
}

build_htp() {
    QNN_SDK_ROOT="${QNN_SDK_ROOT:?QNN_SDK_ROOT must be set}"
    LLVM_TOOL_DIR="${LLVM_TOOL_DIR:?LLVM_TOOL_DIR must be set}"
    HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:?HEXAGON_SDK_ROOT must be set for HTP build}"
    mkdir -p "${ARTIFACT_DIR}"
    local htp_build="${BUILD_DIR}/htp"
    rm -rf "${htp_build}"

    # Step 1: generate skeleton
    PYTHONPATH="${QNN_SDK_ROOT}/lib/python" \
    python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-op-package-generator" \
        -p "${OP_PACKAGE_DIR}/MyAddOpPackageHtp.xml" \
        -o "${htp_build}"

    # Step 2: copy pre-implemented kernel + custom HTP Makefile
    cp "${OP_PACKAGE_DIR}/MyAddHTP.cpp" \
       "${htp_build}/MyAddOpPackage/src/ops/MyAdd.cpp"
    cp "${OP_PACKAGE_DIR}/HTP_Makefile" \
       "${htp_build}/MyAddOpPackage/Makefile"

    # Step 3: build
    QNN_SDK_ROOT="${QNN_SDK_ROOT}" \
    HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT}" \
    PATH="${LLVM_TOOL_DIR}/bin:${PATH}" \
    make -C "${htp_build}/MyAddOpPackage" \
        "X86_CXX=${LLVM_TOOL_DIR}/bin/clang++ -stdlib=libc++" \
        htp_x86

    # Step 4: copy output
    cp "${htp_build}/MyAddOpPackage/build/x86_64-linux-clang/libQnnMyAddOpPackage.so" \
       "${ARTIFACT_DIR}/libMyAddOpPackage_htp.so"
    echo ">>> HTP package: ${ARTIFACT_DIR}/libMyAddOpPackage_htp.so"
}

TARGET="${1:-all}"
case "${TARGET}" in
    cpu)    build_cpu ;;
    htp)    build_htp ;;
    all)    build_cpu; build_htp ;;
    *)
        echo "Usage: $0 [cpu|htp|all]"
        exit 1
        ;;
esac

echo ">>> Done."
