#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
#
# build_op_package.sh -- Build MyAdd QNN op package(s) and Python schema lib.
#
# Usage:
#   ./build_op_package.sh cpu          # build CPU x86 op package
#   ./build_op_package.sh htp          # build HTP x86 op package
#   ./build_op_package.sh schema       # build libMyAddSchema.so (Python schema lib)
#   ./build_op_package.sh all          # build all three
#
# Required environment variables:
#   QNN_SDK_ROOT      – path to QAIRT SDK root (e.g. .../qairt/<version>)
#   LLVM_TOOL_DIR     – path to LLVM bin dir   (e.g. .../LLVM-21.1.8-Linux-X64)
#   ORT_INCLUDE       – path to ORT public headers (e.g. <ort_repo>/include/onnxruntime)
#   ORT_LIB           – path to ORT lib dir       (e.g. <ort_build>/linux-x86_64/Release)
#   HEXAGON_SDK_ROOT  – (HTP only) path to Hexagon SDK version dir (e.g. .../6.5.0.0)
#
# Outputs (relative to this script's directory):
#   ../libMyAddOpPackage_cpu.so   (CPU target)
#   ../libMyAddOpPackage_htp.so   (HTP target)
#   ./libMyAddSchema.so           (Python schema companion lib)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UDO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"   # onnxruntime/test/providers/qnn/udo/
BUILD_DIR="${SCRIPT_DIR}/build"

build_schema() {
    echo ">>> Building Python schema companion lib (libMyAddSchema.so)..."
    ORT_INCLUDE="${ORT_INCLUDE:?ORT_INCLUDE must be set (path to ORT public headers)}"
    ORT_LIB="${ORT_LIB:?ORT_LIB must be set (path to ORT lib dir)}"

    g++ -std=c++17 -fPIC -shared \
        "${SCRIPT_DIR}/register_myadd_schema.cc" \
        -I"${ORT_INCLUDE}" \
        -L"${ORT_LIB}" -lonnxruntime \
        -Wl,-rpath,"${ORT_LIB}" \
        -o "${SCRIPT_DIR}/libMyAddSchema.so"
    echo ">>> Schema lib: ${SCRIPT_DIR}/libMyAddSchema.so"
}

build_cpu() {
    QNN_SDK_ROOT="${QNN_SDK_ROOT:?QNN_SDK_ROOT must be set}"
    LLVM_TOOL_DIR="${LLVM_TOOL_DIR:?LLVM_TOOL_DIR must be set}"
    echo ">>> Building CPU x86 op package..."
    local cpu_build="${BUILD_DIR}/cpu"
    rm -rf "${cpu_build}"

    # Step 1: generate skeleton
    PYTHONPATH="${QNN_SDK_ROOT}/lib/python" \
    python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-op-package-generator" \
        -p "${UDO_DIR}/MyAddOpPackageCpu.xml" \
        -o "${cpu_build}"

    # Step 2: copy pre-implemented kernel
    cp "${UDO_DIR}/MyAddCPU.cpp" \
       "${cpu_build}/MyAddOpPackage/src/ops/MyAdd.cpp"

    # Step 3: build
    QNN_SDK_ROOT="${QNN_SDK_ROOT}" \
    PATH="${LLVM_TOOL_DIR}/bin:${PATH}" \
    make -C "${cpu_build}/MyAddOpPackage" \
        "CXX=${LLVM_TOOL_DIR}/bin/clang++ -stdlib=libc++ -static-libstdc++ -Wl,--exclude-libs,ALL" \
        all_x86

    # Step 4: copy output
    cp "${cpu_build}/MyAddOpPackage/libs/x86_64-linux-clang/libMyAddOpPackage.so" \
       "${UDO_DIR}/libMyAddOpPackage_cpu.so"
    echo ">>> CPU package: ${UDO_DIR}/libMyAddOpPackage_cpu.so"
}

build_htp() {
    QNN_SDK_ROOT="${QNN_SDK_ROOT:?QNN_SDK_ROOT must be set}"
    LLVM_TOOL_DIR="${LLVM_TOOL_DIR:?LLVM_TOOL_DIR must be set}"
    HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:?HEXAGON_SDK_ROOT must be set for HTP build}"
    local htp_build="${BUILD_DIR}/htp"
    rm -rf "${htp_build}"

    # Step 1: generate skeleton
    PYTHONPATH="${QNN_SDK_ROOT}/lib/python" \
    python3 "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-op-package-generator" \
        -p "${UDO_DIR}/MyAddOpPackageHtp.xml" \
        -o "${htp_build}"

    # Step 2: copy pre-implemented kernel + custom HTP Makefile
    cp "${UDO_DIR}/MyAddHTP.cpp" \
       "${htp_build}/MyAddOpPackage/src/ops/MyAdd.cpp"
    cp "${UDO_DIR}/HTP_Makefile" \
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
       "${UDO_DIR}/libMyAddOpPackage_htp.so"
    echo ">>> HTP package: ${UDO_DIR}/libMyAddOpPackage_htp.so"
}

TARGET="${1:-all}"
case "${TARGET}" in
    cpu)    build_cpu ;;
    htp)    build_htp ;;
    schema) build_schema ;;
    all)    build_schema; build_cpu; build_htp ;;
    *)
        echo "Usage: $0 [cpu|htp|schema|all]"
        exit 1
        ;;
esac

echo ">>> Done."
