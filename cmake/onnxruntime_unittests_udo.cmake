# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# This CMake script builds the QNN UDO library for unit tests.
# It performs the full end‑to‑end steps required to generate the library.
# The resulting library is used by ONNX Runtime unit tests.


# QNN EP udo tests not require CPU EP op implementations for accuracy evaluation
find_package(Python REQUIRED COMPONENTS Interpreter)
set(_LLVM_VERSION "21.1.8")
set(_HEXAGON_SDK_VERSION "6.5.0.0")
# The CPU UDO op-package is built on both Linux x86_64 and Windows x86 hosts.
# HTP UDO compilation requires the Hexagon SDK toolchain, which is wired up only on Linux x86_64.
if(UNIX)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND onnxruntime_target_platform STREQUAL "x86_64")
        find_program(MAKE_EXECUTABLE make)

        # Linux CPU
        find_program(CLANGXX NAMES clang++)
        set(_TOOLS_DIR "$ENV{ORT_BUILD_TOOLS_PATH}")
        if(NOT _TOOLS_DIR)
            set(_TOOLS_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../tools")
        endif()
        get_filename_component(LLVM_TOOL_DIR
            "LLVM-${_LLVM_VERSION}-Linux-X64"
            REALPATH
            BASE_DIR "${_TOOLS_DIR}"
        )
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so

            # clean stale build dir before rebuilding
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu

            # generate op package
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
            ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageCpu.xml -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu

            # copy pre-implement op package source file
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/MyAddCPU.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage/src/ops/MyAdd.cpp
            # build op package
            COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                                            PATH=${LLVM_TOOL_DIR}/bin/:$ENV{PATH}
            ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage
                               "CXX=${CLANGXX} -stdlib=libc++ -static-libstdc++ -Wl,--exclude-libs,ALL"
                               all_x86

            # copy built op package
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage/libs/x86_64-linux-clang/libMyAddOpPackage.so
                                            ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
            DEPENDS
                ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageCpu.xml
                ${TEST_SRC_DIR}/providers/qnn/udo/MyAddCPU.cpp
        )
        add_custom_target(QnnUDO_MyAdd
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
        )
        list(APPEND onnxruntime_test_providers_dependencies QnnUDO_MyAdd)

        # Linux HTP (reuses _TOOLS_DIR and LLVM_TOOL_DIR from the CPU block above)
        get_filename_component(HEXAGON_SDK_ROOT
            "hexagon_linux_x86_64-${_HEXAGON_SDK_VERSION}/Hexagon_SDK"
            REALPATH
            BASE_DIR "${_TOOLS_DIR}"
        )
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so

            # clean stale build dir before rebuilding
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp

            # generate op package
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
            ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageHtp.xml
                                                                                                         -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp

            # copy pre-implement op package source file
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/MyAddHTP.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/src/ops/MyAdd.cpp
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/Makefile
            # build op package
            COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                                            PATH=${LLVM_TOOL_DIR}/bin/:$ENV{PATH}
                                            HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}/${_HEXAGON_SDK_VERSION}
            ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage
                               "X86_CXX=clang++ -stdlib=libc++"
                               htp_x86

            # copy built op package
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/build/x86_64-linux-clang/libQnnMyAddOpPackage.so
                                             ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
            DEPENDS
                ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageHtp.xml
                ${TEST_SRC_DIR}/providers/qnn/udo/MyAddHTP.cpp
                ${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile
        )
        add_custom_target(QnnUDO_MyAdd_HTP
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
        )
        list(APPEND onnxruntime_test_providers_dependencies QnnUDO_MyAdd_HTP)
    endif()
elseif(WIN32)
    # Windows CPU only (HTP UDO is not supported on Windows).
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MyAddOpPackage_cpu.dll

        # clean stale build dir and prior artifact before rebuilding
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
        COMMAND ${CMAKE_COMMAND} -E rm -f ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MyAddOpPackage_cpu.dll

        # generate op package (Windows requires --gen_cmakelists to emit CMakeLists.txt instead of Makefile)
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
        ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-windows-msvc/qnn-op-package-generator
                             -p ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageCpu.xml
                             -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
                             --gen_cmakelists

        # copy pre-implement op package source file
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/MyAddCPU.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage/src/ops/MyAdd.cpp

        # configure + build op package via cmake (VS 2022 + ClangCL)
        COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
        ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage
                         -B ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
                         -DCMAKE_CXX_STANDARD=17
                         -G "Visual Studio 17 2022"
                         -T ClangCL
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu --config Release

        # copy built op package next to the test binary
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/Release/MyAddOpPackage.dll
                                         ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MyAddOpPackage_cpu.dll
        DEPENDS
            ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageCpu.xml
            ${TEST_SRC_DIR}/providers/qnn/udo/MyAddCPU.cpp
    )
    add_custom_target(QnnUDO_MyAdd
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MyAddOpPackage_cpu.dll
    )
    list(APPEND onnxruntime_test_providers_dependencies QnnUDO_MyAdd)
endif()
