# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# This CMake script builds the QNN UDO library for unit tests.
# It performs the full end‑to‑end steps required to generate the library.
# The resulting library is used by ONNX Runtime unit tests.


# QNN EP udo tests not require CPU EP op implementations for accuracy evaluation
find_package(Python REQUIRED COMPONENTS Interpreter)
# Must match qcom/packages.yml:llvm_linux_x86_64.version.
set(_LLVM_VERSION "21.1.8")
# Must match qcom/packages.yml:hexagon_linux_x86_64.version (also referenced from
# onnxruntime/test/providers/qnn/udo/HTP_Makefile HEXAGON_SDK_ROOT_V*).
set(_HEXAGON_SDK_VERSION "6.5.0.0")
# qnn-op-package-generator requires Python 3.10. Skip the UDO unit test build when the
# discovered interpreter is any other version (e.g. Windows CI uses 3.11+).
if(NOT (Python_VERSION_MAJOR EQUAL 3 AND Python_VERSION_MINOR EQUAL 10))
    message(STATUS "Skipping QNN UDO unit test build: requires Python 3.10, found ${Python_VERSION}")
    return()
endif()
if(UNIX)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND onnxruntime_target_platform STREQUAL "x86_64")
        find_program(MAKE_EXECUTABLE make)

        # Linux CPU
        set(_TOOLS_DIR "$ENV{ORT_BUILD_TOOLS_PATH}")
        if(NOT _TOOLS_DIR)
            # Mirrors qcom/scripts/linux/tools.sh:get_tools_dir() and qcom/ep_build/tools.py:get_tools_dir().
            # CMAKE_SOURCE_DIR for onnxruntime is <repo>/cmake, so its parent is the repo root.
            set(_TOOLS_DIR "${CMAKE_SOURCE_DIR}/../build/tools")
        endif()
        get_filename_component(LLVM_TOOL_DIR
            "llvm_linux_x86_64-${_LLVM_VERSION}/LLVM-${_LLVM_VERSION}-Linux-X64"
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
                               "CXX=${LLVM_TOOL_DIR}/bin/clang++ -stdlib=libc++ -static-libstdc++ -Wl,--exclude-libs,ALL"
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
                               "X86_CXX=${LLVM_TOOL_DIR}/bin/clang++ -stdlib=libc++"
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

        # Signal to the rest of the unit-test CMake (and the test source files)
        # that the UDO library will actually be built and is available at runtime.
        set(onnxruntime_BUILD_QNN_UDO_TEST ON)
    endif()
endif()
