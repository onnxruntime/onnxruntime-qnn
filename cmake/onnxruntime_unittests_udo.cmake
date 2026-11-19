# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# This CMake script builds the QNN UDO library for unit tests.
# It performs the full end‑to‑end steps required to generate the library.
# The resulting library is used by ONNX Runtime unit tests.


# Build the host registration library for HTP Custom Op tests.
find_package(Python REQUIRED COMPONENTS Interpreter)
# Must match qcom/packages.yml:llvm_linux_x86_64.version.
set(_LLVM_VERSION "21.1.8")
# Must match qcom/packages.yml:hexagon_linux_x86_64.version (also referenced from
# onnxruntime/test/providers/qnn/udo/HTP_Makefile HEXAGON_SDK_ROOT_V*).
set(_HEXAGON_SDK_VERSION "6.5.0.0")
if(UNIX)
    if ((CMAKE_SYSTEM_NAME STREQUAL "Linux" AND onnxruntime_target_platform STREQUAL "x86_64")
        OR (ANDROID AND onnxruntime_target_platform STREQUAL "aarch64" AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux"))
        find_program(MAKE_EXECUTABLE make REQUIRED)
        # The generator interpreter is independent of the wheel's target Python.
        # Prefer the wrapper's host venv, which supplies the generator dependencies.
        set(_UDO_GENERATOR_PYTHON "${Python_EXECUTABLE}")
        if(NOT (Python_VERSION_MAJOR EQUAL 3 AND (Python_VERSION_MINOR EQUAL 10 OR Python_VERSION_MINOR EQUAL 12)))
            find_program(_UDO_SUPPORTED_PYTHON NAMES python3.10 python3.12
                HINTS "${CMAKE_SOURCE_DIR}/../venv/bin" REQUIRED)
            set(_UDO_GENERATOR_PYTHON "${_UDO_SUPPORTED_PYTHON}")
        endif()
        execute_process(COMMAND ${_UDO_GENERATOR_PYTHON} -c "import sys; assert sys.version_info[:2] in ((3, 10), (3, 12))"
            RESULT_VARIABLE _UDO_PYTHON_STATUS)
        if(NOT _UDO_PYTHON_STATUS EQUAL 0)
            message(FATAL_ERROR "QNN op-package generator requires host Python 3.10 or 3.12")
        endif()

        # Locate the host tools used for Linux and Android HTP packages.
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
        get_filename_component(HEXAGON_SDK_ROOT
            "hexagon_linux_x86_64-${_HEXAGON_SDK_VERSION}/Hexagon_SDK"
            REALPATH
            BASE_DIR "${_TOOLS_DIR}"
        )
        set(_UDO_BUILD_TARGETS htp_x86)
        set(_UDO_HOST_ARCH x86_64-linux-clang)
        set(_UDO_DSP_COMMANDS)
        set(_UDO_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/libQnnMyAddOpPackage.so)
        if(ANDROID)
            set(onnxruntime_QNN_UDO_HTP_ARCH "81" CACHE STRING "Hexagon architecture for Android Custom Op tests")
            if(NOT onnxruntime_QNN_UDO_HTP_ARCH MATCHES "^(68|69|73|75|79|81|85)$")
                message(FATAL_ERROR "Unsupported Custom Op HTP architecture: ${onnxruntime_QNN_UDO_HTP_ARCH}")
            endif()
            set(_UDO_BUILD_TARGETS htp_aarch64 htp_v${onnxruntime_QNN_UDO_HTP_ARCH})
            set(_UDO_HOST_ARCH aarch64-android)
            list(APPEND _UDO_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/dsp/libQnnMyAddOpPackage.so)
            set(_UDO_DSP_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/dsp
                COMMAND ${CMAKE_COMMAND} -E copy
                    ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/build/hexagon-v${onnxruntime_QNN_UDO_HTP_ARCH}/libQnnMyAddOpPackage.so
                    ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/dsp/libQnnMyAddOpPackage.so)
        endif()
        add_custom_command(
            OUTPUT ${_UDO_OUTPUTS}

            # clean stale build dir before rebuilding
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp

            # generate op package
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
            ${_UDO_GENERATOR_PYTHON} ${onnxruntime_QNN_HOME}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageHtp.xml
                                                                                                         -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp

            # copy pre-implement op package source file
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/MyAddHTP.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/src/ops/MyAdd.cpp
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/Makefile
            # build op package
            COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                                            PATH=${HEXAGON_SDK_ROOT}/${_HEXAGON_SDK_VERSION}/tools/HEXAGON_Tools/19.0.07/Tools/bin:${LLVM_TOOL_DIR}/bin/:$ENV{PATH}
                                            ANDROID_NDK_ROOT=${ANDROID_NDK_PATH}
                                            HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}/${_HEXAGON_SDK_VERSION}
            ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage
                               "X86_CXX=${LLVM_TOOL_DIR}/bin/clang++ -stdlib=libc++"
                               ${_UDO_BUILD_TARGETS}

            # Produce the packaged lookup directory with the SDK library name.
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/build/${_UDO_HOST_ARCH}/libQnnMyAddOpPackage.so
                                             ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/libQnnMyAddOpPackage.so
            ${_UDO_DSP_COMMANDS}
            DEPENDS
                ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageHtp.xml
                ${TEST_SRC_DIR}/providers/qnn/udo/MyAddHTP.cpp
                ${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile
        )
        add_custom_target(QnnUDO_MyAdd_HTP
          DEPENDS ${_UDO_OUTPUTS}
        )
        list(APPEND onnxruntime_test_providers_dependencies QnnUDO_MyAdd_HTP)

        # Signal to the rest of the unit-test CMake (and the test source files)
        # that the fixture is required at runtime; a missing library is a failure.
        set(onnxruntime_BUILD_QNN_UDO_TEST ON)
    endif()
endif()
