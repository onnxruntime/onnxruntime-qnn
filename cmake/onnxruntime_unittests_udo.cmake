# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# This CMake script builds the QNN UDO library for unit tests.
# It performs the full end‑to‑end steps required to generate the library.
# The resulting library is used by ONNX Runtime unit tests.


# Package tools are available only on Linux x86_64 hosts.
set(_UDO_SUPPORTED_HOST OFF)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_UDO_SUPPORTED_HOST ON)
endif()
set(_UDO_DEFAULT OFF)
if(_UDO_SUPPORTED_HOST AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND onnxruntime_target_platform STREQUAL "x86_64")
    set(_UDO_DEFAULT ON)
endif()
option(onnxruntime_QNN_BUILD_UDO_OP_PACKAGES "Build HTP Custom Op test packages" ${_UDO_DEFAULT})
if(NOT onnxruntime_QNN_BUILD_UDO_OP_PACKAGES)
    return()
endif()
if(NOT _UDO_SUPPORTED_HOST)
    message(FATAL_ERROR "HTP op-package generation requires a Linux x86_64 host; disable onnxruntime_QNN_BUILD_UDO_OP_PACKAGES")
endif()
# Must match qcom/packages.yml.
set(_LLVM_VERSION "21.1.8")
set(_HEXAGON_SDK_VERSION "6.5.0.0")
if(UNIX)
    if ((CMAKE_SYSTEM_NAME STREQUAL "Linux" AND onnxruntime_target_platform STREQUAL "x86_64")
        OR (ANDROID AND onnxruntime_target_platform STREQUAL "aarch64"))
        find_program(MAKE_EXECUTABLE make REQUIRED)
        set(onnxruntime_QNN_UDO_GENERATOR_PYTHON "" CACHE FILEPATH "Host Python 3.10/3.12 with mako and lxml")
        if(NOT onnxruntime_QNN_UDO_GENERATOR_PYTHON)
            find_program(_UDO_SUPPORTED_PYTHON NAMES python3.10 python3.12 REQUIRED)
            set(onnxruntime_QNN_UDO_GENERATOR_PYTHON "${_UDO_SUPPORTED_PYTHON}")
        endif()
        execute_process(COMMAND ${onnxruntime_QNN_UDO_GENERATOR_PYTHON} -c
            "import sys, mako, lxml; assert sys.version_info[:2] in ((3, 10), (3, 12))"
            RESULT_VARIABLE _UDO_PYTHON_STATUS)
        if(NOT _UDO_PYTHON_STATUS EQUAL 0)
            message(FATAL_ERROR "QNN generator requires Python 3.10/3.12 with mako and lxml; set onnxruntime_QNN_UDO_GENERATOR_PYTHON or disable onnxruntime_QNN_BUILD_UDO_OP_PACKAGES")
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
        set(_UDO_BUILD_TARGET htp_x86)
        set(_UDO_HOST_ARCH x86_64-linux-clang)
        set(_UDO_DSP_COMMANDS)
        set(_UDO_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/libQnnMyAddOpPackage.so)
        if(ANDROID)
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                         "${CMAKE_SOURCE_DIR}/../qcom/scripts/all/custom_op_targets.json")
            file(READ "${CMAKE_SOURCE_DIR}/../qcom/scripts/all/custom_op_targets.json" _UDO_TARGET_CONFIG)
            string(JSON _UDO_ARCH_COUNT LENGTH "${_UDO_TARGET_CONFIG}")
            math(EXPR _UDO_LAST_ARCH "${_UDO_ARCH_COUNT} - 1")
            file(STRINGS "${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile" _UDO_TARGETS_LINE REGEX "^SUPPORTED_TARGETS =")
            string(REPLACE "SUPPORTED_TARGETS =" "" _UDO_TARGETS "${_UDO_TARGETS_LINE}")
            separate_arguments(_UDO_TARGETS)
            set(_UDO_BUILD_TARGET htp_aarch64)
            set(_UDO_HOST_ARCH aarch64-android)
            foreach(_UDO_INDEX RANGE ${_UDO_LAST_ARCH})
                string(JSON _UDO_ARCH MEMBER "${_UDO_TARGET_CONFIG}" ${_UDO_INDEX})
                if(NOT "hexagon-v${_UDO_ARCH}" IN_LIST _UDO_TARGETS)
                    message(FATAL_ERROR "Unsupported Custom Op HTP architecture: ${_UDO_ARCH}")
                endif()
                list(APPEND _UDO_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/hexagon-v${_UDO_ARCH}/libQnnMyAddOpPackage.so)
                list(APPEND _UDO_DSP_COMMANDS
                    COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                        HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}/${_HEXAGON_SDK_VERSION}
                        ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage htp_v${_UDO_ARCH}
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/hexagon-v${_UDO_ARCH}
                    COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/build/hexagon-v${_UDO_ARCH}/libQnnMyAddOpPackage.so
                        ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/hexagon-v${_UDO_ARCH}/libQnnMyAddOpPackage.so)
            endforeach()
        endif()
        add_custom_command(
            OUTPUT ${_UDO_OUTPUTS}

            # clean stale build dir before rebuilding
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp

            # generate op package
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
            ${onnxruntime_QNN_UDO_GENERATOR_PYTHON} ${onnxruntime_QNN_HOME}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageHtp.xml
                                                                                                         -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp

            # copy pre-implement op package source file
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/MyAddHTP.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/src/ops/MyAdd.cpp
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/Makefile
            # build op package
            COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                                            PATH=${LLVM_TOOL_DIR}/bin/:$ENV{PATH}
                                            ANDROID_NDK_ROOT=${ANDROID_NDK_PATH}
                                            HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}/${_HEXAGON_SDK_VERSION}
            ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage
                               "X86_CXX=${LLVM_TOOL_DIR}/bin/clang++ -stdlib=libc++"
                               ${_UDO_BUILD_TARGET}

            # Produce the packaged lookup directory with the SDK library name.
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/build/${_UDO_HOST_ARCH}/libQnnMyAddOpPackage.so
                                             ${CMAKE_CURRENT_BINARY_DIR}/custom_op/op_packages/htp/libQnnMyAddOpPackage.so
            ${_UDO_DSP_COMMANDS}
            DEPENDS
                ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageHtp.xml
                ${TEST_SRC_DIR}/providers/qnn/udo/MyAddHTP.cpp
                ${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile
                ${CMAKE_SOURCE_DIR}/../qcom/scripts/all/custom_op_targets.json
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
