# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This CMake script builds the QNN UDO library for unit tests.
# It performs the full end‑to‑end steps required to generate the library.
# The resulting library is used by ONNX Runtime unit tests.
# If the required compiler is not available on the system,
# it will be downloaded from the LLVM GitHub repository.


function(download_and_expand archive toolchain_dir url sha256value)
    # Make toolchain_dir absolute relative to the current binary dir
    if (NOT EXISTS "${toolchain_dir}")
        if (NOT EXISTS "${archive}")
            message(STATUS "Downloading ${url}")
            file(DOWNLOAD "${url}" "${archive}"
                SHOW_PROGRESS
                EXPECTED_SHA256 ${sha256value}
        )
        else()
            message(STATUS "Using cached archive: ${archive}")
        endif()

        message(STATUS "Extracting ${archive} -> ${toolchain_dir}")
        file(MAKE_DIRECTORY "${toolchain_dir}")
        file(ARCHIVE_EXTRACT INPUT "${archive}")
    else()
        message(STATUS "Already extracted at: ${toolchain_dir}")
    endif()
endfunction()


# QNN EP udo tests not require CPU EP op implementations for accuracy evaluation
find_package(Python REQUIRED COMPONENTS Interpreter)
if(UNIX)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_program(MAKE_EXECUTABLE make)

        # Linux CPU
        find_program(CLANGXX NAMES clang++)
        get_filename_component(toolchain_dir
            "LLVM-21.1.8-Linux-X64"
            REALPATH
            BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
        )
        if (NOT CLANGXX)
            download_and_expand(
                LLVM-21.1.8-Linux-X64.tar.xz
                ${toolchain_dir}
                https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/LLVM-21.1.8-Linux-X64.tar.xz
                b3b7f2801d15d50736acea3c73982994d025b01c2f035b91ae3b49d1b575732b
            )
        endif()
        add_custom_target(remove_cpu_udo_lib
            # add this target to ensure udo is always delete before rebuild
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
        )
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so

            # generate op package
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
            ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageCpu.xml -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu

            # copy pre-implement op package source file
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/MyAddCPU.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage/src/ops/MyAdd.cpp
            # build op package
            COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                                            PATH=${toolchain_dir}/bin/:$ENV{PATH}
            ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage all_x86

            # copy built op package
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage/libs/x86_64-linux-clang/libMyAddOpPackage.so
                                            ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
            DEPENDS remove_cpu_udo_lib
        )
        add_custom_target(QnnUDO_MyAdd
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
        )

        # Linux HTP
        get_filename_component(HEXAGON_SDK_ROOT
            "Hexagon_SDK"
            REALPATH
            BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
        )
        download_and_expand(
            Hexagon_SDK_Linux.zip
            ${HEXAGON_SDK_ROOT}
            https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.5.0.0/Hexagon_SDK_Linux.zip
            c8dfd97d339043e3ace673dd17677fc6
        )
        add_custom_target(remove_htp_udo_lib
        # add this target to ensure udo is always delete before rebuild
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
        )
        # message(FATAL_ERROR "HEXAGON_SDK_ROOT:${HEXAGON_SDK_ROOT}")
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
            # generate op package
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python #HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
            ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageHtp.xml
                                                                                                         -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp

            # copy pre-implement op package source file
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/MyAddHTP.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/src/ops/MyAdd.cpp
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/Makefile
            # build op package
            COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                                            PATH=${toolchain_dir}/bin/:$ENV{PATH}
                                            HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}/6.5.0.0
            ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage htp_x86

            # copy built op package
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/MyAddOpPackage/build/x86_64-linux-clang/libQnnMyAddOpPackage.so
                                             ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
            DEPENDS remove_htp_udo_lib
        )
        list(APPEND onnxruntime_test_providers_dependencies QnnUDO_MyAdd_HTP)
        add_custom_target(QnnUDO_MyAdd_HTP
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
        )
    endif()
elseif(WIN32)
    add_custom_target(remove_cpu_udo_lib
        # add this target to ensure udo is always delete before rebuild
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MyAddOpPackage_cpu.dll

        # generate op package
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
        ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-windows-msvc/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/MyAddOpPackageCpu.xml
                                                                                                      -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
                                                                                                      --gen_cmakelists

        # copy pre-implement op package source file
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/MyAddCPU.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage/src/ops/MyAdd.cpp
        # build op package
        COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
        ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/MyAddOpPackage
                         -B ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
                         -DCMAKE_CXX_STANDARD=17
                         -G "Visual Studio 17 2022"
                         -T ClangCL
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu --config Release

        # copy built op package
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/Release/MyAddOpPackage.dll
                                         ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MyAddOpPackage_cpu.dll
        DEPENDS remove_cpu_udo_lib
    )
    add_custom_target(QnnUDO_MyAdd
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MyAddOpPackage_cpu.dll
    )
endif()
list(APPEND onnxruntime_test_providers_dependencies QnnUDO_MyAdd)
