
function(download_and_expand archive toolchain_dir url md5value)
  # Make toolchain_dir absolute relative to the current binary dir
  if (NOT EXISTS "${toolchain_dir}")
    if (NOT EXISTS "${archive}")
      message(STATUS "Downloading ${url}")
      file(DOWNLOAD "${url}" "${archive}"
           SHOW_PROGRESS
           EXPECTED_MD5 ${md5value}
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
list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/udo/udo_op_test.cpp)
find_package(Python REQUIRED COMPONENTS Interpreter)
if(UNIX)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # set -Wno-parentheses for linux to avoid compile error of extra parentheses in included files
        set_source_files_properties(${TEST_SRC_DIR}/providers/qnn/udo/udo_op_test.cpp PROPERTIES COMPILE_FLAGS "-Wno-parentheses")
        find_program(MAKE_EXECUTABLE make)

        # Linux CPU
        find_program(CLANGXX NAMES clang++)
        get_filename_component(toolchain_dir
            "clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04"
            REALPATH
            BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
        )
        if (NOT CLANGXX)
            download_and_expand(
                clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz
                ${toolchain_dir}
                https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz
                aeb379a5688b8d7b7d3c0d8353d30265
            )
        endif()
        add_custom_target(remove_cpu_udo_lib
            # add this target to ensure udo is always delete before rebuild
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_cpu.so
        )
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_cpu.so

            # generate op package
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
            ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/IncrementOpPackageCpu.xml -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu

            # copy pre-implement op package source file
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/IncrementCPU.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/IncrementOpPackage/src/ops/Increment.cpp
            # build op package
            COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                                            PATH=${toolchain_dir}/bin/:$ENV{PATH}
            ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/IncrementOpPackage all_x86

            # copy built op package
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/IncrementOpPackage/libs/x86_64-linux-clang/libIncrementOpPackage.so
                                            ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_cpu.so
            DEPENDS remove_cpu_udo_lib
        )
        add_custom_target(QnnUDO_Increment
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_cpu.so
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
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_htp.so
        )
        # message(FATAL_ERROR "HEXAGON_SDK_ROOT:${HEXAGON_SDK_ROOT}")
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_htp.so
            # generate op package
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python #HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}
            ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-linux-clang/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/IncrementOpPackageHtp.xml -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp

            # copy pre-implement op package source file
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/IncrementHTP.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/IncrementOpPackage/src/ops/Increment.cpp
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/HTP_Makefile
                                             ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/IncrementOpPackage/Makefile
            # build op package
            COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
                                            PATH=${toolchain_dir}/bin/:$ENV{PATH}
                                            HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT}/6.5.0.0
            ${MAKE_EXECUTABLE} -C ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/IncrementOpPackage htp_x86

            # copy built op package
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/htp/IncrementOpPackage/build/x86_64-linux-clang/libQnnIncrementOpPackage.so
                                            ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_htp.so
            DEPENDS remove_htp_udo_lib
        )
        list(APPEND onnxruntime_test_providers_dependencies QnnUDO_Increment_HTP)
        add_custom_target(QnnUDO_Increment_HTP
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_htp.so
        )
    endif()
elseif(WIN32)
    add_custom_target(remove_cpu_udo_lib
        # add this target to ensure udo is always delete before rebuild
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/libIncrementOpPackage_cpu.so
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/IncrementOpPackage_cpu.dll

        # generate op package
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${onnxruntime_QNN_HOME}/lib/python
        ${Python_EXECUTABLE} ${onnxruntime_QNN_HOME}/bin/x86_64-windows-msvc/qnn-op-package-generator -p ${TEST_SRC_DIR}/providers/qnn/udo/IncrementOpPackageCpu.xml -o ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu --gen_cmakelists

        # copy pre-implement op package source file
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SRC_DIR}/providers/qnn/udo/IncrementCPU.cpp
                                        ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/IncrementOpPackage/src/ops/Increment.cpp
        # build op package
        COMMAND ${CMAKE_COMMAND} -E env QNN_SDK_ROOT=${onnxruntime_QNN_HOME}
        ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/IncrementOpPackage
                            -B ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu
                            -DCMAKE_CXX_STANDARD=17
                            -G "Visual Studio 17 2022"
                            -T ClangCL
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu --config Release

        # copy built op package
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/qnn_udo_build/cpu/Release/IncrementOpPackage.dll
                                        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/IncrementOpPackage_cpu.dll
        DEPENDS remove_cpu_udo_lib
    )
    add_custom_target(QnnUDO_Increment
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/IncrementOpPackage_cpu.dll
    )
endif()
list(APPEND onnxruntime_test_providers_dependencies QnnUDO_Increment)
