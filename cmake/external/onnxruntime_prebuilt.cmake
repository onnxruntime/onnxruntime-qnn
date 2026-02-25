# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

include(ExternalProject)

set(ORT_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/ort_repo-src")
set(ORT_BUILD_DIR "${CMAKE_BINARY_DIR}/_deps/ort_repo-build")
set(GIT_TAG "v1.24.1")

# Option to control whether to build ONNX Runtime from source
option(BUILD_ONNXRUNTIME_FROM_SOURCE "Build ONNX Runtime from source using build.py" OFF)

# Conditional BUILD_COMMAND based on BUILD_ONNXRUNTIME_FROM_SOURCE option
if(NOT BUILD_ONNXRUNTIME_FROM_SOURCE)
    message(STATUS "ONNX Runtime will NOT be built from source (BUILD_ONNXRUNTIME_FROM_SOURCE=OFF)")
    set(ORT_BUILD_COMMAND ${CMAKE_COMMAND} -E echo "Skipping ONNX Runtime build (BUILD_ONNXRUNTIME_FROM_SOURCE=OFF)")
else()
    # Use Python to run build.py
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Validate required variables when building from source
    if(NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "Python3_EXECUTABLE is required when BUILD_ONNXRUNTIME_FROM_SOURCE is ON")
    endif()

    if(NOT onnxruntime_QNN_HOME)
        message(FATAL_ERROR "onnxruntime_QNN_HOME is required when BUILD_ONNXRUNTIME_FROM_SOURCE is ON")
    endif()

    string(TOLOWER ${onnxruntime_target_platform} ORT_PLATFORM)

    # Print configuration information
    message(STATUS "ONNX Runtime will be built from source with the following configuration:")
    message(STATUS "  Build Directory: ${ORT_BUILD_DIR}")
    message(STATUS "  Build Config: ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Architecture: ${ORT_PLATFORM}")
    message(STATUS "  QNN Home: ${onnxruntime_QNN_HOME}")
    message(STATUS "  Python Executable: ${Python3_EXECUTABLE}")
    message(STATUS "  CMake Generator: ${CMAKE_GENERATOR}")

    # Build ONNX Runtime from source using the provided build.py command
    set(ORT_BUILD_COMMAND
        ${CMAKE_COMMAND} -E echo "Building ONNX Runtime from source..."
        COMMAND ${Python3_EXECUTABLE} ${ORT_SOURCE_DIR}/tools/ci_build/build.py
        --build_dir ${ORT_BUILD_DIR}
        --config ${CMAKE_BUILD_TYPE}
        --build_shared_lib
        --parallel
        --skip_tests
        --cmake_generator "${CMAKE_GENERATOR}"
        "--${ORT_PLATFORM}"
        --use_qnn
        --qnn_home ${onnxruntime_QNN_HOME}
        --no_kleidiai
        COMMAND ${CMAKE_COMMAND} -E echo "ONNX Runtime build completed successfully"
    )
endif()

ExternalProject_Add(
    ort_repo
    DOWNLOAD_COMMAND
        ${CMAKE_COMMAND} -E remove_directory ${ORT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ORT_SOURCE_DIR}
        COMMAND git clone https://github.com/microsoft/onnxruntime.git ${ORT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E chdir ${ORT_SOURCE_DIR} git sparse-checkout init --cone
        COMMAND ${CMAKE_COMMAND} -E chdir ${ORT_SOURCE_DIR} git sparse-checkout set capi cmake include onnxruntime tools samples
        COMMAND ${CMAKE_COMMAND} -E chdir ${ORT_SOURCE_DIR} git checkout ${GIT_TAG}
    PATCH_COMMAND
        ${CMAKE_COMMAND} -E echo "Applying patch for the test binary onnxruntime_plugin_ep_onnx_test"
        COMMAND ${CMAKE_COMMAND} -E chdir ${ORT_SOURCE_DIR} git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/ort_test_binaries.patch
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${ORT_BUILD_COMMAND}
    INSTALL_COMMAND ""
)

# TODO: In long-term, we aim to remove the dependency of QNN-EP plugin library on ORT Core
set(ONNXRUNTIME_APPLICATION_SOURCE_ROOT "${CMAKE_BINARY_DIR}/_deps/ort_repo-src/onnxruntime")
set(ONNXRUNTIME_APPLICATION_INCLUDE_ROOT "${CMAKE_BINARY_DIR}/_deps/ort_repo-src/include/onnxruntime")

###########################################################################
# Import prebuilt ONNX Runtime libraries/binaries
###########################################################################

message(STATUS "onnxruntime_ORT_HOME: ${onnxruntime_ORT_HOME}")

# Create imported target for ONNX Runtime
add_library(onnxruntime_prebuilt SHARED IMPORTED GLOBAL)
# Add dependency on the external projects to ensure they're downloaded first
add_dependencies(onnxruntime_prebuilt ort_repo)

# Hack since cmake check the existence of INTERFACE_INCLUDE_DIRECTORIES
file(MAKE_DIRECTORY ${ONNXRUNTIME_APPLICATION_INCLUDE_ROOT})

# Platform-specific library configuration
if(WIN32)
    # Windows: Use .dll for IMPORTED_LOCATION and .lib for IMPORTED_IMPLIB
    set_target_properties(onnxruntime_prebuilt PROPERTIES
        IMPORTED_LOCATION ${onnxruntime_ORT_HOME}/lib/onnxruntime.dll
        IMPORTED_IMPLIB ${onnxruntime_ORT_HOME}/lib/onnxruntime.lib
        INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_ORT_HOME}/include
    )
elseif(UNIX AND NOT ANDROID)
    # Linux: Use .so for IMPORTED_LOCATION
    set_target_properties(onnxruntime_prebuilt PROPERTIES
        IMPORTED_LOCATION ${onnxruntime_ORT_HOME}/lib/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_ORT_HOME}/include
    )
else()
    # Fallback for other platforms
    message(WARNING "Unsupported platform for onnxruntime_prebuilt library configuration")
endif()
###########################################################################
