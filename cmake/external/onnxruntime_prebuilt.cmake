# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

include(ExternalProject)

set(ORT_SOURCE_DIR "${ort_core_SOURCE_DIR}")
set(ORT_BUILD_DIR "${ort_core_BINARY_DIR}")
message(STATUS "ORT_SOURCE_DIR: " ${ORT_SOURCE_DIR})
message(STATUS "ORT_BUILD_DIR: " ${ORT_BUILD_DIR})

# Determine the correct path for the test executable based on generator type
# Single-config generators (like Ninja) don't have config subdirectories
# Multi-config generators (like Visual Studio) have config subdirectories
get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(IS_MULTI_CONFIG)
    # Multi-config generators: executable is in config subdirectory
    set(ORT_PREBUILT_DEST "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
else()
    # Single-config generators: executable is directly in build directory
    set(ORT_PREBUILT_DEST "${CMAKE_BINARY_DIR}")
endif()

if(onnxruntime_ORT_HOME)
    message(STATUS "Use prebuilt from MS only at ${onnxruntime_ORT_HOME}. ORT Core will NOT be built from source")
    set(ORT_BUILD_COMMAND ${CMAKE_COMMAND} -E echo "Skipping ORT_BUILD_COMMAND")
    set(ORT_PREBUILT_SOURCE "${onnxruntime_ORT_HOME}/lib")
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
        --use_qnn
        --qnn_home "${onnxruntime_QNN_HOME}"
        --no_kleidiai
        --use_cache
        --targets onnxruntime_perf_test onnxruntime_plugin_ep_onnx_test onnxruntime
        COMMAND ${CMAKE_COMMAND} -E echo "ONNX Runtime build completed successfully"
    )
    # Append "--${ORT_PLATFORM}" to ORT_BUILD_COMMAND if we are on arm64 or arm64ec
    if(ORT_PLATFORM STREQUAL "arm64" OR ORT_PLATFORM STREQUAL "arm64ec")
        list(APPEND ORT_BUILD_COMMAND --${ORT_PLATFORM})
    endif()

    if(IS_MULTI_CONFIG)
        # Multi-config generators: executable is in config subdirectory
        set(ORT_PREBUILT_SOURCE "${ORT_BUILD_DIR}/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}")
    else()
        # Single-config generators: executable is directly in build directory
        set(ORT_PREBUILT_SOURCE "${ORT_BUILD_DIR}/${CMAKE_BUILD_TYPE}")
    endif()
endif()

# Determine ORT_INSTALL_COMMAND here
# Generic install command that copies required files from ORT_PREBUILT_SOURCE to ORT_PREBUILT_DEST
# Handles both Windows (.dll) and Linux (.so) platforms, and allows missing files
set(ORT_INSTALL_COMMAND
    ${CMAKE_COMMAND} -E echo "Copying files from ${ORT_PREBUILT_SOURCE} to ${ORT_PREBUILT_DEST}"
)

# Platform-specific library copying
if(WIN32)
    # Windows: Use file globbing to copy existing .dll and .lib files
    list(APPEND ORT_INSTALL_COMMAND
        COMMAND ${CMAKE_COMMAND} -E echo "Copying Windows ONNX Runtime files"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_PREBUILT_SOURCE}/onnxruntime.dll"
            "${ORT_PREBUILT_SOURCE}/onnxruntime.lib"
            "${ORT_PREBUILT_SOURCE}/onnxruntime_plugin_ep_onnx_test.exe"
            "${ORT_PREBUILT_DEST}"
    )
else()
    # Linux: Use file globbing to copy existing .so files and test executable
    # This will copy libonnxruntime.so, libonnxruntime.so.1, libonnxruntime.so.1.24.1, etc.
    list(APPEND ORT_INSTALL_COMMAND
        COMMAND ${CMAKE_COMMAND} -E echo "Copying Linux ONNX Runtime files"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_PREBUILT_SOURCE}/libonnxruntime.so"
            "${ORT_PREBUILT_SOURCE}/libonnxruntime.so.1"
            "${ORT_PREBUILT_SOURCE}/libonnxruntime.so.${ORT_CORE_VER}"
            "${ORT_PREBUILT_SOURCE}/onnxruntime_plugin_ep_onnx_test"
            "${ORT_PREBUILT_DEST}"
    )
endif()

# Add completion message
list(APPEND ORT_INSTALL_COMMAND
    COMMAND ${CMAKE_COMMAND} -E echo "File copying completed"
)

ExternalProject_Add(
    ort_core_target
    SOURCE_DIR ${ORT_SOURCE_DIR}
    BINARY_DIR ${ORT_BUILD_DIR}
    DOWNLOAD_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${ORT_BUILD_COMMAND}
    INSTALL_COMMAND ${ORT_INSTALL_COMMAND}
    # Enable comprehensive logging for debugging
    LOG_DOWNLOAD ON
    LOG_PATCH ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
    LOG_MERGED_STDOUTERR ON
)

# TODO: In long-term, we aim to remove the dependency of QNN-EP plugin library on ORT Core
set(ONNXRUNTIME_APPLICATION_SOURCE_ROOT "${ORT_SOURCE_DIR}/onnxruntime")
set(ONNXRUNTIME_APPLICATION_INCLUDE_ROOT "${ORT_SOURCE_DIR}/include/onnxruntime")

# Create imported target for ONNX Runtime
add_library(onnxruntime_prebuilt SHARED IMPORTED GLOBAL)
# Add dependency on the external projects to ensure they're downloaded first
add_dependencies(onnxruntime_prebuilt ort_core_target)

# Hack since cmake check the existence of INTERFACE_INCLUDE_DIRECTORIES
file(MAKE_DIRECTORY ${ONNXRUNTIME_APPLICATION_INCLUDE_ROOT})

# Platform-specific library configuration
if(WIN32)
    # Windows: Use .dll for IMPORTED_LOCATION and .lib for IMPORTED_IMPLIB
    set_target_properties(onnxruntime_prebuilt PROPERTIES
        IMPORTED_LOCATION ${ORT_PREBUILT_DEST}/onnxruntime.dll
        IMPORTED_IMPLIB ${ORT_PREBUILT_DEST}/onnxruntime.lib
        # INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_ORT_HOME}/include
    )
else()
    # Linux: Use .so for IMPORTED_LOCATION
    set_target_properties(onnxruntime_prebuilt PROPERTIES
        IMPORTED_LOCATION ${ORT_PREBUILT_DEST}/libonnxruntime.so
        # INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_ORT_HOME}/include
    )
endif()
