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
    set(ONNXRUNTIME_APPLICATION_INCLUDES "${onnxruntime_ORT_HOME}/include")
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

    if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
        set(QNN_LIBRARY_KIND static_lib)
    else()
        set(QNN_LIBRARY_KIND shared_lib)
    endif()

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
        --use_qnn "${QNN_LIBRARY_KIND}"
        --qnn_home "${onnxruntime_QNN_HOME}"
        --no_kleidiai
    )
    if(onnxruntime_BUILD_CACHE)
        list(APPEND ORT_BUILD_COMMAND "--use_cache")
    endif()
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
        list(APPEND ORT_BUILD_COMMAND --android)
        list(APPEND ORT_BUILD_COMMAND --android_sdk_path)
        list(APPEND ORT_BUILD_COMMAND ${ANDROID_SDK_PATH})
        list(APPEND ORT_BUILD_COMMAND --android_ndk_path)
        list(APPEND ORT_BUILD_COMMAND ${ANDROID_NDK_PATH})
        list(APPEND ORT_BUILD_COMMAND --android_abi)
        list(APPEND ORT_BUILD_COMMAND ${ANDROID_ABI})
        list(APPEND ORT_BUILD_COMMAND --android_api)
        list(APPEND ORT_BUILD_COMMAND ${ANDROID_MIN_SDK})
        # Note: For Android builds, we don't add --${ORT_PLATFORM} to avoid architecture conflicts
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        # Handle CMAKE_TOOLCHAIN_FILE and ARM64 for Linux aarch64
        if(onnxruntime_target_platform STREQUAL "aarch64")
            list(APPEND ORT_BUILD_COMMAND --cmake_extra_defines)
            list(APPEND ORT_BUILD_COMMAND "CMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}")
            list(APPEND ORT_BUILD_COMMAND --cmake_extra_defines)
            list(APPEND ORT_BUILD_COMMAND "ARM64:BOOL=TRUE")

            # Disable SVE for aarch64 builds
            list(APPEND ORT_BUILD_COMMAND --no_sve)
        endif()
    else()
        # Windows
        if(NOT (${CMAKE_GENERATOR} STREQUAL "Ninja") AND (${ORT_PLATFORM} STREQUAL "arm64" OR ${ORT_PLATFORM} STREQUAL "arm64ec"))
            list(APPEND ORT_BUILD_COMMAND --${ORT_PLATFORM})
        endif()
        if(DEFINED BUILD_AS_ARM64X)
            list(APPEND ORT_BUILD_COMMAND --buildasx)
            list(APPEND ORT_BUILD_COMMAND --cmake_extra_defines)
            list(APPEND ORT_BUILD_COMMAND "arm64ReproDir=${CMAKE_CURRENT_SOURCE_DIR}/repros_ort_core")
        endif()
    endif()

    if(onnxruntime_CMAKE_DEPS_MIRROR_DIR)
        list(APPEND ORT_BUILD_COMMAND --cmake_deps_mirror_dir)
        list(APPEND ORT_BUILD_COMMAND "${onnxruntime_CMAKE_DEPS_MIRROR_DIR}")
    endif()

    list(APPEND ORT_BUILD_COMMAND --targets)
    list(APPEND ORT_BUILD_COMMAND onnxruntime_perf_test)
    list(APPEND ORT_BUILD_COMMAND onnxruntime_plugin_ep_onnx_test)
    list(APPEND ORT_BUILD_COMMAND onnxruntime)
    if (NOT (${CMAKE_SYSTEM_NAME} STREQUAL "Android"))
        list(APPEND ORT_BUILD_COMMAND onnxruntime_providers_shared)
    endif()

    if(IS_MULTI_CONFIG)
        # Multi-config generators: executable is in config subdirectory
        set(ORT_PREBUILT_SOURCE "${ORT_BUILD_DIR}/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}")
    else()
        # Single-config generators: executable is directly in build directory
        set(ORT_PREBUILT_SOURCE "${ORT_BUILD_DIR}/${CMAKE_BUILD_TYPE}")
    endif()

    set(ONNXRUNTIME_APPLICATION_INCLUDES
        # For onnxruntime_cxx_api.h
        "${ORT_SOURCE_DIR}/include/onnxruntime/core/session"
        # For cpu_provider_factory.h (Note: cpu_provider_factory.h is a public header released in ORT prebuilt)
        "${ORT_SOURCE_DIR}/include/onnxruntime/core/providers/cpu")
endif()

# Define platform-specific file lists.
# ORT_BINARY_FILES: library and test binaries to be copied during install, used as both install sources and build byproducts.
if(WIN32)
    set(ORT_BINARY_FILES
        "onnxruntime.dll"
        "onnxruntime.lib"
        "onnxruntime_providers_shared.dll"
        "onnxruntime_providers_shared.lib"
    )
    if (NOT onnxruntime_ORT_HOME)
        list(APPEND ORT_BINARY_FILES
            "onnxruntime_plugin_ep_onnx_test.exe"
            "onnxruntime_perf_test.exe"
        )
    endif()
elseif(ANDROID)
    set(ORT_BINARY_FILES
        "libonnxruntime.so"
    )
    if (NOT onnxruntime_ORT_HOME)
        list(APPEND ORT_BINARY_FILES
            "onnxruntime_plugin_ep_onnx_test"
            "onnxruntime_perf_test"
        )
    endif()
elseif(UNIX)
    # Linux: libonnxruntime.so.1 and the versioned symlink are extra; only the
    # unversioned .so files are needed as imported targets / build byproducts.
    set(ORT_BINARY_FILES
        "libonnxruntime.so"
        "libonnxruntime.so.1"
        "libonnxruntime.so.${ORT_CORE_VER}"
        "libonnxruntime_providers_shared.so"
    )
    if (NOT onnxruntime_ORT_HOME)
        list(APPEND ORT_BINARY_FILES
            "onnxruntime_plugin_ep_onnx_test"
            "onnxruntime_perf_test"
        )
    endif()
else()
    message(FATAL_ERROR "Unknown platform")
endif()

# Build the list of source paths for the install copy command.
set(ORT_PREBUILT_SOURCE_FILES)
foreach(_file IN LISTS ORT_BINARY_FILES)
    list(APPEND ORT_PREBUILT_SOURCE_FILES "${ORT_PREBUILT_SOURCE}/${_file}")
endforeach()

# Generic install command that copies required files from ORT_PREBUILT_SOURCE to ORT_PREBUILT_DEST.
# Handles both Windows (.dll/.lib) and Linux/Android (.so) platforms.
set(ORT_INSTALL_COMMAND
    ${CMAKE_COMMAND} -E make_directory "${ORT_PREBUILT_DEST}"
    COMMAND ${CMAKE_COMMAND} -E echo "Copying files from ${ORT_PREBUILT_SOURCE} to ${ORT_PREBUILT_DEST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_PREBUILT_SOURCE_FILES}
        "${ORT_PREBUILT_DEST}"
    COMMAND ${CMAKE_COMMAND} -E echo "File copying completed"
)

# Declare build byproducts so Ninja knows ort_core_target produces these files.
# Without BUILD_BYPRODUCTS, Ninja reports "missing and no known rule to make it"
# for IMPORTED_LOCATION / IMPORTED_IMPLIB files that don't exist at configure time.
# Derived directly from ORT_LIBRARY_FILES to avoid repeating filenames.
set(ORT_BUILD_BYPRODUCTS)
foreach(_file IN LISTS ORT_BINARY_FILES)
    list(APPEND ORT_BUILD_BYPRODUCTS "${ORT_PREBUILT_DEST}/${_file}")
endforeach()

message(STATUS "ORT_BUILD_COMMAND for ExternalProject: ${ORT_BUILD_COMMAND}")
ExternalProject_Add(
    ort_core_target
    SOURCE_DIR ${ORT_SOURCE_DIR}
    BINARY_DIR ${ORT_BUILD_DIR}
    DOWNLOAD_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${ORT_BUILD_COMMAND}
    BUILD_ALWAYS ON
    INSTALL_COMMAND ${ORT_INSTALL_COMMAND}
    BUILD_BYPRODUCTS ${ORT_BUILD_BYPRODUCTS}
    # Enable comprehensive logging for debugging
    LOG_DOWNLOAD ON
    LOG_PATCH ON
    LOG_CONFIGURE ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
    LOG_MERGED_STDOUTERR ON

    # Don't run more than one Ninja build at a time
    USES_TERMINAL_BUILD ON
)

# Create imported target for ONNX Runtime
add_library(onnxruntime SHARED IMPORTED GLOBAL)
add_dependencies(onnxruntime ort_core_target)

# onnxruntime_providers_shared is not built for Android as shown on line 252.
if(NOT ANDROID)
    add_library(onnxruntime_providers_shared SHARED IMPORTED GLOBAL)
    add_dependencies(onnxruntime_providers_shared ort_core_target)
endif()

# Platform-specific library configuration
if(WIN32)
    # Windows: Use .dll for IMPORTED_LOCATION and .lib for IMPORTED_IMPLIB
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ORT_PREBUILT_DEST}/onnxruntime.dll
        IMPORTED_IMPLIB ${ORT_PREBUILT_DEST}/onnxruntime.lib
    )
    set_target_properties(onnxruntime_providers_shared PROPERTIES
        IMPORTED_LOCATION ${ORT_PREBUILT_DEST}/onnxruntime_providers_shared.dll
        IMPORTED_IMPLIB ${ORT_PREBUILT_DEST}/onnxruntime_providers_shared.lib
    )
else()
    # Linux: Use .so for IMPORTED_LOCATION
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ORT_PREBUILT_DEST}/libonnxruntime.so
    )
    if(NOT ANDROID)
        set_target_properties(onnxruntime_providers_shared PROPERTIES
            IMPORTED_LOCATION ${ORT_PREBUILT_DEST}/libonnxruntime_providers_shared.so
        )
    endif()
endif()
