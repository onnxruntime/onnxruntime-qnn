# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(ExternalProject)

set(SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/ort_repo-src")
set(GIT_TAG "128eaec7f9ef0c250b7ba9be3f1b0770ad2c4375")

###########################################################################
# TODO: In long-term, we aim to remove the dependency on source code
###########################################################################
ExternalProject_Add(
    ort_repo
    DOWNLOAD_COMMAND
        ${CMAKE_COMMAND} -E remove_directory ${SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SOURCE_DIR}
        COMMAND git clone https://github.qualcomm.com/MLG/onnxruntime-qnn-ep.git ${SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E chdir ${SOURCE_DIR} git sparse-checkout init --cone
        COMMAND ${CMAKE_COMMAND} -E chdir ${SOURCE_DIR} git sparse-checkout set capi cmake include onnxruntime tools samples
        COMMAND ${CMAKE_COMMAND} -E chdir ${SOURCE_DIR} git checkout ${GIT_TAG}
    # TODO: Once MS release 1.24, remove the PATCH_COMMAND
    PATCH_COMMAND
        ${CMAKE_COMMAND} -E echo "Applying patch to ensure ORT_API_VERSION is 23"
        COMMAND ${CMAKE_COMMAND} -E chdir ${SOURCE_DIR} git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/ort_api_version.patch
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set(ONNXRUNTIME_APPLICATION_SOURCE_ROOT "${CMAKE_BINARY_DIR}/_deps/ort_repo-src/onnxruntime")
set(ONNXRUNTIME_APPLICATION_INCLUDE_ROOT "${CMAKE_BINARY_DIR}/_deps/ort_repo-src/include/onnxruntime")
###########################################################################

###########################################################################
# Import prebuilt ONNX Runtime libraries/binaries
###########################################################################

message(STATUS "onnxruntime_ORT_HOME: ${onnxruntime_ORT_HOME}")

# Create imported target for ONNX Runtime
add_library(onnxruntime SHARED IMPORTED GLOBAL)
add_library(onnxruntime_providers_shared SHARED IMPORTED GLOBAL)
# Add dependency on the external projects to ensure they're downloaded first
add_dependencies(onnxruntime onnxruntime_providers_shared ort_repo)

# Hack since cmake check the existence of INTERFACE_INCLUDE_DIRECTORIES
file(MAKE_DIRECTORY ${ONNXRUNTIME_APPLICATION_INCLUDE_ROOT})
# Set the import library (.lib file for linking)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_ORT_HOME}/lib/onnxruntime.dll
    IMPORTED_IMPLIB ${onnxruntime_ORT_HOME}/lib/onnxruntime.lib
    INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_ORT_HOME}/include
)
set_target_properties(onnxruntime_providers_shared PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_ORT_HOME}/lib/onnxruntime_providers_shared.dll
    IMPORTED_IMPLIB ${onnxruntime_ORT_HOME}/lib/onnxruntime.lib
    INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_ORT_HOME}/include
)
###########################################################################
