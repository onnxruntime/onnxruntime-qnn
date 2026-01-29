# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(ExternalProject)

set(SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/ort_repo-src")
set(GIT_TAG "v1.24.1")

###########################################################################
# TODO: In long-term, we aim to remove the dependency on source code
###########################################################################
ExternalProject_Add(
    ort_repo
    DOWNLOAD_COMMAND
        ${CMAKE_COMMAND} -E remove_directory ${SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SOURCE_DIR}
        COMMAND git clone https://github.com/microsoft/onnxruntime.git ${SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E chdir ${SOURCE_DIR} git sparse-checkout init --cone
        COMMAND ${CMAKE_COMMAND} -E chdir ${SOURCE_DIR} git sparse-checkout set capi cmake include onnxruntime tools samples
        COMMAND ${CMAKE_COMMAND} -E chdir ${SOURCE_DIR} git checkout ${GIT_TAG}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set(ONNXRUNTIME_APPLICATION_SOURCE_ROOT "${CMAKE_BINARY_DIR}/_deps/ort_repo-src/onnxruntime")
set(ONNXRUNTIME_APPLICATION_INCLUDE_ROOT "${CMAKE_BINARY_DIR}/_deps/ort_repo-src/include/onnxruntime")
###########################################################################
