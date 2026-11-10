# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Determine the ORT Core version and (for source builds) fetch the ORT Core source.
#
# When a prebuilt ORT is supplied (onnxruntime_ORT_HOME set), the build must use ONLY that
# prebuilt: its version comes from the prebuilt's VERSION_NUMBER file, and the upstream ORT
# source is NOT fetched or patched. Otherwise (source build) the version is parsed from the
# pinned deps.txt URL and the ort_core source is fetched and patched as before.
if(onnxruntime_ORT_HOME)
  file(STRINGS "${PROJECT_SOURCE_DIR}/requirements.txt" _ort_python_requirement
       REGEX "^onnxruntime *>=")
  if(NOT _ort_python_requirement MATCHES ">= *([0-9]+\\.[0-9]+\\.[0-9]+)")
    message(FATAL_ERROR
      "Could not determine the minimum supported ORT version from requirements.txt.")
  endif()
  set(_ort_minimum_version "${CMAKE_MATCH_1}")

  # Pin 1: read ORT_CORE_VER from the prebuilt instead of from the deps.txt URL.
  set(_ort_version_file "${onnxruntime_ORT_HOME}/VERSION_NUMBER")
  if(NOT EXISTS "${_ort_version_file}")
    message(FATAL_ERROR "Prebuilt ORT version file not found: ${_ort_version_file}. "
                        "Expected a VERSION_NUMBER file at the root of onnxruntime_ORT_HOME.")
  endif()
  file(READ "${_ort_version_file}" ORT_CORE_VER)
  string(STRIP "${ORT_CORE_VER}" ORT_CORE_VER)
  if(NOT ORT_CORE_VER MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+([.-][0-9A-Za-z.-]+)?$")
    message(FATAL_ERROR
      "Invalid prebuilt ORT version '${ORT_CORE_VER}' in ${_ort_version_file}. "
      "Expected a semantic version such as 1.24.4.")
  endif()
  if(ORT_CORE_VER VERSION_LESS "${_ort_minimum_version}")
    message(FATAL_ERROR
      "Prebuilt ORT ${ORT_CORE_VER} is unsupported. QNN EP requires ORT ${_ort_minimum_version} or newer.")
  endif()
  message(STATUS "Using prebuilt ORT_CORE_VER from ${_ort_version_file}: ${ORT_CORE_VER}")
  # Pin 2: do NOT fetch or patch the ort_core source in prebuilt mode.
  message(STATUS "onnxruntime_ORT_HOME is set; skipping ort_core source fetch.")
else()
  # Extract version from ort_core URL
  # Expected URL format: https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.24.1.zip
  if(DEP_URL_ort_core MATCHES ".*refs/tags/v([0-9]+\\.[0-9]+\\.[0-9]+)\\.zip$")
    set(ORT_CORE_VER ${CMAKE_MATCH_1})
    message(STATUS "Extracted ORT_CORE_VER: ${ORT_CORE_VER}")
  else()
    message(FATAL_ERROR "Could not extract version from DEP_URL_ort_core: ${DEP_URL_ort_core}")
  endif()

  onnxruntime_fetchcontent_declare(
    ort_core
    URL ${DEP_URL_ort_core}
    URL_HASH SHA1=${DEP_SHA1_ort_core}
    PATCH_COMMAND
      ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/patches/ort_core/0001-cpp-model-test-runner-uses-plugin-EP.patch &&
      ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/patches/ort_core/0002-Allow-users-to-specify-arm64ReproDir-by-cmake-flag.patch &&
      ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/patches/ort_core/0004-deduplicate-autoep-custom-op-domains.patch # TODO: Remove this once the ORT core version we used includes the fix (i.e) 1.31.0.
    EXCLUDE_FROM_ALL)
  FetchContent_Populate(ort_core)
endif()

