# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_QNN=1)

  file(GLOB_RECURSE
       onnxruntime_providers_qnn_ep_srcs CONFIGURE_DEPENDS
       "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.h"
       "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.cc"
  )

  function(extract_qnn_sdk_version_from_yaml QNN_SDK_YAML_FILE QNN_VERSION_OUTPUT)
    file(READ "${QNN_SDK_YAML_FILE}" QNN_SDK_YAML_CONTENT)
    # Match a line of text like "version: 1.33.2"
    string(REGEX MATCH "(^|\n|\r)version: ([0-9]+\\.[0-9]+\\.[0-9]+)" QNN_VERSION_MATCH "${QNN_SDK_YAML_CONTENT}")
    if(QNN_VERSION_MATCH)
      set(${QNN_VERSION_OUTPUT} "${CMAKE_MATCH_2}" PARENT_SCOPE)
      message(STATUS "Extracted QNN SDK version ${CMAKE_MATCH_2} from ${QNN_SDK_YAML_FILE}")
    else()
      message(WARNING "Failed to extract QNN SDK version from ${QNN_SDK_YAML_FILE}")
    endif()
  endfunction()

  if(NOT QNN_SDK_VERSION)
    if(EXISTS "${onnxruntime_QNN_HOME}/sdk.yaml")
      extract_qnn_sdk_version_from_yaml("${onnxruntime_QNN_HOME}/sdk.yaml" QNN_SDK_VERSION)
    else()
      message(WARNING "Cannot open sdk.yaml to extract QNN SDK version")
    endif()
  endif()
  message(STATUS "QNN SDK version ${QNN_SDK_VERSION}")

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_qnn_ep_srcs})

  set(onnxruntime_providers_qnn_all_srcs ${onnxruntime_providers_qnn_ep_srcs})
  if(WIN32)
    # Sets the DLL version info on Windows: https://learn.microsoft.com/en-us/windows/win32/menurc/versioninfo-resource
    list(APPEND onnxruntime_providers_qnn_all_srcs "${ONNXRUNTIME_ROOT}/core/providers/qnn/onnxruntime_providers_qnn.rc")
  endif()

  onnxruntime_add_shared_library_module(onnxruntime_providers_qnn ${onnxruntime_providers_qnn_all_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_qnn ${GSL_TARGET} safeint_interface nlohmann_json::nlohmann_json)

  target_link_libraries(onnxruntime_providers_qnn PRIVATE ${ABSEIL_LIBS})

  add_dependencies(onnxruntime_providers_qnn ort_core_target)

  message(STATUS ONNXRUNTIME_APPLICATION_SOURCE_ROOT ${ONNXRUNTIME_APPLICATION_SOURCE_ROOT})
  target_include_directories(onnxruntime_providers_qnn PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                                               ${ONNXRUNTIME_APPLICATION_INCLUDE_ROOT}/core/session
                                                               ${onnxruntime_QNN_HOME}/include/QNN
                                                               ${onnxruntime_QNN_HOME}/include)

  # Set preprocessor definitions used in onnxruntime_providers_qnn.rc
  if(WIN32)
    if(NOT QNN_SDK_VERSION)
      set(QNN_DLL_FILE_DESCRIPTION "ONNX Runtime QNN Provider")
    else()
      set(QNN_DLL_FILE_DESCRIPTION "ONNX Runtime QNN Provider (QAIRT ${QNN_SDK_VERSION})")
    endif()

    target_compile_definitions(onnxruntime_providers_qnn PRIVATE FILE_DESC=\"${QNN_DLL_FILE_DESCRIPTION}\")
    target_compile_definitions(onnxruntime_providers_qnn PRIVATE FILE_NAME=\"onnxruntime_providers_qnn.dll\")
  endif()

  # Set linker flags for function(s) exported by EP DLL
  if(UNIX)
    target_link_options(onnxruntime_providers_qnn PRIVATE
                        "LINKER:--version-script=${ONNXRUNTIME_ROOT}/core/providers/qnn/version_script.lds"
                        "LINKER:--gc-sections"
                        "LINKER:-rpath=\$ORIGIN"
    )
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_qnn APPEND_STRING PROPERTY LINK_FLAGS
                  "-DEF:${ONNXRUNTIME_ROOT}/core/providers/qnn/symbols.def")
    # Generate PDB for Release builds.
    # /DEBUG tells the linker to emit a .pdb; /OPT:REF and /OPT:ICF re-enable
    # linker optimizations that /DEBUG implicitly turns off via /OPT:NOREF /OPT:NOICF.
    # Note: When developers use this PDB for crash analysis,
    # please be aware that ICF may cause symbol aliasing.
    target_link_options(onnxruntime_providers_qnn PRIVATE
      "$<$<CONFIG:Release>:/DEBUG;/OPT:REF;/OPT:ICF>"
    )
  else()
    message(FATAL_ERROR "onnxruntime_providers_qnn unknown platform, need to specify shared library exports for it")
  endif()

  # Set compile options
  if(MSVC)
    target_compile_options(onnxruntime_providers_qnn PUBLIC /wd4099 /wd4005 /wd4702)
  else()
    # ignore the warning unknown-pragmas on "pragma region"
    target_compile_options(onnxruntime_providers_qnn PRIVATE "-Wno-unknown-pragmas")
  endif()

  set_target_properties(onnxruntime_providers_qnn PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_qnn PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_qnn PROPERTIES FOLDER "ONNXRuntime")

  install(TARGETS onnxruntime_providers_qnn
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

  set(onnxruntime_providers_qnn_target onnxruntime_providers_qnn)

  if (MSVC OR ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    # Create destination directory first to ensure it exists
    add_custom_command(
      TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/onnxruntime_qnn
      COMMENT "Creating QNN library destination directory"
    )

    add_custom_command(
      TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${REPO_ROOT}/VERSION_NUMBER $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>
    )

    # Copy QNN library files with better error handling
    if(QNN_LIB_FILES)
      foreach(QNN_LIB_FILE ${QNN_LIB_FILES})
        add_custom_command(
          TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${QNN_LIB_FILE}" $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>
          COMMENT "Copying QNN library to Build Folder: ${QNN_LIB_FILE}"
        )
        add_custom_command(
          TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${QNN_LIB_FILE}" $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/onnxruntime_qnn
          COMMENT "Copying QNN library to onnxruntime_qnn for Python Wheel: ${QNN_LIB_FILE}"
        )
      endforeach()
    endif()

    # Copy onnxruntime_providers_qnn.dll to onnxruntime_qnn directory
    add_custom_command(
      TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${onnxruntime_providers_qnn_target}> $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/onnxruntime_qnn
      COMMENT "Copying onnxruntime_providers_qnn.dll to onnxruntime_qnn directory"
    )
  endif()
  # Create destination directory first to ensure it exists
  add_custom_command(
    TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/onnxruntime_qnn
    COMMENT "Creating QNN library destination directory"
  )
  # Copy version, license, README, and release notes files to output directory
  add_custom_command(
    TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
    # Copy to output directory, required for zip archive
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${REPO_ROOT}/VERSION_NUMBER
        ${REPO_ROOT}/ThirdPartyNotices.txt
        ${REPO_ROOT}/docs/Privacy.md
        ${REPO_ROOT}/LICENSE
        ${REPO_ROOT}/README.md
        ${REPO_ROOT}/docs/release-notes.md
        $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>
    # Copy to onnxruntime_qnn directory, required for python wheel
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${REPO_ROOT}/VERSION_NUMBER
        ${REPO_ROOT}/ThirdPartyNotices.txt
        ${REPO_ROOT}/docs/Privacy.md
        ${REPO_ROOT}/LICENSE
        ${ONNXRUNTIME_ROOT}/python/__init__.py
        $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/onnxruntime_qnn
    COMMENT "Copying version, license, README, and release notes files"
  )
  # Create document destination directory first to ensure it exists
  add_custom_command(
    TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/docs/execution_providers
        $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/docs/images
    COMMENT "Creating document destination directory"
  )
  # Copy documents to output docs directory to preserve the file structure and maintain document links
  add_custom_command(
    TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${REPO_ROOT}/docs/execution_providers/QNN-ExecutionProvider.md
        ${REPO_ROOT}/docs/execution_providers/build.md
        ${REPO_ROOT}/docs/execution_providers/development.md
        $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/docs/execution_providers
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${REPO_ROOT}/docs/images/qnn_ep_quant_workflow.png
        ${REPO_ROOT}/docs/images/quantization_mixed_precision_1.png
        ${REPO_ROOT}/docs/images/quantization_mixed_precision_2.png
        ${REPO_ROOT}/docs/images/PluginEP-final.png
        ${REPO_ROOT}/docs/images/Q-icon-rgb-blue.png
        ${REPO_ROOT}/docs/images/header.png
        $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/docs/images
  )
  if (EXISTS "${onnxruntime_QNN_HOME}/LICENSE.pdf")
    add_custom_command(
      TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
        # Copy to output directory, required for zip archive
        COMMAND ${CMAKE_COMMAND} -E copy "${onnxruntime_QNN_HOME}/LICENSE.pdf" $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/Qualcomm_LICENSE.pdf
        # Copy to onnxruntime_qnn directory, required for python wheel
        COMMAND ${CMAKE_COMMAND} -E copy "${onnxruntime_QNN_HOME}/LICENSE.pdf" $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/onnxruntime_qnn/Qualcomm_LICENSE.pdf
    )
  endif()
  if (EXISTS "${onnxruntime_QNN_HOME}/Qualcomm AI Hub Proprietary License.pdf")
    add_custom_command(
      TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${onnxruntime_QNN_HOME}/Qualcomm AI Hub Proprietary License.pdf" $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>/onnxruntime_qnn
      )
  endif()

  install(TARGETS onnxruntime_providers_qnn EXPORT ${PROJECT_NAME}Targets
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})

# Code Coverage Configuration
# Currently only supported on Linux with GCC.
# Future: extend to Android (aarch64 cross-compile) — requires ADB-based test execution,
# pulling .gcda files from device, and lcov path substitution for cross-compiled sources.
if(ENABLE_COVERAGE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(onnxruntime_providers_qnn PRIVATE
            --coverage
            # -g and -O0 are explicit so that debug info and no optimization are
            # always present regardless of the CMake build type.
            -g
            -O0
            -fprofile-abs-path
        )
        target_link_options(onnxruntime_providers_qnn PRIVATE --coverage)
    endif()
endif()
