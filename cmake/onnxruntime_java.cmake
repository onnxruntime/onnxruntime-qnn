# Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set(JAVA_ROOT ${REPO_ROOT}/java)
set(JAVA_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java)

set(GRADLE_EXECUTABLE "${JAVA_ROOT}/gradlew")

set(COMMON_GRADLE_ARGS --console=plain)
if(WIN32)
  list(APPEND COMMON_GRADLE_ARGS -Dorg.gradle.daemon=false)
elseif (ANDROID)
  # For Android build, we may run gradle multiple times in same build,
  # sometimes gradle JVM will run out of memory if we keep the daemon running
  # it is better to not keep a daemon running
  list(APPEND COMMON_GRADLE_ARGS --no-daemon)
endif()

if (ANDROID)
  set(JAVA_PACKAGE_OUTPUT_DIR ${JAVA_OUTPUT_DIR}/build)
  file(MAKE_DIRECTORY ${JAVA_PACKAGE_OUTPUT_DIR})
  set(ANDROID_PACKAGE_OUTPUT_DIR ${JAVA_PACKAGE_OUTPUT_DIR}/android)
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_OUTPUT_DIR})

  # onnxruntime-android-qnn AAR: contains libonnxruntime_providers_qnn.so + qnnpluginep Kotlin helpers
  if (onnxruntime_USE_QNN)
    set(ANDROID_QNN_EP_JNILIBS_DIR ${JAVA_OUTPUT_DIR}/android-qnn-ep)
    set(ANDROID_QNN_EP_ABI_DIR ${ANDROID_QNN_EP_JNILIBS_DIR}/${ANDROID_ABI})
    set(ANDROID_QNN_EP_OUTPUT_DIR ${JAVA_OUTPUT_DIR}/build/android-qnn-ep)
    file(MAKE_DIRECTORY ${ANDROID_QNN_EP_JNILIBS_DIR})
    file(MAKE_DIRECTORY ${ANDROID_QNN_EP_ABI_DIR})

    add_custom_command(TARGET onnxruntime_providers_qnn POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:onnxruntime_providers_qnn>
        ${ANDROID_QNN_EP_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime_providers_qnn>)

    # The Android Gradle Plugin's `maven-publish` plugin generates POM files at
    # ${rootBuildDir}/publications/<publication>/pom-default.xml, so we need the
    # root project's build tree to share the qnnpluginep subproject's output
    # directory for the copy_if_different below to find pom-default.xml.
    set(ANDROID_QNN_EP_POM_PATH ${ANDROID_QNN_EP_OUTPUT_DIR}/publications/qnnEp/pom-default.xml)

    # Generate a tiny cmake -P script into the build tree so we can do a
    # build-time if(NOT EXISTS) check without committing a new source file and
    # without relying on bash (which cmake/Ninja may re-wrap in /bin/sh -c,
    # causing compound-syntax errors).
    set(_check_pom_script "${CMAKE_CURRENT_BINARY_DIR}/check_pom_exists.cmake")
    file(WRITE "${_check_pom_script}"
      "if(NOT EXISTS \"${ANDROID_QNN_EP_POM_PATH}\")\n"
      "  message(FATAL_ERROR"
      " \"pom-default.xml was not generated: ${ANDROID_QNN_EP_POM_PATH}\\n\""
      " \"Please check if gradle executes generatePomFileForQnnEpPublication.\")\n"
      "endif()\n")

    add_custom_command(TARGET onnxruntime_providers_qnn POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo "Generating onnxruntime-android-qnn AAR..."
      COMMAND ${GRADLE_EXECUTABLE}
        ${COMMON_GRADLE_ARGS}
        :qnnpluginep:bundleReleaseAar generatePomFileForQnnEpPublication
        -b build-android.gradle -c settings-android.gradle
        -DminSdkVer=${ANDROID_MIN_SDK}
        -DqnnEpJniLibsDir=${ANDROID_QNN_EP_JNILIBS_DIR}
        -DqnnpluginepBuildDir=${ANDROID_QNN_EP_OUTPUT_DIR}
        -DrootBuildDir=${ANDROID_QNN_EP_OUTPUT_DIR}
        --stacktrace
      COMMAND ${CMAKE_COMMAND} -P "${_check_pom_script}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ANDROID_QNN_EP_POM_PATH}
        ${ANDROID_QNN_EP_OUTPUT_DIR}/outputs/aar/onnxruntime-android-qnn.pom
      WORKING_DIRECTORY ${JAVA_ROOT})
  endif()

  if (onnxruntime_BUILD_UNIT_TESTS)
    set(ANDROID_TEST_PACKAGE_ROOT ${JAVA_ROOT}/src/test/android)
    set(ANDROID_TEST_PACKAGE_DIR ${JAVA_OUTPUT_DIR}/androidtest/android)
    file(MAKE_DIRECTORY ${JAVA_OUTPUT_DIR}/androidtest)
    # Copy the test tree, excluding the assets directory (sigmoid.ort is sourced from ort_core at build time)
    file(COPY ${ANDROID_TEST_PACKAGE_ROOT} DESTINATION ${JAVA_OUTPUT_DIR}/androidtest
      PATTERN "assets" EXCLUDE)
    set(ANDROID_TEST_ASSETS_DIR ${ANDROID_TEST_PACKAGE_DIR}/app/src/androidTest/assets)
    file(MAKE_DIRECTORY ${ANDROID_TEST_ASSETS_DIR})
    set(ANDROID_TEST_PACKAGE_LIB_DIR ${ANDROID_TEST_PACKAGE_DIR}/app/libs)
    file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_LIB_DIR})

    if (onnxruntime_USE_QNN)
      add_custom_command(TARGET onnxruntime_providers_qnn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove -f
          ${ANDROID_TEST_PACKAGE_LIB_DIR}/onnxruntime-android-qnn.aar
        COMMAND ${CMAKE_COMMAND} -E copy
          ${ANDROID_QNN_EP_OUTPUT_DIR}/outputs/aar/onnxruntime-android-qnn.aar
          ${ANDROID_TEST_PACKAGE_LIB_DIR}/onnxruntime-android-qnn.aar)

      # Copy sigmoid.ort from ort_core source tree into the test assets directory
      add_custom_command(TARGET onnxruntime_providers_qnn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${ort_core_SOURCE_DIR}/java/src/test/android/app/src/androidTest/assets/sigmoid.ort
          ${ANDROID_TEST_ASSETS_DIR}/sigmoid.ort)

      add_custom_command(TARGET onnxruntime_providers_qnn POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Building Android test APK..."
        COMMAND ${GRADLE_EXECUTABLE}
          ${COMMON_GRADLE_ARGS}
          clean assembleDebug assembleDebugAndroidTest
          -DminSdkVer=${ANDROID_MIN_SDK}
          -DqnnVersion=${QNN_SDK_VERSION}
          -DortVersion=${ORT_CORE_VER}
          --stacktrace
        WORKING_DIRECTORY ${ANDROID_TEST_PACKAGE_DIR})
    endif()
  endif()
endif()
