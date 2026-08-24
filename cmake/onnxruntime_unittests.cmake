# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)

# Exclude files based on CMake options.
function(filter_test_srcs test_srcs_var)
  set(excluded_path_prefixes)

  list(LENGTH excluded_path_prefixes num_excluded_path_prefixes)

  if("${num_excluded_path_prefixes}" GREATER "0")
    set(filtered_test_srcs)

    foreach(test_src ${${test_srcs_var}})
      set(is_excluded false)

      foreach(excluded_path_prefix ${excluded_path_prefixes})
        cmake_path(ABSOLUTE_PATH test_src OUTPUT_VARIABLE test_src_absolute)

        cmake_path(IS_PREFIX excluded_path_prefix ${test_src_absolute} NORMALIZE is_excluded)

        if (is_excluded)
          break()
        endif()
      endforeach()

      if(NOT is_excluded)
        list(APPEND filtered_test_srcs ${test_src})
      endif()
    endforeach()

    set(${test_srcs_var} ${filtered_test_srcs} PARENT_SCOPE)
  endif()
endfunction()

set(disabled_warnings)
function(AddTest)
  cmake_parse_arguments(_UT "DYN" "TARGET" "LIBS;SOURCES;DEPENDS;TEST_ARGS" ${ARGN})
  list(REMOVE_DUPLICATES _UT_SOURCES)

  filter_test_srcs(_UT_SOURCES)

  message(VERBOSE "AddTest() TARGET: ${_UT_TARGET}")
  message(VERBOSE "AddTest() SOURCES:")
  foreach(ut_src ${_UT_SOURCES})
    message(VERBOSE "  ${ut_src}")
  endforeach()

  onnxruntime_add_executable(${_UT_TARGET} ${_UT_SOURCES})

  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)

  if(_UT_LIBS)
    list(REMOVE_DUPLICATES _UT_LIBS)
  endif()

  source_group(TREE ${REPO_ROOT} FILES ${_UT_SOURCES})

  if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    #TODO: fix the warnings, they are dangerous
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4244>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4244>")
  endif()
  if (MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6330>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6330>")
    #Abseil has a lot of C4127/C4324 warnings.
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4127>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4127>")
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4324>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4324>")
  endif()

  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest")

  if (MSVC)
    # set VS debugger working directory to the test program's directory
    set_target_properties(${_UT_TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>)
  endif()

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)

  if(_UT_DYN)
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
            Threads::Threads)
    target_compile_definitions(${_UT_TARGET} PRIVATE -DUSE_ONNXRUNTIME_DLL)
  else()
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
  endif()

  onnxruntime_add_include_to_target(${_UT_TARGET} date::date flatbuffers::flatbuffers)

  if(MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  endif()

  if (WIN32)
    # include dbghelp in case tests throw an ORT exception, as that exception includes a stacktrace, which requires dbghelp.
    target_link_libraries(${_UT_TARGET} PRIVATE debug dbghelp)

    if (MSVC)
      # warning C6326: Potential comparison of a constant with another constant.
      # Lot of such things came from gtest
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
      # Raw new and delete. A lot of such things came from googletest.
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      # "Global initializer calls a non-constexpr function."
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
    endif()
    target_compile_options(${_UT_TARGET} PRIVATE ${disabled_warnings})
  else()
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
    if (${HAS_NOERROR})
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=uninitialized>")
    endif()
    if (${HAS_CHARACTER_CONVERSION})
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=character-conversion>")
    endif()
  endif()

  set(TEST_ARGS ${_UT_TEST_ARGS})
  if (onnxruntime_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      # WebAssembly use a memory file system, so we do not use full path
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<TARGET_FILE_NAME:${_UT_TARGET}>.$<CONFIG>.results.xml")
    else()
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
    endif()
  endif(onnxruntime_GENERATE_TEST_REPORTS)


    add_test(NAME ${_UT_TARGET}
    COMMAND ${_UT_TARGET} ${TEST_ARGS}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
    )
    # Set test timeout to 3 hours.
    set_tests_properties(${_UT_TARGET} PROPERTIES TIMEOUT 10800)

endfunction(AddTest)

# general program entrypoint for C++ unit tests
set(onnxruntime_unittest_main_src "${TEST_SRC_DIR}/unittest_main/test_main.cc")

#Do not add '${TEST_SRC_DIR}/util/include' to your include directories directly
#Use onnxruntime_add_include_to_target or target_link_libraries, so that compile definitions
#can propagate correctly.

file(GLOB onnxruntime_test_utils_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/util/include/*.h"
  "${TEST_SRC_DIR}/util/*.cc"
)

list(LENGTH onnxruntime_test_providers_src_patterns onnxruntime_test_providers_src_patterns_length)
if(onnxruntime_test_providers_src_patterns_length GREATER 0)
  file(GLOB onnxruntime_test_providers_src CONFIGURE_DEPENDS ${onnxruntime_test_providers_src_patterns})
else()
  set(onnxruntime_test_providers_src)
endif()

set(onnxruntime_test_internal_testing_ep_src)
if (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_EXTENDED_MINIMAL_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_internal_testing_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/internal_testing_ep/*"
    )
  list(APPEND onnxruntime_test_internal_testing_ep_src ${onnxruntime_test_providers_internal_testing_src})
endif()

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(onnxruntime_test_common_libs
  GTest::gtest
  GTest::gmock
  ${onnxruntime_EXTERNAL_LIBRARIES}
  onnxruntime
  onnxruntime_test_utils
  onnxruntime_unittest_utils
  cpuinfo::cpuinfo
  ${PROTOBUF_LIB}
)

set (onnxruntime_test_providers_dependencies ${onnxruntime_EXTERNAL_DEPENDENCIES})
set(onnxruntime_test_framework_src_patterns)
if(onnxruntime_USE_QNN AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/*)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/qnn_node_group/*)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/optimizer/*)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/ssr/qnn_ssr_test.cc)
  include(onnxruntime_unittests_udo.cmake)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/unit/*)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/integration/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_qnn)
  if(NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB)
    list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_shared)
  endif()
endif()

file(GLOB onnxruntime_test_framework_src CONFIGURE_DEPENDS
  ${onnxruntime_test_framework_src_patterns}
  )

# TODO: Re-enable the recent op testcases
list(REMOVE_ITEM onnxruntime_test_framework_src
     "${TEST_SRC_DIR}/providers/qnn/optimizer/transpose_optimizer_test.cc")

# qnn_op_tracing_serialization.cc is compiled directly into the test binary so
# that QnnFrameworkOpTraceUnit tests can call ComputeTraceSummary and
# SerializeFrameworkOpTrace without linking against the EP library.
# This translation unit intentionally does NOT include ort_api.h, avoiding
# the ORT_API_MANUAL_INIT mismatch linker error on Windows.  It is needed in
# both shared-lib builds (hidden symbol visibility prevents linking) and
# static-lib builds (test binary has build-order dependency only, not a linker
# dependency, on the EP).
if(onnxruntime_USE_QNN AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  list(APPEND onnxruntime_test_framework_src
       "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/op_tracing/qnn_op_tracing_serialization.cc")
endif()

#This is a small wrapper library that shouldn't use any onnxruntime internal symbols(except onnxruntime_common).
#Because it could dynamically link to onnxruntime. Otherwise you will have two copies of onnxruntime in the same
#process and you won't know which one you are testing.
onnxruntime_add_static_library(onnxruntime_test_utils ${onnxruntime_test_utils_src})
add_dependencies(onnxruntime_test_utils ort_core_target)
if(MSVC)
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
          "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
else()
  if (HAS_CHARACTER_CONVERSION)
    target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=character-conversion>")
  endif()
endif()
onnxruntime_add_include_to_target(onnxruntime_test_utils GTest::gtest GTest::gmock onnx onnx_proto
                                  flatbuffers::flatbuffers nlohmann_json::nlohmann_json Boost::mp11
                                  safeint_interface Eigen3::Eigen ${GSL_TARGET} date::date ${ABSEIL_LIBS})
add_dependencies(onnxruntime_test_utils ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_test_utils PUBLIC "${TEST_SRC_DIR}/util/include"
                           PRIVATE ${ONNXRUNTIME_APPLICATION_INCLUDES}
                           )
set_target_properties(onnxruntime_test_utils PROPERTIES FOLDER "ONNXRuntimeTest")
source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_test_utils_src})

# onnxruntime_unittest_utils
# This is static library containing utilities that are specifically for unit tests.
# Unlike onnxruntime_test_utils, the source files here may have dependencies on internal onnxruntime code.
# Thus, onnxruntime_unittest_utils is not suitable for use in programs that don't link with internal onnxruntime
# libraries.
block()

file(GLOB onnxruntime_unittest_utils_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/unittest_util/*.h"
    "${TEST_SRC_DIR}/unittest_util/*.cc")

onnxruntime_add_static_library(onnxruntime_unittest_utils ${onnxruntime_unittest_utils_src})
add_dependencies(onnxruntime_unittest_utils ort_core_target)

target_include_directories(onnxruntime_unittest_utils PRIVATE
                           ${ONNXRUNTIME_APPLICATION_INCLUDES}
                           "${TEST_SRC_DIR}/util/include"
                           )

target_link_libraries(onnxruntime_unittest_utils PUBLIC
                      onnx
                      GTest::gtest
                      GTest::gmock
                      ${ONNXRUNTIME_TEST_LIBS}
                      ${onnxruntime_EXTERNAL_LIBRARIES}
                      )

set_target_properties(onnxruntime_unittest_utils PROPERTIES FOLDER "ONNXRuntimeTest")

source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_unittest_utils_src})

endblock()

set(all_dependencies ${onnxruntime_test_providers_dependencies} )

if(WIN32)
  list(APPEND onnxruntime_test_providers_libs Advapi32)
endif()

if(WIN32)
  set(wide_get_opt_src_dir ${TEST_SRC_DIR}/win_getopt/wide)
  onnxruntime_add_static_library(win_getopt_wide ${wide_get_opt_src_dir}/getopt.cc ${wide_get_opt_src_dir}/include/getopt.h)
  target_include_directories(win_getopt_wide INTERFACE ${wide_get_opt_src_dir}/include)
  set_target_properties(win_getopt_wide PROPERTIES FOLDER "ONNXRuntimeTest")
  set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs})
  set(GETOPT_LIB_WIDE win_getopt_wide)
endif()

# onnxruntime_provider_test
# Execution provider-related tests.
# These also have some support for dynamically specified plugin EPs.
if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
block()
  list(APPEND onnxruntime_provider_test_srcs
    ${onnxruntime_unittest_main_src}
    ${onnxruntime_test_framework_src}
  )

  if(onnxruntime_USE_QNN)
    # Unlike other QNN tests that link against onnxruntime_providers_qnn.so via the provider
    # interface, the Genie implementation classes (GenieBackendManager, GenieApiLoader,
    # GenieNode) are compiled with hidden symbol visibility and are not exported from the
    # shared library. Re-compiling them directly here is the least-invasive way to make
    # those symbols accessible to genie_unit_test.cc without changing the EP's public ABI.
    # Note: genie_node_compute_info.cc is intentionally excluded because it depends on
    # qnn_utils.cc symbols that are not available to the test binary and none of the
    # unit tests in genie_unit_test.cc exercise GenieNodeComputeInfo.
    list(APPEND onnxruntime_provider_test_srcs
      ${ONNXRUNTIME_ROOT}/core/providers/qnn/genie/genie_api_loader.cc
      ${ONNXRUNTIME_ROOT}/core/providers/qnn/genie/genie_backend_manager.cc
      ${ONNXRUNTIME_ROOT}/core/providers/qnn/genie/genie_node.cc
      # Stub for OrtGetRuntimePath (defined in ort_api.cc, which is EP DLL only).
      ${ONNXRUNTIME_ROOT}/test/providers/qnn/genie_test_stubs.cc
      # ParseOpPackages is consumed by qnn_basic_test.cc; recompile here for the same reason
      # as the genie sources (the EP shared library is loaded via dlopen, not linked).
      ${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/op_package/op_package_parser.cc
    )
  endif()

  set(onnxruntime_provider_test_libs
    ${onnxruntime_test_providers_libs}
    ${onnxruntime_test_common_libs}
  )

  set(onnxruntime_provider_test_deps ${onnxruntime_test_providers_dependencies})

  AddTest(
    TARGET onnxruntime_provider_test
    SOURCES ${onnxruntime_provider_test_srcs}
    LIBS ${onnxruntime_provider_test_libs}
    DEPENDS ${onnxruntime_provider_test_deps}
  )

  # Expose QNN SDK headers to unit tests via an interface target
  add_library(qnn_sdk_headers_include INTERFACE)
  target_include_directories(qnn_sdk_headers_include INTERFACE
    ${onnxruntime_QNN_HOME}/include
    ${onnxruntime_QNN_HOME}/include/QNN)
  target_link_libraries(onnxruntime_provider_test PRIVATE qnn_sdk_headers_include)

  if(onnxruntime_USE_QNN)
    # genie_backend_manager.cc is recompiled here rather than linked from the EP DLL.
    # ort_api.h (included transitively) defines ORT_API_MANUAL_INIT, which must be
    # uniform across all TUs in a binary. Suppress the define so the genie TUs match
    # the rest of the test binary.
    target_compile_definitions(onnxruntime_provider_test PRIVATE ORT_UNIT_TEST_BUILD)
    if(onnxruntime_BUILD_QNN_UDO_TEST)
      target_compile_definitions(onnxruntime_provider_test PRIVATE BUILD_QNN_UDO_TEST)
    endif()
  endif()

  # Dependency on ORT Core public header files
  target_include_directories(onnxruntime_provider_test PRIVATE ${ONNXRUNTIME_APPLICATION_INCLUDES})

  add_custom_command(
    TARGET onnxruntime_provider_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${ort_core_SOURCE_DIR}/onnxruntime/test/testdata
    $<TARGET_FILE_DIR:onnxruntime_provider_test>/testdata
  )

  # Exclude test_dynamic_plugin_ep when using prebuilt ONNX Runtime
  # TODO: Evaluate whether we can enable test_dynamic_plugin_ep with public API
  # enable dynamic plugin EP usage
  # target_compile_definitions(onnxruntime_provider_test PRIVATE ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)

  # TODO fix shorten-64-to-32 warnings
  # there are some in builds where sizeof(size_t) != sizeof(int64_t), e.g., in 'ONNX Runtime Web CI Pipeline'
  if (HAS_SHORTEN_64_TO_32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    target_compile_options(onnxruntime_provider_test PRIVATE -Wno-error=shorten-64-to-32)
  endif()

  # Coverage build: link against the SHARED QNN EP library so tests can call
  # EP-internal functions directly. Coverage is recorded in the .so's .gcda files and
  # collected by lcov --directory <build_dir> (recursive search finds them automatically).
  if(ENABLE_COVERAGE AND UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    target_link_libraries(onnxruntime_provider_test PRIVATE onnxruntime_providers_qnn)
    # QNN_EP_INTERNAL_SYMBOL_ACCESS gates test code that depends on EP-internal symbols.
    # It tracks whether the test binary is link-time bound to the SHARED EP library
    # (i.e., the cmake conditions above hold), not whether any production source is
    # under #if. When the macro is off, the test bodies under unit/ compile to empty
    # translation units, so non-coverage builds do not see undefined references.
    # Today this is only enabled under ENABLE_COVERAGE; once the UT migration plan
    # stabilises, the gate can be widened to other CI build configurations without
    # touching the test code.
    target_compile_definitions(onnxruntime_provider_test PRIVATE QNN_EP_INTERNAL_SYMBOL_ACCESS=1)
  endif()

  if(onnxruntime_USE_QNN AND NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB AND WIN32)
    # ---------------------------------------------------------------------------
    # QnnMockSSR shared library — wraps QnnHtp.dll and injects a QNN_COMMON_ERROR_SYSTEM_COMMUNICATION
    # error (SSR) at a configurable call-site to exercise the SSR recovery paths in QNN EP.
    # ---------------------------------------------------------------------------
    add_library(QnnMockSSR SHARED
      ${TEST_SRC_DIR}/providers/qnn/ssr/qnn_mock_ssr.cc
      ${TEST_SRC_DIR}/providers/qnn/ssr/qnn_mock_ssr.def
    )

    target_include_directories(QnnMockSSR PRIVATE
      ${onnxruntime_QNN_HOME}/include/QNN
    )

    set_target_properties(QnnMockSSR PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      FOLDER "ONNXRuntimeTest"
    )

    # Copy QnnMockSSR next to the test executable so the test can load it by name.
    add_custom_command(
      TARGET QnnMockSSR POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:QnnMockSSR>
        $<TARGET_FILE_DIR:onnxruntime_provider_test>
      COMMENT "Copying QnnMockSSR to test output directory"
    )
  endif()
endblock()
endif()

  if(onnxruntime_USE_QNN)
    #qnn ctx generator
    set(ep_weight_sharing_ctx_gen_src_dir ${TEST_SRC_DIR}/ep_weight_sharing_ctx_gen)
    set(ep_weight_sharing_ctx_gen_src_patterns
    "${ep_weight_sharing_ctx_gen_src_dir}/*.cc"
    "${ep_weight_sharing_ctx_gen_src_dir}/*.h")

    file(GLOB ep_weight_sharing_ctx_gen_src CONFIGURE_DEPENDS
      ${ep_weight_sharing_ctx_gen_src_patterns}
      )
    onnxruntime_add_executable(ep_weight_sharing_ctx_gen ${ep_weight_sharing_ctx_gen_src})
    target_include_directories(ep_weight_sharing_ctx_gen PRIVATE ${ONNXRUNTIME_APPLICATION_INCLUDES})
    if (WIN32)
      target_compile_options(ep_weight_sharing_ctx_gen PRIVATE ${disabled_warnings})
      if (NOT DEFINED SYS_PATH_LIB)
        set(SYS_PATH_LIB shlwapi)
      endif()
    endif()

    if (onnxruntime_BUILD_SHARED_LIB)
      set(ep_weight_sharing_ctx_gen_libs onnxruntime ${onnxruntime_EXTERNAL_LIBRARIES} ${GETOPT_LIB_WIDE})
      target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE ${ep_weight_sharing_ctx_gen_libs})
      if (WIN32)
        target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE debug dbghelp advapi32)
      endif()
    else()
      target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE onnxruntime_session ${onnxruntime_test_providers_libs} ${onnxruntime_EXTERNAL_LIBRARIES} ${GETOPT_LIB_WIDE})
    endif()

    set_target_properties(ep_weight_sharing_ctx_gen PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()

  #some ETW tools
  if(WIN32 AND onnxruntime_ENABLE_INSTRUMENT)
    onnxruntime_add_executable(generate_perf_report_from_etl ${ONNXRUNTIME_ROOT}/tool/etw/main.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(generate_perf_report_from_etl PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(generate_perf_report_from_etl PRIVATE tdh Advapi32)

    onnxruntime_add_executable(compare_two_sessions ${ONNXRUNTIME_ROOT}/tool/etw/compare_two_sessions.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(compare_two_sessions PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(compare_two_sessions PRIVATE ${GETOPT_LIB_WIDE} tdh Advapi32)
  endif()
