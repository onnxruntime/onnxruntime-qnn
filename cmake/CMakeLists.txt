# Copyright (c) Microsoft Corporation. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.28)

# Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 OLD)

# Project
project(onnxruntime_qnn_abi C CXX ASM)

# Disable fast-math for Intel oneAPI compiler
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC-like")
    # Using icx-cl compiler driver with MSVC-like arguments
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:precise")
  else()
    # Using icpx compiler driver
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-fast-math")
  endif()
endif()

# Needed for Java
if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

include(CheckCXXCompilerFlag)
include(CMakeDependentOption)
include(GNUInstallDirs) # onnxruntime_providers_* require CMAKE_INSTALL_* variables

if (NOT CMAKE_CXX_STANDARD)
  # TODO: update this once all system adapt c++20
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_CXX_STANDARD 20)
  else()
    set(CMAKE_CXX_STANDARD 17)
  endif()
endif()

if (MSVC)
  #  Make sure Visual Studio sets __cplusplus macro correctly: https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# NOTE: POSITION INDEPENDENT CODE hurts performance, and it only make sense on POSIX systems
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Options
option(onnxruntime_USE_VCPKG "Build with the vcpkg package manager" OFF)
option(onnxruntime_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)
option(onnxruntime_GENERATE_TEST_REPORTS "Enable test report generation" OFF)
option(onnxruntime_ENABLE_STATIC_ANALYSIS "Enable static analysis" OFF)
option(onnxruntime_USE_CUSTOM_STATIC_ANALYSIS_RULES "Use a custom SDL Rule. It is mainly for our CI build" OFF)
option(onnxruntime_REDIRECT_STATIC_ANALYSIS_OUTPUTS_TO_FILE "Use a custom SDL Rule. It is mainly for our CI build" OFF)
# Enable it may cause LNK1169 error
option(onnxruntime_ENABLE_MEMLEAK_CHECKER "Experimental: Enable memory leak checker in Windows debug build" OFF)
option(onnxruntime_ENABLE_CONVSYMKERNELAVX2_SAT_CHECKER "Experimental: Enable ConvSymKernelAvx2 assembly saturation checker in build" OFF)

# TODO: Turn on the onnxruntime_BUILD_UNIT_TESTS
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" OFF)
option(onnxruntime_USE_PREINSTALLED_EIGEN "Use pre-installed EIGEN. Need to provide eigen_SOURCE_PATH if turn this on." OFF)

option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" OFF)

option(onnxruntime_ENABLE_LTO "Enable link time optimization" OFF)
option(onnxruntime_CROSS_COMPILING "Cross compiling onnx runtime" OFF)
option(onnxruntime_GCOV_COVERAGE "Compile with options necessary to run code coverage" OFF)
option(onnxruntime_DONT_VECTORIZE "Do not vectorize operations in Eigen" OFF)

option(onnxruntime_USE_FULL_PROTOBUF "Link to libprotobuf instead of libprotobuf-lite when this option is ON" OFF)
option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS "Dump debug information about node inputs and outputs when executing the model." OFF)
cmake_dependent_option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB "Build dump debug information about node inputs and outputs with support for sql database." OFF "onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS" OFF)

# When loading a delay loaded DLL, Windows searches the main EXE's folder first.
cmake_dependent_option(onnxruntime_ENABLE_DELAY_LOADING_WIN_DLLS "Delay load some of the dependent DLls that are part of the OS" ON "WIN32;NOT GDK_PLATFORM" OFF)
option(onnxruntime_ENABLE_INSTRUMENT "Enable Instrument with Event Tracing for Windows (ETW)" OFF)
option(onnxruntime_USE_TELEMETRY "Build with Telemetry" OFF)
cmake_dependent_option(onnxruntime_USE_MIMALLOC "Override new/delete and arena allocator with mimalloc" OFF "WIN32" OFF)

# Options related to reducing the binary size produced by the build
# XNNPACK EP requires the internal NHWC contrib ops to be available, so this option must be OFF when onnxruntime_USE_XNNPACK is ON
cmake_dependent_option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" OFF "NOT onnxruntime_USE_XNNPACK" OFF)
option(onnxruntime_DISABLE_ML_OPS "Disable traditional ML ops" OFF)
option(onnxruntime_DISABLE_SPARSE_TENSORS "Disable sparse tensors data types" OFF)
option(onnxruntime_DISABLE_OPTIONAL_TYPE "Disable optional type" OFF)
option(onnxruntime_DISABLE_FLOAT8_TYPES "Disable float 8 types" OFF)
option(onnxruntime_DISABLE_FLOAT4_TYPES "Disable float 4 types" OFF)
option(onnxruntime_MINIMAL_BUILD "Exclude as much as possible from the build. Support ORT format models. No support for ONNX format models." OFF)
option(onnxruntime_CLIENT_PACKAGE_BUILD "Enables default settings that are more appropriate for client/on-device workloads." OFF)
option(onnxruntime_DISABLE_RTTI "Disable RTTI" ON)
# For now onnxruntime_DISABLE_EXCEPTIONS will only work with onnxruntime_MINIMAL_BUILD, more changes (ONNX, non-CPU EP, ...) are required to run this standalone
cmake_dependent_option(onnxruntime_DISABLE_EXCEPTIONS "Disable exception handling. Requires onnxruntime_MINIMAL_BUILD currently." ON "onnxruntime_MINIMAL_BUILD" OFF)
# Even when onnxruntime_DISABLE_ABSEIL is ON, ONNX Runtime still needs to link to abseil.
option(onnxruntime_DISABLE_ABSEIL "Do not use Abseil data structures in ONNX Runtime source code. Redefine Inlined containers to STD containers." OFF)

option(onnxruntime_EXTENDED_MINIMAL_BUILD "onnxruntime_MINIMAL_BUILD with support for execution providers that compile kernels." OFF)
option(onnxruntime_MINIMAL_BUILD_CUSTOM_OPS "Add custom operator kernels support to a minimal build." OFF)
option(onnxruntime_REDUCED_OPS_BUILD "Reduced set of kernels are registered in build via modification of the kernel registration source files." OFF)
option(onnxruntime_DISABLE_EXTERNAL_INITIALIZERS "Don't allow models to load external data" OFF)

#A special option just for debugging and sanitize check. Please do not enable in option in retail builds.
#The option has no effect on Windows.
option(onnxruntime_USE_VALGRIND "Build with valgrind hacks" OFF)

# A special build option only used for gathering code coverage info
option(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE "Run model tests even in debug mode" OFF)

# options for security fuzzing
# build configuration for fuzz testing is in onnxruntime_fuzz_test.cmake
option(onnxruntime_FUZZ_TEST "Enable Fuzz testing" OFF)

# build separate library of schemas of (custom) ops used by ORT (for ONNX to MLIR translation)
option(onnxruntime_BUILD_OPSCHEMA_LIB "Build op schema library" ON)

set(ONNX_CUSTOM_PROTOC_EXECUTABLE "" CACHE STRING "Specify custom protoc executable to build ONNX")

# external transformer src path
option(onnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH "Path to external transformer src dir")

option(onnxruntime_ENABLE_CPUINFO "Enable cpuinfo" ON)

# dlpack support
cmake_dependent_option(onnxruntime_ENABLE_DLPACK "Enable dlpack" ON "onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_ATEN" OFF)
option(onnxruntime_USE_LOCK_FREE_QUEUE "Build with lock-free task queue for threadpool." OFF)
option(onnxruntime_FORCE_GENERIC_ALGORITHMS "Disable optimized arch-specific algorithms. Use only for testing and debugging generic algorithms." OFF)


if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 11.1)
  message(FATAL_ERROR  "GCC version must be greater than or equal to 11.1")
endif()

include(detect_onnxruntime_target_platform.cmake)


# Single output directory for all binaries
set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "Single output directory for all binaries.")

# Local mirror directory of cmake dependencies
set(REPO_ROOT ${PROJECT_SOURCE_DIR}/..)
set(onnxruntime_CMAKE_DEPS_MIRROR_DIR ${REPO_ROOT}/mirror CACHE PATH "Path to the local mirror of cmake dependencies")

function(set_msvc_c_cpp_compiler_warning_level warning_level)
  if (NOT "${warning_level}" MATCHES "^[0-4]$")
    message(FATAL_ERROR "Expected warning_level value of 0-4, got '${warning_level}'.")
  endif()

  if (MSVC)
    set(warning_flag "/W${warning_level}")
    get_property(opts DIRECTORY PROPERTY COMPILE_OPTIONS)
    # only match the generator expression added by this function
    list(FILTER opts
         EXCLUDE REGEX "^\\$<\\$<COMPILE_LANGUAGE:CXX,C>:/W[0-4]>$")
    list(APPEND opts "$<$<COMPILE_LANGUAGE:CXX,C>:${warning_flag}>")
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "${opts}")
  endif()
endfunction()


if(MSVC_Z7_OVERRIDE)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif(MSVC_Z7_OVERRIDE)

# set default MSVC warning level to 3 for external dependencies
set_msvc_c_cpp_compiler_warning_level(3)

# Fuzz test has only been tested with BUILD_SHARED_LIB option,
# using the MSVC compiler and on windows OS and clang/gcc compiler on Linux.
if (onnxruntime_FUZZ_TEST AND onnxruntime_BUILD_SHARED_LIB AND onnxruntime_USE_FULL_PROTOBUF)
  # Fuzz test library dependency, protobuf-mutator,
  # needs the onnx message to be compiled using "non-lite protobuf version"
  set(onnxruntime_FUZZ_ENABLED ON)
endif()

if (onnxruntime_USE_VALGRIND AND NOT WIN32)
  add_definitions(-DRE2_ON_VALGRIND=1)
endif()

if (onnxruntime_ENABLE_NVTX_PROFILE)
  add_definitions(-DENABLE_NVTX_PROFILE=1)
endif()

if (onnxruntime_ENABLE_BITCODE)
  if (NOT (CMAKE_SYSTEM_NAME STREQUAL "iOS"))
    message(FATAL_ERROR "iOS platform required for onnxruntime_ENABLE_BITCODE")
  endif()
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE YES)
  set(CMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE "bitcode")
else()
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)
endif()

if (onnxruntime_ENABLE_MEMORY_PROFILE)
  add_definitions(-DORT_MEMORY_PROFILE=1)
endif()

set(ONNX_ML 1)


if (NOT WIN32)
  #TODO: On Linux we may try https://github.com/microsoft/TraceLogging.git
  if (onnxruntime_ENABLE_INSTRUMENT)
    message(WARNING "Instrument is only supported on Windows now")
    set(onnxruntime_ENABLE_INSTRUMENT OFF)
  endif()
endif()

# 'extended' implies minimal.
if (onnxruntime_EXTENDED_MINIMAL_BUILD AND NOT onnxruntime_MINIMAL_BUILD)
  set(onnxruntime_MINIMAL_BUILD ON)
endif()

set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../onnxruntime)

include(adjust_global_compile_flags.cmake)

# We need to link with libatomic on systems that do not have built-in atomics, or
# don't have built-in support for 8 byte atomics
# Derived from https://github.com/protocolbuffers/protobuf/blob/master/cmake/CMakeLists.txt
set(onnxruntime_LINK_LIBATOMIC false)
# We don't need to link libatomic on iOS
if (NOT MSVC AND NOT (CMAKE_SYSTEM_NAME STREQUAL "iOS"))
  include(CheckCXXSourceCompiles)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} -std=c++11)
  check_cxx_source_compiles("
    #include <atomic>
    int main() {
      return std::atomic<int64_t>{};
    }
  " onnxruntime_HAVE_BUILTIN_ATOMICS)
  if (NOT onnxruntime_HAVE_BUILTIN_ATOMICS)
    set(onnxruntime_LINK_LIBATOMIC true)
  endif()
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endif()


find_package(Threads)
# On Windows we directly use Windows APIs to do the job
# Android NDK doesn't provide the iconv lib.
if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
  find_package(Iconv REQUIRED)
  set(ICONV_LIB Iconv::Iconv)
endif()

find_package(Patch)
if (CMAKE_HOST_WIN32 AND NOT Patch_FOUND)
    # work around CI machines missing patch from the git install by falling back to the binary in this repo.
    # replicate what happens in https://github.com/Kitware/CMake/blob/master/Modules/FindPatch.cmake but without
    # the hardcoded suffixes in the path to the patch binary.
    find_program(Patch_EXECUTABLE NAMES patch PATHS ${PROJECT_SOURCE_DIR}/external/git.Win32.2.41.03.patch)
    if(Patch_EXECUTABLE)
      set(Patch_FOUND 1)
      if (NOT TARGET Patch::patch)
        add_executable(Patch::patch IMPORTED)
        set_property(TARGET Patch::patch PROPERTY IMPORTED_LOCATION ${Patch_EXECUTABLE})
      endif()
    endif()
endif()
if(Patch_FOUND)
  message("Patch found: ${Patch_EXECUTABLE}")
endif()

if (CMAKE_CROSSCOMPILING)
  message("Doing crosscompiling")
endif()

#Need python to generate def file
if (onnxruntime_BUILD_SHARED_LIB)
  find_package(Python 3.10 COMPONENTS Interpreter)
endif()

#Dependencies begin
get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_INCLUDE_DIR ${REPO_ROOT}/include/onnxruntime)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
include(external/onnxruntime_external_deps.cmake)

set(ORT_WARNING_FLAGS)
if (WIN32)
    # class needs to have dll-interface to be used by clients
    list(APPEND ORT_WARNING_FLAGS "/wd4251")
    # issued by thrust nonstandard extension used: nameless struct/union
    list(APPEND ORT_WARNING_FLAGS "/wd4201")
    # structure was padded due to __declspec(align())
    list(APPEND ORT_WARNING_FLAGS "/wd4324")
    # warning C4800: Implicit conversion from 'X' to bool. Possible information loss
    if (onnxruntime_USE_OPENVINO OR onnxruntime_USE_OPENVINO_INTERFACE)
       list(APPEND ORT_WARNING_FLAGS "/wd4800")
    endif()
    # operator 'operator-name': deprecated between enumerations of different types
    list(APPEND ORT_WARNING_FLAGS "/wd5054")
    # Enable warning: data member 'member' will be initialized after data member 'member2' / base class 'base_class'
    list(APPEND ORT_WARNING_FLAGS "/w15038")

    # set linker flags to minimize the binary size.
    if (MSVC)
      foreach(type EXE STATIC SHARED)
        if (NOT type MATCHES STATIC)
          # The WinML internal toolchain does not allow link's "additional options" to contain optimization
          # flags (/OPT#); these are already specified using msbuild properties.
          if (NOT DEFINED onnxruntime_DISABLE_LINKER_OPT_FLAGS)
            set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO")
            set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO")
            set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /INCREMENTAL:NO")
          endif()
        endif()
        if (onnxruntime_ENABLE_LTO)
          set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /LTCG")
        endif()
      endforeach()
    endif()

else()
  # Check for compiler flags
  check_cxx_compiler_flag(-Wambiguous-reversed-operator HAS_AMBIGUOUS_REVERSED_OPERATOR)
  # -Winterference-size was added in GCC 13
  check_cxx_compiler_flag(-Winterference-size HAS_INTERFERENCE_SIZE)
  check_cxx_compiler_flag(-Warray-bounds HAS_ARRAY_BOUNDS)
  check_cxx_compiler_flag(-Wbitwise-instead-of-logical HAS_BITWISE_INSTEAD_OF_LOGICAL)
  check_cxx_compiler_flag(-Wcast-function-type HAS_CAST_FUNCTION_TYPE)
  check_cxx_compiler_flag(-Wcatch-value HAS_CATCH_VALUE)
  check_cxx_compiler_flag(-Wclass-memaccess HAS_CLASS_MEMACCESS)
  check_cxx_compiler_flag(-Wcharacter-conversion HAS_CHARACTER_CONVERSION)
  check_cxx_compiler_flag(-Wdangling-reference HAS_DANGLING_REFERENCE)
  check_cxx_compiler_flag(-Wdeprecated-anon-enum-enum-conversion HAS_DEPRECATED_ANON_ENUM_ENUM_CONVERSION)
  check_cxx_compiler_flag(-Wdeprecated-builtins HAS_DEPRECATED_BUILTINS)
  check_cxx_compiler_flag(-Wdeprecated-copy HAS_DEPRECATED_COPY)
  check_cxx_compiler_flag(-Wdeprecated-declarations HAS_DEPRECATED_DECLARATIONS)
  check_cxx_compiler_flag(-Wdeprecated-literal-operator HAS_DEPRECATED_LITERAL_OPERATOR)
  check_cxx_compiler_flag(-Wdeprecated-this-capture HAS_DEPRECATED_THIS_CAPTURE)
  check_cxx_compiler_flag(-Wenum-constexpr-conversion HAS_ENUM_CONSTEXPR_CONVERSION)
  check_cxx_compiler_flag(-Wformat-truncation HAS_FORMAT_TRUNCATION)
  check_cxx_compiler_flag(-Wignored-attributes HAS_IGNORED_ATTRIBUTES)
  check_cxx_compiler_flag(-Wmaybe-uninitialized HAS_MAYBE_UNINITIALIZED)
  check_cxx_compiler_flag(-Wmissing-braces HAS_MISSING_BRACES)
  check_cxx_compiler_flag(-Wnonnull-compare HAS_NONNULL_COMPARE)
  check_cxx_compiler_flag(-Wparentheses HAS_PARENTHESES)
  check_cxx_compiler_flag(-Wshorten-64-to-32 HAS_SHORTEN_64_TO_32)
  check_cxx_compiler_flag(-Wstrict-aliasing HAS_STRICT_ALIASING)
  check_nvcc_compiler_flag(-Wstrict-aliasing NVCC_HAS_STRICT_ALIASING)
  check_cxx_compiler_flag(-Wstringop-overflow HAS_STRINGOP_OVERFLOW)
  check_cxx_compiler_flag(-Wtautological-pointer-compare HAS_TAUTOLOGICAL_POINTER_COMPARE)
  check_cxx_compiler_flag(-Wundefined-var-template HAS_UNDEFINED_VAR_TEMPLATE)
  check_cxx_compiler_flag(-Wunused-but-set-parameter HAS_UNUSED_BUT_SET_PARAMETER)
  check_cxx_compiler_flag(-Wunused-but-set-variable HAS_UNUSED_BUT_SET_VARIABLE)
  check_cxx_compiler_flag(-Wunused-variable HAS_UNUSED_VARIABLE)
  check_cxx_compiler_flag(-Wuseless-cast HAS_USELESS_CAST)

  check_function_exists(reallocarray HAS_REALLOCARRAY)
  if (NOT APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_target_platform STREQUAL "aarch64")
   check_cxx_compiler_flag(-march=armv8.2-a+bf16 HAS_ARM64_BFLOAT16)
   if(NOT HAS_ARM64_BFLOAT16)
     message(FATAL_ERROR  "The compiler doesn't support BFLOAT16!!!")
   endif()
   check_cxx_compiler_flag(-march=armv8.2-a+fp16 HAS_ARM64_FLOAT16)
   if(NOT HAS_ARM64_FLOAT16)
     message(FATAL_ERROR  "The compiler doesn't support FLOAT16!!!")
   endif()
  endif()

  if (HAS_TAUTOLOGICAL_POINTER_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    list(APPEND ORT_WARNING_FLAGS -Wno-tautological-pointer-compare)
  endif()
  if (HAS_NONNULL_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    list(APPEND ORT_WARNING_FLAGS -Wno-nonnull-compare)
  endif()

  if (HAS_AMBIGUOUS_REVERSED_OPERATOR)
    list(APPEND ORT_WARNING_FLAGS -Wno-ambiguous-reversed-operator)
  endif()
  if (HAS_DEPRECATED_ANON_ENUM_ENUM_CONVERSION)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-anon-enum-enum-conversion)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-enum-enum-conversion)
  endif()
  if (HAS_UNDEFINED_VAR_TEMPLATE)
    list(APPEND ORT_WARNING_FLAGS -Wno-undefined-var-template)
  endif()
  if (HAS_DEPRECATED_BUILTINS)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-builtins)
  endif()
  if (HAS_DEPRECATED_LITERAL_OPERATOR)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-literal-operator)
  endif()
  #see:https://reviews.llvm.org/D131307
  #It was intended that the 'enum-constexpr-conversion' type warnings can not be silenced by -w
  if(HAS_ENUM_CONSTEXPR_CONVERSION AND NOT Protobuf_FOUND)
    if (TARGET libprotobuf)
      target_compile_options(libprotobuf PRIVATE "-Wno-enum-constexpr-conversion")
    endif()
    if (TARGET libprotobuf-lite)
      target_compile_options(libprotobuf-lite PRIVATE "-Wno-enum-constexpr-conversion")
    endif()
  endif()

  if(HAS_INTERFERENCE_SIZE)
    # The warning was from Eigen. Since we do not use Eigen in the public API, we can ignore it.
    # However, it also means that part of our code was compiled based on the build machine's hardware spec.
    # Specifically, the code was tuned based on the build machine's CPU cache line size, which can vary a lot.
    list(APPEND ORT_WARNING_FLAGS -Wno-interference-size)
  endif()
  # enable warning(s) that may not be on by default
  if (HAS_SHORTEN_64_TO_32)
    list(APPEND ORT_WARNING_FLAGS -Wshorten-64-to-32)
  endif()
endif()

include(external/onnxruntime_prebuilt.cmake)

# Generate onnxruntime_config.h
# TODO: Investigate whether we need the VERSION_NUMBER
file (STRINGS "${REPO_ROOT}/VERSION_NUMBER" ORT_VERSION)

set(ORT_PROVIDER_FLAGS)
list(APPEND ORT_PROVIDER_FLAGS -DUSE_QNN=1)

# Collect QNN_LIB_FILES from QNN SDK
if (NOT QNN_ARCH_ABI)
    string(TOLOWER ${onnxruntime_target_platform} GEN_PLATFORM)
    if(MSVC)
    message(STATUS "Building MSVC for architecture ${CMAKE_SYSTEM_PROCESSOR} with CMAKE_GENERATOR_PLATFORM as ${GEN_PLATFORM}")
    if (${GEN_PLATFORM} STREQUAL "arm64")
        set(QNN_ARCH_ABI aarch64-windows-msvc)
    elseif (${GEN_PLATFORM} STREQUAL "arm64ec")
        set(QNN_ARCH_ABI arm64x-windows-msvc)
    else()
        set(QNN_ARCH_ABI x86_64-windows-msvc)
    endif()
    else()
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
        set(QNN_ARCH_ABI aarch64-android-clang6.0)
    elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        if (${GEN_PLATFORM} STREQUAL "x86_64")
        set(QNN_ARCH_ABI x86_64-linux-clang)
        else()
        set(QNN_ARCH_ABI aarch64-android)
        endif()
    endif()
    endif()
endif()

if (MSVC OR ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    file(GLOB QNN_LIB_FILES LIST_DIRECTORIES false "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/libQnn*.so"
                                            "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/Qnn*.dll"
                                            "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/libHtpPrepare.so"
                                            "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/HtpPrepare.dll")
    if (${QNN_ARCH_ABI} STREQUAL "aarch64-windows-msvc" OR ${QNN_ARCH_ABI} STREQUAL "arm64x-windows-msvc")
    file(GLOB EXTRA_HTP_LIB LIST_DIRECTORIES false "${onnxruntime_QNN_HOME}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so"
                                            "${onnxruntime_QNN_HOME}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so"
                                            "${onnxruntime_QNN_HOME}/lib/hexagon-v73/unsigned/libqnnhtpv73.cat"
                                            "${onnxruntime_QNN_HOME}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"
                                            "${onnxruntime_QNN_HOME}/lib/hexagon-v81/unsigned/libqnnhtpv81.cat")
    list(APPEND QNN_LIB_FILES ${EXTRA_HTP_LIB})
    endif()
    message(STATUS "QNN lib files: " ${QNN_LIB_FILES})
endif()

if (onnxruntime_USE_LOCK_FREE_QUEUE)
    add_compile_definitions(USE_LOCK_FREE_QUEUE)
endif()

if (onnxruntime_FORCE_GENERIC_ALGORITHMS)
    add_compile_definitions(FORCE_GENERIC_ALGORITHMS)
endif()

function(onnxruntime_set_compile_flags target_name)
    if (CPUINFO_SUPPORTED)
      onnxruntime_add_include_to_target(${target_name} cpuinfo::cpuinfo)
    endif()
    if(onnxruntime_ENABLE_LAZY_TENSOR)
      target_compile_definitions(${target_name} PRIVATE ENABLE_LAZY_TENSOR)
    endif()
	  # Enable stream for all the non-minimal build
    if (NOT onnxruntime_MINIMAL_BUILD)
      target_compile_definitions(${target_name} PRIVATE ORT_ENABLE_STREAM)
    endif()
	  if (onnxruntime_REDUCED_OPS_BUILD)
      target_compile_definitions(${target_name} PRIVATE REDUCED_OPS_BUILD)
    endif()
	  if (onnxruntime_DISABLE_EXTERNAL_INITIALIZERS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_EXTERNAL_INITIALIZERS)
    endif()
    if(onnxruntime_DISABLE_ABSEIL)
      target_compile_definitions(${target_name} PRIVATE DISABLE_ABSEIL)
    endif()
    if(UNIX)
      target_compile_definitions(${target_name} PRIVATE PLATFORM_POSIX)
    endif()
    target_compile_definitions(${target_name} PRIVATE EIGEN_USE_THREADS)
    if (onnxruntime_DISABLE_CONTRIB_OPS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_CONTRIB_OPS)
    endif()

    if (onnxruntime_DISABLE_ML_OPS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_ML_OPS)
    endif()

    if (onnxruntime_DISABLE_SPARSE_TENSORS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_SPARSE_TENSORS)
    endif()

    if (onnxruntime_DISABLE_OPTIONAL_TYPE)
      target_compile_definitions(${target_name} PRIVATE DISABLE_OPTIONAL_TYPE)
    endif()

    if (onnxruntime_DISABLE_FLOAT8_TYPES)
      target_compile_definitions(${target_name} PRIVATE DISABLE_FLOAT8_TYPES)
    endif()

    if (onnxruntime_ENABLE_ATEN)
      target_compile_definitions(${target_name} PRIVATE ENABLE_ATEN)
    endif()
    # TODO: Narrow scope for Kleidiai compile
    if (onnxruntime_USE_KLEIDIAI)
      target_compile_definitions(${target_name} PRIVATE USE_KLEIDIAI)
    endif()

    set_target_properties(${target_name} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    if (onnxruntime_USE_CUDA)
      # Suppress a "conversion_function_not_usable" warning in gsl/span
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe \"--diag_suppress=conversion_function_not_usable\">")
      target_compile_definitions(${target_name} PRIVATE -DDISABLE_CUSPARSE_DEPRECATED)
    endif()
    if (MSVC)
      foreach(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY}>")
      endforeach()
      if(VCPKG_INSTALLED_DIR)
        cmake_path(NATIVE_PATH VCPKG_INSTALLED_DIR NORMALIZE VCPKG_INSTALLED_DIR_native)
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:SHELL:/external:I ${VCPKG_INSTALLED_DIR_native}>")
      endif()
      foreach(onnxruntime_external_lib IN LISTS onnxruntime_EXTERNAL_LIBRARIES)
        #TODO: the list contains cmake keywords like "debug". We should exclude them.
        if(TARGET ${onnxruntime_external_lib})
          get_target_property(onnxruntime_external_lib_include_dirs ${onnxruntime_external_lib} INTERFACE_INCLUDE_DIRECTORIES)
          foreach(onnxruntime_external_lib_include_dir IN LISTS onnxruntime_external_lib_include_dirs)
            if(onnxruntime_external_lib_include_dir MATCHES "^\\$")
              if(onnxruntime_external_lib_include_dir MATCHES "^\\$<BUILD_INTERFACE:([^>]+)>$")
                string(REGEX REPLACE "^\\$<BUILD_INTERFACE:([^>]+)>$" "\\1" onnxruntime_external_lib_include_dir_cmake "${onnxruntime_external_lib_include_dir}")
                cmake_path(NATIVE_PATH onnxruntime_external_lib_include_dir_cmake NORMALIZE onnxruntime_external_lib_include_dir_native)
                target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${onnxruntime_external_lib_include_dir_native}>")
              endif()
            else()
              cmake_path(NATIVE_PATH onnxruntime_external_lib_include_dir NORMALIZE onnxruntime_external_lib_include_dir_native)
              target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${onnxruntime_external_lib_include_dir_native}>")
            endif()
          endforeach()
        endif()
      endforeach()
      target_compile_definitions(${target_name} PRIVATE -DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
      if (onnxruntime_ENABLE_MEMLEAK_CHECKER)
        target_compile_definitions(${target_name} PRIVATE -DONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
      endif()
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<COMPILE_LANGUAGE:CXX,C>:/utf-8>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /sdl>" "$<$<COMPILE_LANGUAGE:CXX,C>:/sdl>")
      set_target_properties(${target_name}
                      PROPERTIES VS_GLOBAL_CAExcludePath "${ORT_BINARY_DIR};${ORT_SOURCE_DIR}")
      # We do not treat warnings from 3rd-party libraries as errors. In order to do that, we need to add their header files locations to /external:I.
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/experimental:external>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /experimental:external>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:W0>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:W0>")
      #if(NOT onnxruntime_USE_VCPKG)
      #  target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:templates->" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:templates->")
      #endif()
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CURRENT_SOURCE_DIR}>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:I${CMAKE_CURRENT_SOURCE_DIR}>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CURRENT_BINARY_DIR}>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:I${CMAKE_CURRENT_BINARY_DIR}>")
      if (onnxruntime_ENABLE_STATIC_ANALYSIS)
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze>" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze>")
        if (onnxruntime_REDIRECT_STATIC_ANALYSIS_OUTPUTS_TO_FILE)
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze:autolog:ext.sarif>" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze:autolog:ext.sarif>")
        endif()
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze:external->" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze:external->")
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6385>" )
        # There are many such warnings from STL:
        # include\list(148): warning C6011: Dereferencing NULL pointer '_Mycont'. : Lines: 146, 147, 148
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6011>" )
      endif()
    else()
      # Enable warning
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wall>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall>")
      target_compile_options(${target_name} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wextra>")
      if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IBMClang")
        #external/protobuf/src/google/protobuf/arena.h:445:18: error: unused parameter 'p'
        target_compile_options(${target_name} PRIVATE "-Wno-unused-parameter")
      endif()
      if (CMAKE_CXX_COMPILER_ID STREQUAL "IBMClang")
        target_compile_options(${target_name} PRIVATE "-Wno-unused-function")
      endif()
    endif()
    foreach(ORT_FLAG ${ORT_PROVIDER_FLAGS})
      target_compile_definitions(${target_name} PRIVATE ${ORT_FLAG})
    endforeach()

    if (HAS_DEPRECATED_COPY)
      #too many such errors in eigen
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-deprecated-copy>" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>")
    endif()
    foreach(FLAG ${ORT_WARNING_FLAGS})
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${FLAG}>")
    endforeach()
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 13 AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-maybe-uninitialized>")
    endif()
    if (onnxruntime_USE_CUDA)
      foreach(FLAG ${ORT_WARNING_FLAGS})
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options ${FLAG}>")
      endforeach()
      if (NVCC_HAS_STRICT_ALIASING AND "${target_name}" MATCHES "cuda")
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-strict-aliasing>")
      endif()
      if (HAS_STRICT_ALIASING AND NOT "${target_name}" MATCHES "cuda")
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-strict-aliasing>")
      endif()
    endif()
endfunction()

function(onnxruntime_set_source_file_properties target_name)
  get_target_property(srcs ${target_name} SOURCES)

  # enable ARC for Objective-C/C++
  set(objective_c_cc_srcs ${srcs})
  list(FILTER objective_c_cc_srcs INCLUDE REGEX "\\.mm?$")
  set_property(SOURCE ${objective_c_cc_srcs} APPEND PROPERTY COMPILE_OPTIONS "-fobjc-arc")
endfunction()

function(onnxruntime_configure_target target_name)
  target_link_directories(${target_name} PRIVATE ${onnxruntime_LINK_DIRS})
  onnxruntime_set_compile_flags(${target_name})
  onnxruntime_set_source_file_properties(${target_name})
  if(WIN32 AND onnxruntime_ENABLE_STATIC_ANALYSIS AND onnxruntime_USE_CUSTOM_STATIC_ANALYSIS_RULES)
    set_target_properties(${target_name} PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
  endif()
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
  if (onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(${target_name} PRIVATE ${ORTTRAINING_ROOT})
  endif()
  if (onnxruntime_ENABLE_LTO)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
  endif()

  # Keep BinSkim happy
  if(MSVC AND NOT onnxruntime_target_platform MATCHES "ARM")
    target_link_options(${target_name} PRIVATE "/CETCOMPAT")
  endif()

endfunction()

function(onnxruntime_add_shared_library target_name)
  add_library(${target_name} SHARED ${ARGN})
  onnxruntime_configure_target(${target_name})
  if(WIN32)
        target_compile_definitions(${target_name} PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
        target_compile_definitions(${target_name} PRIVATE VER_MINOR=${VERSION_MINOR_PART})
        target_compile_definitions(${target_name} PRIVATE VER_BUILD=${VERSION_BUILD_PART})
        target_compile_definitions(${target_name} PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
        target_compile_definitions(${target_name} PRIVATE VER_STRING=\"${VERSION_STRING}\")
  endif()
endfunction()

function(onnxruntime_add_static_library target_name)
  add_library(${target_name} STATIC ${ARGN})
  onnxruntime_configure_target(${target_name})
endfunction()

#For plugins that are not linked into other targets but may be loaded dynamically at runtime using dlopen-like functionality.
function(onnxruntime_add_shared_library_module target_name)
  if ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin") OR (${CMAKE_SYSTEM_NAME} MATCHES "iOS"))
    add_library(${target_name} SHARED ${ARGN})
  else()
    #On Windows, this target shouldn't generate an import lib, but I don't know how to disable it.
    add_library(${target_name} MODULE ${ARGN})
    if(WIN32)
        target_compile_definitions(${target_name} PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
        target_compile_definitions(${target_name} PRIVATE VER_MINOR=${VERSION_MINOR_PART})
        target_compile_definitions(${target_name} PRIVATE VER_BUILD=${VERSION_BUILD_PART})
        target_compile_definitions(${target_name} PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
        target_compile_definitions(${target_name} PRIVATE VER_STRING=\"${VERSION_STRING}\")
    endif()
  endif()

  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_object_library target_name)
  add_library(${target_name} OBJECT ${ARGN})

  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_executable target_name)
  add_executable(${target_name} ${ARGN})
  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
      if(TARGET ${src_target})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
        target_sources(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_SOURCES>)
      endif()
    endforeach()
endfunction()

#Adjust warning flags
set_msvc_c_cpp_compiler_warning_level(4)

set(ORT_BUILD_INFO "ORT Build Info: ")
find_package(Git)
if (Git_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE ORT_GIT_COMMIT)
  string(STRIP "${ORT_GIT_COMMIT}" ORT_GIT_COMMIT)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE ORT_GIT_BRANCH)
  string(STRIP "${ORT_GIT_BRANCH}" ORT_GIT_BRANCH)
  string(APPEND ORT_BUILD_INFO "git-branch=${ORT_GIT_BRANCH}, git-commit-id=${ORT_GIT_COMMIT}, ")
endif()
string(APPEND ORT_BUILD_INFO "build type=${CMAKE_BUILD_TYPE}")
configure_file(onnxruntime_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_config.h)
get_property(onnxruntime_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)


if (onnxruntime_ENABLE_DLPACK)
  add_compile_definitions(ENABLE_DLPACK)
endif()

if (onnxruntime_CALLER_FRAMEWORK)
  add_definitions(-DORT_CALLER_FRAMEWORK="${onnxruntime_CALLER_FRAMEWORK}")
endif()


# Required by core/providers/qnn-abi/onnxruntime_providers_qnn_abi.rc
set(VERSION_MAJOR_PART   0 CACHE STRING "First part of numeric file/product version.")
set(VERSION_MINOR_PART   0 CACHE STRING "Second part of numeric file/product version.")
set(VERSION_BUILD_PART       0 CACHE STRING "Third part of numeric file/product version.")
set(VERSION_PRIVATE_PART     0 CACHE STRING "Fourth part of numeric file/product version.")
set(VERSION_STRING       "Internal Build" CACHE STRING "String representation of file/product version.")
if(VERSION_MAJOR_PART STREQUAL "0" AND VERSION_MINOR_PART STREQUAL "0" AND VERSION_BUILD_PART STREQUAL "0" AND VERSION_PRIVATE_PART STREQUAL "0")
    string(REPLACE "." ";"  ORT_VERSION_STRING_LIST ${ORT_VERSION})
    list(GET ORT_VERSION_STRING_LIST 0 VERSION_MAJOR_PART)
    list(GET ORT_VERSION_STRING_LIST 1 VERSION_MINOR_PART)
    list(GET ORT_VERSION_STRING_LIST 2 VERSION_BUILD_PART)
    set(VERSION_STRING ${ORT_VERSION})
endif()


if (WIN32)
  if (onnxruntime_USE_MIMALLOC)
   list(APPEND onnxruntime_EXTERNAL_LIBRARIES mimalloc-static)
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${SYS_PATH_LIB})
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES debug Dbghelp)
  # In a onecore build the umbrella libs already contains references to the APIs in advapi32, so in onecore build we do not need to link to advapi32
  # In a non-onecore build, usually we also do not need to link to advapi32 because VC++ by default should have provide everything we need, except when the build target is Windows ARM32.
  # In the future we will add a build option to allow users disabling all API uses from advapi32 because some Windows environments do not have these APIs. For example, some Windows do not have
  # Windows Registry so we cannot query Registry values.
  if(onnxruntime_target_platform STREQUAL "ARM" AND CMAKE_CXX_STANDARD_LIBRARIES MATCHES kernel32.lib)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES advapi32)
  endif()
else()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${ICONV_LIB} ${CMAKE_DL_LIBS} Threads::Threads)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES log)
endif()

# check if we need to link against librt on Linux
include(CheckLibraryExists)
include(CheckFunctionExists)
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)

  if (NOT HAVE_CLOCK_GETTIME)
    set(CMAKE_EXTRA_INCLUDE_FILES time.h)
    check_function_exists(clock_gettime HAVE_CLOCK_GETTIME)
    set(CMAKE_EXTRA_INCLUDE_FILES)
  else()
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES rt)
  endif()
endif()

# From this point on, onnxruntime_EXTERNAL_LIBRARIES should be no longer be modified. It will be used in
# onnxruntime.cmake which will be included next.

if (onnxruntime_BUILD_UNIT_TESTS)
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_unittests)
endif()

# When GDK_PLATFORM is set then WINAPI_FAMILY is defined in gdk_toolchain.cmake (along with other relevant flags/definitions).
if (WIN32 AND NOT GDK_PLATFORM AND NOT CMAKE_CROSSCOMPILING)
  if (NOT CMAKE_CXX_STANDARD_LIBRARIES MATCHES kernel32.lib)
    # On onecore, link to the onecore build of the MSVC runtime
    get_filename_component(msvc_path "${CMAKE_C_COMPILER}/../../../.." ABSOLUTE)
    link_directories(BEFORE "${msvc_path}/lib/onecore/${onnxruntime_target_platform}")
    # The .lib files in the MSVC runtime have a DEFAULITLIB entry for onecore.lib, but it shold not cause any conflict with onecoreuap.lib
  endif()
endif()

include(onnxruntime_providers_qnn_abi.cmake)
# TODO: Enable the onnxruntime_providers_qnn_abi_simulation.cmake if needed
# include(onnxruntime_providers_qnn_abi_simulation.cmake)

if (UNIX)
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" ON)
else()
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" OFF)
endif()
if (BUILD_PKGCONFIG_FILES)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.pc.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc @ONLY)
  install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/pkgconfig )
endif()

if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  add_compile_definitions(DEBUG_NODE_INPUTS_OUTPUTS)
endif()

# Install
include(CMakePackageConfigHelpers)

set(PROJECT_CONFIG_CONTENT "@PACKAGE_INIT@\n")

if (NOT onnxruntime_BUILD_SHARED_LIB)
    string(APPEND PROJECT_CONFIG_CONTENT
        "include(CMakeFindDependencyMacro)\n\
        find_dependency(absl)\n\
        find_dependency(date)\n\
        find_dependency(Eigen3)\n\
        find_dependency(nlohmann_json)\n\
        find_dependency(ONNX)\n\
        find_dependency(re2)\n\
        find_dependency(flatbuffers)\n\
        find_dependency(cpuinfo)\n\
        find_dependency(protobuf)\n\
        find_dependency(Boost COMPONENTS mp11)\n\
        find_dependency(Microsoft.GSL 4.0)\n\
        if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL \"Android\")\n\
        find_dependency(Iconv)\n\
        endif()\n\
        if(WIN32)\n\
        find_dependency(wil)\n\
        endif()\n\
        find_path(safeint_SOURCE_DIR NAMES \"SafeInt.hpp\" REQUIRED)\n\
        add_library(safeint_interface IMPORTED INTERFACE)\n\
        target_include_directories(safeint_interface INTERFACE ${safeint_SOURCE_DIR})\n\
        ")
endif()

string(APPEND PROJECT_CONFIG_CONTENT
       "include(\"\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Targets.cmake\")\n")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/PROJECT_CONFIG_FILE" ${PROJECT_CONFIG_CONTENT})

# TODO: Investigate whether we need the install
# install(EXPORT ${PROJECT_NAME}Targets
#         NAMESPACE ${PROJECT_NAME}::
#         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Create config for find_package()
configure_package_config_file(
    "${CMAKE_CURRENT_BINARY_DIR}/PROJECT_CONFIG_FILE" ${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${ORT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(
    FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

if(DEFINED BUILD_AS_ARM64X)
    set(ARM64X_TARGETS onnxruntime)

    # Add additional ARM64X build targets
    list(APPEND ARM64X_TARGETS onnxruntime_providers_qnn_abi)

    include("${CMAKE_CURRENT_SOURCE_DIR}/arm64x.cmake")
endif()

if(onnxruntime_BUILD_UNIT_TESTS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_test_pch.cmake")
endif()

# Include precompiled header configuration for providers
if(TARGET onnxruntime_providers)
  include("${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_providers_pch.cmake")
endif()
