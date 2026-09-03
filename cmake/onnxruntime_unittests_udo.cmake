# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

# Consumes the prebuilt QNN UDO ("MyAdd") test op-package fixture used by
# onnxruntime/test/providers/qnn/udo_op_test.cc. The fixture is built and validated out-of-band by
# qcom/scripts/linux/build_udo_test_package.py and published to Artifactory keyed on the pinned
# QAIRT SDK version (see qcom/packages.yml:qairt.version and
# .github/workflows/qualcomm-internal-publish-udo-package.yml) -- it is intentionally NOT built
# here anymore, since doing so required ~17 GB of LLVM + Hexagon SDK toolchain for a ~4 MB test
# fixture, and coupled the ORT build to QAIRT SDK internals (see PR #772).
#
# CI downloads the fixture into build/qnn-udo-test-package/ before configuring (see
# qualcomm-internal-build-and-test-single-os.yml); local developers without Artifactory access can
# run qcom/scripts/linux/build_udo_test_package.py directly, which stages into the same path.

if(NOT (UNIX AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND onnxruntime_target_platform STREQUAL "x86_64"))
    return()
endif()

if(DEFINED ENV{ORT_QNN_UDO_TEST_PACKAGE_DIR})
    set(_UDO_TEST_PACKAGE_DIR "$ENV{ORT_QNN_UDO_TEST_PACKAGE_DIR}")
else()
    # CMAKE_SOURCE_DIR for onnxruntime is <repo>/cmake, so its parent is the repo root.
    set(_UDO_TEST_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/../build/qnn-udo-test-package")
endif()

set(_UDO_CPU_SO "${_UDO_TEST_PACKAGE_DIR}/libMyAddOpPackage_cpu.so")
set(_UDO_HTP_SO "${_UDO_TEST_PACKAGE_DIR}/libMyAddOpPackage_htp.so")

if(NOT EXISTS "${_UDO_CPU_SO}" OR NOT EXISTS "${_UDO_HTP_SO}")
    message(STATUS "Skipping QNN UDO unit test build: prebuilt fixture not found at "
                    "${_UDO_TEST_PACKAGE_DIR} (run qcom/scripts/linux/build_udo_test_package.py, "
                    "or set ORT_QNN_UDO_TEST_PACKAGE_DIR).")
    return()
endif()

find_program(_UDO_NM_EXECUTABLE nm)
if(NOT _UDO_NM_EXECUTABLE)
    message(STATUS "Skipping QNN UDO unit test build: `nm` not found, cannot validate the "
                    "prebuilt fixture's ABI against the QAIRT SDK.")
    return()
endif()

# The authoritative ABI check (every mangled C++ symbol the fixture imports resolves against the
# QAIRT SDK's backend library or the host's C++ runtime libraries) runs once, before publishing,
# in qcom/scripts/linux/build_udo_test_package.py -- replicating it exactly here in CMake script
# would mean re-deriving which symbols are satisfied by libc++/libc++abi at configure time, which
# is fragile to get right and, if wrong, would silently disable these tests for every developer.
# This is a lighter, configure-time-only sanity check for the common local-dev mistake of building
# against a --qairt-sdk-root other than the one the fixture was published for: it confirms the
# QAIRT SDK actually ships each backend library, and that the fixture still exports its interface
# symbol (the .so is not corrupt/truncated). A genuine ABI mismatch will still surface as a clear
# dlopen() error inside the test itself rather than a silent skip.
function(_udo_check_interface_symbol so_path backend_lib_path)
    if(NOT EXISTS "${backend_lib_path}")
        message(STATUS "Skipping QNN UDO unit test build: ${backend_lib_path} not found under "
                        "QNN_HOME (is this QAIRT SDK missing the expected backend library?).")
        set(_UDO_CHECKS_OK FALSE PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND "${_UDO_NM_EXECUTABLE}" -D --defined-only "${so_path}"
        OUTPUT_VARIABLE _defined_raw
        ERROR_QUIET
    )
    string(FIND "${_defined_raw}" "MyAddOpPackageInterfaceProvider" _pos)
    if(_pos EQUAL -1)
        message(STATUS "Skipping QNN UDO unit test build: ${so_path} does not export "
                        "MyAddOpPackageInterfaceProvider -- rebuild it with "
                        "qcom/scripts/linux/build_udo_test_package.py.")
        set(_UDO_CHECKS_OK FALSE PARENT_SCOPE)
    endif()
endfunction()

set(_UDO_CHECKS_OK TRUE)
_udo_check_interface_symbol("${_UDO_CPU_SO}" "${onnxruntime_QNN_HOME}/lib/x86_64-linux-clang/libQnnCpu.so")
_udo_check_interface_symbol("${_UDO_HTP_SO}" "${onnxruntime_QNN_HOME}/lib/x86_64-linux-clang/libQnnHtp.so")
if(NOT _UDO_CHECKS_OK)
    return()
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
    COMMAND ${CMAKE_COMMAND} -E copy "${_UDO_CPU_SO}" ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
    DEPENDS "${_UDO_CPU_SO}"
)
add_custom_target(QnnUDO_MyAdd
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_cpu.so
)
list(APPEND onnxruntime_test_providers_dependencies QnnUDO_MyAdd)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
    COMMAND ${CMAKE_COMMAND} -E copy "${_UDO_HTP_SO}" ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
    DEPENDS "${_UDO_HTP_SO}"
)
add_custom_target(QnnUDO_MyAdd_HTP
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libMyAddOpPackage_htp.so
)
list(APPEND onnxruntime_test_providers_dependencies QnnUDO_MyAdd_HTP)

# Signal to the rest of the unit-test CMake (and the test source files) that the UDO library will
# actually be built and is available at runtime.
set(onnxruntime_BUILD_QNN_UDO_TEST ON)
