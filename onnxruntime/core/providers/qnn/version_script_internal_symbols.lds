/* Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 *
 * WARNING: DO NOT use this linker script for production artifacts.
 * This script exports ALL symbols (global: *). It is selected automatically by
 * CMake for QNN EP internal-symbol test builds; never reference it manually.
 *
 * Internal-symbol test version script for libonnxruntime_providers_qnn.so.
 * Used in place of version_script.lds when internal-symbol access is enabled.
 * Exports ALL symbols so that onnxruntime_provider_test can call EP-internal
 * functions directly via target_link_libraries() without requiring a static
 * library build.
 */
/* No local: section — intentional. This test script exports ALL symbols
 * (global: *) so the test binary can call any EP-internal function directly.
 * A local: * entry would be contradictory here. */
VERS_1.0 {
  global:
    *;
};
