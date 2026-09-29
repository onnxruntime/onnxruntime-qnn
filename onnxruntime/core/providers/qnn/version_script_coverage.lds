/* Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 *
 * WARNING: DO NOT use in production or release builds.
 * This script exports ALL symbols (global: *). It is selected automatically by
 * CMake only when ENABLE_COVERAGE=1 is set; never reference it manually.
 *
 * Coverage-only version script for libonnxruntime_providers_qnn.so.
 * Used in place of version_script.lds when ENABLE_COVERAGE=1 is set at build time.
 * Exports ALL symbols so that onnxruntime_provider_test can call EP-internal
 * functions directly via target_link_libraries(), enabling function-level UT
 * coverage without requiring a static library build.
 */
/* No local: section — intentional. This coverage script exports ALL symbols
 * (global: *) so the test binary can call any EP-internal function directly.
 * A local: * entry would be contradictory here. */
VERS_1.0 {
  global:
    *;
};
