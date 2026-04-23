/* Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 *
 * Coverage-only version script for libonnxruntime_providers_qnn.so.
 *
 * Used in place of version_script.lds when ENABLE_COVERAGE=1 is set at build time.
 * Exports ALL symbols so that onnxruntime_provider_test can call EP-internal
 * functions directly via target_link_libraries(), enabling function-level UT
 * coverage without requiring a static library build.
 *
 * DO NOT use in production or release builds — only for coverage instrumentation.
 */
VERS_1.0 {
  global:
    *;
};
