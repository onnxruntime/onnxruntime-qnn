// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Stub implementations for EP-internal symbols required when genie source files
// are recompiled directly into the unit test binary. The real implementations
// live in ort_api.cc which is part of the EP DLL, not the test binary.

#include <string>

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

std::basic_string<ORTCHAR_T> OrtGetRuntimePath() {
  return ORT_TSTR("");
}

}  // namespace onnxruntime
