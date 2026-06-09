// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

// Internal C++ base class for all QNN EP node compute info objects.
// Provides a virtual destructor so derived objects stored as
// `OrtNodeComputeInfo*` can be safely deleted in `ReleaseNodeComputeInfosImpl`.
// The C-API base `OrtNodeComputeInfo` is a POD struct with no virtual
// destructor and cannot get one without breaking ABI.
struct QnnNodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~QnnNodeComputeInfoBase() = default;
};

}  // namespace onnxruntime
