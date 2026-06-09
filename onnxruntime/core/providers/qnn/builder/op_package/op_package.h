// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace onnxruntime {
namespace qnn {

// Description of a UDO (user-defined op) package to register with the QNN backend.
// Defined in its own tiny header so consumers (including unit tests) do not have to pull
// in qnn_backend_manager.h and its transitive QNN-EP-private "common" headers, which
// redefine ORT types (SafeIntExceptionHandler, NodeHashSet, ...) and conflict with the
// public ORT headers a test binary already includes.
struct OpPackage {
  std::string op_type;
  std::string path;
  std::string interface;
  std::string target;
};

}  // namespace qnn
}  // namespace onnxruntime
