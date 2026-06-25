// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_package/op_package.h"

// Forward declaration: callers must have Ort::Logger fully defined at the call site
// (test binaries include "core/session/onnxruntime_cxx_api.h"; the EP includes it via
// "core/providers/qnn/ort_api.h"). Pulling either of those into this header would drag
// in heavy and conflicting transitive includes for the test binary.
namespace Ort {
struct Logger;
}

namespace onnxruntime {

// Parses the `op_packages` provider option string of the form
// "<OpType>:<PackagePath>:<InterfaceSymbolName>[:<Target>][,...]" and appends each well-formed
// entry to `op_packages`. Malformed entries are logged and skipped.
//
// Exposed in a header (rather than kept static) so unit tests can exercise it directly,
// notably the Windows drive-letter merge in qnn_basic_test.cc.
void ParseOpPackages(const std::string& op_packages_string,
                     std::vector<onnxruntime::qnn::OpPackage>& op_packages,
                     const Ort::Logger& logger);

}  // namespace onnxruntime
