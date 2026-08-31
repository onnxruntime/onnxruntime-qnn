// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

// Forward declaration: callers must have Ort::Logger fully defined at the call site
// (the EP includes it via "core/providers/qnn/ort_api.h"; test binaries include it via
// "core/session/onnxruntime_cxx_api.h"). Pulling either of those into this header would
// drag in heavy and conflicting transitive includes for the test binary.
namespace Ort {
struct Logger;
}

namespace onnxruntime {

// Describes one domain entry parsed from ORT_QNN_CUSTOM_OP_DOMAINS.
struct CustomOpDomainSpec {
  std::string domain;
  std::vector<std::string> op_types;
};

// Parses the ORT_QNN_CUSTOM_OP_DOMAINS env-var string of the form
//   domain_name:OpType1[,OpType2[,...]][;domain_name2:OpType3[,...]]
// and appends each well-formed entry to `out`. Malformed entries are logged and skipped.
//
// Exposed in a header (rather than kept static) so unit tests can exercise it directly,
// following the same pattern as ParseOpPackages in op_package_parser.h.
void ParseCustomOpDomains(const std::string& spec,
                          std::vector<CustomOpDomainSpec>& out,
                          const Ort::Logger& logger);

}  // namespace onnxruntime
