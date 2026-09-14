// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

// The "op_affinity" provider option: a JSON config file mapping ONNX op types to the single backend
// allowed to claim them, e.g. { "op_type": { "GroupQueryAttention": "HTP" } }. Heterogeneous
// execution is not supported (one backend per op).
class OpAffinityMap {
 public:
  OpAffinityMap() = default;  // Unconfigured: the option is unset.

  static OpAffinityMap FromConfigFile(const std::filesystem::path& config_file);

  Ort::Status Evaluate(const std::string& op_type, QnnBackendType session_backend) const;

  void SeedDefaultIfAbsent(const std::string& op_type, QnnBackendType default_backend);

  Ort::Status ValidateForSessionBackend(QnnBackendType session_backend) const;

 private:
  std::unordered_map<std::string, QnnBackendType> op_to_backend_;  // op type -> its single backend
};

}  // namespace qnn
}  // namespace onnxruntime
