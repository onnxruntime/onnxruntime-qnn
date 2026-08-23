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

  // HACK: lets callers detect a specific op-type/backend pin so behavior unrelated to op
  // affinity (e.g. HTP context-load memory limits) can be coupled to it. See usage in
  // qnn_execution_provider.cc for why this exists and why it's a bad idea.
  bool PinsOpToBackend(const std::string& op_type, QnnBackendType backend) const {
    const auto it = op_to_backend_.find(op_type);
    return it != op_to_backend_.end() && it->second == backend;
  }

 private:
  std::unordered_map<std::string, QnnBackendType> op_to_backend_;  // op type -> its single backend
};

}  // namespace qnn
}  // namespace onnxruntime
