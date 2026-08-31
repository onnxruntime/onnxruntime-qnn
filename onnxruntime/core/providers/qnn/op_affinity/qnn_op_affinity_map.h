// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

// The "op_affinity" provider option: a JSON config file mapping ONNX op types and/or canonical
// node names to the single backend allowed to claim them. Heterogeneous execution is not supported.
class OpAffinityMap {
 public:
  OpAffinityMap() = default;  // Unconfigured: the option is unset.

  static OpAffinityMap FromConfigFile(const std::filesystem::path& config_file);

  Ort::Status Evaluate(const std::string& op_type, const std::string& op_name,
                       QnnBackendType session_backend) const;

  // Preserves the Phase 1 API for callers and tests that only evaluate op-type rules.
  Ort::Status Evaluate(const std::string& op_type, QnnBackendType session_backend) const {
    return Evaluate(op_type, "", session_backend);
  }

  bool HasOpNameRules() const { return !op_name_to_backend_.empty(); }
  const std::unordered_map<std::string, QnnBackendType>& GetOpNameRules() const {
    return op_name_to_backend_;
  }

  void SeedDefaultIfAbsent(const std::string& op_type, QnnBackendType default_backend);

  Ort::Status ValidateForSessionBackend(QnnBackendType session_backend) const;

 private:
  std::unordered_map<std::string, QnnBackendType> op_type_to_backend_;
  std::unordered_map<std::string, QnnBackendType> op_name_to_backend_;
};

}  // namespace qnn
}  // namespace onnxruntime
