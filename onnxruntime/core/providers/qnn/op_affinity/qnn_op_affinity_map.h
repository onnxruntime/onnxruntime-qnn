// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

// The "op_affinity" provider option: a JSON config file mapping ONNX op types to the single backend
// allowed to claim them, e.g. { "op_type": { "GroupQueryAttention": "HTP" } }. Heterogeneous
// execution is not supported (one backend per op). Evaluated generically for every op type by
// QnnNodeUnitWrapper::IsSupported and the fusion-discard check in GetQnnNodeGroupsImpl
// (qnn_node_group.cc);
class OpAffinityMap {
 public:
  // kError (not an exception) signals a runtime backend mismatch, letting the op builder RETURN_IF it.
  enum class Decision : uint8_t { kProceed,
                                  kReject,
                                  kError };

  OpAffinityMap() = default;  // Unconfigured: the option is unset.

  // Throws std::runtime_error on any problem (unopenable/malformed file, bad "op_type", empty/multi
  // array, unknown backend). The EP caller does not catch, so a bad config fails session creation.
  static OpAffinityMap FromConfigFile(const std::filesystem::path& config_file);

  // True once a config file loaded; distinguishes "no option" from "option set but op absent".
  bool IsConfigured() const { return configured_; }

  Decision Evaluate(const std::string& op_type, QnnBackendType session_backend) const;

  // Sets op_type's pin to default_backend unless a config entry already pins it. Lets a caller with
  // resolved backend knowledge (e.g. "GQA defaults to CPU on HTP") seed a default as ordinary pin
  // data, so Evaluate() stays a single generic lookup. Must be called after backend setup resolves
  // the real backend (not at EP construction, where it is not yet known).
  void SeedDefaultIfAbsent(const std::string& op_type, QnnBackendType default_backend);

  // Returns an error message if any op is pinned to an accelerator that is neither the session backend
  // (htp/htp_fp16 alias-aware) nor CPU; nullopt otherwise. Must be called after backend setup resolves
  // the real backend (not at EP construction). No-op for an unconfigured map.
  std::optional<std::string> ValidateForSessionBackend(QnnBackendType session_backend) const;

 private:
  std::unordered_map<std::string, QnnBackendType> op_to_backend_;  // op type -> its single backend
  bool configured_ = false;
};

}  // namespace qnn
}  // namespace onnxruntime
