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
// execution is not supported (one backend per op). Evaluated generically for every op type by
// QnnNodeUnitWrapper::IsSupported and the fusion-discard check in GetQnnNodeGroupsImpl
// (qnn_node_group.cc). Always safe to hold and query: a default-constructed instance is
// unconfigured and Evaluate() always succeeds against it, so callers never need to check for a
// null/absent map.
class OpAffinityMap {
 public:
  OpAffinityMap() = default;  // Unconfigured: the option is unset.

  // Throws std::runtime_error on any problem (unopenable/malformed file, bad "op_type", empty/multi
  // array, unknown backend). The EP caller does not catch, so a bad config fails session creation.
  static OpAffinityMap FromConfigFile(const std::filesystem::path& config_file);

  // True once a config file loaded; distinguishes "no option" from "option set but op absent".
  bool IsConfigured() const { return configured_; }

  // OK if op_type may proceed onto QNN for session_backend (including the common case of no pin at
  // all); ORT_EP_FAIL if op_type is pinned to CPU (silent fallback intent) or to an accelerator this
  // session isn't running. Callers that only need a boolean can check IsOK(); callers that want to
  // surface the failure should RETURN_IF_ERROR/propagate it, since the message already explains why.
  Ort::Status Evaluate(const std::string& op_type, QnnBackendType session_backend) const;

  // Sets op_type's pin to default_backend unless a config entry already pins it. Lets a caller with
  // resolved backend knowledge (e.g. "GQA defaults to CPU on HTP") seed a default as ordinary pin
  // data, so Evaluate() stays a single generic lookup. Must be called after backend setup resolves
  // the real backend (not at EP construction, where it is not yet known).
  void SeedDefaultIfAbsent(const std::string& op_type, QnnBackendType default_backend);

  // OK unless some op is pinned to an accelerator that is neither the session backend (htp/htp_fp16
  // alias-aware) nor CPU, in which case the returned Status carries a readable error message. Must
  // be called after backend setup resolves the real backend (not at EP construction). Always OK for
  // an unconfigured map.
  Ort::Status ValidateForSessionBackend(QnnBackendType session_backend) const;

 private:
  std::unordered_map<std::string, QnnBackendType> op_to_backend_;  // op type -> its single backend
  bool configured_ = false;
};

}  // namespace qnn
}  // namespace onnxruntime
