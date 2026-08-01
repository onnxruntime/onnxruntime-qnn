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

// Parses and represents the "op_affinity" provider option: a JSON config file mapping ONNX op types
// to the single backend allowed to claim them, e.g. { "op_type": { "GroupQueryAttention": "HTP" } }.
//
// Heterogeneous execution is NOT supported, so each op maps to at most ONE backend; a value that is
// an array of length > 1 is rejected at parse time.
//
// Consumed today only by the GroupQueryAttention op builder's IsOpSupported. The decision truth table
// (see the design spec) is fully encapsulated in Evaluate() so call sites only branch on its result.
class OpAffinityMap {
 public:
  // Result of Evaluate(). kError is returned (not thrown) for a runtime backend mismatch so the op
  // builder can convert it into a fail Status via RETURN_IF, matching the codebase's RETURN_IF_* idiom.
  enum class Decision : uint8_t { kProceed,
                                  kReject,
                                  kError };

  // Unconfigured filter -- the state when the "op_affinity" option is unset.
  OpAffinityMap() = default;

  // Parse a JSON config file into a map. Throws std::runtime_error on ANY problem: unopenable file,
  // malformed JSON, missing/!object "op_type", a value that is neither string nor array, an empty or
  // length>1 array, or an unknown backend name. The EP caller deliberately does NOT catch, so a bad
  // config fails session creation loudly.
  static OpAffinityMap FromConfigFile(const std::filesystem::path& config_file);

  // True when a config file was successfully loaded (even if it lists no ops). Distinguishes
  // "no option given" from "option given but this op absent".
  bool IsConfigured() const { return configured_; }

  // Encapsulates the whole truth table. Does not throw. See the design spec §3 for every cell.
  Decision Evaluate(const std::string& op_type, QnnBackendType session_backend) const;

  // Validate every configured pin against the backend this session will actually run. Returns a
  // human-readable error message if any op is pinned to a concrete accelerator that is neither the
  // session backend (htp/htp_fp16 alias-aware) nor CPU (CPU means "fall back to the CPU EP", a
  // legitimate silent-off intent); returns std::nullopt if all pins are valid. The caller fails
  // session creation on a returned message. Must be called only once the session's real backend is
  // known (i.e. after QnnBackendManager backend setup), NOT at EP construction time where the
  // backend type is not yet resolved. A no-op (nullopt) for an unconfigured map.
  std::optional<std::string> ValidateForSessionBackend(QnnBackendType session_backend) const;

 private:
  // op type -> the single backend allowed to claim it. Populated only from a config file.
  std::unordered_map<std::string, QnnBackendType> op_to_backend_;
  bool configured_ = false;
};

}  // namespace qnn
}  // namespace onnxruntime
