// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/qnn_op_affinity_map.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace qnn {

namespace {

// Lowercase for case-insensitive backend-name matching.
std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// Map a (case-insensitive) backend name to the enum via QnnBackendTypeToString (single source of
// truth, so parsing and matching can't drift). nullopt for an unknown name. Bounded at HTP_FP16,
// not SERIALIZER: SERIALIZER is not a valid execution backend and must not be user-configurable.
std::optional<QnnBackendType> BackendFromName(const std::string& raw_name) {
  const std::string name = ToLower(raw_name);
  for (int i = 0; i <= static_cast<int>(QnnBackendType::HTP_FP16); ++i) {
    const auto backend = static_cast<QnnBackendType>(i);
    if (name == QnnBackendTypeToString(backend)) {
      return backend;
    }
  }
  return std::nullopt;
}

// Treats HTP and HTP_FP16 as one physical backend. Deliberately not IsNpuBackend() (which is
// HTP||DSP) -- op_affinity is scoped to GPU/HTP sessions.
bool BackendMatches(QnnBackendType pinned, QnnBackendType session_backend) {
  const bool pinned_is_htp = (pinned == QnnBackendType::HTP || pinned == QnnBackendType::HTP_FP16);
  const bool session_is_htp = (session_backend == QnnBackendType::HTP ||
                               session_backend == QnnBackendType::HTP_FP16);
  if (pinned_is_htp && session_is_htp) {
    return true;
  }
  return pinned == session_backend;
}

}  // namespace

OpAffinityMap OpAffinityMap::FromConfigFile(const std::filesystem::path& config_file) {
  std::ifstream ifs(config_file);
  if (!ifs) {
    throw std::runtime_error("op_affinity config file could not be opened: " + config_file.string());
  }

  // JSONC comments allowed; parse errors propagate.
  const nlohmann::json j = nlohmann::json::parse(ifs, /*cb*/ nullptr, /*allow_exceptions*/ true,
                                                 /*ignore_comments*/ true);

  if (!j.contains("op_type") || !j.at("op_type").is_object()) {
    throw std::runtime_error("op_affinity config: top-level \"op_type\" object is required.");
  }

  OpAffinityMap result;
  for (const auto& [op_name, value] : j.at("op_type").items()) {
    // Value is a backend string or a single-element array of one.
    std::string backend_name;
    if (value.is_string()) {
      backend_name = value.get<std::string>();
    } else if (value.is_array()) {
      if (value.empty()) {
        throw std::runtime_error("op_affinity config: op type '" + op_name +
                                 "' must specify exactly one backend, but the array is empty.");
      }
      if (value.size() > 1) {
        throw std::runtime_error(
            "op_affinity config: op type '" + op_name +
            "' must map to exactly one backend; heterogeneous execution is not supported.");
      }
      if (!value.at(0).is_string()) {
        throw std::runtime_error("op_affinity config: backend for op type '" + op_name +
                                 "' must be a string.");
      }
      backend_name = value.at(0).get<std::string>();
    } else {
      throw std::runtime_error("op_affinity config: value for op type '" + op_name +
                               "' must be a string or a single-element array of strings.");
    }

    const std::optional<QnnBackendType> backend = BackendFromName(backend_name);
    if (!backend.has_value()) {
      throw std::runtime_error("op_affinity config: unknown backend '" + backend_name +
                               "' for op type '" + op_name + "'.");
    }
    result.op_to_backend_[op_name] = *backend;
  }

  result.configured_ = true;
  return result;
}

OpAffinityMap::Decision OpAffinityMap::Evaluate(const std::string& op_type,
                                                QnnBackendType session_backend) const {
  const auto it = op_to_backend_.find(op_type);
  if (it == op_to_backend_.end()) {
    return Decision::kProceed;  // No pin (config or default): opt-out.
  }
  if (BackendMatches(it->second, session_backend)) {
    return Decision::kProceed;
  }
  if (it->second == QnnBackendType::CPU) {
    return Decision::kReject;  // Pinned to CPU: silent fallback to the CPU EP.
  }
  return Decision::kError;  // Pinned to an accelerator this session isn't running.
}

void OpAffinityMap::SeedDefaultIfAbsent(const std::string& op_type, QnnBackendType default_backend) {
  op_to_backend_.emplace(op_type, default_backend);  // no-op if a config pin already exists
}

std::optional<std::string> OpAffinityMap::ValidateForSessionBackend(QnnBackendType session_backend) const {
  for (const auto& [op_type, pinned] : op_to_backend_) {
    // CPU pin is a legitimate silent-off (Evaluate -> kReject). Any other non-matching backend can
    // never be honored, so report it.
    if (pinned != QnnBackendType::CPU && !BackendMatches(pinned, session_backend)) {
      return "op_affinity pins op type '" + op_type + "' to backend '" +
             QnnBackendTypeToString(pinned) + "', but this session runs backend '" +
             QnnBackendTypeToString(session_backend) +
             "'. Heterogeneous execution is not supported; pin it to the running backend or to 'cpu'.";
    }
  }
  return std::nullopt;
}

}  // namespace qnn
}  // namespace onnxruntime
