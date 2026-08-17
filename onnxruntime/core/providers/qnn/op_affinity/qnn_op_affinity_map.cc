// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/op_affinity/qnn_op_affinity_map.h"

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

// Map a (case-insensitive) backend name to the enum via QnnBackendTypeToString
std::optional<QnnBackendType> BackendFromName(const std::string& raw_name) {
  const std::string name = ToLower(raw_name);
  // Only real execution backends are valid op_affinity targets
  for (int i = 0; i <= static_cast<int>(QnnBackendType::HTP); ++i) {
    const auto backend = static_cast<QnnBackendType>(i);
    if (name == QnnBackendTypeToString(backend)) {
      return backend;
    }
  }
  return std::nullopt;
}

bool BackendMatches(QnnBackendType pinned, QnnBackendType session_backend) {
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

  return result;
}

Ort::Status OpAffinityMap::Evaluate(const std::string& op_type,
                                    QnnBackendType session_backend) const {
  const auto it = op_to_backend_.find(op_type);
  if (it == op_to_backend_.end()) {
    return Ort::Status();  // No pin (config or default): opt-out.
  }
  if (BackendMatches(it->second, session_backend)) {
    return Ort::Status();
  }
  if (it->second == QnnBackendType::CPU) {
    return MAKE_EP_FAIL((op_type + " filtered off QNN by the op_affinity provider option.").c_str());
  }
  return MAKE_EP_FAIL((op_type + " op_affinity pins it to a backend this session is not running.").c_str());
}

void OpAffinityMap::SeedDefaultIfAbsent(const std::string& op_type, QnnBackendType default_backend) {
  op_to_backend_.emplace(op_type, default_backend);  // no-op if a config pin already exists
}

Ort::Status OpAffinityMap::ValidateForSessionBackend(QnnBackendType session_backend) const {
  for (const auto& [op_type, pinned] : op_to_backend_) {
    // CPU pin is a legitimate silent-off (Evaluate -> !IsOK() but a benign fallback). Any other
    // non-matching backend can never be honored, so report it.
    if (pinned != QnnBackendType::CPU && !BackendMatches(pinned, session_backend)) {
      const std::string message = "op_affinity pins op type '" + op_type + "' to backend '" +
                                  QnnBackendTypeToString(pinned) + "', but this session runs backend '" +
                                  QnnBackendTypeToString(session_backend) +
                                  "'. Heterogeneous execution is not supported; pin it to the running backend or to 'cpu'.";
      return MAKE_EP_FAIL(message.c_str());
    }
  }
  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
