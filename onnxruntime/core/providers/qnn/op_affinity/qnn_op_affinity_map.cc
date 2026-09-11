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

void ParseRuleMap(const nlohmann::json& rules,
                  const std::string& rule_kind,
                  std::unordered_map<std::string, QnnBackendType>& output) {
  for (const auto& [selector, value] : rules.items()) {
    std::string backend_name;
    if (value.is_string()) {
      backend_name = value.get<std::string>();
    } else if (value.is_array()) {
      if (value.empty()) {
        throw std::runtime_error("op_affinity config: " + rule_kind + " '" + selector +
                                 "' must specify exactly one backend, but the array is empty.");
      }
      if (value.size() > 1) {
        throw std::runtime_error("op_affinity config: " + rule_kind + " '" + selector +
                                 "' must map to exactly one backend; heterogeneous execution is not supported.");
      }
      if (!value.at(0).is_string()) {
        throw std::runtime_error("op_affinity config: backend for " + rule_kind + " '" + selector +
                                 "' must be a string.");
      }
      backend_name = value.at(0).get<std::string>();
    } else {
      throw std::runtime_error("op_affinity config: value for " + rule_kind + " '" + selector +
                               "' must be a string or a single-element array of strings.");
    }

    const std::optional<QnnBackendType> backend = BackendFromName(backend_name);
    if (!backend.has_value()) {
      throw std::runtime_error("op_affinity config: unknown backend '" + backend_name +
                               "' for " + rule_kind + " '" + selector + "'.");
    }
    output[selector] = *backend;
  }
}

Ort::Status ValidateRuleBackends(const std::unordered_map<std::string, QnnBackendType>& rules,
                                 const std::string& rule_kind,
                                 QnnBackendType session_backend) {
  for (const auto& [selector, pinned] : rules) {
    if (pinned != QnnBackendType::CPU && !BackendMatches(pinned, session_backend)) {
      const std::string message = "op_affinity pins " + rule_kind + " '" + selector + "' to backend '" +
                                  QnnBackendTypeToString(pinned) + "', but this session runs backend '" +
                                  QnnBackendTypeToString(session_backend) +
                                  "'. Heterogeneous execution is not supported; pin it to the running backend or to 'cpu'.";
      return MAKE_EP_FAIL(message.c_str());
    }
  }
  return Ort::Status();
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

  const bool has_op_type = j.contains("op_type");
  const bool has_op_name = j.contains("op_name");
  if (!has_op_type && !has_op_name) {
    throw std::runtime_error("op_affinity config: a top-level \"op_type\" or \"op_name\" object is required.");
  }
  if (has_op_type && !j.at("op_type").is_object()) {
    throw std::runtime_error("op_affinity config: top-level \"op_type\" must be an object.");
  }
  if (has_op_name && !j.at("op_name").is_object()) {
    throw std::runtime_error("op_affinity config: top-level \"op_name\" must be an object.");
  }

  OpAffinityMap result;
  if (has_op_type) {
    ParseRuleMap(j.at("op_type"), "op type", result.op_type_to_backend_);
  }
  if (has_op_name) {
    ParseRuleMap(j.at("op_name"), "op name", result.op_name_to_backend_);
  }

  return result;
}

Ort::Status OpAffinityMap::Evaluate(const std::string& op_type, const std::string& op_name,
                                    QnnBackendType session_backend) const {
  const QnnBackendType* pinned_backend = nullptr;
  std::string selector_description;
  const auto op_name_it = op_name.empty() ? op_name_to_backend_.end() : op_name_to_backend_.find(op_name);
  if (op_name_it != op_name_to_backend_.end()) {
    pinned_backend = &op_name_it->second;
    selector_description = "op name '" + op_name + "'";
  } else {
    const auto op_type_it = op_type_to_backend_.find(op_type);
    if (op_type_it == op_type_to_backend_.end()) {
      return Ort::Status();  // No pin (config or default): opt-out.
    }
    pinned_backend = &op_type_it->second;
    selector_description = "op type '" + op_type + "'";
  }

  if (BackendMatches(*pinned_backend, session_backend)) {
    return Ort::Status();
  }
  if (*pinned_backend == QnnBackendType::CPU) {
    return MAKE_EP_FAIL((selector_description + " filtered off QNN by the op_affinity provider option.").c_str());
  }
  return MAKE_EP_FAIL((selector_description + " is pinned to a backend this session is not running.").c_str());
}

void OpAffinityMap::SeedDefaultIfAbsent(const std::string& op_type, QnnBackendType default_backend) {
  op_type_to_backend_.emplace(op_type, default_backend);  // no-op if a config pin already exists
}

Ort::Status OpAffinityMap::ValidateForSessionBackend(QnnBackendType session_backend) const {
  Ort::Status status = ValidateRuleBackends(op_type_to_backend_, "op type", session_backend);
  if (!status.IsOK()) {
    return status;
  }
  return ValidateRuleBackends(op_name_to_backend_, "op name", session_backend);
}

}  // namespace qnn
}  // namespace onnxruntime
