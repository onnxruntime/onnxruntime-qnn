// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/qnn_onnx_op_affinity.h"

#include <exception>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "nlohmann/json.hpp"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Only the exact "include" spelling selects include mode; anything else (the default, or a typo)
// resolves to the safer exclude mode.
OnnxOpAffinity::Mode ParseMode(const std::string& mode_str) {
  return (mode_str == "include") ? OnnxOpAffinity::Mode::kInclude : OnnxOpAffinity::Mode::kExclude;
}

// Recognized backend scope names, derived from QnnBackendTypeToString() (the single source of truth)
// rather than a hand-maintained list, so the two can't drift. "htp" also covers HTP_FP16 -- see
// AppliesToBackend().
bool IsKnownBackendName(std::string_view name) {
  for (uint8_t i = 0; i <= static_cast<uint8_t>(QnnBackendType::SERIALIZER); ++i) {
    if (name == QnnBackendTypeToString(static_cast<QnnBackendType>(i))) {
      return true;
    }
  }
  return false;
}

// Phase-2 syntax guard. '[' is reserved for a future op-name qualifier, "OpType[name=...]". Phase 1
// doesn't implement name matching, so reject any token carrying it (rather than treat it as a literal
// op type that would silently never match) -- this keeps a phase-1 spec a strict subset of the future
// grammar. Colons are NOT reserved (UDO op types use them). Throws so FromOptionValue() degrades the
// whole filter to inactive.
void RejectReservedOpTypeChars(const std::string& op_type) {
  if (op_type.find('[') != std::string::npos) {
    throw std::runtime_error("op_affinity op type '" + op_type +
                             "' uses the reserved '[' character (op-name qualifier); "
                             "op-name filtering is not supported in this version.");
  }
}

}  // namespace

OnnxOpAffinity::OnnxOpAffinity(const std::string& inline_spec) {
  // Inline spec: "[backend:][mode:]<comma-separated op types>". Only exact reserved words are treated
  // as prefixes -- an op type may itself contain a colon (e.g. UDO "custom:MyOp"), so an unrecognized
  // prefix stays part of the literal op-type list.
  std::string remainder = inline_spec;

  // Optional backend scope prefix (checked before mode: it is the outermost segment).
  if (const auto colon = remainder.find(':'); colon != std::string::npos) {
    const std::string maybe_backend = remainder.substr(0, colon);
    if (IsKnownBackendName(maybe_backend)) {
      backend_scope_ = maybe_backend;
      remainder = remainder.substr(colon + 1);
    }
  }

  // Optional mode prefix.
  std::string mode_str = "exclude";
  if (const auto colon = remainder.find(':'); colon != std::string::npos) {
    const std::string maybe_mode = remainder.substr(0, colon);
    if (maybe_mode == "exclude" || maybe_mode == "include") {
      mode_str = maybe_mode;
      remainder = remainder.substr(colon + 1);
    }
  }

  mode_ = ParseMode(mode_str);
  for (const auto& token : utils::SplitString(remainder, ",", /*keep_empty*/ false)) {
    if (!token.empty()) {
      std::string op_type(token);
      RejectReservedOpTypeChars(op_type);  // phase-2 "[name=...]" syntax -> degrade to inactive
      selected_op_types_.emplace(std::move(op_type));
    }
  }
}

OnnxOpAffinity::OnnxOpAffinity(const std::filesystem::path& config_file) {
  // JSON config file: { "backend": "htp", "mode": "exclude"|"include", "op_types": [...] }
  // ("backend" optional). Anything malformed -- unopenable file, bad JSON, or a field of the wrong
  // type/value -- throws; FromOptionValue() catches and degrades to an inactive filter + WARNING. The
  // per-field type/value checks below keep this path as loud as the inline path rather than silently
  // coercing a typo'd config to a surprising default.
  std::ifstream ifs(config_file);
  if (!ifs) {
    throw std::runtime_error("op_affinity config file could not be opened: " +
                             config_file.string());
  }

  const nlohmann::json j = nlohmann::json::parse(ifs, /*cb*/ nullptr, /*allow_exceptions*/ true,
                                                 /*ignore_comments*/ true);

  if (j.contains("backend")) {
    if (!j.at("backend").is_string()) {
      throw std::runtime_error("op_affinity config: \"backend\" must be a string.");
    }
    std::string backend = utils::TrimWhitespace(j.at("backend").get<std::string>());
    if (!backend.empty()) {
      backend_scope_ = std::move(backend);
    }
  }

  std::string mode_str = "exclude";  // default mode when not specified
  if (j.contains("mode")) {
    if (!j.at("mode").is_string()) {
      throw std::runtime_error("op_affinity config: \"mode\" must be a string.");
    }
    mode_str = j.at("mode").get<std::string>();
    // Unlike the inline path (where an unrecognized token falls into the op-type list), a config file
    // names the mode explicitly, so a value that is neither "exclude" nor "include" is a typo -- reject
    // it loudly rather than let ParseMode coerce it to exclude.
    if (mode_str != "exclude" && mode_str != "include") {
      throw std::runtime_error("op_affinity config: \"mode\" must be \"exclude\" or \"include\", got \"" +
                               mode_str + "\".");
    }
  }
  mode_ = ParseMode(mode_str);

  if (j.contains("op_types")) {
    if (!j.at("op_types").is_array()) {
      throw std::runtime_error("op_affinity config: \"op_types\" must be an array of strings.");
    }
    for (const auto& item : j.at("op_types")) {
      if (!item.is_string()) {
        throw std::runtime_error("op_affinity config: every entry in \"op_types\" must be a string.");
      }
      // File values may be whitespace-padded; trim so entries match node op types.
      std::string op_type = utils::TrimWhitespace(item.get<std::string>());
      if (!op_type.empty()) {
        RejectReservedOpTypeChars(op_type);  // phase-2 "[name=...]" syntax -> degrade to inactive
        selected_op_types_.emplace(std::move(op_type));
      }
    }
  }
}

OnnxOpAffinity OnnxOpAffinity::FromOptionValue(const std::string& option_value, const Ort::Logger& logger) {
  const std::string trimmed = utils::TrimWhitespace(option_value);
  if (trimmed.empty()) {
    return OnnxOpAffinity{};  // No filter configured; keep default (exclude + empty list).
  }

  OnnxOpAffinity filter;
  try {
    if (trimmed[0] == '@') {
      // Value is a path to a JSON config file.
      const std::string path = utils::TrimWhitespace(std::string_view(trimmed).substr(1));
      filter = OnnxOpAffinity(std::filesystem::path(path));
    } else {
      filter = OnnxOpAffinity(trimmed);
    }
  } catch (const std::exception& ex) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                (std::string("Failed to parse op_affinity value '") + trimmed +
                 "': " + ex.what() + ". Falling back to no filtering.")
                    .c_str());
    return OnnxOpAffinity{};  // Bad value degrades to "no filtering" rather than failing the session.
  }

  if (filter.mode_ == Mode::kInclude && filter.selected_op_types_.empty()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                "op_affinity mode is 'include' but the op-type list is empty; "
                "ALL op types will fall back off QNN.");
  }

  // A backend scope that is not a recognized backend name can never match a real session; warn so a
  // typo (e.g. "hpt:exclude:...") is visible rather than a silent no-op.
  if (filter.backend_scope_.has_value() && !IsKnownBackendName(*filter.backend_scope_)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("op_affinity backend scope '" + *filter.backend_scope_ +
                 "' is not a recognized backend name; the filter will not apply to any session.")
                    .c_str());
  }

  if (filter.IsActive()) {
    std::string joined;
    for (const auto& op_type : filter.selected_op_types_) {
      joined += (joined.empty() ? "" : ", ") + op_type;
    }
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("op_affinity backend=" +
                 (filter.backend_scope_.has_value() ? *filter.backend_scope_ : std::string("<any>")) +
                 " mode=" +
                 std::string(filter.mode_ == Mode::kInclude ? "include" : "exclude") +
                 " op_types=[" + joined + "]")
                    .c_str());
  }

  return filter;
}

bool OnnxOpAffinity::AppliesToBackend(QnnBackendType backend_type) const {
  if (!backend_scope_.has_value()) {
    return true;  // Unscoped filter applies to any backend.
  }
  // "htp" and "htp_fp16" are the same physical backend at different precision settings, so a scope
  // written as either name matches a session running the other.
  if (backend_type == QnnBackendType::HTP || backend_type == QnnBackendType::HTP_FP16) {
    return *backend_scope_ == "htp" || *backend_scope_ == "htp_fp16";
  }
  return *backend_scope_ == QnnBackendTypeToString(backend_type);
}

bool OnnxOpAffinity::ShouldFilterOff(const OrtNodeUnit& target_node_unit, QnnBackendType backend_type) const {
  if (!IsActive() || !AppliesToBackend(backend_type)) {
    return false;  // Inactive, or scoped to a different backend: never keeps anything off QNN.
  }
  const std::string target_op_type = target_node_unit.OpType();
  const bool in_list = selected_op_types_.count(target_op_type) != 0;
  if (in_list) {
    matched_op_types_.emplace(target_op_type);
  }
  //   exclude mode: filter off the op types in the list.
  //   include mode: filter off every op type NOT in the list.
  return (mode_ == Mode::kExclude) ? in_list : !in_list;
}

void OnnxOpAffinity::WarnUnmatchedEntries(QnnBackendType backend_type, const Ort::Logger& logger) const {
  // GetSupportedNodes runs multiple times per session (see header), so matched_op_types_ must not
  // survive past this call -- otherwise an op matched only in a later pass would be wrongly flagged
  // as unmatched here. Cleared unconditionally on every exit path below.

  // A filter scoped to a backend this session isn't running can never match by design, so the
  // per-entry "did you make a typo?" warnings would be misleading. Suppress them and log one INFO.
  if (IsActive() && !AppliesToBackend(backend_type)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
                ("op_affinity is scoped to backend '" + *backend_scope_ +
                 "' but this session runs '" + QnnBackendTypeToString(backend_type) +
                 "' -- filter skipped.")
                    .c_str());
    matched_op_types_.clear();
    return;
  }

  for (const auto& op_type : selected_op_types_) {
    if (matched_op_types_.count(op_type) == 0) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                  ("op_affinity entry '" + op_type +
                   "' did not match any node in the graph; check for a typo in the op type name.")
                      .c_str());
    }
  }
  matched_op_types_.clear();
}

}  // namespace qnn
}  // namespace onnxruntime
