// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Encapsulates the "op_affinity" provider option: which ONNX op types the QNN EP claims in
// GetSupportedNodes. It filters on the ONNX op type (e.g. "Softmax"), NOT on QNN op names -- the
// "Onnx" in the class name is a deliberate reminder of that. Parsing (inline spec or "@<path>" JSON
// config file), the exclude/include policy, the optional backend scope, and the diagnostic logging
// all live here so qnn_execution_provider.cc only needs to construct one via FromOptionValue() and
// call ShouldFilterOff() per node group.
//
// Two modes:
//   kExclude (default): every op goes to QNN EXCEPT the listed op types, which fall back to another
//                       EP (typically CPU). Empty list = no filtering (the default).
//   kInclude:           ONLY the listed op types are claimed by QNN; everything else falls back. An
//                       empty list forces ALL ops off QNN.
//
// Optional backend scope: a filter may be scoped to a single backend (e.g. "htp") using the same
// backend-name strings as the "backend_type" option. A session runs one backend, so scope is a
// single filter-wide constant. A filter scoped to a backend the session isn't running is inert
// (ShouldFilterOff() returns false) -- see AppliesToBackend().
//
// ShouldFilterOff() takes the whole OrtNodeUnit (not just the op-type string) so a future node-name
// filter can be added here without touching call sites.
class OnnxOpAffinity {
 public:
  enum class Mode : uint8_t { kExclude,
                              kInclude };

  // An inactive filter (kExclude + empty list) -- the default when the option is unset.
  OnnxOpAffinity() = default;

  // Build a filter from an inline spec "[backend:][mode:]<comma-separated op types>" (backend
  // optional, mode defaults to "exclude"). Parsing is pure string work and does not throw.
  explicit OnnxOpAffinity(const std::string& inline_spec);

  // const char* overload delegating to the std::string constructor. Without it, a string literal
  // (e.g. OnnxOpAffinity("exclude:Softmax")) is an ambiguous conversion between std::string and
  // std::filesystem::path (the config-file constructor below); this exact-match overload resolves it.
  explicit OnnxOpAffinity(const char* inline_spec) : OnnxOpAffinity(std::string(inline_spec)) {}

  // Build a filter from a JSON config file
  // { "backend": "htp", "mode": "exclude"|"include", "op_types": [...] } ("backend" optional).
  // Throws std::runtime_error if the file cannot be opened, and lets nlohmann::json parse errors
  // propagate. FromOptionValue() is the caller that catches these and degrades to an inactive
  // filter, so construction itself is free to throw on bad input.
  explicit OnnxOpAffinity(const std::filesystem::path& config_file);

  // Parse a provider-option value into a filter -- either an inline spec "[backend:][mode:]<op types>"
  // (mode defaults to "exclude") or "@<path>" to a JSON config file. On any parse failure this logs a
  // WARNING and returns an inactive filter, so a bad value degrades to "no filtering" rather than
  // failing session creation. Also logs the include-with-empty-list warning and a VERBOSE summary.
  static OnnxOpAffinity FromOptionValue(const std::string& option_value, const Ort::Logger& logger);

  // True if the filter would keep at least one op off QNN. An empty exclude list is inactive; include
  // mode is always active. Backend scope is not considered here -- that's AppliesToBackend()'s job.
  bool IsActive() const {
    return !selected_op_types_.empty() || mode_ == Mode::kInclude;
  }

  // True if this filter is the EP's built-in default (seeded because the user did not set op_affinity),
  // rather than a value the user typed. Diagnostics differ for the two: an unmatched default entry is
  // not a user typo, and a default filter's disable_cpu_ep_fallback interaction gets an actionable
  // (rather than generic) warning. Set via MarkAsDefault() by the EP right after construction.
  bool IsDefault() const { return is_default_; }

  // Mark this filter as the EP's built-in default. Only the EP calls this, on the instance it seeds
  // when the user leaves op_affinity unset. See IsDefault().
  void MarkAsDefault() { is_default_ = true; }

  // True if this filter applies to the session's backend. An unscoped filter applies to any backend;
  // a scoped filter applies only when its scope matches the running backend's name -- except "htp"
  // and "htp_fp16" are aliases for the same physical backend. When false, ShouldFilterOff() returns
  // false unconditionally.
  bool AppliesToBackend(QnnBackendType backend_type) const;

  // Decide whether the given node group's target op should be kept off QNN. An inactive filter, or one
  // scoped to a different backend, always returns false. Records a match for WarnUnmatchedEntries().
  // const: only mutates the mutable diagnostic bookkeeping, so it's callable from the const
  // GetSupportedNodes().
  bool ShouldFilterOff(const OrtNodeUnit& target_node_unit, QnnBackendType backend_type) const;

  // Warn about list entries that never matched any node group since the last call -- almost always a
  // misspelled op type, which would otherwise be a silent no-op. Clears the match bookkeeping on every
  // exit path, so repeated calls (GetSupportedNodes runs multiple times per session) each start clean.
  // If the filter is scoped to a backend the session isn't running, no entry can match by design, so
  // the per-entry warnings are suppressed in favor of a single INFO line. Takes the session backend to
  // tell the two cases apart. A default filter (IsDefault()) suppresses the per-entry typo warnings
  // entirely: the user never typed those entries, so an unmatched default op type (e.g. a model with
  // no GroupQueryAttention) is expected, not a typo.
  void WarnUnmatchedEntries(QnnBackendType backend_type, const Ort::Logger& logger) const;

 private:
  Mode mode_ = Mode::kExclude;
  // The op types listed in the filter spec. In exclude mode these are kept off QNN; in include mode
  // these are the only op types claimed by QNN. Empty in the default (inactive) exclude filter.
  std::unordered_set<std::string> selected_op_types_;
  // Optional backend scope. nullopt = applies to any backend. Stored as the backend-name string
  // ("htp"/"gpu"/...) rather than the QnnBackendType enum so it compares directly against
  // QnnBackendTypeToString() and stays decoupled from the enum's internal ordering.
  std::optional<std::string> backend_scope_;
  // Entries matched by ShouldFilterOff() since the last WarnUnmatchedEntries() call (which clears this
  // on every exit path). Mutable diagnostic bookkeeping only -- does not affect filtering. Scoped to
  // "since the last call" because GetSupportedNodes can run multiple times per session.
  mutable std::unordered_set<std::string> matched_op_types_;
  // True if this filter is the EP's built-in default rather than a user-supplied value. Affects only
  // diagnostics (typo warnings, disable_cpu_ep_fallback message), never filtering. See IsDefault().
  bool is_default_ = false;
};

}  // namespace qnn
}  // namespace onnxruntime
