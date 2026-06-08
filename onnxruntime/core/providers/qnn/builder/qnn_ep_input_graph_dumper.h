// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// EP-internal helper that serializes the ONNX graph the QNN EP receives in
// GetCapabilityImpl (compile-time, before partitioning) into a JSON file using
// the same schema as the post-compile `dump_json_qnn_graph` output, so the
// result can be opened in QNN Netron and consumed by the offline
// source->optimized op-trace matcher.
//
// This captures the graph AFTER ORT Level 1 optimizations (QDQ preserved,
// `/duplicated` DQ copies, Transpose rearrangements) but BEFORE the QNN EP
// compiles its partition, which is otherwise not observable: the ORT optimized-
// model serializer refuses graphs containing compiled nodes.

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <unordered_set>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Walks the EP-input OrtGraph and writes a QNN-Netron-schema JSON description
// to `output_path`. Returns true on success; logs a WARNING and returns false
// on any failure (the caller treats the dump as a best-effort diagnostic and
// continues compilation regardless).
bool DumpQnnEpInputGraphToJson(const OrtGraph* graph,
                               const std::filesystem::path& output_path,
                               const Ort::Logger& logger);

// Returns a filename-safe form of `graph_name`: any character outside
// [A-Za-z0-9._-] is replaced with `_`, leading dots/hyphens are stripped (so
// the result is never `..` or a hidden-file form), and trailing dots and
// spaces are dropped (a Windows-reserved suffix). Windows reserved device
// names (CON, PRN, AUX, NUL, COM1..9, LPT1..9, case-insensitive) are
// suffixed with `_` so they no longer match the device-name pattern. An
// input that sanitizes to the empty string returns "graph", giving callers a
// non-empty fallback they can disambiguate further.
//
// Defined inline so the unit tests in onnxruntime_provider_test (which does
// not link onnxruntime_providers_qnn) can call it directly.
inline std::string SanitizeGraphNameForFilename(const std::string& graph_name) {
  std::string result;
  result.reserve(graph_name.size());
  for (char c : graph_name) {
    unsigned char uc = static_cast<unsigned char>(c);
    bool safe = (uc >= '0' && uc <= '9') ||
                (uc >= 'A' && uc <= 'Z') ||
                (uc >= 'a' && uc <= 'z') ||
                uc == '.' || uc == '_' || uc == '-';
    result.push_back(safe ? static_cast<char>(uc) : '_');
  }

  size_t first = 0;
  while (first < result.size() && (result[first] == '.' || result[first] == '-')) {
    ++first;
  }
  if (first > 0) {
    result.erase(0, first);
  }

  while (!result.empty() && (result.back() == '.' || result.back() == ' ')) {
    result.pop_back();
  }

  if (result.empty()) {
    return "graph";
  }

  std::string stem = result.substr(0, result.find('.'));
  std::string upper_stem = stem;
  std::transform(upper_stem.begin(), upper_stem.end(), upper_stem.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  static const std::unordered_set<std::string> kReserved = {
      "CON", "PRN", "AUX", "NUL",
      "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
      "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};
  if (kReserved.count(upper_stem)) {
    result.insert(stem.size(), "_");
  }

  return result;
}

}  // namespace qnn
}  // namespace onnxruntime
