// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Pure-STL filename sanitizer extracted from qnn_ep_input_graph_dumper.h so
// callers can use it without dragging in the QNN EP private API surface
// (`core/providers/qnn/ort_api.h`). Used by both the dumper itself and the
// dumper unit tests.

#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>

namespace onnxruntime {
namespace qnn {

// Returns a filename-safe form of `graph_name`: any character outside
// [A-Za-z0-9._-] is replaced with `_` (so spaces and path separators become
// `_`), leading dots/hyphens are stripped (so the result is never `..` or a
// hidden-file form), and trailing dots are dropped (Windows treats a trailing
// dot as ignorable). Windows reserved device names (CON, PRN, AUX, NUL,
// COM1..9, LPT1..9, case-insensitive) are suffixed with `_` so they no longer
// match the device-name pattern. An input that sanitizes to the empty string
// returns "graph", giving callers a non-empty fallback they can disambiguate
// further.
//
// Defined inline so the unit tests in onnxruntime_provider_test (which does
// not link libonnxruntime_providers_qnn.so) can call it directly.
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

  // Trailing dot strip. Windows treats trailing dots as ignorable when
  // resolving filenames (`foo.` and `foo` open the same file), so dropping
  // them avoids ambiguity. Trailing spaces are already replaced with `_` by
  // the safe-set pass above and so cannot reach this loop.
  while (!result.empty() && result.back() == '.') {
    result.pop_back();
  }

  if (result.empty()) {
    return "graph";
  }

  // Windows reserves a small set of device names regardless of extension:
  // creating a file named `CON`, `CON.json`, `nul`, etc. fails with
  // ERROR_ACCESS_DENIED. The check is case-insensitive and applies to the
  // stem only (the portion before the first `.`). On Linux/macOS these are
  // ordinary filenames, but we suffix unconditionally to keep dump output
  // portable across hosts.
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
