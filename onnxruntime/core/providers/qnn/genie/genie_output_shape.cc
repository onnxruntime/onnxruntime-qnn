// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/genie/genie_output_shape.h"

#include <charconv>
#include <cctype>
#include <string_view>

namespace onnxruntime {

bool ParseGenieOutputShape(const char* output_config,
                           std::vector<int64_t>& output_shape) noexcept {
  output_shape.clear();
  if (output_config == nullptr) {
    return false;
  }

  try {
    const std::string_view output_info{output_config};
    const size_t first_bracket = output_info.find('[');
    const size_t second_bracket = output_info.find(']', first_bracket);
    if (first_bracket == std::string_view::npos || second_bracket == std::string_view::npos ||
        second_bracket <= first_bracket + 1) {
      return false;
    }

    std::string_view dimensions = output_info.substr(first_bracket + 1, second_bracket - first_bracket - 1);
    while (!dimensions.empty()) {
      const size_t comma = dimensions.find(',');
      std::string_view dimension = dimensions.substr(0, comma);
      while (!dimension.empty() && std::isspace(static_cast<unsigned char>(dimension.front()))) {
        dimension.remove_prefix(1);
      }
      while (!dimension.empty() && std::isspace(static_cast<unsigned char>(dimension.back()))) {
        dimension.remove_suffix(1);
      }

      int64_t value = 0;
      const auto [end, error] = std::from_chars(dimension.data(), dimension.data() + dimension.size(), value);
      if (dimension.empty() || error != std::errc{} || end != dimension.data() + dimension.size() || value <= 0) {
        output_shape.clear();
        return false;
      }
      output_shape.push_back(value);

      if (comma == std::string_view::npos) {
        break;
      }
      dimensions.remove_prefix(comma + 1);
      if (dimensions.empty()) {
        output_shape.clear();
        return false;
      }
    }

    if (output_shape.empty()) {
      return false;
    }

    // Genie omits the singleton sequence dimension from the logit output configuration.
    output_shape.insert(output_shape.begin() + 1, 1);
    return true;
  } catch (...) {
    output_shape.clear();
    return false;
  }
}

}  // namespace onnxruntime
