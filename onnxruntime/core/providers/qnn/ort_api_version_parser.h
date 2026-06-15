// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <charconv>
#include <cstdint>
#include <string_view>
#include <system_error>

namespace onnxruntime::qnn::detail {

// Parse the host ORT version string ("1.X.Y") and return the API version (the
// minor component). Returns 0 if the string is null, malformed, or major != 1.
// The "minor == API version" contract is documented for ORT 1.x; rejecting
// non-1 majors keeps a hypothetical ORT 2.x from being silently misparsed.
inline uint32_t ParseRuntimeOrtApiVersion(const char* version_str) {
  if (version_str == nullptr) {
    return 0;
  }
  std::string_view sv{version_str};
  const char* end = sv.data() + sv.size();

  int major = 0;
  auto [p1, ec1] = std::from_chars(sv.data(), end, major);
  if (ec1 != std::errc{} || p1 == end || *p1 != '.' || major != 1) {
    return 0;
  }

  uint32_t minor = 0;
  auto [p2, ec2] = std::from_chars(p1 + 1, end, minor);
  if (ec2 != std::errc{}) {
    return 0;
  }
  return minor;
}

}  // namespace onnxruntime::qnn::detail
