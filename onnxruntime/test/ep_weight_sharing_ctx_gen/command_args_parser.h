// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <core/session/onnxruntime_c_api.h>

namespace onnxruntime {
namespace qnnctxgen {

struct TestConfig;

inline std::string ToUTF8String(const std::string& s) { return s; }
#ifdef _WIN32
/**
 * Convert a wide character string to a UTF-8 string
 */
std::string ToUTF8String(std::wstring_view s);
inline std::string ToUTF8String(const wchar_t* s) {
  return ToUTF8String(std::wstring_view{s});
}
inline std::string ToUTF8String(const std::wstring& s) {
  return ToUTF8String(std::wstring_view{s});
}
#endif

class CommandLineParser {
 public:
  static void ShowUsage();
  static bool ParseArguments(TestConfig& test_config, int argc, ORTCHAR_T* argv[]);
};

}  // namespace qnnctxgen
}  // namespace onnxruntime
