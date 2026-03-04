// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <core/session/onnxruntime_c_api.h>

namespace onnxruntime {
namespace qnnctxgen {

// char type for filesystem paths
using PathChar = ORTCHAR_T;
// string type for filesystem paths
using PathString = std::basic_string<PathChar>;

struct TestConfig;

// QNN-EP COPY START
// Below are string utilities copied from MS onnxruntime\core\common\common.h directly.
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
// QNN-EP COPY END

// QNN-EP COPY START
// Below are string utilities copied from MS onnxruntime\core\common\common.h directly.
#ifdef _WIN32
std::wstring ToWideString(std::string_view s);
inline std::wstring ToWideString(const char* s) {
  return ToWideString(std::string_view{s});
}
inline std::wstring ToWideString(const std::string& s) {
  return ToWideString(std::string_view{s});
}
inline std::wstring ToWideString(const std::wstring& s) { return s; }
inline std::wstring ToWideString(std::wstring_view s) { return std::wstring{s}; }

static_assert(std::is_same<PathString, std::wstring>::value, "PathString is not std::wstring!");

inline PathString ToPathString(std::string_view s) {
  return ToWideString(s);
}
inline PathString ToPathString(const char* s) {
  return ToWideString(s);
}
inline PathString ToPathString(const std::string& s) {
  return ToWideString(s);
}
#else
inline std::string ToWideString(const std::string& s) { return s; }
inline std::string ToWideString(const char* s) { return s; }
inline std::string ToWideString(std::string_view s) { return std::string{s}; }

static_assert(std::is_same<PathString, std::string>::value, "PathString is not std::string!");

inline PathString ToPathString(const char* s) {
  return s;
}

inline PathString ToPathString(std::string_view s) {
  return PathString{s};
}
// QNN-EP COPY END
#endif

class CommandLineParser {
 public:
  static void ShowUsage();
  static bool ParseArguments(TestConfig& test_config, int argc, ORTCHAR_T* argv[]);
};

}  // namespace qnnctxgen
}  // namespace onnxruntime
