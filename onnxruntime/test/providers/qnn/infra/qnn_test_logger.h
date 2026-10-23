// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

inline constexpr uintptr_t kQnnTestLoggerToken = 1;

// Constructs an Ort::Logger whose cached severity is FATAL, so ORT_CXX_LOG
// short-circuits without dereferencing the null logger pointer.
inline Ort::Logger MakeNullLogger() {
  static_assert(sizeof(Ort::Logger) == 2 * sizeof(void*),
                "Ort::Logger layout changed - update MakeNullLogger()");
  Ort::Logger logger{std::nullptr_t{}};
  OrtLoggingLevel fatal = ORT_LOGGING_LEVEL_FATAL;
  std::memcpy(reinterpret_cast<char*>(&logger) + sizeof(const OrtLogger*),
              &fatal, sizeof(OrtLoggingLevel));
  return logger;
}

// Direct QnnBackendManager component tests bypass QnnEpFactory, which normally
// installs the process default logger used by a few Linux ARM64 backend paths.
// Initialize that provider-side logger with a FATAL/no-op test sink before a
// real HTP backend is loaded. The FATAL severity keeps subsequent logging from
// dereferencing the opaque token after the temporary API override is restored.
inline void EnsureQnnTestDefaultLogger() {
#if defined(__linux__) && defined(__aarch64__) && !defined(__ANDROID__)
  static OrtApi logger_api = Ort::GetApi();
  static const bool logger_initialized = []() {
    logger_api.Logger_GetLoggingSeverityLevel =
        [](const OrtLogger*, OrtLoggingLevel* out) noexcept -> OrtStatus* {
      *out = ORT_LOGGING_LEVEL_FATAL;
      return nullptr;
    };
    logger_api.Logger_LogMessage =
        [](const OrtLogger*, OrtLoggingLevel, const char*, const ORTCHAR_T*, int,
           const char*) noexcept -> OrtStatus* { return nullptr; };

    const OrtApi* original_api = Ort::detail::Global::Api();
    Ort::detail::Global::Api(&logger_api);
    OrtLoggingManager::SetDefaultLogger(
        reinterpret_cast<const OrtLogger*>(kQnnTestLoggerToken));
    (void)OrtLoggingManager::GetDefaultLogger();
    Ort::detail::Global::Api(original_api);
    return true;
  }();
  (void)logger_initialized;

  // Some component fixtures intentionally clear the process default logger.
  // Restore the test token whenever a real-backend context is created.
  OrtLoggingManager::SetDefaultLogger(
      reinterpret_cast<const OrtLogger*>(kQnnTestLoggerToken));
#endif
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
