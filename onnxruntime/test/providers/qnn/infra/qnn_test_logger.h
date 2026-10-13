// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
