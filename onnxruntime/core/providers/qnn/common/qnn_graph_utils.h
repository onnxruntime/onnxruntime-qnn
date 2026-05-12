// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// QNN-EP COPY START
// Below are GSL utilities copied from core/common/span_utils.h directly.
// Below are constants copied from core/graph/constants.h directly.
#pragma once

#include <gsl/gsl>

namespace onnxruntime {

template <class U, class T>
[[nodiscard]] inline gsl::span<U> ReinterpretAsSpan(gsl::span<T> src) {
  // adapted from gsl-lite span::as_span():
  // https://github.com/gsl-lite/gsl-lite/blob/4720a2980a30da085b4ddb4a0ea2a71af7351a48/include/gsl/gsl-lite.hpp#L4102-L4108
  Expects(src.size_bytes() % sizeof(U) == 0);
  return gsl::span<U>(reinterpret_cast<U*>(src.data()), src.size_bytes() / sizeof(U));
}

inline constexpr const char* kOnnxDomain = "";
inline constexpr const char* kMSDomain = "com.microsoft";
inline constexpr const char* kMSInternalNHWCDomain = "com.ms.internal.nhwc";

}  // namespace onnxruntime
// QNN-EP COPY END
