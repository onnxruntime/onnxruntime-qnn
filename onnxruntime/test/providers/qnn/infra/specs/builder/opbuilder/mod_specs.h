// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime::test {

struct ModSpec {
  const char* name;
  bool divisor_is_initializer;
  std::vector<int64_t> dividend;
  std::vector<int64_t> divisor;
};

inline const std::vector<int64_t> kModDividend = {
    -100, -93, -42, -17, -10,
    -9, -5, -1, 0, 1,
    5, 9, 10, 17, 42,
    93, 99, 7, -33, 64};

inline const std::vector<int64_t> kModPositiveDivisor = {3, 5, 7, 9, 10};
inline const std::vector<int64_t> kModNegativeDivisor = {-3, -5, -7, -9, -10};

inline const std::vector<ModSpec> kModSpecs = {
    {"Dynamic_PositiveDivisor", false, kModDividend, kModPositiveDivisor},
    {"Dynamic_NegativeDivisor", false, kModDividend, kModNegativeDivisor},
    {"Static_PositiveDivisor", true, kModDividend, kModPositiveDivisor},
    {"Static_NegativeDivisor", true, kModDividend, kModNegativeDivisor},
};

}  // namespace onnxruntime::test
