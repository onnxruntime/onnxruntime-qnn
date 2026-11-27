// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include "test/unittest_util/model_test_builder.h"

namespace onnxruntime::test {

struct ModSpec {
  const char* name;
  bool divisor_is_initializer;
  std::vector<int64_t> dividend;
  std::vector<int64_t> divisor;
};

inline const std::vector<ModSpec>& GetModSpecs() {
  static const std::vector<ModSpec> specs = [] {
    // Preserve the legacy generator, ranges and three successive draws.
    RandomValueGenerator random{std::optional<RandomValueGenerator::RandomSeedType>{2345}};
    auto dividend = random.Uniform<int64_t>(std::vector<int64_t>{1, 4, 5}, -100.0f, 100.0f);
    auto positive = random.Uniform<int64_t>(std::vector<int64_t>{1, 5}, 1.0f, 10.0f);
    auto negative = random.Uniform<int64_t>(std::vector<int64_t>{1, 5}, -10.0f, -1.0f);
    return std::vector<ModSpec>{
        {"Dynamic_PositiveDivisor", false, dividend, positive},
        {"Dynamic_NegativeDivisor", false, dividend, negative},
        {"Static_PositiveDivisor", true, dividend, positive},
        {"Static_NegativeDivisor", true, dividend, negative},
    };
  }();
  return specs;
}

}  // namespace onnxruntime::test
