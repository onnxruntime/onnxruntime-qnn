// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {

constexpr uint32_t kUInt16QMax = std::numeric_limits<uint16_t>::max();
constexpr double kReplacementMaskMagnitude = 100.0;

TEST(AttnMaskValueReductionFusionTests, DerivesReplacementEncodingForDifferentMaskMagnitudes) {
  struct TestCase {
    double original_mask_magnitude;
    float original_scale;
    int32_t original_offset;
  };

  constexpr std::array<TestCase, 2> test_cases{{{1000.0, 0.025f, -40000}, {10000.0, 0.25f, -40000}}};
  for (const TestCase& test_case : test_cases) {
    float replacement_scale = 0.0f;
    int32_t replacement_offset = 0;
    const double old_min = static_cast<double>(test_case.original_offset) * test_case.original_scale;
    const double old_max = (static_cast<double>(kUInt16QMax) + test_case.original_offset) * test_case.original_scale;
    const double new_min = old_min + (test_case.original_mask_magnitude - kReplacementMaskMagnitude);
    ASSERT_TRUE(qnn::DeriveUInt16EncodingWithMin(test_case.original_scale, test_case.original_offset, new_min,
                                                 replacement_scale, replacement_offset));

    const double expected_scale = (old_max - new_min) / kUInt16QMax;
    const int32_t expected_offset = -static_cast<int32_t>(std::lround(-new_min / expected_scale));
    EXPECT_NEAR(replacement_scale, expected_scale, 1e-7);
    EXPECT_EQ(replacement_offset, expected_offset);

    // The replacement encoding preserves the original upper real-value bound.
    const double replacement_max = (static_cast<double>(kUInt16QMax) + replacement_offset) * replacement_scale;
    EXPECT_NEAR(replacement_max, old_max, replacement_scale);
  }
}

TEST(AttnMaskValueReductionFusionTests, RejectsNonPositiveScale) {
  float replacement_scale = 0.0f;
  int32_t replacement_offset = 0;
  EXPECT_FALSE(qnn::DeriveUInt16EncodingWithMin(0.0f, 0, 0.0,
                                                replacement_scale, replacement_offset));
}

}  // namespace
}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD)
