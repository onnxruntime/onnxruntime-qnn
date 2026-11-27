// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

struct MeanInputSpec {
  std::vector<int64_t> shape;
  std::vector<float> data;
  // Empty data selects the legacy GetFloatDataInRange sequence, not random data.
  float range_min = 0.0f;
  float range_max = 0.0f;
};

struct MeanSpec {
  const char* name;
  bool qdq;
  bool enhancement;
  std::vector<MeanInputSpec> inputs;
  std::vector<int64_t> output_shape;
  int opset = 13;
};

// The first four scenarios map 1:1 to QnnHTPBackendTests.Mean_*.
inline const std::vector<MeanSpec> kMeanSpecs = {
    {"Mean_TwoInputs", false, false, {{{4}, {1.0f, 2.0f, 3.0f, 4.0f}}, {{4}, {5.0f, 6.0f, 7.0f, 8.0f}}}, {4}},
    {"Mean_FourInputs", false, false, {{{4}, {1.0f, 1.0f, 1.0f, 1.0f}}, {{4}, {2.0f, 2.0f, 2.0f, 2.0f}}, {{4}, {3.0f, 3.0f, 3.0f, 3.0f}}, {{4}, {4.0f, 4.0f, 4.0f, 4.0f}}}, {4}},
    {"Mean_TwoInputs_QU8", true, false, {{{1, 2, 2}, {}, 0.0f, 10.0f}, {{1, 2, 2}, {}, 10.0f, 20.0f}}, {1, 2, 2}},
    {"Mean_FourInputs_QU8", true, false, {{{1, 2, 2}, {}, 0.0f, 10.0f}, {{1, 2, 2}, {}, 10.0f, 20.0f}, {{1, 2, 2}, {}, 20.0f, 30.0f}, {{1, 2, 2}, {}, 30.0f, 40.0f}}, {1, 2, 2}},
    // Enhancement: exercise multidirectional broadcasting with nonconstant outputs.
    {"Mean_TwoInputs_Broadcast_F32", false, true, {{{2, 1}, {-2.0f, 4.0f}}, {{1, 3}, {1.0f, 3.0f, 5.0f}}}, {2, 3}},
};

}  // namespace test
}  // namespace onnxruntime
