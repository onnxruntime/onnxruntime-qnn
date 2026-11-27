// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared CumSum specifications for focused graph and accuracy tiers.

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

// Exact legacy input literals, shared by the Accuracy and Snapshot setup.
inline const std::vector<float> kCumSumInputData = {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f};

enum class CumSumDataType {
  Float,
  UInt8,
};

struct CumSumSpec {
  const char* name;
  CumSumDataType data_type;
  int64_t axis;
  bool axis_is_int64;
};

// Maps 1:1 to the enabled HTP CumSum legacy contracts. QNN CPU behavior is
// intentionally excluded because that backend is being retired; these specs
// retain the common HTP builder, graph, assignment, and numerical contracts.
inline const std::vector<CumSumSpec> kCumSumSpecs = {
    {"CumSum_F32_Int32_Axis0", CumSumDataType::Float, 0, false},
    {"CumSum_F32_Int32_AxisNeg1", CumSumDataType::Float, -1, false},
    {"CumSum_F32_Int64_Axis1", CumSumDataType::Float, 1, true},
    {"CumSum_U8_Int32_Axis0", CumSumDataType::UInt8, 0, false},
};

}  // namespace test
}  // namespace onnxruntime
