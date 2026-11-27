// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared spec literals for GlobalMaxPool tier-based tests.

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

enum class GlobalMaxPoolDataType {
  UInt8,
  UInt16,
};

struct GlobalMaxPoolSpec {
  const char* name;
  std::vector<int64_t> input_shape;
  GlobalMaxPoolDataType data_type;
  bool use_contrib_qdq;
  // Use ModelTestBuilder random input (seed 2345), as in the legacy case.
  bool use_random_input = false;
};

// Maps 1:1 to enabled QnnHTPBackendTests GlobalMaxPool cases in
// maxpool_test.cc. QNN CPU GlobalMaxPool coverage is intentionally excluded:
// QNN CPU is a retiring backend, while the common HTP graph behavior is
// preserved by these backend-independent Session Snapshot and Accuracy specs.
inline const std::vector<GlobalMaxPoolSpec> kGlobalMaxPoolSpecs = {
    {"GlobalMaxPool_U8", {1, 2, 3, 3}, GlobalMaxPoolDataType::UInt8, false},
    {"GlobalMaxPool_U16", {1, 2, 3, 3}, GlobalMaxPoolDataType::UInt16, true},
    {"GlobalMaxPool_U8_LargeInput", {1, 128, 16, 113}, GlobalMaxPoolDataType::UInt8, false, true},
    {"GlobalMaxPool_U8_LargeInput2", {1, 64, 384, 576}, GlobalMaxPoolDataType::UInt8, false, true},
    {"GlobalMaxPool_U8_Rank3", {1, 8, 5}, GlobalMaxPoolDataType::UInt8, false},
    {"GlobalMaxPool_U8_Rank3LargeInput", {1, 8400, 80}, GlobalMaxPoolDataType::UInt8, false, true},
    {"GlobalMaxPool_U16_Rank3", {1, 8, 5}, GlobalMaxPoolDataType::UInt16, true},
};

}  // namespace test
}  // namespace onnxruntime
