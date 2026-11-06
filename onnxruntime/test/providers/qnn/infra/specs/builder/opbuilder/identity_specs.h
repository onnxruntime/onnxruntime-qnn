// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared spec literals for Identity op-builder tests.

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime {
namespace test {

enum class IdentityDataKind {
  Float,
  Int32,
  QDQUint8,
};

struct IdentitySpec {
  const char* name;
  IdentityDataKind kind;
  std::vector<int64_t> shape;
  std::vector<float> float_data;
  std::vector<int32_t> int32_data;
  float input_scale;
  float output_scale;
  uint8_t zero_point;
  int opset;
};

// Mirrors QnnHTPBackendTests.IdentityU8. The slightly different output scale
// exercises IdentityOpBuilder's qparam-equality adjustment.
inline const IdentitySpec kIdentityU8QParamSnapRank4Spec = {
    "Identity_U8_QParamSnap_Rank4",
    IdentityDataKind::QDQUint8,
    {1, 3, 4, 4},
    {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
     0.8f, 0.9f, 1.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
     0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 0.1f, 0.2f, 0.3f,
     0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 0.0f,
     0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
     0.9f, 1.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f},
    {},
    0.01f,
    0.010001f,
    0,
    18};

inline const IdentitySpec kIdentityF32Rank4Spec = {
    "Identity_F32_Rank4",
    IdentityDataKind::Float,
    {1, 3, 4, 4},
    {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
     -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f,
     4.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
     3.0f, 4.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f,
     2.0f, 3.0f, 4.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
     1.0f, 2.0f, 3.0f, 4.0f, -4.0f, -3.0f, -2.0f, -1.0f},
    {},
    0.0f,
    0.0f,
    0,
    13};

inline const IdentitySpec kIdentityI32Rank4Spec = {
    "Identity_I32_Rank4",
    IdentityDataKind::Int32,
    {1, 3, 4, 4},
    {},
    {-100, -50, -1, 0, 1, 50, 100, -75,
     75, -25, 25, -10, 10, -5, 5, -2,
     2, -3, 3, -4, 4, -6, 6, -7,
     7, -8, 8, -9, 9, -11, 11, -12,
     12, -13, 13, -14, 14, -15, 15, -16,
     16, -17, 17, -18, 18, -19, 19, -20},
    0.0f,
    0.0f,
    0,
    13};

inline const IdentitySpec kIdentityF32Rank1Spec = {
    "Identity_F32_Rank1",
    IdentityDataKind::Float,
    {16},
    {-8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
     0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
    {},
    0.0f,
    0.0f,
    0,
    13};

inline const std::vector<IdentitySpec> kIdentitySpecs = {
    kIdentityU8QParamSnapRank4Spec,
    kIdentityF32Rank4Spec,
    kIdentityI32Rank4Spec,
    kIdentityF32Rank1Spec,
};

}  // namespace test
}  // namespace onnxruntime
