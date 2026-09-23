// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared spec literals for Tile op-builder tests.
//
// One literal per migrated HTP QDQ case is shared by the op-builder snapshot
// tier and the paired accuracy tier. The `name` field is the single source of
// truth for the gtest case name and snapshot golden basename.

#pragma once

#include <cstdint>
#include <vector>

#include "onnxruntime_c_api.h"

namespace onnxruntime {
namespace test {

enum class TileBackend { HTP };

struct TileQDQSpec {
  const char* name;
  TileBackend snapshot_backend;
  TileBackend accuracy_backend;
  ONNXTensorElementDataType qdq_dtype;
  float scale;
  uint32_t zero_point;
  std::vector<int64_t> input_shape;
  std::vector<float> input_data;
  std::vector<int64_t> repeats;
  int opset;
  bool use_contrib_qdq;
};

inline const TileQDQSpec kTileU8Rank4Spec = {
    "Tile_U8_Rank4",
    TileBackend::HTP,
    TileBackend::HTP,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    0.1f,
    128,
    {1, 2, 2, 2},
    {-4.0f, -3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f},
    {1, 2, 1, 1},
    /*opset=*/13,
    /*use_contrib_qdq=*/false};

inline const TileQDQSpec kTileU16Rank4Spec = {
    "Tile_U16_Rank4",
    TileBackend::HTP,
    TileBackend::HTP,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    0.1f,
    32768,
    {1, 2, 2, 2},
    {-4.0f, -3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f},
    {1, 2, 1, 1},
    /*opset=*/13,
    /*use_contrib_qdq=*/true};

inline const std::vector<TileQDQSpec> kTileQDQSpecs = {
    kTileU8Rank4Spec,
    kTileU16Rank4Spec,
};

}  // namespace test
}  // namespace onnxruntime
