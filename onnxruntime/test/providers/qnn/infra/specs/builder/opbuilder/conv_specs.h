// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared spec literals for Conv/ConvTranspose op-builder tests.
//
// 74 spec constants (71 ConvSpec + 3 ConvFusionSpec) are referenced by:
//   * session-level snapshot tests  (conv_session_test.cc)
//   * paired session-routed accuracy tests (conv_accuracy_test.cc)
//
// Drift = 0 by construction: both sides reference the named constant, so a
// spec change automatically propagates to both tiers.
//
// This header pulls only <cstdint>, <string_view>, and <vector> so it can be
// included from the full-ORT-internals world (conv_accuracy_test.cc →
// qnn_test_utils.h) without triggering kOnnxDomain double-define issues.

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Backend enum (shared with clip_specs.h)
// ---------------------------------------------------------------------------

enum class SnapshotBackend { CPU,
                             HTP };

// ---------------------------------------------------------------------------
// TensorSpec: lightweight replacement for TestInputDef<float>.
//
// When fixed_data is non-empty it is used verbatim (e.g. 1D conv tests).
// Otherwise the builder calls GetFloatDataInRange(min_val, max_val, count)
// where count = product(shape).
// shape empty = this input is absent (no bias, no bias).
// ---------------------------------------------------------------------------

struct TensorSpec {
  std::vector<int64_t> shape;
  bool is_static = true;
  float min_val = 0.0f;
  float max_val = 1.0f;
  std::vector<float> fixed_data;  // verbatim when non-empty; overrides min/max
};

// ---------------------------------------------------------------------------
// Quantization-related enums
// ---------------------------------------------------------------------------

enum class ConvInputType { F32,
                           U8,
                           U16,
                           S8 };

enum class ConvWeightType { F32,
                            U8,
                            S8,
                            S4,
                            U16,
                            S16 };

enum class ConvQuantMode { None,
                           PerTensor,
                           PerChannel };

// ---------------------------------------------------------------------------
// ConvSpec: covers Conv and ConvTranspose, all quantization modes.
//
// cluster_rep == name  for KEEP tests and cluster representatives.
// cluster_rep == rep_name  for cluster members (accuracy tier only).
// ---------------------------------------------------------------------------

struct ConvSpec {
  // --- metadata ---
  std::string_view name;
  std::string_view cluster_rep;     // == name for non-cluster and reps
  SnapshotBackend snapshot_backend;
  SnapshotBackend accuracy_backend;
  int opset;

  // --- graph topology ---
  std::string_view op_type;          // "Conv" or "ConvTranspose"
  TensorSpec input;
  TensorSpec weights;
  TensorSpec bias;                   // shape empty = no bias
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  int64_t group = 1;
  std::string_view auto_pad = "NOTSET";
  std::vector<int64_t> output_shape;  // ConvTranspose only

  // --- quantization ---
  ConvInputType input_type = ConvInputType::F32;
  ConvWeightType weight_type = ConvWeightType::F32;
  ConvQuantMode quant_mode = ConvQuantMode::None;
  int64_t weight_quant_axis = 0;   // 0=Conv, 1=ConvTranspose, -4=negative axis test
  bool use_contrib_qdq = false;
  float bias_scale_multiplier = 1.0f;  // != 1.0 triggers requantization path
};

// ---------------------------------------------------------------------------
// ConvFusionSpec: Conv + Relu/Clip output activation.
//
// Graph topology differs from plain Conv (has an extra activation node), so
// a separate struct is used. Input/weight/bias shapes are fixed:
//   input={1,2,4,4}, weights={3,2,2,2}, bias={3}
//   data: GetFloatDataInRange(0.0f, 1.0f, ...) for input/weights/bias
// ---------------------------------------------------------------------------

enum class ConvFusionType { Relu,
                            ClipRedundant };

enum class ConvFusionInputType { U8,
                                 S8 };

struct ConvFusionSpec {
  std::string_view name;
  std::string_view cluster_rep;
  SnapshotBackend snapshot_backend;
  SnapshotBackend accuracy_backend;
  int opset;
  ConvFusionType fusion_type;
  ConvFusionInputType input_type;
  bool per_channel;     // false=per-tensor U8/U8, true=per-channel S8/S8
  bool use_contrib_qdq;
};

// ===========================================================================
// CPU F32 specs (13)
// All CPU tests: snapshot_backend=CPU, accuracy_backend=CPU, opset=13
// ===========================================================================

// Integration test: QnnCPUBackendTests.Convf32_dynamic_bias
inline const ConvSpec kConv2D_f32_DynamicBias = {
    .name = "Conv2D_f32_DynamicBias",
    .cluster_rep = "Conv2D_f32_DynamicBias",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 3, 3}, false, 0.0f, 10.0f},
    .weights = {{2, 1, 2, 2}, true, 0.0f, 1.0f},
    .bias = {{2}, false, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
};

// Integration test: QnnCPUBackendTests.Convf32_bias_initializer
inline const ConvSpec kConv2D_f32_StaticBias = {
    .name = "Conv2D_f32_StaticBias",
    .cluster_rep = "Conv2D_f32_StaticBias",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 3, 3}, false, 0.0f, 10.0f},
    .weights = {{2, 1, 2, 2}, true, 0.0f, 1.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
};

// Integration test: QnnCPUBackendTests.Convf32_AutoPadUpper
inline const ConvSpec kConv2D_f32_AutoPadSameUpper = {
    .name = "Conv2D_f32_AutoPadSameUpper",
    .cluster_rep = "Conv2D_f32_AutoPadSameUpper",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 3, 3}, false, -3.0f, 3.0f},
    .weights = {{2, 1, 2, 2}, true, -1.0f, 1.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "SAME_UPPER",
};

// Integration test: QnnCPUBackendTests.ConvTransposef32_AutoPadUpper
inline const ConvSpec kConvTranspose2D_f32_AutoPadSameUpper = {
    .name = "ConvTranspose2D_f32_AutoPadSameUpper",
    .cluster_rep = "ConvTranspose2D_f32_AutoPadSameUpper",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 1, 3, 3}, false, -3.0f, 3.0f},
    .weights = {{1, 2, 2, 2}, true, -1.0f, 1.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "SAME_UPPER",
};

// Integration test: QnnCPUBackendTests.Convf32_AutoPadLower
inline const ConvSpec kConv2D_f32_AutoPadSameLower = {
    .name = "Conv2D_f32_AutoPadSameLower",
    .cluster_rep = "Conv2D_f32_AutoPadSameLower",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 3, 3}, false, -3.0f, 3.0f},
    .weights = {{2, 1, 2, 2}, false, -1.0f, 1.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "SAME_LOWER",
};

// Integration test: QnnCPUBackendTests.ConvTransposef32_AutoPadLower
inline const ConvSpec kConvTranspose2D_f32_AutoPadSameLower = {
    .name = "ConvTranspose2D_f32_AutoPadSameLower",
    .cluster_rep = "ConvTranspose2D_f32_AutoPadSameLower",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 1, 3, 3}, false, -3.0f, 3.0f},
    .weights = {{1, 2, 2, 2}, false, -1.0f, 1.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "SAME_LOWER",
};

// Integration test: QnnCPUBackendTests.ConvTranspose3D_f32_AutoPadLower
inline const ConvSpec kConvTranspose3D_f32_AutoPadSameLower = {
    .name = "ConvTranspose3D_f32_AutoPadSameLower",
    .cluster_rep = "ConvTranspose3D_f32_AutoPadSameLower",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 1, 3, 3, 3}, false, -3.0f, 3.0f},
    .weights = {{1, 2, 2, 2, 2}, false, -1.0f, 1.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1, 1},
    .pads = {},
    .dilations = {1, 1, 1},
    .auto_pad = "SAME_LOWER",
};

// Integration test: QnnCPUBackendTests.Convf32_large_input1_pad_bias_initializer
inline const ConvSpec kConv2D_f32_LargePads = {
    .name = "Conv2D_f32_LargePads",
    .cluster_rep = "Conv2D_f32_LargePads",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 3, 60, 452}, false, 0.0f, 10.0f},
    .weights = {{16, 3, 3, 3}, true, 0.0f, 1.0f},
    .bias = {{16}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {1, 1, 1, 1},
    .dilations = {1, 1},
};

// Integration test: QnnCPUBackendTests.Convf32_large_input2_nopad_bias_initializer
inline const ConvSpec kConv2D_f32_LargeInput = {
    .name = "Conv2D_f32_LargeInput",
    .cluster_rep = "Conv2D_f32_LargeInput",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 32, 16, 113}, false, -3.0f, 3.0f},
    .weights = {{16, 32, 1, 1}, false, -1.0f, 1.0f},
    .bias = {{16}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
};

// 1D conv uses explicit data so both sides produce identical inputs.
// Integration test: QnnCPUBackendTests.Conv1Df32_StaticWeights_DefaultBias
inline const ConvSpec kConv1D_f32_StaticWeights = {
    .name = "Conv1D_f32_StaticWeights",
    .cluster_rep = "Conv1D_f32_StaticWeights",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{1, 2, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {0, 0},
    .dilations = {1},
};

// Integration test: QnnCPUBackendTests.Conv1Df32_DynamicWeights_DefaultBias
inline const ConvSpec kConv1D_f32_DynamicWeights = {
    .name = "Conv1D_f32_DynamicWeights",
    .cluster_rep = "Conv1D_f32_DynamicWeights",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{1, 2, 2}, false, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {},  // no bias
    .strides = {1},
    .pads = {0, 0},
    .dilations = {1},
};

// Integration test: QnnCPUBackendTests.ConvTranspose1Df32_StaticWeights_DefaultBias
inline const ConvSpec kConvTranspose1D_f32_StaticWeights = {
    .name = "ConvTranspose1D_f32_StaticWeights",
    .cluster_rep = "ConvTranspose1D_f32_StaticWeights",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{2, 1, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {0.0f}},
    .strides = {1},
    .pads = {0, 0},
    .dilations = {1},
};

// Integration test: QnnCPUBackendTests.ConvTranspose1Df32_DynamicWeights_DefaultBias
inline const ConvSpec kConvTranspose1D_f32_DynamicWeights = {
    .name = "ConvTranspose1D_f32_DynamicWeights",
    .cluster_rep = "ConvTranspose1D_f32_DynamicWeights",
    .snapshot_backend = SnapshotBackend::CPU,
    .accuracy_backend = SnapshotBackend::CPU,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{2, 1, 2}, false, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {0.0f}},
    .strides = {1},
    .pads = {0, 0},
    .dilations = {1},
};

// ===========================================================================
// HTP Cluster representatives (8)
// ===========================================================================

// Cluster: Conv1DU8U8S32_AutoPadUpper (4 members)
inline const ConvSpec kConv1DU8U8S32_AutoPadUpper = {
    .name = "Conv1DU8U8S32_AutoPadUpper",
    .cluster_rep = "Conv1DU8U8S32_AutoPadUpper",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{1, 2, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {},
    .dilations = {1},
    .auto_pad = "SAME_UPPER",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster: ConvTranspose1DU8U8S32_AutoPadLower (5 members)
inline const ConvSpec kConvTranspose1DU8U8S32_AutoPadLower = {
    .name = "ConvTranspose1DU8U8S32_AutoPadLower",
    .cluster_rep = "ConvTranspose1DU8U8S32_AutoPadLower",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{2, 1, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {},
    .dilations = {1},
    .auto_pad = "SAME_LOWER",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster: ConvU8U8S32_AutoPadValid (4 members)
inline const ConvSpec kConvU8U8S32_AutoPadValid = {
    .name = "ConvU8U8S32_AutoPadValid",
    .cluster_rep = "ConvU8U8S32_AutoPadValid",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 4, 4}, true, -1.0f, 1.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "VALID",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster: ConvTransposeU8U8S32_OutputShape (3 members)
inline const ConvSpec kConvTransposeU8U8S32_OutputShape = {
    .name = "ConvTransposeU8U8S32_OutputShape",
    .cluster_rep = "ConvTransposeU8U8S32_OutputShape",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 1, 4, 4}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 2, 2}, true, -1.0f, 1.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {2, 2},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .auto_pad = "SAME_UPPER",
    .output_shape = {6, 6},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster: ConvU8S8S32_PerChannel (2 members)
inline const ConvSpec kConvU8S8S32_PerChannel = {
    .name = "ConvU8S8S32_PerChannel",
    .cluster_rep = "ConvU8S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
};

// Cluster: ConvU16S4S32_PerChannel_NegativeWeightQuantAxis (2 members)
// weight DQ axis=-4 normalizes to 0 for a 4-D weight tensor.
inline const ConvSpec kConvU16S4S32_PerChannel_NegativeWeightQuantAxis = {
    .name = "ConvU16S4S32_PerChannel_NegativeWeightQuantAxis",
    .cluster_rep = "ConvU16S4S32_PerChannel_NegativeWeightQuantAxis",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, 0.0f, 1.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S4,
    .quant_mode = ConvQuantMode::PerChannel,
    .weight_quant_axis = -4,
};

// Cluster: ConvU8U8S32_LargeInput_Dilations_Pads (2 members)
inline const ConvSpec kConvU8U8S32_LargeInput_Dilations_Pads = {
    .name = "ConvU8U8S32_LargeInput_Dilations_Pads",
    .cluster_rep = "ConvU8U8S32_LargeInput_Dilations_Pads",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 3, 768, 1152}, false, 0.0f, 10.0f},
    .weights = {{64, 3, 7, 7}, true, -1.0f, 1.0f},
    .bias = {{64}, true, -1.0f, 1.0f},
    .strides = {2, 2},
    .pads = {3, 3, 3, 3},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster: ConvU16S4_PerChannel_NoBias (2 members)
inline const ConvSpec kConvU16S4_PerChannel_NoBias = {
    .name = "ConvU16S4_PerChannel_NoBias",
    .cluster_rep = "ConvU16S4_PerChannel_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, 0.0f, 1.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S4,
    .quant_mode = ConvQuantMode::PerChannel,
};

// ===========================================================================
// HTP Cluster members — accuracy tier only (16)
// ===========================================================================

// ---- Cluster: Conv1DU8U8S32_AutoPadUpper ----

// Cluster member of Conv1DU8U8S32_AutoPadUpper. Integration test: QnnHTPBackendTests.Conv1DU8U8S32_AutoPadLower
inline const ConvSpec kConv1DU8U8S32_AutoPadLower = {
    .name = "Conv1DU8U8S32_AutoPadLower",
    .cluster_rep = "Conv1DU8U8S32_AutoPadUpper",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{1, 2, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {},
    .dilations = {1},
    .auto_pad = "SAME_LOWER",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster member of Conv1DU8U8S32_AutoPadUpper. Integration test: QnnHTPBackendTests.Conv1DU8U8S32_AutoPadValid
inline const ConvSpec kConv1DU8U8S32_AutoPadValid = {
    .name = "Conv1DU8U8S32_AutoPadValid",
    .cluster_rep = "Conv1DU8U8S32_AutoPadUpper",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{1, 2, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {},
    .dilations = {1},
    .auto_pad = "VALID",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster member of Conv1DU8U8S32_AutoPadUpper. Integration test: QnnHTPBackendTests.Conv1DU8U8S32_bias_initializer
inline const ConvSpec kConv1DU8U8S32_bias_initializer = {
    .name = "Conv1DU8U8S32_bias_initializer",
    .cluster_rep = "Conv1DU8U8S32_AutoPadUpper",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{1, 2, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {0, 0},
    .dilations = {1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// ---- Cluster: ConvTranspose1DU8U8S32_AutoPadLower ----

// Cluster member of ConvTranspose1DU8U8S32_AutoPadLower. Integration test: QnnHTPBackendTests.ConvTranspose1DU8U8S32_AutoPadUpper
inline const ConvSpec kConvTranspose1DU8U8S32_AutoPadUpper = {
    .name = "ConvTranspose1DU8U8S32_AutoPadUpper",
    .cluster_rep = "ConvTranspose1DU8U8S32_AutoPadLower",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{2, 1, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {},
    .dilations = {1},
    .auto_pad = "SAME_UPPER",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster member of ConvTranspose1DU8U8S32_AutoPadLower. Integration test: QnnHTPBackendTests.ConvTranspose1DU8U8S32_AutoPadValid
inline const ConvSpec kConvTranspose1DU8U8S32_AutoPadValid = {
    .name = "ConvTranspose1DU8U8S32_AutoPadValid",
    .cluster_rep = "ConvTranspose1DU8U8S32_AutoPadLower",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{2, 1, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {},
    .dilations = {1},
    .auto_pad = "VALID",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster member of ConvTranspose1DU8U8S32_AutoPadLower. Integration test: QnnHTPBackendTests.ConvTranspose1DU8U8S32_bias_initializer
inline const ConvSpec kConvTranspose1DU8U8S32_bias_initializer = {
    .name = "ConvTranspose1DU8U8S32_bias_initializer",
    .cluster_rep = "ConvTranspose1DU8U8S32_AutoPadLower",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4}, false, 0.0f, 0.0f,
              {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
    .weights = {{2, 1, 2}, true, 0.0f, 0.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1},
    .pads = {0, 0},
    .dilations = {1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster member of ConvTranspose1DU8U8S32_AutoPadLower. Integration test: QnnHTPBackendTests.ConvTranspose1DU8U8S32_OutputShape
inline const ConvSpec kConvTranspose1DU8U8S32_OutputShape = {
    .name = "ConvTranspose1DU8U8S32_OutputShape",
    .cluster_rep = "ConvTranspose1DU8U8S32_AutoPadLower",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 1, 4}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 2}, true, -1.0f, 1.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {2},
    .pads = {0, 0},
    .dilations = {1},
    .auto_pad = "SAME_UPPER",
    .output_shape = {6},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// ---- Cluster: ConvU8U8S32_AutoPadValid ----

// Cluster member of ConvU8U8S32_AutoPadValid. Integration test: QnnHTPBackendTests.ConvU8U8S32_AutoPadUpper
inline const ConvSpec kConvU8U8S32_AutoPadUpper = {
    .name = "ConvU8U8S32_AutoPadUpper",
    .cluster_rep = "ConvU8U8S32_AutoPadValid",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 4, 4}, true, -1.0f, 1.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "SAME_UPPER",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster member of ConvU8U8S32_AutoPadValid. Integration test: QnnHTPBackendTests.ConvU8U8S32_AutoPadLower
inline const ConvSpec kConvU8U8S32_AutoPadLower = {
    .name = "ConvU8U8S32_AutoPadLower",
    .cluster_rep = "ConvU8U8S32_AutoPadValid",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 4, 4}, true, -1.0f, 1.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "SAME_LOWER",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster member of ConvU8U8S32_AutoPadValid. Integration test: QnnHTPBackendTests.ConvU8U8S32_bias_initializer
inline const ConvSpec kConvU8U8S32_bias_initializer = {
    .name = "ConvU8U8S32_bias_initializer",
    .cluster_rep = "ConvU8U8S32_AutoPadValid",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 3, 3}, true, -10.0f, 10.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {2.0f}},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// ---- Cluster: ConvTransposeU8U8S32_OutputShape ----

// Cluster member of ConvTransposeU8U8S32_OutputShape. Integration test: QnnHTPBackendTests.ConvTransposeU8U8S32_AutoPadLower
inline const ConvSpec kConvTransposeU8U8S32_AutoPadLower = {
    .name = "ConvTransposeU8U8S32_AutoPadLower",
    .cluster_rep = "ConvTransposeU8U8S32_OutputShape",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 1, 5, 5}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 4, 4}, true, -1.0f, 1.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "SAME_LOWER",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Cluster member of ConvTransposeU8U8S32_OutputShape. Integration test: QnnHTPBackendTests.ConvTransposeU8U8S32_AutoPadValid
inline const ConvSpec kConvTransposeU8U8S32_AutoPadValid = {
    .name = "ConvTransposeU8U8S32_AutoPadValid",
    .cluster_rep = "ConvTransposeU8U8S32_OutputShape",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 1, 5, 5}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 4, 4}, true, -1.0f, 1.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {1.0f}},
    .strides = {1, 1},
    .pads = {},
    .dilations = {1, 1},
    .auto_pad = "VALID",
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// ---- Cluster: ConvU8S8S32_PerChannel ----

// BiasRequantization: same graph as PerChannel but bias_scale_multiplier != 1
// (intentionally mis-scaled bias triggers requantization path in op-builder).
// Cluster member of ConvU8S8S32_PerChannel. Integration test: QnnHTPBackendTests.ConvU8S8S32_PerChannel_BiasRequantization
inline const ConvSpec kConvU8S8S32_PerChannel_BiasRequantization = {
    .name = "ConvU8S8S32_PerChannel_BiasRequantization",
    .cluster_rep = "ConvU8S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .bias_scale_multiplier = 0.5f,  // != 1.0 → requantization path
};

// ---- Cluster: ConvU16S4S32_PerChannel_NegativeWeightQuantAxis ----

// AccuracyIssue: explicit input/weight/bias data to reproduce the QNN accuracy issue.
// (Data is GetFloatDataInRange written out explicitly for reproducibility.)
// Cluster member of ConvU16S4S32_PerChannel_NegativeWeightQuantAxis. Integration test: QnnHTPBackendTests.ConvU16S4S32_PerChannel_AccuracyIssue
inline const ConvSpec kConvU16S4S32_PerChannel_AccuracyIssue = {
    .name = "ConvU16S4S32_PerChannel_AccuracyIssue",
    .cluster_rep = "ConvU16S4S32_PerChannel_NegativeWeightQuantAxis",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, 0.0f, 0.0f,
              {-10.000f, -9.355f, -8.710f, -8.065f, -7.419f, -6.774f, -6.129f, -5.484f,
               -4.839f, -4.194f, -3.548f, -2.903f, -2.258f, -1.613f, -0.968f, -0.323f,
               0.323f, 0.968f, 1.613f, 2.258f, 2.903f, 3.548f, 4.194f, 4.839f,
               5.484f, 6.129f, 6.774f, 7.419f, 8.065f, 8.710f, 9.355f, 10.000f}},
    .weights = {{3, 2, 2, 2}, true, 0.0f, 0.0f,
                {-1.000f, -0.913f, -0.826f, -0.739f, -0.652f, -0.565f, -0.478f, -0.391f,
                 -0.304f, -0.217f, -0.130f, -0.043f, 0.043f, 0.130f, 0.217f, 0.304f,
                 0.391f, 0.478f, 0.565f, 0.652f, 0.739f, 0.826f, 0.913f, 1.000f}},
    .bias = {{3}, true, 0.0f, 0.0f, {-1.000f, 0.000f, 1.000f}},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::S4,
    .quant_mode = ConvQuantMode::PerChannel,
};

// ---- Cluster: ConvU8U8S32_LargeInput_Dilations_Pads ----

// Cluster member of ConvU8U8S32_LargeInput_Dilations_Pads. Integration test: QnnHTPBackendTests.ConvU8U8S32_large_input2_bias_initializer
inline const ConvSpec kConvU8U8S32_large_input2_bias_initializer = {
    .name = "ConvU8U8S32_large_input2_bias_initializer",
    .cluster_rep = "ConvU8U8S32_LargeInput_Dilations_Pads",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 128, 8, 56}, false, 0.0f, 10.0f},
    .weights = {{32, 128, 1, 1}, true, -1.0f, 1.0f},
    .bias = {{32}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// ---- Cluster: ConvU16S4_PerChannel_NoBias ----

// Cluster member of ConvU16S4_PerChannel_NoBias. Integration test: QnnHTPBackendTests.ConvU16S4_PerChannel_NoBias_LargeINT4Weight
inline const ConvSpec kConvU16S4_PerChannel_NoBias_LargeINT4Weight = {
    .name = "ConvU16S4_PerChannel_NoBias_LargeINT4Weight",
    .cluster_rep = "ConvU16S4_PerChannel_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .op_type = "Conv",
    .input = {{1, 3072, 1, 512}, false, 0.0f, 1.0f},
    .weights = {{9216, 3072, 1, 1}, true, -1.0f, 5.0f},
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S4,
    .quant_mode = ConvQuantMode::PerChannel,
};

// ===========================================================================
// HTP KEEP — non-fusion (24)
// ===========================================================================

inline const ConvSpec kConvU8U8S32_bias_dynamic_input = {
    .name = "ConvU8U8S32_bias_dynamic_input",
    .cluster_rep = "ConvU8U8S32_bias_dynamic_input",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, 0.0f, 10.0f},
    .weights = {{1, 1, 3, 3}, true, -10.0f, 10.0f},
    .bias = {{1}, false, 0.0f, 0.0f, {2.0f}},  // dynamic bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

inline const ConvSpec kConvU8U8S32_BiasRequantization = {
    .name = "ConvU8U8S32_BiasRequantization",
    .cluster_rep = "ConvU8U8S32_BiasRequantization",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
    .bias_scale_multiplier = 10.0f,
};

inline const ConvSpec kConvU16U8_PerTensor_NoBias = {
    .name = "ConvU16U8_PerTensor_NoBias",
    .cluster_rep = "ConvU16U8_PerTensor_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, 0.0f, 1.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

inline const ConvSpec kConvU16S4S32_PerChannel = {
    .name = "ConvU16S4S32_PerChannel",
    .cluster_rep = "ConvU16S4S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, 0.0f, 1.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S4,
    .quant_mode = ConvQuantMode::PerChannel,
};

inline const ConvSpec kConv3D_U8S8S32_PerChannel = {
    .name = "Conv3D_U8S8S32_PerChannel",
    .cluster_rep = "Conv3D_U8S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 4, 3, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{8, 4, 3, 3, 3}, true, -1.0f, 5.0f},
    .bias = {{8}, true, -1.0f, 1.0f},
    .strides = {1, 1, 1},
    .pads = {0, 0, 0, 0, 0, 0},
    .dilations = {1, 1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
};

inline const ConvSpec kConvDepthwiseU8S8S32_PerChannel = {
    .name = "ConvDepthwiseU8S8S32_PerChannel",
    .cluster_rep = "ConvDepthwiseU8S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{2, 1, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .group = 2,
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
};

inline const ConvSpec kConv3D_U8S8S32_PerChannel2 = {
    .name = "Conv3D_U8S8S32_PerChannel2",
    .cluster_rep = "Conv3D_U8S8S32_PerChannel2",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{2, 1, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1, 1},
    .pads = {0, 0, 0, 0, 0, 0},
    .dilations = {1, 1, 1},
    .group = 2,
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
};

inline const ConvSpec kConvTransposeU8S8S32_PerChannel = {
    .name = "ConvTransposeU8S8S32_PerChannel",
    .cluster_rep = "ConvTransposeU8S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{2, 3, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .weight_quant_axis = 1,
};

inline const ConvSpec kConvTranspose3D_U8S8S32_PerChannel = {
    .name = "ConvTranspose3D_U8S8S32_PerChannel",
    .cluster_rep = "ConvTranspose3D_U8S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{2, 3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1, 1},
    .pads = {0, 0, 0, 0, 0, 0},
    .dilations = {1, 1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .weight_quant_axis = 1,
};

inline const ConvSpec kConvU16S8S32_PerChannel = {
    .name = "ConvU16S8S32_PerChannel",
    .cluster_rep = "ConvU16S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .use_contrib_qdq = true,
};

inline const ConvSpec kConv3D_U16S8S32_PerChannel = {
    .name = "Conv3D_U16S8S32_PerChannel",
    .cluster_rep = "Conv3D_U16S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{3, 2, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1, 1},
    .pads = {0, 0, 0, 0, 0, 0},
    .dilations = {1, 1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .use_contrib_qdq = true,
};

inline const ConvSpec kConvTransposeU16S8S32_PerChannel = {
    .name = "ConvTransposeU16S8S32_PerChannel",
    .cluster_rep = "ConvTransposeU16S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{2, 3, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .weight_quant_axis = 1,
    .use_contrib_qdq = true,
};

inline const ConvSpec kConvTranspose3D_U16S8S32_PerChannel = {
    .name = "ConvTranspose3D_U16S8S32_PerChannel",
    .cluster_rep = "ConvTranspose3D_U16S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 2, 4, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{2, 3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1, 1},
    .pads = {0, 0, 0, 0, 0, 0},
    .dilations = {1, 1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .weight_quant_axis = 1,
    .use_contrib_qdq = true,
};

inline const ConvSpec kConvDepthwiseU16S8S32_PerChannel = {
    .name = "ConvDepthwiseU16S8S32_PerChannel",
    .cluster_rep = "ConvDepthwiseU16S8S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{2, 1, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .group = 2,
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .use_contrib_qdq = true,
};

inline const ConvSpec kConv3D_U16S8S32_PerChannel2 = {
    .name = "Conv3D_U16S8S32_PerChannel2",
    .cluster_rep = "Conv3D_U16S8S32_PerChannel2",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{2, 1, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{2}, true, -1.0f, 1.0f},
    .strides = {1, 1, 1},
    .pads = {0, 0, 0, 0, 0, 0},
    .dilations = {1, 1, 1},
    .group = 2,
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S8,
    .quant_mode = ConvQuantMode::PerChannel,
    .use_contrib_qdq = true,
};

inline const ConvSpec kConvU16U8S32_StaticBias = {
    .name = "ConvU16U8S32_StaticBias",
    .cluster_rep = "ConvU16U8S32_StaticBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 5, 5}, false, 0.0f, 1.0f},   // BuildConvU16U8S32_2D_StaticBiasFn: 0–1
    .weights = {{1, 2, 3, 3}, true, -1.0f, 5.0f},
    .bias = {{1}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

inline const ConvSpec kConvU16U8S32_DynamicBias = {
    .name = "ConvU16U8S32_DynamicBias",
    .cluster_rep = "ConvU16U8S32_DynamicBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 2, 3, 3}, true, -1.0f, 5.0f},
    .bias = {{1}, false, 0.0f, 0.0f, {2.0f}},  // dynamic bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

inline const ConvSpec kConvU16U8S32_NoBias = {
    .name = "ConvU16U8S32_NoBias",
    .cluster_rep = "ConvU16U8S32_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 2, 3, 3}, true, -1.0f, 5.0f},
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

// Matches integration test QnnHTPBackendTests.ConvU8U8S32_DynamicWeight_NoBias (2D subcase).
inline const ConvSpec kConvU8U8S32_DynamicWeight_NoBias = {
    .name = "ConvU8U8S32_DynamicWeight_NoBias",
    .cluster_rep = "ConvU8U8S32_DynamicWeight_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 3, 32, 32}, false, -10.0f, 10.0f},
    .weights = {{1, 3, 4, 4}, false, -10.0f, 10.0f},  // dynamic weight
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Matches integration test QnnHTPBackendTests.ConvU8U8S32_DynamicWeight_NoBias (3D subcase).
// Uses QNN_OP_CONV_3D, a different code path from the 2D case above.
inline const ConvSpec kConv3D_U8U8S32_DynamicWeight_NoBias = {
    .name = "Conv3D_U8U8S32_DynamicWeight_NoBias",
    .cluster_rep = "Conv3D_U8U8S32_DynamicWeight_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 3, 32, 32, 32}, false, -10.0f, 10.0f},
    .weights = {{1, 3, 4, 4, 4}, false, -10.0f, 10.0f},  // dynamic weight
    .bias = {},  // no bias
    .strides = {1, 1, 1},
    .pads = {0, 0, 0, 0, 0, 0},
    .dilations = {1, 1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// Matches integration test QnnHTPBackendTests.ConvTransposeU8U8S32_DynamicWeight_NoBias.
inline const ConvSpec kConvTransposeU8U8S32_DynamicWeight_NoBias = {
    .name = "ConvTransposeU8U8S32_DynamicWeight_NoBias",
    .cluster_rep = "ConvTransposeU8U8S32_DynamicWeight_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 3, 32, 32}, false, -10.0f, 10.0f},
    .weights = {{3, 1, 4, 4}, false, -10.0f, 10.0f},  // dynamic weight
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

inline const ConvSpec kConvTranspose3D_U8U8S32_DynamicWeight_NoBias = {
    .name = "ConvTranspose3D_U8U8S32_DynamicWeight_NoBias",
    .cluster_rep = "ConvTranspose3D_U8U8S32_DynamicWeight_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "ConvTranspose",
    .input = {{1, 3, 32, 32, 32}, false, -10.0f, 10.0f},
    .weights = {{3, 1, 4, 4, 4}, false, -10.0f, 10.0f},  // dynamic weight
    .bias = {},  // no bias
    .strides = {1, 1, 1},
    .pads = {0, 0, 0, 0, 0, 0},
    .dilations = {1, 1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

inline const ConvSpec kDepthwiseConvU16U8S32_StaticBias = {
    .name = "DepthwiseConvU16U8S32_StaticBias",
    .cluster_rep = "DepthwiseConvU16U8S32_StaticBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 1, 3, 3}, true, -1.0f, 5.0f},
    .bias = {{1}, true, 0.0f, 0.0f, {2.0f}},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

inline const ConvSpec kDepthwiseConvU16U8S32_DynamicBias = {
    .name = "DepthwiseConvU16U8S32_DynamicBias",
    .cluster_rep = "DepthwiseConvU16U8S32_DynamicBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 1, 3, 3}, true, -1.0f, 5.0f},
    .bias = {{1}, false, 0.0f, 0.0f, {2.0f}},  // dynamic bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

inline const ConvSpec kDepthwiseConvU16U8S32_NoBias = {
    .name = "DepthwiseConvU16U8S32_NoBias",
    .cluster_rep = "DepthwiseConvU16U8S32_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 1, 3, 3}, true, -1.0f, 5.0f},
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

// ===========================================================================
// HTP DISABLED — S16 variants (6)
// These tests require Windows/Android hardware; disabled on Linux x86.
// Snapshots exist with DISABLED_ prefix; accuracy tests are also DISABLED.
// ===========================================================================

// DISABLED: ConvU16S16S32_PerChannel (not __linux__)
inline const ConvSpec kDISABLED_ConvU16S16S32_PerChannel = {
    .name = "ConvU16S16S32_PerChannel",
    .cluster_rep = "ConvU16S16S32_PerChannel",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, -10.0f, 10.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {{3}, true, -1.0f, 1.0f},
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S16,
    .quant_mode = ConvQuantMode::PerChannel,
    .use_contrib_qdq = true,
};

// DISABLED: ConvU16U16_PerTensor_NoBias (not __linux__)
inline const ConvSpec kDISABLED_ConvU16U16_PerTensor_NoBias = {
    .name = "ConvU16U16_PerTensor_NoBias",
    .cluster_rep = "ConvU16U16_PerTensor_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .op_type = "Conv",
    .input = {{1, 2, 4, 4}, false, 0.0f, 1.0f},
    .weights = {{3, 2, 2, 2}, true, -1.0f, 5.0f},
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::U16,
    .quant_mode = ConvQuantMode::PerTensor,
};

// DISABLED: ConvU16S16S32_DynamicBias (x86 HTP emulator unsupported)
inline const ConvSpec kDISABLED_ConvU16S16S32_DynamicBias = {
    .name = "ConvU16S16S32_DynamicBias",
    .cluster_rep = "ConvU16S16S32_DynamicBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 2, 3, 3}, false, -5.0f, 5.0f},  // dynamic weight
    .bias = {{1}, false, 0.0f, 0.0f, {2.0f}},        // dynamic bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S16,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

// DISABLED: DepthwiseConvU16S16S32_DynamicBias (x86 HTP emulator unsupported)
inline const ConvSpec kDISABLED_DepthwiseConvU16S16S32_DynamicBias = {
    .name = "DepthwiseConvU16S16S32_DynamicBias",
    .cluster_rep = "DepthwiseConvU16S16S32_DynamicBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 1, 3, 3}, false, -5.0f, 5.0f},  // dynamic weight
    .bias = {{1}, false, 0.0f, 0.0f, {2.0f}},        // dynamic bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S16,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

// DISABLED: ConvU16S16S32_NoBias (x86 HTP emulator unsupported)
inline const ConvSpec kDISABLED_ConvU16S16S32_NoBias = {
    .name = "ConvU16S16S32_NoBias",
    .cluster_rep = "ConvU16S16S32_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 2, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 2, 3, 3}, false, -5.0f, 5.0f},  // dynamic weight
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S16,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

// DISABLED: DepthwiseConvU16S16S32_NoBias (x86 HTP emulator unsupported)
inline const ConvSpec kDISABLED_DepthwiseConvU16S16S32_NoBias = {
    .name = "DepthwiseConvU16S16S32_NoBias",
    .cluster_rep = "DepthwiseConvU16S16S32_NoBias",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 1, 5, 5}, false, -10.0f, 10.0f},
    .weights = {{1, 1, 3, 3}, false, -10.0f, 10.0f},  // dynamic weight
    .bias = {},  // no bias
    .strides = {1, 1},
    .pads = {0, 0, 0, 0},
    .dilations = {1, 1},
    .input_type = ConvInputType::U16,
    .weight_type = ConvWeightType::S16,
    .quant_mode = ConvQuantMode::PerTensor,
    .use_contrib_qdq = true,
};

// DISABLED: ConvU8U8S32_large_input1_padding_bias_initializer
// No snapshot; kept as DISABLED accuracy spec to track the tolerance regression
// (QNN SDK 2.19.2 introduced 0.76% tolerance requirement).
inline const ConvSpec kDISABLED_ConvU8U8S32_large_input1_padding_bias_initializer = {
    .name = "ConvU8U8S32_large_input1_padding_bias_initializer",
    .cluster_rep = "ConvU8U8S32_large_input1_padding_bias_initializer",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .op_type = "Conv",
    .input = {{1, 3, 60, 452}, false, 0.0f, 10.0f},
    .weights = {{16, 3, 3, 3}, true, -1.0f, 1.0f},
    .bias = {{16}, true, 0.0f, 0.0f, std::vector<float>(16, 1.0f)},
    .strides = {1, 1},
    .pads = {1, 1, 1, 1},
    .dilations = {1, 1},
    .input_type = ConvInputType::U8,
    .weight_type = ConvWeightType::U8,
    .quant_mode = ConvQuantMode::PerTensor,
};

// ===========================================================================
// HTP Fusion specs (3) — ConvFusionSpec
//
// All fusion tests use fixed shapes: input={1,2,4,4}, weights={3,2,2,2}, bias={3}
// with GetFloatDataInRange(0.0f, 1.0f, ...) for all tensors.
// ===========================================================================

// Mirrors ConvU8U8S32_ReluClipFusion integration test.
// U8/U8 per-tensor Conv + Relu → fused into Conv2d with activation params.
inline const ConvFusionSpec kConvU8U8S32_ReluClipFusion = {
    .name = "ConvU8U8S32_ReluClipFusion",
    .cluster_rep = "ConvU8U8S32_ReluClipFusion",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .fusion_type = ConvFusionType::Relu,
    .input_type = ConvFusionInputType::U8,
    .per_channel = false,
    .use_contrib_qdq = false,
};

// Mirrors ConvU8U8S32_RedundantClipQDQ integration test.
// U8/U8 per-tensor Conv + Clip (bounds wider than activation range → redundant).
inline const ConvFusionSpec kConvU8U8S32_RedundantClipQDQ = {
    .name = "ConvU8U8S32_RedundantClipQDQ",
    .cluster_rep = "ConvU8U8S32_RedundantClipQDQ",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 13,
    .fusion_type = ConvFusionType::ClipRedundant,
    .input_type = ConvFusionInputType::U8,
    .per_channel = false,
    .use_contrib_qdq = true,
};

// Mirrors ConvS8S8S32_PerChannel_ReluClipFusion integration test.
// S8/S8 per-channel Conv + Relu.
inline const ConvFusionSpec kConvS8S8S32_PerChannel_ReluClipFusion = {
    .name = "ConvS8S8S32_PerChannel_ReluClipFusion",
    .cluster_rep = "ConvS8S8S32_PerChannel_ReluClipFusion",
    .snapshot_backend = SnapshotBackend::HTP,
    .accuracy_backend = SnapshotBackend::HTP,
    .opset = 21,
    .fusion_type = ConvFusionType::Relu,
    .input_type = ConvFusionInputType::S8,
    .per_channel = true,
    .use_contrib_qdq = false,
};

}  // namespace test
}  // namespace onnxruntime
