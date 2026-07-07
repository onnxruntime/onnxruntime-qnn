// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Session-level snapshot tests for Conv / ConvTranspose.
//
// Conv/ConvTranspose require a pre-partition NCHW→NHWC layout transform
// (transform_layout_fn) before ConvOpBuilder receives the node — this
// transform only runs inside a full ORT InferenceSession. Op-builder-level
// snapshots (AssertSnapshotJson) bypass the session entirely and therefore
// cannot drive Conv correctly. Session-level snapshots are the primary
// snapshot layer for all layout-sensitive ops.
//
// Separate translation unit (no qnn_unit_test_utils.h / ort_api.h) because
// session_snapshot.h pulls full ORT internal headers (core/graph/constants.h
// etc.) that double-define kOnnxDomain against the QNN-EP-internal copy.
//
// All 54 tests (48 enabled + 6 DISABLED) reference conv_specs.h constants
// through BuildConvModelFn / BuildConvFusionModelFn to guarantee zero drift
// between snapshot and accuracy tiers.
//
// Golden files: unit/goldens/builder/opbuilder/conv_session/<test_name>.json
// To generate/update goldens:
//   QNN_UPDATE_GOLDENS=1 ./onnxruntime_provider_test
//     --gtest_filter='QnnUnit_SessionSnapshot_ConvTest.*'

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/unit/session_snapshot.h"

#include "test/providers/qnn/unit/builder/opbuilder/conv_model_builders.h"

namespace onnxruntime {
namespace test {

namespace {

// Dispatch ConvSpec → snapshot assertion.
// F32 specs (quant_mode == None) use compile-only AssertSessionSnapshotJson.
// QDQ specs run CPU F32 inference first to get correct output_qparams.
// Mirrors RunConvAccuracy in conv_accuracy_test.cc.
void RunConvSnapshotQDQ(const ConvSpec& s) {
  if (s.quant_mode == ConvQuantMode::None) {
    AssertSessionSnapshotJson(BuildConvModelFn(s), BackendOptions(s.snapshot_backend),
                              s.opset, std::string(s.name));
    return;
  }

  const auto f32_fn = BuildConvF32ReferenceFn(s);
  const auto opts = BackendOptions(s.snapshot_backend);

  if (s.input_type == ConvInputType::U8) {
    if (s.weight_type == ConvWeightType::U8)
      AssertSessionSnapshotJsonQDQ<uint8_t>(
          f32_fn, BuildConvQDQFn<uint8_t, uint8_t>(s), opts, s.opset, std::string(s.name));
    else if (s.weight_type == ConvWeightType::S8)
      AssertSessionSnapshotJsonQDQ<uint8_t>(
          f32_fn, BuildConvQDQFn<uint8_t, int8_t>(s), opts, s.opset, std::string(s.name));
    else if (s.weight_type == ConvWeightType::S4)
      AssertSessionSnapshotJsonQDQ<uint8_t>(
          f32_fn, BuildConvQDQFn<uint8_t, Int4x2>(s), opts, s.opset, std::string(s.name));
  } else if (s.input_type == ConvInputType::U16) {
    if (s.weight_type == ConvWeightType::U8)
      AssertSessionSnapshotJsonQDQ<uint16_t>(
          f32_fn, BuildConvQDQFn<uint16_t, uint8_t>(s), opts, s.opset, std::string(s.name));
    else if (s.weight_type == ConvWeightType::S8)
      AssertSessionSnapshotJsonQDQ<uint16_t>(
          f32_fn, BuildConvQDQFn<uint16_t, int8_t>(s), opts, s.opset, std::string(s.name));
    else if (s.weight_type == ConvWeightType::S4)
      AssertSessionSnapshotJsonQDQ<uint16_t>(
          f32_fn, BuildConvQDQFn<uint16_t, Int4x2>(s), opts, s.opset, std::string(s.name));
    else if (s.weight_type == ConvWeightType::S16)
      AssertSessionSnapshotJsonQDQ<uint16_t>(
          f32_fn, BuildConvQDQFn<uint16_t, int16_t>(s), opts, s.opset, std::string(s.name));
    else if (s.weight_type == ConvWeightType::U16)
      AssertSessionSnapshotJsonQDQ<uint16_t>(
          f32_fn, BuildConvQDQFn<uint16_t, uint16_t>(s), opts, s.opset, std::string(s.name));
  }
}

// ORT_ENABLE_BASIC prevents ConvActivationFusion on the CPU F32 reference
// session (same reason as RunConvFusionAccuracy in conv_accuracy_test.cc).
void RunConvFusionSnapshotQDQ(const ConvFusionSpec& s) {
  const auto f32_fn = BuildConvFusionF32ReferenceFn(s);
  const auto opts = BackendOptions(s.snapshot_backend);
  const auto opt_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
  if (s.input_type == ConvFusionInputType::U8)
    AssertSessionSnapshotJsonQDQ<uint8_t>(
        f32_fn, BuildConvFusionQDQFn<uint8_t>(s), opts, s.opset, std::string(s.name),
        "", opt_level);
  else
    AssertSessionSnapshotJsonQDQ<int8_t>(
        f32_fn, BuildConvFusionQDQFn<int8_t>(s), opts, s.opset, std::string(s.name),
        "", opt_level);
}

}  // namespace

// ===========================================================================
// Phase A: CPU F32 (13 tests)
// ===========================================================================

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv2D_f32_DynamicBias) {
  const auto& s = kConv2D_f32_DynamicBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv2D_f32_StaticBias) {
  const auto& s = kConv2D_f32_StaticBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv2D_f32_AutoPadSameUpper) {
  const auto& s = kConv2D_f32_AutoPadSameUpper;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose2D_f32_AutoPadSameUpper) {
  const auto& s = kConvTranspose2D_f32_AutoPadSameUpper;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv2D_f32_AutoPadSameLower) {
  const auto& s = kConv2D_f32_AutoPadSameLower;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose2D_f32_AutoPadSameLower) {
  const auto& s = kConvTranspose2D_f32_AutoPadSameLower;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose3D_f32_AutoPadSameLower) {
  const auto& s = kConvTranspose3D_f32_AutoPadSameLower;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv2D_f32_LargePads) {
  const auto& s = kConv2D_f32_LargePads;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv2D_f32_LargeInput) {
  const auto& s = kConv2D_f32_LargeInput;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv1D_f32_StaticWeights) {
  const auto& s = kConv1D_f32_StaticWeights;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv1D_f32_DynamicWeights) {
  const auto& s = kConv1D_f32_DynamicWeights;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose1D_f32_StaticWeights) {
  const auto& s = kConvTranspose1D_f32_StaticWeights;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose1D_f32_DynamicWeights) {
  const auto& s = kConvTranspose1D_f32_DynamicWeights;
  RunConvSnapshotQDQ(s);
}

// ===========================================================================
// Phase B: HTP per-channel (U8/S8, U16/S8, U16/S4)
// ===========================================================================

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU8S8S32_PerChannel) {
  const auto& s = kConvU8S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv3D_U8S8S32_PerChannel) {
  const auto& s = kConv3D_U8S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU16S4S32_PerChannel) {
  const auto& s = kConvU16S4S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU16S4S32_PerChannel_NegativeWeightQuantAxis) {
  const auto& s = kConvU16S4S32_PerChannel_NegativeWeightQuantAxis;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTransposeU8S8S32_PerChannel) {
  const auto& s = kConvTransposeU8S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose3D_U8S8S32_PerChannel) {
  const auto& s = kConvTranspose3D_U8S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvDepthwiseU8S8S32_PerChannel) {
  const auto& s = kConvDepthwiseU8S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv3D_U8S8S32_PerChannel2) {
  const auto& s = kConv3D_U8S8S32_PerChannel2;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU16S8S32_PerChannel) {
  const auto& s = kConvU16S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv3D_U16S8S32_PerChannel) {
  const auto& s = kConv3D_U16S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTransposeU16S8S32_PerChannel) {
  const auto& s = kConvTransposeU16S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose3D_U16S8S32_PerChannel) {
  const auto& s = kConvTranspose3D_U16S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvDepthwiseU16S8S32_PerChannel) {
  const auto& s = kConvDepthwiseU16S8S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv3D_U16S8S32_PerChannel2) {
  const auto& s = kConv3D_U16S8S32_PerChannel2;
  RunConvSnapshotQDQ(s);
}

// DISABLED: x86 HTP emulator does not support U16+S16 Conv.
TEST(QnnUnit_SessionSnapshot_ConvTest, DISABLED_ConvU16S16S32_PerChannel) {
  const auto& s = kDISABLED_ConvU16S16S32_PerChannel;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU16S4_PerChannel_NoBias) {
  const auto& s = kConvU16S4_PerChannel_NoBias;
  RunConvSnapshotQDQ(s);
}

// ===========================================================================
// Phase 4: HTP KEEP — U8/U8 per-tensor, U16/U8 per-tensor, dynamic weight
// ===========================================================================

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU8U8S32_bias_dynamic_input) {
  const auto& s = kConvU8U8S32_bias_dynamic_input;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU8U8S32_BiasRequantization) {
  const auto& s = kConvU8U8S32_BiasRequantization;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU8U8S32_LargeInput_Dilations_Pads) {
  const auto& s = kConvU8U8S32_LargeInput_Dilations_Pads;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv1DU8U8S32_AutoPadUpper) {
  const auto& s = kConv1DU8U8S32_AutoPadUpper;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose1DU8U8S32_AutoPadLower) {
  const auto& s = kConvTranspose1DU8U8S32_AutoPadLower;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU8U8S32_AutoPadValid) {
  const auto& s = kConvU8U8S32_AutoPadValid;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTransposeU8U8S32_OutputShape) {
  const auto& s = kConvTransposeU8U8S32_OutputShape;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU8U8S32_DynamicWeight_NoBias) {
  const auto& s = kConvU8U8S32_DynamicWeight_NoBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTransposeU8U8S32_DynamicWeight_NoBias) {
  const auto& s = kConvTransposeU8U8S32_DynamicWeight_NoBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, Conv3D_U8U8S32_DynamicWeight_NoBias) {
  const auto& s = kConv3D_U8U8S32_DynamicWeight_NoBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvTranspose3D_U8U8S32_DynamicWeight_NoBias) {
  const auto& s = kConvTranspose3D_U8U8S32_DynamicWeight_NoBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU16U8_PerTensor_NoBias) {
  const auto& s = kConvU16U8_PerTensor_NoBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU16U8S32_StaticBias) {
  const auto& s = kConvU16U8S32_StaticBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU16U8S32_DynamicBias) {
  const auto& s = kConvU16U8S32_DynamicBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU16U8S32_NoBias) {
  const auto& s = kConvU16U8S32_NoBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, DepthwiseConvU16U8S32_StaticBias) {
  const auto& s = kDepthwiseConvU16U8S32_StaticBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, DepthwiseConvU16U8S32_DynamicBias) {
  const auto& s = kDepthwiseConvU16U8S32_DynamicBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, DepthwiseConvU16U8S32_NoBias) {
  const auto& s = kDepthwiseConvU16U8S32_NoBias;
  RunConvSnapshotQDQ(s);
}

// DISABLED: U16+U16 — guarded #ifndef __linux__ in integration test.
TEST(QnnUnit_SessionSnapshot_ConvTest, DISABLED_ConvU16U16_PerTensor_NoBias) {
  const auto& s = kDISABLED_ConvU16U16_PerTensor_NoBias;
  RunConvSnapshotQDQ(s);
}

// DISABLED: x86 HTP emulator does not support U16+S16 Conv.
TEST(QnnUnit_SessionSnapshot_ConvTest, DISABLED_ConvU16S16S32_DynamicBias) {
  const auto& s = kDISABLED_ConvU16S16S32_DynamicBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, DISABLED_DepthwiseConvU16S16S32_DynamicBias) {
  const auto& s = kDISABLED_DepthwiseConvU16S16S32_DynamicBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, DISABLED_ConvU16S16S32_NoBias) {
  const auto& s = kDISABLED_ConvU16S16S32_NoBias;
  RunConvSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, DISABLED_DepthwiseConvU16S16S32_NoBias) {
  const auto& s = kDISABLED_DepthwiseConvU16S16S32_NoBias;
  RunConvSnapshotQDQ(s);
}

// ===========================================================================
// Fusion tests: Conv + Relu / redundant Clip (3 tests)
// ===========================================================================

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU8U8S32_ReluClipFusion) {
  const auto& s = kConvU8U8S32_ReluClipFusion;
  RunConvFusionSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvU8U8S32_RedundantClipQDQ) {
  const auto& s = kConvU8U8S32_RedundantClipQDQ;
  RunConvFusionSnapshotQDQ(s);
}

TEST(QnnUnit_SessionSnapshot_ConvTest, ConvS8S8S32_PerChannel_ReluClipFusion) {
  const auto& s = kConvS8S8S32_PerChannel_ReluClipFusion;
  RunConvFusionSnapshotQDQ(s);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
