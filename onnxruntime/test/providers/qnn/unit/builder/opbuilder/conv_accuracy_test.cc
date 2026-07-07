// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Session-routed accuracy tests for Conv / ConvTranspose.
//
// 1:1 mapping with QnnUnit_SessionSnapshot_ConvTest snapshot cases plus all
// cluster members (accuracy-only): every snapshot case has a paired
// QnnUnit_SessionAccuracy_ConvTest.<Case> here that runs the same ONNX
// graph end-to-end and diffs QNN EP output against an ORT CPU EP reference.
//
// Cluster members (specs whose cluster_rep != name) are accuracy-only —
// they share a snapshot with their cluster representative but each exercises
// a distinct ONNX attribute variation (auto_pad, output_shape, ...).
//
// Builder shared with conv_session_test.cc via conv_model_builders.h to
// guarantee zero drift: changing a spec constant in conv_specs.h propagates
// to both snapshot and accuracy tiers automatically.
//
// Gated on QNN_EP_ACCURACY_LIGHT_UT (requires ENABLE_COVERAGE=ON on Linux
// x86_64 — see cmake/onnxruntime_unittests.cmake for details).
//
// Tolerance policy (aligns with integration test's TestQDQModelAccuracy):
//   F32 CPU backend: 1e-4f  (conv accumulation can drift slightly)
//   F32 HTP backend: 5e-3f  (HTP converts FP32→FP16 internally since 2.35)
//   QDQ (any backend): QDQTolerance() = 0.4% of output range, or passes if
//     QNN EP is at least as accurate as CPU EP (BiasRequantization tests pass
//     via this "is_as_accurate_as_cpu_ep" check even with bsm != 1.0).
//
// Per-channel bias fix in BuildConvQDQFn: uses per-channel scales
//   bias_scale[i] = in_scale * w_scales[i]  (vs single w_scales[0] in snapshot).
// This eliminates the ~8.0f absolute error from QNN EP's per-channel
// requantization of a mis-scaled bias and allows 0.4% relative tolerance.
//
// Golden files: unit/goldens/builder/opbuilder/conv_session/<test_name>.json
// (snapshot tier only; accuracy tests have no golden files of their own).

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_LIGHT_UT

#include <chrono>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/unit/builder/opbuilder/conv_model_builders.h"

namespace onnxruntime {
namespace test {

namespace {

// F32 specs use RunQnnModelTest (no quantization, absolute threshold).
// 1e-4f: integration test uses 1e-5f, but its F32 CPU tests use small shapes
// (e.g., {1,1,3,3}). Our Conv2D_f32_LargePads ({1,3,60,452}, 27 MADs/output)
// accumulates ~1.14e-5 float error between ORT and QNN CPU EP due to different
// summation order, so 1e-5f is too tight for large shapes.
inline float ConvF32Tol(const ConvSpec& /*s*/) {
  return 1e-4f;
}

// QDQ specs: dispatch ConvSpec → correct TestQDQModelAccuracy<AType> instantiation.
// Tolerance: QDQTolerance() = 0.4% of output range (integration-test parity).
// BiasRequantization tests (bsm != 1.0) pass via is_as_accurate_as_cpu_ep check.
void RunConvAccuracy(const ConvSpec& s) {
  if (s.quant_mode == ConvQuantMode::None) {
    RunQnnModelTest(BuildConvModelFn(s), BackendOptions(s.accuracy_backend), s.opset,
                    EPVerificationParams{ExpectedEPNodeAssignment::All,
                                         ElementwiseAbsoluteVerifier(ConvF32Tol(s))});
    return;
  }

  const auto f32_fn = BuildConvF32ReferenceFn(s);
  const auto opts = BackendOptions(s.accuracy_backend);

  if (s.input_type == ConvInputType::U8) {
    if (s.weight_type == ConvWeightType::U8)
      TestQDQModelAccuracy<uint8_t>(f32_fn, BuildConvQDQFn<uint8_t, uint8_t>(s),
                                    opts, s.opset, ExpectedEPNodeAssignment::All);
    else if (s.weight_type == ConvWeightType::S8)
      TestQDQModelAccuracy<uint8_t>(f32_fn, BuildConvQDQFn<uint8_t, int8_t>(s),
                                    opts, s.opset, ExpectedEPNodeAssignment::All);
    else if (s.weight_type == ConvWeightType::S4)
      TestQDQModelAccuracy<uint8_t>(f32_fn, BuildConvQDQFn<uint8_t, Int4x2>(s),
                                    opts, s.opset, ExpectedEPNodeAssignment::All);
  } else if (s.input_type == ConvInputType::U16) {
    if (s.weight_type == ConvWeightType::U8)
      TestQDQModelAccuracy<uint16_t>(f32_fn, BuildConvQDQFn<uint16_t, uint8_t>(s),
                                     opts, s.opset, ExpectedEPNodeAssignment::All);
    else if (s.weight_type == ConvWeightType::S8)
      TestQDQModelAccuracy<uint16_t>(f32_fn, BuildConvQDQFn<uint16_t, int8_t>(s),
                                     opts, s.opset, ExpectedEPNodeAssignment::All);
    else if (s.weight_type == ConvWeightType::S4)
      TestQDQModelAccuracy<uint16_t>(f32_fn, BuildConvQDQFn<uint16_t, Int4x2>(s),
                                     opts, s.opset, ExpectedEPNodeAssignment::All);
    else if (s.weight_type == ConvWeightType::S16)
      TestQDQModelAccuracy<uint16_t>(f32_fn, BuildConvQDQFn<uint16_t, int16_t>(s),
                                     opts, s.opset, ExpectedEPNodeAssignment::All);
    else if (s.weight_type == ConvWeightType::U16)
      TestQDQModelAccuracy<uint16_t>(f32_fn, BuildConvQDQFn<uint16_t, uint16_t>(s),
                                     opts, s.opset, ExpectedEPNodeAssignment::All);
  }
}

void RunConvFusionAccuracy(const ConvFusionSpec& s) {
  const auto f32_fn = BuildConvFusionF32ReferenceFn(s);
  const auto opts = BackendOptions(s.accuracy_backend);
  // ORT_ENABLE_BASIC prevents ConvActivationFusion from running on the CPU reference
  // session. Without it, contrib Q/DQ + Conv + Clip causes a temporary duplicate node
  // name that ORT's ONNX checker rejects. QNN EP's own fusion is unaffected (it runs
  // inside the QNN SDK, not via ORT graph transformers).
  const auto opt_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
  if (s.input_type == ConvFusionInputType::U8)
    TestQDQModelAccuracy<uint8_t>(f32_fn, BuildConvFusionQDQFn<uint8_t>(s),
                                  opts, s.opset, ExpectedEPNodeAssignment::All,
                                  QDQTolerance(), OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                  "", {}, opt_level);
  else
    TestQDQModelAccuracy<int8_t>(f32_fn, BuildConvFusionQDQFn<int8_t>(s),
                                 opts, s.opset, ExpectedEPNodeAssignment::All,
                                 QDQTolerance(), OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                 "", {}, opt_level);
}

}  // namespace

// ===========================================================================
// Phase A: CPU F32 (13 tests)
// ===========================================================================

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv2D_f32_DynamicBias) {
  RunConvAccuracy(kConv2D_f32_DynamicBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv2D_f32_StaticBias) {
  RunConvAccuracy(kConv2D_f32_StaticBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv2D_f32_AutoPadSameUpper) {
  RunConvAccuracy(kConv2D_f32_AutoPadSameUpper);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose2D_f32_AutoPadSameUpper) {
  RunConvAccuracy(kConvTranspose2D_f32_AutoPadSameUpper);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv2D_f32_AutoPadSameLower) {
  RunConvAccuracy(kConv2D_f32_AutoPadSameLower);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose2D_f32_AutoPadSameLower) {
  RunConvAccuracy(kConvTranspose2D_f32_AutoPadSameLower);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose3D_f32_AutoPadSameLower) {
  RunConvAccuracy(kConvTranspose3D_f32_AutoPadSameLower);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv2D_f32_LargePads) {
  RunConvAccuracy(kConv2D_f32_LargePads);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv2D_f32_LargeInput) {
  RunConvAccuracy(kConv2D_f32_LargeInput);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv1D_f32_StaticWeights) {
  RunConvAccuracy(kConv1D_f32_StaticWeights);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv1D_f32_DynamicWeights) {
  RunConvAccuracy(kConv1D_f32_DynamicWeights);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose1D_f32_StaticWeights) {
  RunConvAccuracy(kConvTranspose1D_f32_StaticWeights);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose1D_f32_DynamicWeights) {
  RunConvAccuracy(kConvTranspose1D_f32_DynamicWeights);
}

// ===========================================================================
// Phase B: HTP Cluster representatives (8 tests)
// ===========================================================================

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv1DU8U8S32_AutoPadUpper) {
  RunConvAccuracy(kConv1DU8U8S32_AutoPadUpper);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose1DU8U8S32_AutoPadLower) {
  RunConvAccuracy(kConvTranspose1DU8U8S32_AutoPadLower);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_AutoPadValid) {
  RunConvAccuracy(kConvU8U8S32_AutoPadValid);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTransposeU8U8S32_OutputShape) {
  RunConvAccuracy(kConvTransposeU8U8S32_OutputShape);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8S8S32_PerChannel) {
  RunConvAccuracy(kConvU8S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16S4S32_PerChannel_NegativeWeightQuantAxis) {
  RunConvAccuracy(kConvU16S4S32_PerChannel_NegativeWeightQuantAxis);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_LargeInput_Dilations_Pads) {
  RunConvAccuracy(kConvU8U8S32_LargeInput_Dilations_Pads);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16S4_PerChannel_NoBias) {
  RunConvAccuracy(kConvU16S4_PerChannel_NoBias);
}

// ===========================================================================
// Phase C: HTP Cluster members — accuracy-only (16 tests)
// ===========================================================================

// ---- Cluster: Conv1DU8U8S32_AutoPadUpper ----

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv1DU8U8S32_AutoPadLower) {
  RunConvAccuracy(kConv1DU8U8S32_AutoPadLower);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv1DU8U8S32_AutoPadValid) {
  RunConvAccuracy(kConv1DU8U8S32_AutoPadValid);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv1DU8U8S32_bias_initializer) {
  RunConvAccuracy(kConv1DU8U8S32_bias_initializer);
}

// ---- Cluster: ConvTranspose1DU8U8S32_AutoPadLower ----

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose1DU8U8S32_AutoPadUpper) {
  RunConvAccuracy(kConvTranspose1DU8U8S32_AutoPadUpper);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose1DU8U8S32_AutoPadValid) {
  RunConvAccuracy(kConvTranspose1DU8U8S32_AutoPadValid);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose1DU8U8S32_bias_initializer) {
  RunConvAccuracy(kConvTranspose1DU8U8S32_bias_initializer);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose1DU8U8S32_OutputShape) {
  RunConvAccuracy(kConvTranspose1DU8U8S32_OutputShape);
}

// ---- Cluster: ConvU8U8S32_AutoPadValid ----

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_AutoPadUpper) {
  RunConvAccuracy(kConvU8U8S32_AutoPadUpper);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_AutoPadLower) {
  RunConvAccuracy(kConvU8U8S32_AutoPadLower);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_bias_initializer) {
  RunConvAccuracy(kConvU8U8S32_bias_initializer);
}

// ---- Cluster: ConvTransposeU8U8S32_OutputShape ----

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTransposeU8U8S32_AutoPadLower) {
  RunConvAccuracy(kConvTransposeU8U8S32_AutoPadLower);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTransposeU8U8S32_AutoPadValid) {
  RunConvAccuracy(kConvTransposeU8U8S32_AutoPadValid);
}

// ---- Cluster: ConvU8S8S32_PerChannel ----

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8S8S32_PerChannel_BiasRequantization) {
  RunConvAccuracy(kConvU8S8S32_PerChannel_BiasRequantization);
}

// ---- Cluster: ConvU16S4S32_PerChannel_NegativeWeightQuantAxis ----

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16S4S32_PerChannel_AccuracyIssue) {
  RunConvAccuracy(kConvU16S4S32_PerChannel_AccuracyIssue);
}

// ---- Cluster: ConvU8U8S32_LargeInput_Dilations_Pads ----

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_large_input2_bias_initializer) {
  RunConvAccuracy(kConvU8U8S32_large_input2_bias_initializer);
}

// ---- Cluster: ConvU16S4_PerChannel_NoBias ----

TEST(QnnUnit_SessionAccuracy_ConvTest,
     DISABLED_ConvU16S4_PerChannel_NoBias_LargeINT4Weight) {
  // DISABLED: QNN EP (x86 HTP emulator) produces incorrect output (~6171)
  // instead of the expected ~-4.5 for this 28M-weight (9216x3072) S4 model.
  // Previously masked by old RunQnnModelTest approach which used input-scale as
  // output scale, clipping both CPU/QNN outputs to [0,1] before comparison.
  // TestQDQModelAccuracy uses the correct output scale and reveals the real issue.
  RunConvAccuracy(kConvU16S4_PerChannel_NoBias_LargeINT4Weight);
}

// ===========================================================================
// Phase D: HTP KEEP — non-fusion (24 tests)
// ===========================================================================

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_bias_dynamic_input) {
  RunConvAccuracy(kConvU8U8S32_bias_dynamic_input);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_BiasRequantization) {
  RunConvAccuracy(kConvU8U8S32_BiasRequantization);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16U8_PerTensor_NoBias) {
  RunConvAccuracy(kConvU16U8_PerTensor_NoBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16S4S32_PerChannel) {
  RunConvAccuracy(kConvU16S4S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv3D_U8S8S32_PerChannel) {
  RunConvAccuracy(kConv3D_U8S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvDepthwiseU8S8S32_PerChannel) {
  RunConvAccuracy(kConvDepthwiseU8S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv3D_U8S8S32_PerChannel2) {
  RunConvAccuracy(kConv3D_U8S8S32_PerChannel2);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTransposeU8S8S32_PerChannel) {
  RunConvAccuracy(kConvTransposeU8S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose3D_U8S8S32_PerChannel) {
  RunConvAccuracy(kConvTranspose3D_U8S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16S8S32_PerChannel) {
  RunConvAccuracy(kConvU16S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv3D_U16S8S32_PerChannel) {
  RunConvAccuracy(kConv3D_U16S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTransposeU16S8S32_PerChannel) {
  RunConvAccuracy(kConvTransposeU16S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose3D_U16S8S32_PerChannel) {
  RunConvAccuracy(kConvTranspose3D_U16S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvDepthwiseU16S8S32_PerChannel) {
  RunConvAccuracy(kConvDepthwiseU16S8S32_PerChannel);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv3D_U16S8S32_PerChannel2) {
  RunConvAccuracy(kConv3D_U16S8S32_PerChannel2);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16U8S32_StaticBias) {
  RunConvAccuracy(kConvU16U8S32_StaticBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16U8S32_DynamicBias) {
  RunConvAccuracy(kConvU16U8S32_DynamicBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU16U8S32_NoBias) {
  RunConvAccuracy(kConvU16U8S32_NoBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_DynamicWeight_NoBias) {
  RunConvAccuracy(kConvU8U8S32_DynamicWeight_NoBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTransposeU8U8S32_DynamicWeight_NoBias) {
  RunConvAccuracy(kConvTransposeU8U8S32_DynamicWeight_NoBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, Conv3D_U8U8S32_DynamicWeight_NoBias) {
  RunConvAccuracy(kConv3D_U8U8S32_DynamicWeight_NoBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvTranspose3D_U8U8S32_DynamicWeight_NoBias) {
  RunConvAccuracy(kConvTranspose3D_U8U8S32_DynamicWeight_NoBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, DepthwiseConvU16U8S32_StaticBias) {
  RunConvAccuracy(kDepthwiseConvU16U8S32_StaticBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, DepthwiseConvU16U8S32_DynamicBias) {
  RunConvAccuracy(kDepthwiseConvU16U8S32_DynamicBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, DepthwiseConvU16U8S32_NoBias) {
  RunConvAccuracy(kDepthwiseConvU16U8S32_NoBias);
}

// ===========================================================================
// Phase E: HTP Fusion tests (3 tests)
// ===========================================================================

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_ReluClipFusion) {
  RunConvFusionAccuracy(kConvU8U8S32_ReluClipFusion);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvU8U8S32_RedundantClipQDQ) {
  RunConvFusionAccuracy(kConvU8U8S32_RedundantClipQDQ);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, ConvS8S8S32_PerChannel_ReluClipFusion) {
  RunConvFusionAccuracy(kConvS8S8S32_PerChannel_ReluClipFusion);
}

// ===========================================================================
// Phase F: DISABLED — S16 variants not supported on Linux x86 HTP emulator (6)
// ===========================================================================

// DISABLED: Linux does not support U16+S16 Conv; Windows/Android only.
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_ConvU16S16S32_PerChannel) {
  RunConvAccuracy(kDISABLED_ConvU16S16S32_PerChannel);
}

// DISABLED: Linux does not support U16+U16 Conv; guarded #ifndef __linux__
// in integration tier.
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_ConvU16U16_PerTensor_NoBias) {
  RunConvAccuracy(kDISABLED_ConvU16U16_PerTensor_NoBias);
}

// DISABLED: x86 HTP emulator does not support U16+S16 Conv.
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_ConvU16S16S32_DynamicBias) {
  RunConvAccuracy(kDISABLED_ConvU16S16S32_DynamicBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_DepthwiseConvU16S16S32_DynamicBias) {
  RunConvAccuracy(kDISABLED_DepthwiseConvU16S16S32_DynamicBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_ConvU16S16S32_NoBias) {
  RunConvAccuracy(kDISABLED_ConvU16S16S32_NoBias);
}

TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_DepthwiseConvU16S16S32_NoBias) {
  RunConvAccuracy(kDISABLED_DepthwiseConvU16S16S32_NoBias);
}

// ===========================================================================
// Phase G: DISABLED — accuracy-only (no snapshot), tolerance regression (1)
// ===========================================================================

// DISABLED: QNN SDK 2.19.2 introduced a ~0.76% tolerance requirement for
// large padded inputs. Kept as a spec to track the regression.
TEST(QnnUnit_SessionAccuracy_ConvTest,
     DISABLED_ConvU8U8S32_large_input1_padding_bias_initializer) {
  RunConvAccuracy(kDISABLED_ConvU8U8S32_large_input1_padding_bias_initializer);
}

// ===========================================================================
// Benchmarks: CPU EP F32 inference overhead (DISABLED — run manually)
// Usage: ./onnxruntime_provider_test
//   --gtest_filter="QnnUnit_SessionAccuracy_ConvTest.DISABLED_Bench*"
//   --gtest_also_run_disabled_tests
// ===========================================================================

namespace {

void BenchCpuEpInference(const ConvSpec& s, int n_warmup = 3, int n_runs = 10) {
  ModelTestBuilder helper;
  BuildConvF32ReferenceFn(s)(helper);
  for (const auto& [domain, version] :
       std::unordered_map<std::string, int>{{"", s.opset}, {kMSDomain, 1}}) {
    auto* opset = helper.model_.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  std::vector<Ort::Value> cpu_outputs;

  // warmup
  for (int i = 0; i < n_warmup; ++i) {
    InferenceModelCPU(model_data, "bench_warmup", helper.feeds_, cpu_outputs, std::nullopt);
  }

  // measure session_create + run (combined, as AssertSessionSnapshotJsonQDQ would do)
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n_runs; ++i) {
    InferenceModelCPU(model_data, "bench_run", helper.feeds_, cpu_outputs, std::nullopt);
  }
  auto elapsed_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count() /
                    n_runs;

  GTEST_LOG_(INFO) << s.name << ": CPU F32 inference avg = " << elapsed_ms << " ms  (n=" << n_runs << ")";
}

// Measures QNN HTP: compile (session create) and Run() separately.
// Uses BuildConvModelFn (QDQ model) to match snapshot test's compile path.
void BenchQnnEpCompileAndRun(const ConvSpec& s, int n_warmup = 2, int n_runs = 5) {
  ModelTestBuilder helper;
  BuildConvModelFn(s)(helper);
  for (const auto& [domain, version] :
       std::unordered_map<std::string, int>{{"", s.opset}, {kMSDomain, 1}}) {
    auto* opset = helper.model_.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  auto provider_options = BackendOptions(s.accuracy_backend);

  // ── compile (session create) ──────────────────────────────────────────────
  // warmup compile
  for (int i = 0; i < n_warmup; ++i) {
    RegisteredEpDeviceUniquePtr ep_dev;
    Ort::SessionOptions so;
    RegisterQnnEpLibrary(ep_dev, so, "QNNExecutionProvider", provider_options);
    Ort::Session sess(*GetOrtEnv(), model_data.data(), model_data.size(), so);
    (void)sess;
  }
  // measure compile
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n_runs; ++i) {
    RegisteredEpDeviceUniquePtr ep_dev;
    Ort::SessionOptions so;
    RegisterQnnEpLibrary(ep_dev, so, "QNNExecutionProvider", provider_options);
    Ort::Session sess(*GetOrtEnv(), model_data.data(), model_data.size(), so);
    (void)sess;
  }
  double compile_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count() /
                      n_runs;

  // ── run (inference only, reuse one compiled session) ─────────────────────
  RegisteredEpDeviceUniquePtr ep_dev;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(ep_dev, so, "QNNExecutionProvider", provider_options);
  Ort::Session sess(*GetOrtEnv(), model_data.data(), model_data.size(), so);

  std::vector<std::string> in_names = sess.GetInputNames();
  std::vector<std::string> out_names = sess.GetOutputNames();
  std::vector<const char*> in_cstr, out_cstr;
  for (auto& n : in_names) in_cstr.push_back(n.c_str());
  for (auto& n : out_names) out_cstr.push_back(n.c_str());

  // build input tensors from feeds
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> ort_inputs;
  for (auto& name : in_names) {
    auto& feed = helper.feeds_.at(name);
    auto shape = feed.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    ort_inputs.emplace_back(Ort::Value::CreateTensor(
        mem_info, (void*)feed.GetTensorRawData(), feed.GetTensorSizeInBytes(),
        shape.data(), shape.size(),
        feed.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType()));
  }

  // warmup run
  for (int i = 0; i < n_warmup; ++i) {
    sess.Run(Ort::RunOptions{nullptr}, in_cstr.data(), ort_inputs.data(),
             ort_inputs.size(), out_cstr.data(), out_cstr.size());
  }
  // measure run
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n_runs; ++i) {
    sess.Run(Ort::RunOptions{nullptr}, in_cstr.data(), ort_inputs.data(),
             ort_inputs.size(), out_cstr.data(), out_cstr.size());
  }
  double run_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count() /
                  n_runs;

  GTEST_LOG_(INFO) << s.name << ": QNN EP compile = " << compile_ms
                   << " ms   run = " << run_ms << " ms  (n=" << n_runs << ")";
}

}  // namespace

TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_Bench_ConvU8S8S32_PerChannel) {
  BenchCpuEpInference(kConvU8S8S32_PerChannel);  // QNN compile = 99ms
}
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_Bench_ConvU16S8S32_PerChannel) {
  BenchCpuEpInference(kConvU16S8S32_PerChannel);  // QNN compile = 17ms
}
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_Bench_ConvU8U8S32_BiasRequantization) {
  BenchCpuEpInference(kConvU8U8S32_BiasRequantization);  // QNN compile = 10ms
}
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_Bench_ConvU8U8S32_LargeInput_Dilations_Pads) {
  BenchCpuEpInference(kConvU8U8S32_LargeInput_Dilations_Pads);  // QNN compile = 445ms
}

TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_BenchQnn_ConvU8S8S32_PerChannel) {
  BenchQnnEpCompileAndRun(kConvU8S8S32_PerChannel);
}
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_BenchQnn_ConvU16S8S32_PerChannel) {
  BenchQnnEpCompileAndRun(kConvU16S8S32_PerChannel);
}
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_BenchQnn_ConvU8U8S32_BiasRequantization) {
  BenchQnnEpCompileAndRun(kConvU8U8S32_BiasRequantization);
}
TEST(QnnUnit_SessionAccuracy_ConvTest, DISABLED_BenchQnn_ConvU8U8S32_LargeInput_Dilations_Pads) {
  BenchQnnEpCompileAndRun(kConvU8U8S32_LargeInput_Dilations_Pads);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_LIGHT_UT
