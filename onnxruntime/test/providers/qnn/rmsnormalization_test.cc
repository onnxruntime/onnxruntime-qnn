// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>

#include "gtest/gtest.h"

#include "test/unittest_util/qdq_test_utils.h"
#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__linux__)

static void RunRMSNormCpuTest(const TestInputDef<float>& input_def,
                              const TestInputDef<float>& scale_def,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("rms_norm", "RMSNormalization", {input_def, scale_def}, {}, attrs),
                  provider_options,
                  23,
                  EPVerificationParams{expected_ep_assignment});
}

TEST_F(QnnCPUBackendTests, RMSNorm) {
  RunRMSNormCpuTest(TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                    TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, RMSNorm1D_Axis0) {
  RunRMSNormCpuTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                    TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, RMSNorm2D) {
  RunRMSNormCpuTest(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                    TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, RMSNorm3D) {
  RunRMSNormCpuTest(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                    TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNormBroadcastScale) {
  // HTP computes in fp16; this also covers the f16 ones-gamma path.
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  auto input_def = TestInputDef<float>({2, 4, 8}, false, GetFloatDataInRange(0.0f, 4.0f, 64));
  auto scale_def = TestInputDef<float>({2, 1, 8}, false, GetFloatDataInRange(0.9f, 1.1f, 16));
  auto attrs = {test::MakeAttribute("axis", static_cast<int64_t>(-1))};
  TestFp16ModelAccuracy(
      BuildOpTestCase<float>("rms_norm", "RMSNormalization", {input_def, scale_def}, {}, attrs),
      BuildOpTestCase<Ort::Float16_t>("rms_norm", "RMSNormalization",
                                      {ConvertToFP16InputDef(input_def), ConvertToFP16InputDef(scale_def)}, {}, attrs),
      provider_options,
      23,
      ExpectedEPNodeAssignment::All);
}

template <typename InputQType, typename ScaleQType>
GetTestQDQModelFn<InputQType> BuildQDQRMSNormTestCase(const TestInputDef<float>& input_def,
                                                      const TestInputDef<float>& scale_def,
                                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                      bool use_contrib_qdq_ops) {
  return [input_def, scale_def, attrs,
          use_contrib_qdq_ops](ModelTestBuilder& builder,
                               std::vector<QuantParams<InputQType>>& output_qparams) {
    // Input QDQ pair
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    std::string input_qdq = AddQDQNodePair<InputQType>(builder, "qdq_input", "input",
                                                       input_qparams.scale, input_qparams.zero_point,
                                                       use_contrib_qdq_ops);

    // Scale QDQ pair
    std::string scale_qdq;
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);

    if (scale_def.IsInitializer() && scale_def.IsRawData()) {
      std::vector<float> scale_scales = {scale_qparams.scale};
      std::vector<ScaleQType> scale_zps = {scale_qparams.zero_point};
      const std::vector<int64_t>& scale_shape = scale_def.GetShape();
      std::vector<ScaleQType> quantized_scales(SizeOfShape(scale_shape));
      QuantizeValues<float, ScaleQType>(scale_def.GetRawData(), quantized_scales, scale_shape,
                                        scale_scales, scale_zps, std::nullopt);

      builder.MakeInitializer<ScaleQType>("scale_initializer", scale_shape, quantized_scales);
      scale_qdq = "scale_dq_out";
      builder.AddDequantizeLinearNode<ScaleQType>("scale_dq", "scale_initializer",
                                                  scale_scales, scale_zps, scale_qdq,
                                                  {}, use_contrib_qdq_ops);
    } else {
      MakeTestInput<float>(builder, "scale", scale_def);
      scale_qdq = AddQDQNodePair<ScaleQType>(builder, "qdq_scale", "scale",
                                             scale_qparams.scale, scale_qparams.zero_point,
                                             use_contrib_qdq_ops);
    }

    // RMSNormalization node
    builder.AddNode("rms_norm", "RMSNormalization", {input_qdq, scale_qdq}, {"rms_norm_output"}, "", attrs);

    // Output QDQ pair
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "qdq_output", "rms_norm_output",
                                                      output_qparams[0].scale,
                                                      output_qparams[0].zero_point, use_contrib_qdq_ops);
  };
}

// `json_qnn_graph_dir`, when non-empty, dumps the compiled QNN graph there so the caller can assert
// exactly which ops landed on QNN (via AssertOpInQnnGraph) -- stronger than ExpectedEPNodeAssignment,
// which only counts assigned nodes and can't tell a specific op's CPU fallback from a fully-QNN graph.
// `verify_outputs` defaults to false to preserve existing callers; pass true to additionally check
// QNN's output against CPU EP's within ElementwiseAbsoluteVerifier(fp32_abs_err), e.g. to validate a
// gamma requantization path actually preserves numerical accuracy rather than just compiling/running.
// `fp32_abs_err` defaults to 1e-5 for lossless paths; the symmetric INT16 gamma re-encode is lossy
// by one quant step (see RMSNorm1D_LastAxis_StaticScale_AS16_WS16Shifted), so callers that exercise
// it pass a looser bound.
template <typename InputQType, typename ScaleQType>
static void RunRMSNormQDQTest(const TestInputDef<float>& input_def,
                              const TestInputDef<float>& scale_def,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              bool use_contrib_qdq_ops = false,
                              const std::string& json_qnn_graph_dir = "",
                              bool verify_outputs = false,
                              float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  if (!json_qnn_graph_dir.empty()) {
    provider_options["dump_json_qnn_graph"] = "1";
    provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir;
  }

  auto qdq_model_fn = BuildQDQRMSNormTestCase<InputQType, ScaleQType>(input_def, scale_def, attrs,
                                                                      use_contrib_qdq_ops);
  GetTestModelFn model_fn = [qdq_model_fn, input_def](ModelTestBuilder& builder) {
    std::pair<float, float> input_range = input_def.GetRange();
    QuantParams<InputQType> output_qparams = QuantParams<InputQType>::Compute(input_range.first, input_range.second);
    std::vector<QuantParams<InputQType>> output_qparams_vec = {output_qparams};

    qdq_model_fn(builder, output_qparams_vec);
  };

  RunQnnModelTest(model_fn,
                  provider_options,
                  23,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, verify_outputs);
}

TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, 0.0f, 10.0f),
                                      TestInputDef<float>({3}, true, 0.0f, 10.0f),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AU8_WU8) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AU16_WU8) {
  RunRMSNormQDQTest<uint16_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                       TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                       {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                       ExpectedEPNodeAssignment::All,
                                       true);
}

// A nonnegative INT16 gamma quantizes to a nonzero QNN offset, which HTP rejects. A static one is
// re-encoded as UINT16 and still runs on QNN. verify_outputs=true checks the re-encode preserves
// accuracy against CPU EP, not just that the model compiles and runs. The looser bound isn't about
// this re-encode (it's exact, just an offset shift) -- RMSNorm's own quantized computation carries
// ~1.5e-4 of noise against the CPU EP reference on some backends, independent of gamma handling.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AU16_WS16Shifted) {
  RunRMSNormQDQTest<uint16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                       TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                       {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                       ExpectedEPNodeAssignment::All,
                                       true,
                                       /*json_qnn_graph_dir=*/"",
                                       /*verify_outputs=*/true,
                                       /*fp32_abs_err=*/1e-3f);
}

// The same shifted INT16 gamma computed in-graph has no data to re-encode, so the RMSNorm node unit
// is rejected and falls back to CPU. The rejection must stay soft: the surrounding QDQ nodes are
// still claimed by QNN, the session initializes, and the results stay correct.
// ExpectedEPNodeAssignment::Some alone would also pass for a fully-QNN graph, so this dumps the
// compiled graph and asserts RmsNorm never lands on QNN. RMSNorm falls back to CPU along with every
// other node except the two leading Quantize ops (input, dynamic gamma), which end up as their own
// standalone QNN partition. Correct fallback placement alone doesn't prove correct results -- the
// mixed CPU/QNN execution could still produce wrong numbers -- so this also checks output accuracy
// against the CPU EP reference.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_DynamicScale_AU16_WS16Shifted) {
  // RunRMSNormQDQTest's GTEST_SKIP() for HTP arch <= V68 only returns out of that helper, not out of
  // this test body -- without this guard, the AssertOpInQnnGraph calls below still run on such
  // runners and fail because no QNN graph was ever dumped (the run itself was skipped).
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "RMSNorm1D_LastAxis_DynamicScale_AU16_WS16Shifted";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  RunRMSNormQDQTest<uint16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                       TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                       {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                       ExpectedEPNodeAssignment::Some,
                                       true,
                                       json_qnn_graph_dir.string(),
                                       /*verify_outputs=*/true,
                                       /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "RmsNorm", 0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", 0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Quantize", 2);  // input activation + dynamic gamma
}

// A nonnegative INT16 gamma quantizes to a nonzero QNN offset. When the activation is also INT16,
// HTP additionally rejects mixing a (would-be) UINT16 gamma with an SFIXED_POINT_16 activation, so a
// static gamma is instead re-encoded as symmetric SFIXED_POINT_16 (offset forced to 0, scale doubled)
// and still runs on QNN, at the cost of up to one quant step of gamma precision. The float range here
// starts at 0.0f, so the computed zero_point lands at the INT16 minimum (-32768), which is also where
// the doubled-scale re-encode saturates by exactly one quant level (see rmsnormalization_op_builder.cc).
// verify_outputs=true checks that the one-quant-step saturation this comment describes still stays
// within tolerance of CPU EP, rather than only checking the model compiles and runs on QNN. The
// symmetric re-encode is lossy by design, so this uses a looser bound than the default 1e-5.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AS16_WS16Shifted) {
  RunRMSNormQDQTest<int16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All,
                                      true,
                                      /*json_qnn_graph_dir=*/"",
                                      /*verify_outputs=*/true,
                                      /*fp32_abs_err=*/1e-3f);
}

// The same shifted INT16 gamma computed in-graph has no data to re-encode, so the RMSNorm node unit
// is rejected and falls back to CPU, same as the AU16 case above. The rejection must stay soft: the
// surrounding QDQ nodes are still claimed by QNN, the session initializes, and the results stay correct.
// Same graph-level assertion as the AU16 case: ExpectedEPNodeAssignment::Some alone would also pass
// for a fully-QNN graph. RMSNorm falls back to CPU along with every other node except the two
// leading Quantize ops (input, dynamic gamma), which end up as their own standalone QNN partition.
// Also checks output accuracy against the CPU EP reference, same reasoning as the AU16 case above.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_DynamicScale_AS16_WS16Shifted) {
  // Same reason as the AU16 case above: the post-run AssertOpInQnnGraph calls need their own skip,
  // since RunRMSNormQDQTest's internal GTEST_SKIP() for HTP arch <= V68 doesn't propagate out of
  // this test body on its own.
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "RMSNorm1D_LastAxis_DynamicScale_AS16_WS16Shifted";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  RunRMSNormQDQTest<int16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::Some,
                                      true,
                                      json_qnn_graph_dir.string(),
                                      /*verify_outputs=*/true,
                                      /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "RmsNorm", 0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", 0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Quantize", 2);  // input activation + dynamic gamma
}

// Test names below are aligned to the QNN offset (the value the builder/HTP actually operate on),
// which is the negative of the ONNX zero point used elsewhere in this file -- e.g. a gamma range
// that produces ONNX zero_point +414 produces QNN offset -414, i.e. a *negative* QNN offset.

// Gamma range [-4, 3.9] produces ONNX zero_point +414, i.e. QNN offset -414: a moderate
// (non-saturating) negative QNN offset, clearly nonzero but well short of the INT16 minimum, so the
// re-encode shifts without hitting the saturation edge case covered by
// RMSNorm1D_LastAxis_StaticScale_AS16_WS16Shifted above.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AS16_NegativeOffset) {
  RunRMSNormQDQTest<int16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, true, GetFloatDataInRange(-4.0f, 3.9f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All,
                                      true,
                                      /*json_qnn_graph_dir=*/"",
                                      /*verify_outputs=*/true,
                                      /*fp32_abs_err=*/1e-3f);
}

// Gamma range [-2, 8] produces ONNX zero_point -19661, i.e. QNN offset +19661: a large-magnitude
// positive QNN offset, distinct from the near-saturating case above.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AS16_PositiveOffset) {
  RunRMSNormQDQTest<int16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, true, GetFloatDataInRange(-2.0f, 8.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All,
                                      true,
                                      /*json_qnn_graph_dir=*/"",
                                      /*verify_outputs=*/true,
                                      /*fp32_abs_err=*/1e-3f);
}

// Gamma range [-3.9, 4.0] produces ONNX zero_point -415, i.e. a genuinely small positive QNN offset
// (+415) -- the case the two tests above don't cover (one is negative, one is large-magnitude).
// Asserts the re-encoded tensor's actual scale/offset rather than only checking output tolerance,
// since a wrong-but-close re-encode could still pass ElementwiseAbsoluteVerifier by coincidence.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AS16_SmallPositiveOffset) {
  // RunRMSNormQDQTest's GTEST_SKIP() for HTP arch <= V68 only returns out of that helper, not out of
  // this test body -- without this guard, the AssertTensorQuantParamsInQnnGraph call below still runs
  // on such runners and fails because no QNN graph was ever dumped (the run itself was skipped).
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "RMSNorm1D_LastAxis_StaticScale_AS16_SmallPositiveOffset";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  TestInputDef<float> scale_def({3}, true, GetFloatDataInRange(-3.9f, 4.0f, 3));
  RunRMSNormQDQTest<int16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      scale_def,
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All,
                                      true,
                                      json_qnn_graph_dir.string(),
                                      /*verify_outputs=*/true,
                                      /*fp32_abs_err=*/1e-3f);

  // Symmetric re-encode: offset forced to 0, scale doubled from the original asymmetric encoding.
  const QuantParams<int16_t> original = GetTestInputQuantParams<int16_t>(scale_def);
  AssertTensorQuantParamsInQnnGraph(json_qnn_graph_dir, "scale_initializer_s16sym",
                                    original.scale * 2.0f, /*expected_offset=*/0);
}

// Gamma values genuinely straddle zero (as opposed to the nonnegative-only gamma used by every case
// above), exercising the re-encode against an asymmetric mixed-sign distribution rather than a
// one-sided range that happens to produce a nonzero zero_point.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AS16_MixedSignGamma) {
  RunRMSNormQDQTest<int16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, true, GetFloatDataInRange(-3.0f, 5.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All,
                                      true,
                                      /*json_qnn_graph_dir=*/"",
                                      /*verify_outputs=*/true,
                                      /*fp32_abs_err=*/1e-3f);
}

// Regression guard: an (almost) symmetric gamma range solves zero_point = qmin - rmin/scale = 0, so
// the builder's offset-is-already-zero check should take the pass-through path and leave gamma
// untouched -- the re-encode above must not fire (or touch anything) when there's nothing to fix.
// The looser output bound is unrelated to the pass-through itself; see AU16_WS16Shifted above for
// why. Output tolerance alone can't tell "gamma untouched" from "gamma re-encoded to something that
// happens to land within tolerance", so this also asserts the tensor QNN actually receives keeps
// its original scale/offset -- i.e. the re-encode truly did not fire.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AS16_OffsetZero) {
  // Same reason as SmallPositiveOffset above: the post-run AssertTensorQuantParamsInQnnGraph call
  // needs its own skip, since RunRMSNormQDQTest's internal GTEST_SKIP() for HTP arch <= V68 doesn't
  // propagate out of this test body on its own.
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "RMSNorm1D_LastAxis_StaticScale_AS16_OffsetZero";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  TestInputDef<float> scale_def({3}, true, GetFloatDataInRange(-32768.0f / 32767.0f * 4.0f, 4.0f, 3));
  RunRMSNormQDQTest<int16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      scale_def,
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All,
                                      true,
                                      json_qnn_graph_dir.string(),
                                      /*verify_outputs=*/true,
                                      /*fp32_abs_err=*/1e-3f);

  const QuantParams<int16_t> original = GetTestInputQuantParams<int16_t>(scale_def);
  AssertTensorQuantParamsInQnnGraph(json_qnn_graph_dir, "scale_initializer",
                                    original.scale, -original.zero_point);
}

// Gamma shape has leading 1-dims ([1, 1, 3]) that the builder must squeeze down to match X's
// normalized dimensions, on top of the INT16 sign re-encode -- exercises the two features composing,
// not just each one individually.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AS16_WS16Shifted_LeadingOnesShape) {
  RunRMSNormQDQTest<int16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({1, 1, 3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All,
                                      true,
                                      /*json_qnn_graph_dir=*/"",
                                      /*verify_outputs=*/true,
                                      /*fp32_abs_err=*/1e-3f);
}

TEST_F(QnnHTPBackendTests, RMSNormU8U8_4D_LastAxis) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18)),
                                      TestInputDef<float>({3}, true, GetFloatDataInRange(-2.0f, 2.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_DynamicScale) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

// ONNX RMSNormalization allows `scale` to be any shape unidirectionally broadcastable to X, but
// QNN RmsNorm requires gamma to match the normalized dimensions. The builder squeezes leading 1-dims
// to bridge the two; the tests below cover the shapes that squeeze is responsible for.
//
// The rank-3 scale shape comes from Pi05ActionExpert (tetracode issue #20549), which normalizes
// X [1, 50, 1024] with a scale of shape [1, 1, 1024]. Dims are scaled down here for speed; the
// rank relationship is what is under test.
static void RunRMSNormFp32Test(const TestInputDef<float>& input_def,
                               const TestInputDef<float>& scale_def,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               float fp32_abs_err = 0.01f) {
#if defined(_WIN32)
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_htp_fp16_precision"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(BuildOpTestCase<float>("rms_norm", "RMSNormalization", {input_def, scale_def}, {}, attrs),
                  provider_options,
                  23,  // opset
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}

// Static scale: squeezed by re-declaring the dims, no Reshape node.
TEST_F(QnnHTPBackendTests, RMSNorm_Rank3Scale_LeadingOnes_StaticScale) {
  RunRMSNormFp32Test(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),
                     TestInputDef<float>({1, 1, 3}, true, GetFloatDataInRange(0.5f, 1.5f, 3)),
                     {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                     ExpectedEPNodeAssignment::All);
}

// Computed (non-initializer) scale, as in Pi05ActionExpert: only an in-graph Reshape can fix the rank.
TEST_F(QnnHTPBackendTests, RMSNorm_Rank3Scale_LeadingOnes_DynamicScale) {
  RunRMSNormFp32Test(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),
                     TestInputDef<float>({1, 1, 3}, false, GetFloatDataInRange(0.5f, 1.5f, 3)),
                     {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                     ExpectedEPNodeAssignment::All);
}

// Squeeze is driven by size(axes), not by a fixed source rank.
TEST_F(QnnHTPBackendTests, RMSNorm_Rank2Scale_LeadingOnes) {
  RunRMSNormFp32Test(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),
                     TestInputDef<float>({1, 3}, true, GetFloatDataInRange(0.5f, 1.5f, 3)),
                     {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                     ExpectedEPNodeAssignment::All);
}

// Multiple leading 1-dims must all be dropped.
TEST_F(QnnHTPBackendTests, RMSNorm_Rank4Scale_LeadingOnes) {
  RunRMSNormFp32Test(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 18)),
                     TestInputDef<float>({1, 1, 1, 3}, true, GetFloatDataInRange(0.5f, 1.5f, 3)),
                     {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                     ExpectedEPNodeAssignment::All);
}

// FP32 model at FP16 precision: covers the FLOAT_32 ones-gamma that the FP16-model test does not.
TEST_F(QnnHTPBackendTests, RMSNormBroadcastScale_FP32_StaticScale) {
  for (const auto& scale_shape : {std::vector<int64_t>{2, 1, 8}, std::vector<int64_t>{1}}) {
    RunRMSNormFp32Test(TestInputDef<float>({2, 4, 8}, false, GetFloatDataInRange(0.0f, 4.0f, 64)),
                       TestInputDef<float>(scale_shape, true, GetFloatDataInRange(0.9f, 1.1f, SizeOfShape(scale_shape))),
                       {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                       ExpectedEPNodeAssignment::All);
  }
}

TEST_F(QnnHTPBackendTests, RMSNormBroadcastScale_FP32_DynamicScale) {
  for (const auto& scale_shape : {std::vector<int64_t>{2, 1, 8}, std::vector<int64_t>{1}}) {
    RunRMSNormFp32Test(TestInputDef<float>({2, 4, 8}, false, GetFloatDataInRange(0.0f, 4.0f, 64)),
                       TestInputDef<float>(scale_shape, false, GetFloatDataInRange(0.9f, 1.1f, SizeOfShape(scale_shape))),
                       {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                       ExpectedEPNodeAssignment::All);
  }
}

TEST_F(QnnHTPBackendTests, RMSNorm_MultipleAxes) {
  for (int64_t axis : {1, -2}) {
    RunRMSNormFp32Test(TestInputDef<float>({2, 3, 4}, false, GetFloatDataInRange(-2.0f, 2.0f, 24)),
                       TestInputDef<float>({2, 1, 4}, false, GetFloatDataInRange(0.5f, 1.5f, 8)),
                       {test::MakeAttribute("axis", axis), test::MakeAttribute("epsilon", 0.1f)},
                       ExpectedEPNodeAssignment::All);
  }
}

TEST_F(QnnHTPBackendTests, RMSNorm_Rank5) {
  RunRMSNormFp32Test(TestInputDef<float>({1, 2, 3, 4, 8}, false, GetFloatDataInRange(-2.0f, 2.0f, 192)),
                     TestInputDef<float>({8}, true, GetFloatDataInRange(0.5f, 1.5f, 8)),
                     {},
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNorm_Decomposition_QuantizedNotSupported) {
  for (int64_t axis : {1, -1}) {
    RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({2, 3, 4}, false, 0.0f, 4.0f),
                                        TestInputDef<float>({axis == 1 ? 4 : 1}, true, 0.5f, 1.5f),
                                        {test::MakeAttribute("axis", axis)},
                                        ExpectedEPNodeAssignment::None);
  }
}

// Quantized squeeze: exercises the scale's quant params being carried onto the squeezed tensor and
// the dummy beta (SDK < 2.49) inheriting the post-squeeze rank.
TEST_F(QnnHTPBackendTests, RMSNorm_Rank3Scale_LeadingOnes_QDQ_StaticScale) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({1, 1, 3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, RMSNorm_Rank3Scale_LeadingOnes_QDQ_DynamicScale) {
  RunRMSNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                      TestInputDef<float>({1, 1, 3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                      ExpectedEPNodeAssignment::All);
}

// axis=1 over X {2, 3, 4} normalizes 2 axes, so the scale squeezes to rank 2 rather than rank 1.
// CPU uses the native multi-axis RMSNorm kernel.
TEST_F(QnnCPUBackendTests, RMSNorm_Rank3Scale_LeadingOne_MultipleAxes) {
  RunRMSNormCpuTest(TestInputDef<float>({2, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 24)),
                    TestInputDef<float>({1, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 12)),
                    {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                    ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
