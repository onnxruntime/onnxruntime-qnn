// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "gtest/gtest.h"

#include "test/unittest_util/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

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

template <typename InputQType, typename ScaleQType>
static void RunRMSNormQDQTest(const TestInputDef<float>& input_def,
                              const TestInputDef<float>& scale_def,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              ExpectedEPNodeAssignment expected_ep_assignment,
                              bool use_contrib_qdq_ops = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

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
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(1e-5)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, false);
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
// re-encoded as UINT16 and still runs on QNN.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_StaticScale_AU16_WS16Shifted) {
  RunRMSNormQDQTest<uint16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                       TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                       {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                       ExpectedEPNodeAssignment::All,
                                       true);
}

// The same shifted INT16 gamma computed in-graph has no data to re-encode, so the RMSNorm node unit
// is rejected and falls back to CPU. The rejection must stay soft: the surrounding QDQ nodes are
// still claimed by QNN, the session initializes, and the results stay correct.
TEST_F(QnnHTPBackendTests, RMSNorm1D_LastAxis_DynamicScale_AU16_WS16Shifted) {
  RunRMSNormQDQTest<uint16_t, int16_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                       TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                       {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                       ExpectedEPNodeAssignment::Some,
                                       true);
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
// QNN's RmsNorm OpDef requires rank(gamma) == size(axes). The builder squeezes leading 1-dims to
// bridge the two; the tests below cover the shapes that squeeze is responsible for.
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
// Uses the CPU backend because the HTP path requires the last axis only.
TEST_F(QnnCPUBackendTests, RMSNorm_Rank3Scale_LeadingOne_MultipleAxes) {
  RunRMSNormCpuTest(TestInputDef<float>({2, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 24)),
                    TestInputDef<float>({1, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 12)),
                    {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                    ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
