// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Runs an LayerNorm model on the QNN CPU backend. Checks the graph node assignment and that inference
// outputs for QNN and CPU match.
static void RunLayerNormCpuTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& scale_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("layer_norm_node", "LayerNormalization", {input_def, scale_def}, {}, attrs),
                  provider_options,
                  17,
                  expected_ep_assignment);
}

// Disabled all QNN CPU LayerNorm tests due to bug in 2.42 SDK

TEST_F(QnnCPUBackendTests, DISABLED_LayerNorm) {
  RunLayerNormCpuTest(TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, DISABLED_LayerNorm1D_Axis0) {
  RunLayerNormCpuTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, DISABLED_LayerNorm1D_AxisLast) {
  RunLayerNormCpuTest(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                      TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 10.0f, 3)),
                      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, DISABLED_LayerNorm2D) {
  RunLayerNormCpuTest(TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                      TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
                      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, DISABLED_LayerNorm3D) {
  RunLayerNormCpuTest(TestInputDef<float>({1, 2, 3, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 72)),
                      TestInputDef<float>({1, 2, 3, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 72)),
                      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
                      ExpectedEPNodeAssignment::All);
}

template <typename InputQType, typename ScaleQType>
GetTestQDQModelFn<InputQType> BuildQDQLayerNormTestCase(const TestInputDef<float>& input_def,
                                                        const TestInputDef<float>& scale_def,
                                                        const TestInputDef<float>& bias_def,
                                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                        bool use_contrib_qdq_ops) {
  return [input_def, scale_def, bias_def, attrs,
          use_contrib_qdq_ops](ModelTestBuilder& builder,
                               std::vector<QuantParams<InputQType>>& output_qparams) {
    std::vector<std::string> layer_norm_inputs;

    // X -> Q -> DQ ->
    MakeTestInput(builder, "X", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    std::string x_qdq_name = AddQDQNodePair<InputQType>(builder, "qdq0", "X", input_qparams.scale, input_qparams.zero_point,
                                                        use_contrib_qdq_ops);
    layer_norm_inputs.push_back(x_qdq_name);

    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);

    if (scale_def.IsInitializer() && scale_def.IsRawData()) {
      // Quantized(scale weights) -> DQ ->
      std::vector<float> scale_scales = {scale_qparams.scale};
      std::vector<ScaleQType> scale_zps = {scale_qparams.zero_point};
      std::vector<int64_t> scale_shape = scale_def.GetShape();
      std::vector<ScaleQType> quantized_scales(SizeOfShape(scale_shape));
      QuantizeValues<float, ScaleQType>(scale_def.GetRawData(), quantized_scales, scale_shape,
                                        scale_scales, scale_zps, std::nullopt);

      builder.MakeInitializer<ScaleQType>("scale", scale_shape, quantized_scales);
      const std::string scale_qdq = "scale_dq_out";
      builder.AddDequantizeLinearNode<ScaleQType>("scale_dq", "scale", scale_qparams.scale, scale_qparams.zero_point,
                                                  scale_qdq, use_contrib_qdq_ops);
      layer_norm_inputs.push_back(scale_qdq);
    } else {
      // scale input -> Q -> DQ ->
      MakeTestInput(builder, "scale", scale_def);
      auto scale_qdq = AddQDQNodePair<ScaleQType>(builder, "scale_qdq", "scale", scale_qparams.scale, scale_qparams.zero_point,
                                                  use_contrib_qdq_ops);
      layer_norm_inputs.push_back(scale_qdq);
    }

    if (!bias_def.GetShape().empty()) {
      const float bias_scale = input_qparams.scale * scale_qparams.scale;
      layer_norm_inputs.push_back(MakeTestQDQBiasInput(builder, "bias", bias_def, bias_scale, use_contrib_qdq_ops));
    }

    // LayerNormalization
    builder.AddNode(
        "ln_node",
        "LayerNormalization",
        layer_norm_inputs,
        {"Y"},
        "",
        attrs);

    // layer_norm_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "final_qdq", "Y", output_qparams[0].scale,
                                                      output_qparams[0].zero_point, use_contrib_qdq_ops);
  };
}

// Runs a QDQ LayerNorm model on the QNN HTP backend. Checks the graph node assignment and that inference
// outputs for QNN are as accurate as CPU EP (compares against f32 model and QDQ model).
template <typename InputQType, typename ScaleQType>
static void RunLayerNormQDQTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& scale_def,
                                const TestInputDef<float>& bias_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                bool use_contrib_qdq_ops = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>("layer_norm_node", "LayerNormalization", {input_def, scale_def}, {}, attrs),
                       BuildQDQLayerNormTestCase<InputQType, ScaleQType>(input_def, scale_def, bias_def, attrs,
                                                                         use_contrib_qdq_ops),
                       provider_options,
                       17,  // opset
                       expected_ep_assignment);
}

// Test that QNN HTP only supports axis = -1 (i.e., last dimension).
TEST_F(QnnHTPBackendTests, LayerNorm1D_Axis0_Unsupported) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, 0.0f, 10.0f),
                                        TestInputDef<float>({1, 2, 3}, true, 0.0f, 10.0f),
                                        TestInputDef<float>(),
                                        {test::MakeAttribute("axis", static_cast<int64_t>(0))},  // Unsupported axis
                                        ExpectedEPNodeAssignment::None);
}

// Test accuracy of 8-bit QDQ LayerNorm with a static scale input.
TEST_F(QnnHTPBackendTests, LayerNorm1D_LastAxis_StaticScale_AU8_WU8) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                        TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                        TestInputDef<float>(),  // Implicit bias input
                                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                        ExpectedEPNodeAssignment::All);
}

// Test accuracy of 8-bit QDQ LayerNorm with a static scale input and an explicit bias input (static).
TEST_F(QnnHTPBackendTests, LayerNorm1D_LastAxis_StaticScale_StaticBias_AU8_WU8_BU8) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                        TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                        TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LayerNorm1D_QNN2_24_ImplicitBias_ValidationBug) {
  // QNN 2.24 to 2.27: LayerNorm fails validation (intermittent) if the bias input is not provided. QNN EP will provide
  // an explicit bias of all zeros to get around this bug.
  // QNN 2.28.0: Validation bug is fixed, but get accuracy errors.
  // QNN 2.28.2: All fixed.
  for (size_t i = 0; i < 15; i++) {  // Run it multiple times since this is an intermittent bug.
    RunLayerNormQDQTest<uint16_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 1.0f, 6)),
                                           TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
                                           TestInputDef<float>(),  // Implicit bias input
                                           {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                           ExpectedEPNodeAssignment::All,
                                           true);
  }
}

TEST_F(QnnHTPBackendTests, LayerNorm1D_LastAxis_StaticScale_AU16_WU8) {
  // QNN 2.28.0: Get accuracy errors.
  // QNN 2.28.2: All fixed.
  RunLayerNormQDQTest<uint16_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                         TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),  // Static
                                         TestInputDef<float>(),
                                         {test::MakeAttribute("axis", static_cast<int64_t>(-1))},  // Last axis
                                         ExpectedEPNodeAssignment::All,
                                         true);  // Use 'com.microsoft' Q/DQ ops
}

// Test accuracy of 8-bit QDQ LayerNorm with a dynamic scale input.
//
// TODO(adrianlizarraga): Fails to finalize with QNN SDK 2.22. Still fails on QNN SDK 2.36.1.
// Verbose logs:
// Starting stage: Graph Transformations and Optimizations
// C:\...\QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::flat_to_vtcm
// C:\...\QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x102800000013 preparation failed with err:-1
// Completed stage: Graph Transformations and Optimizations (6247 us)
// QnnDsp <E> "node_token_15" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// Failed to finalize QNN graph.
TEST_F(QnnHTPBackendTests, DISABLED_LayerNorm1D_LastAxis_DynamicScale) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                                        TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),  // Dynamic
                                        TestInputDef<float>(),
                                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},  // Last axis
                                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LayerNorm_Decomposed_ScaleAndBiasMisaligned) {
  // scale + bias both misaligned -> LN, Mul (intermediate), Add (final)
  RunLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      // Full-rank scale with non-1 dim before the normalized axis -> externalize_scale.
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(0.1f, 1.0f, 6)),
      // Full-rank bias with non-1 dim before the normalized axis -> externalize_bias.
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(0.0f, 1.0f, 6)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// final_tensor_type / final_output_info.{qnn_data_type, quant_param, shape}).
TEST_F(QnnHTPBackendTests, LayerNorm_Decomposed_ScaleMisaligned_NoBias) {
  //scale misaligned, no bias -> LN, Mul (final)
  RunLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(0.1f, 1.0f, 6)),
      TestInputDef<float>(),  // No bias.
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LayerNorm_Decomposed_BiasMisaligned_ScaleAligned) {
  // scale aligned, bias misaligned -> LN(scale), Add (final)
  RunLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      // 1D scale aligned with X.shape[axis:]=[3], does not need externalization.
      TestInputDef<float>({3}, true, GetFloatDataInRange(0.1f, 1.0f, 3)),
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(0.0f, 1.0f, 6)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// scale misaligned, bias shape is legal on its own -> scale-out forces bias-out, so still
// LN, Mul (intermediate), Add (final). Verifies the policy that bias gets pulled out alongside
// scale even when its own shape would have been consumable by LN.
TEST_F(QnnHTPBackendTests, LayerNorm_Decomposed_ScaleMisaligned_BiasAligned) {
  RunLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      // Full-rank scale with non-1 dim before normalized axis -> externalize_scale.
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(0.1f, 1.0f, 6)),
      // 1D bias aligned with X.shape[axis:]=[3]; legal inside LN, but bias-out is forced by scale-out.
      TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// ----- Non-QDQ (FP32 lowered to FP16) tests on the HTP backend -----------------------
// QNN HTP requires fp16 to run float models; enable_htp_fp16_precision lowers fp32
// inputs/weights to fp16 internally. These tests exercise the LayerNorm op-builder paths
// without QDQ Q/DQ pairs, so intermediate quant params don't enter the picture and the
// decomposition path is validated purely on shape/dtype handling.

static void RunLayerNormHtpFp16Test(const TestInputDef<float>& input_def,
                                    const TestInputDef<float>& scale_def,
                                    const TestInputDef<float>& bias_def,
                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    float fp32_abs_err = 0.01f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_htp_fp16_precision"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  GetTestModelFn model_fn =
      bias_def.GetShape().empty()
          ? BuildOpTestCase<float>("layer_norm_node", "LayerNormalization",
                                    {input_def, scale_def}, {}, attrs)
          : BuildOpTestCase<float, int64_t>("layer_norm_node", "LayerNormalization",
                                             {input_def, scale_def}, {}, {bias_def}, attrs);

  RunQnnModelTest(model_fn,
                  provider_options,
                  17,  // opset
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Standard LN: 1D scale/bias on the last axis. Single LN node; no decomposition.
TEST_F(QnnHTPBackendTests, LayerNorm_FP32_LastAxis_StandardLN) {
  RunLayerNormHtpFp16Test(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(0.5f, 1.5f, 3)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(-0.1f, 0.1f, 3)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// Standard LN with no bias.
TEST_F(QnnHTPBackendTests, LayerNorm_FP32_LastAxis_NoBias) {
  RunLayerNormHtpFp16Test(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(0.5f, 1.5f, 3)),
      TestInputDef<float>(),  // No bias.
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// Decomposition path: scale + bias both misaligned. Lowers to LN, Mul (intermediate), Add (final).
TEST_F(QnnHTPBackendTests, LayerNorm_FP32_Decomposed_ScaleAndBiasMisaligned) {
  RunLayerNormHtpFp16Test(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(0.5f, 1.5f, 6)),
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(-0.1f, 0.1f, 6)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// Decomposition path: scale misaligned, no bias. Lowers to LN, Mul (final).
TEST_F(QnnHTPBackendTests, LayerNorm_FP32_Decomposed_ScaleMisaligned_NoBias) {
  RunLayerNormHtpFp16Test(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(0.5f, 1.5f, 6)),
      TestInputDef<float>(),  // No bias.
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// Decomposition path: bias misaligned, scale aligned. Lowers to LN(scale), Add (final).
TEST_F(QnnHTPBackendTests, LayerNorm_FP32_Decomposed_BiasMisaligned_ScaleAligned) {
  RunLayerNormHtpFp16Test(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 6)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(0.5f, 1.5f, 3)),
      TestInputDef<float>({1, 2, 3}, true, GetFloatDataInRange(-0.1f, 0.1f, 6)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
