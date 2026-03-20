// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/node_attr_utils.h"

#include "gtest/gtest.h"

#include "test/unittest_util/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Runs a float32 SimplifiedLayerNormalization model on the QNN CPU backend.
static void RunSimplifiedLayerNormCpuTest(const TestInputDef<float>& input_def,
                                          const TestInputDef<float>& scale_def,
                                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                          ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildOpTestCase<float>("simplified_layernorm", "SimplifiedLayerNormalization", {input_def, scale_def}, {}, attrs),
      provider_options,
      17,  // opset version
      expected_ep_assignment);
}

// Builds a QDQ SimplifiedLayerNormalization test case (single Y output).
template <typename InputQType, typename ScaleQType>
GetTestQDQModelFn<InputQType> BuildQDQSimplifiedLayerNormTestCase(
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& scale_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    bool use_contrib_qdq_ops) {
  return [input_def, scale_def, attrs,
          use_contrib_qdq_ops](ModelTestBuilder& builder,
                               std::vector<QuantParams<InputQType>>& output_qparams) {
    // input -> Q -> DQ
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    std::string input_qdq = AddQDQNodePair<InputQType>(builder, "qdq_input", "input",
                                                       input_qparams.scale,
                                                       input_qparams.zero_point, use_contrib_qdq_ops);

    // scale: static initializer -> DQ, or dynamic -> Q -> DQ
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
      scale_qdq = "scale_qdq_output";
      builder.AddDequantizeLinearNode<ScaleQType>("scale_dq", "scale_initializer", scale_scales, scale_zps,
                                                  scale_qdq, {}, use_contrib_qdq_ops);
    } else {
      MakeTestInput<float>(builder, "scale", scale_def);
      scale_qdq = AddQDQNodePair<ScaleQType>(builder, "qdq_scale", "scale", scale_qparams.scale,
                                             scale_qparams.zero_point, use_contrib_qdq_ops);
    }

    // SimplifiedLayerNormalization (single output Y)
    builder.AddNode("sln_node", "SimplifiedLayerNormalization", {input_qdq, scale_qdq},
                    {"sln_output"}, "", attrs);

    // output -> Q -> DQ -> graph output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "qdq_output", "sln_output",
                                                      output_qparams[0].scale,
                                                      output_qparams[0].zero_point, use_contrib_qdq_ops);
  };
}

// Runs a QDQ SimplifiedLayerNormalization model on the QNN HTP backend.
template <typename InputQType, typename ScaleQType>
static void RunSimplifiedLayerNormQDQTest(const TestInputDef<float>& input_def,
                                          const TestInputDef<float>& scale_def,
                                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                          ExpectedEPNodeAssignment expected_ep_assignment,
                                          bool use_contrib_qdq_ops = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto qdq_model_fn = BuildQDQSimplifiedLayerNormTestCase<InputQType, ScaleQType>(
      input_def, scale_def, attrs, use_contrib_qdq_ops);

  GetTestModelFn model_fn = [qdq_model_fn, input_def](ModelTestBuilder& builder) {
    std::pair<float, float> input_range = input_def.GetRange();
    QuantParams<InputQType> output_qparams = QuantParams<InputQType>::Compute(
        input_range.first, input_range.second);
    std::vector<QuantParams<InputQType>> output_qparams_vec = {output_qparams};
    qdq_model_fn(builder, output_qparams_vec);
  };

  RunQnnModelTest(model_fn, provider_options, 17, expected_ep_assignment,
                  1e-5f, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, false);
}

// Basic 2D input, axis=0.
TEST_F(QnnCPUBackendTests, SimplifiedLayerNorm_2D_Axis0) {
  RunSimplifiedLayerNormCpuTest(
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      TestInputDef<float>({2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
      ExpectedEPNodeAssignment::All);
}

// 3D input, axis=0.
TEST_F(QnnCPUBackendTests, SimplifiedLayerNorm_3D_Axis0) {
  RunSimplifiedLayerNormCpuTest(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
      ExpectedEPNodeAssignment::All);
}

// 4D input, axis=0.
TEST_F(QnnCPUBackendTests, SimplifiedLayerNorm_4D_Axis0) {
  RunSimplifiedLayerNormCpuTest(
      TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
      TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 18)),
      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
      ExpectedEPNodeAssignment::All);
}

// Custom epsilon value on CPU backend.
// QNN CPU backend requires scale to have the same shape as the input, so axis=0 is used.
TEST_F(QnnCPUBackendTests, SimplifiedLayerNorm_CustomEpsilon) {
  RunSimplifiedLayerNormCpuTest(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      {test::MakeAttribute("axis", static_cast<int64_t>(0)),
       test::MakeAttribute("epsilon", 1e-3f)},
      ExpectedEPNodeAssignment::All);
}

// QNN EP must reject SimplifiedLayerNormalization when the optional inv_std_var output is requested.
// The QNN RMSNorm op only supports a single output (Y).
TEST_F(QnnCPUBackendTests, SimplifiedLayerNorm_InvStdVar_Unsupported) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  // Build a model with 2 outputs: Y and inv_std_var.
  auto build_model_with_inv_std_var = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 2, 3}, 0.0f, 10.0f);
    builder.MakeInitializer<float>("scale", {3}, 0.0f, 1.0f);
    builder.MakeOutput("output_y");
    builder.MakeOutput("output_inv_std_var");
    builder.AddNode("sln_node", "SimplifiedLayerNormalization",
                    {"input", "scale"},
                    {"output_y", "output_inv_std_var"},
                    "",
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(-1))});
  };

  // QNN EP should reject this node because inv_std_var is not supported.
  // The node falls back to CPU EP.
  RunQnnModelTest(build_model_with_inv_std_var, provider_options, 17,
                  ExpectedEPNodeAssignment::None,
                  1e-5f, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  false /* verify_outputs */);
}

// QNN HTP only supports axis = last dimension. axis=0 must be rejected.
// Uses a non-QDQ float model so that Q/DQ nodes don't inflate the assigned-node count.
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_Axis0_Unsupported) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildOpTestCase<float>("simplified_layernorm",
                             "SimplifiedLayerNormalization",
                             {TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
                              TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6))},
                             {},
                             {test::MakeAttribute("axis", static_cast<int64_t>(0))}),
      provider_options,
      17,
      ExpectedEPNodeAssignment::None);
}

// Input rank > 4 must be rejected.
// Uses a non-QDQ float model so that Q/DQ nodes don't inflate the assigned-node count.
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_Rank5_Unsupported) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildOpTestCase<float>("simplified_layernorm",
                             "SimplifiedLayerNormalization",
                             {TestInputDef<float>({1, 2, 3, 3, 4}, false, GetFloatDataInRange(0.0f, 10.0f, 72)),
                              TestInputDef<float>({4}, false, GetFloatDataInRange(0.0f, 1.0f, 4))},
                             {},
                             {test::MakeAttribute("axis", static_cast<int64_t>(-1))}),
      provider_options,
      17,
      ExpectedEPNodeAssignment::None);
}

// 3D input, last axis, static scale, U8 activations / U8 weights.
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_LastAxis_StaticScale_AU8_WU8) {
  RunSimplifiedLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// 3D input, last axis, static scale, U16 activations / U8 weights (contrib QDQ ops).
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_LastAxis_StaticScale_AU16_WU8) {
  RunSimplifiedLayerNormQDQTest<uint16_t, uint8_t>(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All,
      true /* use_contrib_qdq_ops */);
}

// 3D input, last axis, dynamic (non-initializer) scale, U8/U8.
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_LastAxis_DynamicScale_AU8_WU8) {
  RunSimplifiedLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      TestInputDef<float>({3}, false, GetFloatDataInRange(0.0f, 1.0f, 3)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// 4D input, last axis, static scale, U8/U8.
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_4D_LastAxis_AU8_WU8) {
  RunSimplifiedLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(-2.0f, 2.0f, 3)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// 3D input, last axis, custom epsilon, U8/U8.
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_LastAxis_CustomEpsilon_AU8_WU8) {
  RunSimplifiedLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({1, 2, 3}, false, GetFloatDataInRange(0.0f, 10.0f, 6)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(0.0f, 1.0f, 3)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1)),
       test::MakeAttribute("epsilon", 1e-3f)},
      ExpectedEPNodeAssignment::All);
}

// 2D input, last axis (axis=1), static scale, U8/U8.
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_2D_LastAxis_AU8_WU8) {
  RunSimplifiedLayerNormQDQTest<uint8_t, uint8_t>(
      TestInputDef<float>({4, 8}, false, GetFloatDataInRange(0.0f, 10.0f, 32)),
      TestInputDef<float>({8}, true, GetFloatDataInRange(0.0f, 1.0f, 8)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All);
}

// 4D input, last axis, static scale, U16/U8 (contrib QDQ ops).
TEST_F(QnnHTPBackendTests, SimplifiedLayerNorm_4D_LastAxis_AU16_WU8) {
  RunSimplifiedLayerNormQDQTest<uint16_t, uint8_t>(
      TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18)),
      TestInputDef<float>({3}, true, GetFloatDataInRange(-2.0f, 2.0f, 3)),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All,
      true /* use_contrib_qdq_ops */);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
