// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/inference_session.h"
#include "core/framework/session_options.h"

#include "test/unittest_util/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Function that builds a QDQ model with an InstanceNormalization operator.
template <typename ActivationQType, typename ScaleQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQInstanceNormTestCase(const TestInputDef<float>& input_def,
                                                                       const TestInputDef<float>& scale_def,
                                                                       const TestInputDef<float>& bias_def,
                                                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                                       bool use_contrib_qdq = false) {
  return [input_def, scale_def, bias_def, attrs,
          use_contrib_qdq](ModelTestBuilder& builder,
                           std::vector<QuantParams<ActivationQType>>& output_qparams) {
    // input => Q => DQ =>
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<ActivationQType> input_qparams = GetTestInputQuantParams<ActivationQType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair(builder, input, input_qparams.scale, input_qparams.zero_point,
                                        use_contrib_qdq);

    // scale => Q => DQ =>
    NodeArg* scale = MakeTestInput(builder, scale_def);
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);
    NodeArg* scale_qdq = AddQDQNodePair(builder, scale, scale_qparams.scale, scale_qparams.zero_point,
                                        use_contrib_qdq);

    // bias (as int32) => DQ =>
    NodeArg* bias_qdq = MakeTestQDQBiasInput(builder, bias_def, input_qparams.scale * scale_qparams.scale,
                                             use_contrib_qdq);

    // InstanceNormalization operator.
    auto* instance_norm_output = builder.MakeIntermediate();
    Node& inst_norm_node = builder.AddNode("InstanceNormalization", {input_qdq, scale_qdq, bias_qdq},
                                           {instance_norm_output});
    for (const auto& attr : attrs) {
      inst_norm_node.AddAttributeProto(attr);
    }

    // Add instance_norm_output -> Q -> output_u8
    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(builder, instance_norm_output, output_qparams[0].scale,
                                                           output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, SSRInstanceNormU8U8) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_ssr_handling"] = "1";
  // fails with QNN 2.15.1 with the following fixed input.
  std::vector<float> input_data = {3.21289f, -5.9981f, -1.72799f, 6.27263f, 3.36205f, -1.93515f, -5.40113f, 3.75648f, 6.15357f,
                                   -5.25769f, 2.73637f, -0.901382f, -6.55612f, 1.99497f, -4.79228f, 2.69813f, 8.3064f, 0.0362501f};
  std::vector<float> scale_data = {-0.148738f, -1.45158f};
  std::vector<float> bias_data = {-2.2785083772f, 2.3338717017f};
  auto input_def = TestInputDef<float>({1, 2, 3, 3}, false, input_data).OverrideValueRange(-10.0f, 10.0f);
  auto scale_def = TestInputDef<float>({2}, true, scale_data).OverrideValueRange(-2.0f, 2.0f);
  auto bias_def = TestInputDef<float>({2}, true, bias_data).OverrideValueRange(-3.0f, 3.0f);
  TestQDQModelAccuracy(BuildOpTestCase<float>("InstanceNormalization", {input_def, scale_def, bias_def}, {}, {}),
                       BuildQDQInstanceNormTestCase<uint8_t, uint8_t>(input_def, scale_def, bias_def, {}, true),
                       provider_options,
                       18,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       logging::Severity::kVERBOSE,
                       "",
                       {},
                       nullptr, 
                       true);
}

TEST_F(QnnHTPBackendTests, SSRInferenceSession) {
  std::vector<float> input_data = {3.21289f, -5.9981f, -1.72799f, 6.27263f, 3.36205f, -1.93515f, -5.40113f, 3.75648f, 6.15357f,
                                   -5.25769f, 2.73637f, -0.901382f, -6.55612f, 1.99497f, -4.79228f, 2.69813f, 8.3064f, 0.0362501f};
  std::vector<float> scale_data = {-0.148738f, -1.45158f};
  std::vector<float> bias_data = {-2.2785083772f, 2.3338717017f};
  auto input_def = TestInputDef<float>({1, 2, 3, 3}, false, input_data).OverrideValueRange(-10.0f, 10.0f);
  auto scale_def = TestInputDef<float>({2}, true, scale_data).OverrideValueRange(-2.0f, 2.0f);
  auto bias_def = TestInputDef<float>({2}, true, bias_data).OverrideValueRange(-3.0f, 3.0f);

  ProviderOptions provider_options;
  provider_options["backend_path"] = "QnnMockSSR.dll";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_ssr_handling"] = "1";
  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  auto qnn_ep = QnnExecutionProviderWithOptions(provider_options, &so);
  std::string provider_type = qnn_ep->Type();
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(qnn_ep)));

  auto& logging_manager = DefaultLoggingManager();
  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), {}, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  BuildOpTestCase<float>("InstanceNormalization", {input_def, scale_def, bias_def}, {}, {})(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_object.Initialize());

}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
