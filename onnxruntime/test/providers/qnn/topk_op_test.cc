// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/node_attr_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

// Returns a function that builds a model with a TopK operator.
template <typename DataType>
inline GetTestModelFn BuildTopKTestCase(const TestInputDef<DataType>& input_def,
                                        const TestInputDef<int64_t>& k_def,
                                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, k_def, attrs](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput<DataType>(builder, input_def);
    NodeArg* k_input = MakeTestInput<int64_t>(builder, k_def);

    NodeArg* values_output = builder.MakeOutput();
    NodeArg* indices_output = builder.MakeOutput();
    Node& topk_node = builder.AddNode("TopK", {input, k_input}, {values_output, indices_output});

    for (const auto& attr : attrs) {
      topk_node.AddAttributeProto(attr);
    }
  };
}

// Runs a model with a TopK operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunTopKTestOnCPU(const TestInputDef<DataType>& input_def,
                             const TestInputDef<int64_t>& k_def,
                             const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             int opset = 19) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "cpu";

  RunQnnModelTest(BuildTopKTestCase<DataType>(input_def, k_def, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that TopK with a dynamic K input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, TopK_DynamicK_Unsupported) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, false /* is_initializer */, {2}),
                          {},                               // Attributes
                          ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test that TopK with an axis attribute that is not the last dimension.
TEST_F(QnnCPUBackendTests, TopK_NonLastAxis) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
                          ExpectedEPNodeAssignment::All);
}

// Test that TopK with an axis attribute that is not the last dimension. Set largest to 0.
TEST_F(QnnCPUBackendTests, TopK_NonLastAxis_Largest_0) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {utils::MakeAttribute("axis", static_cast<int64_t>(1)),
                           utils::MakeAttribute("largest", static_cast<int64_t>(0))},
                          ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest floats from last axis
TEST_F(QnnCPUBackendTests, TopK_LargestFloats_LastAxis) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {},  // Attributes
                          ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest floats from last axis. Largest to 0.
TEST_F(QnnCPUBackendTests, TopK_LargestFloats_LastAxis_Largest_0) {
  RunTopKTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                          {utils::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                          ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest int32s from last axis
TEST_F(QnnCPUBackendTests, TopK_LargestInt32s_LastAxis) {
  std::vector<int32_t> input_data = {-6, -5, -4, -3, -2, 0, 1, 2, 3, 4, 5, 6};
  RunTopKTestOnCPU<int32_t>(TestInputDef<int32_t>({1, 2, 2, 3}, false, input_data),
                            TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                            {},  // Attributes
                            ExpectedEPNodeAssignment::All);
}

// Test TopK on CPU backend: top 2 largest int32s from last axis. Set largest to 0.
TEST_F(QnnCPUBackendTests, TopK_LargestInt32s_LastAxis_Largest_0) {
  std::vector<int32_t> input_data = {-6, -5, -4, -3, -2, 0, 1, 2, 3, 4, 5, 6};
  RunTopKTestOnCPU<int32_t>(TestInputDef<int32_t>({1, 2, 2, 3}, false, input_data),
                            TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                            {utils::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                            ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that creates a graph with a QDQ TopK operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQTopKTestCase(const TestInputDef<float>& input_def,
                                                  const TestInputDef<int64_t>& k_def,
                                                  const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                  bool use_contrib_qdq = false) {
  return [input_def, k_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                                    std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                   use_contrib_qdq);

    // K input
    NodeArg* k_input = MakeTestInput(builder, k_def);

    // TopK_values_output -> Q -> DQ -> output
    // NOTE: Create output QDQ nodes before the TopK node so that TopK's 'values' output is the graph's first output.
    NodeArg* values_output = builder.MakeIntermediate();
    output_qparams[0] = input_qparams;  // Input and output qparams must be equal.
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, values_output, input_qparams.scale,
                                                     input_qparams.zero_point, use_contrib_qdq);
    // TopK node
    NodeArg* indices_output = builder.MakeOutput();
    Node& topk_node = builder.AddNode("TopK", {input_qdq, k_input}, {values_output, indices_output});

    for (const auto& attr : attrs) {
      topk_node.AddAttributeProto(attr);
    }
  };
}

// Unit test to load a model and run.
TEST_F(QnnHTPBackendTests, topk_model_test) {
  const std::string path{"/local/mnt/workspace/0922_topk_model.onnx"};
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  TryEnableQNNSaver(provider_options);
  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  verification_params.fp32_abs_err = 1e-2f;
  verification_params.graph_verifier = nullptr;
  NameMLValMap feeds;
  AllocatorPtr allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{1414}};

  const std::vector<int64_t> input_shape{249216};
  auto input_tokens = rand_gen_.Uniform<float>(input_shape, -1000.0f, 1000.0f);
  OrtValue input_token_value;
  CreateMLValue<float>(allocator, input_shape, std::move(input_tokens), &input_token_value);
  feeds.insert(std::make_pair("/img_view_transformer/Reshape_11_output_0", input_token_value));

  RunAndVerifyOutputsWithEP(path, "QNN_EP_TestLogID",
                            QnnExecutionProviderWithOptions(provider_options),
                            /*feeds=*/feeds,
                            /*params=*/verification_params,
                            /*session_options_updater=*/{},
                            /*verify_outputs=*/true);
}

// Runs a QDQ TopK model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename QType>
static void RunQDQTopKTestOnHTP(const TestInputDef<float>& input_def,
                                const TestInputDef<int64_t>& k_def,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 19,
                                bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildTopKTestCase<float>(input_def, k_def, attrs);
  auto qdq_model_builder = BuildQDQTopKTestCase<QType>(input_def, k_def, attrs, use_contrib_qdq);
  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test 8-bit QDQ TopK on HTP backend: top 2 largest floats from last axis
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U8_LastAxis) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ TopK on HTP backend: top 2 largest floats from last axis. Set largest to 0.
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U8_LastAxis_Largest_0) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {utils::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ TopK on HTP backend: non-last axis
TEST_F(QnnHTPBackendTests, TopK_U8_NonLastAxis) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {utils::MakeAttribute("axis", static_cast<int64_t>(1))},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ TopK on HTP backend: non-last axis. Set largest to 0.
TEST_F(QnnHTPBackendTests, TopK_U8_NonLastAxis_Largest_0) {
  RunQDQTopKTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                               {utils::MakeAttribute("axis", static_cast<int64_t>(1)),
                                utils::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ TopK on HTP backend: top 2 largest floats from last axis
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U16_LastAxis) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}

// Test 16-bit QDQ TopK on HTP backend: top 2 largest floats from last axis. Set largest to 0.
TEST_F(QnnHTPBackendTests, TopK_LargestFloats_U16_LastAxis_Largest_0) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {utils::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}

// Test 16-bit QDQ TopK on HTP backend: non-last axis
TEST_F(QnnHTPBackendTests, TopK_U16_NonLastAxis) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {utils::MakeAttribute("axis", static_cast<int64_t>(1))},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}

// Test 16-bit QDQ TopK on HTP backend: non-last axis. Set largest to 0.
TEST_F(QnnHTPBackendTests, TopK_U16_NonLastAxis_Largest_0) {
  RunQDQTopKTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-20.0f, 20.0f, 48)),
                                TestInputDef<int64_t>({1}, true /* is_initializer */, {2}),
                                {utils::MakeAttribute("axis", static_cast<int64_t>(1)),
                                 utils::MakeAttribute("largest", static_cast<int64_t>(0))},  // Attributes
                                ExpectedEPNodeAssignment::All,
                                21);  // opset
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
