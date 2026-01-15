// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// // Runs a model with a ThresholdedRelu operator on the QNN CPU backend. Checks the graph node assignment
// // and that inference outputs for QNN EP and CPU EP match.
// template <typename DataType>
// static void RunThresholdedReluTest(const std::vector<TestInputDef<DataType>>& input_defs,
//                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
//                                    ExpectedEPNodeAssignment expected_ep_assignment,
//                                    const std::string& backend_name = "cpu",
//                                    float fp32_abs_err = 1e-5f,
//                                    int opset = 13) {
//   ProviderOptions provider_options;

//   provider_options["backend_type"] = backend_name;
//   provider_options["offload_graph_io_quantization"] = "0";

//   RunQnnModelTest(BuildOpTestCase<DataType>("ThresholdedRelu", input_defs, {}, attrs),
//                   provider_options,
//                   opset,
//                   expected_ep_assignment,
//                   fp32_abs_err);
// }

//
// CPU tests:
//
// TEST_F(QnnCPUBackendTests, ThresholdedRelu) {
//   // Test that ThresholdedRelu with fp32 input.
//   RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
//   const std::vector<int64_t> dividend_shape{1, 4, 5};
//   auto input = rand_gen_.Uniform<float>(dividend_shape, -100.0f, 100.0f);

//   RunThresholdedReluTest<float>({TestInputDef<float>({1, 4, 5}, false, input)},
//                                 {utils::MakeAttribute("alpha", 4.5f)},
//                                 ExpectedEPNodeAssignment::All);
// }

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that builds a model with a QDQ ThresholdedRelu node.
// template <typename InputAQType, typename InputBQType>
// inline GetTestQDQModelFn<InputAQType> BuildQDQThresholdedReluTestCase(const std::vector<TestInputDef<float>>& input_defs,
//                                                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
//                                                                       bool use_contrib_qdq = false) {
//   return [input_defs, attrs, use_contrib_qdq](ModelTestBuilder& builder,
//                                               std::vector<QuantParams<InputAQType>>& output_qparams) {
//     const size_t num_inputs = input_defs.size();
//     std::vector<NodeArg*> op_inputs;
//     op_inputs.reserve(num_inputs);

//     // Process input 0
//     NodeArg* input0 = MakeTestInput<float>(builder, input_defs[0]);
//     QuantParams<InputAQType> input0_qparams = GetTestInputQuantParams<InputAQType>(input_defs[0]);
//     NodeArg* input0_after_qdq = AddQDQNodePair<InputAQType>(builder, input0, input0_qparams.scale,
//                                                             input0_qparams.zero_point, use_contrib_qdq);
//     op_inputs.push_back(input0_after_qdq);

//     // Op -> op_output
//     auto* ThresholdedRelu_output = builder.MakeIntermediate();
//     Node& ThresholdedRelu_node = builder.AddNode("ThresholdedRelu", op_inputs, {ThresholdedRelu_output});

//     for (const auto& attr : attrs) {
//       ThresholdedRelu_node.AddAttributeProto(attr);
//     }

//     // op_output -> Q -> DQ -> output
//     AddQDQNodePairWithOutputAsGraphOutput<InputAQType>(builder, ThresholdedRelu_output, output_qparams[0].scale,
//                                                        output_qparams[0].zero_point, use_contrib_qdq);
//   };
// }

// template <typename InputAQType, typename InputBQType>
// static void RunQDQThresholdedReluTestOnHTP(const std::vector<TestInputDef<float>>& input_defs,
//                                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
//                                            ExpectedEPNodeAssignment expected_ep_assignment,
//                                            int opset = 13,
//                                            bool use_contrib_qdq = false,
//                                            QDQTolerance tolerance = QDQTolerance()) {
//   ProviderOptions provider_options;

//   provider_options["backend_type"] = "htp";
//   provider_options["offload_graph_io_quantization"] = "0";

//   auto f32_model_builder = BuildOpTestCase<float>("ThresholdedRelu", input_defs, {}, attrs);
//   auto qdq_model_builder = BuildQDQThresholdedReluTestCase<InputAQType, InputBQType>(input_defs, attrs, use_contrib_qdq);
//   TestQDQModelAccuracy<InputAQType>(f32_model_builder,
//                                     qdq_model_builder,
//                                     provider_options,
//                                     opset,
//                                     expected_ep_assignment,
//                                     tolerance);
// }

TEST_F(QnnHTPBackendTests, DumpDlcTest) {
  // const std::string path{"/local/mnt/workspace/1217_pruned_model.onnx"};

  const std::string path{"/local/mnt/workspace/roialign_example.onnx"};
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["soc_model"] = "75";
  TryEnableQNNSaver(provider_options);
  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  verification_params.fp32_abs_err = 1e-2f;
  verification_params.graph_verifier = nullptr;
  NameMLValMap feeds;
  AllocatorPtr allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};

  // const std::vector<int64_t> input_tokens_shape{1, 384};
  // auto input_tokens_value = rand_gen_.Uniform<int32_t>(input_tokens_shape, 0.0f, 10000.0f);
  // OrtValue input_tokens;
  // CreateMLValue<int32_t>(allocator, input_tokens_shape, std::move(input_tokens_value), &input_tokens);
  // feeds.insert(std::make_pair("input_tokens", input_tokens));

  // const std::vector<int64_t> attention_masks_shape{1, 384, 128};
  // auto attention_masks_value = rand_gen_.Uniform<float>(attention_masks_shape, 0.0f, 1.0f);
  // OrtValue attention_masks;
  // CreateMLValue<float>(allocator, attention_masks_shape, std::move(attention_masks_value), &attention_masks);
  // feeds.insert(std::make_pair("/model/predictions/LayerNorm/LayerNormalization_output_0", attention_masks));

  // const std::vector<int64_t> mask_indices_shape{1};
  // auto mask_indices_value = rand_gen_.Uniform<int32_t>(mask_indices_shape, 0.0f, 10.0f);
  // OrtValue mask_indices;
  // CreateMLValue<int32_t>(allocator, mask_indices_shape, std::move(mask_indices_value), &mask_indices);
  // feeds.insert(std::make_pair("mask_indices", mask_indices));

  const std::vector<int64_t> x_shape{1, 1024, 50, 80};
  auto x = rand_gen_.Uniform<float>(x_shape, 0.0f, 1.0f);
  OrtValue x_value;
  CreateMLValue<float>(allocator, x_shape, std::move(x), &x_value);
  feeds.insert(std::make_pair("x", x_value));

  const std::vector<int64_t> batch_indices_shape{2};
  auto batch_indices = rand_gen_.Uniform<int64_t>(batch_indices_shape, 0.0f, 1.0f);
  OrtValue batch_indices_value;
  CreateMLValue<int64_t>(allocator, batch_indices_shape, std::move(batch_indices), &batch_indices_value);
  feeds.insert(std::make_pair("batch_indices", batch_indices_value));

  RunAndVerifyOutputsWithEP(path, "QNN_EP_TestLogID",
                            QnnExecutionProviderWithOptions(provider_options),
                            /*feeds=*/feeds,
                            /*params=*/verification_params,
                            /*session_options_updater=*/{},
                            /*verify_outputs=*/true);
}



#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
