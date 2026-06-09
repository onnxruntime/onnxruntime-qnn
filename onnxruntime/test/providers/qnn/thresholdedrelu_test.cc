// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a ThresholdedRelu operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunThresholdedReluTest(const std::vector<TestInputDef<DataType>>& input_defs,
                                   const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   const std::string& backend_name = "cpu",
                                   float fp32_abs_err = 1e-5f,
                                   int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<DataType>("ThresholdedRelu_node", "ThresholdedRelu", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//
TEST_F(QnnCPUBackendTests, ThresholdedRelu) {
  // Test that ThresholdedRelu with fp32 input.
  RandomValueGenerator rand_gen_{std::optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto input = rand_gen_.Uniform<float>(dividend_shape, -100.0f, 100.0f);

  RunThresholdedReluTest<float>({TestInputDef<float>({1, 4, 5}, false, input)},
                                {test::MakeAttribute("alpha", 4.5f)},
                                ExpectedEPNodeAssignment::All);
}

// Case 1: alpha=0, equivalent to ReLU (output = input > 0 ? input : 0).
TEST_F(QnnCPUBackendTests, ThresholdedRelu_Alpha0) {
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 3.0f, -2.0f, 4.0f,
                              -1.5f, 0.1f, -0.1f, 5.0f, -4.0f, 2.5f, -2.5f, 1.5f, -3.5f, 0.2f};
  RunThresholdedReluTest<float>({TestInputDef<float>({1, 4, 5}, false, input)},
                                {test::MakeAttribute("alpha", 0.0f)},
                                ExpectedEPNodeAssignment::All);
}

// Case 2: alpha=-1, values <= -1 are zeroed out (output = input > -1 ? input : 0).
TEST_F(QnnCPUBackendTests, ThresholdedRelu_AlphaNeg1) {
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 3.0f, -2.0f, 4.0f,
                              -1.5f, 0.1f, -0.1f, 5.0f, -4.0f, 2.5f, -2.5f, 1.5f, -3.5f, 0.2f};
  RunThresholdedReluTest<float>({TestInputDef<float>({1, 4, 5}, false, input)},
                                {test::MakeAttribute("alpha", -1.0f)},
                                ExpectedEPNodeAssignment::All);
}

// Case 3: alpha=x (arbitrary positive threshold, output = input > x ? input : 0).
TEST_F(QnnCPUBackendTests, ThresholdedRelu_AlphaX) {
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 3.0f, -2.0f, 4.0f,
                              -1.5f, 0.1f, -0.1f, 5.0f, -4.0f, 2.5f, -2.5f, 1.5f, -3.5f, 0.2f};
  RunThresholdedReluTest<float>({TestInputDef<float>({1, 4, 5}, false, input)},
                                {test::MakeAttribute("alpha", 2.0f)},
                                ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that builds a model with a QDQ ThresholdedRelu node.
template <typename InputAQType, typename InputBQType>
inline GetTestQDQModelFn<InputAQType> BuildQDQThresholdedReluTestCase(const std::vector<TestInputDef<float>>& input_defs,
                                                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                                      bool use_contrib_qdq = false) {
  return [input_defs, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                              std::vector<QuantParams<InputAQType>>& output_qparams) {
    const size_t num_inputs = input_defs.size();
    QNN_ASSERT(num_inputs == 1);

    // input
    MakeTestInput<float>(builder, "input0", input_defs[0]);

    // input0 -> Q -> DQ -> input0_qdq
    const QuantParams<InputAQType> input0_qparams = GetTestInputQuantParams<InputAQType>(input_defs[0]);
    const std::string input0_qdq =
        AddQDQNodePair<InputAQType>(builder, "qdq_in", "input0",
                                    input0_qparams.scale, input0_qparams.zero_point, use_contrib_qdq);

    // ThresholdedRelu -> op_out
    builder.AddNode("ThresholdedRelu",
                    "ThresholdedRelu",
                    {input0_qdq},
                    {"op_out"},
                    kOnnxDomain,
                    attrs);

    // op_out -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputAQType>(builder,
                                                       "qdq_out",
                                                       "op_out",
                                                       output_qparams[0].scale,
                                                       output_qparams[0].zero_point,
                                                       use_contrib_qdq);
  };
}

template <typename InputAQType, typename InputBQType>
static void RunQDQThresholdedReluTestOnHTP(const std::vector<TestInputDef<float>>& input_defs,
                                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                           ExpectedEPNodeAssignment expected_ep_assignment,
                                           int opset = 13,
                                           bool use_contrib_qdq = false,
                                           QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildOpTestCase<float>("ThresholdedRelu_node", "ThresholdedRelu", input_defs, {}, attrs);
  auto qdq_model_builder = BuildQDQThresholdedReluTestCase<InputAQType, InputBQType>(input_defs, attrs, use_contrib_qdq);
  TestQDQModelAccuracy<InputAQType>(f32_model_builder,
                                    qdq_model_builder,
                                    provider_options,
                                    opset,
                                    expected_ep_assignment,
                                    tolerance);
}

// Test ThresholdedRelu QDQ.
TEST_F(QnnHTPBackendTests, ThresholdedRelu_qdq) {
  std::vector<float> input = GetFloatDataInRange(-10.0f, 10.0f, 20);
  RunQDQThresholdedReluTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 4, 5}, false, input)},
                                                   {test::MakeAttribute("alpha", 4.5f)},
                                                   ExpectedEPNodeAssignment::All);
}

// Case 1: alpha=0, equivalent to ReLU (output = input > 0 ? input : 0).
TEST_F(QnnHTPBackendTests, ThresholdedRelu_Alpha0) {
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 3.0f, -2.0f, 4.0f,
                              -1.5f, 0.1f, -0.1f, 5.0f, -4.0f, 2.5f, -2.5f, 1.5f, -3.5f, 0.2f};
  RunQDQThresholdedReluTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 4, 5}, false, input)},
                                                   {test::MakeAttribute("alpha", 0.0f)},
                                                   ExpectedEPNodeAssignment::All);
}

// Case 2: alpha=-1, values <= -1 are zeroed out (output = input > -1 ? input : 0).
TEST_F(QnnHTPBackendTests, ThresholdedRelu_AlphaNeg1) {
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 3.0f, -2.0f, 4.0f,
                              -1.5f, 0.1f, -0.1f, 5.0f, -4.0f, 2.5f, -2.5f, 1.5f, -3.5f, 0.2f};
  RunQDQThresholdedReluTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 4, 5}, false, input)},
                                                   {test::MakeAttribute("alpha", -1.0f)},
                                                   ExpectedEPNodeAssignment::All);
}

// Case 3: alpha=x (arbitrary positive threshold, output = input > x ? input : 0).
TEST_F(QnnHTPBackendTests, ThresholdedRelu_AlphaX) {
  std::vector<float> input = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 3.0f, -2.0f, 4.0f,
                              -1.5f, 0.1f, -0.1f, 5.0f, -4.0f, 2.5f, -2.5f, 1.5f, -3.5f, 0.2f};
  RunQDQThresholdedReluTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 4, 5}, false, input)},
                                                   {test::MakeAttribute("alpha", 2.0f)},
                                                   ExpectedEPNodeAssignment::All);
}

// Test ThresholdedRelu.
TEST_F(QnnHTPBackendTests, ThresholdedRelu_fp32) {
  RandomValueGenerator rand_gen_{std::optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto input = rand_gen_.Uniform<float>(dividend_shape, -10.0f, 10.0f);

  RunThresholdedReluTest<float>({TestInputDef<float>({1, 4, 5}, false, input)},
                                {test::MakeAttribute("alpha", 4.5f)},
                                ExpectedEPNodeAssignment::All,
                                "htp",
                                0.004f);  // Tolerance. Comparing fp16 (QNN) with fp32 (CPU EP), so expect to need larger tolerance.);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
