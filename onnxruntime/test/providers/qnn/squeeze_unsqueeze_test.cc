// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "core/graph/node_attr_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Squeeze (or Unsqueeze) operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunSqueezeTestOnCPU(const std::string& op_type,  // Squeeze or Unsqueeze
                                const TestInputDef<DataType>& input_def,
                                const TestInputDef<int64_t>& axes_def,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "cpu";

  RunQnnModelTest(BuildOpTestCase<DataType, int64_t>(op_type + "_node", op_type, {input_def}, {axes_def}, {}),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Squeeze with a dynamic axes input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Squeeze_DynamicAxes_Unsupported) {
  RunSqueezeTestOnCPU("Squeeze",
                      TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                      TestInputDef<int64_t>({1}, false /* is_initializer */, {0}),
                      ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test that Unsqueeze with a dynamic axes input is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Unsqueeze_DynamicAxes_Unsupported) {
  RunSqueezeTestOnCPU("Unsqueeze",
                      TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                      TestInputDef<int64_t>({1}, false /* is_initializer */, {0}),
                      ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test Squeeze of rank 5 -> rank 2.
TEST_F(QnnCPUBackendTests, Squeeze_Rank5_Rank2_f32) {
  RunSqueezeTestOnCPU("Squeeze",
                      TestInputDef<float>({1, 3, 1, 2, 4}, false, -10.0f, 10.0f),
                      TestInputDef<int64_t>({2}, true, {0, 2}),  // Squeeze axes 0 and 2 => (3, 2, 4)
                      ExpectedEPNodeAssignment::All);
}

// Test Squeeze of rank 4 -> rank 3 with a negative axes value.
TEST_F(QnnCPUBackendTests, Squeeze_Rank4_Rank3_NegAxes_f32) {
  RunSqueezeTestOnCPU("Squeeze",
                      TestInputDef<float>({1, 3, 2, 1}, false, -10.0f, 10.0f),
                      TestInputDef<int64_t>({1}, true, {-1}),  // Squeeze last axis => (1, 3, 2)
                      ExpectedEPNodeAssignment::All);
}

// Test Unsqueeze of rank 3 -> rank 5.
TEST_F(QnnCPUBackendTests, Unsqueeze_Rank3_Rank5_f32) {
  RunSqueezeTestOnCPU("Unsqueeze",
                      TestInputDef<float>({3, 2, 4}, false, -10.0f, 10.0f),
                      TestInputDef<int64_t>({2}, true, {0, 2}),  // Add 1's => (1, 3, 1, 2, 4)
                      ExpectedEPNodeAssignment::All);
}

// Test Unsqueeze of rank 3 -> rank 4 with a negative axes value.
TEST_F(QnnCPUBackendTests, Unsqueeze_Rank3_Rank4_NegAxes_f32) {
  RunSqueezeTestOnCPU("Unsqueeze",
                      TestInputDef<float>({1, 3, 2}, false, -10.0f, 10.0f),
                      TestInputDef<int64_t>({1}, true, {-1}),  // Add 1 as last axis => (1, 3, 2, 1)
                      ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that creates a graph with a QDQ (Un)Squeeze operator.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQSqueezeTestCase(const std::string& op_type,  // Squeeze or Unsqueeze
                                                     const TestInputDef<float>& input_def,
                                                     const TestInputDef<int64_t>& axes_def,
                                                     bool use_contrib_qdq = false) {
  return [op_type, input_def, axes_def,
          use_contrib_qdq](ModelTestBuilder& builder,
                           std::vector<QuantParams<QuantType>>& output_qparams) {
    // input
    MakeTestInput(builder, "input", input_def);

    // input -> Q -> DQ -> input_qdq
    const QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input",
                                  input_qparams.scale, input_qparams.zero_point, use_contrib_qdq);

    // axes input
    MakeTestInput(builder, "axes", axes_def);

    // (Un)Squeeze op
    builder.AddNode(op_type,
                    op_type,
                    {input_qdq, "axes"},
                    {"op_out"});

    // op_out -> Q -> DQ -> graph output
    // NOTE: Input and output quantization parameters must be equal for (Un)Squeeze.
    output_qparams[0] = input_qparams;
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder,
                                                     "qdq_out",
                                                     "op_out",
                                                     input_qparams.scale,
                                                     input_qparams.zero_point,
                                                     use_contrib_qdq);
  };
}

// Runs a model with a non-QDQ (Un)Squeeze operator on the QNN HTP backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunSqueezeTestOnHTP(const std::string& op_type,  // Squeeze or Unsqueeze
                                const TestInputDef<DataType>& input_def,
                                const TestInputDef<int64_t>& axes_def,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";

  RunQnnModelTest(BuildOpTestCase<DataType, int64_t>(op_type + "_node", op_type, {input_def}, {axes_def}, {}),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ (Un)Squeeze model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and
// that inference running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP
// (when compared to the baseline float32 model).
template <typename QType>
static void RunQDQSqueezeTestOnHTP(const std::string& op_type,
                                   const TestInputDef<float>& input_def,
                                   const TestInputDef<int64_t>& axes_def,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   int opset = 13,
                                   bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildOpTestCase<float, int64_t>(op_type + "_node", op_type, {input_def}, {axes_def}, {});
  auto qdq_model_builder = BuildQDQSqueezeTestCase<QType>(op_type, input_def, axes_def, use_contrib_qdq);

  TestQDQModelAccuracy(f32_model_builder,
                       qdq_model_builder,
                       provider_options,
                       opset,
                       expected_ep_assignment);
}

// Test that QDQ Squeeze with a dynamic axes input is not supported by QNN EP.
TEST_F(QnnHTPBackendTests, Squeeze_DynamicAxes_Unsupported) {
  RunQDQSqueezeTestOnHTP<uint8_t>("Squeeze",
                                  TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                  TestInputDef<int64_t>({1}, false /* is_initializer */, {0}),
                                  ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test that Unsqueeze with a dynamic axes input is not supported by QNN EP.
TEST_F(QnnHTPBackendTests, Unsqueeze_DynamicAxes_Unsupported) {
  RunQDQSqueezeTestOnHTP<uint8_t>("Unsqueeze",
                                  TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                                  TestInputDef<int64_t>({1}, false /* is_initializer */, {0}),
                                  ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test Squeeze of rank 5 -> rank 2.
TEST_F(QnnHTPBackendTests, Squeeze_Rank5_Rank2_f32) {
  // We can't use the usual model-building functions because they add standalone Quantize and Dequantize nodes
  // at the input and output. These Q/DQ ops get lowered to QNN's Quantize and Dequantize operators, which DO NOT
  // support rank 5 tensors. Therefore, we have to create a test model that only instantiates the DQ -> Squeeze -> Q
  // QDQ node group, which gets lowered to a single QNN Reshape node.
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    // input (u8) -> DQ -> input_dq
    builder.MakeInput<uint8_t>("quant_input", {1, 3, 1, 2, 4}, 0, 255);
    builder.AddDequantizeLinearNode<uint8_t>("dq_in", "quant_input", 1.0f, 0, "input_dq");  // scale = 1.0, zp = 0

    // axes initializer
    builder.Make1DInitializer<int64_t>("axes", {0, 2});  // Squeeze axes 0 and 2 => (3, 2, 4)

    // Squeeze -> squeeze_out
    builder.AddNode("Squeeze",
                    "Squeeze",
                    {"input_dq", "axes"},
                    {"squeeze_out"});

    // Q -> output (u8)
    builder.MakeOutput("Y");
    builder.AddQuantizeLinearNode<uint8_t>("q_out", "squeeze_out", 1.0f, 0, "Y");  // scale = 1.0, zp = 0
  };

  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Squeeze of rank 4 -> rank 3 with a negative axes value.
TEST_F(QnnHTPBackendTests, Squeeze_Rank4_Rank3_NegAxes_u8) {
  RunQDQSqueezeTestOnHTP<uint8_t>("Squeeze",
                                  TestInputDef<float>({1, 3, 2, 1}, false, -10.0f, 10.0f),
                                  TestInputDef<int64_t>({1}, true, {-1}),  // Squeeze last axis => (1, 3, 2)
                                  ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Squeeze of rank 4 -> rank 3 with a negative axes value.
TEST_F(QnnHTPBackendTests, Squeeze_Rank4_Rank3_NegAxes_u16) {
  RunQDQSqueezeTestOnHTP<uint16_t>("Squeeze",
                                   TestInputDef<float>({1, 3, 2, 1}, false, -10.0f, 10.0f),
                                   TestInputDef<int64_t>({1}, true, {-1}),  // Squeeze last axis => (1, 3, 2)
                                   ExpectedEPNodeAssignment::All,
                                   13,     // opset
                                   true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ Unsqueeze of rank 3 -> rank 5.
TEST_F(QnnHTPBackendTests, Unsqueeze_Rank3_Rank5_f32) {
  // We can't use the usual model-building functions because they add standalone Quantize and Dequantize nodes
  // at the input and output. These Q/DQ ops get lowered to QNN's Quantize and Dequantize operators, which DO NOT
  // support rank 5 tensors. Therefore, we have to create a test model that only instantiates the DQ -> Unsqueeze -> Q
  // QDQ node group, which gets lowered to a single QNN Reshape node.
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    // input (u8) -> DQ -> input_dq
    builder.MakeInput<uint8_t>("quant_input", {3, 2, 4}, 0, 255);
    builder.AddDequantizeLinearNode<uint8_t>("dq_in", "quant_input", 1.0f, 0, "input_dq");  // scale = 1.0, zp = 0

    // axes initializer
    builder.Make1DInitializer<int64_t>("axes", {0, 2});  // Add 1's => (1, 3, 1, 2, 4)

    // Unsqueeze -> unsqueeze_out
    builder.AddNode("Unsqueeze",
                    "Unsqueeze",
                    {"input_dq", "axes"},
                    {"unsqueeze_out"});

    // Q -> output (u8)
    builder.MakeOutput("Y");
    builder.AddQuantizeLinearNode<uint8_t>("q_out", "unsqueeze_out", 1.0f, 0, "Y");  // scale = 1.0, zp = 0
  };

  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ Unsqueeze of rank 3 -> rank 4 with a negative axes value.
TEST_F(QnnHTPBackendTests, Unsqueeze_Rank3_Rank4_NegAxes_u8) {
  RunQDQSqueezeTestOnHTP<uint8_t>("Unsqueeze",
                                  TestInputDef<float>({1, 3, 2}, false, -10.0f, 10.0f),
                                  TestInputDef<int64_t>({1}, true, {-1}),  // Add 1 as last axis => (1, 3, 2, 1)
                                  ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Unsqueeze of rank 3 -> rank 4 with a negative axes value.
TEST_F(QnnHTPBackendTests, Unsqueeze_Rank3_Rank4_NegAxes_u16) {
  RunQDQSqueezeTestOnHTP<uint16_t>("Unsqueeze",
                                   TestInputDef<float>({1, 3, 2}, false, -10.0f, 10.0f),
                                   TestInputDef<int64_t>({1}, true, {-1}),  // Add 1 as last axis => (1, 3, 2, 1)
                                   ExpectedEPNodeAssignment::All,
                                   13,     // opset
                                   true);  // Use com.microsoft Q/DQ ops
}

// Test that int32 Squeeze runs on HTP backend.
TEST_F(QnnHTPBackendTests, Squeeze_Int32_Rank4_Rank3) {
  std::vector<int32_t> input_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  RunSqueezeTestOnHTP<int32_t>("Squeeze",
                               TestInputDef<int32_t>({1, 3, 2, 2}, false, input_data),
                               TestInputDef<int64_t>({1}, true, {0}),  // Squeeze 0th axis => (3, 2, 2)
                               ExpectedEPNodeAssignment::All);
}

// Test that int32 Unsqueeze runs on HTP backend.
TEST_F(QnnHTPBackendTests, Unsqueeze_Int32_Rank3_Rank4) {
  std::vector<int32_t> input_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  RunSqueezeTestOnHTP<int32_t>("Unsqueeze",
                               TestInputDef<int32_t>({3, 2, 2}, false, input_data),
                               TestInputDef<int64_t>({1}, true, {0}),  // Unsqueeze 0th axis => (1, 3, 2, 2)
                               ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
