// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Gemm operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunGemmTest(const std::vector<TestInputDef<DataType>>& input_defs,
                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        const std::string& backend_name = "cpu",
                        int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>("Gemm_node", "Gemm", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment});
}

//
// CPU tests:
//

// Test that Gemm with non-default 'alpha' or 'beta' attributes is not supported by QNN EP.
TEST_F(QnnCPUBackendTests, Gemm_NonDefaultAlphaBeta_Unsupported) {
  // Check that alpha != 1.0f is not supported.
  RunGemmTest<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -10.0f, 10.0f)},
                     {test::MakeAttribute("alpha", 1.5f)},
                     ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.

  // Check that non-zero, non-default beta is not supported.
  RunGemmTest<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -10.0f, 10.0f),
                      TestInputDef<float>({1, 4}, false, -1.0f, 1.0f)},
                     {test::MakeAttribute("beta", 1.2f)},
                     ExpectedEPNodeAssignment::None);  // Should not be assigned to QNN EP.
}

// Test Gemm with 2D bias is supported.
TEST_F(QnnCPUBackendTests, Gemm_2D_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 12);

  // 2D matrix mul with bias is supported.
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, input_a_data),
                      TestInputDef<float>({3, 4}, false, input_b_data),
                      TestInputDef<float>({2, 4}, false, -1.0f, 1.0f)},
                     {},
                     ExpectedEPNodeAssignment::All);  // Assigned to QNN EP.

  // However, 2D matrix mul without a bias is supported. Input A's 0th dimension is interpreted as `batch_size`.
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, input_a_data),
                      TestInputDef<float>({3, 4}, false, input_b_data)},
                     {},
                     ExpectedEPNodeAssignment::All);  // Assigned to QNN EP.
}

// Test Gemm with dynamic (i.e., not initializer) inputs (A, B, Bias).
TEST_F(QnnCPUBackendTests, Gemm_Dynamic_A_B_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, false, input_b_data),
                      TestInputDef<float>({1, 4}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

// Test Gemm with static B and Bias inputs.
TEST_F(QnnCPUBackendTests, Gemm_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

// Test Gemm with beta=0.0: bias is present but must be ignored.
TEST_F(QnnCPUBackendTests, Gemm_ZeroBeta_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {test::MakeAttribute("beta", 0.0f)},
                     ExpectedEPNodeAssignment::All);
}

// Test Gemm with transposed A/B and static B and Bias inputs.
TEST_F(QnnCPUBackendTests, Gemm_TransAB_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                      TestInputDef<float>({4, 6}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {test::MakeAttribute("transA", static_cast<int64_t>(1)),
                      test::MakeAttribute("transB", static_cast<int64_t>(1))},
                     ExpectedEPNodeAssignment::All);
}

// Test Gemm with transposed A/B and dynamic (i.e., not initializer) B and Bias inputs.
TEST_F(QnnCPUBackendTests, Gemm_TransAB_Dynamic_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                      TestInputDef<float>({4, 6}, false, input_b_data),
                      TestInputDef<float>({1, 4}, false, input_c_data)},
                     {test::MakeAttribute("transA", static_cast<int64_t>(1)),
                      test::MakeAttribute("transB", static_cast<int64_t>(1))},
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Gemm_Broadcast_Bias_DynamicInputs) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // All dynamic inputs
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, false, input_b_data),
                      TestInputDef<float>({3}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_DynamicC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // Dynamic A, static B, dynamic C
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, true, input_b_data),
                      TestInputDef<float>({3}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_StaticC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // Dynamic A, static B, static C
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, true, input_b_data),
                      TestInputDef<float>({3}, true, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

// FullyConnected cannot consume C=[M,1] as a bias, but ElementWiseAdd can broadcast it over N.
TEST_F(QnnCPUBackendTests, Gemm_ColumnBroadcast_DynamicC) {
  constexpr int64_t M = 2;
  constexpr int64_t K = 4;
  constexpr int64_t N = 3;
  RunGemmTest<float>({TestInputDef<float>({M, K}, false, -1.0f, 1.0f),
                      TestInputDef<float>({K, N}, true, -1.0f, 1.0f),
                      TestInputDef<float>({M, 1}, false, -1.0f, 1.0f)},
                     {},
                     ExpectedEPNodeAssignment::All);
}

namespace {
GetTestModelFn BuildReshapeGemmTestCase(const TestInputDef<float>& input, const TestInputDef<int64_t>& shape,
                                        const TestInputDef<float>& weight, const TestInputDef<float>& bias) {
  return [input, shape, weight, bias](ModelTestBuilder& builder) {
    // Inputs
    MakeTestInput(builder, "X", input);
    MakeTestInput(builder, "shape", shape);

    // Reshape
    builder.AddNode("reshape", "Reshape", {"X", "shape"}, {"reshaped"});

    // Weights + bias
    MakeTestInput(builder, "W", weight);
    MakeTestInput(builder, "B", bias);

    // Gemm
    builder.AddNode("gemm", "Gemm", {"reshaped", "W", "B"}, {"Y"});

    builder.MakeOutput("Y");
  };
}

void RunReshapeGemmTest(const TestInputDef<float>& input, const TestInputDef<int64_t>& shape,
                        const TestInputDef<float>& weight, const TestInputDef<float>& bias,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        const std::string& backend_name = "cpu", float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  auto build_fn = BuildReshapeGemmTestCase(input, shape, weight, bias);
  RunQnnModelTest(build_fn,
                  provider_options,
                  18,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}

}  // namespace

TEST_F(QnnCPUBackendTests, ReshapeGemmFusion) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<int64_t> shape_data = {4, 2};
  std::vector<float> weight_data(6, 1.0f);
  std::vector<float> bias_data = {1.0f, 2.0f, 3.0f};
// GCC 13 with -O2 inlines this call chain deeply enough that its data flow analyzer loses track of
// std::variant's initialization state inside the copy constructor (variant:224), triggering a false
// positive -Wmaybe-uninitialized. The warning is suppressed here because TestInputDef members are
// properly initialized in all constructors; this is a known GCC 13 analysis limitation with
// std::variant + lambda capture + deep inlining.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  RunReshapeGemmTest(TestInputDef<float>({2, 2, 2}, false, input_data), TestInputDef<int64_t>({2}, true, shape_data),
                     TestInputDef<float>({2, 3}, true, weight_data), TestInputDef<float>({3}, true, bias_data),
                     ExpectedEPNodeAssignment::All);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that builds a model with a QDQ Gemm node.
template <typename InputAQType, typename InputBQType>
inline GetTestQDQModelFn<InputAQType> BuildQDQGemmTestCase(const std::vector<TestInputDef<float>>& input_defs,
                                                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                           bool use_contrib_qdq = false) {
  return [input_defs, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                              std::vector<QuantParams<InputAQType>>& output_qparams) {
    const size_t num_inputs = input_defs.size();
    QNN_ASSERT(num_inputs == 2 || num_inputs == 3);

    builder.graph_->set_name("qdq_gemm_graph");

    // A (fp32) -> Q -> DQ
    MakeTestInput(builder, "A", input_defs[0]);
    QuantParams<InputAQType> a_qparams = GetTestInputQuantParams<InputAQType>(input_defs[0]);
    const std::string a_qdq = AddQDQNodePair<InputAQType>(
        builder, "qdq_a", "A", a_qparams.scale, a_qparams.zero_point, use_contrib_qdq);

    // B (fp32) -> Q -> DQ
    MakeTestInput(builder, "B", input_defs[1]);
    QuantParams<InputBQType> b_qparams = GetTestInputQuantParams<InputBQType>(input_defs[1]);
    const std::string b_qdq = AddQDQNodePair<InputBQType>(
        builder, "qdq_b", "B", b_qparams.scale, b_qparams.zero_point, use_contrib_qdq);

    std::vector<std::string> gemm_inputs;
    gemm_inputs.reserve(num_inputs);
    gemm_inputs.push_back(a_qdq);
    gemm_inputs.push_back(b_qdq);

    // Bias (optional): int32 -> DQ
    if (num_inputs == 3) {
      const std::string bias_dq = MakeTestQDQBiasInput(
          builder, "C", input_defs[2], a_qparams.scale * b_qparams.scale, use_contrib_qdq);
      gemm_inputs.push_back(bias_dq);
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes = attrs;
    builder.AddNode("gemm", "Gemm", gemm_inputs, {"Y"}, "", attributes);

    // Output: Y -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<InputAQType>(
        builder, "qdq_out", "Y", output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Runs a QDQ Gemm model on the QNN (HTP) EP and the ORT CPU EP. Checks the graph node assignment and that inference
// running the QDQ model on QNN EP is at least as accurate as on ORT CPU EP (compared to the baseline float32 model).
template <typename InputAQType, typename InputBQType>
static void RunQDQGemmTestOnHTP(const std::vector<TestInputDef<float>>& input_defs,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 13,
                                bool use_contrib_qdq = false,
                                QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildOpTestCase<float>("Gemm_node", "Gemm", input_defs, {}, attrs);
  auto qdq_model_builder = BuildQDQGemmTestCase<InputAQType, InputBQType>(input_defs, attrs, use_contrib_qdq);
  TestQDQModelAccuracy<InputAQType>(f32_model_builder,
                                    qdq_model_builder,
                                    provider_options,
                                    opset,
                                    expected_ep_assignment,
                                    tolerance);
}

// Test 8-bit QDQ Gemm with dynamic inputs A and Bias. The B input is an initializer.
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_Static_B_Dynamic_Bias_U8) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                         TestInputDef<float>({6, 4}, true, input_b_data),
                                         TestInputDef<float>({1, 4}, false, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All);
}

#ifndef __linux__
// Test 16-bit QDQ Gemm with dynamic inputs A and Bias. The B input is an initializer.
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_Dynamic_B_Dynamic_Bias_U16) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint16_t, uint16_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                           TestInputDef<float>({6, 4}, false, input_b_data),
                                           TestInputDef<float>({1, 4}, false, input_c_data)},
                                          {},
                                          ExpectedEPNodeAssignment::All,
                                          13,     // opset
                                          true);  // Use com.microsoft Q/DQ ops
}
#endif

// Test broadcasting of bias input. All inputs are dynamic.
TEST_F(QnnHTPBackendTests, Gemm_Broadcast_Bias_DynamicInputs) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // All dynamic inputs
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({2, 4}, false, input_a_data),
                                         TestInputDef<float>({4, 3}, false, input_b_data),
                                         TestInputDef<float>({3}, false, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All,
                                        13,
                                        false,
                                        QDQTolerance(0.00410f));
}

TEST_F(QnnHTPBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_DynamicC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // Dynamic A, static B, dynamic C
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({2, 4}, false, input_a_data),
                                         TestInputDef<float>({4, 3}, true, input_b_data),
                                         TestInputDef<float>({3}, false, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All,
                                        13,
                                        false,
                                        QDQTolerance(0.00410f));
}

TEST_F(QnnHTPBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_StaticC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};
  // Expected output (2,3):
  // 11.0f, 12.0f, 13.0f,
  // -9.0f, -8.0f, -7.0f

  // Dynamic A, static B, static C
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({2, 4}, false, input_a_data),
                                         TestInputDef<float>({4, 3}, true, input_b_data),
                                         TestInputDef<float>({3}, true, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All,
                                        13,
                                        false,
                                        QDQTolerance(0.00410f));
}

// Test 16-bit QDQ Gemm with dynamic inputs A and Bias. The B input is an initializer.
// TODO: Inaccuracy detected for output 'output_0', element 0.
// Output quant params: scale=0.001872879103757441, zero_point=0.
// Expected val: 120.73912048339844
// QNN QDQ val: 0 (err 120.73912048339844)
// CPU QDQ val: 120.73889923095703 (err 0.00022125244140625)
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_Static_B_Dynamic_Bias_U16) {
  QNN_SKIP_TEST_ON_LINUX("Output value mismatch with QNN SDK 2.31");
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint16_t, uint16_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                           TestInputDef<float>({6, 4}, true, input_b_data),
                                           TestInputDef<float>({1, 4}, false, input_c_data)},
                                          {},
                                          ExpectedEPNodeAssignment::All,
                                          13,     // opset
                                          true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ Gemm (16bit act, 8bit weight) with dynamic inputs A and Bias. The B input is an initializer.
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_Static_B_Dynamic_Bias_U16Act_U8Weight) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint16_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                          TestInputDef<float>({6, 4}, true, input_b_data),
                                          TestInputDef<float>({1, 4}, false, input_c_data)},
                                         {},
                                         ExpectedEPNodeAssignment::All,
                                         13,     // opset
                                         true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ Gemm with dynamic A and B inputs. The Bias is static.
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.48132994771003723, zero_point=0.
// Expected val: 120.73912048339844
// QNN QDQ val: 77.012794494628906 (err 43.726325988769531)
// CPU QDQ val: 119.85115814208984 (err 0.88796234130859375)
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, Gemm_Dynamic_A_B_Static_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                         TestInputDef<float>({6, 4}, false, input_b_data),  // Dynamic => inaccuracy
                                         TestInputDef<float>({1, 4}, true, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Gemm with static B and Bias inputs.
TEST_F(QnnHTPBackendTests, Gemm_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                         TestInputDef<float>({6, 4}, true, input_b_data),
                                         TestInputDef<float>({1, 4}, true, input_c_data)},
                                        {},
                                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Gemm with beta=0.0: bias is present but must be ignored.
TEST_F(QnnHTPBackendTests, Gemm_ZeroBeta_Static_B_And_Bias_U8) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({1, 6}, false, input_a_data),
                                         TestInputDef<float>({6, 4}, true, input_b_data),
                                         TestInputDef<float>({1, 4}, true, input_c_data)},
                                        {test::MakeAttribute("beta", 0.0f)},
                                        ExpectedEPNodeAssignment::All);
}

// Broken on v79 and v81 devices:
// Inaccuracy detected for output 'output_0', element 0
// output_range=31.434787750244141, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): 29.434776306152344
// qdq@QNN_EP val: 28.229671478271484 (err: 1.2051048278808594, err/output_range: 3.8336660861968994%)
// qdq@CPU_EP val: 29.092588424682617 (err: 0.34218788146972656, err/output_range: 1.0885642766952515%)
// abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = 2.7451016902923584%
// Test 8-bit QDQ Gemm with transposed A/B and static B and Bias inputs.
TEST_F(QnnHTPBackendTests, Gemm_TransAB_Static_B_And_Bias_U8) {
  QNN_SKIP_TEST_ON_ARM64("QDQ accuracy below tolerance on v79 and v81 devices");
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({6, 1}, false, input_a_data),
                                         TestInputDef<float>({4, 6}, true, input_b_data),
                                         TestInputDef<float>({1, 4}, true, input_c_data)},
                                        {test::MakeAttribute("transA", static_cast<int64_t>(1)),
                                         test::MakeAttribute("transB", static_cast<int64_t>(1))},
                                        ExpectedEPNodeAssignment::All);
}

// Test QDQ Gemm (16bit activation, 8bit weight) with transposed A/B and static B and Bias inputs.
TEST_F(QnnHTPBackendTests, Gemm_TransAB_Static_B_And_Bias_U16Act_U8Weight) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint16_t, uint8_t>({TestInputDef<float>({6, 1}, false, input_a_data),
                                          TestInputDef<float>({4, 6}, true, input_b_data),
                                          TestInputDef<float>({1, 4}, true, input_c_data)},
                                         {test::MakeAttribute("transA", static_cast<int64_t>(1)),
                                          test::MakeAttribute("transB", static_cast<int64_t>(1))},
                                         ExpectedEPNodeAssignment::All,
                                         13,     // opset
                                         true);  // Use com.microsoft Q/DQ ops
}

// Broken on v79 and v81 devices:
// Inaccuracy detected for output 'output_0', element 0
// output_range=31.434787750244141, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): 29.434776306152344
// qdq@QNN_EP val: 28.229671478271484 (err: 1.2051048278808594, err/output_range: 3.8336660861968994%)
// qdq@CPU_EP val: 29.092588424682617 (err: 0.34218788146972656, err/output_range: 1.0885642766952515%)
// abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = 2.7451016902923584%
// Test QDQ Gemm with transposed A/B and dynamic (i.e., not initializer) B and Bias inputs.
TEST_F(QnnHTPBackendTests, Gemm_TransAB_Dynamic_B_And_Bias) {
  QNN_SKIP_TEST_ON_ARM64("QDQ accuracy below tolerance on v79 and v81 devices");
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunQDQGemmTestOnHTP<uint8_t, uint8_t>({TestInputDef<float>({6, 1}, false, input_a_data),
                                         TestInputDef<float>({4, 6}, false, input_b_data),
                                         TestInputDef<float>({1, 4}, false, input_c_data)},
                                        {test::MakeAttribute("transA", static_cast<int64_t>(1)),
                                         test::MakeAttribute("transB", static_cast<int64_t>(1))},
                                        ExpectedEPNodeAssignment::All);
}

// Reproduces the CLIP text projection averaging pattern where ORT's MatMulAddFusion
// creates Gemm nodes with intermediate (NATIVE) bias:
//   A1 -> MatMul(W) -> mm1  (stays as MatMul)
//   A2 -> Gemm(W, C=mm1)   -> add1  (C is NATIVE)
//   A3 -> Gemm(W, C=add1)  -> add2  (C is NATIVE)
//   A4 -> Gemm(W, C=add2)  -> add3  (C is NATIVE)
namespace {
GetTestModelFn BuildGemmFromMatMulAddTestCase(int64_t K, int64_t N) {
  return [K, N](ModelTestBuilder& builder) {
    constexpr int64_t batch = 1;
    const std::vector<int64_t> input_shape = {batch, K};
    const std::vector<int64_t> weight_shape = {K, N};

    // 4 dynamic inputs
    builder.MakeInput<float>("A1", input_shape, -1.0f, 1.0f);
    builder.MakeInput<float>("A2", input_shape, -1.0f, 1.0f);
    builder.MakeInput<float>("A3", input_shape, -1.0f, 1.0f);
    builder.MakeInput<float>("A4", input_shape, -1.0f, 1.0f);

    // Shared static weight
    builder.MakeInitializer<float>("W", weight_shape, -1.0f, 1.0f);

    // 4 MatMul nodes
    builder.AddNode("matmul_1", "MatMul", {"A1", "W"}, {"mm1"});
    builder.AddNode("matmul_2", "MatMul", {"A2", "W"}, {"mm2"});
    builder.AddNode("matmul_3", "MatMul", {"A3", "W"}, {"mm3"});
    builder.AddNode("matmul_4", "MatMul", {"A4", "W"}, {"mm4"});

    // Chain of Adds: add1 = mm1 + mm2, add2 = add1 + mm3, add3 = add2 + mm4
    builder.AddNode("add_1", "Add", {"mm1", "mm2"}, {"add1"});
    builder.AddNode("add_2", "Add", {"add1", "mm3"}, {"add2"});
    builder.AddNode("add_3", "Add", {"add2", "mm4"}, {"add3"});

    builder.MakeOutput("add3");
  };
}

GetTestModelFn BuildGemmFromMatMulAddColumnBroadcastTestCase(int64_t M, int64_t K, int64_t N) {
  return [M, K, N](ModelTestBuilder& builder) {
    builder.MakeInput<float>("A", {M, K}, -1.0f, 1.0f);
    builder.MakeInput<float>("C_A", {M, K}, -1.0f, 1.0f);
    builder.MakeInitializer<float>("W", {K, N}, -1.0f, 1.0f);
    builder.MakeInitializer<float>("C_W", {K, 1}, -1.0f, 1.0f);

    builder.AddNode("column_matmul", "MatMul", {"C_A", "C_W"}, {"C"});
    builder.AddNode("matmul", "MatMul", {"A", "W"}, {"matmul_out"});
    builder.AddNode("add", "Add", {"matmul_out", "C"}, {"Y"});
    builder.MakeOutput("Y");
  };
}

void RunGemmFromMatMulAddColumnBroadcastTest(const std::string& backend_name, float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildGemmFromMatMulAddColumnBroadcastTestCase(/*M=*/2, /*K=*/4, /*N=*/3),
                  provider_options,
                  /*opset=*/18,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}
}  // namespace

TEST_F(QnnHTPBackendTests, GemmFromMatMulAddNonStaticBias) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildGemmFromMatMulAddTestCase(/*K=*/4, /*N=*/3),
                  provider_options,
                  /*opset=*/18,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

TEST_F(QnnCPUBackendTests, GemmFromMatMulAddNonStaticBias) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildGemmFromMatMulAddTestCase(/*K=*/4, /*N=*/3),
                  provider_options,
                  /*opset=*/18,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

TEST_F(QnnHTPBackendTests, GemmFromMatMulAddColumnBroadcastBias) {
  RunGemmFromMatMulAddColumnBroadcastTest("htp", 2e-3f);
}

TEST_F(QnnCPUBackendTests, GemmFromMatMulAddColumnBroadcastBias) {
  RunGemmFromMatMulAddColumnBroadcastTest("cpu");
}

namespace {

// Builds an ONNX QDQ graph for a Gemm with a block-quantized (BW_FLOAT_BLOCK) weight.
//   - activation A: float → Q(uint16) → DQ, shape [M, K]
//   - weight B: INT4/INT8 (or UINT4/UINT8) initializer + DQ with block_size attribute and a
//               rank-2 scale (blocked on K axis); axis/scale shape depend on transB.
//     transB=0: B=[K,N], scale=[K/block_size,N], axis=0.
//     transB=1: B=[N,K], scale=[N,K/block_size], axis=1.
//   - optional bias C: INT32 quantized (per-tensor), shape [N].
//   - output: Gemm → Q(uint16) → DQ → graph output, shape [M, N].
GetQDQTestCaseFn BuildBQGemmTestCase(int64_t M, int64_t K, int64_t N, int64_t block_size,
                                     int64_t trans_b = 0, bool include_bias = false,
                                     int weight_bits = 4, bool weight_is_unsigned = false,
                                     int64_t trans_a = 0) {
  return [M, K, N, block_size, trans_b, include_bias, weight_bits,
          weight_is_unsigned, trans_a](ModelTestBuilder& builder) -> void {
    const int64_t num_blocks = K / block_size;  // caller ensures K % block_size == 0

    // ── Activation A: float → Q(uint16) → DQ ─────────────────────────────────
    // transA=0: A=[M,K]; transA=1: A=[K,M].
    const std::vector<int64_t> act_shape = trans_a == 0 ? std::vector<int64_t>{M, K}
                                                        : std::vector<int64_t>{K, M};
    auto input_def = TestInputDef<float>(act_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);
    const float act_scale = 2.0f / 65534.0f;
    const uint16_t act_zp = 32767;
    const std::string act_dql_out = AddQDQNodePair<uint16_t>(builder, "act", "input", act_scale, act_zp);

    // ── Weight B initializer + DQ(block_size) ─────────────────────────────────
    // transB=0: B=[K,N], scale=[K/bs, N], axis=0.
    // transB=1: B=[N,K], scale=[N, K/bs], axis=1.
    const std::vector<int64_t> weight_shape = trans_b == 0 ? std::vector<int64_t>{K, N}
                                                           : std::vector<int64_t>{N, K};
    const std::vector<int64_t> scale_shape = trans_b == 0 ? std::vector<int64_t>{num_blocks, N}
                                                          : std::vector<int64_t>{N, num_blocks};
    const int64_t block_axis = trans_b == 0 ? 0 : 1;
    builder.MakeInitializer<float>("weight_scale", scale_shape, 0.01f, 0.05f);

    const size_t num_elems = static_cast<size_t>(K * N);
    if (weight_bits == 4 && !weight_is_unsigned) {
      std::vector<Int4x2> wd(Int4x2::CalcNumInt4Pairs(num_elems));
      for (size_t i = 0; i < num_elems; ++i) wd[i >> 1].SetElem(i & 1, static_cast<int8_t>((i % 7) - 3));
      builder.MakeInitializer<Int4x2>("weight_quant", weight_shape, wd);
    } else if (weight_bits == 4 && weight_is_unsigned) {
      std::vector<UInt4x2> wd(UInt4x2::CalcNumInt4Pairs(num_elems));
      for (size_t i = 0; i < num_elems; ++i) wd[i >> 1].SetElem(i & 1, static_cast<uint8_t>(i % 15));
      builder.MakeInitializer<UInt4x2>("weight_quant", weight_shape, wd);
    } else if (weight_is_unsigned) {
      std::vector<uint8_t> wd(num_elems);
      for (size_t i = 0; i < num_elems; ++i) wd[i] = static_cast<uint8_t>(i % 127);
      builder.MakeInitializer<uint8_t>("weight_quant", weight_shape, wd);
    } else {
      std::vector<int8_t> wd(num_elems);
      for (size_t i = 0; i < num_elems; ++i) wd[i] = static_cast<int8_t>((i % 127) - 63);
      builder.MakeInitializer<int8_t>("weight_quant", weight_shape, wd);
    }
    builder.AddNode("weight_dql", "DequantizeLinear",
                    {"weight_quant", "weight_scale"}, {"weight_dql_out"}, "",
                    {builder.MakeScalarAttribute("axis", block_axis),
                     builder.MakeScalarAttribute("block_size", block_size)});

    // ── Gemm ─────────────────────────────────────────────────────────────────
    std::vector<std::string> gemm_inputs = {act_dql_out, "weight_dql_out"};
    std::vector<ONNX_NAMESPACE::AttributeProto> gemm_attrs;
    gemm_attrs.push_back(builder.MakeScalarAttribute("transB", trans_b));
    if (trans_a != 0) {
      gemm_attrs.push_back(builder.MakeScalarAttribute("transA", trans_a));
    }
    if (include_bias) {
      // INT32-quantized bias (per-tensor scale). Matches Conv BQ bias pattern.
      const float bias_scale = act_scale * 0.03f;
      builder.MakeScalarInitializer<float>("bias_scale", bias_scale);
      builder.MakeScalarInitializer<int32_t>("bias_zp", 0);
      builder.Make1DInitializer<int32_t>("bias_quant", std::vector<int32_t>(static_cast<size_t>(N), 0));
      builder.AddNode("bias_dql", "DequantizeLinear",
                      {"bias_quant", "bias_scale", "bias_zp"}, {"bias_dql_out"});
      gemm_inputs.push_back("bias_dql_out");
    }
    builder.AddNode("gemm", "Gemm", gemm_inputs, {"gemm_out"}, kOnnxDomain, gemm_attrs);

    // ── Output: Gemm → Q(uint16) → DQ → graph output ─────────────────────────
    const float out_scale = 4.0f / 65534.0f;
    const uint16_t out_zp = 32767;
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "out", "gemm_out", out_scale, out_zp);
  };
}

ProviderOptions GetBQGemmProviderOptions() {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return opts;
}

}  // namespace

// INT4 weight transB=0, [K,N]=[16,4], block_size=8, no bias.
TEST_F(QnnHTPBackendTests, GemmBQ_U16Int4_TransB0_NoBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8, /*transB=*/0),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT4 weight transB=1, [N,K]=[4,16], block_size=8, no bias.
TEST_F(QnnHTPBackendTests, GemmBQ_U16Int4_TransB1_NoBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8, /*transB=*/1),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// transA=1: ONNX activation is [K, M]; QNN EP inserts a Transpose to [M, K] before the FC.
TEST_F(QnnHTPBackendTests, GemmBQ_U16Int4_TransA1_TransB0) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8, /*transB=*/0,
                                      /*include_bias=*/false, /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false, /*transA=*/1),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// transA=1 with transB=1: both A and B transposed.
TEST_F(QnnHTPBackendTests, GemmBQ_U16Int4_TransA1_TransB1) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8, /*transB=*/1,
                                      /*include_bias=*/false, /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false, /*transA=*/1),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT4 transB=0, larger K with multiple blocks. Guards scale reordering.
TEST_F(QnnHTPBackendTests, GemmBQ_U16Int4_TransB0_MultiBlock) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/32, /*N=*/8, /*block_size=*/8, /*transB=*/0),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT4 transB=0 with INT32-quantized bias.
TEST_F(QnnHTPBackendTests, GemmBQ_U16Int4_TransB0_WithBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8, /*transB=*/0,
                                      /*include_bias=*/true),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT8, block_size=4, transB=0.
TEST_F(QnnHTPBackendTests, GemmBQ_U16Int8_TransB0_BlockSize4) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/4, /*transB=*/0,
                                      /*include_bias=*/false, /*weight_bits=*/8),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// UINT4 transB=0: exercises unsigned→signed conversion.
TEST_F(QnnHTPBackendTests, GemmBQ_U16UInt4_TransB0_NoBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8, /*transB=*/0,
                                      /*include_bias=*/false, /*weight_bits=*/4, /*weight_is_unsigned=*/true),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// INT2 DISABLED — CPU lacks 2-bit Q/DQ; HTP 2-bit BQ requires QAIRT >= 2.47 (float MatMul/FC kernel).
TEST_F(QnnHTPBackendTests, DISABLED_GemmBQ_U16Int2_TransB0_BlockSize16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQGemmTestCase(/*M=*/2, /*K=*/32, /*N=*/4, /*block_size=*/16, /*transB=*/0,
                                      /*include_bias=*/false, /*weight_bits=*/2),
                  GetBQGemmProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// MatMulAddFusion (Gemm sandwiched by Reshapes) supergroup tests. The QDQ selector absorbs
// the trailing Reshape (and Relu when Q's encoding is bounded to [0, +inf)) into the Gemm
// unit; the op-builder emits FC (rank-2, encoded) + QNN Reshape (rank-N, same encoding).
namespace {

//   input (rank-3, float) -> Q(ActQType) -> DQ ─┐
//                                               ├─ MatMul -> Add(DQ'd int32 bias) -> [Relu] -> Q -> DQ -> output
//   weight (rank-2, float, static) -> Q(WtQType) -> DQ ┘
// The activation is rank-3 so MatMulAddFusion inserts the Reshape wrappers around
// the resulting rank-2 Gemm. The output Q is unsigned with zero_point == 0, so
// encoding_min = scale * (qmin - zp) = 0 and the Relu fold is encoding-safe.
template <typename ActQType, typename WtQType>
GetTestModelFn BuildMatMulAddFusionQDQTestCase(int64_t M, int64_t K, int64_t N, bool include_relu) {
  return [M, K, N, include_relu](ModelTestBuilder& builder) {
    const std::vector<int64_t> act_shape{1, M, K};
    const std::vector<int64_t> weight_shape{K, N};
    const std::vector<int64_t> bias_shape{N};

    MakeTestInput<float>(builder, "input", TestInputDef<float>(act_shape, false, -1.0f, 1.0f));
    QuantParams<ActQType> act_qp = QuantParams<ActQType>::Compute(-1.0f, 1.0f, /*symmetric=*/false);
    const std::string act_qdq = AddQDQNodePair<ActQType>(builder, "act", "input", act_qp.scale,
                                                         act_qp.zero_point, /*use_contrib_qdq=*/true);

    // Static weight -> DQ.
    QuantParams<WtQType> wt_qp = QuantParams<WtQType>::Compute(-0.5f, 0.5f, /*symmetric=*/true);
    TestInputDef<float> wt_def(weight_shape, /*is_initializer=*/true,
                               GetFloatDataInRange(-0.5f, 0.5f, static_cast<size_t>(K * N)));
    std::vector<WtQType> wt_quantized(static_cast<size_t>(K * N));
    const std::vector<float> wt_scales{wt_qp.scale};
    const std::vector<WtQType> wt_zps{wt_qp.zero_point};
    QuantizeValues<float, WtQType>(wt_def.GetRawData(), wt_quantized, weight_shape,
                                   wt_scales, wt_zps, std::nullopt);
    builder.MakeInitializer<WtQType>("weight_q", weight_shape, wt_quantized);
    builder.AddDequantizeLinearNode<WtQType>("weight_dq", "weight_q", wt_qp.scale,
                                             wt_qp.zero_point, "weight_dq", /*use_contrib_qdq=*/true);

    // MatMul.
    builder.AddNode("MatMul", "MatMul", {act_qdq, "weight_dq"}, {"mm_out"}, kOnnxDomain);

    // DQ'd int32 bias with bias_scale = act_scale * weight_scale.
    TestInputDef<float> bias_def(bias_shape, /*is_initializer=*/true,
                                 GetFloatDataInRange(-0.2f, 0.2f, static_cast<size_t>(N)));
    const std::string bias_dq = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                     act_qp.scale * wt_qp.scale, /*use_contrib_qdq=*/true);
    builder.AddNode("Add", "Add", {"mm_out", bias_dq}, {"add_out"}, kOnnxDomain);

    // Optional Relu.
    const std::string post_activation = include_relu ? "relu_out" : "add_out";
    if (include_relu) {
      builder.AddNode("Relu", "Relu", {"add_out"}, {"relu_out"});
    }

    // Output Q(zp=0) makes the encoding safe for the Relu fold; ORT normally drops
    // Relu itself in this case, but we assert that even if the Relu survives (e.g.,
    // because L2 cleanup decides to keep it), the group forms and QNN accepts the FC.
    QuantParams<ActQType> out_qp = QuantParams<ActQType>::Compute(0.0f, 8.0f, /*symmetric=*/false);
    AddQDQNodePairWithOutputAsGraphOutput<ActQType>(builder, "out", post_activation,
                                                    out_qp.scale, out_qp.zero_point,
                                                    /*use_contrib_qdq=*/true);
  };
}

ProviderOptions GetMatMulAddFusionProviderOptions() {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  // On x86_64 Linux, the default HTP validator is v68 which lacks 16-bit-weight FC support.
  // Pin to SM8550 (v73) to validate 16-bit-weight paths without a real device.
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8550);
#endif
  return opts;
}

}  // namespace

// U16 activation, U8 weight, with Relu
TEST_F(QnnHTPBackendTests, GemmMatMulAddFusion_U16Act_U8Weight_WithRelu) {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  RunQnnModelTest(BuildMatMulAddFusionQDQTestCase<uint16_t, uint8_t>(
                      /*M=*/8, /*K=*/16, /*N=*/12, /*include_relu=*/true),
                  opts, /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// U16 activation, S16 weight, with Relu.
// On x86 Linux the v68 default validator rejects 16-bit-weight FC (QNN error 3110); the
// provider options above pin to SM8550 (v73) to validate.
TEST_F(QnnHTPBackendTests, GemmMatMulAddFusion_U16Act_S16Weight_WithRelu) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildMatMulAddFusionQDQTestCase<uint16_t, int16_t>(
                      /*M=*/8, /*K=*/16, /*N=*/12, /*include_relu=*/true),
                  GetMatMulAddFusionProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// Direct QDQ Gemm -> Reshape -> Q graphs (no MatMul+Add) — exercise the absorb-Reshape
// selector's guard-rails on shapes; MatMulAddFusion never emits (transB=1, NATIVE bias).
namespace {
// activation: rank-2 [M, K] float -> Q(u16) -> DQ
// weight:    rank-2 [K, N] (or [N, K] when trans_b=1) static -> DQ
// bias (opt): static INT32 -> DQ; rank-1 [N] by default
// -> Gemm -> Reshape (rank-3 [1, M, N]) -> Q -> DQ -> output
struct DirectGemmReshapeQConfig {
  int64_t M = 4;
  int64_t K = 8;
  int64_t N = 6;
  int64_t trans_b = 0;                  // 0 → weight [K,N]; 1 → weight [N,K]
  bool include_bias = true;             // rank-1 bias by default
  bool bias_from_intermediate = false;  // if true, bias is produced by an intermediate MatMul (NATIVE bias)
  std::optional<std::vector<int64_t>> bias_shape;
};

GetTestModelFn BuildDirectGemmReshapeQTestCase(const DirectGemmReshapeQConfig& cfg) {
  return [cfg](ModelTestBuilder& builder) {
    const std::vector<int64_t> act_shape{cfg.M, cfg.K};
    const std::vector<int64_t> weight_shape = cfg.trans_b == 0
                                                  ? std::vector<int64_t>{cfg.K, cfg.N}
                                                  : std::vector<int64_t>{cfg.N, cfg.K};

    MakeTestInput<float>(builder, "input", TestInputDef<float>(act_shape, false, -1.0f, 1.0f));
    QuantParams<uint16_t> act_qp = QuantParams<uint16_t>::Compute(-1.0f, 1.0f, /*symmetric=*/false);
    const std::string act_qdq = AddQDQNodePair<uint16_t>(builder, "act", "input", act_qp.scale,
                                                         act_qp.zero_point, /*use_contrib_qdq=*/true);

    QuantParams<uint8_t> wt_qp = QuantParams<uint8_t>::Compute(-0.5f, 0.5f, /*symmetric=*/true);
    TestInputDef<float> wt_def(weight_shape, /*is_initializer=*/true,
                               GetFloatDataInRange(-0.5f, 0.5f, static_cast<size_t>(cfg.K * cfg.N)));
    std::vector<uint8_t> wt_quantized(static_cast<size_t>(cfg.K * cfg.N));
    const std::vector<float> wt_scales{wt_qp.scale};
    const std::vector<uint8_t> wt_zps{wt_qp.zero_point};
    QuantizeValues<float, uint8_t>(wt_def.GetRawData(), wt_quantized, weight_shape,
                                   wt_scales, wt_zps, std::nullopt);
    builder.MakeInitializer<uint8_t>("weight_q", weight_shape, wt_quantized);
    builder.AddDequantizeLinearNode<uint8_t>("weight_dq", "weight_q", wt_qp.scale,
                                             wt_qp.zero_point, "weight_dq", /*use_contrib_qdq=*/true);

    std::vector<std::string> gemm_inputs{act_qdq, "weight_dq"};
    if (cfg.include_bias) {
      if (cfg.bias_from_intermediate) {
        // Build a NATIVE bias: MatMul produces the tensor consumed by Gemm as C. This is what
        // MatMulAddFusion normally rewrites into Gemm(C=intermediate), but here we assemble
        // the pattern by hand so the absorb-Reshape selector must reject it.
        std::vector<int64_t> mm_a_shape{cfg.M, cfg.K};
        std::vector<int64_t> mm_w_shape{cfg.K, cfg.N};
        builder.MakeInput<float>("bias_mm_a", mm_a_shape, -1.0f, 1.0f);
        builder.MakeInitializer<float>("bias_mm_w", mm_w_shape, -0.5f, 0.5f);
        const std::string bias_mm_dq_a = AddQDQNodePair<uint16_t>(builder, "bias_mm_a_qdq",
                                                                  "bias_mm_a", act_qp.scale,
                                                                  act_qp.zero_point, /*use_contrib_qdq=*/true);
        builder.AddNode("bias_mm", "MatMul", {bias_mm_dq_a, "bias_mm_w"}, {"bias_native"}, kOnnxDomain);
        gemm_inputs.push_back("bias_native");
      } else {
        const std::vector<int64_t> bias_shape = cfg.bias_shape.value_or(std::vector<int64_t>{cfg.N});
        TestInputDef<float> bias_def(bias_shape, /*is_initializer=*/true,
                                     GetFloatDataInRange(-0.2f, 0.2f, SizeOfShape(bias_shape)));
        const std::string bias_dq = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                         act_qp.scale * wt_qp.scale, /*use_contrib_qdq=*/true);
        gemm_inputs.push_back(bias_dq);
      }
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> gemm_attrs;
    gemm_attrs.push_back(test::MakeAttribute("transB", cfg.trans_b));
    builder.AddNode("gemm", "Gemm", gemm_inputs, {"gemm_out"}, kOnnxDomain, gemm_attrs);

    // Explicit Reshape rank-2 -> rank-3.
    const std::vector<int64_t> reshape_target{1, cfg.M, cfg.N};
    builder.Make1DInitializer<int64_t>("reshape_shape", reshape_target);
    builder.AddNode("reshape", "Reshape", {"gemm_out", "reshape_shape"}, {"reshape_out"});

    QuantParams<uint16_t> out_qp = QuantParams<uint16_t>::Compute(0.0f, 8.0f, /*symmetric=*/false);
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "out", "reshape_out",
                                                    out_qp.scale, out_qp.zero_point,
                                                    /*use_contrib_qdq=*/true);
  };
}

ProviderOptions GetHtpProviderOptions() {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  return opts;
}
}  // namespace

// Positive: direct Gemm -> Reshape -> Q with transB=0 and rank-1 static bias.
// The selector absorbs the Reshape and the group is assigned to QNN EP.
TEST_F(QnnHTPBackendTests, GemmReshapeQ_Direct_TransB0_1DBias) {
  DirectGemmReshapeQConfig cfg;
  RunQnnModelTest(BuildDirectGemmReshapeQTestCase(cfg), GetHtpProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// Negative gate (M-1): direct Gemm(transB=1) -> Reshape -> Q must NOT be absorbed by the
// absorb-Reshape selector — the builder's absorbed path hard-asserts transB=0. The graph
// still runs on QNN EP via the regular Gemm path with a standalone Reshape.
TEST_F(QnnHTPBackendTests, GemmReshapeQ_Direct_TransB1_NotAbsorbed) {
  DirectGemmReshapeQConfig cfg;
  cfg.trans_b = 1;
  // Successful build proves the selector did not force the absorbed-Reshape path
  // (which would abort at graph build with transB=1).
  RunQnnModelTest(BuildDirectGemmReshapeQTestCase(cfg), GetHtpProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// Negative gate (M-2): direct Gemm with a NATIVE bias (produced by an intermediate MatMul)
// -> Reshape -> Q. The absorb-Reshape path must reject it (unsafe: NATIVE bias would have
// required a separate Add). The graph still runs through the regular Gemm path.
TEST_F(QnnHTPBackendTests, GemmReshapeQ_Direct_NativeBias_NotAbsorbed) {
  DirectGemmReshapeQConfig cfg;
  cfg.bias_from_intermediate = true;
  RunQnnModelTest(BuildDirectGemmReshapeQTestCase(cfg), GetHtpProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// C=[M,1] and C=[1,1] broadcast in ONNX but are not FullyConnected bias vectors. Keep the
// standalone Reshape so the builder emits FC + Add.
TEST_F(QnnHTPBackendTests, GemmReshapeQ_Direct_IncompatibleBiasBroadcast_NotAbsorbed) {
  DirectGemmReshapeQConfig column_bias;
  column_bias.bias_shape = std::vector<int64_t>{column_bias.M, 1};
  RunQnnModelTest(BuildDirectGemmReshapeQTestCase(column_bias), GetHtpProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(2e-2f)});

  DirectGemmReshapeQConfig unit_bias;
  unit_bias.bias_shape = std::vector<int64_t>{1, 1};
  RunQnnModelTest(BuildDirectGemmReshapeQTestCase(unit_bias), GetHtpProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::Some, ElementwiseAbsoluteVerifier(2e-2f)});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// Gemm basic default attributes.
// QNN's FullyConnected operator only supports `outputVector = ( inputAsVector * weightsMatrix ) + biasesVector`
// Input A's 0th dimension is interpreted as `batch_size`.
TEST_F(QnnGPUBackendTests, Gemm_Basic) {
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, -10.0f, 10.0f),
                      TestInputDef<float>({3, 4}, false, -10.0f, 10.0f)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Gemm with 'alpha' or 'beta' attributes is not supported by QNN EP.
TEST_F(QnnGPUBackendTests, Gemm_AlphaBetaUnsupported) {
  // Check that alpha != 1.0f is not supported.
  RunGemmTest<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -10.0f, 10.0f)},
                     {test::MakeAttribute("alpha", 1.5f)},
                     ExpectedEPNodeAssignment::None,  // Should not be assigned to QNN EP.
                     "gpu");

  // Check that beta != 1.0f is not supported.
  RunGemmTest<float>({TestInputDef<float>({1, 2}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -10.0f, 10.0f),
                      TestInputDef<float>({1, 4}, false, -1.0f, 1.0f)},
                     {test::MakeAttribute("beta", 1.2f)},
                     ExpectedEPNodeAssignment::None,  // Should not be assigned to QNN EP.
                     "gpu");
}

// Gemm with matrix bias ie 2D (M, N) is supported.
// When vector bias ie M == 1
// QNN's FullyConnected operator only supports `outputVector = ( inputAsVector * weightsMatrix ) + biasesVector`
// When 2D bias i.e. M != 1, N != 1.
// When 2D bias i.e. M != 1, N != 1.
// QNN's Gemm will be split in to FullyConnected and ElementwiseAdd.
TEST_F(QnnGPUBackendTests, Gemm_2D_Bias) {
  // 2D matrix mul with 2D bias is supported when Gemm is not a QDQ node.
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, -10.0f, 10.0f),
                      TestInputDef<float>({3, 4}, false, -10.0f, 10.0f),
                      TestInputDef<float>({2, 4}, false, -1.0f, 1.0f)},
                     {},
                     ExpectedEPNodeAssignment::All,  // Should be assigned to QNN EP.
                     "gpu");
}

// Gemm with vector bias is supported ie when M == 1.
// Bias is broadcast across input batches.
// `outputVector = ( inputAsVector * weightsMatrix ) + biasesVector`
TEST_F(QnnGPUBackendTests, Gemm_1DBiasBcast) {
  // 2D matrix mul with 1D bias supported.
  RunGemmTest<float>({TestInputDef<float>({2, 3}, false, -10.0f, 10.0f),
                      TestInputDef<float>({3, 4}, false, -10.0f, 10.0f),
                      TestInputDef<float>({1, 4}, false, -1.0f, 1.0f)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Test Gemm with dynamic (i.e., not initializer) inputs (A, B, Bias).
TEST_F(QnnGPUBackendTests, Gemm_Dynamic_A_B_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, false, input_b_data),
                      TestInputDef<float>({1, 4}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Test Gemm with static B and Bias inputs.
TEST_F(QnnGPUBackendTests, Gemm_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({1, 6}, false, input_a_data),
                      TestInputDef<float>({6, 4}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Test Gemm with transposed A/B and static B and Bias inputs.
TEST_F(QnnGPUBackendTests, Gemm_TransposeAB_Static_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                      TestInputDef<float>({4, 6}, true, input_b_data),
                      TestInputDef<float>({1, 4}, true, input_c_data)},
                     {test::MakeAttribute("transA", static_cast<int64_t>(1)),
                      test::MakeAttribute("transB", static_cast<int64_t>(1))},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Test Gemm with transposed A/B and dynamic (i.e., not initializer) B and Bias inputs.
TEST_F(QnnGPUBackendTests, Gemm_TransAB_Dynamic_B_And_Bias) {
  std::vector<float> input_a_data = GetFloatDataInRange(-10.0f, 10.0f, 6);
  std::vector<float> input_b_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  std::vector<float> input_c_data = GetFloatDataInRange(-1.0f, 1.0f, 4);
  RunGemmTest<float>({TestInputDef<float>({6, 1}, false, input_a_data),
                      TestInputDef<float>({4, 6}, false, input_b_data),
                      TestInputDef<float>({1, 4}, false, input_c_data)},
                     {test::MakeAttribute("transA", static_cast<int64_t>(1)),
                      test::MakeAttribute("transB", static_cast<int64_t>(1))},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Bias broadcast across batches.
TEST_F(QnnGPUBackendTests, Gemm_Broadcast_Bias_DynamicInputs) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};

  // All dynamic inputs
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, false, input_b_data),
                      TestInputDef<float>({3}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

TEST_F(QnnGPUBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_DynamicC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};

  // Dynamic A, static B, dynamic C
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, true, input_b_data),
                      TestInputDef<float>({3}, false, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

TEST_F(QnnGPUBackendTests, Gemm_Broadcast_Bias_DynamicA_StaticB_StaticC) {
  std::vector<float> input_a_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input_b_data(12, 1.0f);
  std::vector<float> input_c_data = {1.0f, 2.0f, 3.0f};

  // Dynamic A, static B, static C
  RunGemmTest<float>({TestInputDef<float>({2, 4}, false, input_a_data),
                      TestInputDef<float>({4, 3}, true, input_b_data),
                      TestInputDef<float>({3}, true, input_c_data)},
                     {},
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

// Tests fusion of Reshape inpout followed by Gemm.
TEST_F(QnnGPUBackendTests, ReshapeGemmFusion) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<int64_t> shape_data = {4, 2};
  std::vector<float> weight_data(6, 1.0f);
  std::vector<float> bias_data = {1.0f, 2.0f, 3.0f};
  RunReshapeGemmTest(TestInputDef<float>({2, 2, 2}, false, input_data), TestInputDef<int64_t>({2}, true, shape_data),
                     TestInputDef<float>({2, 3}, true, weight_data), TestInputDef<float>({3}, true, bias_data),
                     ExpectedEPNodeAssignment::All,
                     "gpu");
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
