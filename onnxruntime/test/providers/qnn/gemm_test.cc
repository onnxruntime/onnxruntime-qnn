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
