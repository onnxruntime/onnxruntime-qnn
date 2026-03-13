// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/constants.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a QuickGelu operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunQuickGeluTest(const TestInputDef<DataType>& input_def,
                             float alpha,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             const std::string& backend_name = "cpu",
                             float fp32_abs_err = 5e-3f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;

  if (backend_name == "htp") {
#if defined(_WIN32)
    if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
      GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
    }
#endif
#if defined(__linux__) && !defined(__aarch64__)
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  auto model_builder = [input_def, alpha](ModelTestBuilder& builder) {
    MakeTestInput<DataType>(builder, "input", input_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("alpha", alpha));

    builder.AddNode("quickgelu", "QuickGelu", {"input"}, {"output"}, kMSDomain, attributes);
    builder.MakeOutput("output");
  };

  RunQnnModelTest(model_builder,
                  provider_options,
                  13,  // opset version for contrib ops
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Tests the accuracy of a QDQ QuickGelu model on QNN EP by comparing to CPU EP.
template <typename QType>
static void RunQDQQuickGeluTest(const TestInputDef<float>& input_def,
                                float alpha,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                const std::string& backend_name = "htp",
                                bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestModelFn model_builder_fn = [input_def, alpha](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("alpha", alpha));

    builder.AddNode("quickgelu", "QuickGelu", {"input"}, {"output"}, kMSDomain, attributes);
    builder.MakeOutput("output");
  };

  GetTestQDQModelFn<QType> qdq_model_builder_fn = [input_def, alpha, use_contrib_qdq](ModelTestBuilder& builder, std::vector<QuantParams<QType>>& output_qparams) {
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<QType> input_qparams = GetTestInputQuantParams<QType>(input_def);
    std::string input_after_qdq = AddQDQNodePair<QType>(builder, "qdq_input", "input", input_qparams.scale,
                                                        input_qparams.zero_point, use_contrib_qdq);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("alpha", alpha));

    // QuickGelu -> op_output
    builder.AddNode("quickgelu", "QuickGelu", {input_after_qdq}, {"op_output"}, kMSDomain, attributes);

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QType>(builder, "qdq_output", "op_output", output_qparams[0].scale,
                                                 output_qparams[0].zero_point, use_contrib_qdq);
  };

  TestQDQModelAccuracy(model_builder_fn,
                       qdq_model_builder_fn,
                       provider_options,
                       13,  // opset version for contrib ops
                       expected_ep_assignment,
                       QDQTolerance(5e-3f));
}

//
// CPU tests:
//

// Test QuickGelu with default alpha value (1.0)
TEST_F(QnnCPUBackendTests, QuickGelu_Default_Alpha) {
  RunQuickGeluTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          1.0f,  // alpha
                          ExpectedEPNodeAssignment::All);
}

// Test QuickGelu with custom alpha value
TEST_F(QnnCPUBackendTests, QuickGelu_Custom_Alpha) {
  RunQuickGeluTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          1.702f,  // alpha
                          ExpectedEPNodeAssignment::All);
}

// Test QuickGelu with negative alpha value
TEST_F(QnnCPUBackendTests, QuickGelu_Negative_Alpha) {
  RunQuickGeluTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          -1.702f,  // alpha
                          ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, QuickGelu_Default_Alpha) {
  RunQuickGeluTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          1.0f,
                          ExpectedEPNodeAssignment::All,
                          "htp",
                          0.01f);
}

// Test QuickGelu with custom alpha value on HTP
TEST_F(QnnHTPBackendTests, QuickGelu_Custom_Alpha) {
  RunQuickGeluTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          1.702f,  // alpha
                          ExpectedEPNodeAssignment::All,
                          "htp");
}

// Test QuickGelu with negative alpha value on HTP
TEST_F(QnnHTPBackendTests, QuickGelu_Negative_Alpha) {
  RunQuickGeluTest<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                          -1.702f,  // alpha
                          ExpectedEPNodeAssignment::All,
                          "htp");
}

TEST_F(QnnHTPBackendTests, QuickGelu_Float16_Default_Alpha) {
  RunQuickGeluTest<Ort::Float16_t>(ConvertToFP16InputDef(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))),
                                   1.0f,
                                   ExpectedEPNodeAssignment::All,
                                   "htp",
                                   0.01f);
}

// Test QuickGelu with float16 inputs and custom alpha on HTP
TEST_F(QnnHTPBackendTests, QuickGelu_Float16_Custom_Alpha) {
  RunQuickGeluTest<Ort::Float16_t>(ConvertToFP16InputDef(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))),
                                   1.702f,  // alpha
                                   ExpectedEPNodeAssignment::All,
                                   "htp");
}

// Test QuickGelu with float16 inputs and negative alpha on HTP
TEST_F(QnnHTPBackendTests, QuickGelu_Float16_Negative_Alpha) {
  RunQuickGeluTest<Ort::Float16_t>(ConvertToFP16InputDef(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))),
                                   -1.702f,  // alpha
                                   ExpectedEPNodeAssignment::All,
                                   "htp");
}

// Test 8-bit QDQ QuickGelu with default alpha value on HTP
TEST_F(QnnHTPBackendTests, QuickGelu_QDQ_U8_Default_Alpha) {
  RunQDQQuickGeluTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               1.0f,  // alpha
                               ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ QuickGelu with custom alpha value on HTP
TEST_F(QnnHTPBackendTests, QuickGelu_QDQ_U8_Custom_Alpha) {
  RunQDQQuickGeluTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                               1.702f,  // alpha
                               ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ QuickGelu with default alpha value on HTP
TEST_F(QnnHTPBackendTests, QuickGelu_QDQ_U16_Default_Alpha) {
  RunQDQQuickGeluTest<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                1.0f,  // alpha
                                ExpectedEPNodeAssignment::All,
                                "htp",
                                true);  // Use com.microsoft Q/DQ ops
}

// Test 16-bit QDQ QuickGelu with custom alpha value on HTP
TEST_F(QnnHTPBackendTests, QuickGelu_QDQ_U16_Custom_Alpha) {
  RunQDQQuickGeluTest<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                1.702f,  // alpha
                                ExpectedEPNodeAssignment::All,
                                "htp",
                                true);  // Use com.microsoft Q/DQ ops
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
