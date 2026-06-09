// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs an AveragePool model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN and CPU match.
static void RunAveragePoolOpTest(const std::string& op_type,
                                 const std::vector<TestInputDef<float>>& input_defs,
                                 const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 const std::string& backend_name = "cpu", int opset = 18) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

// Runs a QDQ AveragePool model on the QNN HTP backend. Checks the graph node assignment, and that accuracy
// on QNN EP is at least as good as on CPU EP.
template <typename QuantType>
static void RunQDQAveragePoolOpTest(const std::string& op_type,
                                    const std::vector<TestInputDef<float>>& input_defs,
                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    int opset = 18,
                                    QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, attrs),
                       BuildQDQOpTestCase<QuantType>(op_type + "_node", op_type, input_defs, {}, attrs),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

//
// CPU tests:
//

// AveragePool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnCPUBackendTests, AveragePool_AsGlobal) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                        test::MakeAttribute("strides", std::vector<int64_t>{3, 3})},
                       ExpectedEPNodeAssignment::All);
}

// Test GlobalAveragePool on QNN CPU backend.
TEST_F(QnnCPUBackendTests, GlobalAveragePool) {
  RunAveragePoolOpTest("GlobalAveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {},
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that counts padding.
TEST_F(QnnCPUBackendTests, AveragePool_CountIncludePad) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                        test::MakeAttribute("count_include_pad", static_cast<int64_t>(1))},
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnCPUBackendTests, AveragePool_AutopadSameUpper) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                        test::MakeAttribute("count_include_pad", static_cast<int64_t>(1)),
                        test::MakeAttribute("auto_pad", "SAME_UPPER")},
                       ExpectedEPNodeAssignment::All);
}

// AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnCPUBackendTests, AveragePool_AutopadSameLower) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                        test::MakeAttribute("count_include_pad", static_cast<int64_t>(1)),
                        test::MakeAttribute("auto_pad", "SAME_LOWER")},
                       ExpectedEPNodeAssignment::All);
}

// AveragePool 3D as GlobalAveragePool.
TEST_F(QnnCPUBackendTests, AveragePool_3D_AsGlobal) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3, 3}, false, -10.0f, 10.0f)},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                        test::MakeAttribute("strides", std::vector<int64_t>{3, 3, 3})},
                       ExpectedEPNodeAssignment::All);
}

// GlobalAveragePool 3D.
TEST_F(QnnCPUBackendTests, GlobalAveragePool_3D) {
  RunAveragePoolOpTest("GlobalAveragePool",
                       {TestInputDef<float>({1, 2, 3, 3, 3}, false, -10.0f, 10.0f)},
                       {},
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, GlobalAveragePoolRank3) {
  RunAveragePoolOpTest("GlobalAveragePool",
                       {TestInputDef<float>({1, 8, 5}, false, -10.0f, 10.0f)},
                       {},
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, AveragePoolRank3) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 3, 5}, false, -10.0f, 10.0f)},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                        test::MakeAttribute("strides", std::vector<int64_t>{1}),
                        test::MakeAttribute("pads", std::vector<int64_t>{1, 1})},
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, AveragePoolRank3AutopadSameUpper) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 3, 4}, false, -10.0f, 10.0f)},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                        test::MakeAttribute("strides", std::vector<int64_t>{2}),
                        test::MakeAttribute("auto_pad", "SAME_UPPER")},
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, AveragePoolRank3AutopadSameLower) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 3, 4}, false, -10.0f, 10.0f)},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                        test::MakeAttribute("strides", std::vector<int64_t>{2}),
                        test::MakeAttribute("auto_pad", "SAME_LOWER")},
                       ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// QDQ AveragePool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnHTPBackendTests, AveragePool_AsGlobal) {
  std::vector<float> input = {32.1289f, -59.981f, -17.2799f, 62.7263f, 33.6205f, -19.3515f, -54.0113f, 37.5648f, 61.5357f,
                              -52.5769f, 27.3637f, -9.01382f, -65.5612f, 19.9497f, -47.9228f, 26.9813f, 83.064f, 0.362503f};
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                                    test::MakeAttribute("strides", std::vector<int64_t>{3, 3})},
                                   ExpectedEPNodeAssignment::All);
}

// Test accuracy for 8-bit QDQ GlobalAveragePool with input of rank 4.
TEST_F(QnnHTPBackendTests, GlobalAveragePool) {
  std::vector<float> input = GetFloatDataInRange(-32.0f, 32.0f, 18);

  RunQDQAveragePoolOpTest<uint8_t>("GlobalAveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {},
                                   ExpectedEPNodeAssignment::All);
}

// QDQ AveragePool that counts padding.
TEST_F(QnnHTPBackendTests, AveragePool_CountIncludePad_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                    test::MakeAttribute("count_include_pad", static_cast<int64_t>(1))},
                                   ExpectedEPNodeAssignment::All,
                                   18);
}

// QDQ AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnHTPBackendTests, AveragePool_AutopadSameUpper_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                    test::MakeAttribute("auto_pad", "SAME_UPPER")},
                                   ExpectedEPNodeAssignment::All,
                                   18);
}

// QDQ AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnHTPBackendTests, AveragePool_AutopadSameLower_HTP_u8) {
  std::vector<float> input = {-9.0f, -7.33f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f,
                              1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 3, 3}, false, input)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}),
                                    test::MakeAttribute("auto_pad", "SAME_LOWER")},
                                   ExpectedEPNodeAssignment::All,
                                   18);
}

// QDQ AveragePool 3D.
TEST_F(QnnHTPBackendTests, AveragePool_3D_u8) {
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 8, 8, 8}, false, -10.0f, 10.0f)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3, 3}),
                                    test::MakeAttribute("strides", std::vector<int64_t>{2, 2, 2})},
                                   ExpectedEPNodeAssignment::All);
}

// QDQ AveragePool 3D with auto_pad SAME_UPPER.
TEST_F(QnnHTPBackendTests, AveragePool_3D_AutoPad_SAME_UPPER_u8) {
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 8, 8, 8}, false, -10.0f, 10.0f)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                                    test::MakeAttribute("auto_pad", "SAME_UPPER")},
                                   ExpectedEPNodeAssignment::All);
}

// QDQ AveragePool 3D with auto_pad SAME_LOWER.
TEST_F(QnnHTPBackendTests, AveragePool_3D_AutoPad_SAME_LOWER_u8) {
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 2, 8, 8, 8}, false, -10.0f, 10.0f)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2, 2}),
                                    test::MakeAttribute("auto_pad", "SAME_LOWER")},
                                   ExpectedEPNodeAssignment::All);
}

// Covers the NHWC reshape-back path in pool_op_builder.
TEST_F(QnnHTPBackendTests, GlobalAveragePoolRank3U8) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 1 * 8 * 5);
  RunQDQAveragePoolOpTest<uint8_t>("GlobalAveragePool",
                                   {TestInputDef<float>({1, 8, 5}, false, input_data)},
                                   {},
                                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, AveragePool1DFusedQnnNodePresent) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", TestInputDef<float>({1, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 9)));
    builder.MakeOutput("output");

    builder.AddNode("avgpool", "AveragePool",
                    {"input"},
                    {"output"},
                    "",
                    {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                     test::MakeAttribute("strides", std::vector<int64_t>{3}),
                     test::MakeAttribute("pads", std::vector<int64_t>{0, 0}),
                     test::MakeAttribute("auto_pad", "NOTSET")});
  };

  ProviderOptions options;
  options["backend_type"] = "htp";

  std::function<void(const Ort::Session&)> check_num_nodes = [](const Ort::Session& session) {
    // The Reshape -> Pool -> Reshape gets fused to a single QNN node, so there should be
    // exactly 1 QNN EP subgraph.
    size_t num_qnn_subgraphs = 0;
    for (const auto& subgraph : session.GetEpGraphAssignmentInfo()) {
      if (subgraph.GetEpName() == kQnnExecutionProvider) {
        num_qnn_subgraphs++;
      }
    }
    EXPECT_EQ(num_qnn_subgraphs, 1u) << "Expected 1 fused QNN node for AveragePool rank-3 input.";
  };

  RunQnnModelTest(build_test_case,
                  options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  1e-5,
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  true,
                  &check_num_nodes);
}

TEST_F(QnnHTPBackendTests, AveragePoolRank3U8) {
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 3, 5}, false, -10.0f, 10.0f)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                                    test::MakeAttribute("strides", std::vector<int64_t>{1}),
                                    test::MakeAttribute("pads", std::vector<int64_t>{1, 1})},
                                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, AveragePoolRank3CountIncludePadU8) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 1 * 3 * 5);
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 3, 5}, false, input_data)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                                    test::MakeAttribute("strides", std::vector<int64_t>{1}),
                                    test::MakeAttribute("pads", std::vector<int64_t>{1, 1}),
                                    test::MakeAttribute("count_include_pad", static_cast<int64_t>(1))},
                                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, AveragePoolRank3AutoPadSameUpperU8) {
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 3, 4}, false, -10.0f, 10.0f)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                                    test::MakeAttribute("strides", std::vector<int64_t>{2}),
                                    test::MakeAttribute("auto_pad", "SAME_UPPER")},
                                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, AveragePoolRank3AutoPadSameLowerU8) {
  RunQDQAveragePoolOpTest<uint8_t>("AveragePool",
                                   {TestInputDef<float>({1, 3, 4}, false, -10.0f, 10.0f)},
                                   {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3}),
                                    test::MakeAttribute("strides", std::vector<int64_t>{2}),
                                    test::MakeAttribute("auto_pad", "SAME_LOWER")},
                                   ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// AveragePool with kernel size equal to the spatial dimension of input tensor.
TEST_F(QnnGPUBackendTests, AveragePool_AsGlobal) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                        test::MakeAttribute("strides", std::vector<int64_t>{3, 3})},
                       ExpectedEPNodeAssignment::All, "gpu");
}

// Test GlobalAveragePool on QNN GPU backend.
TEST_F(QnnGPUBackendTests, GlobalAveragePool) {
  RunAveragePoolOpTest("GlobalAveragePool",
                       {TestInputDef<float>({1, 2, 3, 3}, false, GetFloatDataInRange(-10.0f, 10.0f, 18))},
                       {},
                       ExpectedEPNodeAssignment::All, "gpu");
}

// AveragePool that counts padding.
TEST_F(QnnGPUBackendTests, AveragePool_CountIncludePad) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 3, 4, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 3 * 4 * 5))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                        test::MakeAttribute("count_include_pad", static_cast<int64_t>(1))},
                       ExpectedEPNodeAssignment::All, "gpu");
}

// AveragePool that use auto_pad 'SAME_UPPER'.
TEST_F(QnnGPUBackendTests, AveragePool_AutopadSameUpper) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 3, 4, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 3 * 4 * 5))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                        test::MakeAttribute("count_include_pad", static_cast<int64_t>(1)),
                        test::MakeAttribute("auto_pad", "SAME_UPPER")},
                       ExpectedEPNodeAssignment::All, "gpu");
}

// AveragePool that use auto_pad 'SAME_LOWER'.
TEST_F(QnnGPUBackendTests, AveragePool_AutopadSameLower) {
  RunAveragePoolOpTest("AveragePool",
                       {TestInputDef<float>({1, 3, 4, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 3 * 4 * 5))},
                       {test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 3}),
                        test::MakeAttribute("count_include_pad", static_cast<int64_t>(1)),
                        test::MakeAttribute("auto_pad", "SAME_LOWER")},
                       ExpectedEPNodeAssignment::All, "gpu");
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
