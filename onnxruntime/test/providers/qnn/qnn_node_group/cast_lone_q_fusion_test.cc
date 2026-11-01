// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// cast_input -> Cast -> Q -> DQ ----
//                                   Add -> Q -> DQ -> output
//             input2 -> Q -> DQ ----
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildCastAddQDQTestCase() {
  return [](ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    builder.graph_->set_name("cast_lone_q_fusion_graph");

    MakeTestInput<uint8_t>(builder, "cast_input",
                           TestInputDef<uint8_t>({2, 3}, false, {0, 1, 0, 1, 0, 1}));

    const std::vector<float> add_input2_data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    MakeTestInput<float>(builder, "add_input2",
                         TestInputDef<float>({2, 3}, false, add_input2_data));

    builder.AddNode(
        "cast", "Cast", {"cast_input"}, {"cast_output"}, kOnnxDomain,
        {builder.MakeScalarAttribute(
            "to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});

    const QuantParams<QuantType> input_qparams =
        GetDataQuantParams<QuantType>(gsl::make_span(add_input2_data));
    const std::string cast_after_qdq =
        AddQDQNodePair<QuantType>(builder, "cast_qdq", "cast_output",
                                  input_qparams.scale, input_qparams.zero_point);
    const std::string input2_after_qdq =
        AddQDQNodePair<QuantType>(builder, "input2_qdq", "add_input2",
                                  input_qparams.scale, input_qparams.zero_point);

    builder.AddNode("add", "Add", {cast_after_qdq, input2_after_qdq}, {"add_output"});
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "output_qdq", "add_output",
        output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

GetTestModelFn BuildCastAddTestCase() {
  return [](ModelTestBuilder& builder) {
    builder.graph_->set_name("cast_add_graph");

    MakeTestInput<uint8_t>(builder, "cast_input",
                           TestInputDef<uint8_t>({2, 3}, false, {0, 1, 0, 1, 0, 1}));
    MakeTestInput<float>(builder, "add_input2",
                         TestInputDef<float>({2, 3}, false,
                                             {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}));

    builder.AddNode(
        "cast", "Cast", {"cast_input"}, {"cast_output"}, kOnnxDomain,
        {builder.MakeScalarAttribute(
            "to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.AddNode("add", "Add", {"cast_output", "add_input2"}, {"output"});
    builder.MakeOutput("output");
  };
}

ProviderOptions GetHTPProviderOptions(const std::filesystem::path& json_dir) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

template <typename QuantType>
void RunCastLoneQTest(const std::filesystem::path& json_dir,
                      ExpectedEPNodeAssignment expected_ep_assignment) {
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));

  TestQDQModelAccuracy<QuantType>(BuildCastAddTestCase(),
                                  BuildCastAddQDQTestCase<QuantType>(),
                                  GetHTPProviderOptions(json_dir),
                                  /*opset_version=*/21,
                                  expected_ep_assignment);
}

}  // namespace

TEST_F(QnnHTPBackendTests, CastLoneQFusion_U8_EmitsConvert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "CastLoneQFusion_U8_EmitsConvert";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunCastLoneQTest<uint8_t>(json_dir, ExpectedEPNodeAssignment::All);
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpWithNameInQnnGraph(json_dir, "Convert", "cast_ort_qnn_ep_convert");
  AssertOpInQnnGraph(json_dir, "Cast", /*count=*/0);
}

TEST_F(QnnHTPBackendTests, CastLoneQFusion_U16_EmitsConvert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "CastLoneQFusion_U16_EmitsConvert";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunCastLoneQTest<uint16_t>(json_dir, ExpectedEPNodeAssignment::All);
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpWithNameInQnnGraph(json_dir, "Convert", "cast_ort_qnn_ep_convert");
  AssertOpInQnnGraph(json_dir, "Cast", /*count=*/0);
}

TEST_F(QnnHTPBackendTests, CastLoneQFusion_S8_EmitsConvert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "CastLoneQFusion_S8_EmitsConvert";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  RunCastLoneQTest<int8_t>(json_dir, ExpectedEPNodeAssignment::All);
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpWithNameInQnnGraph(json_dir, "Convert", "cast_ort_qnn_ep_convert");
  AssertOpInQnnGraph(json_dir, "Cast", /*count=*/0);
}

TEST_F(QnnHTPBackendTests, CastLoneQFusion_S16_EmitsConvert) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "CastLoneQFusion_S16_EmitsConvert";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  // Standalone S16 Q/DQ support is partial, but the Cast -> Q portion should
  // still be composed as a Convert in the QNN partition.
  RunCastLoneQTest<int16_t>(json_dir, ExpectedEPNodeAssignment::Some);
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpWithNameInQnnGraph(json_dir, "Convert", "cast_ort_qnn_ep_convert");
  AssertOpInQnnGraph(json_dir, "Cast", /*count=*/0);
}

TEST_F(QnnHTPBackendTests, CastLoneQFusion_NoQChild_DoesNotFuse) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_dir = "CastLoneQFusion_NoQChild_DoesNotFuse";
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  RunQnnModelTest(BuildCastAddTestCase(),
                  GetHTPProviderOptions(json_dir),
                  /*opset_version=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpWithNameInQnnGraph(json_dir, "Cast", "cast");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
