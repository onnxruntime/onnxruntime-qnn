// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>

#include <gsl/util>
#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

static GetTestQDQModelFn<uint16_t> BuildQDQGroupNormTestCase(
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& scale_def,
    const TestInputDef<float>& bias_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, scale_def, bias_def, attrs](ModelTestBuilder& builder,
                                                 std::vector<QuantParams<uint16_t>>& output_qparams) {
    MakeTestInput(builder, "X", input_def);
    const QuantParams<uint16_t> input_qparams = GetTestInputQuantParams<uint16_t>(input_def);
    const std::string input_dq = AddQDQNodePair<uint16_t>(builder, "qdq_input", "X", input_qparams.scale,
                                                          input_qparams.zero_point, /*use_contrib_qdq=*/false);

    MakeTestInput(builder, "scale", scale_def);
    const QuantParams<uint16_t> scale_qparams = GetTestInputQuantParams<uint16_t>(scale_def);
    const std::string scale_dq = AddQDQNodePair<uint16_t>(builder, "qdq_scale", "scale", scale_qparams.scale,
                                                          scale_qparams.zero_point, /*use_contrib_qdq=*/false);

    const std::string bias_dq = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                     input_qparams.scale * scale_qparams.scale,
                                                     /*use_contrib_qdq=*/false);

    // Preserve the intermediate output shape through the NCHW-to-NHWC layout transformation.
    auto* output_value_info = builder.model_.mutable_graph()->add_value_info();
    output_value_info->set_name("Y");
    auto* output_tensor_type = output_value_info->mutable_type()->mutable_tensor_type();
    output_tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* output_shape = output_tensor_type->mutable_shape();
    for (const int64_t dim : input_def.GetShape()) {
      output_shape->add_dim()->set_dim_value(dim);
    }

    builder.AddNode("group_norm", "GroupNormalization", {input_dq, scale_dq, bias_dq}, {"Y"}, "", attrs);
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_output", "Y", output_qparams[0].scale,
                                                    output_qparams[0].zero_point, /*use_contrib_qdq=*/false);
  };
}

TEST_F(QnnHTPBackendTests, GroupNormQdqU16U16) {
  namespace fs = std::filesystem;
  const fs::path graph_dir = fs::temp_directory_path() / "GroupNormQdqU16U16";
  fs::remove_all(graph_dir);
  ASSERT_TRUE(fs::create_directories(graph_dir));
  auto cleanup = gsl::finally([&graph_dir]() { fs::remove_all(graph_dir); });

  const TestInputDef<float> input_def({1, 8, 4, 4}, false, -1.0f, 1.0f);
  const TestInputDef<float> scale_def({8}, true, 0.5f, 1.5f);
  const TestInputDef<float> bias_def({8}, true, -0.1f, 0.1f);
  const std::vector<ONNX_NAMESPACE::AttributeProto> attrs{
      test::MakeAttribute("num_groups", static_cast<int64_t>(4)),
      test::MakeAttribute("epsilon", 1e-05f)};

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = graph_dir.string();
#ifdef __linux__
  provider_options["htp_arch"] = "73";
#endif

  TestQDQModelAccuracy(
      BuildOpTestCase<float>("group_norm", "GroupNormalization", {input_def, scale_def, bias_def}, {}, attrs),
      BuildQDQGroupNormTestCase(input_def, scale_def, bias_def, attrs),
      provider_options,
      /*opset=*/21,
      ExpectedEPNodeAssignment::All);

  if (::testing::Test::IsSkipped()) {
    return;
  }
  AssertOpInQnnGraph(graph_dir, "GroupNorm", 1);
  // Graph I/O contributes one Q and one DQ; a leaked bias DQ would increase the latter count.
  AssertOpInQnnGraph(graph_dir, "Dequantize", 1);
  AssertOpInQnnGraph(graph_dir, "Quantize", 1);
  AssertQnnGraphInputCount(graph_dir, 1);
}

// Test GroupNormalization operator on HTP backend with default parameters
TEST_F(QnnHTPBackendTests, GroupNorm_Float_Default) {
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f, 0.25f, 0.45f, 0.65f, 0.85f,
      0.12f, 0.32f, 0.52f, 0.72f, 0.22f, 0.42f, 0.62f, 0.82f};
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {0.1f, 0.2f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", TestInputDef<float>({1, 2, 3, 4}, false, input_data));
    MakeTestInput<float>(builder, "scale", TestInputDef<float>({2}, true, scale_data));
    MakeTestInput<float>(builder, "bias", TestInputDef<float>({2}, true, bias_data));

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("num_groups", static_cast<int64_t>(1)));
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-05f));

    builder.AddNode("group_norm", "GroupNormalization", {"X", "scale", "bias"}, {"Y"}, "", attributes);
    builder.MakeOutput<float>("Y", std::vector<int64_t>{1, 2, 3, 4});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.01f)});
}

// Test GroupNormalization operator on CPU backend
TEST_F(QnnCPUBackendTests, GroupNorm_Float_CPU) {
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f, 0.25f, 0.45f, 0.65f, 0.85f,
      0.12f, 0.32f, 0.52f, 0.72f, 0.22f, 0.42f, 0.62f, 0.82f};
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {0.1f, 0.2f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", TestInputDef<float>({1, 2, 3, 4}, false, input_data));
    MakeTestInput<float>(builder, "scale", TestInputDef<float>({2}, true, scale_data));
    MakeTestInput<float>(builder, "bias", TestInputDef<float>({2}, true, bias_data));

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("num_groups", static_cast<int64_t>(1)));
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-05f));

    builder.AddNode("group_norm", "GroupNormalization", {"X", "scale", "bias"}, {"Y"}, "", attributes);
    builder.MakeOutput<float>("Y", std::vector<int64_t>{1, 2, 3, 4});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.01f)});
}

// Test GroupNormalization operator with multiple groups
TEST_F(QnnHTPBackendTests, GroupNorm_Float_MultipleGroups) {
  // Input with 4 channels, to be divided into 2 groups
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f, 0.25f, 0.45f, 0.65f, 0.85f,
      0.12f, 0.32f, 0.52f, 0.72f, 0.22f, 0.42f, 0.62f, 0.82f,
      0.11f, 0.31f, 0.51f, 0.71f, 0.21f, 0.41f, 0.61f, 0.81f,
      0.13f, 0.33f, 0.53f, 0.73f, 0.23f, 0.43f, 0.63f, 0.83f,
      0.14f, 0.34f, 0.54f, 0.74f, 0.24f, 0.44f, 0.64f, 0.84f};
  std::vector<float> scale_data = {1.0f, 2.0f, 0.5f, 1.5f};
  std::vector<float> bias_data = {0.1f, 0.2f, 0.3f, 0.4f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", TestInputDef<float>({1, 4, 3, 4}, false, input_data));
    MakeTestInput<float>(builder, "scale", TestInputDef<float>({4}, true, scale_data));
    MakeTestInput<float>(builder, "bias", TestInputDef<float>({4}, true, bias_data));

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("num_groups", static_cast<int64_t>(2)));  // 4 channels / 2 groups = 2 channels per group
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-05f));

    builder.AddNode("group_norm", "GroupNormalization", {"X", "scale", "bias"}, {"Y"}, "", attributes);
    builder.MakeOutput<float>("Y", std::vector<int64_t>{1, 4, 3, 4});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.01f)});
}

// Test GroupNormalization operator with different epsilon value
TEST_F(QnnHTPBackendTests, GroupNorm_Float_LargeEpsilon) {
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f, 0.25f, 0.45f, 0.65f, 0.85f,
      0.12f, 0.32f, 0.52f, 0.72f, 0.22f, 0.42f, 0.62f, 0.82f};
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {0.1f, 0.2f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", TestInputDef<float>({1, 2, 3, 4}, false, input_data));
    MakeTestInput<float>(builder, "scale", TestInputDef<float>({2}, true, scale_data));
    MakeTestInput<float>(builder, "bias", TestInputDef<float>({2}, true, bias_data));

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("num_groups", static_cast<int64_t>(1)));
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 0.1f));  // Larger epsilon value

    builder.AddNode("group_norm", "GroupNormalization", {"X", "scale", "bias"}, {"Y"}, "", attributes);
    builder.MakeOutput<float>("Y", std::vector<int64_t>{1, 2, 3, 4});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.01f)});
}

// Test GroupNormalization operator with 3D input
TEST_F(QnnHTPBackendTests, GroupNorm_Float_3D) {
  std::vector<float> input_data = {
      0.1f, 0.3f, 0.5f, 0.7f,
      0.2f, 0.4f, 0.6f, 0.8f,
      0.15f, 0.35f, 0.55f, 0.75f,
      0.25f, 0.45f, 0.65f, 0.85f};
  std::vector<float> scale_data = {1.0f, 2.0f, 0.5f, 1.5f};
  std::vector<float> bias_data = {0.1f, 0.2f, 0.3f, 0.4f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", TestInputDef<float>({1, 4, 4}, false, input_data));
    MakeTestInput<float>(builder, "scale", TestInputDef<float>({4}, true, scale_data));
    MakeTestInput<float>(builder, "bias", TestInputDef<float>({4}, true, bias_data));

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("num_groups", static_cast<int64_t>(2)));  // 4 channels / 2 groups = 2 channels per group
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-05f));

    builder.AddNode("group_norm", "GroupNormalization", {"X", "scale", "bias"}, {"Y"}, "", attributes);
    builder.MakeOutput<float>("Y", std::vector<int64_t>{1, 4, 4});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.01f)});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
