// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

static void AddFloatValueInfo(ModelTestBuilder& builder, const std::string& name,
                              const std::vector<int64_t>& shape) {
  auto* value_info = builder.graph_->add_value_info();
  value_info->set_name(name);
  auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (const int64_t dim : shape) {
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
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

// Regression tests for the NHWC layout-transform shape-loss bug: ORT's layout transformer could
// duplicate a GroupNormalization node when its NCHW input (or output) fed multiple consumers, one
// layout-sensitive (e.g. MaxPool, which is wrapped in NCHW<->NHWC Transposes by the transformer) and
// one not (e.g. Relu). The duplicated node's output tensor did not always inherit a resolved shape,
// which failed QNN EP's own shape lookup ("Cannot get shape") and stranded the node with no EP able
// to run it. GroupNormalizationOpBuilder now performs the NCHW<->NHWC conversion internally and
// ShouldConvertDataLayoutForOp suppresses ORT's automatic layout transform for GroupNormalization,
// so this fan-out no longer triggers node duplication.

// Input-side fan-out: X feeds GroupNorm (not layout-sensitive per QNN EP) and MaxPool (layout-sensitive).
TEST_F(QnnHTPBackendTests, GroupNorm_Float_InputFanout) {
  std::vector<float> input_data = GetFloatDataInRange(0.0f, 1.0f, 24);
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {0.1f, 0.2f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", TestInputDef<float>({1, 2, 3, 4}, false, input_data));
    MakeTestInput<float>(builder, "scale", TestInputDef<float>({2}, true, scale_data));
    MakeTestInput<float>(builder, "bias", TestInputDef<float>({2}, true, bias_data));

    std::vector<ONNX_NAMESPACE::AttributeProto> gn_attrs;
    gn_attrs.push_back(builder.MakeScalarAttribute("num_groups", static_cast<int64_t>(1)));
    gn_attrs.push_back(builder.MakeScalarAttribute("epsilon", 1e-05f));
    builder.AddNode("group_norm", "GroupNormalization", {"X", "scale", "bias"}, {"gn_out"}, "", gn_attrs);
    builder.MakeOutput<float>("gn_out", std::vector<int64_t>{1, 2, 3, 4});

    // Second consumer of X: MaxPool is layout-sensitive, so ORT's layout transformer wraps it in
    // NCHW<->NHWC Transposes. Before the fix, this is what triggered GroupNormalization duplication.
    std::vector<ONNX_NAMESPACE::AttributeProto> pool_attrs;
    pool_attrs.push_back(test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}));
    builder.AddNode("max_pool", "MaxPool", {"X"}, {"pool_out"}, "", pool_attrs);
    builder.MakeOutput<float>("pool_out", std::vector<int64_t>{1, 2, 3, 4});
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

// Output-side fan-out: GroupNorm's NCHW output feeds MaxPool (layout-sensitive) and Relu (not).
TEST_F(QnnHTPBackendTests, GroupNorm_Float_OutputFanout) {
  std::vector<float> input_data = GetFloatDataInRange(0.0f, 1.0f, 24);
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {0.1f, 0.2f};

  auto build_test_case = [&](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", TestInputDef<float>({1, 2, 3, 4}, false, input_data));
    MakeTestInput<float>(builder, "scale", TestInputDef<float>({2}, true, scale_data));
    MakeTestInput<float>(builder, "bias", TestInputDef<float>({2}, true, bias_data));

    std::vector<ONNX_NAMESPACE::AttributeProto> gn_attrs;
    gn_attrs.push_back(builder.MakeScalarAttribute("num_groups", static_cast<int64_t>(1)));
    gn_attrs.push_back(builder.MakeScalarAttribute("epsilon", 1e-05f));
    builder.AddNode("group_norm", "GroupNormalization", {"X", "scale", "bias"}, {"gn_out"}, "", gn_attrs);
    AddFloatValueInfo(builder, "gn_out", {1, 2, 3, 4});

    std::vector<ONNX_NAMESPACE::AttributeProto> pool_attrs;
    pool_attrs.push_back(test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}));
    builder.AddNode("max_pool", "MaxPool", {"gn_out"}, {"pool_out"}, "", pool_attrs);
    builder.MakeOutput<float>("pool_out", std::vector<int64_t>{1, 2, 3, 4});

    builder.AddNode("relu", "Relu", {"gn_out"}, {"relu_out"}, "");
    builder.MakeOutput<float>("relu_out", std::vector<int64_t>{1, 2, 3, 4});
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

// QDQ variant of the input-side fan-out regression test.
template <typename QuantType>
static GetTestQDQModelFn<QuantType> BuildQDQGroupNormInputFanoutTestCase(
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& scale_def,
    const TestInputDef<float>& bias_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& gn_attrs) {
  return [input_def, scale_def, bias_def, gn_attrs](ModelTestBuilder& builder,
                                                    std::vector<QuantParams<QuantType>>& output_qparams) {
    MakeTestInput(builder, "X", input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    std::string x_qdq = AddQDQNodePair<QuantType>(builder, "x_qdq", "X", input_qparams.scale,
                                                  input_qparams.zero_point);

    MakeTestInput(builder, "scale", scale_def);
    QuantParams<QuantType> scale_qparams = GetTestInputQuantParams<QuantType>(scale_def);
    std::string scale_qdq = AddQDQNodePair<QuantType>(builder, "scale_qdq", "scale", scale_qparams.scale,
                                                      scale_qparams.zero_point);

    MakeTestInput(builder, "bias", bias_def);
    QuantParams<QuantType> bias_qparams = GetTestInputQuantParams<QuantType>(bias_def);
    std::string bias_qdq = AddQDQNodePair<QuantType>(builder, "bias_qdq", "bias", bias_qparams.scale,
                                                     bias_qparams.zero_point);

    builder.AddNode("group_norm", "GroupNormalization", {x_qdq, scale_qdq, bias_qdq}, {"gn_out"}, "", gn_attrs);
    AddFloatValueInfo(builder, "gn_out", {1, 2, 3, 4});
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "gn_out_qdq", "gn_out", output_qparams[0].scale,
                                                     output_qparams[0].zero_point);

    // Second consumer of the quantized X: MaxPool is layout-sensitive, so ORT's layout transformer
    // wraps it in NCHW<->NHWC Transposes. Before the fix, this triggered GroupNormalization duplication.
    std::vector<ONNX_NAMESPACE::AttributeProto> pool_attrs;
    pool_attrs.push_back(test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}));
    builder.AddNode("max_pool", "MaxPool", {x_qdq}, {"pool_out"}, "", pool_attrs);
    AddFloatValueInfo(builder, "pool_out", {1, 2, 3, 4});
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "pool_out_qdq", "pool_out", output_qparams[1].scale,
                                                     output_qparams[1].zero_point);
  };
}

// Float reference model matching BuildQDQGroupNormInputFanoutTestCase's graph shape (same two
// outputs), required so TestQDQModelAccuracy sizes output_qparams to 2 entries.
static GetTestModelFn BuildGroupNormInputFanoutTestCase(
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& scale_def,
    const TestInputDef<float>& bias_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& gn_attrs) {
  return [input_def, scale_def, bias_def, gn_attrs](ModelTestBuilder& builder) {
    MakeTestInput(builder, "X", input_def);
    MakeTestInput(builder, "scale", scale_def);
    MakeTestInput(builder, "bias", bias_def);

    builder.AddNode("group_norm", "GroupNormalization", {"X", "scale", "bias"}, {"gn_out"}, "", gn_attrs);
    builder.MakeOutput("gn_out");

    std::vector<ONNX_NAMESPACE::AttributeProto> pool_attrs;
    pool_attrs.push_back(test::MakeAttribute("kernel_shape", std::vector<int64_t>{1, 1}));
    builder.AddNode("max_pool", "MaxPool", {"X"}, {"pool_out"}, "", pool_attrs);
    builder.MakeOutput("pool_out");
  };
}

TEST_F(QnnHTPBackendTests, GroupNorm_QDQ_U8_InputFanout) {
  std::vector<ONNX_NAMESPACE::AttributeProto> gn_attrs;
  gn_attrs.push_back(test::MakeAttribute("num_groups", static_cast<int64_t>(1)));
  gn_attrs.push_back(test::MakeAttribute("epsilon", 1e-05f));

  TestInputDef<float> input_def({1, 2, 3, 4}, false, GetFloatDataInRange(0.0f, 1.0f, 24));
  TestInputDef<float> scale_def({2}, true, {1.0f, 2.0f});
  TestInputDef<float> bias_def({2}, true, {0.1f, 0.2f});

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  TestQDQModelAccuracy(BuildGroupNormInputFanoutTestCase(input_def, scale_def, bias_def, gn_attrs),
                       BuildQDQGroupNormInputFanoutTestCase<uint8_t>(input_def, scale_def, bias_def, gn_attrs),
                       provider_options,
                       21,
                       ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
