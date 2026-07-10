// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/unittest_util/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

static void RunSkipSimplifiedLayerNormHtpFloatTest(const TestInputDef<float>& input_def,
                                                   const TestInputDef<float>& skip_def,
                                                   const TestInputDef<float>& gamma_def,
                                                   const std::vector<TestInputDef<float>>& bias_defs,
                                                   const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                   ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(
      BuildOpTestCase<float, float>("skip_simplified_layernorm",
                                    "SkipSimplifiedLayerNormalization",
                                    {input_def, skip_def, gamma_def},
                                    bias_defs,
                                    attrs,
                                    kMSDomain),
      provider_options,
      13,  // ai.onnx opset for the graph
      EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(1e-3f)});
}

// HTP: 2D float, no bias.
TEST_F(QnnHTPBackendTests, SkipSimplifiedLayerNorm_Float_2D_NoBias) {
  RunSkipSimplifiedLayerNormHtpFloatTest(
      TestInputDef<float>({2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 8)),
      TestInputDef<float>({2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 8)),
      TestInputDef<float>({4}, true, GetFloatDataInRange(0.5f, 1.5f, 4)),
      {},
      {test::MakeAttribute("epsilon", 1e-5f)},
      ExpectedEPNodeAssignment::All);
}

// HTP: 3D float, no bias.
TEST_F(QnnHTPBackendTests, SkipSimplifiedLayerNorm_Float_3D_NoBias) {
  RunSkipSimplifiedLayerNormHtpFloatTest(
      TestInputDef<float>({1, 2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 8)),
      TestInputDef<float>({1, 2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 8)),
      TestInputDef<float>({4}, true, GetFloatDataInRange(0.5f, 1.5f, 4)),
      {},
      {test::MakeAttribute("epsilon", 1e-5f)},
      ExpectedEPNodeAssignment::All);
}

// HTP: 3D float with bias.
TEST_F(QnnHTPBackendTests, SkipSimplifiedLayerNorm_Float_3D_WithBias) {
  RunSkipSimplifiedLayerNormHtpFloatTest(
      TestInputDef<float>({1, 2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 8)),
      TestInputDef<float>({1, 2, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 8)),
      TestInputDef<float>({4}, true, GetFloatDataInRange(0.5f, 1.5f, 4)),
      {TestInputDef<float>({4}, true, GetFloatDataInRange(-0.1f, 0.1f, 4))},
      {test::MakeAttribute("epsilon", 1e-5f)},
      ExpectedEPNodeAssignment::All);
}

// HTP: training outputs (mean) must be rejected.
TEST_F(QnnHTPBackendTests, SkipSimplifiedLayerNorm_TrainingOutputs_Unsupported) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 2, 4}, -1.0f, 1.0f);
    builder.MakeInput<float>("skip", {1, 2, 4}, -1.0f, 1.0f);
    builder.MakeInitializer<float>("gamma", {4}, 0.5f, 1.5f);
    builder.MakeOutput("output_y");
    builder.MakeOutput("output_mean");
    builder.AddNode("skip_sln", "SkipSimplifiedLayerNormalization",
                    {"input", "skip", "gamma"},
                    {"output_y", "output_mean"},
                    kMSDomain,
                    {test::MakeAttribute("epsilon", 1e-5f)});
  };

  RunQnnModelTest(build_model, provider_options, 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::None},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  false /* verify_outputs */);
}

// HTP: output[3] (input_skip_bias_sum) exposed as graph output.
TEST_F(QnnHTPBackendTests, SkipSimplifiedLayerNorm_Float_3D_WithOutput3) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 2, 4}, -1.0f, 1.0f);
    builder.MakeInput<float>("skip", {1, 2, 4}, -1.0f, 1.0f);
    builder.MakeInitializer<float>("gamma", {4}, 0.5f, 1.5f);
    builder.MakeOutput("output_y");
    builder.MakeOutput("output_sum");
    builder.AddNode("skip_sln", "SkipSimplifiedLayerNormalization",
                    {"input", "skip", "gamma"},
                    {"output_y", "", "", "output_sum"},
                    kMSDomain,
                    {test::MakeAttribute("epsilon", 1e-5f)});
  };

  RunQnnModelTest(build_model, provider_options, 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// HTP: 3D float with bias and output[3].
TEST_F(QnnHTPBackendTests, SkipSimplifiedLayerNorm_Float_3D_WithBiasAndOutput3) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto build_model = [](ModelTestBuilder& builder) {
    builder.MakeInput<float>("input", {1, 2, 4}, -1.0f, 1.0f);
    builder.MakeInput<float>("skip", {1, 2, 4}, -1.0f, 1.0f);
    builder.MakeInitializer<float>("gamma", {4}, 0.5f, 1.5f);
    builder.MakeInitializer<float>("bias", {4}, -0.1f, 0.1f);
    builder.MakeOutput("output_y");
    builder.MakeOutput("output_sum");
    builder.AddNode("skip_sln", "SkipSimplifiedLayerNormalization",
                    {"input", "skip", "gamma", "bias"},
                    {"output_y", "", "", "output_sum"},
                    kMSDomain,
                    {test::MakeAttribute("epsilon", 1e-5f)});
  };

  RunQnnModelTest(build_model, provider_options, 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

#endif  // aarch64 / ARM64 / linux
}  // namespace test
}  // namespace onnxruntime

#endif  // !ORT_MINIMAL_BUILD
