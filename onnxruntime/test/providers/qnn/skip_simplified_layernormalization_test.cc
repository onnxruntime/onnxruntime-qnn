// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "gtest/gtest.h"

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

#endif  // aarch64 / ARM64 / linux
}  // namespace test
}  // namespace onnxruntime

#endif  // !ORT_MINIMAL_BUILD
