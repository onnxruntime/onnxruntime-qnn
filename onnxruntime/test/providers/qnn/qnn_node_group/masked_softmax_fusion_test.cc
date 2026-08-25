// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

GetTestModelFn BuildMaskedSoftmaxPatternATestCase() {
  return [](ModelTestBuilder& builder) -> void {
    constexpr float kScale = 0.01f;
    constexpr uint16_t kZeroPoint = 32768;
    const TestInputDef<float> score_def({1, 1, 2, 4}, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "score", score_def);
    const std::string score_qdq = AddQDQNodePair<uint16_t>(builder, "score_qdq", "score", kScale, kZeroPoint);

    // PSL attention_mask uses one for valid tokens. The fusion compares the
    // PSL's legacy mask chain is 1 - attention_mask. The fusion recognizes
    // this PSL-specific shape but compares the original mask directly.
    builder.MakeInitializer<float>("mask", {1, 1, 2, 4}, std::vector<float>(8, 1.0f));
    const std::string mask_qdq = AddQDQNodePair<uint16_t>(builder, "mask_qdq", "mask", kScale, kZeroPoint);
    builder.MakeScalarInitializer<float>("one", 1.0f);
    builder.AddNode("mask_sub", "Sub", {"one", mask_qdq}, {"mask_sub_out"});
    const std::string inverted_mask =
        AddQDQNodePair<uint16_t>(builder, "mask_sub_qdq", "mask_sub_out", kScale, kZeroPoint);
    builder.AddNode("mask_mul", "Mul", {inverted_mask, "one"}, {"mask_mul_out"});
    const std::string additive_mask =
        AddQDQNodePair<uint16_t>(builder, "additive_mask_qdq", "mask_mul_out", kScale, kZeroPoint);

    builder.MakeScalarInitializer<float>("divisor", 1.0f);
    builder.AddNode("score_div", "Div", {score_qdq, "divisor"}, {"score_div_out"});
    const std::string score = AddQDQNodePair<uint16_t>(builder, "score_div_qdq", "score_div_out", kScale, kZeroPoint);
    builder.AddNode("masked_add", "Add", {score, additive_mask}, {"masked_add_out"});
    const std::string masked_add =
        AddQDQNodePair<uint16_t>(builder, "masked_add_qdq", "masked_add_out", kScale, kZeroPoint);

    builder.MakeInitializer<float>("gate", {1, 1, 2, 4}, std::vector<float>(8, 0.0f));
    const std::string gate_qdq = AddQDQNodePair<uint16_t>(builder, "gate_qdq", "gate", kScale, kZeroPoint);
    builder.MakeScalarInitializer<float>("gate_scale", 1.0f);
    builder.AddNode("gate_mul", "Mul", {gate_qdq, "gate_scale"}, {"gate_mul_out"});
    const std::string gate = AddQDQNodePair<uint16_t>(builder, "gate_mul_qdq", "gate_mul_out", kScale, kZeroPoint);

    builder.AddNode("gated_add", "Add", {masked_add, gate}, {"gated_add_out"});
    const std::string softmax_input =
        AddQDQNodePair<uint16_t>(builder, "gated_add_qdq", "gated_add_out", kScale, kZeroPoint);
    builder.AddNode("softmax", "Softmax", {softmax_input}, {"softmax_out"}, "",
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(3))});
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "output_qdq", "softmax_out", kScale, kZeroPoint);
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

}  // namespace

TEST_F(QnnHTPBackendTests, MaskedSoftmaxPatternAFusion) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const std::filesystem::path json_qnn_graph_dir = "MaskedSoftmaxPatternAFusion";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  RunQnnModelTest(BuildMaskedSoftmaxPatternATestCase(), provider_options, /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_qnn_graph_dir, "ReduceMin", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "ElementWiseSelect", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Softmax", 1);
  AssertNodeNameContainsInQnnGraph(json_qnn_graph_dir, "_NotEqual", 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
