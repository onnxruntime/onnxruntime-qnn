// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

GetTestModelFn BuildTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    const int64_t num_channels = 12;
    const std::vector<int64_t> input_shape{1, num_channels, 8, 8};
    const auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // Conv1 weights
    const std::vector<int64_t> conv1_weight_shape = {num_channels, num_channels / 2, 1, 1};
    builder.MakeInitializer<float>("conv1_weight", conv1_weight_shape, -2.f, 2.f);

    // Conv1: input + conv1_weight -> conv1_out
    {
      std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
      attrs.push_back(test::MakeAttribute("group", static_cast<int64_t>(2)));
      builder.AddNode("Conv1",
                      "Conv",
                      {"input", "conv1_weight"},
                      {"conv1_out"},
                      kOnnxDomain,
                      attrs);
    }

    // Reshape1: conv1_out + shape1 -> reshape1_out
    builder.Make1DInitializer<int64_t>("shape1",
                                       {input_shape[0], 2, num_channels / 2, input_shape[2], input_shape[3]});
    builder.AddNode("Reshape1",
                    "Reshape",
                    {"conv1_out", "shape1"},
                    {"reshape1_out"});

    // Transpose: reshape1_out -> transpose_out
    {
      std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
      attrs.push_back(test::MakeAttribute("perm", std::vector<int64_t>{0, 2, 1, 3, 4}));
      builder.AddNode("Transpose1",
                      "Transpose",
                      {"reshape1_out"},
                      {"transpose_out"},
                      kOnnxDomain,
                      attrs);
    }

    // Reshape2: transpose_out + shape2 -> reshape2_out
    builder.Make1DInitializer<int64_t>("shape2", input_shape);
    builder.AddNode("Reshape2",
                    "Reshape",
                    {"transpose_out", "shape2"},
                    {"reshape2_out"});

    // Conv2 weights
    const std::vector<int64_t> conv2_weight_shape = {num_channels, 1, 3, 1};
    builder.MakeInitializer<float>("conv2_weight", conv2_weight_shape, -2.f, 2.f);

    // Conv2: reshape2_out + conv2_weight -> Y
    {
      std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
      attrs.push_back(test::MakeAttribute("group", static_cast<int64_t>(num_channels)));
      attrs.push_back(test::MakeAttribute("kernel_shape", std::vector<int64_t>{3, 1}));
      builder.MakeOutput("Y");
      builder.AddNode("Conv2",
                      "Conv",
                      {"reshape2_out", "conv2_weight"},
                      {"Y"},
                      kOnnxDomain,
                      attrs);
    }
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

}  // namespace

TEST_F(QnnHTPBackendTests, ChannelShuffleFusion) {
  const std::filesystem::path json_qnn_graph_dir = "ChannelShuffleFusion";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(BuildTestCase(),
                  provider_options,
                  /*opset_version=*/10,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);

  AssertOpInQnnGraph(json_qnn_graph_dir, "ChannelShuffle");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
