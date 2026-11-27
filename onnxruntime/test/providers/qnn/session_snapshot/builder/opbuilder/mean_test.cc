// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/mean_test_utils.h"
#include "test/providers/qnn/session_snapshot/session_snapshot.h"

namespace onnxruntime {
namespace test {
namespace {

void CheckMeanGraph(const nlohmann::json& graph, const MeanSpec& spec) {
  const auto& nodes = graph.at("graph").at("nodes");
  const auto& tensors = graph.at("graph").at("tensors");
  std::vector<const nlohmann::json*> adds;
  const nlohmann::json* divide = nullptr;
  std::set<std::string> add_outputs;
  for (const auto& node : nodes) {
    const auto type = node.at("type").get<std::string>();
    if (type == "ElementWiseAdd") {
      ASSERT_EQ(node.at("input_names").size(), 2u);
      ASSERT_EQ(node.at("output_names").size(), 1u);
      adds.push_back(&node);
      add_outputs.insert(node.at("output_names")[0].get<std::string>());
    } else if (type == "ElementWiseDivide") {
      ASSERT_EQ(divide, nullptr) << "Expected one Mean divide node";
      divide = &node;
    }
  }
  ASSERT_EQ(adds.size(), spec.inputs.size() - 1);
  ASSERT_NE(divide, nullptr);
  std::string sum_output;
  std::set<std::string> inputs;
  for (size_t i = 0; i < spec.inputs.size() - 1; ++i) {
    auto it = std::find_if(adds.begin(), adds.end(), [&](const auto* node) {
      const auto first = node->at("input_names")[0].template get<std::string>();
      return i == 0 ? add_outputs.count(first) == 0 : first == sum_output;
    });
    ASSERT_NE(it, adds.end()) << "Add nodes must form one connected accumulation chain";
    const auto& node = **it;
    if (i == 0) inputs.insert(node.at("input_names")[0].get<std::string>());
    EXPECT_TRUE(inputs.insert(node.at("input_names")[1].get<std::string>()).second);
    sum_output = node.at("output_names")[0].get<std::string>();
    const auto& tensor = tensors.at(sum_output);
    EXPECT_EQ(tensor.at("dims").get<std::vector<int64_t>>(), spec.output_shape);
    EXPECT_EQ(tensor.at("data_type").get<uint32_t>(), static_cast<uint32_t>(QNN_DATATYPE_FLOAT_32));
    adds.erase(it);
  }
  EXPECT_EQ(inputs.size(), spec.inputs.size());
  ASSERT_EQ(divide->at("input_names").size(), 2u);
  ASSERT_EQ(divide->at("output_names").size(), 1u);
  EXPECT_EQ(divide->at("input_names")[0].get<std::string>(), sum_output);
  const auto& divisor = tensors.at(divide->at("input_names")[1].get<std::string>());
  EXPECT_EQ(divisor.at("type").get<uint32_t>(), static_cast<uint32_t>(QNN_TENSOR_TYPE_STATIC));
  EXPECT_EQ(divisor.at("data_type").get<uint32_t>(), static_cast<uint32_t>(QNN_DATATYPE_FLOAT_32));
  EXPECT_EQ(divisor.at("dims").get<std::vector<int64_t>>(), (std::vector<int64_t>{1}));
  // General op tensor JSON omits static bytes; Component verifies the divisor value.
  const auto& output = tensors.at(divide->at("output_names")[0].get<std::string>());
  EXPECT_EQ(output.at("dims").get<std::vector<int64_t>>(), spec.output_shape);
  EXPECT_EQ(output.at("data_type").get<uint32_t>(), static_cast<uint32_t>(QNN_DATATYPE_FLOAT_32));
}

}  // namespace

// Mean is not registered as a QDQ node group: keep the actual session's Q/DQ and
// float Add/Divide lowering rather than manufacturing a quantized Mean mock.
class QnnSnapshot_Mean_SessionTest : public QnnHTPBackendTests,
                                     public ::testing::WithParamInterface<MeanSpec> {};

TEST_P(QnnSnapshot_Mean_SessionTest, Case) {
  const auto& spec = GetParam();
  auto options = MeanProviderOptions(spec);
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC))
  options["num_graph_prepare_threads"] = "1";
#endif
  AssertSessionSnapshotJson(
      BuildMeanSessionModel(spec), options, spec.opset, spec.name, "", __FILE__,
      [&spec](const nlohmann::json& graph) { CheckMeanGraph(graph, spec); });
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_Mean_SessionTest, ::testing::ValuesIn(kMeanSpecs),
    [](const ::testing::TestParamInfo<MeanSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
