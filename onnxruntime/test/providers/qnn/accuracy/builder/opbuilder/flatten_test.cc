// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS && \
    (defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__linux__))

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/infra/flatten_graph_assertions.h"
#include "test/providers/qnn/infra/flatten_test_data.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/flatten_specs.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename QType>
GetTestQDQModelFn<QType> BuildCheckedQdqFlatten(const FlattenSpec& s,
                                                const TestInputDef<float>& input,
                                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  const auto data = GetFlattenQdqData(s.input_shape, std::is_same_v<QType, uint16_t>);
  auto build = BuildQDQOpTestCase<QType>("flatten_node", "Flatten", {input}, {}, attrs,
                                         kOnnxDomain, std::is_same_v<QType, uint16_t>);
  return [build, data](ModelTestBuilder& model, std::vector<QuantParams<QType>>& output_params) {
    ASSERT_EQ(output_params.size(), 1u);
    // TestQDQModelAccuracy obtains these from a real CPU float session.
    // Equality also qualifies the paired Snapshot's derived metadata.
    EXPECT_FLOAT_EQ(output_params[0].scale, data.output.scale);
    EXPECT_EQ(output_params[0].zero_point, data.output.zero_point);
    build(model, output_params);
    const auto& actual = model.feeds_.at("quant_input_defs_0");
    ASSERT_EQ(actual.GetTensorTypeAndShapeInfo().GetElementCount(), data.input_values.size());
    const auto* values = actual.GetTensorData<float>();
    for (size_t i = 0; i < data.input_values.size(); ++i) EXPECT_EQ(values[i], data.input_values[i]);
  };
}

GetTestModelFn BuildRank5GraphIO() {
  return [](ModelTestBuilder& builder) {
    builder.MakeInput<uint8_t>("X", {1, 2, 3, 4, 5}, 0, 255);
    builder.AddDequantizeLinearNode<uint8_t>("dq", "X", 1.0f, 0, "dq_out");
    builder.AddNode("flatten", "Flatten", {"dq_out"}, {"flatten_out"}, "",
                    {builder.MakeScalarAttribute("axis", int64_t{2})});
    builder.AddQuantizeLinearNode<uint8_t>("q", "flatten_out", 1.0f, 0, "Y");
    builder.MakeOutput("Y");
  };
}

void RunGraphIOAccuracy(const FlattenSpec& s) {
  ProviderOptions options{{"backend_type", "htp"}, {"offload_graph_io_quantization", "0"}};
  // Preserve the general legacy helper's broad ARM64 V68 skip before staging
  // any files. This is policy parity, not a claimed U8 hardware restriction.
  CONDITIONAL_SKIP_TEST_ON_LINUX_ARM64(options, QNN_HTP_DEVICE_ARCH_V68, "FP16");
  const auto slot = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto directory = std::filesystem::temp_directory_path() / ("qnn_flatten_rank5_accuracy_" + slot);
  ASSERT_TRUE(std::filesystem::create_directory(directory));
  struct Cleanup {
    std::filesystem::path directory;
    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove_all(directory, ec);
    }
  } cleanup{directory};
  options["dump_json_qnn_graph"] = "1";
  options["json_qnn_graph_dir"] = directory.string();
  std::function<void(const Ort::Session&)> checker = [&](const Ort::Session&) {
    std::vector<std::filesystem::path> dumps;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
      if (entry.path().extension() == ".json" && entry.path().filename().string().find("_tensor_log") == std::string::npos) {
        dumps.push_back(entry.path());
      }
    }
    ASSERT_EQ(dumps.size(), 1u);
    std::ifstream stream(dumps.front());
    nlohmann::json graph;
    stream >> graph;
    // QNN's quantized UINT8 encoding is UFIXED_POINT_8 (0x0408).
    AssertFlattenReshapeGraph(graph, "X", "Y", s.input_shape, s.output_shape, 0x0408);
  };
  RunQnnModelTest(BuildRank5GraphIO(), options, 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier{}, &checker});
}

}  // namespace

class QnnAcc_Flatten_AccuracyTest : public QnnHTPBackendTests,
                                    public ::testing::WithParamInterface<FlattenSpec> {};

TEST_P(QnnAcc_Flatten_AccuracyTest, Case) {
  const auto& s = GetParam();
  if (s.profile == FlattenProfile::QdqU8GraphIO) {
    RunGraphIOAccuracy(s);
    return;
  }
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  if (s.axis) attrs.push_back(test::MakeAttribute("axis", *s.axis));
  ProviderOptions options{{"backend_type", "htp"}};
  if (s.profile == FlattenProfile::Int32) {
    const auto count = std::accumulate(s.input_shape.begin(), s.input_shape.end(), int64_t{1}, std::multiplies<>{});
    std::vector<int32_t> values(static_cast<size_t>(count));
    std::iota(values.begin(), values.end(), 0);
    RunQnnModelTest(BuildOpTestCase<int32_t>("flatten_node", "Flatten",
                                             {TestInputDef<int32_t>(s.input_shape, false, values)}, {}, attrs),
                    options, 13, EPVerificationParams{ExpectedEPNodeAssignment::All});
    return;
  }
  const TestInputDef<float> input(s.input_shape, false, -10.0f, 10.0f);
  auto float_model = BuildOpTestCase<float>("flatten_node", "Flatten", {input}, {}, attrs);
  if (s.profile == FlattenProfile::Float) {
    RunQnnModelTest(float_model, options, 13, EPVerificationParams{ExpectedEPNodeAssignment::All});
    return;
  }
  options["offload_graph_io_quantization"] = "0";
  if (s.profile == FlattenProfile::QdqU16) {
    TestQDQModelAccuracy(float_model,
                         BuildCheckedQdqFlatten<uint16_t>(s, input, attrs),
                         options, 13, ExpectedEPNodeAssignment::All);
  } else {
    TestQDQModelAccuracy(float_model,
                         BuildCheckedQdqFlatten<uint8_t>(s, input, attrs),
                         options, 13, ExpectedEPNodeAssignment::All);
  }
}

INSTANTIATE_TEST_SUITE_P(
    , QnnAcc_Flatten_AccuracyTest, ::testing::ValuesIn(kFlattenSpecs),
    [](const ::testing::TestParamInfo<FlattenSpec>& info) { return std::string(info.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // non-minimal Accuracy build on legacy HTP platforms
