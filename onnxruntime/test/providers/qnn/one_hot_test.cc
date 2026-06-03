// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that builds a single OneHot node with three inputs:
//   input[0]: indices  (IndicesType)
//   input[1]: depth    (int64 or int32 constant initializer)
//   input[2]: values   (ValuesType constant initializer, 2 elements: [off_value, on_value])
template <typename IndicesType = int64_t, typename ValuesType = float>
static GetTestModelFn BuildOneHotTestCase(
    const TestInputDef<IndicesType>& indices_def,
    const TestInputDef<int64_t>& depth_def,
    const TestInputDef<ValuesType>& values_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [indices_def, depth_def, values_def, attrs](ModelTestBuilder& builder) {
    MakeTestInput<IndicesType>(builder, "indices", indices_def);
    MakeTestInput<int64_t>(builder, "depth", depth_def);
    MakeTestInput<ValuesType>(builder, "values", values_def);

    builder.MakeOutput("Y");
    builder.AddNode("OneHot_node", "OneHot", {"indices", "depth", "values"}, {"Y"}, "", attrs);
  };
}

// Returns a function that builds a QDQ OneHot node.
// Only the output is quantized (indices is int, depth/values are constants).
template <typename QuantType>
static GetTestQDQModelFn<QuantType> BuildQDQOneHotTestCase(
    const TestInputDef<int64_t>& indices_def,
    const TestInputDef<int64_t>& depth_def,
    const TestInputDef<float>& values_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    bool use_contrib_qdq = false) {
  return [indices_def, depth_def, values_def, attrs, use_contrib_qdq](
             ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    MakeTestInput<int64_t>(builder, "indices", indices_def);
    MakeTestInput<int64_t>(builder, "depth", depth_def);
    MakeTestInput<float>(builder, "values", values_def);

    const std::string one_hot_out = "one_hot_out";
    builder.AddNode("OneHot_node", "OneHot", {"indices", "depth", "values"}, {one_hot_out}, "", attrs);

    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out", one_hot_out, output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Runs a float32 OneHot model on the given QNN backend and compares output to CPU EP.
template <typename IndicesType = int64_t, typename ValuesType = float>
static void RunOneHotTest(
    const TestInputDef<IndicesType>& indices_def,
    const TestInputDef<int64_t>& depth_def,
    const TestInputDef<ValuesType>& values_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    const std::string& backend_name,
    ExpectedEPNodeAssignment expected_ep_assignment,
    int opset = 11,
    float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOneHotTestCase<IndicesType, ValuesType>(indices_def, depth_def, values_def, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Runs a QDQ OneHot model on HTP and checks accuracy against a float32 reference.
template <typename QuantType>
static void RunQDQOneHotTest(
    const TestInputDef<int64_t>& indices_def,
    const TestInputDef<int64_t>& depth_def,
    const TestInputDef<float>& values_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    ExpectedEPNodeAssignment expected_ep_assignment,
    int opset = 11,
    bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_fn = BuildOneHotTestCase<int64_t, float>(indices_def, depth_def, values_def, attrs);
  auto qdq_model_fn = BuildQDQOneHotTestCase<QuantType>(indices_def, depth_def, values_def, attrs, use_contrib_qdq);

  TestQDQModelAccuracy<QuantType>(f32_model_fn, qdq_model_fn, provider_options, opset, expected_ep_assignment);
}

//
// CPU backend tests
//

TEST_F(QnnCPUBackendTests, OneHot_FP32_Axis_Default) {
  RunOneHotTest(
      TestInputDef<int64_t>({4}, false, {0, 2, 1, 3}),
      TestInputDef<int64_t>({1}, true, {5}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {},
      "cpu",
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, OneHot_FP32_Axis0) {
  RunOneHotTest(
      TestInputDef<int64_t>({4}, false, {0, 2, 1, 3}),
      TestInputDef<int64_t>({1}, true, {5}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
      "cpu",
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, OneHot_FP32_2DIndices) {
  RunOneHotTest(
      TestInputDef<int64_t>({2, 3}, false, {0, 1, 2, 1, 0, 2}),
      TestInputDef<int64_t>({1}, true, {4}),
      TestInputDef<float>({2}, true, {-1.0f, 1.0f}),
      {},
      "cpu",
      ExpectedEPNodeAssignment::All);
}

// depth must be a constant initializer — dynamic depth is not supported.
TEST_F(QnnCPUBackendTests, OneHot_DynamicDepth_Unsupported) {
  RunOneHotTest(
      TestInputDef<int64_t>({4}, false, {0, 1, 2, 3}),
      TestInputDef<int64_t>({1}, false, {5}),  // false = dynamic (graph input)
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {},
      "cpu",
      ExpectedEPNodeAssignment::None);
}

//
// HTP and GPU backend tests — run on Linux, ARM64 Windows
//

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, OneHot_FP32_Axis_Default) {
  RunOneHotTest(
      TestInputDef<int64_t>({4}, false, {0, 2, 1, 3}),
      TestInputDef<int64_t>({1}, true, {5}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {},
      "htp",
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, OneHot_FP32_Axis0) {
  RunOneHotTest(
      TestInputDef<int64_t>({4}, false, {0, 2, 1, 3}),
      TestInputDef<int64_t>({1}, true, {5}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
      "htp",
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, OneHot_FP32_NegativeAxis) {
  RunOneHotTest(
      TestInputDef<int64_t>({2, 3}, false, {0, 1, 2, 1, 0, 2}),
      TestInputDef<int64_t>({1}, true, {4}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
      "htp",
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, OneHot_FP32_2DIndices) {
  RunOneHotTest(
      TestInputDef<int64_t>({2, 3}, false, {0, 1, 2, 1, 0, 2}),
      TestInputDef<int64_t>({1}, true, {4}),
      TestInputDef<float>({2}, true, {-1.0f, 1.0f}),
      {},
      "htp",
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, OneHot_FP32_LargeDepth) {
  RunOneHotTest(
      TestInputDef<int64_t>({8}, false, {0, 1, 2, 3, 4, 5, 6, 7}),
      TestInputDef<int64_t>({1}, true, {32}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {},
      "htp",
      ExpectedEPNodeAssignment::All);
}

// QDQ uint8 test
TEST_F(QnnHTPBackendTests, OneHot_QDQ_U8) {
  RunQDQOneHotTest<uint8_t>(
      TestInputDef<int64_t>({4}, false, {0, 2, 1, 3}),
      TestInputDef<int64_t>({1}, true, {5}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {},
      ExpectedEPNodeAssignment::All);
}

// QDQ uint16 test — uses com.microsoft Q/DQ ops since ONNX QuantizeLinear
// only supports uint16 zero-points from opset 21.
TEST_F(QnnHTPBackendTests, OneHot_QDQ_U16) {
  RunQDQOneHotTest<uint16_t>(
      TestInputDef<int64_t>({4}, false, {0, 2, 1, 3}),
      TestInputDef<int64_t>({1}, true, {5}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {},
      ExpectedEPNodeAssignment::All,
      11,
      true);  // use_contrib_qdq
}

// QDQ with axis attribute
TEST_F(QnnHTPBackendTests, OneHot_QDQ_U8_Axis1) {
  RunQDQOneHotTest<uint8_t>(
      TestInputDef<int64_t>({2, 3}, false, {0, 1, 2, 1, 0, 2}),
      TestInputDef<int64_t>({1}, true, {4}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {test::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// BFloat16 and GPU tests — ARM64 Windows only
//

#if defined(__aarch64__) || defined(_M_ARM64)

static void RunOneHotHTPBF16Test(
    const TestInputDef<int64_t>& indices_def,
    const TestInputDef<int64_t>& depth_def,
    const TestInputDef<float>& values_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    ExpectedEPNodeAssignment expected_ep_assignment,
    float tolerance = 0.008f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["htp_bf16_enable"] = "1";
  provider_options["soc_model"] = "88";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOneHotTestCase<int64_t, float>(indices_def, depth_def, values_def, attrs),
                  provider_options,
                  11,
                  expected_ep_assignment,
                  tolerance);
}

TEST_F(QnnHTPBackendTests, OneHot_BF16_Axis_Default) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunOneHotHTPBF16Test(
      TestInputDef<int64_t>({4}, false, {0, 2, 1, 3}),
      TestInputDef<int64_t>({1}, true, {5}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {},
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, OneHot_BF16_Axis0) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V79);
  RunOneHotHTPBF16Test(
      TestInputDef<int64_t>({4}, false, {0, 2, 1, 3}),
      TestInputDef<int64_t>({1}, true, {5}),
      TestInputDef<float>({2}, true, {0.0f, 1.0f}),
      {test::MakeAttribute("axis", static_cast<int64_t>(0))},
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
