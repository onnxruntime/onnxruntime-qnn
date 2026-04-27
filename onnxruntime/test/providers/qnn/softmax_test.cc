// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Regression: Softmax(axis=1) fed by a MatMul+Add that ORT's MatMulAddFusion rewrites to Gemm.
// The Softmax OpBuilder's transpose-insertion path must register a tensor wrapper for its
// input during GetCapability validation, since upstream node groups do not populate the
// shared QnnModelWrapper's tensor map in the validate phase. Without that, the op falls
// to CPU and fragments the graph.
namespace {
GetTestModelFn BuildMatMulAddSoftmaxNonLastAxisTestCase(int64_t K, int64_t N) {
  return [K, N](ModelTestBuilder& builder) {
    const std::vector<int64_t> input_shape = {1, 2, K};
    const std::vector<int64_t> weight_shape = {K, N};
    const std::vector<int64_t> bias_shape = {N};

    builder.MakeInput<float>("X", input_shape, -1.0f, 1.0f);
    builder.MakeInitializer<float>("W", weight_shape, -1.0f, 1.0f);
    builder.MakeInitializer<float>("B", bias_shape, -1.0f, 1.0f);

    builder.AddNode("node_MatMul", "MatMul", {"X", "W"}, {"val_mm"});
    builder.AddNode("node_linear_17", "Add", {"val_mm", "B"}, {"linear_17"});
    builder.AddNode("node_softmax", "Softmax", {"linear_17"}, {"softmax_out"},
                    /*domain=*/"",
                    {test::MakeAttribute("axis", static_cast<int64_t>(1))});

    builder.MakeOutput("softmax_out");
  };
}
}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

template <typename InputQType = uint8_t>
static void RunQDQOpTest(const std::string& op_type,
                         const std::vector<TestInputDef<float>>& input_defs,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         int opset_version,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         const std::string& op_domain = kOnnxDomain,
                         bool use_contrib_qdq = false,
                         QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain),
                       BuildQDQOpTestCase<InputQType>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain, use_contrib_qdq),
                       provider_options,
                       opset_version,
                       expected_ep_assignment,
                       tolerance);
}

TEST_F(QnnHTPBackendTests, Softmax13DefaultAxis) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Softmax13DefaultAxisU16) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint16_t>("Softmax",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// QNN EP wraps Softmax with transposes when axis != rank-1.
TEST_F(QnnHTPBackendTests, Softmax13NonLastAxis) {
  const std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 100.0f, 110.0f, 120.0f,
                                         1.0856307f, 0.99734545f, 0.2829785f, 1.5062947f, 0.5786002f, 1.6514366f,
                                         2.4266791f, 0.42891264f, 1.2659363f};
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3, 3}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Partner-model shape.
TEST_F(QnnHTPBackendTests, Softmax13NonLastAxisLargeInput) {
  const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 124, 1}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Softmax13NonLastAxisLargeInputU16) {
  const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
  RunQDQOpTest<uint16_t>("Softmax",
                         {TestInputDef<float>({1, 124, 1}, false, input_data)},
                         {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

TEST_F(QnnHTPBackendTests, Softmax11DefaultAxis) {
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Softmax11LastAxis) {
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        11,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogSoftmax13DefaultAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogSoftmax13NonLastAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogSoftmax11DefaultAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogSoftmax11LastAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        11,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Softmax13NonLastAxisAfterMatMulAddFusion) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildMatMulAddSoftmaxNonLastAxisTestCase(/*K=*/128, /*N=*/1),
                  provider_options,
                  /*opset=*/18,
                  ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
