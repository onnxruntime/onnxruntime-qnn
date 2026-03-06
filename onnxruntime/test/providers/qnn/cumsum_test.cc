// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Runs a non-QDQ model on HTP and compares output to CPU EP.
template <typename InputType1 = float, typename InputType2 = float>
static void RunCumSumOpTest(const std::string& op_type,
                            const TestInputDef<InputType1>& input_def_1,
                            const TestInputDef<InputType2>& input_def_2,
                            const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                            int opset_version,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            float fp32_abs_err = 2e-3f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  // Runs model with a Q/DQ binary op and compares the outputs of the CPU and QNN EPs.
  RunQnnModelTest(BuildOpTestCase<InputType1, InputType2>("CumSum_node", op_type, {input_def_1}, {input_def_2}, attrs),
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Non-QDQ model, CumSum with float input and axis input as initializer with axis 0
TEST_F(QnnHTPBackendTests, CumSum_float_int32_e0_r0_axis_0) {
  RunCumSumOpTest<float, int32_t>("CumSum",
                                  TestInputDef<float>({3, 2}, false, {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f}),
                                  TestInputDef<int32_t>({}, true, {0}),
                                  {test::MakeAttribute("exclusive", static_cast<int64_t>(0)),
                                   test::MakeAttribute("reverse", static_cast<int64_t>(0))},
                                  17,
                                  ExpectedEPNodeAssignment::All);
}

// Non-QDQ model, CumSum with float input and axis input as initializer with axis -1
TEST_F(QnnHTPBackendTests, CumSum_float_int32_e0_r0_axis_neg1) {
  RunCumSumOpTest<float, int32_t>("CumSum",
                                  TestInputDef<float>({3, 2}, false, {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f}),
                                  TestInputDef<int32_t>({}, true, {-1}),
                                  {test::MakeAttribute("exclusive", static_cast<int64_t>(0)),
                                   test::MakeAttribute("reverse", static_cast<int64_t>(0))},
                                  17,
                                  ExpectedEPNodeAssignment::All);
}

// Test int64 axis
TEST_F(QnnHTPBackendTests, CumSum_float_int64_e0_r0_axis_1) {
  RunCumSumOpTest<float, int64_t>("CumSum",
                                  TestInputDef<float>({3, 2}, false, {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f}),
                                  TestInputDef<int64_t>({}, true, {1}),
                                  {test::MakeAttribute("exclusive", static_cast<int64_t>(0)),
                                   test::MakeAttribute("reverse", static_cast<int64_t>(0))},
                                  17,
                                  ExpectedEPNodeAssignment::All);
}

// Returns a function that creates a graph with a QDQ CumSum operator.
template <typename QuantType, typename AxisType>
GetTestQDQModelFn<QuantType> BuildQDQCumSumTestCase(const TestInputDef<float>& input_def,
                                                    const TestInputDef<AxisType>& axis_def,
                                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                    bool use_contrib_qdq = false) {
  return [input_def, axis_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                                       std::vector<QuantParams<QuantType>>& output_qparams) {
    ORT_UNUSED_PARAMETER(use_contrib_qdq);  // Build using standard ONNX Q/DQ nodes.

    builder.graph_->set_name("qdq_cumsum_graph");

    // Input (fp32).
    MakeTestInput(builder, "X", input_def);

    // Input Q/DQ: X -> Q -> DQ -> X_qdq
    const QuantParams<QuantType> input_qparam = GetTestInputQuantParams<QuantType>(input_def);

    auto x_qdq = AddQDQNodePair<QuantType>(builder, "X_qdq", "X", input_qparam.scale, input_qparam.zero_point);

    // Axis input.
    MakeTestInput(builder, "axis", axis_def);

    // CumSum op.
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes = attrs;
    builder.AddNode(
        "cumsum",
        "CumSum",
        {x_qdq, "axis"},
        {"cumsum_out"},
        "",
        attributes);

    // Output Q/DQ: cumsum_out -> Q -> DQ -> Y
    auto qdq_out = AddQDQNodePair<QuantType>(builder, "cumsum_out_qdq", "cumsum_out", output_qparams[0].scale, output_qparams[0].zero_point);

    builder.MakeOutput(qdq_out);
  };
}

// Test the accuracy of a QDQ CumSum model on QNN EP. Checks if the QDQ model on QNN EP is as accurate as the QDQ model on CPU EP
// (compared to float32 model).
template <typename QuantType, typename AxisType>
static void RunQDQCumSumOpTest(const TestInputDef<float>& input_def,
                               const TestInputDef<AxisType>& axis_def,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                               int opset,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_model_builder = BuildOpTestCase<float, AxisType>("CumSum_node", "CumSum", {input_def}, {axis_def}, attrs);
  auto qdq_model_builder = BuildQDQCumSumTestCase<QuantType, AxisType>(input_def, axis_def, attrs,
                                                                       use_contrib_qdq);

  TestQDQModelAccuracy<QuantType>(f32_model_builder,
                                  qdq_model_builder,
                                  provider_options,
                                  opset,
                                  expected_ep_assignment);
}

// Test creates a DQ -> CumSum -> Q -> DQ graph, and checks that all
// nodes are supported by the QNN EP, and that the inference results are as accurate as CPU EP.
//
// QDQ model, CumSum with uint8 input and axis input as initializer
TEST_F(QnnHTPBackendTests, CumSum_uint8_int32_e0_r0) {
  RunQDQCumSumOpTest<uint8_t, int32_t>(TestInputDef<float>({3, 2}, false, {1.3f, 7.2f, 0.4f, 3.4f, 5.7f, 0.8f}),
                                       TestInputDef<int32_t>({}, true, {0}),
                                       {test::MakeAttribute("exclusive", static_cast<int64_t>(0)),
                                        test::MakeAttribute("reverse", static_cast<int64_t>(0))},
                                       17,
                                       ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
