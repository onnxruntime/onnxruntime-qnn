// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Verifies that standalone Q and DQ nodes used around a Transpose produce bit-exact output
// on QNN EP. QNN keeps the data in quantized form through the Transpose, so no precision is
// lost. ORT CPU EP actually dequantizes, transposes in float, then re-quantizes, which can
// introduce rounding differences — hence the test uses RunQnnModelTest (QNN vs CPU) rather
// than TestQDQModelAccuracy.
TEST_F(QnnHTPBackendTests, QuantAccuracyTest) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto builder_func = [](ModelTestBuilder& builder) {
    const TestInputDef<float> input0_def({1, 2, 3}, false, {1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f});

    // input -> Q -> Transpose -> DQ -> output
    MakeTestInput<float>(builder, "input0", input0_def);
    const QuantParams<uint8_t> qparams = GetTestInputQuantParams<uint8_t>(input0_def);

    builder.AddQuantizeLinearNode<uint8_t>("q_in", "input0", qparams.scale, qparams.zero_point, "quant_input");

    builder.AddNode("Transpose",
                    "Transpose",
                    {"quant_input"},
                    {"op_output"});

    builder.MakeOutput("Y");
    builder.AddDequantizeLinearNode<uint8_t>("dq_out", "op_output", qparams.scale, qparams.zero_point, "Y");
  };

  RunQnnModelTest(builder_func,
                  provider_options,
                  13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

// Builds a graph where a (DQ -> Q) sequence at the graph's output is fuse into a QNN Convert operator.
// ONNX Graph: DQ -> Add -> Q -> DQ -> Q -> graph_output
// QNN Graph:  DQ -> Add -> Q -> Convert -> graph_output
template <typename InQuantType, typename OutQuantType>
static GetTestModelFn BuildDQQConvertAtOutputTestCase(const TestInputDef<float>& input0_def,
                                                      const TestInputDef<float>& input1_def,
                                                      const QuantParams<OutQuantType>& output_qparams) {
  return [input0_def, input1_def, output_qparams](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input0", input0_def);
    MakeTestInput<float>(builder, "input1", input1_def);

    // Input0 -> Quantize(InQuantType) -> Dequantize(InQuantType to float) -> input0_after_qdq
    const QuantParams<InQuantType> input0_qparams = GetTestInputQuantParams<InQuantType>(input0_def);
    const std::string input0_after_qdq =
        AddQDQNodePair<InQuantType>(builder, "qdq0", "input0", input0_qparams.scale, input0_qparams.zero_point);

    // Input1 -> Quantize(InQuantType) -> Dequantize(InQuantType to float) -> input1_after_qdq
    const QuantParams<InQuantType> input1_qparams = GetTestInputQuantParams<InQuantType>(input1_def);
    const std::string input1_after_qdq =
        AddQDQNodePair<InQuantType>(builder, "qdq1", "input1", input1_qparams.scale, input1_qparams.zero_point);

    // Add op -> add_out
    builder.AddNode("Add",
                    "Add",
                    {input0_after_qdq, input1_after_qdq},
                    {"add_out"});

    // op_output -> Quantize(InQuantType) -> add_out_q
    QuantParams<InQuantType> add_out_qparams = ConvertQuantParams<OutQuantType, InQuantType>(output_qparams);
    add_out_qparams.scale *= 1.01f;  // Make qparams slightly different so DQ->Q are not optimized out.

    auto add_qdq_name = AddQDQNodePair(builder, "add_qdq", "add_out", add_out_qparams.scale, add_out_qparams.zero_point);

    // Add a Q to quantize to OutQuantType.
    // The previous DQ and this Q will be fused into a QNN Convert.
    builder.MakeOutput("Y");
    builder.AddQuantizeLinearNode<OutQuantType>("final_q", add_qdq_name, output_qparams.scale, output_qparams.zero_point, "Y");
  };
}

// Test fusion of (DQ -> Q) into QNN's Convert op using the same quant type.
TEST_F(QnnHTPBackendTests, DQ_Q_ConvertFusion_SameType) {
  std::vector<float> input0_data = {-8.0f, -6.0, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  std::vector<float> input1_data = {-8.0f, -6.0, -2.0f, 0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
  TestInputDef<float> input0_def({1, 2, 2, 2}, false, input0_data);
  TestInputDef<float> input1_def({1, 2, 2, 2}, false, input1_data);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  QuantParams<uint8_t> out_qparams_u8 = {1.0f, 128};
  QuantParams<uint16_t> out_qparams_u16 = {1.0f, 32768};

  // QNN Convert op converts uint8 to uint8 at the graph output. Slightly different scale values.
  RunQnnModelTest(BuildDQQConvertAtOutputTestCase<uint8_t, uint8_t>(input0_def, input1_def, out_qparams_u8),
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});

  // QNN Convert op converts uint16 to uint16 at the graph output. Slightly different scale values.
  RunQnnModelTest(BuildDQQConvertAtOutputTestCase<uint16_t, uint16_t>(input0_def, input1_def, out_qparams_u16),
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

// Standalone per-channel Quantize/Dequantize support on the HTP backend.
//
// QnnQuantizeOpBuilder::ValidateQdqNode rejects a standalone per-channel Q/DQ node when its
// data input is NOT a compile-time constant (graph input / activation), because QNN has no
// per-channel encoding for such a tensor. When the data input IS constant, the node is instead
// constant-folded into a STATIC tensor and stays on QNN EP. The tests below cover both branches
// for Q and DQ.

// Per-channel DQ on a NON-constant (graph) input must be rejected -> whole graph falls back to CPU.
TEST_F(QnnHTPBackendTests, StandalonePerChannelDequantizeLinear_NonConstInput_Unsupported) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    // Quantized activation as a graph input (non-constant) with shape [2, 3].
    auto* input = builder.MakeInput<uint8_t>("input", {2, 3}, static_cast<uint8_t>(0), static_cast<uint8_t>(255));

    // Per-channel scales/zero-points along axis 0 (3 columns per row -> 2 channels).
    const std::vector<float> scales = {0.1f, 0.2f};
    const std::vector<uint8_t> zero_points = {0, 0};

    builder.AddDequantizeLinearNode<uint8_t>("dq", input->name(), scales, zero_points, "output",
                                             {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
    builder.MakeOutput("output");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // Per-channel standalone DQ on a non-constant input is unsupported: expect no nodes on QNN EP.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// Per-channel Q on a NON-constant (graph) input must be rejected -> whole graph falls back to CPU.
TEST_F(QnnHTPBackendTests, StandalonePerChannelQuantizeLinear_NonConstInput_Unsupported) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    // Float activation as a graph input (non-constant) with shape [2, 3].
    auto* input = builder.MakeInput<float>("input", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Per-channel scales/zero-points along axis 0 (2 channels).
    const std::vector<float> scales = {0.1f, 0.2f};
    const std::vector<uint8_t> zero_points = {0, 0};

    builder.AddQuantizeLinearNode<uint8_t>("q", input->name(), scales, zero_points, "output",
                                           {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
    builder.MakeOutput("output");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // Per-channel standalone Q on a non-constant input is unsupported: expect no nodes on QNN EP.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// Per-channel DQ on a CONSTANT input is constant-folded into a STATIC tensor and stays on QNN EP.
TEST_F(QnnHTPBackendTests, StandalonePerChannelDequantizeLinear_ConstInput_Folded) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    // Quantized weights as a constant initializer with shape [2, 3].
    const std::vector<uint8_t> weights = {10, 20, 30, 40, 50, 60};
    auto* weights_init = builder.MakeInitializer<uint8_t>("weights", {2, 3}, weights);

    // Per-channel scales/zero-points along axis 0 (2 channels).
    const std::vector<float> scales = {0.1f, 0.2f};
    const std::vector<uint8_t> zero_points = {0, 0};

    // DQ(const) -> Mul(graph input) so the DQ output feeds a supported op rather than a graph output.
    builder.MakeInput<float>("activation", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    builder.AddDequantizeLinearNode<uint8_t>("dq", weights_init->name(), scales, zero_points, "dq_out",
                                             {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
    builder.AddNode("Mul", "Mul", {"dq_out", "activation"}, {"output"});
    builder.MakeOutput("output");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // Constant per-channel DQ is folded to a STATIC tensor; the whole graph stays on QNN EP.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

// Per-channel Q on a CONSTANT input is constant-folded into a STATIC tensor and stays on QNN EP.
TEST_F(QnnHTPBackendTests, StandalonePerChannelQuantizeLinear_ConstInput_Folded) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    // Float weights as a constant initializer with shape [2, 3].
    const std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto* weights_init = builder.MakeInitializer<float>("weights", {2, 3}, weights);

    // Per-channel scales/zero-points along axis 0 (2 channels).
    const std::vector<float> scales = {0.1f, 0.2f};
    const std::vector<uint8_t> zero_points = {0, 0};

    // Q(const) -> DQ -> Mul(graph input) so the quantized output feeds a supported op
    // and the graph output remains float (comparable between QNN EP and CPU EP).
    builder.MakeInput<float>("activation", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    builder.AddQuantizeLinearNode<uint8_t>("q", weights_init->name(), scales, zero_points, "q_out",
                                           {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
    builder.AddDequantizeLinearNode<uint8_t>("dq", "q_out", scales, zero_points, "dq_out",
                                             {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
    builder.AddNode("Mul", "Mul", {"dq_out", "activation"}, {"output"});
    builder.MakeOutput("output");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // Constant per-channel Q is folded to a STATIC tensor; the whole graph stays on QNN EP.
  RunQnnModelTest(build_test_case,
                  provider_options,
                  21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
