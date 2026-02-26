// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.
//
// Tests for miscellaneous Add/Convert/quantization patterns that don't belong
// in the per-arithmetic-op files:
//   - QuantAccuracyTest: verifies Q/DQ accuracy with a Transpose model
//   - Add_U8_U16_Convert: mixed-precision (u8->u16) Add via QNN Convert
//   - DQ_Q_ConvertFusion_SameType: DQ->Q fusion into QNN Convert at graph output
//   - RandomUniformLikeAddTest: RandomUniformLike + Add assignment test

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/node_attr_utils.h"
#include "core/graph/graph.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Verifies that a graph input -> Q -> Transpose -> DQ -> output is handled
// accurately.  QNN optimizes away the quantization on graph inputs (perfect
// accuracy), while CPU EP actually quantizes/dequantizes, leading to a small
// difference — this test confirms we handle that correctly.
TEST_F(QnnHTPBackendTests, QuantAccuracyTest) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto builder_func = [](ModelTestBuilder& builder) {
    const TestInputDef<float> input0_def({1, 2, 3}, false, {1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f});
    NodeArg* input0 = MakeTestInput(builder, input0_def);
    QuantParams<uint8_t> qparams = GetTestInputQuantParams<uint8_t>(input0_def);

    auto* quant_input = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input0, qparams.scale, qparams.zero_point, quant_input);

    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("Transpose", {quant_input}, {op_output});

    NodeArg* output = builder.MakeOutput();
    builder.AddDequantizeLinearNode<uint8_t>(op_output, qparams.scale, qparams.zero_point, output);
  };

  RunQnnModelTest(builder_func,
                  provider_options,
                  13,
                  ExpectedEPNodeAssignment::All);
}

// Builds a mixed-precision Add: input0 is cast u8->u16 before Add, input1 is u16.
static GetTestQDQModelFn<uint16_t> BuildQDQConvertAddTestCase(const TestInputDef<float>& input0_def,
                                                              const TestInputDef<float>& input1_def) {
  return [input0_def, input1_def](ModelTestBuilder& builder, std::vector<QuantParams<uint16_t>>& output_qparams) {
    constexpr bool use_contrib_qdq = true;

    NodeArg* input0 = MakeTestInput<float>(builder, input0_def);
    QuantParams<uint8_t> input0_u8_qparams = GetTestInputQuantParams<uint8_t>(input0_def);
    NodeArg* input0_after_qdq = AddQDQNodePair<uint8_t>(builder, input0, input0_u8_qparams.scale,
                                                        input0_u8_qparams.zero_point, use_contrib_qdq);

    QuantParams<uint16_t> input0_u16_qparams = GetTestInputQuantParams<uint16_t>(input0_def);
    NodeArg* input0_after_convert = AddQDQNodePair<uint16_t>(builder, input0_after_qdq, input0_u16_qparams.scale,
                                                             input0_u16_qparams.zero_point, use_contrib_qdq);

    NodeArg* input1 = MakeTestInput<float>(builder, input1_def);
    QuantParams<uint16_t> input1_qparams = GetTestInputQuantParams<uint16_t>(input1_def);
    NodeArg* input1_after_qdq = AddQDQNodePair<uint16_t>(builder, input1, input1_qparams.scale,
                                                         input1_qparams.zero_point, use_contrib_qdq);

    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("Add", {input0_after_convert, input1_after_qdq}, {op_output});

    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, op_output, output_qparams[0].scale,
                                                    output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Test quantization type conversion (mixed precision) with Add.
// First input is converted from uint8_t to uint16_t.
TEST_F(QnnHTPBackendTests, Add_U8_U16_Convert) {
  std::vector<float> input0_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  std::vector<float> input1_data = GetFloatDataInRange(-20.0f, 20.0f, 8);
  TestInputDef<float> input0_def({1, 2, 2, 2}, false, input0_data);
  TestInputDef<float> input1_def({1, 2, 2, 2}, false, input1_data);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>("Add", {input0_def, input1_def}, {}, {}, kOnnxDomain),
                       BuildQDQConvertAddTestCase(input0_def, input1_def),
                       provider_options,
                       18,
                       ExpectedEPNodeAssignment::All);
}

// Builds a graph where a (DQ -> Q) sequence at the graph's output is fused into a QNN Convert op.
// ONNX Graph: DQ -> Add -> Q -> DQ -> Q -> graph_output
// QNN Graph:  DQ -> Add -> Q -> Convert -> graph_output
template <typename InQuantType, typename OutQuantType>
static GetTestModelFn BuildDQQConvertAtOutputTestCase(const TestInputDef<float>& input0_def,
                                                      const TestInputDef<float>& input1_def,
                                                      const QuantParams<OutQuantType>& output_qparams) {
  return [input0_def, input1_def, output_qparams](ModelTestBuilder& builder) {
    NodeArg* input0 = MakeTestInput<float>(builder, input0_def);
    QuantParams<InQuantType> input0_qparams = GetTestInputQuantParams<InQuantType>(input0_def);
    NodeArg* input0_after_qdq = AddQDQNodePair<InQuantType>(builder, input0, input0_qparams.scale,
                                                            input0_qparams.zero_point);

    NodeArg* input1 = MakeTestInput<float>(builder, input1_def);
    QuantParams<InQuantType> input1_qparams = GetTestInputQuantParams<InQuantType>(input1_def);
    NodeArg* input1_after_qdq = AddQDQNodePair<InQuantType>(builder, input1, input1_qparams.scale,
                                                            input1_qparams.zero_point);

    auto* op_output = builder.MakeIntermediate();
    builder.AddNode("Add", {input0_after_qdq, input1_after_qdq}, {op_output});

    QuantParams<InQuantType> add_out_qparams = ConvertQuantParams<OutQuantType, InQuantType>(output_qparams);
    add_out_qparams.scale *= 1.01f;  // Slightly different so DQ->Q are not optimized out.
    NodeArg* add_out_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<InQuantType>(op_output, add_out_qparams.scale,
                                               add_out_qparams.zero_point, add_out_q);
    NodeArg* add_out_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<InQuantType>(add_out_q, add_out_qparams.scale,
                                                 add_out_qparams.zero_point, add_out_dq);

    NodeArg* q_conv_out = builder.MakeOutput();
    builder.AddQuantizeLinearNode<OutQuantType>(add_out_dq, output_qparams.scale, output_qparams.zero_point,
                                                q_conv_out);
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

  RunQnnModelTest(BuildDQQConvertAtOutputTestCase<uint8_t, uint8_t>(input0_def, input1_def, out_qparams_u8),
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All);

  RunQnnModelTest(BuildDQQConvertAtOutputTestCase<uint16_t, uint16_t>(input0_def, input1_def, out_qparams_u16),
                  provider_options,
                  21,
                  ExpectedEPNodeAssignment::All);
}

// Test RandomUniformLike + Add: both ops should be assigned to QNN EP.
// Outputs are not verified because RandomUniformLike's randomness algorithm differs from ORT CPU.
TEST_F(QnnHTPBackendTests, RandomUniformLikeAddTest) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    auto* input = builder.MakeInput<float>({1, 4, 3}, input_data);

    auto* random_output = builder.MakeIntermediate();
    Node& random_node = builder.AddNode("RandomUniformLike", {input}, {random_output});
    random_node.AddAttribute("low", 0.0f);
    random_node.AddAttribute("high", 10.0f);
    random_node.AddAttribute("seed", 42.0f);

    auto* final_output = builder.MakeOutput();
    builder.AddNode("Add", {input, random_output}, {final_output});
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnModelTest(build_test_case,
                  provider_options,
                  14,
                  ExpectedEPNodeAssignment::All,
                  1e-5f,
                  logging::Severity::kERROR,
                  false);  // Don't verify outputs (non-deterministic)
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
