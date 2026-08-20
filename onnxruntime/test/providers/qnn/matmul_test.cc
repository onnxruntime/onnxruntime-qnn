// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_test_utils.h"
#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Returns a function that creates a graph with MatMul operator.
static GetTestModelFn BuildMatMulOpTestCase(const TestInputDef<float>& input1_def,
                                            const TestInputDef<float>& input2_def) {
  return [input1_def, input2_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input0", input1_def);
    MakeTestInput<float>(builder, "input1", input2_def);

    builder.MakeOutput("Y");

    builder.AddNode("MatMul",
                    "MatMul",
                    {"input0", "input1"},
                    {"Y"},
                    kOnnxDomain);
  };
}

static void RunMatMulOpTest(const std::vector<int64_t>& shape_0,
                            const std::vector<int64_t>& shape_1, bool is_initializer_0, bool is_initializer_1,
                            ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                            const std::string& backend_name = "cpu",
                            int opset = 18, float f32_abs_err = 1e-4f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildMatMulOpTestCase(
                      TestInputDef<float>(shape_0, is_initializer_0, GetSequentialFloatData(shape_0, 0.01f, 0.02f)),
                      TestInputDef<float>(shape_1, is_initializer_1, GetSequentialFloatData(shape_1, 0.02f, 0.02f))),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(f32_abs_err)});
}

// Returns a function that creates a graph with a QDQ MatMul operator.
template <typename Input0QType, typename Input1QType, typename OutputQType>
static GetTestQDQModelFn<OutputQType> BuildMatMulOpQDQTestCase(const TestInputDef<float>& input0_def,
                                                               const TestInputDef<float>& input1_def,
                                                               bool use_contrib_qdq) {
  return [input0_def, input1_def, use_contrib_qdq](ModelTestBuilder& builder,
                                                   std::vector<QuantParams<OutputQType>>& output_qparams) {
    // inputs
    MakeTestInput<float>(builder, "input0", input0_def);
    MakeTestInput<float>(builder, "input1", input1_def);

    // input0 -> Q -> DQ -> input0_qdq
    const QuantParams<Input0QType> input0_qparams = GetTestInputQuantParams<Input0QType>(input0_def);
    const std::string input0_qdq =
        AddQDQNodePair<Input0QType>(builder, "qdq_in0", "input0",
                                    input0_qparams.scale, input0_qparams.zero_point, use_contrib_qdq);

    // input1 -> Q -> DQ -> input1_qdq
    const QuantParams<Input1QType> input1_qparams = GetTestInputQuantParams<Input1QType>(input1_def);
    const std::string input1_qdq =
        AddQDQNodePair<Input1QType>(builder, "qdq_in1", "input1",
                                    input1_qparams.scale, input1_qparams.zero_point, use_contrib_qdq);

    // MatMul -> Y
    builder.AddNode("MatMul",
                    "MatMul",
                    {input0_qdq, input1_qdq},
                    {"Y"},
                    kOnnxDomain);

    // Y -> Q -> DQ -> (graph output)
    AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(builder,
                                                       "qdq_out",
                                                       "Y",
                                                       output_qparams[0].scale,
                                                       output_qparams[0].zero_point,
                                                       use_contrib_qdq);
  };
}

/// Returns a function that creates a graph with a per-channel (weights) QDQ MatMul operator.
template <typename Input0QType, typename WeightQType, typename OutputQType>
static GetTestQDQModelFn<OutputQType> BuildQDQPerChannelMatMulTestCase(const TestInputDef<float>& input_def,
                                                                       const TestInputDef<float>& weights_def,
                                                                       int64_t weight_quant_axis,
                                                                       bool use_contrib_qdq = false) {
  return [input_def, weights_def, weight_quant_axis, use_contrib_qdq](
             ModelTestBuilder& builder, std::vector<QuantParams<OutputQType>>& output_qparams) {
    QNN_ASSERT(weights_def.IsInitializer() && weights_def.IsRawData());

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // input -> Q/DQ -> input_qdq
    const QuantParams<Input0QType> input_qparams = GetTestInputQuantParams<Input0QType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<Input0QType>(builder, "qdq_in", "input",
                                    input_qparams.scale, input_qparams.zero_point, use_contrib_qdq);

    // Quantized(weights) -> DQ ->
    auto weight_shape = weights_def.GetShape();
    std::vector<float> weight_scales;
    std::vector<WeightQType> weight_zero_points;
    int64_t pos_weight_quant_axis = weight_quant_axis;
    if (pos_weight_quant_axis < 0) {
      pos_weight_quant_axis += static_cast<int64_t>(weight_shape.size());
    }

    GetTestInputQuantParamsPerChannel<WeightQType>(weights_def, weight_scales, weight_zero_points,
                                                   static_cast<size_t>(pos_weight_quant_axis), true);

    std::vector<WeightQType> quantized_weights;
    size_t num_weight_storage_elems = SizeOfShape(weight_shape);
    if constexpr (std::is_same_v<WeightQType, Int4x2> || std::is_same_v<WeightQType, UInt4x2>) {
      num_weight_storage_elems = Int4x2::CalcNumInt4Pairs(SizeOfShape(weight_shape));
    }
    quantized_weights.resize(num_weight_storage_elems);

    QuantizeValues<float, WeightQType>(weights_def.GetRawData(), quantized_weights, weight_shape, weight_scales,
                                       weight_zero_points, pos_weight_quant_axis);

    builder.MakeInitializer<WeightQType>("weights", weights_def.GetShape(), quantized_weights);

    // weights -> DQ -> weights_dq
    builder.AddDequantizeLinearNode<WeightQType>(
        "weights_dq",
        "weights",
        weight_scales,
        weight_zero_points,
        "weights_dq",
        {builder.MakeScalarAttribute("axis", static_cast<int64_t>(weight_quant_axis))},
        use_contrib_qdq);

    // MatMul(input_qdq, weights_dq) -> Y
    builder.AddNode("MatMul",
                    "MatMul",
                    {input_qdq, "weights_dq"},
                    {"Y"},
                    kOnnxDomain);

    // Y -> Q -> DQ -> (graph output)
    AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(builder,
                                                       "qdq_out",
                                                       "Y",
                                                       output_qparams[0].scale,
                                                       output_qparams[0].zero_point,
                                                       use_contrib_qdq);
  };
}

template <typename Input0QType, typename Input1QType, typename OutputQType>
static void RunQDQMatMulOpTest(const std::vector<int64_t>& shape_0, const std::vector<int64_t>& shape_1,
                               bool is_initializer_0, bool is_initializer_1,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               int opset = 21, bool use_contrib_qdq = false,
                               QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestInputDef<float> input0_def(
      shape_0, is_initializer_0,
      GetFloatDataInRange(-0.1f, 0.1f,
                          static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                              std::multiplies<int64_t>()))));
  TestInputDef<float> input1_def(
      shape_1, is_initializer_1,
      GetFloatDataInRange(-0.1f, 0.1f,
                          static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                              std::multiplies<int64_t>()))));

  TestQDQModelAccuracy(
      BuildMatMulOpTestCase(input0_def, input1_def),
      BuildMatMulOpQDQTestCase<Input0QType, Input1QType, OutputQType>(input0_def, input1_def, use_contrib_qdq),
      provider_options, opset, expected_ep_assignment, tolerance);
}

template <typename InputQType, typename WeightQType, typename OutputQType>
static void RunQDQPerChannelMatMulOpTest(
    const std::vector<int64_t>& shape_input, const std::vector<int64_t>& shape_weight, int64_t weight_quant_axis,
    QDQTolerance tolerance = QDQTolerance(),
    ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All, int opset = 21,
    bool use_contrib_qdq = false, bool enable_fp16_precision = true) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  if (enable_fp16_precision) {
#if defined(_WIN32)
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
#if defined(__linux__) && !defined(__aarch64__)
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
    provider_options["enable_htp_fp16_precision"] = "1";
  } else {
    provider_options["enable_htp_fp16_precision"] = "0";
  }

  TestInputDef<float> input_def(
      shape_input, false,
      GetFloatDataInRange(-0.1f, 0.1f,
                          static_cast<size_t>(std::accumulate(shape_input.begin(), shape_input.end(),
                                                              static_cast<int64_t>(1), std::multiplies<int64_t>()))));
  TestInputDef<float> weight_def(
      shape_weight, true,
      GetFloatDataInRange(-0.1f, 0.1f,
                          static_cast<size_t>(std::accumulate(shape_weight.begin(), shape_weight.end(),
                                                              static_cast<int64_t>(1), std::multiplies<int64_t>()))));

  TestQDQModelAccuracy(BuildMatMulOpTestCase(input_def, weight_def),
                       BuildQDQPerChannelMatMulTestCase<InputQType, WeightQType, OutputQType>(
                           input_def, weight_def, weight_quant_axis, use_contrib_qdq),
                       provider_options, opset, expected_ep_assignment, tolerance);
}

/// Returns a function that creates a graph with a block-quantized (BQ) weight MatMul operator.
template <typename Input0QType, typename WeightQType, typename OutputQType>
static GetTestQDQModelFn<OutputQType> BuildQDQBlockQuantMatMulTestCase(
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    int64_t block_size,
    int64_t weight_quant_axis,
    bool use_contrib_qdq = false) {
  return [input_def, weights_def, block_size, weight_quant_axis, use_contrib_qdq](
             ModelTestBuilder& builder, std::vector<QuantParams<OutputQType>>& output_qparams) {
    QNN_ASSERT(weights_def.IsInitializer() && weights_def.IsRawData());

    // input -> Q/DQ -> input_qdq
    MakeTestInput<float>(builder, "input", input_def);
    const QuantParams<Input0QType> input_qparams = GetTestInputQuantParams<Input0QType>(input_def);
    const std::string input_qdq = AddQDQNodePair<Input0QType>(
        builder, "qdq_in", "input", input_qparams.scale, input_qparams.zero_point, use_contrib_qdq);

    // Compute per-block quantization parameters (symmetric)
    const auto& weight_shape = weights_def.GetShape();
    int64_t pos_weight_quant_axis = weight_quant_axis;
    if (pos_weight_quant_axis < 0) {
      pos_weight_quant_axis += static_cast<int64_t>(weight_shape.size());
    }

    std::vector<float> weight_scales;
    std::vector<WeightQType> weight_zero_points;
    GetTestInputQuantParamsBlockQuant<WeightQType>(weights_def, weight_scales, weight_zero_points,
                                                   block_size, pos_weight_quant_axis, true);

    // Quantize weight data with per-block params
    const size_t num_weight_elems = SizeOfShape(weight_shape);
    size_t num_weight_storage_elems = num_weight_elems;
    if constexpr (std::is_same_v<WeightQType, Int4x2> || std::is_same_v<WeightQType, UInt4x2>) {
      num_weight_storage_elems = Int4x2::CalcNumInt4Pairs(num_weight_elems);
    }
    std::vector<WeightQType> quantized_weights(num_weight_storage_elems);
    QuantizeValuesBlockQuant<float, WeightQType>(
        weights_def.GetRawData(), quantized_weights, weight_shape,
        weight_scales, weight_zero_points, block_size, pos_weight_quant_axis);

    builder.MakeInitializer<WeightQType>("weights", weight_shape, quantized_weights);

    // Compute 2D scale shape: [num_blocks, non_axis_dim] for axis=0
    //                          [non_axis_dim, num_blocks] for axis=1
    const int64_t axis_dim = weight_shape[static_cast<size_t>(pos_weight_quant_axis)];
    const int64_t non_axis_dim = weight_shape[static_cast<size_t>(1 - pos_weight_quant_axis)];
    const int64_t num_blocks = (axis_dim + block_size - 1) / block_size;
    const std::vector<int64_t> scale_shape = (pos_weight_quant_axis == 0)
                                                 ? std::vector<int64_t>{num_blocks, non_axis_dim}
                                                 : std::vector<int64_t>{non_axis_dim, num_blocks};

    builder.MakeInitializer<float>("weights_scale", scale_shape, weight_scales);
    builder.MakeInitializer<WeightQType>("weights_zp", scale_shape, weight_zero_points);

    // weights -> DQ -> weights_dq (with block_size and axis attributes)
    builder.AddNode("weights_dq", "DequantizeLinear",
                    {"weights", "weights_scale", "weights_zp"},
                    {"weights_dq"},
                    "",
                    {builder.MakeScalarAttribute("axis", weight_quant_axis),
                     builder.MakeScalarAttribute("block_size", block_size)});

    // MatMul(input_qdq, weights_dq) -> Y
    builder.AddNode("MatMul", "MatMul", {input_qdq, "weights_dq"}, {"Y"}, kOnnxDomain);

    // Y -> Q -> DQ -> (graph output)
    AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(builder, "qdq_out", "Y",
                                                       output_qparams[0].scale, output_qparams[0].zero_point,
                                                       use_contrib_qdq);
  };
}

// Returns a function that creates a graph with MatMul + Add operators.
// input0 --|
//           MatMul -> Add -> output
// input1 --|
// bias ----------------|
static GetTestModelFn BuildMatMulAddOpTestCase(const TestInputDef<float>& input0_def,
                                               const TestInputDef<float>& input1_def,
                                               const TestInputDef<float>& bias_def) {
  return [input0_def, input1_def, bias_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input0", input0_def);
    MakeTestInput<float>(builder, "input1", input1_def);
    MakeTestInput<float>(builder, "bias", bias_def);

    builder.AddNode("MatMul", "MatMul", {"input0", "input1"}, {"matmul_out"}, kOnnxDomain);
    builder.AddNode("Add", "Add", {"matmul_out", "bias"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  };
}

// Returns a function that creates a QDQ MatMul + Add graph with:
//   - Per-tensor input Q->DQ
//   - Per-channel weight (DQ-only, pre-quantized initializer)
//   - Float bias (initializer, no quantization)
//   - Per-tensor output Q->DQ
//
//   input[f32] -> Q -> DQ - |
//                           MatMul -> Add -> Q -> DQ -> output[f32]
//   weight[per-ch] -> DQ -- |        /
//   bias[f32 initializer] ----------/
template <typename InputQType, typename WeightQType, typename OutputQType>
static GetTestQDQModelFn<OutputQType> BuildQDQPerChannelMatMulAddTestCase(
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    int64_t weight_quant_axis) {
  return [input_def, weights_def, bias_def, weight_quant_axis](
             ModelTestBuilder& builder, std::vector<QuantParams<OutputQType>>& output_qparams) {
    QNN_ASSERT(weights_def.IsInitializer() && weights_def.IsRawData());
    QNN_ASSERT(bias_def.IsInitializer() && bias_def.IsRawData());

    // input
    MakeTestInput<float>(builder, "input", input_def);

    // input -> Q -> DQ -> input_qdq
    const QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<InputQType>(builder, "qdq_in", "input",
                                   input_qparams.scale, input_qparams.zero_point);

    // Quantized(weights) -> DQ -> weights_dq (per-channel, pre-quantized initializer)
    auto weight_shape = weights_def.GetShape();
    std::vector<float> weight_scales;
    std::vector<WeightQType> weight_zero_points;
    int64_t pos_weight_quant_axis = weight_quant_axis;
    if (pos_weight_quant_axis < 0) {
      pos_weight_quant_axis += static_cast<int64_t>(weight_shape.size());
    }

    GetTestInputQuantParamsPerChannel<WeightQType>(weights_def, weight_scales, weight_zero_points,
                                                   static_cast<size_t>(pos_weight_quant_axis), true);

    std::vector<WeightQType> quantized_weights;
    size_t num_weight_storage_elems = SizeOfShape(weight_shape);
    if constexpr (std::is_same_v<WeightQType, Int4x2> || std::is_same_v<WeightQType, UInt4x2>) {
      num_weight_storage_elems = Int4x2::CalcNumInt4Pairs(SizeOfShape(weight_shape));
    }
    quantized_weights.resize(num_weight_storage_elems);

    QuantizeValues<float, WeightQType>(weights_def.GetRawData(), quantized_weights, weight_shape, weight_scales,
                                       weight_zero_points, pos_weight_quant_axis);

    builder.MakeInitializer<WeightQType>("weights", weights_def.GetShape(), quantized_weights);

    builder.AddDequantizeLinearNode<WeightQType>(
        "weights_dq",
        "weights",
        weight_scales,
        weight_zero_points,
        "weights_dq",
        {builder.MakeScalarAttribute("axis", static_cast<int64_t>(weight_quant_axis))});

    // bias as fp32 initializer (no quantization)
    MakeTestInput<float>(builder, "bias", bias_def);

    // MatMul(input_qdq, weights_dq) -> matmul_out
    builder.AddNode("MatMul", "MatMul", {input_qdq, "weights_dq"}, {"matmul_out"}, kOnnxDomain);

    // Add(matmul_out, bias) -> Y
    builder.AddNode("Add", "Add", {"matmul_out", "bias"}, {"Y"}, kOnnxDomain);

    // Y -> Q -> DQ -> (graph output)
    AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(builder, "qdq_out", "Y",
                                                       output_qparams[0].scale,
                                                       output_qparams[0].zero_point);
  };
}

template <typename InputQType, typename WeightQType, typename OutputQType>
static void RunQDQBlockQuantMatMulOpTest(
    const std::vector<int64_t>& shape_input,
    const std::vector<int64_t>& shape_weight,
    int64_t block_size,
    int64_t weight_quant_axis,
    QDQTolerance tolerance = QDQTolerance(),
    ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
    int opset = 21,
    bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_block_quant_weight_optimization"] = "1";

  const size_t num_input_elems = static_cast<size_t>(
      std::accumulate(shape_input.begin(), shape_input.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
  const size_t num_weight_elems = static_cast<size_t>(
      std::accumulate(shape_weight.begin(), shape_weight.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

  TestInputDef<float> input_def(shape_input, false, GetFloatDataInRange(-0.1f, 0.1f, num_input_elems));
  TestInputDef<float> weight_def(shape_weight, true, GetFloatDataInRange(-0.1f, 0.1f, num_weight_elems));

  TestQDQModelAccuracy(
      BuildMatMulOpTestCase(input_def, weight_def),
      BuildQDQBlockQuantMatMulTestCase<InputQType, WeightQType, OutputQType>(
          input_def, weight_def, block_size, weight_quant_axis, use_contrib_qdq),
      provider_options, opset, expected_ep_assignment, tolerance);
}

template <typename InputQType, typename WeightQType, typename OutputQType>
static void RunQDQPerChannelMatMulAddOpTest(
    const std::vector<int64_t>& shape_input, const std::vector<int64_t>& shape_weight,
    const std::vector<int64_t>& bias_shape, int64_t weight_quant_axis,
    QDQTolerance tolerance = QDQTolerance(),
    ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All, int opset = 21) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto num_elements = [](const std::vector<int64_t>& shape) -> size_t {
    return static_cast<size_t>(
        std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
  };

  TestInputDef<float> input_def(shape_input, false, GetFloatDataInRange(-0.1f, 0.1f, num_elements(shape_input)));
  TestInputDef<float> weight_def(shape_weight, true, GetFloatDataInRange(-0.1f, 0.1f, num_elements(shape_weight)));
  TestInputDef<float> bias_def(bias_shape, true, GetFloatDataInRange(-0.1f, 0.1f, num_elements(bias_shape)));

  TestQDQModelAccuracy(BuildMatMulAddOpTestCase(input_def, weight_def, bias_def),
                       BuildQDQPerChannelMatMulAddTestCase<InputQType, WeightQType, OutputQType>(
                           input_def, weight_def, bias_def, weight_quant_axis),
                       provider_options, opset, expected_ep_assignment, tolerance);
}

//
// CPU tests:
//
TEST_F(QnnCPUBackendTests, MatMulOp) {
  // RunMatMulOpTest(shape_0, shape_1, is_initializer_0, is_initializer_1)
  RunMatMulOpTest({2, 3}, {3, 2}, false, false);
  RunMatMulOpTest({2, 3}, {3, 2}, false, true);
  RunMatMulOpTest({2, 3}, {3, 2}, true, false);
  RunMatMulOpTest({2, 3}, {3, 2}, true, true);  // constant folding
  RunMatMulOpTest({2, 3}, {2, 3, 2}, false, false);
  RunMatMulOpTest({3, 3, 3}, {3, 2}, true, false);
  RunMatMulOpTest({2, 3, 3, 3}, {3, 2}, false, true);
  RunMatMulOpTest({2, 3, 3, 3}, {2, 3, 3, 2}, false, true);

  RunMatMulOpTest({2, 1, 2, 3}, {3, 3, 2}, false, false);
  RunMatMulOpTest({3}, {3}, false, false);
  RunMatMulOpTest({3}, {3}, false, true);
  RunMatMulOpTest({3}, {3}, true, false);
  RunMatMulOpTest({3}, {3, 2}, false, false);
  RunMatMulOpTest({3}, {3, 2}, false, true);
  RunMatMulOpTest({3}, {3, 3, 2}, true, false);
  RunMatMulOpTest({2, 3}, {3}, false, false);
  RunMatMulOpTest({2, 3}, {3}, true, false);
  RunMatMulOpTest({2, 3, 4}, {4, 2}, false, true);
  RunMatMulOpTest({2, 3, 3, 3}, {3}, false, false);
  RunMatMulOpTest({1, 1, 2, 2, 4}, {4, 2}, false, true);

  // Failed randomly on Linux
  // Expected: contains 36 values, where each value and its corresponding value in 16-byte object
  // <24-00 00-00 00-00 00-00 40-4A 47-42 4D-56 00-00> are an almost-equal pair
  // Actual: 16-byte object <24-00 00-00 00-00 00-00 80-39 2B-42 4D-56 00-00>, where the value pair (0.104199991, 0)
  // at index #18 don't match, which is -0.1042 from 0.1042
  // RunMatMulOpTest({2, 3, 3, 3}, {3, 2}, true, false);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

//
// HTP tests:
//

namespace {

// Builds an ONNX QDQ graph for a MatMul with a block-quantized (BW_FLOAT_BLOCK) weight.
//   - activation A: float → Q(uint16) → DQ, shape [M, K]
//   - weight B: INT4/INT8 (or UINT4/UINT8) initializer + DQ with block_size attribute and a rank-2
//               float scale [K/block_size, N] (axis=0, K is the blocked contraction dimension)
//   - output:  MatMul → Q(uint16) → DQ → graph output, shape [M, N]
//
// weight_bits: 4 for INT4/UINT4 (default), 8 for INT8/UINT8, 2 for INT2/UINT2.
// block_size must be a multiple of 8 (4-bit), 4 (8-bit), or 16 (2-bit) per HTP.
// weight_is_unsigned: true → use UINT weight type; exercises the unsigned→signed conversion path.
GetQDQTestCaseFn BuildBQMatMulTestCase(int64_t M, int64_t K, int64_t N, int64_t block_size,
                                       int weight_bits = 4, bool weight_is_unsigned = false,
                                       std::vector<int64_t> act_shape_override = {},
                                       std::vector<int64_t> weight_shape_override = {}) {
  return [M, K, N, block_size, weight_bits, weight_is_unsigned, act_shape_override,
          weight_shape_override](ModelTestBuilder& builder) -> void {
    const int64_t num_blocks = K / block_size;  // caller ensures K % block_size == 0

    // ── Activation A: float → Q(uint16) → DQ ─────────────────────────────────
    const std::vector<int64_t> act_shape = act_shape_override.empty() ? std::vector<int64_t>{M, K}
                                                                      : act_shape_override;
    auto input_def = TestInputDef<float>(act_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    const float act_scale = 2.0f / 65534.0f;  // uint16 symmetric per-tensor, ~[-1, 1]
    const uint16_t act_zp = 32767;
    const std::string act_dql_out = AddQDQNodePair<uint16_t>(builder, "act", "input", act_scale, act_zp);

    // ── Weight B initializer + DQ(block_size, axis=rank-2) ──────────────────
    // Scale rank == weight rank per ONNX opset 21.
    const std::vector<int64_t> weight_shape = weight_shape_override.empty()
                                                  ? std::vector<int64_t>{K, N}
                                                  : weight_shape_override;
    // Build scale shape: same as weight shape with the K-axis (rank-2) replaced by num_blocks.
    std::vector<int64_t> scale_shape = weight_shape;
    scale_shape[scale_shape.size() - 2] = num_blocks;
    const int64_t block_axis = static_cast<int64_t>(weight_shape.size()) - 2;
    builder.MakeInitializer<float>("weight_scale", scale_shape, 0.01f, 0.05f);

    const size_t num_elems = static_cast<size_t>(K * N);
    if (weight_bits == 4 && !weight_is_unsigned) {
      std::vector<Int4x2> weight_data(Int4x2::CalcNumInt4Pairs(num_elems));
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i >> 1].SetElem(i & 1, static_cast<int8_t>((i % 7) - 3));
      }
      builder.MakeInitializer<Int4x2>("weight_quant", weight_shape, weight_data);
    } else if (weight_bits == 4 && weight_is_unsigned) {
      std::vector<UInt4x2> weight_data(UInt4x2::CalcNumInt4Pairs(num_elems));
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i >> 1].SetElem(i & 1, static_cast<uint8_t>(i % 15));
      }
      builder.MakeInitializer<UInt4x2>("weight_quant", weight_shape, weight_data);
    } else if (weight_bits == 2 && !weight_is_unsigned) {
      std::vector<Int2x4> weight_data(Int2x4::CalcNumInt2Quads(num_elems));
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i >> 2].SetElem(i & 3, static_cast<int8_t>((i % 3) - 1));
      }
      builder.MakeInitializer<Int2x4>("weight_quant", weight_shape, weight_data);
    } else if (weight_bits == 2 && weight_is_unsigned) {
      std::vector<UInt2x4> weight_data(UInt2x4::CalcNumInt2Quads(num_elems));
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i >> 2].SetElem(i & 3, static_cast<uint8_t>(i % 4));
      }
      builder.MakeInitializer<UInt2x4>("weight_quant", weight_shape, weight_data);
    } else if (weight_is_unsigned) {
      std::vector<uint8_t> weight_data(num_elems);
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i] = static_cast<uint8_t>(i % 127);
      }
      builder.MakeInitializer<uint8_t>("weight_quant", weight_shape, weight_data);
    } else {
      std::vector<int8_t> weight_data(num_elems);
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i] = static_cast<int8_t>((i % 127) - 63);
      }
      builder.MakeInitializer<int8_t>("weight_quant", weight_shape, weight_data);
    }

    // DQ with block_size; omit zero_point (symmetric). axis=0: K is the blocked dimension.
    builder.AddNode("weight_dql", "DequantizeLinear",
                    {"weight_quant", "weight_scale"}, {"weight_dql_out"}, "",
                    {builder.MakeScalarAttribute("axis", block_axis),
                     builder.MakeScalarAttribute("block_size", block_size)});

    // ── MatMul ───────────────────────────────────────────────────────────────
    builder.AddNode("matmul", "MatMul", {act_dql_out, "weight_dql_out"}, {"matmul_out"}, kOnnxDomain);

    // ── Output: MatMul → Q(uint16) → DQ → graph output ───────────────────────
    const float out_scale = 4.0f / 65534.0f;
    const uint16_t out_zp = 32767;
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "out", "matmul_out", out_scale, out_zp);
  };
}

ProviderOptions GetBQMatMulProviderOptions() {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["enable_block_quant_weight_optimization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  // On the x86_64 Linux HTP simulator, specify SM8850 to enable BW_FLOAT_BLOCK support.
  // On real ARM64 hardware, the SoC model is auto-detected by QNN EP.
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return opts;
}

}  // namespace

// INT4 weight, K=16, N=4, block_size=8 (2 blocks/N), uint16 activation, no bias.
// Checks: all nodes assigned to QNN EP; output matches CPU EP within 1e-2.
TEST_F(QnnHTPBackendTests, MatMulBQ_U16Int4_NoBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Larger K with more blocks per output channel. Guards the [num_blocks, N] → [N, num_blocks]
// scale reordering: a wrong order fails on accuracy, not on QNN validation.
TEST_F(QnnHTPBackendTests, MatMulBQ_U16Int4_MultiBlock) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/32, /*N=*/8, /*block_size=*/8),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT4, block_size=16: still a valid HTP multiple-of-8 block size.
TEST_F(QnnHTPBackendTests, MatMulBQ_U16Int4_BlockSize16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/32, /*N=*/4, /*block_size=*/16),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT8, block_size=4: minimum valid HTP multiple-of-4 block size for 8-bit.
TEST_F(QnnHTPBackendTests, MatMulBQ_U16Int8_BlockSize4) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/4,
                                        /*weight_bits=*/8),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// UINT4 weight: exercises the unsigned→signed conversion path (TransformUnsignedToSignedFixedPoint).
TEST_F(QnnHTPBackendTests, MatMulBQ_U16UInt4_NoBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8,
                                        /*weight_bits=*/4, /*weight_is_unsigned=*/true),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// UINT8 weight: unsigned 8-bit path.
TEST_F(QnnHTPBackendTests, MatMulBQ_U16UInt8_BlockSize4) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/4,
                                        /*weight_bits=*/8, /*weight_is_unsigned=*/true),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// INT2, block_size=16: DISABLED. Two independent blockers (same as Conv BQ):
//   1. ORT CPU backend does not support 2-bit Q/DQ (rejects tensor(int2)).
//   2. QAIRT HTP backend does not support 2-bit BQ until QAIRT 2.47.
TEST_F(QnnHTPBackendTests, DISABLED_MatMulBQ_U16Int2_BlockSize16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/32, /*N=*/4, /*block_size=*/16,
                                        /*weight_bits=*/2),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// Rank-3 activation [1, M, K]: leading dim=1, reshapes to [1, 1, M, K] matching weight [1, 1, K, N].
TEST_F(QnnHTPBackendTests, MatMulBQ_U16Int4_Rank3Activation) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8,
                                        /*weight_bits=*/4, /*weight_is_unsigned=*/false,
                                        /*act_shape_override=*/{1, 2, 16}),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Rank-3 weight [1, K, N]: leading dim = 1, reshapeable to [1, 1, K, N].
TEST_F(QnnHTPBackendTests, MatMulBQ_U16Int4_Rank3Weight) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8,
                                        /*weight_bits=*/4, /*weight_is_unsigned=*/false,
                                        /*act_shape_override=*/{}, /*weight_shape_override=*/{1, 16, 4}),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Rank-4 activation [1, 1, M, K]: already in 4-D form, no reshape needed.
TEST_F(QnnHTPBackendTests, MatMulBQ_U16Int4_Rank4Activation) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8,
                                        /*weight_bits=*/4, /*weight_is_unsigned=*/false,
                                        /*act_shape_override=*/{1, 1, 2, 16}),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Rank-4 weight [1, 1, K, N]: already in the [1,1,K,N] form QNN requires.
TEST_F(QnnHTPBackendTests, MatMulBQ_U16Int4_Rank4Weight) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQMatMulTestCase(/*M=*/2, /*K=*/16, /*N=*/4, /*block_size=*/8,
                                        /*weight_bits=*/4, /*weight_is_unsigned=*/false,
                                        /*act_shape_override=*/{}, /*weight_shape_override=*/{1, 1, 16, 4}),
                  GetBQMatMulProviderOptions(), /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Test QDQ MatMul + Add with per-channel int8 weight and int32 bias, uint16 output.
// Pattern: input -> Q(u8) -> DQ -> MatMul -> Add -> Q(u16) -> DQ -> output
//          weight[i8 per-ch] -> DQ ---|        |
//          bias[i32 per-ch] -> DQ -------------|
//
// NOTE: Inputs must be rank-2 (2D). With rank>2 inputs, ORT's MatMulAddFusion optimizer
// inserts Reshape nodes around Gemm (Reshape -> Gemm -> Reshape), which breaks the QDQ
// pattern because QNN EP cannot form valid QDQ node units when Reshape sits between DQ and Gemm.
//
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_U8_I8_U16) {
  RunQDQPerChannelMatMulAddOpTest<uint8_t, int8_t, uint16_t>(
      {2, 3}, {3, 2}, {2}, /*weight_quant_axis=*/-1, QDQTolerance(0.002f));
  // Larger shape
  RunQDQPerChannelMatMulAddOpTest<uint8_t, int8_t, uint16_t>(
      {32, 32}, {32, 32}, {32}, /*weight_quant_axis=*/-1, QDQTolerance(0.002f));
}

// Test QDQ MatMul + Add with per-channel int8 weight, fp32 output (no QDQ on output).
// Pattern: input -> Q -> DQ -> MatMul -> Add -> output(fp32)
//     weight[i8 per-ch] -> DQ ---|        /
//             bias[f32 initializer] -----/
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_U8_I8_FP32) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto num_elements = [](const std::vector<int64_t>& shape) -> size_t {
    return static_cast<size_t>(
        std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
  };

  std::vector<int64_t> shape_input = {2, 3};
  std::vector<int64_t> shape_weight = {3, 2};
  std::vector<int64_t> bias_shape = {2};

  TestInputDef<float> input_def(shape_input, false, GetFloatDataInRange(-0.1f, 0.1f, num_elements(shape_input)));
  TestInputDef<float> weight_def(shape_weight, true, GetFloatDataInRange(-0.1f, 0.1f, num_elements(shape_weight)));
  TestInputDef<float> bias_def(bias_shape, true, GetFloatDataInRange(-0.1f, 0.1f, num_elements(bias_shape)));

  // Float reference model: MatMul -> Add -> output (f32)
  auto f32_model_fn = BuildMatMulAddOpTestCase(input_def, weight_def, bias_def);

  // QDQ model: QDQ(u8) -> MatMul -> Add -> output (f32)
  auto qdq_model_fn = [input_def, weight_def, bias_def](
                          ModelTestBuilder& builder, std::vector<QuantParams<uint8_t>>& output_qparams) {
    QNN_TEST_UNUSED_PARAMETER(output_qparams);

    // input -> Q -> DQ
    MakeTestInput<float>(builder, "input", input_def);
    const QuantParams<uint8_t> input_qparams = GetTestInputQuantParams<uint8_t>(input_def);
    const std::string input_qdq =
        AddQDQNodePair<uint8_t>(builder, "qdq_in", "input",
                                input_qparams.scale, input_qparams.zero_point);

    // weights as fp32 initializer
    MakeTestInput<float>(builder, "weights", weight_def);

    // bias as fp32 initializer
    MakeTestInput<float>(builder, "bias", bias_def);

    // MatMul -> Add
    builder.AddNode("MatMul", "MatMul", {input_qdq, "weights"}, {"matmul_out"}, kOnnxDomain);
    builder.AddNode("Add", "Add", {"matmul_out", "bias"}, {"Y"}, kOnnxDomain);

    // Output is already fp32, no Cast needed
    builder.MakeOutput("Y");
  };

  TestQDQModelAccuracy<uint8_t>(f32_model_fn,
                                qdq_model_fn,
                                provider_options, 21, ExpectedEPNodeAssignment::All);
}
template <typename InputQType = float, typename WeightQType = int8_t, typename OutputQType = float>
static void RunMixedPrecisionPerChannelMatMulAddTest(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& bias_shape) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto num_elements = [](const std::vector<int64_t>& shape) -> size_t {
    return static_cast<size_t>(
        std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
  };

  TestInputDef<float> input_def(input_shape, false, GetFloatDataInRange(-0.1f, 0.1f, num_elements(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true, GetFloatDataInRange(-0.1f, 0.1f, num_elements(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true, GetFloatDataInRange(-0.1f, 0.1f, num_elements(bias_shape)));

  auto f32_model_fn = BuildMatMulAddOpTestCase(input_def, weight_def, bias_def);

  // Determine the quant type used for TestQDQModelAccuracy template param.
  // Priority: OutputQType if quantized, else InputQType if quantized, else uint8_t fallback.
  using TestQType = std::conditional_t<!std::is_same_v<OutputQType, float>, OutputQType,
                                       std::conditional_t<!std::is_same_v<InputQType, float>, InputQType, uint8_t>>;

  auto qdq_model_fn = [input_def, weight_def, bias_def](
                          ModelTestBuilder& builder, std::vector<QuantParams<TestQType>>& output_qparams) {
    QNN_TEST_UNUSED_PARAMETER(output_qparams);

    MakeTestInput<float>(builder, "input", input_def);
    std::string matmul_input_name;
    if constexpr (std::is_same_v<InputQType, float>) {
      matmul_input_name = "input";
    } else {
      const QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
      matmul_input_name =
          AddQDQNodePair<InputQType>(builder, "qdq_in", "input",
                                     input_qparams.scale, input_qparams.zero_point);
    }

    auto w_shape = weight_def.GetShape();
    std::vector<float> weight_scales;
    std::vector<WeightQType> weight_zero_points;
    GetTestInputQuantParamsPerChannel<WeightQType>(weight_def, weight_scales, weight_zero_points, 1, true);

    std::vector<WeightQType> quantized_weights(SizeOfShape(w_shape));
    QuantizeValues<float, WeightQType>(weight_def.GetRawData(), quantized_weights, w_shape,
                                       weight_scales, weight_zero_points, 1);
    builder.MakeInitializer<WeightQType>("weights", w_shape, quantized_weights);
    builder.template AddDequantizeLinearNode<WeightQType>(
        "weights_dq", "weights", weight_scales, weight_zero_points, "weights_dq",
        {builder.MakeScalarAttribute("axis", static_cast<int64_t>(-1))});

    MakeTestInput<float>(builder, "bias", bias_def);

    builder.AddNode("MatMul", "MatMul", {matmul_input_name, "weights_dq"}, {"matmul_out"}, kOnnxDomain);
    builder.AddNode("Add", "Add", {"matmul_out", "bias"}, {"Y"}, kOnnxDomain);

    if constexpr (std::is_same_v<OutputQType, float>) {
      builder.MakeOutput("Y");
    } else {
      AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(builder, "qdq_out", "Y",
                                                         output_qparams[0].scale,
                                                         output_qparams[0].zero_point);
    }
  };

  TestQDQModelAccuracy<TestQType>(f32_model_fn, qdq_model_fn,
                                  provider_options, 21, ExpectedEPNodeAssignment::All);
}
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_U8_I8_NoOutputQDQ) {
  RunMixedPrecisionPerChannelMatMulAddTest<uint8_t, int8_t>({2, 3}, {3, 2}, {2});
}

// QDQ(u16) input, per-channel int16 weights, fp32 output (no output QDQ).
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_U16_I16_NoOutputQDQ) {
  RunMixedPrecisionPerChannelMatMulAddTest<uint16_t, int16_t>({2, 3}, {3, 2}, {2});
}

// fp32 input, per-channel int8 weights, QDQ(u8) output (no input QDQ).
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_NoInputQDQ_U8) {
  RunMixedPrecisionPerChannelMatMulAddTest<float, int8_t, uint8_t>({2, 3}, {3, 2}, {2});
}

// fp32 input, per-channel int8 weights, QDQ(u16) output (no input QDQ).
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_NoInputQDQ_U16) {
  RunMixedPrecisionPerChannelMatMulAddTest<float, int8_t, uint16_t>({2, 3}, {3, 2}, {2});
}
// fp32 input, per-channel int8 weights, fp32 output (no input or output QDQ)
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_NoQDQ_FP32) {
  RunMixedPrecisionPerChannelMatMulAddTest<float, int8_t, float>({2, 3}, {3, 2}, {2});
}

// The three tests below use a rank-2 [M, N] bias instead of rank-1 [N]. MatMulAddFusion still folds
// MatMul+Add into a Gemm (C=[M, N] broadcasts unidirectionally), but QNN FullyConnected only accepts
// a [N] / [1, N] bias, so GemmOpBuilder decomposes into FullyConnected + ElementWiseAdd ("split_gemm").
// A rank-2 bias also falls outside ORT's WeightBiasQuantization (rank-1 only), so it stays fp32 —
// which is why only the two float-FC mixed-precision sub-cases can reach the split path: a quantized
// FC would need a quantized bias, and ExplictOpCheck rejects a quantized C with shape[0] != 1.
//
// Regression coverage for the split path: the FC intermediate used to be typed off the ONNX-side
// activation type (wrong once ProcessInputs re-typed the activation ahead of FC), and the
// mixed-precision output tail used to be emitted only on the non-split branches. Every rank-1-bias
// test above takes the non-split path, so both bugs were invisible.

// QDQ(u8) input, per-channel int8 weights, fp32 output, rank-2 bias.
// quant-to-float + split_gemm: ProcessInputs inserts Dequantize on the activation and pre-dequantizes
// the weights, so FC runs in fp32. The FC intermediate feeding the ElementWiseAdd must therefore be
// FLOAT_32; typing it off the ONNX activation (UFIXED_8) hands FC mismatched in[0]/out[0] types.
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_U8_I8_NoOutputQDQ_2DBias) {
  RunMixedPrecisionPerChannelMatMulAddTest<uint8_t, int8_t>({2, 3}, {3, 2}, {2, 2});
}

// fp32 input, per-channel int8 weights, QDQ(u16) output, rank-2 bias.
// float-to-quant + split_gemm: FC and the ElementWiseAdd both run in fp32 and a Quantize tail
// re-encodes to u16. The tail has to be appended after the Add, not after the FC — before the fix the
// split branch emitted no tail at all, leaving the node unit's output tensor uncreated.
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_NoInputQDQ_U16_2DBias) {
  RunMixedPrecisionPerChannelMatMulAddTest<float, int8_t, uint16_t>({2, 3}, {3, 2}, {2, 2});
}

// fp32 in/out with quantized weights and a rank-2 bias: split_gemm with no mixed-precision tail.
// Pins the baseline so the two tests above isolate the mixed-precision handling.
TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_NoQDQ_FP32_2DBias) {
  RunMixedPrecisionPerChannelMatMulAddTest<float, int8_t, float>({2, 3}, {3, 2}, {2, 2});
}

TEST_F(QnnHTPBackendTests, MatMulAddOp_QDQ_PerChannel_U16_I8_U8) {
  RunQDQPerChannelMatMulAddOpTest<uint16_t, int8_t, uint8_t>(
      {2, 3}, {3, 2}, {2}, /*weight_quant_axis=*/-1);
  // Larger shape
  RunQDQPerChannelMatMulAddOpTest<uint8_t, int8_t, uint16_t>(
      {32, 32}, {32, 32}, {32}, /*weight_quant_axis=*/-1);
}

TEST_F(QnnHTPBackendTests, MatMulOp) {
  // RunMatMulOpTest(shape_0, shape_1, is_initializer_0, is_initializer_1, expected_ep_assignment,
  // opset, f32_abs_err)
  RunMatMulOpTest({2, 3}, {3, 2}, false, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3}, {3, 2}, true, true, ExpectedEPNodeAssignment::All, "htp");  // constant folding
  RunMatMulOpTest({2, 3}, {2, 3, 2}, false, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3, 3, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3, 3, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3, 3, 3}, {2, 3, 3, 2}, false, true, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 1, 2, 3}, {3, 3, 2}, false, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({3}, {3}, false, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({3}, {3}, false, true, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({3}, {3}, true, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({3}, {3, 2}, false, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({3}, {3, 3, 2}, true, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3}, {3}, false, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3}, {3}, true, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3, 4}, {4, 2}, false, true, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({2, 3, 3, 3}, {3}, false, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
  RunMatMulOpTest({1, 1, 2, 2, 4}, {4, 2}, false, true, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);

  // Failed randomly on Linux
  // Expected: contains 18 values, where each value and its corresponding value in 16-byte object
  // <12-00 00-00 00-00 00-00 40-3D CC-A5 5A-7A 00-00> are an almost-equal pair
  // Actual: 16-byte object <12-00 00-00 00-00 00-00 80-E8 CF-8F 5B-7A 00-00>, where the value pair
  // (0.0393999927, 98304.0078) at index #6 don't match, which is 98304 from 0.0394
  // RunMatMulOpTest({3, 3, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, "htp", 18, 1e-2f);
}

// Broken on v79 and v81 devices with several results outside of acceptable tolerance.
// Example:
// Inaccuracy detected for output 'output_0', element 0
// output_range=0.010000000707805157, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): 0.010000000707805157
// qdq@QNN_EP val: 0.0099215693771839142 (err: 7.8431330621242523e-05, err/output_range: 0.78431320190429688%)
// qdq@CPU_EP val: 0.010000000707805157 (err: 0, err/output_range: 0%)
// abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = 0.78431320190429688%
TEST_F(QnnHTPBackendTests, MatMulOp_QDQ) {
  QNN_SKIP_TEST_ON_ARM64("QDQ accuracy below tolerance on v79 and v81 devices");
  // UINT8
  // RunQDQMatMulOpTest(shape_0, shape_1, is_initializer_0, is_initializer_1, expected_ep_assignment, opset,
  // use_contrib_qdq)
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3, 2}, false, false);
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, 21,
                                                false, QDQTolerance(0.008f));
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 2, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, 18,
                                                true);
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 1, 3, 3}, {3, 3, 2}, false, true);
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({3}, {3}, false, false);
  RunQDQMatMulOpTest<uint8_t, uint8_t, uint8_t>({2, 3}, {3}, true, false);

  // UINT16, UINT8
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({2, 3}, {3, 2}, false, false);
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({2, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, 18, true);
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({2, 3, 3, 3}, {3, 2}, true, false);
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({3}, {3, 2}, false, true);
  RunQDQMatMulOpTest<uint16_t, uint8_t, uint16_t>({2, 3, 3, 3}, {3}, false, false);

  // UINT16, per-channel signed 4-bit weight
  // RunQDQPerChannelMatMulOpTest(shape_input, shape_weight, weight_quant_axis, tolerance, expected_ep_assignment,
  // opset, use_contrib_qdq, enable_fp16_precision)
  RunQDQPerChannelMatMulOpTest<uint16_t, Int4x2, uint16_t>({2, 3}, {3, 2}, 1);
  RunQDQPerChannelMatMulOpTest<uint16_t, Int4x2, uint16_t>({2, 3, 3, 3}, {3, 2}, -1, QDQTolerance(),
                                                           ExpectedEPNodeAssignment::All, 18, true);

  // UINT16, per-channel INT8 weight
  RunQDQPerChannelMatMulOpTest<uint16_t, int8_t, uint16_t>({2, 3}, {3, 2}, 1, QDQTolerance(),
                                                           ExpectedEPNodeAssignment::All, 21, false, false);
  RunQDQPerChannelMatMulOpTest<uint16_t, int8_t, uint16_t>({2, 3, 3}, {3}, -1, QDQTolerance(0.0041f));
}

// Tests MatMul with two uint16 (quantized) inputs that are both dynamic.
// This exercises a logic in QNN EP that inserts a QNN Convert op before input[1] to convert asymmetric uint16 into
// symmetric one.
// Got specific shapes and input ranges (quant params) from customer model.
TEST_F(QnnHTPBackendTests, MatMulOp_QDQ_Regression_uint16_dynamic_inputs) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#ifdef __linux__
  // W16A16 requires minimum HTP arch v73.
  provider_options["htp_arch"] = "73";
#endif

  // Test with rank 4 inputs
  {
    std::vector<int64_t> shape_0 = {1, 12, 512, 96};
    TestInputDef<float> input0_def(
        {1, 12, 512, 96}, false,
        GetFloatDataInRange(-5.087f, 4.992f,
                            static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));
    std::vector<int64_t> shape_1 = {1, 12, 96, 512};
    TestInputDef<float> input1_def(
        shape_1, false,
        GetFloatDataInRange(-6.772f, 7.258f,
                            static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));

    TestQDQModelAccuracy(
        BuildMatMulOpTestCase(input0_def, input1_def),
        BuildMatMulOpQDQTestCase<uint16_t, uint16_t, uint16_t>(input0_def, input1_def, false),
        provider_options, 21, ExpectedEPNodeAssignment::All, QDQTolerance());
  }

  // Test with input[1] as rank 1
  {
    std::vector<int64_t> shape_0 = {1, 12, 512, 96};
    TestInputDef<float> input0_def(
        {1, 12, 512, 96}, false,
        GetFloatDataInRange(-5.087f, 4.992f,
                            static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));
    std::vector<int64_t> shape_1 = {96};
    TestInputDef<float> input1_def(
        shape_1, false,
        GetFloatDataInRange(-6.772f, 7.258f,
                            static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));

    TestQDQModelAccuracy(
        BuildMatMulOpTestCase(input0_def, input1_def),
        BuildMatMulOpQDQTestCase<uint16_t, uint16_t, uint16_t>(input0_def, input1_def, false),
        provider_options, 21, ExpectedEPNodeAssignment::All, QDQTolerance());
  }
}

// Tests MatMul with two uint16 (quantized) inputs with weight as static.
// This exercises a workaround in QNN EP that inserts a QNN Convert op before input[1] (converts from uint16 to sint16).
// This workaround prevents a validation error for this specific MatMul configuration.
// Got specific shapes and input ranges (quant params) from customer model.
TEST_F(QnnHTPBackendTests, MatMulOp_QDQ_Regression_uint16_static_weight) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#ifdef __linux__
  // W16A16 requires minimum HTP arch v73.
  provider_options["htp_arch"] = "73";
#endif

  // Test with rank 4 inputs
  {
    std::vector<int64_t> shape_0 = {1, 12, 512, 96};
    TestInputDef<float> input0_def(
        {1, 12, 512, 96}, false,
        GetFloatDataInRange(-5.087f, 4.992f,
                            static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));
    std::vector<int64_t> shape_1 = {1, 12, 96, 512};
    TestInputDef<float> input1_def(
        shape_1, true,
        GetFloatDataInRange(-6.772f, 7.258f,
                            static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));

    TestQDQModelAccuracy(
        BuildMatMulOpTestCase(input0_def, input1_def),
        BuildMatMulOpQDQTestCase<uint16_t, uint16_t, uint16_t>(input0_def, input1_def, false),
        provider_options, 21, ExpectedEPNodeAssignment::All, QDQTolerance());
  }

  // Test with input[1] as rank 1
  {
    std::vector<int64_t> shape_0 = {1, 12, 512, 96};
    TestInputDef<float> input0_def(
        {1, 12, 512, 96}, false,
        GetFloatDataInRange(-5.087f, 4.992f,
                            static_cast<size_t>(std::accumulate(shape_0.begin(), shape_0.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));
    std::vector<int64_t> shape_1 = {96};
    TestInputDef<float> input1_def(
        shape_1, true,
        GetFloatDataInRange(-6.772f, 7.258f,
                            static_cast<size_t>(std::accumulate(shape_1.begin(), shape_1.end(), static_cast<int64_t>(1),
                                                                std::multiplies<int64_t>()))));

    TestQDQModelAccuracy(
        BuildMatMulOpTestCase(input0_def, input1_def),
        BuildMatMulOpQDQTestCase<uint16_t, uint16_t, uint16_t>(input0_def, input1_def, false),
        provider_options, 21, ExpectedEPNodeAssignment::All, QDQTolerance());
  }
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(__linux__)

// Tests MatMul with ONNX block-quantized (BQ) weight using the BQ -> QNN LPBQ conversion path.
// Currently BQ -> LPBQ conversion is only supported on Linux. It will be later enabled for windows as well.
TEST_F(QnnHTPBackendTests, MatMulOp_QDQ_BlockQuant) {
  RunQDQBlockQuantMatMulOpTest<uint16_t, Int4x2, uint16_t>({4, 16}, {16, 8}, 8, 0, QDQTolerance(0.05f));
  RunQDQBlockQuantMatMulOpTest<int16_t, Int4x2, int16_t>({4, 128}, {128, 64}, 32, 0, QDQTolerance(0.05f));
  RunQDQBlockQuantMatMulOpTest<int16_t, Int4x2, int16_t>({4, 128}, {128, 64}, 64, 0, QDQTolerance(0.05f));
  RunQDQBlockQuantMatMulOpTest<uint16_t, Int4x2, uint16_t>({2, 4, 16}, {16, 8}, 8, 0, QDQTolerance(0.05f));
  RunQDQBlockQuantMatMulOpTest<uint16_t, Int4x2, uint16_t>({2, 3, 4, 16}, {16, 8}, 8, 0, QDQTolerance(0.05f));
}

#endif  // defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// RunMatMulOpTest(shape_0, shape_1, is_initializer_0, is_initializer_1, expected_ep_assignment, backend);

TEST_F(QnnGPUBackendTests, MatMulOp_simple) {
  RunMatMulOpTest({2, 3}, {3, 2}, false, false, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({2, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({2, 3}, {3, 2}, true, false, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({2, 3}, {3, 2}, true, true, ExpectedEPNodeAssignment::All, "gpu");  // constant folding
}

TEST_F(QnnGPUBackendTests, MatMulOp_batches) {
  RunMatMulOpTest({3, 3, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({2, 3, 3, 3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
}

TEST_F(QnnGPUBackendTests, MatMulOp_batchesWtsSameDim) {
  RunMatMulOpTest({3, 3, 3}, {3, 3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
}

TEST_F(QnnGPUBackendTests, MatMulOp_batchesWtsSameDim2) {
  RunMatMulOpTest({2, 3, 3, 3}, {2, 3, 3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
}

TEST_F(QnnGPUBackendTests, MatMulOp_wtsDimBcast) {
  RunMatMulOpTest({3, 3, 3}, {1, 3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
}

TEST_F(QnnGPUBackendTests, DISABLED_MatMulOp_batchesDimBcast) {
  RunMatMulOpTest({1, 3, 3}, {3, 3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
}

TEST_F(QnnGPUBackendTests, DISABLED_MatMulOp_batchesDimBcast2) {
  RunMatMulOpTest({2, 1, 3, 3}, {3, 3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
}

TEST_F(QnnGPUBackendTests, MatMulOp_inp0DimBcast) {
  RunMatMulOpTest({3, 3}, {3, 3, 2}, false, false, ExpectedEPNodeAssignment::All, "gpu");
}

TEST_F(QnnGPUBackendTests, MatMulOp_inp1DimBcast) {
  RunMatMulOpTest({2, 3, 3}, {3, 2}, false, false, ExpectedEPNodeAssignment::All, "gpu");
}

TEST_F(QnnGPUBackendTests, MatMulOp_rank1) {
  RunMatMulOpTest({3}, {3}, false, false, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({3}, {3}, false, true, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({3}, {3}, true, false, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({3}, {3, 2}, false, false, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({3}, {3, 2}, false, true, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({3}, {3, 3, 2}, true, false, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({2, 3}, {3}, false, false, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({2, 3}, {3}, true, false, ExpectedEPNodeAssignment::All, "gpu");
  RunMatMulOpTest({2, 3, 3, 3}, {3}, false, false, ExpectedEPNodeAssignment::All, "gpu");
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
