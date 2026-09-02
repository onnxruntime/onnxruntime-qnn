// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <optional>
#include <string>

#include <gsl/gsl_util>
#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

// Information for activation node placed between the Conv and Q.
struct OutputActivationInfo {
  std::string op_type;  // Relu or Clip
  std::vector<float> const_inputs;
};

// Creates a graph with a single float32 Conv operator. Used for testing CPU backend.
static GetTestModelFn BuildF32ConvTestCase(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                                           const TestInputDef<float>& weights_def,
                                           const TestInputDef<float>& bias_def,
                                           const std::vector<int64_t>& strides,
                                           const std::vector<int64_t>& pads,
                                           const std::vector<int64_t>& dilations,
                                           std::optional<int64_t> group,
                                           const std::string& auto_pad = "NOTSET",
                                           std::optional<OutputActivationInfo> output_activation = std::nullopt,
                                           const std::optional<std::vector<int64_t>>& output_shape = std::nullopt) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, group, auto_pad, output_activation, output_shape](ModelTestBuilder& builder) {
    // inputs
    MakeTestInput<float>(builder, "input", input_def);
    MakeTestInput<float>(builder, "weights", weights_def);
    std::vector<std::string> conv_input_names{"input", "weights"};
    if (!bias_def.GetShape().empty()) {
      MakeTestInput<float>(builder, "bias", bias_def);
      conv_input_names.push_back("bias");
    }

    // attributes (must be provided at node creation)
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", auto_pad));

    if (group.has_value()) {
      conv_attrs.push_back(builder.MakeScalarAttribute("group", group.value()));
    }

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_attrs.push_back(builder.MakeIntsAttribute("pads", pads));
    }

    if (!strides.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("strides", strides));
    }

    if (!dilations.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("dilations", dilations));
    }

    if (output_shape.has_value()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("output_shape", output_shape.value()));
    }

    // Conv -> conv_out (either final output, or feeds activation)
    const char* conv_out_name = output_activation.has_value() ? "conv_out" : "output";
    if (output_activation.has_value()) {
      builder.MakeOutput("output");
    } else {
      builder.MakeOutput(conv_out_name);
    }

    builder.AddNode("Conv",
                    conv_op_type,
                    conv_input_names,
                    {conv_out_name},
                    kOnnxDomain,
                    conv_attrs);

    if (output_activation.has_value()) {
      // conv_out -> activation -> output
      std::vector<std::string> activation_inputs{"conv_out"};
      for (size_t i = 0; i < output_activation->const_inputs.size(); ++i) {
        const std::string name = "act_const_" + std::to_string(i);
        builder.MakeScalarInitializer<float>(name, output_activation->const_inputs[i]);
        activation_inputs.push_back(name);
      }

      builder.AddNode("Activation",
                      output_activation->op_type,
                      activation_inputs,
                      {"output"},
                      kOnnxDomain);
    }
  };
}

// Creates a graph with a single Q/DQ Conv operator with mismatched bias scales to test bias requantization.
template <typename ActivationQType, typename WeightQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQConvBiasRequantTestCase(
    const std::string& conv_op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    std::optional<int64_t> group,
    const std::string& auto_pad = "NOTSET",
    bool use_contrib_qdq = false) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, group, auto_pad, use_contrib_qdq](ModelTestBuilder& builder,
                                                       std::vector<QuantParams<ActivationQType>>& output_qparams) {
    std::vector<std::string> conv_input_names;

    // input -> Q/DQ ->
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<ActivationQType> input_qparams = GetTestInputQuantParams<ActivationQType>(input_def);
    std::string input_qdq = AddQDQNodePair<ActivationQType>(builder, "input_qdq", "input", input_qparams.scale, input_qparams.zero_point,
                                                            use_contrib_qdq);
    conv_input_names.push_back(input_qdq);

    // weights -> Q/DQ ->
    MakeTestInput<float>(builder, "weights", weights_def);
    QuantParams<WeightQType> weights_qparams = GetTestInputQuantParams<WeightQType>(weights_def);
    std::string weights_qdq = AddQDQNodePair<WeightQType>(builder, "weights_qdq", "weights", weights_qparams.scale,
                                                          weights_qparams.zero_point, use_contrib_qdq);
    conv_input_names.push_back(weights_qdq);

    // bias -> Create bias with MISMATCHED scale to trigger requantization
    if (!bias_def.GetShape().empty()) {
      // Intentionally use a WRONG bias scale that doesn't match (input_scale * weight_scale)
      // This should trigger the bias requantization logic in QNN EP
      const float correct_bias_scale = input_qparams.scale * weights_qparams.scale;
      const float wrong_bias_scale = correct_bias_scale * 2.5f;  // Intentionally wrong scale

      conv_input_names.push_back(MakeTestQDQBiasInput(builder, "bias", bias_def, wrong_bias_scale, use_contrib_qdq));
    }

    // Conv attrs (must be provided at node creation)
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", auto_pad));

    if (group.has_value()) {
      conv_attrs.push_back(builder.MakeScalarAttribute("group", group.value()));
    }

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_attrs.push_back(builder.MakeIntsAttribute("pads", pads));
    }
    if (!strides.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("strides", strides));
    }
    if (!dilations.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("dilations", dilations));
    }

    builder.AddNode("Conv",
                    conv_op_type,
                    conv_input_names,
                    {"conv_output"},
                    kOnnxDomain,
                    conv_attrs);

    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(builder, "output_qdq", "conv_output", output_qparams[0].scale,
                                                           output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Creates a graph with a single Q/DQ Conv operator with per-channel weights and mismatched bias scales to test bias requantization.
template <typename ActivationQType, typename WeightQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQConvPerChannelBiasRequantTestCase(
    const std::string& conv_op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    int64_t weight_quant_axis,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    std::optional<int64_t> group,
    const std::string& auto_pad = "NOTSET",
    bool use_contrib_qdq = false) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, group, auto_pad, use_contrib_qdq,
          weight_quant_axis](ModelTestBuilder& builder,
                             std::vector<QuantParams<ActivationQType>>& output_qparams) {
    std::vector<std::string> conv_input_names;

    // input -> Q/DQ ->
    MakeTestInput<float>(builder, "input", input_def);
    const QuantParams<ActivationQType> input_qparams = GetTestInputQuantParams<ActivationQType>(input_def);
    conv_input_names.push_back(
        AddQDQNodePair<ActivationQType>(builder, "qdq_input", "input", input_qparams.scale, input_qparams.zero_point,
                                        use_contrib_qdq));

    // Quantized(weights) -> DQ -> (per-channel quantization)
    QNN_ASSERT(weights_def.IsInitializer() && weights_def.IsRawData());
    std::vector<float> weight_scales;
    std::vector<WeightQType> weight_zero_points;

    auto weights_shape = weights_def.GetShape();
    int64_t pos_weight_quant_axis = weight_quant_axis;
    if (pos_weight_quant_axis < 0) {
      pos_weight_quant_axis += static_cast<int64_t>(weights_shape.size());
    }

    GetTestInputQuantParamsPerChannel<WeightQType>(weights_def, weight_scales, weight_zero_points,
                                                   static_cast<size_t>(pos_weight_quant_axis), true);

    size_t num_weight_storage_elems = SizeOfShape(weights_shape);
    if constexpr (std::is_same_v<WeightQType, Int4x2> || std::is_same_v<WeightQType, UInt4x2>) {
      num_weight_storage_elems = Int4x2::CalcNumInt4Pairs(SizeOfShape(weights_shape));
    }

    std::vector<WeightQType> quantized_weights(num_weight_storage_elems);
    QuantizeValues<float, WeightQType>(weights_def.GetRawData(), quantized_weights,
                                       weights_def.GetShape(), weight_scales, weight_zero_points,
                                       pos_weight_quant_axis);

    builder.MakeInitializer<WeightQType>("weights_quant", weights_def.GetShape(), quantized_weights);

    std::vector<ONNX_NAMESPACE::AttributeProto> weights_dq_attrs;
    weights_dq_attrs.push_back(builder.MakeScalarAttribute("axis", weight_quant_axis));

    builder.AddDequantizeLinearNode(
        "WeightDQ",
        "weights_quant",
        weight_scales,
        weight_zero_points,
        "weights_dq",
        weights_dq_attrs,
        use_contrib_qdq);
    conv_input_names.push_back("weights_dq");

    // Quantized(bias) -> DQ -> (per-channel quantization with WRONG scales)
    if (!bias_def.GetShape().empty()) {
      // Create INTENTIONALLY WRONG bias scales that don't match input_scale * weight_scale[i]
      // This should cause QDQ to fail against CPU, but our requantization should fix it
      QNN_ASSERT(bias_def.IsInitializer() && bias_def.IsRawData());
      std::vector<float> wrong_bias_scales = weight_scales;
      std::vector<int32_t> bias_zero_points(weight_scales.size(), 0);

      // Apply wrong scaling factors to each channel - this will make QDQ fail without requantization
      for (size_t i = 0; i < wrong_bias_scales.size(); i++) {
        // Use different wrong multipliers for each channel to make it really wrong
        float wrong_multiplier = 1.5f + (i * 0.3f);  // 1.5, 1.8, 2.1, etc.
        wrong_bias_scales[i] = (input_qparams.scale * weight_scales[i]) * wrong_multiplier;
      }

      auto bias_shape = bias_def.GetShape();
      std::vector<int32_t> quantized_biases(SizeOfShape(bias_shape));
      QuantizeValues<float, int32_t>(bias_def.GetRawData(), quantized_biases,
                                     bias_def.GetShape(), wrong_bias_scales, bias_zero_points,
                                     0 /* axis */);

      builder.MakeInitializer<int32_t>("bias_quant", bias_def.GetShape(), quantized_biases);
      builder.MakeInitializer<float>("bias_scale", {static_cast<int64_t>(wrong_bias_scales.size())}, wrong_bias_scales);
      builder.MakeInitializer<int32_t>("bias_zp", {static_cast<int64_t>(bias_zero_points.size())}, bias_zero_points);

      std::vector<ONNX_NAMESPACE::AttributeProto> bias_dq_attrs;
      bias_dq_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));

      builder.AddNode("BiasDQ",
                      "DequantizeLinear",
                      {"bias_quant", "bias_scale", "bias_zp"},
                      {"bias_dq"},
                      use_contrib_qdq ? kMSDomain : kOnnxDomain,
                      bias_dq_attrs);
      conv_input_names.push_back("bias_dq");
    }

    // Conv attrs (must be provided at node creation)
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", auto_pad));

    if (group.has_value()) {
      conv_attrs.push_back(builder.MakeScalarAttribute("group", group.value()));
    }

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_attrs.push_back(builder.MakeIntsAttribute("pads", pads));
    }
    if (!strides.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("strides", strides));
    }
    if (!dilations.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("dilations", dilations));
    }

    const char* conv_out_name = "Y";
    builder.AddNode("Conv",
                    conv_op_type,
                    conv_input_names,
                    {conv_out_name},
                    kOnnxDomain,
                    conv_attrs);

    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(
        builder, "qdq_out", conv_out_name, output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Runs a Conv model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
static void RunConvOpTest(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                          const TestInputDef<float>& weights_def,
                          const TestInputDef<float>& bias_def,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& pads,
                          const std::vector<int64_t>& dilations,
                          std::optional<int64_t> group,
                          const std::string& auto_pad,
                          ExpectedEPNodeAssignment expected_ep_assignment,
                          const std::string& backend_name = "cpu",
                          int opset = 13,
                          float fp32_abs_err = 1e-5f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  auto build_fn = BuildF32ConvTestCase(conv_op_type, input_def, weights_def, bias_def, strides, pads,
                                       dilations, group, auto_pad);
  RunQnnModelTest(build_fn,
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}

// Creates a graph with a single Q/DQ Conv operator. Used for testing HTP backend.
template <typename ActivationQType, typename WeightQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQConvTestCase(
    const std::string& conv_op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    std::optional<int64_t> group,
    const std::string& auto_pad = "NOTSET",
    bool use_contrib_qdq = false,
    std::optional<OutputActivationInfo> output_activation = std::nullopt,
    std::optional<std::vector<int64_t>> output_shape = std::nullopt,
    bool use_float_bias = false) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, group, auto_pad,
          use_contrib_qdq, use_float_bias, output_activation, output_shape](ModelTestBuilder& builder,
                                                                            std::vector<QuantParams<ActivationQType>>& output_qparams) {
    std::vector<std::string> conv_input_names;

    // input -> Q/DQ ->
    MakeTestInput<float>(builder, "input", input_def);
    const QuantParams<ActivationQType> input_qparams = GetTestInputQuantParams<ActivationQType>(input_def);
    conv_input_names.push_back(
        AddQDQNodePair<ActivationQType>(builder, "qdq_input", "input", input_qparams.scale, input_qparams.zero_point,
                                        use_contrib_qdq));

    // weights -> Q/DQ ->
    MakeTestInput<float>(builder, "weights", weights_def);
    const QuantParams<WeightQType> weights_qparams = GetTestInputQuantParams<WeightQType>(weights_def);
    conv_input_names.push_back(
        AddQDQNodePair<WeightQType>(builder, "qdq_weights", "weights", weights_qparams.scale, weights_qparams.zero_point,
                                    use_contrib_qdq));

    // bias ->
    if (!bias_def.GetShape().empty()) {
      if (use_float_bias) {
        ASSERT_TRUE(bias_def.IsInitializer() && bias_def.IsRawData()) << "Float bias must be an initializer with raw data";
        builder.MakeInitializer<float>("bias", bias_def.GetShape(), bias_def.GetRawData());
        conv_input_names.push_back("bias");
      } else {
        // Bias requirement taken from python quantization tool: onnx_quantizer.py::quantize_bias_static()
        const float bias_scale = input_qparams.scale * weights_qparams.scale;
        conv_input_names.push_back(MakeTestQDQBiasInput(builder, "bias", bias_def, bias_scale, use_contrib_qdq));
      }
    }

    // Conv attrs (must be provided at node creation)
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", auto_pad));

    if (group.has_value()) {
      conv_attrs.push_back(builder.MakeScalarAttribute("group", group.value()));
    }

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_attrs.push_back(builder.MakeIntsAttribute("pads", pads));
    }
    if (!strides.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("strides", strides));
    }
    if (!dilations.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("dilations", dilations));
    }
    if (output_shape.has_value()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("output_shape", output_shape.value()));
    }

    const char* conv_out_name = output_activation.has_value() ? "conv_out" : "Y";
    builder.AddNode("Conv",
                    conv_op_type,
                    conv_input_names,
                    {conv_out_name},
                    kOnnxDomain,
                    conv_attrs);

    // (optional) activation after Conv
    std::string q_input_name = conv_out_name;
    if (output_activation.has_value()) {
      std::vector<std::string> activation_inputs{"conv_out"};
      for (size_t i = 0; i < output_activation->const_inputs.size(); ++i) {
        const std::string name = "act_const_" + std::to_string(i);
        builder.MakeScalarInitializer<float>(name, output_activation->const_inputs[i]);
        activation_inputs.push_back(name);
      }

      builder.AddNode("Activation",
                      output_activation->op_type,
                      activation_inputs,
                      {"act_out"},
                      kOnnxDomain);
      q_input_name = "act_out";
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(
        builder, "qdq_out", q_input_name, output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
  };
}

template <typename ActivationQType, typename WeightQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQPerChannelConvTestCase(
    const std::string& conv_op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    int64_t weight_quant_axis,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    std::optional<int64_t> group,
    const std::string& auto_pad = "NOTSET",
    bool use_contrib_qdq = false,
    std::optional<OutputActivationInfo> output_activation = std::nullopt,
    bool use_float_bias = false) {
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, group, auto_pad, use_contrib_qdq, use_float_bias,
          weight_quant_axis, output_activation](ModelTestBuilder& builder,
                                                std::vector<QuantParams<ActivationQType>>& output_qparams) {
    std::vector<std::string> conv_input_names;

    // input -> Q/DQ ->
    MakeTestInput<float>(builder, "input", input_def);
    const QuantParams<ActivationQType> input_qparams = GetTestInputQuantParams<ActivationQType>(input_def);
    conv_input_names.push_back(
        AddQDQNodePair<ActivationQType>(builder, "qdq_input", "input", input_qparams.scale, input_qparams.zero_point,
                                        use_contrib_qdq));

    // Quantized(weights) -> DQ ->
    QNN_ASSERT(weights_def.IsInitializer() && weights_def.IsRawData());
    std::vector<float> weight_scales;
    std::vector<WeightQType> weight_zero_points;

    auto weights_shape = weights_def.GetShape();
    int64_t pos_weight_quant_axis = weight_quant_axis;
    if (pos_weight_quant_axis < 0) {
      pos_weight_quant_axis += static_cast<int64_t>(weights_shape.size());
    }

    GetTestInputQuantParamsPerChannel<WeightQType>(weights_def, weight_scales, weight_zero_points,
                                                   static_cast<size_t>(pos_weight_quant_axis), true);

    size_t num_weight_storage_elems = SizeOfShape(weights_shape);
    if constexpr (std::is_same_v<WeightQType, Int4x2> || std::is_same_v<WeightQType, UInt4x2>) {
      num_weight_storage_elems = Int4x2::CalcNumInt4Pairs(SizeOfShape(weights_shape));
    }

    std::vector<WeightQType> quantized_weights(num_weight_storage_elems);
    QuantizeValues<float, WeightQType>(weights_def.GetRawData(), quantized_weights,
                                       weights_def.GetShape(), weight_scales, weight_zero_points,
                                       pos_weight_quant_axis);

    builder.MakeInitializer<WeightQType>("weights_quant", weights_def.GetShape(), quantized_weights);

    std::vector<ONNX_NAMESPACE::AttributeProto> weights_dq_attrs;
    weights_dq_attrs.push_back(builder.MakeScalarAttribute("axis", weight_quant_axis));

    builder.AddDequantizeLinearNode(
        "WeightDQ",
        "weights_quant",
        weight_scales,
        weight_zero_points,
        "weights_dq",
        weights_dq_attrs,
        use_contrib_qdq);
    conv_input_names.push_back("weights_dq");

    // bias ->
    if (!bias_def.GetShape().empty()) {
      QNN_ASSERT(bias_def.IsInitializer() && bias_def.IsRawData());
      if (use_float_bias) {
        builder.MakeInitializer<float>("bias", bias_def.GetShape(), bias_def.GetRawData());
        conv_input_names.push_back("bias");
      } else {
        // Quantized(bias) -> DQ -> (per-channel quantization)
        // bias_scale = input_scale * weight_scale (per-channel)
        std::vector<float> bias_scales(weight_scales);
        for (float& s : bias_scales) {
          s *= input_qparams.scale;
        }

        std::vector<int32_t> bias_zero_points(bias_scales.size(), 0);
        auto bias_shape = bias_def.GetShape();

        std::vector<int32_t> quantized_biases(SizeOfShape(bias_shape));
        QuantizeValues<float, int32_t>(bias_def.GetRawData(), quantized_biases,
                                       bias_def.GetShape(), bias_scales, bias_zero_points,
                                       0 /* axis */);

        builder.MakeInitializer<int32_t>("bias_quant", bias_def.GetShape(), quantized_biases);
        builder.MakeInitializer<float>("bias_scale", {static_cast<int64_t>(bias_scales.size())}, bias_scales);
        builder.MakeInitializer<int32_t>("bias_zp", {static_cast<int64_t>(bias_zero_points.size())}, bias_zero_points);

        std::vector<ONNX_NAMESPACE::AttributeProto> bias_dq_attrs;
        bias_dq_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));

        builder.AddNode("BiasDQ",
                        "DequantizeLinear",
                        {"bias_quant", "bias_scale", "bias_zp"},
                        {"bias_dq"},
                        use_contrib_qdq ? kMSDomain : kOnnxDomain,
                        bias_dq_attrs);
        conv_input_names.push_back("bias_dq");
      }
    }

    // Conv attrs (must be provided at node creation)
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", auto_pad));

    if (group.has_value()) {
      conv_attrs.push_back(builder.MakeScalarAttribute("group", group.value()));
    }

    if (!pads.empty() && auto_pad == "NOTSET") {
      conv_attrs.push_back(builder.MakeIntsAttribute("pads", pads));
    }
    if (!strides.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("strides", strides));
    }
    if (!dilations.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("dilations", dilations));
    }

    const char* conv_out_name = output_activation.has_value() ? "conv_out" : "Y";
    builder.AddNode("Conv",
                    conv_op_type,
                    conv_input_names,
                    {conv_out_name},
                    kOnnxDomain,
                    conv_attrs);

    // (optional) activation after Conv
    std::string q_input_name = conv_out_name;
    if (output_activation.has_value()) {
      std::vector<std::string> activation_inputs{"conv_out"};
      for (size_t i = 0; i < output_activation->const_inputs.size(); ++i) {
        const std::string name = "act_const_" + std::to_string(i);
        builder.MakeScalarInitializer<float>(name, output_activation->const_inputs[i]);
        activation_inputs.push_back(name);
      }

      builder.AddNode("Activation",
                      output_activation->op_type,
                      activation_inputs,
                      {"act_out"},
                      kOnnxDomain);
      q_input_name = "act_out";
    }

    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(
        builder, "qdq_out", q_input_name, output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Runs a Conv model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename ActivationQType, typename WeightQType>
static void RunHTPConvOpTest(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                             const TestInputDef<float>& weights_def,
                             const TestInputDef<float>& bias_def,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& pads,
                             const std::vector<int64_t>& dilations,
                             std::optional<int64_t> group,
                             const std::string& auto_pad,
                             ExpectedEPNodeAssignment expected_ep_assignment,
                             bool use_contrib_qdq = false,
                             int opset = 13,
                             QDQTolerance tolerance = QDQTolerance(),
                             std::optional<OutputActivationInfo> output_activation = std::nullopt,
                             std::optional<std::vector<int64_t>> output_shape = std::nullopt) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildF32ConvTestCase(conv_op_type, input_def, weights_def, bias_def, strides, pads, dilations,
                                            group, auto_pad, output_activation, output_shape),
                       BuildQDQConvTestCase<ActivationQType, WeightQType>(conv_op_type, input_def, weights_def,
                                                                          bias_def, strides, pads, dilations,
                                                                          group, auto_pad, use_contrib_qdq,
                                                                          output_activation, output_shape),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

// Runs a QDQ Conv model (per-axis quantization on weight/bias) on the QNN HTP backend.
// Checks the graph node assignment, and that inference outputs for QNN EP and CPU EP match.
template <typename ActivationQType, typename WeightQType>
static void RunHTPConvOpPerChannelTest(const std::string& conv_op_type, const TestInputDef<float>& input_def,
                                       const TestInputDef<float>& weights_def,
                                       const TestInputDef<float>& bias_def,
                                       int64_t weight_quant_axis,
                                       const std::vector<int64_t>& strides,
                                       const std::vector<int64_t>& pads,
                                       const std::vector<int64_t>& dilations,
                                       std::optional<int64_t> group,
                                       const std::string& auto_pad,
                                       ExpectedEPNodeAssignment expected_ep_assignment,
                                       bool use_contrib_qdq = false,
                                       int opset = 13,
                                       QDQTolerance tolerance = QDQTolerance(),
                                       std::optional<OutputActivationInfo> output_activation = std::nullopt) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto f32_fn = BuildF32ConvTestCase(conv_op_type, input_def, weights_def, bias_def, strides, pads, dilations,
                                     group, auto_pad, output_activation);
  auto qdq_fn = BuildQDQPerChannelConvTestCase<ActivationQType, WeightQType>(conv_op_type, input_def, weights_def,
                                                                             bias_def, weight_quant_axis, strides,
                                                                             pads, dilations, group, auto_pad,
                                                                             use_contrib_qdq, output_activation);
  TestQDQModelAccuracy(f32_fn, qdq_fn, provider_options, opset, expected_ep_assignment, tolerance);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as a dynamic input.
// TODO: Segfaults when calling graphFinalize(). v2.13
// fixed by QNN 2.32
TEST_F(QnnCPUBackendTests, Convf32_dynamic_bias) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, false, -1.0f, 1.0f),           // Random dynamic bias
                {1, 1},                                                 // default strides
                {0, 0, 0, 0},                                           // default pads
                {1, 1},                                                 // default dilations
                1,                                                      // default group
                "NOTSET",                                               // No auto-padding
                ExpectedEPNodeAssignment::All);

  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, false, -1.0f, 1.0f),              // Random dynamic bias
                {1, 1, 1},                                                 // default strides
                {0, 0, 0, 0, 0, 0},                                        // default pads
                {1, 1, 1},                                                 // default dilations
                1,                                                         // default group
                "NOTSET",                                                  // No auto-padding
                ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnCPUBackendTests, Convf32_bias_initializer) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // default strides
                {0, 0, 0, 0},                                           // default pads
                {1, 1},                                                 // default dilations
                1,                                                      // default group
                "NOTSET",                                               // No auto-padding
                ExpectedEPNodeAssignment::All);

  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                {1, 1, 1},                                                 // default strides
                {0, 0, 0, 0, 0, 0},                                        // default pads
                {1, 1, 1},                                                 // default dilations
                1,                                                         // default group
                "NOTSET",                                                  // No auto-padding
                ExpectedEPNodeAssignment::All);
}

// Tests Conv's auto_pad value "SAME_UPPER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, Convf32_AutoPadUpper) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2}, true, -1.0f, 1.0f),   // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // strides
                {},                                                     // pads
                {1, 1},                                                 // dilations
                1,                                                      // default group
                "SAME_UPPER",                                           // auto_pad
                ExpectedEPNodeAssignment::All);

  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2, 2}, true, -1.0f, 1.0f),   // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                {1, 1, 1},                                                 // strides
                {},                                                        // pads
                {1, 1, 1},                                                 // dilations
                1,                                                         // default group
                "SAME_UPPER",                                              // auto_pad
                ExpectedEPNodeAssignment::All);
}

// Tests ConvTranspose's auto_pad value "SAME_UPPER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, ConvTransposef32_AutoPadUpper) {
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 1, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({1, 2, 2, 2}, true, -1.0f, 1.0f),   // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // strides
                {},                                                     // pads
                {1, 1},                                                 // dilations
                1,                                                      // default group
                "SAME_UPPER",                                           // auto_pad
                ExpectedEPNodeAssignment::All);

  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 1, 3, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({1, 2, 2, 2, 2}, true, -1.0f, 1.0f),   // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                {1, 1, 1},                                                 // strides
                {},                                                        // pads
                {1, 1, 1},                                                 // dilations
                1,                                                         // default group
                "SAME_UPPER",                                              // auto_pad
                ExpectedEPNodeAssignment::All);
}

// Tests Conv's auto_pad value "SAME_LOWER" (compares to CPU EP).
TEST_F(QnnCPUBackendTests, Convf32_AutoPadLower) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2}, false, -1.0f, 1.0f),  // Random dynamic weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // strides
                {},                                                     // pads
                {1, 1},                                                 // dilations
                1,                                                      // default group
                "SAME_LOWER",                                           // auto_pad
                ExpectedEPNodeAssignment::All);

  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2, 2}, false, -1.0f, 1.0f),  // Random dynamic weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                {1, 1, 1},                                                 // strides
                {},                                                        // pads
                {1, 1, 1},                                                 // dilations
                1,                                                         // default group
                "SAME_LOWER",                                              // auto_pad
                ExpectedEPNodeAssignment::All);
}

// Tests ConvTranspose's auto_pad value "SAME_LOWER" (compares to CPU EP).
// 2.31 Exception from qnn_interface.graphAddNode
// unknown file: error: SEH exception with code 0xc0000005 thrown in the test body
// fixed by QNN 2.32
TEST_F(QnnCPUBackendTests, ConvTransposef32_AutoPadLower) {
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 1, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({1, 2, 2, 2}, false, -1.0f, 1.0f),  // Random dynamic weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // strides
                {},                                                     // pads
                {1, 1},                                                 // dilations
                1,                                                      // default group
                "SAME_LOWER",                                           // auto_pad
                ExpectedEPNodeAssignment::All);
}

// Tests ConvTranspose's auto_pad value "SAME_LOWER" (compares to CPU EP).
// Exception from graphFinalize
// Exception thrown at 0x00007FFFB7651630 (QnnCpu.dll) in onnxruntime_test_all.exe:
// 0xC0000005: Access violation reading location 0x0000000000000000.
// fixed by QNN 2.32
TEST_F(QnnCPUBackendTests, ConvTranspose3D_f32_AutoPadLower) {
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 1, 3, 3, 3}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({1, 2, 2, 2, 2}, false, -1.0f, 1.0f),  // Random dynamic weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                {1, 1, 1},                                                 // strides
                {},                                                        // pads
                {1, 1, 1},                                                 // dilations
                1,                                                         // default group
                "SAME_LOWER",                                              // auto_pad
                ExpectedEPNodeAssignment::All);
}

// large input,output, pads
TEST_F(QnnCPUBackendTests, Convf32_large_input1_pad_bias_initializer) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 3, 60, 452}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({16, 3, 3, 3}, true, 0.0f, 1.0f),      // Random dynamic weights
                TestInputDef<float>({16}, true, -1.0f, 1.0f),              // Random static bias
                {1, 1},
                {1, 1, 1, 1},
                {1, 1},
                1,  // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "cpu",
                13,
                1e-4f);

  RunConvOpTest("Conv",
                TestInputDef<float>({1, 3, 60, 452, 20}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({16, 3, 3, 3, 3}, true, 0.0f, 1.0f),       // Random dynamic weights
                TestInputDef<float>({16}, true, -1.0f, 1.0f),                  // Random static bias
                {1, 1, 1},
                {1, 1, 1, 1, 1, 1},
                {1, 1, 1},
                1,  // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "cpu",
                13,
                2e-4f);
}

TEST_F(QnnCPUBackendTests, Convf32_large_input2_nopad_bias_initializer) {
#if defined(_WIN32)
  // Tolerance needs to be > 1.52588e-05 on Windows x64
  // TODO: Investigate why
  float fp32_abs_err = 1e-4f;
#else
  float fp32_abs_err = 1e-5f;  // default value
#endif

  RunConvOpTest("Conv",
                TestInputDef<float>({1, 32, 16, 113}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({16, 32, 1, 1}, false, -1.0f, 1.0f),    // Random dynamic weights
                TestInputDef<float>({16}, true, -1.0f, 1.0f),               // Random static bias
                {1, 1},
                {0, 0, 0, 0},
                {1, 1},
                1,  // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "cpu",
                13,  // opset
                fp32_abs_err);

  RunConvOpTest("Conv",
                TestInputDef<float>({1, 32, 16, 113, 12}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({16, 32, 1, 1, 1}, false, -1.0f, 1.0f),     // Random dynamic weights
                TestInputDef<float>({16}, true, -1.0f, 1.0f),                   // Random static bias
                {1, 1, 1},
                {0, 0, 0, 0, 0, 0},
                {1, 1, 1},
                1,  // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "cpu",
                13,  // opset
                fp32_abs_err);
}

// Test 1D Conv with static weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, Conv1Df32_StaticWeights_DefaultBias) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 2, 4}, false, input_data),               // Dynamic input
                TestInputDef<float>({1, 2, 2}, true, {1.0f, 2.0f, 3.0f, 4.0f}),  // Static weights
                TestInputDef<float>({1}, true, {1.0f}),                          // Initializer Bias
                {1},                                                             // Strides
                {0, 0},                                                          // Pads
                {1},                                                             // Dilations
                1,                                                               // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All);
}

// Test 1D Conv with dynamic weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, Conv1Df32_DynamicWeights_DefaultBias) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 2, 4}, false, input_data),                // Dynamic input
                TestInputDef<float>({1, 2, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),  // Dynamic weights
                TestInputDef<float>(),                                            // Default bias
                {1},                                                              // Strides
                {0, 0},                                                           // Pads
                {1},                                                              // Dilations
                1,                                                                // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All);
}

// Test 1D ConvTranspose with static weights (implemented in QNN EP as 2D convolution with height of 1).
TEST_F(QnnCPUBackendTests, ConvTranspose1Df32_StaticWeights_DefaultBias) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 2, 4}, false, input_data),               // Dynamic input
                TestInputDef<float>({2, 1, 2}, true, {1.0f, 2.0f, 3.0f, 4.0f}),  // Static weights
                TestInputDef<float>({1}, true, {0.0f}),                          // Zero bias
                {1},                                                             // Strides
                {0, 0},                                                          // Pads
                {1},                                                             // Dilations
                1,                                                               // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All);
}

// Test 1D ConvTranspose with dynamic weights (implemented in QNN EP as 2D convolution with height of 1).
// 2.31 Exception from qnn_interface.graphAddNode
// unknown file: error: SEH exception with code 0xc0000005 thrown in the test body
// fixed by QNN 2.32
TEST_F(QnnCPUBackendTests, ConvTranspose1Df32_DynamicWeights_DefaultBias) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 2, 4}, false, input_data),                // Dynamic input
                TestInputDef<float>({2, 1, 2}, false, {1.0f, 2.0f, 3.0f, 4.0f}),  // Dynamic weights
                TestInputDef<float>({1}, true, {0.0f}),                           // Zero bias
                {1},                                                              // Strides
                {0, 0},                                                           // Pads
                {1},                                                              // Dilations
                1,                                                                // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All);
}

// Builds: weight_q0 (int8 init) -> DQ -> Q -> DQ -> Conv.
// Used to regression-test chained folding; differing scale0/scale1 exercises real requant
// on the intermediate STATIC tensor rather than a byte round-trip.
static GetTestModelFn BuildPerChannelQDQChainConstWeightConvTestCase(
    const std::vector<float>& scale0,
    const std::vector<int8_t>& zp0,
    const std::vector<float>& scale1,
    const std::vector<int8_t>& zp1) {
  return [scale0, zp0, scale1, zp1](ModelTestBuilder& builder) {
    constexpr int64_t out_ch = 2;
    constexpr int64_t in_ch = 3;
    const std::vector<int64_t> input_shape = {1, in_ch, 1, 1};
    const std::vector<int64_t> weight_shape = {out_ch, in_ch, 1, 1};

    builder.MakeInput<float>("input", input_shape, -1.0f, 1.0f);

    builder.MakeInitializer<int8_t>("weight_q0", weight_shape, std::vector<int8_t>{1, 2, 3, 4, 5, 6});
    builder.MakeInitializer<float>("scale0", {out_ch}, scale0);
    builder.MakeInitializer<int8_t>("zp0", {out_ch}, zp0);
    builder.MakeInitializer<float>("scale1", {out_ch}, scale1);
    builder.MakeInitializer<int8_t>("zp1", {out_ch}, zp1);

    std::vector<ONNX_NAMESPACE::AttributeProto> axis_attrs;
    axis_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));

    builder.AddNode("WeightDQ0", "DequantizeLinear", {"weight_q0", "scale0", "zp0"}, {"weight_dq0"},
                    kOnnxDomain, axis_attrs);
    builder.AddNode("WeightQ1", "QuantizeLinear", {"weight_dq0", "scale1", "zp1"}, {"weight_q1"},
                    kOnnxDomain, axis_attrs);
    builder.AddNode("WeightDQ1", "DequantizeLinear", {"weight_q1", "scale1", "zp1"}, {"weight_dq1"},
                    kOnnxDomain, axis_attrs);

    builder.MakeOutput("output");
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", "NOTSET"));
    conv_attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    conv_attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    conv_attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    conv_attrs.push_back(builder.MakeScalarAttribute("group", static_cast<int64_t>(1)));

    builder.AddNode("Conv", "Conv", {"input", "weight_dq1"}, {"output"}, kOnnxDomain, conv_attrs);
  };
}

TEST_F(QnnCPUBackendTests, Convf32_PerChannelQDQChainConstWeight_Regression) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildPerChannelQDQChainConstWeightConvTestCase(
                      /*scale0*/ {0.1f, 0.2f}, /*zp0*/ {0, 0},
                      /*scale1*/ {0.1f, 0.2f}, /*zp1*/ {0, 0}),
                  provider_options,
                  /*opset*/ 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});
}

TEST_F(QnnCPUBackendTests, Convf32_PerChannelQDQChainConstWeight_NonIdentity_Regression) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildPerChannelQDQChainConstWeightConvTestCase(
                      /*scale0*/ {0.1f, 0.2f}, /*zp0*/ {0, 0},
                      /*scale1*/ {0.05f, 0.4f}, /*zp1*/ {-2, 3}),
                  provider_options,
                  /*opset*/ 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});
}

// Builds: weight_q (int4 init) -> per-channel DQ -> Conv. QNN cannot represent a standalone
// per-channel DQ, so it is folded to an fp32 static by decoding the initializer's bytes as int8.
// Weights span the negative INT4 range, which a sign-handling regression would decode as q + 16.
static GetTestModelFn BuildPerChannelInt4DQConstWeightConvTestCase(const std::vector<float>& scales) {
  return [scales](ModelTestBuilder& builder) {
    constexpr int64_t out_ch = 2;
    constexpr int64_t in_ch = 3;
    const std::vector<int64_t> input_shape = {1, in_ch, 1, 1};
    const std::vector<int64_t> weight_shape = {out_ch, in_ch, 1, 1};
    const std::vector<int8_t> weight_values{-8, -7, -1, 1, 5, 7};

    builder.MakeInput<float>("input", input_shape, -1.0f, 1.0f);

    std::vector<Int4x2> weight_data(Int4x2::CalcNumInt4Pairs(weight_values.size()));
    for (size_t i = 0; i < weight_values.size(); ++i) {
      weight_data[i >> 1].SetElem(i & 1, weight_values[i]);
    }
    builder.MakeInitializer<Int4x2>("weight_q", weight_shape, weight_data);
    builder.MakeInitializer<float>("weight_scale", {out_ch}, scales);

    std::vector<ONNX_NAMESPACE::AttributeProto> axis_attrs;
    axis_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));
    builder.AddNode("WeightDQ", "DequantizeLinear", {"weight_q", "weight_scale"}, {"weight_dq"},
                    kOnnxDomain, axis_attrs);

    builder.MakeOutput("output");
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", "NOTSET"));
    conv_attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    conv_attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    conv_attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    conv_attrs.push_back(builder.MakeScalarAttribute("group", static_cast<int64_t>(1)));
    builder.AddNode("Conv", "Conv", {"input", "weight_dq"}, {"output"}, kOnnxDomain, conv_attrs);
  };
}

TEST_F(QnnCPUBackendTests, Convf32_PerChannelInt4DQConstWeight_SignRegression) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildPerChannelInt4DQConstWeightConvTestCase(/*scales*/ {0.1f, 0.2f}),
                  provider_options,
                  /*opset*/ 21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-4f)});
}

// Tests for reuse_sparse_indices parameter (always false, verifies the parameter is accepted by QNN without errors).
// Conv2d: reuse_sparse_indices should be added to the QNN node parameters.
TEST_F(QnnCPUBackendTests, Conv2D_ReuseSparseIndices) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 2, 5, 5}, false, -10.0f, 10.0f),  // Dynamic input
                TestInputDef<float>({4, 2, 3, 3}, true, -1.0f, 1.0f),     // Static weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),              // Static bias
                {1, 1},                                                   // Strides
                {0, 0, 0, 0},                                             // Pads
                {1, 1},                                                   // Dilations
                1,                                                        // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All);
}

// Conv3d: reuse_sparse_indices should be added using QNN_OP_CONV_3D_PARAM_REUSE_SPARSE_INDICIES.
TEST_F(QnnCPUBackendTests, Conv3D_ReuseSparseIndices) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 2, 4, 4, 4}, false, -10.0f, 10.0f),  // Dynamic input
                TestInputDef<float>({4, 2, 2, 2, 2}, true, -1.0f, 1.0f),     // Static weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),                 // Static bias
                {1, 1, 1},                                                   // Strides
                {0, 0, 0, 0, 0, 0},                                          // Pads
                {1, 1, 1},                                                   // Dilations
                1,                                                           // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All);
}

// DepthwiseConv2d: reuse_sparse_indices should NOT be added (group == input_channels == output_channels).
TEST_F(QnnCPUBackendTests, DepthwiseConv2D_NoReuseSparseIndices) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 4, 5, 5}, false, -10.0f, 10.0f),  // Dynamic input
                TestInputDef<float>({4, 1, 3, 3}, true, -1.0f, 1.0f),     // Depthwise weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),              // Static bias
                {1, 1},                                                   // Strides
                {0, 0, 0, 0},                                             // Pads
                {1, 1},                                                   // Dilations
                4,                                                        // group == input_channels == output_channels -> DepthwiseConv2d
                "NOTSET",
                ExpectedEPNodeAssignment::All);
}

// ConvTranspose: reuse_sparse_indices should NOT be added.
TEST_F(QnnCPUBackendTests, ConvTranspose2D_NoReuseSparseIndices) {
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 2, 4, 4}, false, -10.0f, 10.0f),  // Dynamic input
                TestInputDef<float>({2, 4, 3, 3}, true, -1.0f, 1.0f),     // Static weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),              // Static bias
                {1, 1},                                                   // Strides
                {0, 0, 0, 0},                                             // Pads
                {1, 1},                                                   // Dilations
                1,                                                        // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// The bug is from a QDQ model, and Conv node gets processed before it's producer Mul node
// A Transpose node gets inserted between Mul and the dynamic weight tensor shape on Conv
// to make Conv weight with shape HWNC
// However it changes Mul output shape to HWNC and cause issue
// It has to be QDQ model, because the DQ node with initializer on Conv gets processed first
// and DQ node requires its node unit to be processed
// So, Conv gets processed before Mul node
//
// Since at least QAIRT 2.33 value pair (3.549, 3.588) at index #12709 don't match, which is 0.039 from 3.549
TEST_F(QnnHTPBackendTests, DISABLED_Test_QDQConvWithDynamicWeightsFromMul) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto BuildConvMulGraph = [](ModelTestBuilder& builder) {
    // DQ node for Conv input
    builder.MakeInitializer<uint8_t>("conv_q_input", {1, 32, 16, 113}, static_cast<uint8_t>(0), static_cast<uint8_t>(127));
    builder.AddNode("ConvInputDQ",
                    "DequantizeLinear",
                    {"conv_q_input"},
                    {"dq_i_output"},
                    kOnnxDomain);

    // DQ node for Conv bias
    builder.MakeInitializer<int32_t>("bias_q", {16}, static_cast<int32_t>(0), static_cast<int32_t>(127));
    builder.AddNode("BiasDQ",
                    "DequantizeLinear",
                    {"bias_q"},
                    {"dq_bias_output"},
                    kOnnxDomain);

    // Mul node: DQ(mul_input1) * DQ(mul_input2)
    builder.MakeInput<uint8_t>("mul_q_input1", {16, 32, 1, 1}, static_cast<uint8_t>(0), static_cast<uint8_t>(127));
    builder.MakeInitializer<uint8_t>("mul_q_input2", {16, 1, 1, 1}, static_cast<uint8_t>(0), static_cast<uint8_t>(127));

    builder.AddNode("MulInput1DQ",
                    "DequantizeLinear",
                    {"mul_q_input1"},
                    {"mul_dq1_output"},
                    kOnnxDomain);

    builder.AddNode("MulInput2DQ",
                    "DequantizeLinear",
                    {"mul_q_input2"},
                    {"mul_dq2_output"},
                    kOnnxDomain);

    builder.AddNode("Mul",
                    "Mul",
                    {"mul_dq1_output", "mul_dq2_output"},
                    {"mul_output"},
                    kOnnxDomain);

    AddQDQNodePair<uint8_t>(builder, "qdq_mul_out", "mul_output", 0.03f, static_cast<uint8_t>(0));

    // Conv node
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", "NOTSET"));
    conv_attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    conv_attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    conv_attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));

    builder.AddNode("Conv",
                    "Conv",
                    {"dq_i_output", "qdq_mul_out_dq", "dq_bias_output"},
                    {"conv_output"},
                    kOnnxDomain,
                    conv_attrs);

    // Conv output -> Q
    builder.AddNode("ConvOutputQ",
                    "QuantizeLinear",
                    {"conv_output"},
                    {"q_output"},
                    kOnnxDomain);

    // Q -> DQ -> graph output
    builder.AddNode("ConvOutputDQ",
                    "DequantizeLinear",
                    {"q_output"},
                    {"output"},
                    kOnnxDomain);
    builder.MakeOutput("output");
  };

  RunQnnModelTest(BuildConvMulGraph,
                  provider_options,
                  13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

TEST_F(QnnHTPBackendTests, Convf32_PerChannelQDQChainConstWeight_Regression) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildPerChannelQDQChainConstWeightConvTestCase(
                      /*scale0*/ {0.1f, 0.2f}, /*zp0*/ {0, 0},
                      /*scale1*/ {0.1f, 0.2f}, /*zp1*/ {0, 0}),
                  provider_options,
                  /*opset*/ 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

TEST_F(QnnHTPBackendTests, Convf32_PerChannelQDQChainConstWeight_NonIdentity_Regression) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildPerChannelQDQChainConstWeightConvTestCase(
                      /*scale0*/ {0.1f, 0.2f}, /*zp0*/ {0, 0},
                      /*scale1*/ {0.05f, 0.4f}, /*zp1*/ {-2, 3}),
                  provider_options,
                  /*opset*/ 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// Smoke test for the enable_htp_fp16_clamp_overflow HTP option.
// The option is compiled unconditionally: on QAIRT < 2.49 the EP logs a warning
// and ignores it (see qnn_execution_provider.cc), so this degrades to a plain
// fp16 Conv smoke test; on QAIRT >= 2.49 it exercises the clamp-overflow path.
TEST_F(QnnHTPBackendTests, Conv_Fp16ClampOverflow_Smoke) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V75);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_htp_fp16_precision"] = "1";
  provider_options["enable_htp_fp16_clamp_overflow"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  auto input_def = TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-1.0f, 1.0f, 32));
  auto weights_def = TestInputDef<float>({2, 2, 2, 2}, true, GetFloatDataInRange(-1.0f, 1.0f, 16));
  auto bias_def = TestInputDef<float>({2}, true, {1.0f, -1.0f});

  RunQnnModelTest(BuildF32ConvTestCase("Conv", input_def, weights_def, bias_def,
                                       {1, 1},        // strides
                                       {0, 0, 0, 0},  // pads
                                       {1, 1},        // dilations
                                       1),            // group
                  provider_options,
                  /*opset*/ 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.01f)});
}

TEST_F(QnnHTPBackendTests, Convf32_PerChannelInt4DQConstWeight_SignRegression) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildPerChannelInt4DQConstWeightConvTestCase(/*scales*/ {0.1f, 0.2f}),
                  provider_options,
                  /*opset*/ 21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-3f)});
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as a dynamic input.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_bias_dynamic_input) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                                     TestInputDef<float>({1, 1, 3, 3}, true, -10.0f, 10.0f),  // Random static input
                                     TestInputDef<float>({1}, false, {2.0f}),                 // Dynamic bias
                                     {1, 1},                                                  // Strides
                                     {0, 0, 0, 0},                                            // Pads
                                     {1, 1},                                                  // Dilations
                                     1,                                                       // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13);    // opset

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                                     TestInputDef<float>({1, 1, 3, 3, 3}, true, -10.0f, 10.0f),  // Random static input
                                     TestInputDef<float>({1}, false, {2.0f}),                    // Dynamic bias
                                     {1, 1, 1},                                                  // Strides
                                     {0, 0, 0, 0, 0, 0},                                         // Pads
                                     {1, 1, 1},                                                  // Dilations
                                     1,                                                          // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13);    // opset
}

// Test per-channel QDQ Conv. in0: u8, in1 (weight): s8, in2 (bias): s32, out: u8
TEST_F(QnnHTPBackendTests, ConvU8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// Test per-channel QDQ Conv with INT4 weights. in0: u16, in1 (weight): s4, in2 (bias): s32, out: u8
TEST_F(QnnHTPBackendTests, ConvU16S4S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, Int4x2>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               false,  // use_qdq_contrib_ops
                                               21);    // opset
}

// Test bias requantization when bias scale doesn't match (weight_scale * activation_scale)
// This test uses a bias with intentionally wrong scale to trigger the requantization logic in QNN EP
TEST_F(QnnHTPBackendTests, ConvU8U8S32_BiasRequantization) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildF32ConvTestCase("Conv",
                                            TestInputDef<float>({1, 2, 4, 4}, false, -10.0f, 10.0f),  // Input
                                            TestInputDef<float>({3, 2, 2, 2}, true, -1.0f, 5.0f),     // Weights
                                            TestInputDef<float>({3}, true, -1.0f, 1.0f),              // Bias
                                            {1, 1},                                                   // Strides
                                            {0, 0, 0, 0},                                             // Pads
                                            {1, 1},                                                   // Dilations
                                            1,                                                        // Group
                                            "NOTSET"),                                                // Auto pad
                       BuildQDQConvBiasRequantTestCase<uint8_t, uint8_t>("Conv",
                                                                         TestInputDef<float>({1, 2, 4, 4}, false, -10.0f, 10.0f),  // Input
                                                                         TestInputDef<float>({3, 2, 2, 2}, true, -1.0f, 5.0f),     // Weights
                                                                         TestInputDef<float>({3}, true, -1.0f, 1.0f),              // Bias (will get wrong scale)
                                                                         {1, 1},                                                   // Strides
                                                                         {0, 0, 0, 0},                                             // Pads
                                                                         {1, 1},                                                   // Dilations
                                                                         1,                                                        // Group
                                                                         "NOTSET"),                                                // Auto pad
                       provider_options,
                       13,  // opset
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(0.015f));
}

// Test per-channel bias requantization when bias scales don't match (weight_scale[i] * activation_scale)
// This test uses a bias with intentionally wrong scales that would cause QDQ to fail against CPU,
// but the requantization logic should correct it, allowing the test to pass.
TEST_F(QnnHTPBackendTests, ConvU8S8S32_PerChannel_BiasRequantization) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestInputDef<float> input_def({1, 2, 4, 4}, false, -10.0f, 10.0f);
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  std::vector<int64_t> bias_shape = {3};
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  TestQDQModelAccuracy(BuildF32ConvTestCase("Conv",
                                            input_def,
                                            weight_def,
                                            bias_def,
                                            {1, 1},        // Strides
                                            {0, 0, 0, 0},  // Pads
                                            {1, 1},        // Dilations
                                            1,             // Group
                                            "NOTSET"),     // Auto pad
                       BuildQDQConvPerChannelBiasRequantTestCase<uint8_t, int8_t>("Conv",
                                                                                  input_def,
                                                                                  weight_def,
                                                                                  bias_def,
                                                                                  0,             // weight quant axis
                                                                                  {1, 1},        // Strides
                                                                                  {0, 0, 0, 0},  // Pads
                                                                                  {1, 1},        // Dilations
                                                                                  1,             // Group
                                                                                  "NOTSET",      // Auto pad
                                                                                  false),        // use_contrib_qdq
                       provider_options,
                       13,  // opset
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(0.015f));
}

// Tests QDQ Conv where activation and weight are per-tensor quantized but bias is a plain float
// initializer.
TEST_F(QnnHTPBackendTests, ConvU8U8_FloatBias) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestInputDef<float> input_def({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32));
  TestInputDef<float> weight_def({3, 2, 2, 2}, true, GetFloatDataInRange(-1.0f, 5.0f, 24));
  TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));

  TestQDQModelAccuracy(
      BuildF32ConvTestCase("Conv", input_def, weight_def, bias_def,
                           {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, "NOTSET"),
      BuildQDQConvTestCase<uint8_t, uint8_t>("Conv", input_def, weight_def, bias_def,
                                             {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, "NOTSET",
                                             /*use_contrib_qdq=*/false, std::nullopt, std::nullopt,
                                             /*use_float_bias=*/true),
      provider_options, 13, ExpectedEPNodeAssignment::All, QDQTolerance(0.015f));
}

// Tests QDQ Conv where activation is per-tensor quantized, weight is per-channel quantized (int8),
// and bias is a plain float initializer.
TEST_F(QnnHTPBackendTests, ConvU8S8_PerChannel_FloatBias) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestInputDef<float> input_def({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32));
  TestInputDef<float> weight_def({3, 2, 2, 2}, true, GetFloatDataInRange(-1.0f, 5.0f, 24));
  TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));

  TestQDQModelAccuracy(
      BuildF32ConvTestCase("Conv", input_def, weight_def, bias_def,
                           {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, "NOTSET"),
      BuildQDQPerChannelConvTestCase<uint8_t, int8_t>("Conv", input_def, weight_def, bias_def,
                                                      0,  // weight quant axis
                                                      {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, "NOTSET",
                                                      /*use_contrib_qdq=*/false, std::nullopt,
                                                      /*use_float_bias=*/true),
      provider_options, 13, ExpectedEPNodeAssignment::All, QDQTolerance(0.015f));
}

// Test per-channel QDQ Conv with INT4 weights and no bias.
// in0: u16, in1 (weight): s4, out: u8
// Tests bug in QNN SDK 2.25 when validating Conv without a bias (QNN EP adds a dummy bias).
TEST_F(QnnHTPBackendTests, ConvU16S4_PerChannel_NoBias) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, Int4x2>("Conv",
                                               input_def,
                                               weight_def,
                                               TestInputDef<float>(),
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               false,  // use_qdq_contrib_ops
                                               21);    // opset
}

// Test per-channel QDQ Conv with uint16 input[0], uint8 weights, and no bias.
// in0: u16, in1 (weight): s4, out: u8
// Tests bug in QNN SDK 2.25 when validating Conv without a bias (QNN EP adds a dummy bias).
TEST_F(QnnHTPBackendTests, ConvU16U8_PerTensor_NoBias) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));

  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      input_def,
                                      weight_def,
                                      TestInputDef<float>(),
                                      {1, 1},        // Strides
                                      {0, 0, 0, 0},  // Pads
                                      {1, 1},        // Dilations
                                      1,             // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      false,  // use_qdq_contrib_ops
                                      21);    // opset
}

#ifndef __linux__
// Test per-channel QDQ Conv with uint16 input[0], uint8 weights, and no bias.
// in0: u16, in1 (weight): s4, out: u8
// Tests bug in QNN SDK 2.25 when validating Conv without a bias (QNN EP adds a dummy bias).
TEST_F(QnnHTPBackendTests, ConvU16U16_PerTensor_NoBias) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));

  RunHTPConvOpTest<uint16_t, uint16_t>("Conv",
                                       input_def,
                                       weight_def,
                                       TestInputDef<float>(),
                                       {1, 1},        // Strides
                                       {0, 0, 0, 0},  // Pads
                                       {1, 1},        // Dilations
                                       1,             // default group
                                       "NOTSET",
                                       ExpectedEPNodeAssignment::All,
                                       false,  // use_qdq_contrib_ops
                                       21);    // opset
}
#endif

TEST_F(QnnHTPBackendTests, ConvU16S4_PerChannel_NoBias_LargeINT4Weight) {
  std::vector<int64_t> input_shape = {1, 3072, 1, 512};
  std::vector<int64_t> weight_shape = {9216, 3072, 1, 1};
  std::vector<float> input_data(SizeOfShape(input_shape), 0.1f);
  input_data[0] = 0.2f;
  std::vector<float> weight_data(SizeOfShape(weight_shape), -0.1f);
  for (size_t c = 0; c < static_cast<size_t>(weight_shape[0]); c++) {
    size_t i = c * 3072;
    weight_data[i] = 0.1f;
  }

  TestInputDef<float> input_def(input_shape, false, input_data);
  TestInputDef<float> weight_def(weight_shape, true, weight_data);

  RunHTPConvOpPerChannelTest<uint16_t, Int4x2>("Conv",
                                               input_def,
                                               weight_def,
                                               TestInputDef<float>(),
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               false,  // use_qdq_contrib_ops
                                               21);    // opset
}

// Test fusion of DQs -> Conv -> Relu/Clip -> Q.
// User per-tensor quantization.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_ReluClipFusion) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  // DQs -> Conv (w/ bias) -> Relu -> Q
  OutputActivationInfo relu_info = {"Relu", {}};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     input_def,
                                     weight_def,
                                     bias_def,
                                     {1, 1},        // Strides
                                     {0, 0, 0, 0},  // Pads
                                     {1, 1},        // Dilations
                                     1,             // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     21,     // opset
                                     QDQTolerance(),
                                     relu_info);

  // DQs -> Conv (NO bias) -> Relu -> Q
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     input_def,
                                     weight_def,
                                     TestInputDef<float>(),
                                     {1, 1},        // Strides
                                     {0, 0, 0, 0},  // Pads
                                     {1, 1},        // Dilations
                                     1,             // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     21,     // opset
                                     QDQTolerance(),
                                     relu_info);

  // DQs -> Conv (w/ bias) -> Clip -> Q
  // Opset 6 Clip uses attributes for min/max
  OutputActivationInfo clip_info = {"Clip", {0.0f, 2.0f}};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     input_def,
                                     weight_def,
                                     bias_def,
                                     {1, 1},        // Strides
                                     {0, 0, 0, 0},  // Pads
                                     {1, 1},        // Dilations
                                     1,             // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     19,     // opset
                                     QDQTolerance(),
                                     clip_info);

  // DQs -> Conv (NO bias) -> Clip -> Q
  OutputActivationInfo clip_info_2 = {"Clip", {-6.0f, 6.0f}};
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      input_def,
                                      weight_def,
                                      TestInputDef<float>(),
                                      {1, 1},        // Strides
                                      {0, 0, 0, 0},  // Pads
                                      {1, 1},        // Dilations
                                      1,             // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      false,  // use_qdq_contrib_ops
                                      21,     // opset
                                      QDQTolerance(),
                                      clip_info_2);
}

// Redundant Clip between Conv and Q in a QDQ model should be accepted on HTP.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_RedundantClipQDQ) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  OutputActivationInfo clip_info = {"Clip", {-2.0f, 2.0f}};
  auto f32_fn = BuildF32ConvTestCase("Conv",
                                     input_def,
                                     weight_def,
                                     bias_def,
                                     {1, 1},        // Strides
                                     {0, 0, 0, 0},  // Pads
                                     {1, 1},        // Dilations
                                     1,             // default group
                                     "NOTSET",
                                     clip_info);

  auto qdq_fn = [input_def, weight_def, bias_def](ModelTestBuilder& builder,
                                                  std::vector<QuantParams<uint8_t>>& output_qparams) {
    std::vector<std::string> conv_inputs;

    // input -> Q/DQ ->
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<uint8_t> input_qparams = GetTestInputQuantParams<uint8_t>(input_def);
    conv_inputs.push_back(AddQDQNodePair<uint8_t>(builder, "input_qdq", "input", input_qparams.scale,
                                                  input_qparams.zero_point, /*use_contrib_qdq=*/true));

    // weights -> Q/DQ ->
    MakeTestInput<float>(builder, "weights", weight_def);
    QuantParams<uint8_t> weights_qparams = GetTestInputQuantParams<uint8_t>(weight_def);
    conv_inputs.push_back(AddQDQNodePair<uint8_t>(builder, "weights_qdq", "weights", weights_qparams.scale,
                                                  weights_qparams.zero_point, /*use_contrib_qdq=*/true));

    // bias ->
    if (!bias_def.GetShape().empty()) {
      const float bias_scale = input_qparams.scale * weights_qparams.scale;
      conv_inputs.push_back(MakeTestQDQBiasInput(builder, "bias", bias_def, bias_scale,
                                                 /*use_contrib_qdq=*/true));
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", "NOTSET"));
    conv_attrs.push_back(builder.MakeIntsAttribute("pads", {0, 0, 0, 0}));
    conv_attrs.push_back(builder.MakeIntsAttribute("strides", {1, 1}));
    conv_attrs.push_back(builder.MakeIntsAttribute("dilations", {1, 1}));
    conv_attrs.push_back(builder.MakeScalarAttribute("group", static_cast<int64_t>(1)));

    const std::string conv_output = "conv_output";
    builder.AddNode("Conv", "Conv", conv_inputs, {conv_output}, "", conv_attrs);

    // Clip float min/max initializers -> Q -> DQ ->
    builder.MakeScalarInitializer<float>("clip_min_f", -2.0f);
    builder.MakeScalarInitializer<float>("clip_max_f", 2.0f);
    builder.AddQuantizeLinearNode<uint8_t>("clip_min_q", "clip_min_f", input_qparams.scale,
                                           input_qparams.zero_point, "clip_min_q_out");
    builder.AddQuantizeLinearNode<uint8_t>("clip_max_q", "clip_max_f", input_qparams.scale,
                                           input_qparams.zero_point, "clip_max_q_out");

    builder.AddDequantizeLinearNode<uint8_t>("clip_min_dq", "clip_min_q_out", input_qparams.scale,
                                             input_qparams.zero_point, "clip_min_dq_out");
    builder.AddDequantizeLinearNode<uint8_t>("clip_max_dq", "clip_max_q_out", input_qparams.scale,
                                             input_qparams.zero_point, "clip_max_dq_out");

    // Clip ->
    const std::string clip_output = "clip_output";
    builder.AddNode("Clip", "Clip", {conv_output, "clip_min_dq_out", "clip_max_dq_out"}, {clip_output});

    // Q -> output
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "output_qdq", clip_output, output_qparams[0].scale,
                                                   output_qparams[0].zero_point, /*use_contrib_qdq=*/true);
  };

  TestQDQModelAccuracy<uint8_t>(f32_fn,
                                qdq_fn,
                                provider_options,
                                13,  // opset
                                ExpectedEPNodeAssignment::All,
                                QDQTolerance());
}

// Test fusion of DQs -> Conv -> Relu/Clip -> Q.
// User per-channel quantization.
TEST_F(QnnHTPBackendTests, ConvS8S8S32_PerChannel_ReluClipFusion) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  // DQs -> Conv (w/ bias) -> Relu -> Q
  OutputActivationInfo relu_info = {"Relu", {}};
  RunHTPConvOpPerChannelTest<int8_t, int8_t>("Conv",
                                             input_def,
                                             weight_def,
                                             bias_def,
                                             0,             // weight quant axis
                                             {1, 1},        // Strides
                                             {0, 0, 0, 0},  // Pads
                                             {1, 1},        // Dilations
                                             1,             // default group
                                             "NOTSET",
                                             ExpectedEPNodeAssignment::All,
                                             false,  // use_qdq_contrib_ops
                                             21,     // opset
                                             QDQTolerance(),
                                             relu_info);

  // DQs -> Conv (w/ bias) -> Clip -> Q
  OutputActivationInfo clip_info = {"Clip", {0.0f, 6.0f}};
  RunHTPConvOpPerChannelTest<int8_t, int8_t>("Conv",
                                             input_def,
                                             weight_def,
                                             bias_def,
                                             0,             // weight quant axis
                                             {1, 1},        // Strides
                                             {0, 0, 0, 0},  // Pads
                                             {1, 1},        // Dilations
                                             1,             // default group
                                             "NOTSET",
                                             ExpectedEPNodeAssignment::All,
                                             false,  // use_qdq_contrib_ops
                                             21,     // opset
                                             QDQTolerance(),
                                             clip_info);
}

// Test per-channel QDQ Conv with INT4 weights and a negative weight quantization axis that still points to dimension 0.
TEST_F(QnnHTPBackendTests, ConvU16S4S32_PerChannel_NegativeWeightQuantAxis) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(0.0f, 1.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, Int4x2>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              -4,            // negative weight quant axis (same as 0)
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              21);    // opset
}

// Test per-channel QDQ Conv with INT4 weights. in0: u16, in1 (weight): s4, in2 (bias): s32, out: u8
// TODO(adrianlizarraga): Investigate inaccuracy for QNN EP.
//
// Output values for all EPs:
// CPU EP (f32 model): 25.143 21.554 17.964 10.785 7.195 3.605  -3.574  -7.164  -10.753
// CPU EP (qdq model): 24.670 21.103 17.536 10.254 6.689 2.972  -4.161  -7.728  -10.700
// QNN EP (qdq model): 27.186 27.186 27.186 21.541 6.685 -8.022 -10.548 -10.548 -10.548
TEST_F(QnnHTPBackendTests, ConvU16S4S32_PerChannel_AccuracyIssue) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  // Wrote out input data explicitly for easier reproduction.
  // std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape));)
  std::vector<float> input_data = {-10.000f, -9.355f, -8.710f, -8.065f, -7.419f, -6.774f, -6.129f, -5.484f, -4.839f,
                                   -4.194f, -3.548f, -2.903f, -2.258f, -1.613f, -0.968f, -0.323f, 0.323f, 0.968f,
                                   1.613f, 2.258f, 2.903f, 3.548f, 4.194f, 4.839f, 5.484f, 6.129f, 6.774f,
                                   7.419f, 8.065f, 8.710f, 9.355f, 10.000f};

  // std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(weight_shape));
  std::vector<float> weight_data = {-1.000f, -0.913f, -0.826f, -0.739f, -0.652f, -0.565f, -0.478f, -0.391f, -0.304f,
                                    -0.217f, -0.130f, -0.043f, 0.043f, 0.130f, 0.217f, 0.304f, 0.391f, 0.478f,
                                    0.565f, 0.652f, 0.739f, 0.826f, 0.913f, 1.000f};

  // std::vector<float> bias_data = GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape));
  std::vector<float> bias_data = {-1.000f, 0.000f, 1.000f};

  TestInputDef<float> input_def(input_shape, false, input_data);
  TestInputDef<float> weight_def(weight_shape, true, weight_data);
  TestInputDef<float> bias_def(bias_shape, true, bias_data);

  RunHTPConvOpPerChannelTest<uint8_t, Int4x2>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              21,     // opset
                                              QDQTolerance(0.005f));
}

// Test per-channel QDQ Conv is rejected with weight axis != 0
TEST_F(QnnHTPBackendTests, Conv_PerChannel_UnsupportedAxis) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 3, 3};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              2,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::None,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// Test per-channel QDQ Conv. in0: u8, in1 (weight): s8, in2 (bias): s32, out: u8
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::QNN_Conv3d_w_scale
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x1a preparation failed with err:-1
// QnnDsp <E> "Conv" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// onnxruntime::qnn::QnnModel::FinalizeGraphs] Failed to finalize QNN graph.
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, Conv3D_U8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,                   // weight quant axis
                                              {1, 1, 1},           // Strides
                                              {0, 0, 0, 0, 0, 0},  // Pads
                                              {1, 1, 1},           // Dilations
                                              1,                   // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,
                                              13);
}

// Test per-channel QDQ Conv that maps to QNN's DepthwiseConv2d (input_chans == output_chans == group).
// in0: u8, in1 (weight): s8, in2 (bias): s32, out: u8
TEST_F(QnnHTPBackendTests, ConvDepthwiseU8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};   // (N, C, H, W)
  std::vector<int64_t> weight_shape = {2, 1, 2, 2};  // (C, M/group, kH, kW)
  std::vector<int64_t> bias_shape = {2};             // (M)

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              2,             // group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// Conv3D per-channel
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::QNN_Conv3d_w_scale
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x1a preparation failed with err:-1
// QnnDsp <E> "Conv" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// onnxruntime::qnn::QnnModel::FinalizeGraphs] Failed to finalize QNN graph.
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, Conv3D_U8S8S32_PerChannel2) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {2, 1, 2, 2, 2};
  std::vector<int64_t> bias_shape = {2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,                   // weight quant axis
                                              {1, 1, 1},           // Strides
                                              {0, 0, 0, 0, 0, 0},  // Pads
                                              {1, 1, 1},           // Dilations
                                              2,                   // group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,
                                              13);
}

// Test per-channel QDQ ConvTranspose. in0: u8, in1 (weight): s8, in2 (bias): s32, out: u8
TEST_F(QnnHTPBackendTests, ConvTransposeU8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("ConvTranspose",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              1,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// Test per-channel QDQ ConvTranspose is unsupported with weight axis != 1.
TEST_F(QnnHTPBackendTests, ConvTranspose_PerChannel_UnsupportedAxis) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 3, 3};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("ConvTranspose",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              2,             // weight quant axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              1,             // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::None,
                                              false,  // use_qdq_contrib_ops
                                              13);    // opset
}

// ConvTranspose3D per-channel
// Disable it for 2.21 since it failed, re-enabled it for 2.22
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, ConvTranspose3D_U8S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("ConvTranspose",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              1,                   // weight quant axis
                                              {1, 1, 1},           // Strides
                                              {0, 0, 0, 0, 0, 0},  // Pads
                                              {1, 1, 1},           // Dilations
                                              1,                   // default group
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All,
                                              false,
                                              13);
}

#ifndef __linux__
// Test per-channel QDQ Conv. in0: u16, in1 (weight): s8, in2 (bias): s32, out: u16
TEST_F(QnnHTPBackendTests, ConvU16S16S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, int16_t>("Conv",
                                                input_def,
                                                weight_def,
                                                bias_def,
                                                0,             // weight quant axis
                                                {1, 1},        // Strides
                                                {0, 0, 0, 0},  // Pads
                                                {1, 1},        // Dilations
                                                1,             // default group
                                                "NOTSET",
                                                ExpectedEPNodeAssignment::All,
                                                true,  // use_qdq_contrib_ops
                                                13);   // opset
}
#endif

// Test per-channel QDQ Conv. in0: u16, in1 (weight): s8, in2 (bias): s32, out: u16
TEST_F(QnnHTPBackendTests, ConvU16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,  // use_qdq_contrib_ops
                                               13);   // opset
}

// Helper: Builds QDQ Conv model with fused activation (Relu or Clip) between Conv and output Q.
// Pattern: DQ(input) + DQ(weight) + float_bias -> Conv -> Relu/Clip -> Q(forced encoding) -> DQ
// Uses a forced output encoding with zp > 0 (encoding allows negatives) to trigger the
// non-fusion path. Without the fix, HTP won't clamp and the test fails.
template <typename ActivationQType, typename WeightQType>
static GetTestQDQModelFn<ActivationQType> BuildQDQConvWithFusedActivationTestCase(
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    const std::string& activation_type,
    float forced_output_scale,
    ActivationQType forced_output_zp,
    float clip_min = 0.0f,
    float clip_max = 6.0f) {
  return [input_def, weights_def, bias_def, activation_type, forced_output_scale, forced_output_zp,
          clip_min, clip_max](
             ModelTestBuilder& builder,
             std::vector<QuantParams<ActivationQType>>& output_qparams) {
    (void)output_qparams;

    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<ActivationQType> input_qparams = GetTestInputQuantParams<ActivationQType>(input_def);
    std::string input_dq = AddQDQNodePair<ActivationQType>(builder, "input_qdq", "input",
                                                           input_qparams.scale, input_qparams.zero_point, true);

    std::vector<float> weight_scales;
    std::vector<WeightQType> weight_zps;
    GetTestInputQuantParamsPerChannel<WeightQType>(weights_def, weight_scales, weight_zps, 0, true);
    std::vector<WeightQType> quantized_weights(SizeOfShape(weights_def.GetShape()));
    QuantizeValues<float, WeightQType>(weights_def.GetRawData(), quantized_weights,
                                       weights_def.GetShape(), weight_scales, weight_zps, 0);
    builder.MakeInitializer<WeightQType>("weights_quant", weights_def.GetShape(), quantized_weights);
    std::vector<ONNX_NAMESPACE::AttributeProto> w_dq_attrs;
    w_dq_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));
    builder.AddDequantizeLinearNode("WeightDQ", "weights_quant", weight_scales, weight_zps,
                                    "weights_dq", w_dq_attrs, true);

    builder.MakeInitializer<float>("bias", bias_def.GetShape(), bias_def.GetRawData());

    builder.AddNode("Conv", "Conv", {input_dq, "weights_dq", "bias"}, {"conv_out"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("kernel_shape", {1, 1}),
                     builder.MakeIntsAttribute("strides", {1, 1}),
                     builder.MakeIntsAttribute("pads", {0, 0, 0, 0})});

    if (activation_type == "Relu") {
      builder.AddNode("Relu", "Relu", {"conv_out"}, {"act_out"});
    } else {
      builder.MakeScalarInitializer<float>("clip_min", clip_min);
      builder.MakeScalarInitializer<float>("clip_max", clip_max);
      builder.AddNode("Clip", "Clip", {"conv_out", "clip_min", "clip_max"}, {"act_out"});
    }

    // Use FORCED output encoding (not auto-calibrated) to guarantee encoding_min < activation_min
    AddQDQNodePairWithOutputAsGraphOutput<ActivationQType>(builder, "output_qdq", "act_out",
                                                           forced_output_scale, forced_output_zp, true);
  };
}

// Test 1: Conv+Relu where output encoding min < 0 (Relu's activation_min).
TEST_F(QnnHTPBackendTests, ConvReluFusion_EncodingMinBelowZero) {
  std::vector<int64_t> input_shape = {1, 128, 4, 4};
  std::vector<int64_t> weight_shape = {64, 128, 1, 1};
  std::vector<int64_t> bias_shape = {64};

  TestInputDef<float> input_def(input_shape, false, GetFloatDataInRange(0.0f, 40.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true, GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true, GetFloatDataInRange(-3.0f, 10.0f, SizeOfShape(bias_shape)));

  auto build_f32_model = [input_def, weight_def, bias_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    MakeTestInput<float>(builder, "weights", weight_def);
    MakeTestInput<float>(builder, "bias", bias_def);
    builder.AddNode("Conv", "Conv", {"input", "weights", "bias"}, {"conv_out"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("kernel_shape", {1, 1}),
                     builder.MakeIntsAttribute("strides", {1, 1}),
                     builder.MakeIntsAttribute("pads", {0, 0, 0, 0})});
    builder.AddNode("Relu", "Relu", {"conv_out"}, {"relu_out"});
    builder.MakeOutput("relu_out");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Force output encoding: scale=0.000196, zp=14601 → encoding_min = 0.000196*(0-14601) = -2.86 < 0
  GetTestQDQModelFn<uint16_t> qdq_fn = BuildQDQConvWithFusedActivationTestCase<uint16_t, int8_t>(
      input_def, weight_def, bias_def, "Relu",
      0.000196270834f, static_cast<uint16_t>(14601));
  TestQDQModelAccuracy(build_f32_model, qdq_fn, provider_options, 21,
                       ExpectedEPNodeAssignment::All, QDQTolerance(0.02f));
}

// Test 2: Conv+Clip where output encoding min < clip_min.
TEST_F(QnnHTPBackendTests, ConvClipFusion_EncodingMinBelowClipMin) {
  std::vector<int64_t> input_shape = {1, 64, 4, 4};
  std::vector<int64_t> weight_shape = {32, 64, 1, 1};
  std::vector<int64_t> bias_shape = {32};

  TestInputDef<float> input_def(input_shape, false, GetFloatDataInRange(-5.0f, 40.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true, GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true, GetFloatDataInRange(-5.0f, 5.0f, SizeOfShape(bias_shape)));

  auto build_f32_model = [input_def, weight_def, bias_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    MakeTestInput<float>(builder, "weights", weight_def);
    MakeTestInput<float>(builder, "bias", bias_def);
    builder.AddNode("Conv", "Conv", {"input", "weights", "bias"}, {"conv_out"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("kernel_shape", {1, 1}),
                     builder.MakeIntsAttribute("strides", {1, 1}),
                     builder.MakeIntsAttribute("pads", {0, 0, 0, 0})});
    builder.MakeScalarInitializer<float>("clip_min_val", -1.0f);
    builder.MakeScalarInitializer<float>("clip_max_val", 6.0f);
    builder.AddNode("Clip", "Clip", {"conv_out", "clip_min_val", "clip_max_val"}, {"clip_out"});
    builder.MakeOutput("clip_out");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Force output encoding: scale=0.0003, zp=20000 → encoding_min = 0.0003*(0-20000) = -6.0 < clip_min(-1)
  GetTestQDQModelFn<uint16_t> qdq_fn = BuildQDQConvWithFusedActivationTestCase<uint16_t, int8_t>(
      input_def, weight_def, bias_def, "Clip",
      0.0003f, static_cast<uint16_t>(20000),
      -1.0f, 6.0f);
  TestQDQModelAccuracy(build_f32_model, qdq_fn, provider_options, 21,
                       ExpectedEPNodeAssignment::All, QDQTolerance(0.02f));
}

// Test 3: Conv+Clip where output encoding max > clip_max.
TEST_F(QnnHTPBackendTests, ConvClipFusion_EncodingMaxAboveClipMax) {
  std::vector<int64_t> input_shape = {1, 64, 4, 4};
  std::vector<int64_t> weight_shape = {32, 64, 1, 1};
  std::vector<int64_t> bias_shape = {32};

  TestInputDef<float> input_def(input_shape, false, GetFloatDataInRange(0.0f, 50.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true, GetFloatDataInRange(-0.5f, 0.5f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true, GetFloatDataInRange(0.0f, 8.0f, SizeOfShape(bias_shape)));

  auto build_f32_model = [input_def, weight_def, bias_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    MakeTestInput<float>(builder, "weights", weight_def);
    MakeTestInput<float>(builder, "bias", bias_def);
    builder.AddNode("Conv", "Conv", {"input", "weights", "bias"}, {"conv_out"}, kOnnxDomain,
                    {builder.MakeIntsAttribute("kernel_shape", {1, 1}),
                     builder.MakeIntsAttribute("strides", {1, 1}),
                     builder.MakeIntsAttribute("pads", {0, 0, 0, 0})});
    builder.MakeScalarInitializer<float>("clip_min_val", 0.0f);
    builder.MakeScalarInitializer<float>("clip_max_val", 6.0f);
    builder.AddNode("Clip", "Clip", {"conv_out", "clip_min_val", "clip_max_val"}, {"clip_out"});
    builder.MakeOutput("clip_out");
  };

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Force output encoding: scale=0.0002, zp=0 → encoding_max = 0.0002*65535 = 13.1 > clip_max(6)
  GetTestQDQModelFn<uint16_t> qdq_fn = BuildQDQConvWithFusedActivationTestCase<uint16_t, int8_t>(
      input_def, weight_def, bias_def, "Clip",
      0.0002f, static_cast<uint16_t>(0),
      0.0f, 6.0f);
  TestQDQModelAccuracy(build_f32_model, qdq_fn, provider_options, 21,
                       ExpectedEPNodeAssignment::All, QDQTolerance(0.02f));
}

// Conv3D per-channel (known issue)
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::QNN_Conv3d_w_scale
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x1a preparation failed with err:-1
// QnnDsp <E> "Conv" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// onnxruntime::qnn::QnnModel::FinalizeGraphs] Failed to finalize QNN graph.
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, Conv3D_U16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {3, 2, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,                   // weight quant axis
                                               {1, 1, 1},           // Strides
                                               {0, 0, 0, 0, 0, 0},  // Pads
                                               {1, 1, 1},           // Dilations
                                               1,                   // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,
                                               13);
}

// Test per-channel QDQ ConvTranspose. in0: u16, in1 (weight): s8, in2 (bias): s32, out: u16
TEST_F(QnnHTPBackendTests, ConvTransposeU16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("ConvTranspose",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               1,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               1,             // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,  // use_qdq_contrib_ops
                                               13);   // opset
}

// Disable it for 2.21, re-enable it for 2.22
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, ConvTranspose3D_U16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {2, 3, 2, 2, 2};
  std::vector<int64_t> bias_shape = {3};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("ConvTranspose",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               1,                   // weight quant axis
                                               {1, 1, 1},           // Strides
                                               {0, 0, 0, 0, 0, 0},  // Pads
                                               {1, 1, 1},           // Dilations
                                               1,                   // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,
                                               13);
}

// Test per-channel QDQ Conv that maps to QNN's DepthwiseConv2d (input_chans == output_chans == group).
// in0: u16, in1 (weight): s8, in2 (bias): s32, out: u16
TEST_F(QnnHTPBackendTests, ConvDepthwiseU16S8S32_PerChannel) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4};   // (N, C, H, W)
  std::vector<int64_t> weight_shape = {2, 1, 2, 2};  // (C, M/group, kH, kW)
  std::vector<int64_t> bias_shape = {2};             // (M)

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,             // weight quant axis
                                               {1, 1},        // Strides
                                               {0, 0, 0, 0},  // Pads
                                               {1, 1},        // Dilations
                                               2,             // group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,  // use_qdq_contrib_ops
                                               13);   // opset
}

// Test per-channel QDQ Conv3D
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:203:ERROR:could not create op: q::QNN_Conv3d_w_scale
// \QNN\HTP\HTP\src\hexagon\prepare\graph_prepare.cc:1187:ERROR:Op 0x1a preparation failed with err:-1
// QnnDsp <E> "Conv" generated: could not create op
// QnnDsp <E> RouterWindows graph prepare failed 12
// QnnDsp <E> Failed to finalize graph (id: 1) with err 1002
// QnnDsp <V> Wake up free backend 1 thread(s)
// QnnDsp <I> QnnGraph_finalize done. status 0x3ea
// onnxruntime::qnn::QnnModel::FinalizeGraphs] Failed to finalize QNN graph.
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, Conv3D_U16S8S32_PerChannel2) {
  std::vector<int64_t> input_shape = {1, 2, 4, 4, 4};
  std::vector<int64_t> weight_shape = {2, 1, 2, 2, 2};
  std::vector<int64_t> bias_shape = {2};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint16_t, int8_t>("Conv",
                                               input_def,
                                               weight_def,
                                               bias_def,
                                               0,                   // weight quant axis
                                               {1, 1, 1},           // Strides
                                               {0, 0, 0, 0, 0, 0},  // Pads
                                               {1, 1, 1},           // Dilations
                                               2,                   // default group
                                               "NOTSET",
                                               ExpectedEPNodeAssignment::All,
                                               true,
                                               13);
}

// Tests 16-bit QDQ Conv with dynamic weights and bias (uses QNN's Conv2d)
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.0040235077030956745, zero_point=0.
// Expected val: 87.354057312011719
// QNN QDQ val: 0 (err 87.354057312011719)
// CPU QDQ val: 87.3583984375 (err 0.00434112548828125)
TEST_F(QnnHTPBackendTests, DISABLED_ConvU16S16S32_DynamicBias) {
  TestInputDef<float> input_def({1, 2, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 50));
  TestInputDef<float> weight_def({1, 2, 3, 3}, false, GetFloatDataInRange(-1.0f, 5.0f, 18));
  RunHTPConvOpTest<uint16_t, int16_t>("Conv",
                                      input_def,                                   // Input
                                      weight_def.OverrideValueRange(-5.0f, 5.0f),  // Weights (symmetric quant range)
                                      TestInputDef<float>({1}, false, {2.0f}),     // Bias
                                      {1, 1},                                      // Strides
                                      {0, 0, 0, 0},                                // Pads
                                      {1, 1},                                      // Dilations
                                      1,                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit QDQ Conv with dynamic weights and bias (uses QNN's DepthwiseConv2d)
// TODO(adrianlizarraga): FAIL: Failed to finalize QNN graph. Error code 1002
TEST_F(QnnHTPBackendTests, DISABLED_DepthwiseConvU16S16S32_DynamicBias) {
  TestInputDef<float> input_def({1, 1, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 25));
  TestInputDef<float> weight_def({1, 1, 3, 3}, false, GetFloatDataInRange(-1.0f, 5.0f, 9));
  RunHTPConvOpTest<uint16_t, int16_t>("Conv",
                                      input_def,                                   // Input
                                      weight_def.OverrideValueRange(-5.0f, 5.0f),  // Weights (symmetric quant range)
                                      TestInputDef<float>({1}, false, {2.0f}),     // Bias
                                      {1, 1},                                      // Strides
                                      {0, 0, 0, 0},                                // Pads
                                      {1, 1},                                      // Dilations
                                      1,                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit QDQ Conv with dynamic weights and no bias.
// TODO: Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.0039929896593093872, zero_point=0.
// Expected val: 85.354057312011719
// QNN QDQ val: 0 (err 85.354057312011719)
// CPU QDQ val: 85.358139038085938 (err 0.00408172607421875)
TEST_F(QnnHTPBackendTests, DISABLED_ConvU16S16S32_NoBias) {
  TestInputDef<float> input_def({1, 2, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 50));
  TestInputDef<float> weight_def({1, 2, 3, 3}, false, GetFloatDataInRange(-1.0f, 5.0f, 18));
  RunHTPConvOpTest<uint16_t, int16_t>("Conv",
                                      input_def,                                   // Input
                                      weight_def.OverrideValueRange(-5.0f, 5.0f),  // Weights (symmetric quant range)
                                      TestInputDef<float>(),                       // Bias
                                      {1, 1},                                      // Strides
                                      {0, 0, 0, 0},                                // Pads
                                      {1, 1},                                      // Dilations
                                      1,                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit QDQ Conv with dynamic weights and no bias (uses QNN's DepthWiseConv2d)
// TODO(adrianlizarraga): FAIL: Failed to finalize QNN graph. Error code 1002
TEST_F(QnnHTPBackendTests, DISABLED_DepthwiseConvU16S16S32_NoBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 25);
  std::vector<float> weight_data = GetFloatDataInRange(-10.0f, 10.0f, 9);
  RunHTPConvOpTest<uint16_t, int16_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5}, false, input_data),   // Input
                                      TestInputDef<float>({1, 1, 3, 3}, false, weight_data),  // Weights
                                      TestInputDef<float>(),                                  // Bias
                                      {1, 1},                                                 // Strides
                                      {0, 0, 0, 0},                                           // Pads
                                      {1, 1},                                                 // Dilations
                                      1,                                                      // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true);  // Use com.microsoft QDQ ops for 16-bit
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with static bias.
// Uses QNN's DepthwiseConv2d operator.
TEST_F(QnnHTPBackendTests, DepthwiseConvU16U8S32_StaticBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 25);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 9);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 1, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>({1}, true, {2.0f}),                // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 125);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 27);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5, 5}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 1, 3, 3, 3}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>({1}, true, {2.0f}),                      // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with static bias.
TEST_F(QnnHTPBackendTests, ConvU16U8S32_StaticBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 50);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 18);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 2, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>({1}, true, {2.0f}),                // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 150);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 36);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 2, 3, 3, 2}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>({1}, true, {2.0f}),                      // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with dynamic bias.
// Uses QNN's DepthwiseConv2d operator.
TEST_F(QnnHTPBackendTests, DepthwiseConvU16U8S32_DynamicBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 25);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 9);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 1, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>({1}, false, {2.0f}),               // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 75);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 27);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 1, 3, 3, 3}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>({1}, false, {2.0f}),                     // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with dynamic bias.
TEST_F(QnnHTPBackendTests, ConvU16U8S32_DynamicBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 50);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 18);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 2, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>({1}, false, {2.0f}),               // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 150);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 36);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 2, 3, 3, 2}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>({1}, false, {2.0f}),                     // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with no bias
TEST_F(QnnHTPBackendTests, ConvU16U8S32_NoBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 50);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 18);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 2, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>(),                                 // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 150);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 36);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 2, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 2, 3, 3, 2}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>(),                                       // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Tests 16-bit activations, 8-bit static weights QDQ Conv with no bias
// Uses QNN's DepthwiseConv2d operator.
TEST_F(QnnHTPBackendTests, DepthwiseConvU16U8S32_NoBias) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 25);
  std::vector<float> weight_data = GetFloatDataInRange(-1.0f, 5.0f, 9);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5}, false, input_data),  // Input
                                      TestInputDef<float>({1, 1, 3, 3}, true, weight_data),  // Weights
                                      TestInputDef<float>(),                                 // Bias
                                      {1, 1},                                                // Strides
                                      {0, 0, 0, 0},                                          // Pads
                                      {1, 1},                                                // Dilations
                                      1,                                                     // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);

  std::vector<float> input_data_3d = GetFloatDataInRange(-10.0f, 10.0f, 75);
  std::vector<float> weight_data_3d = GetFloatDataInRange(-1.0f, 5.0f, 18);
  RunHTPConvOpTest<uint16_t, uint8_t>("Conv",
                                      TestInputDef<float>({1, 1, 5, 5, 3}, false, input_data_3d),  // Input
                                      TestInputDef<float>({1, 1, 3, 3, 2}, true, weight_data_3d),  // Weights
                                      TestInputDef<float>(),                                       // Bias
                                      {1, 1, 1},                                                   // Strides
                                      {0, 0, 0, 0, 0, 0},                                          // Pads
                                      {1, 1, 1},                                                   // Dilations
                                      1,                                                           // default group
                                      "NOTSET",
                                      ExpectedEPNodeAssignment::All,
                                      true,  // Use com.microsoft QDQ ops for 16-bit
                                      13);
}

// Test that dynamic weights with default bias works for Conv. This was previously not working
// on older versions of QNN sdk.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_DynamicWeight_NoBias) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 3, 32, 32}, false, -10.0f, 10.0f),  // Input
                                     TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),    // Weights
                                     TestInputDef<float>(),                                      // Bias
                                     {1, 1},                                                     // Strides
                                     {0, 0, 0, 0},                                               // Pads
                                     {1, 1},                                                     // Dilations
                                     1,                                                          // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 3, 32, 32, 32}, false, -10.0f, 10.0f),  // Input
                                     TestInputDef<float>({1, 3, 4, 4, 4}, false, -10.0f, 10.0f),     // Weights
                                     TestInputDef<float>(),                                          // Bias
                                     {1, 1, 1},                                                      // Strides
                                     {0, 0, 0, 0, 0, 0},                                             // Pads
                                     {1, 1, 1},                                                      // Dilations
                                     1,                                                              // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Test that dynamic weights with default bias works for ConvTranspose. This was previously not working
// on older versions of QNN sdk.
TEST_F(QnnHTPBackendTests, ConvTransposeU8U8S32_DynamicWeight_NoBias) {
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 3, 32, 32}, false, -10.0f, 10.0f),  // Input
                                     TestInputDef<float>({3, 1, 4, 4}, false, -10.0f, 10.0f),    // Weights
                                     TestInputDef<float>(),                                      // Bias
                                     {1, 1},                                                     // Strides
                                     {0, 0, 0, 0},                                               // Pads
                                     {1, 1},                                                     // Dilations
                                     1,                                                          // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// QNN op validation crash. Run correctly if by pass the QNN op validation
// Exception from backendValidateOpConfig:
// Exception thrown at 0x00007FFF9E0128B0 (QnnHtpPrepare.dll) in onnxruntime_test_all.exe:
// 0xC0000005: Access violation reading location 0x7079745F656C706D.
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, ConvTranspose3D_U8U8S32_DynamicWeight_NoBias) {
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 3, 32, 32, 32}, false, -10.0f, 10.0f),  // Input
                                     TestInputDef<float>({3, 1, 4, 4, 4}, false, -10.0f, 10.0f),     // Weights
                                     TestInputDef<float>(),                                          // Bias
                                     {1, 1, 1},                                                      // Strides
                                     {0, 0, 0, 0, 0, 0},                                             // Pads
                                     {1, 1, 1},                                                      // Dilations
                                     1,                                                              // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
// Tests bias as an initializer.
TEST_F(QnnHTPBackendTests, ConvU8U8S32_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                                     TestInputDef<float>({1, 1, 3, 3}, true, -10.0f, 10.0f),  // Random static weight
                                     TestInputDef<float>({1}, true, {2.0f}),                  // Initializer bias
                                     {1, 1},                                                  // Strides
                                     {0, 0, 0, 0},                                            // Pads
                                     {1, 1},                                                  // Dilations
                                     1,                                                       // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13);    // opset

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.0f, 10.0f),   // Random dynamic input
                                     TestInputDef<float>({1, 1, 3, 3, 3}, true, -10.0f, 10.0f),  // Random static weight
                                     TestInputDef<float>({1}, true, {2.0f}),                     // Initializer bias
                                     {1, 1, 1},                                                  // Strides
                                     {0, 0, 0, 0, 0, 0},                                         // Pads
                                     {1, 1, 1},                                                  // Dilations
                                     1,                                                          // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13);    // opset
}

// Tests 1D Conv with bias as an initializer.
TEST_F(QnnHTPBackendTests, Conv1DU8U8S32_bias_initializer) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0, 0},                                                      // pads
                                     {1},                                                         // dilations
                                     1,                                                           // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Tests 1D ConvTranspose with bias as an initializer.
TEST_F(QnnHTPBackendTests, ConvTranspose1DU8U8S32_bias_initializer) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0, 0},                                                      // pads
                                     {1},                                                         // dilations
                                     1,                                                           // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Tests auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvU8U8S32_AutoPadUpper) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {1, 1},                                               // strides
                                     {},                                                   // pads
                                     {1, 1},                                               // dilations
                                     1,                                                    // default group
                                     "SAME_UPPER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {1, 1, 1},                                               // strides
                                     {},                                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // default group
                                     "SAME_UPPER",                                            // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests Conv1d auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, Conv1DU8U8S32_AutoPadUpper) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     1,                                                           // default group
                                     "SAME_UPPER",                                                // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests TransposeConv1d auto_pad value "SAME_UPPER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvTranspose1DU8U8S32_AutoPadUpper) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     1,                                                           // default group
                                     "SAME_UPPER",                                                // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests Conv's auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {1, 1},                                               // strides
                                     {},                                                   // pads
                                     {1, 1},                                               // dilations
                                     1,                                                    // default group
                                     "SAME_LOWER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {1, 1, 1},                                               // strides
                                     {},                                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // default group
                                     "SAME_LOWER",                                            // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests ConvTranspose's auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvTransposeU8U8S32_AutoPadLower) {
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {1, 1},                                               // strides
                                     {},                                                   // pads
                                     {1, 1},                                               // dilations
                                     1,                                                    // default group
                                     "SAME_LOWER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);

  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {1, 1, 1},                                               // strides
                                     {},                                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // default group
                                     "SAME_LOWER",                                            // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests Conv1d auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, Conv1DU8U8S32_AutoPadLower) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     1,                                                           // default group
                                     "SAME_LOWER",                                                // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests ConvTranspose 1d auto_pad value "SAME_LOWER" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvTranspose1DU8U8S32_AutoPadLower) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     1,                                                           // default group
                                     "SAME_LOWER",                                                // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests Conv's auto_pad value "VALID" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvU8U8S32_AutoPadValid) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {1, 1},                                               // strides
                                     {},                                                   // pads
                                     {1, 1},                                               // dilations
                                     1,                                                    // default group
                                     "VALID",                                              // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);

  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {1, 1, 1},                                               // strides
                                     {},                                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // default group
                                     "VALID",                                                 // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests ConvTranspose's auto_pad value "VALID" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvTransposeU8U8S32_AutoPadValid) {
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {1, 1},                                               // strides
                                     {},                                                   // pads
                                     {1, 1},                                               // dilations
                                     1,                                                    // default group
                                     "VALID",                                              // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);

  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 5, 5, 5}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 4, 4, 4}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {1, 1, 1},                                               // strides
                                     {},                                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // default group
                                     "VALID",                                                 // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Test ConvTranspose with output_shape attribute
// This test verifies that when 'output_shape' is provided, the QNN EP correctly
// calculates and applies padding for ConvTranspose, overriding any 'pads' attribute,
// and correctly distributes the padding according to 'auto_pad' rules.
TEST_F(QnnHTPBackendTests, ConvTransposeU8U8S32_OutputShape) {
  std::vector<int64_t> output_shape = {6, 6};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 4, 4}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 2, 2}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),               // Initializer bias
                                     {2, 2},                                               // strides
                                     {0, 0, 0, 0},                                         // pads
                                     {1, 1},                                               // dilations
                                     1,                                                    // group
                                     "SAME_UPPER",                                         // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13,     // opset
                                     QDQTolerance(),
                                     std::nullopt,   // No output activation
                                     output_shape);  // Pass the output_shape attribute

  std::vector<int64_t> output_shape_3d = {6, 6, 6};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 4, 4, 4}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 2, 2, 2}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),                  // Initializer bias
                                     {2, 2, 2},                                               // strides
                                     {0, 0, 0, 0, 0, 0},                                      // pads
                                     {1, 1, 1},                                               // dilations
                                     1,                                                       // group
                                     "SAME_UPPER",                                            // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13,     // opset
                                     QDQTolerance(),
                                     std::nullopt,      // No output activation
                                     output_shape_3d);  // Pass the output_shape attribute
}

TEST_F(QnnHTPBackendTests, ConvTranspose1DU8U8S32_OutputShape) {
  std::vector<int64_t> output_shape = {6};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 1, 4}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({1, 1, 2}, true, -1.f, 1.f),   // Static weights
                                     TestInputDef<float>({1}, true, {1.0f}),            // Initializer bias
                                     {2},                                               // strides
                                     {0, 0},                                            // pads
                                     {1},                                               // dilations
                                     1,                                                 // group
                                     "SAME_UPPER",                                      // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13,     // opset
                                     QDQTolerance(),
                                     std::nullopt,   // No output activation
                                     output_shape);  // Pass the output_shape attribute
}

// Tests Conv1d auto_pad value "VALID" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, Conv1DU8U8S32_AutoPadValid) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({1, 2, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     1,                                                           // default group
                                     "VALID",                                                     // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Tests ConvTranspose 1d auto_pad value "VALID" on HTP backend (compares to CPU EP).
TEST_F(QnnHTPBackendTests, ConvTranspose1DU8U8S32_AutoPadValid) {
  std::vector<float> input_data = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 2, 4}, false, input_data),           // Dynamic input
                                     TestInputDef<float>({2, 1, 2}, true, {1.f, 2.f, 3.f, 4.f}),  // Static weight
                                     TestInputDef<float>({1}, true, {1.0f}),                      // Initializer bias
                                     {1},                                                         // strides
                                     {0},                                                         // pads
                                     {1},                                                         // dilations
                                     1,                                                           // default group
                                     "VALID",                                                     // auto_pad
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_contrib_qdq
                                     13);
}

// Fails with QNN SDK 2.35.0:
// value pair (-4.54545403, -4.54687548) at index #3 don't match, which is -0.00142145 from -4.54545
TEST_F(QnnHTPBackendTests, DISABLED_ConvU8U8S32_large_input1_padding_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 3, 60, 452}, false, 0.f, 10.f),        // Dynamic input
                                     TestInputDef<float>({16, 3, 3, 3}, true, -1.f, 1.f),           // Static weights
                                     TestInputDef<float>({16}, true, std::vector<float>(16, 1.f)),  // Initializer bias
                                     {1, 1},
                                     {1, 1, 1, 1},
                                     {1, 1},
                                     1,  // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,  // use_qdq_contrib_ops
                                     13,     // opset
                                     // Need tolerance of 0.76% of output range after QNN SDK 2.19.2
                                     QDQTolerance(0.0076f));
}

TEST_F(QnnHTPBackendTests, ConvU8U8S32_large_input2_bias_initializer) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 128, 8, 56}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({32, 128, 1, 1}, true, -1.f, 1.f),   // Random static weights
                                     TestInputDef<float>({32}, true, -1.f, 1.f),              // Random initializer bias
                                     {1, 1},
                                     {0, 0, 0, 0},
                                     {1, 1},
                                     1,  // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All,
                                     false,
                                     13);
}

TEST_F(QnnHTPBackendTests, ConvU8U8S32_LargeInput_Dilations_Pads) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 3, 768, 1152}, false, 0.f, 10.f),  // Dynamic input
                                     TestInputDef<float>({64, 3, 7, 7}, true, -1.f, 1.f),       // Static weights
                                     TestInputDef<float>({64}, true, -1.f, 1.f),                // Initializer bias
                                     {2, 2},                                                    // strides
                                     {3, 3, 3, 3},                                              // pads
                                     {1, 1},                                                    // dilations
                                     1,                                                         // default group
                                     "NOTSET",                                                  // auto_pad
                                     ExpectedEPNodeAssignment::All);
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-Quantized Conv (BQ Phase 1)
//
// ONNX graph pattern:
//   input → Q(u16) → DQ → Conv ← DQ(Int4, block_size, axis=0) → Q(u16) → DQ
//
// The weight DQ node uses a rank-2 float scale tensor [OC, num_blocks] where
// num_blocks = IC / block_size. QNN EP maps this to the BW_FLOAT_BLOCK kernel
// (FP16 activation) with an INT16→FP16 Convert before Conv and FP16→INT16
// Convert after Conv.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Builds the ONNX QDQ graph for a BQ (block-quantized weight) 2D Conv.
//   - activation: uint16 per-tensor symmetric Q/DQ
//   - weight: INT4 or INT8 initializer + DQ with block_size attribute and rank-4
//             float scale [OC, IC/block_size, 1, 1] (axis=1, IC is the blocked dimension)
//   - output: uint16 per-tensor symmetric Q/DQ
//
// weight_bits: 4 for INT4/UINT4 (default), 8 for INT8/UINT8, 2 for INT2.
// block_size constraints: must be a multiple of 8 (4-bit), 4 (8-bit), or 16 (2-bit) per HTP.
// weight_is_unsigned: true → use UINT weight type (UINT4 or UINT8); tests the
//   unsigned→signed conversion path in conv_op_builder.cc (TransformUnsignedToSignedFixedPoint).
// bias_per_channel: when include_bias is true, true → per-channel bias scale ([OC] values,
//   axis=0); false → per-tensor bias scale (scalar). Exercises both bias dequant paths.
GetQDQTestCaseFn BuildBQConvTestCase(const std::vector<int64_t>& input_shape,
                                     const std::vector<int64_t>& weight_shape,
                                     int64_t block_size,
                                     bool include_bias = false,
                                     int weight_bits = 4,
                                     bool weight_is_unsigned = false,
                                     bool bias_per_channel = false,
                                     bool is_bias_quantized = true) {
  return [input_shape, weight_shape, block_size, include_bias, weight_bits,
          weight_is_unsigned, bias_per_channel, is_bias_quantized](ModelTestBuilder& builder) -> void {
    const int64_t OC = weight_shape[0];
    const int64_t IC = weight_shape[1];
    const int64_t kH = weight_shape.size() >= 4 ? weight_shape[2] : 1;
    const int64_t kW = weight_shape.size() >= 4 ? weight_shape[3] : 1;
    const int64_t num_blocks = IC / block_size;  // caller ensures IC % block_size == 0

    // ── Activation: float → Q(uint16) → DQ ──────────────────────────────────
    auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "input", input_def);

    // uint16 symmetric per-tensor: scale = 2/65534, zp = 32767 (~[-1, 1])
    const float act_scale = 2.0f / 65534.0f;
    const uint16_t act_zp = 32767;
    const std::string act_dql_out = AddQDQNodePair<uint16_t>(builder, "act", "input", act_scale, act_zp);

    // ── Weight initializer + DQ(block_size, axis=1) ──────────────────────────
    // Scale rank == weight rank per ONNX opset 21: [OC, IC/block_size, 1, 1].
    const std::vector<int64_t> scale_shape{OC, num_blocks, 1, 1};
    builder.MakeInitializer<float>("weight_scale", scale_shape, 0.01f, 0.05f);

    const size_t num_elems = static_cast<size_t>(OC * IC * kH * kW);
    if (weight_bits == 4 && !weight_is_unsigned) {
      // INT4 weight in range [-3, 3] (symmetric).
      std::vector<Int4x2> weight_data(Int4x2::CalcNumInt4Pairs(num_elems));
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i >> 1].SetElem(i & 1, static_cast<int8_t>((i % 7) - 3));
      }
      builder.MakeInitializer<Int4x2>("weight_quant", weight_shape, weight_data);
    } else if (weight_bits == 4 && weight_is_unsigned) {
      // UINT4 weight in range [0, 14] (asymmetric-like; symmetric around 7 with zp omitted).
      std::vector<UInt4x2> weight_data(UInt4x2::CalcNumInt4Pairs(num_elems));
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i >> 1].SetElem(i & 1, static_cast<uint8_t>(i % 15));
      }
      builder.MakeInitializer<UInt4x2>("weight_quant", weight_shape, weight_data);
    } else if (weight_bits == 2 && !weight_is_unsigned) {
      // INT2 weight in range [-1, 1] (symmetric, 4 elements per byte).
      std::vector<Int2x4> weight_data(Int2x4::CalcNumInt2Quads(num_elems));
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i >> 2].SetElem(i & 3, static_cast<int8_t>((i % 3) - 1));
      }
      builder.MakeInitializer<Int2x4>("weight_quant", weight_shape, weight_data);
    } else if (weight_bits == 2 && weight_is_unsigned) {
      // UINT2 weight in range [0, 3] (full unsigned 2-bit range, 4 elements per byte).
      std::vector<UInt2x4> weight_data(UInt2x4::CalcNumInt2Quads(num_elems));
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i >> 2].SetElem(i & 3, static_cast<uint8_t>(i % 4));
      }
      builder.MakeInitializer<UInt2x4>("weight_quant", weight_shape, weight_data);
    } else if (weight_is_unsigned) {
      // UINT8 weight in range [0, 126] (symmetric around 63 with zp omitted).
      std::vector<uint8_t> weight_data(num_elems);
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i] = static_cast<uint8_t>(i % 127);
      }
      builder.MakeInitializer<uint8_t>("weight_quant", weight_shape, weight_data);
    } else {
      // INT8 weight in range [-63, 63] (symmetric).
      std::vector<int8_t> weight_data(num_elems);
      for (size_t i = 0; i < num_elems; ++i) {
        weight_data[i] = static_cast<int8_t>((i % 127) - 63);
      }
      builder.MakeInitializer<int8_t>("weight_quant", weight_shape, weight_data);
    }

    // DQ with block_size; omit zero_point (symmetric). axis=1: IC is the blocked dimension.
    builder.AddNode("weight_dql", "DequantizeLinear",
                    {"weight_quant", "weight_scale"},
                    {"weight_dql_out"}, "",
                    {builder.MakeScalarAttribute("axis", static_cast<int64_t>(1)),
                     builder.MakeScalarAttribute("block_size", block_size)});

    // ── Conv ─────────────────────────────────────────────────────────────────
    std::vector<std::string> conv_inputs{act_dql_out, "weight_dql_out"};
    if (include_bias) {
      if (!is_bias_quantized) {
        // Float bias: pass directly as a float initializer (no DQ node).
        std::vector<float> bias_data(static_cast<size_t>(OC), 0.01f);
        builder.Make1DInitializer<float>("bias", bias_data);
        conv_inputs.push_back("bias");
      } else if (bias_per_channel) {
        // Per-channel quantized bias: distinct quantized value and scale per output channel (DQ axis=0).
        // Use INT32-quantized bias directly (no QL node — avoids ORT QL opset validation for INT32).
        // OrtConvNodeGroupSelector requires bias DQL input type == INT32 (qnn_ep_utils.cc:741).
        // Non-zero quant values ensure the per-channel scale indexing is actually exercised.
        // Omit zero_point (symmetric): ORT per-axis DQ requires zp be null or 1D of size OC.
        std::vector<int32_t> bias_quant(static_cast<size_t>(OC));
        std::vector<float> bias_scales(static_cast<size_t>(OC));
        for (size_t i = 0; i < bias_quant.size(); ++i) {
          bias_quant[i] = static_cast<int32_t>(i) - static_cast<int32_t>(OC) / 2;
          bias_scales[i] = act_scale * (0.02f + 0.01f * static_cast<float>(i));
        }
        builder.Make1DInitializer<int32_t>("bias_quant", bias_quant);
        builder.Make1DInitializer<float>("bias_scale", bias_scales);
        builder.AddNode("bias_dql", "DequantizeLinear",
                        {"bias_quant", "bias_scale"}, {"bias_dql_out"}, "",
                        {builder.MakeScalarAttribute("axis", static_cast<int64_t>(0))});
        conv_inputs.push_back("bias_dql_out");
      } else {
        // Per-tensor quantized bias.
        const float bias_scale = act_scale * 0.03f;
        builder.MakeScalarInitializer<float>("bias_scale", bias_scale);
        builder.MakeScalarInitializer<int32_t>("bias_zp", 0);
        builder.Make1DInitializer<int32_t>("bias_quant", std::vector<int32_t>(static_cast<size_t>(OC), 0));
        builder.AddNode("bias_dql", "DequantizeLinear",
                        {"bias_quant", "bias_scale", "bias_zp"}, {"bias_dql_out"});
        conv_inputs.push_back("bias_dql_out");
      }
    }
    builder.AddNode("conv", "Conv",
                    conv_inputs, {"conv_out"}, kOnnxDomain,
                    {builder.MakeStringAttribute("auto_pad", "NOTSET"),
                     builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}),
                     builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0})});

    // ── Output: Conv → Q(uint16) → DQ → graph output ─────────────────────────
    const float out_scale = 2.0f / 65534.0f;
    const uint16_t out_zp = 32767;
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "out", "conv_out", out_scale, out_zp);
  };
}

ProviderOptions GetBQConvProviderOptions() {
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

// Provider options for the LPBQ (BLOCKWISE_EXPANSION) path.
// enable_block_quant_weight_optimization=1 triggers BQ -> LPBQ conversion.
ProviderOptions GetLPBQConvProviderOptions() {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
  opts["enable_block_quant_weight_optimization"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return opts;
}

// Runs a BQ Conv model on the QNN HTP backend and verifies which BQ encoding path QNN EP selected by inspecting the
// dumped QNN graph JSON.
//   - Native BQ  (QNN_QUANTIZATION_ENCODING_BLOCK): no INT16→FP16 activation Dequantize and no
//     FP16→INT16 output Quantize are inserted → Quantize=1, Dequantize=1.
//   - Fallback   (QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK): activation Dequantize and output
//     Quantize are inserted → Quantize=2, Dequantize=2.
void RunNativeBQConvTest(GetQDQTestCaseFn build_fn,
                         bool expect_native_bq,
                         float fp32_abs_err = 1e-2f) {
  ProviderOptions provider_options = GetBQConvProviderOptions();

  // Dump JSON graph for verifying native BQ working as expected.
  const std::filesystem::path json_qnn_graph_dir = "ConvNativeBQ";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();

  RunQnnModelTest(build_fn,
                  provider_options,
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(fp32_abs_err)});

#ifndef QNN_HTP_NATIVE_BQ_AVAILABLE
  // HTP native BQ support starts from QAIRT 2.51 (QNN API 2.40).
  // Disable here to avoid guarding everywhere.
  expect_native_bq = false;
#endif

  if (expect_native_bq) {
    AssertOpInQnnGraph(json_qnn_graph_dir, "Quantize", 1);
    AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", 1);
  } else {
    AssertOpInQnnGraph(json_qnn_graph_dir, "Quantize", 2);
    AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", 2);
  }
}

}  // namespace

// 1x1 Conv, INT4 weight, block_size=8, uint16 activation, no bias.
// in0: u16, weight: int4 (scale=[4,2,1,1], block_size=8), out: u16
// Checks: all nodes assigned to QNN EP; output matches CPU EP within 1e-2.
TEST_F(QnnHTPBackendTests, ConvBQ_U16Int4_1x1_NoBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 16, 4, 4},
                                      /*weight=*/{4, 16, 1, 1},
                                      /*block_size=*/8,
                                      /*bias=*/false),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// 1x1 Conv with bias. Exercises the INT32→FP16 bias dequantization path.
// in0: u16, weight: int4 (scale=[4,2,1,1], block_size=8), bias: int32 DQ, out: u16
// Checks: all nodes assigned to QNN EP; output matches CPU EP within 1e-2.
TEST_F(QnnHTPBackendTests, ConvBQ_U16Int4_1x1_WithBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 16, 4, 4},
                                      /*weight=*/{4, 16, 1, 1},
                                      /*block_size=*/8,
                                      /*bias=*/true),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// 1x1 Conv with per-channel quantized bias (DQ axis=0, [OC] scales).
// Exercises the per-channel branch of the INT32→FP16 bias dequantization.
TEST_F(QnnHTPBackendTests, ConvBQ_U16Int4_1x1_WithBiasPerChannel) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 16, 4, 4},
                                      /*weight=*/{4, 16, 1, 1},
                                      /*block_size=*/8,
                                      /*bias=*/true,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false,
                                      /*bias_per_channel=*/true),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// 1x1 Conv with larger IC and more blocks per channel.
// weight: int4 (IC=32, block_size=8, 4 blocks/OC), scale=[8,4,1,1]
// Checks: all nodes assigned to QNN EP; output matches CPU EP within 1e-2.
TEST_F(QnnHTPBackendTests, ConvBQ_U16Int4_1x1_MultiBlock) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                      /*weight=*/{8, 32, 1, 1},
                                      /*block_size=*/8),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// Regression: existing per-channel INT4 Conv (no block_size) continues to work.
// Reuses RunHTPConvOpPerChannelTest to confirm the BQ detection does not interfere.
TEST_F(QnnHTPBackendTests, ConvBQ_ExistingPerChannel_Unaffected) {
  // This duplicates ConvU16S4S32_PerChannel to act as a regression guard.
  RunHTPConvOpPerChannelTest<uint16_t, Int4x2>(
      "Conv",
      TestInputDef<float>({1, 2, 4, 4}, false,
                          GetFloatDataInRange(0.0f, 1.0f, SizeOfShape({1, 2, 4, 4}))),
      TestInputDef<float>({3, 2, 2, 2}, true,
                          GetFloatDataInRange(-1.0f, 5.0f, SizeOfShape({3, 2, 2, 2}))),
      TestInputDef<float>({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3)),
      0,             // weight quant axis
      {1, 1},        // strides
      {0, 0, 0, 0},  // pads
      {1, 1},        // dilations
      1,             // group
      "NOTSET",
      ExpectedEPNodeAssignment::All,
      false,  // use_qdq_contrib_ops
      21);    // opset
}

// ── BQ Conv bitwidth / block_size variants ───────────────────────────────────
// INT4, block_size=16: still a valid HTP multiple-of-8 block size.
TEST_F(QnnHTPBackendTests, ConvBQ_U16Int4_1x1_BlockSize16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                      /*weight=*/{4, 32, 1, 1},
                                      /*block_size=*/16,
                                      /*bias=*/false,
                                      /*weight_bits=*/4),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT8, block_size=4: minimum valid HTP multiple-of-4 block size for 8-bit.
TEST_F(QnnHTPBackendTests, ConvBQ_U16Int8_1x1_BlockSize4) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 16, 4, 4},
                                      /*weight=*/{4, 16, 1, 1},
                                      /*block_size=*/4,
                                      /*bias=*/false,
                                      /*weight_bits=*/8),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT8, block_size=8: larger block size, still a valid HTP multiple-of-4.
TEST_F(QnnHTPBackendTests, ConvBQ_U16Int8_1x1_BlockSize8) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                      /*weight=*/{4, 32, 1, 1},
                                      /*block_size=*/8,
                                      /*bias=*/false,
                                      /*weight_bits=*/8),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// INT2, block_size=16: DISABLED. Two independent blockers:
//   1. ORT CPU backend does not support 2-bit Q/DQ — it rejects tensor(int2) as a
//      DequantizeLinear input type at model load time (ONNX type check).
//   2. QAIRT HTP backend does not support 2-bit BQ until QAIRT 2.47.
// Re-enable once both are available (with verify_outputs=false until the CPU DQ
// kernel supports int2 for accuracy comparison).
TEST_F(QnnHTPBackendTests, DISABLED_ConvBQ_U16Int2_1x1_BlockSize16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                      /*weight=*/{4, 32, 1, 1},
                                      /*block_size=*/16,
                                      /*bias=*/false,
                                      /*weight_bits=*/2),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

// UINT2, block_size=16: DISABLED. Same blockers as ConvBQ_U16Int2_1x1_BlockSize16 —
// ORT CPU backend does not support 2-bit Q/DQ, and QAIRT HTP backend does not
// support 2-bit BQ until QAIRT 2.47.
TEST_F(QnnHTPBackendTests, DISABLED_ConvBQ_U16UInt2_1x1_BlockSize16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                      /*weight=*/{4, 32, 1, 1},
                                      /*block_size=*/16,
                                      /*bias=*/false,
                                      /*weight_bits=*/2,
                                      /*weight_is_unsigned=*/true),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/false);
}

// UINT4 weight, block_size=8: exercises TransformUnsignedToSignedFixedPoint for 4-bit.
// in0: u16, weight: uint4 (scale=[4,2,1,1], block_size=8), out: u16
TEST_F(QnnHTPBackendTests, ConvBQ_U16UInt4_1x1_NoBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 16, 4, 4},
                                      /*weight=*/{4, 16, 1, 1},
                                      /*block_size=*/8,
                                      /*bias=*/false,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/true),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// UINT4 weight with bias: verifies unsigned weight path works with the FP16 bias dequantization.
TEST_F(QnnHTPBackendTests, ConvBQ_U16UInt4_1x1_WithBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 16, 4, 4},
                                      /*weight=*/{4, 16, 1, 1},
                                      /*block_size=*/8,
                                      /*bias=*/true,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/true),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});
}

// UINT8 weight, block_size=4: exercises TransformUnsignedToSignedFixedPoint for 8-bit.
// block_size=4 is the minimum valid HTP multiple for 8-bit block quantization.
// Tolerance 2e-2f: UINT8 weights [0,126] produce larger FP16 intermediate values than INT4/INT8,
// leading to slightly larger rounding differences between CPU (FP32) and QNN (FP16).
TEST_F(QnnHTPBackendTests, ConvBQ_U16UInt8_1x1_BlockSize4) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 16, 4, 4},
                                      /*weight=*/{4, 16, 1, 1},
                                      /*block_size=*/4,
                                      /*bias=*/false,
                                      /*weight_bits=*/8,
                                      /*weight_is_unsigned=*/true),
                  GetBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// Tests Conv2D with ONNX block-quantized (BQ) weights using the BQ -> QNN LPBQ conversion path.

// LPBQ: 1x1 Conv, INT4 weight, block_size=8, no bias. IC=16, 2 blocks/OC.
TEST_F(QnnHTPBackendTests, ConvLPBQ_U16Int4_1x1_NoBias_BS8) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 16, 4, 4},
                                      /*weight=*/{4, 16, 1, 1},
                                      /*block_size=*/8,
                                      /*include_bias=*/false),
                  GetLPBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// LPBQ: 1x1 Conv, INT4 weight, block_size=16, no bias.
TEST_F(QnnHTPBackendTests, ConvLPBQ_U16Int4_1x1_NoBias_BS16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 64, 4, 4},
                                      /*weight=*/{4, 64, 1, 1},
                                      /*block_size=*/16,
                                      /*include_bias=*/false),
                  GetLPBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// LPBQ: 1x1 Conv, INT4 weight, block_size=16, with per-tensor quantized bias.
TEST_F(QnnHTPBackendTests, ConvLPBQ_U16Int4_1x1_WithBias_BS16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 64, 4, 4},
                                      /*weight=*/{4, 64, 1, 1},
                                      /*block_size=*/16,
                                      /*include_bias=*/true,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false,
                                      /*bias_per_channel=*/false),
                  GetLPBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// LPBQ: 1x1 Conv, INT4 weight, block_size=64, with per-tensor quantized bias.
TEST_F(QnnHTPBackendTests, ConvLPBQ_U16Int4_1x1_WithBias_BS64) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 128, 1, 1},
                                      /*weight=*/{4, 128, 1, 1},
                                      /*block_size=*/64,
                                      /*include_bias=*/true,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false,
                                      /*bias_per_channel=*/false),
                  GetLPBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// LPBQ: 1x1 Conv, INT4 weight, block_size=16, with per-channel quantized bias.
TEST_F(QnnHTPBackendTests, ConvLPBQ_U16Int4_1x1_WithBiasPerChannel_BS16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                      /*weight=*/{4, 32, 1, 1},
                                      /*block_size=*/16,
                                      /*include_bias=*/true,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false,
                                      /*bias_per_channel=*/true),
                  GetLPBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// LPBQ: 1x1 Conv, INT4 weight, block_size=32, with per-channel quantized bias.
TEST_F(QnnHTPBackendTests, ConvLPBQ_U16Int4_1x1_WithBiasPerChannel_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 128, 4, 4},
                                      /*weight=*/{4, 128, 1, 1},
                                      /*block_size=*/32,
                                      /*include_bias=*/true,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false,
                                      /*bias_per_channel=*/true),
                  GetLPBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// LPBQ: 1x1 Conv, INT4 weight, block_size=8, float (unquantized) bias.
TEST_F(QnnHTPBackendTests, ConvLPBQ_U16Int4_1x1_WithFloatBias_BS8) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                      /*weight=*/{4, 32, 1, 1},
                                      /*block_size=*/8,
                                      /*include_bias=*/true,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false,
                                      /*bias_per_channel=*/false,
                                      /*is_bias_quantized=*/false),
                  GetLPBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// LPBQ: 1x1 Conv, INT4 weight, block_size=32, float (unquantized) bias.
TEST_F(QnnHTPBackendTests, ConvLPBQ_U16Int4_1x1_WithFloatBias_BS32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunQnnModelTest(BuildBQConvTestCase(/*input=*/{1, 64, 4, 4},
                                      /*weight=*/{8, 64, 1, 1},
                                      /*block_size=*/32,
                                      /*include_bias=*/true,
                                      /*weight_bits=*/4,
                                      /*weight_is_unsigned=*/false,
                                      /*bias_per_channel=*/false,
                                      /*is_bias_quantized=*/false),
                  GetLPBQConvProviderOptions(),
                  /*opset=*/21,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-2f)});
}

// ── HTP native BQ path ────────────────────────────────────────────────────────
// HTP native BQ kernel adopts QNN_QUANTIZATION_ENCODING_BLOCK encoding type.
// Compared to non-native BQ kernel adopting QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK, no INT16→FP16 activation dequant
// and no FP16→INT16 output quantize are inserted.

// Native path: INT4, block_size=32, IC=32, OC=32.
TEST_F(QnnHTPBackendTests, ConvBQ_Native_U16Int4_BlockSize32) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto htp_arch = GetPlatformAttributes().htp_arch;

  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                          /*weight=*/{32, 32, 1, 1},
                                          /*block_size=*/32,
                                          /*bias=*/false,
                                          /*weight_bits=*/4),
                      /*expect_native_bq=*/htp_arch >= QNN_HTP_DEVICE_ARCH_V81);
}

// Native path: INT4, block_size=64, IC=64, OC=32.
TEST_F(QnnHTPBackendTests, ConvBQ_Native_U16Int4_BlockSize64) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto htp_arch = GetPlatformAttributes().htp_arch;

  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 64, 4, 4},
                                          /*weight=*/{32, 64, 1, 1},
                                          /*block_size=*/64,
                                          /*bias=*/false,
                                          /*weight_bits=*/4),
                      /*expect_native_bq=*/htp_arch >= QNN_HTP_DEVICE_ARCH_V81);
}

// Native path: INT4, block_size=128, IC=128, OC=32.
TEST_F(QnnHTPBackendTests, ConvBQ_Native_U16Int4_BlockSize128) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto htp_arch = GetPlatformAttributes().htp_arch;

  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 128, 4, 4},
                                          /*weight=*/{32, 128, 1, 1},
                                          /*block_size=*/128,
                                          /*bias=*/false,
                                          /*weight_bits=*/4),
                      /*expect_native_bq=*/htp_arch >= QNN_HTP_DEVICE_ARCH_V81);
}

// Native path: INT4, block_size=32, IC=64, OC=64.
TEST_F(QnnHTPBackendTests, ConvBQ_Native_U16Int4_MultiBlock) {
  QNN_SKIP_TEST_IF_NO_PLATFORM_ATTRS();
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  auto htp_arch = GetPlatformAttributes().htp_arch;

  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 64, 4, 4},
                                          /*weight=*/{64, 64, 1, 1},
                                          /*block_size=*/32,
                                          /*bias=*/false,
                                          /*weight_bits=*/4),
                      /*expect_native_bq=*/htp_arch >= QNN_HTP_DEVICE_ARCH_V81);
}

// Fallback edge: Unsupported bitwidth 8.
TEST_F(QnnHTPBackendTests, ConvBQ_NativeFallback_Int8BlockSize32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                          /*weight=*/{32, 32, 1, 1},
                                          /*block_size=*/32,
                                          /*bias=*/false,
                                          /*weight_bits=*/8),
                      /*expect_native_bq=*/false);
}

// Fallback edge: Unsupported block size 16.
TEST_F(QnnHTPBackendTests, ConvBQ_NativeFallback_BlockSize16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                          /*weight=*/{32, 32, 1, 1},
                                          /*block_size=*/16,
                                          /*bias=*/false,
                                          /*weight_bits=*/4),
                      /*expect_native_bq=*/false);
}

// Fallback edge: Unsupported asymmetric offsets.
TEST_F(QnnHTPBackendTests, ConvBQ_NativeFallback_UInt4Asymmetric) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                          /*weight=*/{32, 32, 1, 1},
                                          /*block_size=*/32,
                                          /*bias=*/false,
                                          /*weight_bits=*/4,
                                          /*weight_is_unsigned=*/true),
                      /*expect_native_bq=*/false);
}

// Fallback edge: Unsupported non-32 multiplier OC.
TEST_F(QnnHTPBackendTests, ConvBQ_NativeFallback_OcNotMultipleOf32) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                          /*weight=*/{16, 32, 1, 1},
                                          /*block_size=*/32,
                                          /*bias=*/false,
                                          /*weight_bits=*/4),
                      /*expect_native_bq=*/false);
}

// Fallback edge: Unsupported Conv with bias.
TEST_F(QnnHTPBackendTests, ConvBQ_NativeFallback_ConvWithBias) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunNativeBQConvTest(BuildBQConvTestCase(/*input=*/{1, 32, 4, 4},
                                          /*weight=*/{32, 32, 1, 1},
                                          /*block_size=*/32,
                                          /*bias=*/true,
                                          /*weight_bits=*/4),
                      /*expect_native_bq=*/false);
}

// Tests for reuse_sparse_indices parameter (always false, verifies the parameter is accepted by QNN without errors).
// Conv2d: reuse_sparse_indices should be added to the QNN node parameters.
TEST_F(QnnHTPBackendTests, Conv2D_ReuseSparseIndices) {
  RunHTPConvOpTest<uint8_t, uint8_t>("Conv",
                                     TestInputDef<float>({1, 2, 5, 5}, false, -10.0f, 10.0f),  // Dynamic input
                                     TestInputDef<float>({4, 2, 3, 3}, true, -1.0f, 1.0f),     // Static weights
                                     TestInputDef<float>({4}, true, -1.0f, 1.0f),              // Static bias
                                     {1, 1},                                                   // Strides
                                     {0, 0, 0, 0},                                             // Pads
                                     {1, 1},                                                   // Dilations
                                     1,                                                        // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

// Conv3d: reuse_sparse_indices should be added using QNN_OP_CONV_3D_PARAM_REUSE_SPARSE_INDICIES.
TEST_F(QnnHTPBackendTests, Conv3D_ReuseSparseIndices) {
  RunHTPConvOpTest<uint8_t, int8_t>("Conv",
                                    TestInputDef<float>({1, 2, 4, 4, 4}, false, -10.0f, 10.0f),  // Dynamic input
                                    TestInputDef<float>({4, 2, 2, 2, 2}, true, -1.0f, 1.0f),     // Static weights
                                    TestInputDef<float>({4}, true, -1.0f, 1.0f),                 // Static bias
                                    {1, 1, 1},                                                   // Strides
                                    {0, 0, 0, 0, 0, 0},                                          // Pads
                                    {1, 1, 1},                                                   // Dilations
                                    1,                                                           // default group
                                    "NOTSET",
                                    ExpectedEPNodeAssignment::All);
}

// DepthwiseConv2d: reuse_sparse_indices should NOT be added (group == input_channels == output_channels).
TEST_F(QnnHTPBackendTests, DepthwiseConv2D_NoReuseSparseIndices) {
  std::vector<int64_t> input_shape = {1, 4, 5, 5};
  std::vector<int64_t> weight_shape = {4, 1, 3, 3};
  std::vector<int64_t> bias_shape = {4};

  TestInputDef<float> input_def(input_shape, false,
                                GetFloatDataInRange(-10.0f, 10.0f, SizeOfShape(input_shape)));
  TestInputDef<float> weight_def(weight_shape, true,
                                 GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(weight_shape)));
  TestInputDef<float> bias_def(bias_shape, true,
                               GetFloatDataInRange(-1.0f, 1.0f, SizeOfShape(bias_shape)));

  RunHTPConvOpPerChannelTest<uint8_t, int8_t>("Conv",
                                              input_def,
                                              weight_def,
                                              bias_def,
                                              0,             // weight_quant_axis
                                              {1, 1},        // Strides
                                              {0, 0, 0, 0},  // Pads
                                              {1, 1},        // Dilations
                                              4,             // group == input_channels == output_channels -> DepthwiseConv2d
                                              "NOTSET",
                                              ExpectedEPNodeAssignment::All);
}

// ConvTranspose: reuse_sparse_indices should NOT be added.
TEST_F(QnnHTPBackendTests, ConvTranspose2D_NoReuseSparseIndices) {
  RunHTPConvOpTest<uint8_t, uint8_t>("ConvTranspose",
                                     TestInputDef<float>({1, 2, 4, 4}, false, -10.0f, 10.0f),  // Dynamic input
                                     TestInputDef<float>({2, 4, 3, 3}, true, -1.0f, 1.0f),     // Static weights
                                     TestInputDef<float>({4}, true, -1.0f, 1.0f),              // Static bias
                                     {1, 1},                                                   // Strides
                                     {0, 0, 0, 0},                                             // Pads
                                     {1, 1},                                                   // Dilations
                                     1,                                                        // default group
                                     "NOTSET",
                                     ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_M_ARM64)
//
// GPU tests:
//

// Convolution 2D GPU test.
TEST_F(QnnGPUBackendTests, Conv2D) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // default strides
                {0, 0, 0, 0},                                           // default pads
                {1, 1},                                                 // default dilations
                1,                                                      // default group
                "NOTSET",                                               // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Convolution 3D GPU test.
// Disable Reason : 3D Conv is currently not supported by the GPU.
TEST_F(QnnGPUBackendTests, DISABLED_Conv3D) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),               // Random static bias
                {1, 1, 1},                                                 // default strides
                {0, 0, 0, 0, 0, 0},                                        // default pads
                {1, 1, 1},                                                 // default dilations
                1,                                                         // default group
                "NOTSET",                                                  // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Convolution 2D dynamic bias GPU test.
TEST_F(QnnGPUBackendTests, Conv2D_biasDynamic) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, false, -1.0f, 1.0f),           // Random static bias
                {1, 1},                                                 // default strides
                {0, 0, 0, 0},                                           // default pads
                {1, 1},                                                 // default dilations
                1,                                                      // default group
                "NOTSET",                                               // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Convolution 2D GPU test, large input feature map, more output feature maps.
TEST_F(QnnGPUBackendTests, Conv2D_largeInput) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 3, 60, 452}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({16, 3, 3, 3}, true, 0.0f, 1.0f),      // Random dynamic weights
                TestInputDef<float>({16}, true, -1.0f, 1.0f),              // Random static bias
                {1, 1},                                                    // default strides
                {0, 0, 0, 0},                                              // default pads
                {1, 1},                                                    // default dilations
                1,                                                         // default group
                "NOTSET",                                                  // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu",
                13,
                1e-4f);
}

// Convolution 2D GPU test, reduce featuremaps with pointwise conv.
TEST_F(QnnGPUBackendTests, Conv2D_bottleneckSqueeze) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 32, 16, 113}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({16, 32, 1, 1}, false, -1.0f, 1.0f),    // Random dynamic weights
                TestInputDef<float>({16}, true, -1.0f, 1.0f),               // Random static bias
                {1, 1},                                                     // default strides
                {0, 0, 0, 0},                                               // default pads
                {1, 1},                                                     // default dilations
                1,                                                          // default group
                "NOTSET",                                                   // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu",
                13,
                1e-4f);
}

// Convolution 2D SAME_UPPER pad GPU test.
TEST_F(QnnGPUBackendTests, Conv2D_padSameUpper) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({2, 1, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // default strides
                {},                                                     // unspecified pads
                {1, 1},                                                 // default dilations
                1,                                                      // default group
                "SAME_UPPER",                                           // auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Convolution Transpose 2D GPU test.
TEST_F(QnnGPUBackendTests, ConvTranspose2D) {
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({1, 2, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // default strides
                {0, 0, 0, 0},                                           // default pads
                {1, 1},                                                 // default dilations
                1,                                                      // default group
                "NOTSET",                                               // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Convolution Transpose 2D SAME_LOWER pad GPU test.
TEST_F(QnnGPUBackendTests, ConvTranspose2D_padSameLower) {
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 1, 3, 3}, false, 0.0f, 10.0f),  // Random dynamic input
                TestInputDef<float>({1, 2, 2, 2}, true, 0.0f, 1.0f),    // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),            // Random static bias
                {1, 1},                                                 // default strides
                {},                                                     // unspecified pads
                {1, 1},                                                 // default dilations
                1,                                                      // default group
                "SAME_LOWER",                                           // auto_pad
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Depthwise Convolution 2D GPU test, depthwise conv.
TEST_F(QnnGPUBackendTests, Conv2D_depthwise) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 3, 3, 3}, false, 0.0f, 1.0f),  // Random dynamic input
                TestInputDef<float>({3, 1, 2, 2}, true, 0.0f, 1.0f),   // Random static weights
                TestInputDef<float>({3}, true, -1.0f, 1.0f),           // Random static bias
                {1, 1},                                                // default strides
                {0, 0, 0, 0},                                          // default pads
                {1, 1},                                                // default dilations
                3,                                                     // 3 groups
                "NOTSET",                                              // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Convolution 2D GPU test, reduce featuremaps with depthwise-pointwise conv.
TEST_F(QnnGPUBackendTests, Conv2D_depthwiseSeparable) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 6, 16, 16}, false, -3.0f, 3.0f),  // Random dynamic input
                TestInputDef<float>({3, 2, 1, 1}, false, -1.0f, 1.0f),    // Random dynamic weights
                TestInputDef<float>({3}, true, -1.0f, 1.0f),              // Random static bias
                {1, 1},                                                   // default strides
                {0, 0, 0, 0},                                             // default pads
                {1, 1},                                                   // default dilations
                3,                                                        // 3 groups
                "NOTSET",                                                 // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu",
                13,
                1e-4f);
}

// Convolution 2D groups GPU test, use grouping.
TEST_F(QnnGPUBackendTests, Conv2D_groups) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 4, 3, 3}, false, 0.0f, 1.0f),  // Random dynamic input
                TestInputDef<float>({2, 2, 2, 2}, true, 0.0f, 1.0f),   // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),           // Random static bias
                {1, 1},                                                // default strides
                {0, 0, 0, 0},                                          // default pads
                {1, 1},                                                // default dilations
                2,                                                     // 2 groups
                "NOTSET",                                              // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu",
                13,
                1e-4f);
}

// Convolution 2D groups GPU test, use grouping, more than 1 output per group.
TEST_F(QnnGPUBackendTests, Conv2D_groupsExpand) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 4, 3, 3}, false, 0.0f, 1.0f),  // Random dynamic input
                TestInputDef<float>({4, 2, 2, 2}, true, 0.0f, 1.0f),   // Random static weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),           // Random static bias
                {1, 1},                                                // default strides
                {0, 0, 0, 0},                                          // default pads
                {1, 1},                                                // default dilations
                2,                                                     // 2 groups
                "NOTSET",                                              // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu",
                13,
                1e-4f);
}

// Convolution 2D groups GPU test, use grouping, 1 group of 3.
TEST_F(QnnGPUBackendTests, Conv2D_1groupOf3) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 3, 3, 3}, false, 0.0f, 1.0f),  // Random dynamic input
                TestInputDef<float>({2, 3, 2, 2}, true, 0.0f, 1.0f),   // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),           // Random static bias
                {1, 1},                                                // default strides
                {0, 0, 0, 0},                                          // default pads
                {1, 1},                                                // default dilations
                1,                                                     // 1 groups
                "NOTSET",                                              // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu",
                13,
                1e-4f);
}

// Convolution 2D groups GPU test, use grouping, more than 1 group of 3.
// Disable Reason : doesn't work.
TEST_F(QnnGPUBackendTests, DISABLED_Conv2D_2groupsOf3) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 6, 3, 3}, false, 0.0f, 1.0f),  // Random dynamic input
                TestInputDef<float>({2, 3, 2, 2}, true, 0.0f, 1.0f),   // Random static weights
                TestInputDef<float>({2}, true, -1.0f, 1.0f),           // Random static bias
                {1, 1},                                                // default strides
                {0, 0, 0, 0},                                          // default pads
                {1, 1},                                                // default dilations
                2,                                                     // 2 groups
                "NOTSET",                                              // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu",
                13,
                1e-4f);
}

// Convolution 2D groups GPU test, use grouping, more than 1 group of 2.
TEST_F(QnnGPUBackendTests, Conv2D_3groupsOf2) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 6, 3, 3}, false, 0.0f, 1.0f),  // Random dynamic input
                TestInputDef<float>({3, 2, 2, 2}, true, 0.0f, 1.0f),   // Random static weights
                TestInputDef<float>({3}, true, -1.0f, 1.0f),           // Random static bias
                {1, 1},                                                // default strides
                {0, 0, 0, 0},                                          // default pads
                {1, 1},                                                // default dilations
                3,                                                     // 3 groups
                "NOTSET",                                              // No auto-padding
                ExpectedEPNodeAssignment::All,
                "gpu",
                13,
                1e-4f);
}

// Convolution 1D GPU test.
TEST_F(QnnGPUBackendTests, Conv1D) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 2, 4}, false, input_data),               // Dynamic input
                TestInputDef<float>({1, 2, 2}, true, {1.0f, 2.0f, 3.0f, 4.0f}),  // Static weights
                TestInputDef<float>({1}, true, {1.0f}),                          // Initializer Bias
                {1},                                                             // Strides
                {0, 0},                                                          // Pads
                {1},                                                             // Dilations
                1,                                                               // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Convolution Transpose 1D GPU test.
TEST_F(QnnGPUBackendTests, ConvTranspose1D) {
  std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 2, 4}, false, input_data),               // Dynamic input
                TestInputDef<float>({2, 1, 2}, true, {1.0f, 2.0f, 3.0f, 4.0f}),  // Static weights
                TestInputDef<float>({1}, true, {0.0f}),                          // Zero bias
                {1},                                                             // Strides
                {0, 0},                                                          // Pads
                {1},                                                             // Dilations
                1,                                                               // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Tests for reuse_sparse_indices parameter (always false, verifies the parameter is accepted by QNN without errors).
// Conv2d: reuse_sparse_indices should be added to the QNN node parameters.
TEST_F(QnnGPUBackendTests, Conv2D_ReuseSparseIndices) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 2, 5, 5}, false, -10.0f, 10.0f),  // Dynamic input
                TestInputDef<float>({4, 2, 3, 3}, true, -1.0f, 1.0f),     // Static weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),              // Static bias
                {1, 1},                                                   // Strides
                {0, 0, 0, 0},                                             // Pads
                {1, 1},                                                   // Dilations
                1,                                                        // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// Conv3d: GPU does not support Conv3D.
TEST_F(QnnGPUBackendTests, DISABLED_Conv3D_ReuseSparseIndices) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 2, 4, 4, 4}, false, -10.0f, 10.0f),  // Dynamic input
                TestInputDef<float>({4, 2, 2, 2, 2}, true, -1.0f, 1.0f),     // Static weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),                 // Static bias
                {1, 1, 1},                                                   // Strides
                {0, 0, 0, 0, 0, 0},                                          // Pads
                {1, 1, 1},                                                   // Dilations
                1,                                                           // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// DepthwiseConv2d: reuse_sparse_indices should NOT be added (group == input_channels == output_channels).
TEST_F(QnnGPUBackendTests, DepthwiseConv2D_NoReuseSparseIndices) {
  RunConvOpTest("Conv",
                TestInputDef<float>({1, 4, 5, 5}, false, -10.0f, 10.0f),  // Dynamic input
                TestInputDef<float>({4, 1, 3, 3}, true, -1.0f, 1.0f),     // Depthwise weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),              // Static bias
                {1, 1},                                                   // Strides
                {0, 0, 0, 0},                                             // Pads
                {1, 1},                                                   // Dilations
                4,                                                        // group == input_channels == output_channels -> DepthwiseConv2d
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "gpu");
}

// ConvTranspose: reuse_sparse_indices should NOT be added.
TEST_F(QnnGPUBackendTests, ConvTranspose2D_NoReuseSparseIndices) {
  RunConvOpTest("ConvTranspose",
                TestInputDef<float>({1, 2, 4, 4}, false, -10.0f, 10.0f),  // Dynamic input
                TestInputDef<float>({2, 4, 3, 3}, true, -1.0f, 1.0f),     // Static weights
                TestInputDef<float>({4}, true, -1.0f, 1.0f),              // Static bias
                {1, 1},                                                   // Strides
                {0, 0, 0, 0},                                             // Pads
                {1, 1},                                                   // Dilations
                1,                                                        // default group
                "NOTSET",
                ExpectedEPNodeAssignment::All,
                "gpu");
}

#endif  // defined(_M_ARM64) GPU tests

}  // namespace test
}  // namespace onnxruntime

#endif
