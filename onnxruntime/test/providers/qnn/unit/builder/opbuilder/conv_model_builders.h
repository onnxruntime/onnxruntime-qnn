// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Inline model-builder helpers for Conv/ConvTranspose tests.
//
// Shared between conv_session_test.cc (snapshot tier) and
// conv_accuracy_test.cc (accuracy tier).  All functions are inline so this
// header can be included from either translation unit without ODR violations.
//
// Do NOT include from any TU that uses qnn_unit_test_utils.h / ort_api.h:
// qnn_test_utils.h (pulled here) and ort_api.h define kOnnxDomain
// independently, causing a double-define ODR error.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <optional>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/unit/builder/opbuilder/conv_specs.h"

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Spec → TestInputDef
// ---------------------------------------------------------------------------

inline TestInputDef<float> ToInputDef(const TensorSpec& ts) {
  if (ts.shape.empty()) return {};
  std::vector<int64_t> shape(ts.shape.begin(), ts.shape.end());
  if (!ts.fixed_data.empty())
    return {shape, ts.is_static, ts.fixed_data};
  return {shape, ts.is_static, ts.min_val, ts.max_val};
}

// ---------------------------------------------------------------------------
// Backend provider options
// ---------------------------------------------------------------------------

inline ProviderOptions BackendOptions(SnapshotBackend b) {
  ProviderOptions opts;
  opts["backend_type"] = (b == SnapshotBackend::CPU) ? "cpu" : "htp";
  opts["offload_graph_io_quantization"] = "0";
  return opts;
}

// ---------------------------------------------------------------------------
// Conv attribute helper — avoids repeating the same attrs push_back pattern.
// ---------------------------------------------------------------------------

inline void AddConvNodeAttrs(ModelTestBuilder& builder,
                              std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                              const std::string& auto_pad,
                              int64_t group,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& pads,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& output_shape) {
  attrs.push_back(builder.MakeStringAttribute("auto_pad", auto_pad));
  attrs.push_back(builder.MakeScalarAttribute("group", group));
  if (!pads.empty() && auto_pad == "NOTSET")
    attrs.push_back(builder.MakeIntsAttribute("pads", pads));
  if (!strides.empty())
    attrs.push_back(builder.MakeIntsAttribute("strides", strides));
  if (!dilations.empty())
    attrs.push_back(builder.MakeIntsAttribute("dilations", dilations));
  if (!output_shape.empty())
    attrs.push_back(builder.MakeIntsAttribute("output_shape", output_shape));
}

// ---------------------------------------------------------------------------
// Per-channel weight setup helpers
// ---------------------------------------------------------------------------

// Materializes a range-based TestInputDef<float> into a fixed-data one so
// that GetRangePerChannel() returns actual per-channel ranges (not the global
// rand_min/rand_max repeated for every channel). No-op for raw-data inputs.
inline TestInputDef<float> MaterializeWeightDef(const TestInputDef<float>& def) {
  if (def.IsRawData()) return def;
  const auto& rand_info = def.GetRandomDataInfo();
  return TestInputDef<float>(
      def.GetShape(), def.IsInitializer(),
      GetFloatDataInRange(rand_info.min, rand_info.max, SizeOfShape(def.GetShape())));
}

// Generates random float data for a range-based TestInputDef using the builder's
// random engine (seed=2345), matching what BuildConvF32ReferenceFn's MakeTestInput
// calls produce. This ensures F32 and QDQ models see identical weight/bias values.
// No-op for raw-data inputs (already have fixed data).
inline TestInputDef<float> MaterializeWithBuilderRng(const TestInputDef<float>& def,
                                                      ModelTestBuilder& builder,
                                                      const std::string& temp_name) {
  if (def.IsRawData()) return def;
  const auto& rand_info = def.GetRandomDataInfo();
  // MakeInitializer<float> uses builder.rand_gen_.Uniform<float> (same as
  // the F32 reference builder's MakeTestInput call), advancing the random
  // engine by exactly the same number of steps.
  const auto* proto = builder.MakeInitializer<float>(
      temp_name,
      std::vector<int64_t>(def.GetShape()),
      rand_info.min, rand_info.max);
  // Extract raw float data from the TensorProto (ORT always uses raw_data format).
  const auto& raw = proto->raw_data();
  const float* floats = reinterpret_cast<const float*>(raw.data());
  return TestInputDef<float>(def.GetShape(), def.IsInitializer(),
                              std::vector<float>(floats, floats + raw.size() / sizeof(float)));
}

// S8 per-channel: returns "weights_dq"; use_contrib gates the DQ domain.
inline void SetupPerChannelS8Weight(
    ModelTestBuilder& builder,
    const TestInputDef<float>& w_def,
    int64_t axis,
    bool use_contrib,
    std::vector<float>& w_scales_out) {
  const auto w_mat = MaterializeWeightDef(w_def);
  std::vector<int8_t> w_zps;
  GetTestInputQuantParamsPerChannel<int8_t>(w_mat, w_scales_out, w_zps, axis, true);
  std::vector<int8_t> w_quant(SizeOfShape(w_mat.GetShape()));
  QuantizeValues<float, int8_t>(w_mat.GetRawData(), w_quant,
                                 w_mat.GetShape(), w_scales_out, w_zps,
                                 static_cast<size_t>(axis));
  builder.MakeInitializer<int8_t>("weights_quant", w_mat.GetShape(), w_quant);
  std::vector<ONNX_NAMESPACE::AttributeProto> w_dq_attrs;
  w_dq_attrs.push_back(builder.MakeScalarAttribute("axis", axis));
  builder.AddDequantizeLinearNode("WeightDQ", "weights_quant", w_scales_out, w_zps,
                                   "weights_dq", w_dq_attrs, use_contrib);
}

// S4 per-channel (INT4, opset 21, no contrib). quant_axis is the positive axis
// used for quantization; attr_axis is the value stored in the DQ attribute
// (may be negative, e.g. -4 normalises to 0 for a 4-D tensor).
inline void SetupPerChannelS4Weight(
    ModelTestBuilder& builder,
    const TestInputDef<float>& w_def,
    int64_t quant_axis,
    int64_t attr_axis,
    std::vector<float>& w_scales_out) {
  const auto w_mat = MaterializeWeightDef(w_def);
  std::vector<Int4x2> w_zps;
  GetTestInputQuantParamsPerChannel<Int4x2>(w_mat, w_scales_out, w_zps, quant_axis, true);
  const size_t w_storage = Int4x2::CalcNumInt4Pairs(SizeOfShape(w_mat.GetShape()));
  std::vector<Int4x2> w_quant(w_storage);
  QuantizeValues<float, Int4x2>(w_mat.GetRawData(), w_quant,
                                 w_mat.GetShape(), w_scales_out, w_zps,
                                 static_cast<size_t>(quant_axis));
  builder.MakeInitializer<Int4x2>("weights_quant", w_mat.GetShape(), w_quant);
  std::vector<ONNX_NAMESPACE::AttributeProto> w_dq_attrs;
  w_dq_attrs.push_back(builder.MakeScalarAttribute("axis", attr_axis));
  builder.AddDequantizeLinearNode("WeightDQ", "weights_quant", w_scales_out, w_zps,
                                   "weights_dq", w_dq_attrs);
}

// S16 per-channel (contrib QDQ, U16/S16 tests).
inline void SetupPerChannelS16Weight(
    ModelTestBuilder& builder,
    const TestInputDef<float>& w_def,
    int64_t axis,
    std::vector<float>& w_scales_out) {
  const auto w_mat = MaterializeWeightDef(w_def);
  std::vector<int16_t> w_zps;
  GetTestInputQuantParamsPerChannel<int16_t>(w_mat, w_scales_out, w_zps, axis, true);
  std::vector<int16_t> w_quant(SizeOfShape(w_mat.GetShape()));
  QuantizeValues<float, int16_t>(w_mat.GetRawData(), w_quant,
                                  w_mat.GetShape(), w_scales_out, w_zps,
                                  static_cast<size_t>(axis));
  builder.MakeInitializer<int16_t>("weights_quant", w_mat.GetShape(), w_quant);
  std::vector<ONNX_NAMESPACE::AttributeProto> w_dq_attrs;
  w_dq_attrs.push_back(builder.MakeScalarAttribute("axis", axis));
  builder.AddDequantizeLinearNode("WeightDQ", "weights_quant", w_scales_out, w_zps,
                                   "weights_dq", w_dq_attrs, /*use_contrib_qdq=*/true);
}

// ---------------------------------------------------------------------------
// BuildF32ConvFn — Float32 Conv / ConvTranspose (CPU backend).
// ---------------------------------------------------------------------------

inline GetTestModelFn BuildF32ConvFn(
    const std::string& conv_op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    std::optional<int64_t> group,
    const std::string& auto_pad = "NOTSET",
    const std::vector<int64_t>& output_shape = {}) {
  const int64_t grp = group.value_or(1);
  return [conv_op_type, input_def, weights_def, bias_def, strides, pads,
          dilations, grp, auto_pad, output_shape](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    MakeTestInput<float>(builder, "weights", weights_def);
    std::vector<std::string> conv_input_names{"input", "weights"};
    if (!bias_def.GetShape().empty()) {
      MakeTestInput<float>(builder, "bias", bias_def);
      conv_input_names.push_back("bias");
    }
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    AddConvNodeAttrs(builder, conv_attrs, auto_pad, grp, strides, pads, dilations, output_shape);
    builder.MakeOutput("output");
    builder.AddNode("Conv", conv_op_type, conv_input_names, {"output"},
                    kOnnxDomain, conv_attrs);
  };
}

// ---------------------------------------------------------------------------
// BuildConvU8U8S32Fn — U8/U8 per-tensor QDQ Conv / ConvTranspose.
// Handles 1D/2D/3D, with/without bias, static/dynamic, any auto_pad.
// bias_scale_multiplier != 1.0 triggers the requantization path.
// output_shape_attr used for ConvTranspose tests with explicit output_shape.
// ---------------------------------------------------------------------------

inline GetTestModelFn BuildConvU8U8S32Fn(
    const std::string& op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    int64_t group = 1,
    const std::string& auto_pad = "NOTSET",
    float bias_scale_multiplier = 1.0f,
    std::vector<int64_t> output_shape_attr = {}) {
  return [op_type, input_def, weights_def, bias_def, strides, pads, dilations,
          group, auto_pad, bias_scale_multiplier, output_shape_attr](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    const auto in_qp = GetTestInputQuantParams<uint8_t>(input_def);
    const std::string in_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_input", "input",
                                                        in_qp.scale, in_qp.zero_point);
    MakeTestInput<float>(builder, "weights", weights_def);
    const auto w_qp = GetTestInputQuantParams<uint8_t>(weights_def);
    const std::string w_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_weights", "weights",
                                                       w_qp.scale, w_qp.zero_point);
    std::vector<std::string> conv_inputs{in_qdq, w_qdq};
    if (!bias_def.GetShape().empty()) {
      conv_inputs.push_back(MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                  in_qp.scale * w_qp.scale * bias_scale_multiplier));
    }
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    AddConvNodeAttrs(builder, attrs, auto_pad, group, strides, pads, dilations, output_shape_attr);
    builder.AddNode("Conv", op_type, conv_inputs, {"Y"}, kOnnxDomain, attrs);
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "Y",
                                                   in_qp.scale, in_qp.zero_point);
  };
}

// ---------------------------------------------------------------------------
// BuildConvU16U8S32Fn — U16/U8 per-tensor QDQ Conv / ConvTranspose.
// Uses com.microsoft contrib Q/DQ domain (use_ms_domain=true).
// ---------------------------------------------------------------------------

inline GetTestModelFn BuildConvU16U8S32Fn(
    const std::string& op_type,
    const TestInputDef<float>& input_def,
    const TestInputDef<float>& weights_def,
    const TestInputDef<float>& bias_def,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    int64_t group = 1,
    const std::string& auto_pad = "NOTSET") {
  return [op_type, input_def, weights_def, bias_def, strides, pads, dilations,
          group, auto_pad](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    const auto in_qp = GetTestInputQuantParams<uint16_t>(input_def);
    const std::string in_qdq = AddQDQNodePair<uint16_t>(builder, "qdq_input", "input",
                                                         in_qp.scale, in_qp.zero_point,
                                                         /*use_ms_domain=*/true);
    MakeTestInput<float>(builder, "weights", weights_def);
    const auto w_qp = GetTestInputQuantParams<uint8_t>(weights_def);
    const std::string w_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_weights", "weights",
                                                        w_qp.scale, w_qp.zero_point,
                                                        /*use_ms_domain=*/true);
    std::vector<std::string> conv_inputs{in_qdq, w_qdq};
    if (!bias_def.GetShape().empty()) {
      conv_inputs.push_back(MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                  in_qp.scale * w_qp.scale,
                                                  /*use_contrib_qdq=*/true));
    }
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    AddConvNodeAttrs(builder, attrs, auto_pad, group, strides, pads, dilations, {});
    builder.AddNode("Conv", op_type, conv_inputs, {"Y"}, kOnnxDomain, attrs);
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_out", "Y",
                                                    in_qp.scale, in_qp.zero_point,
                                                    /*use_ms_domain=*/true);
  };
}

// ---------------------------------------------------------------------------
// Fusion builders (fixed shapes: input={1,2,4,4}, weights={3,2,2,2}, bias={3})
// ---------------------------------------------------------------------------

inline GetTestModelFn BuildConvU8U8S32_ReluFn() {
  const TestInputDef<float> input_def({1, 2, 4, 4}, false,
                                      GetFloatDataInRange(0.0f, 1.0f, 32));
  const TestInputDef<float> weights_def({3, 2, 2, 2}, true,
                                        GetFloatDataInRange(-1.0f, 5.0f, 24));
  const TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));
  return [input_def, weights_def, bias_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    const auto in_qp = GetTestInputQuantParams<uint8_t>(input_def);
    const std::string in_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_input", "input",
                                                        in_qp.scale, in_qp.zero_point);
    MakeTestInput<float>(builder, "weights", weights_def);
    const auto w_qp = GetTestInputQuantParams<uint8_t>(weights_def);
    const std::string w_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_weights", "weights",
                                                       w_qp.scale, w_qp.zero_point);
    const std::string bias_in = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                      in_qp.scale * w_qp.scale);
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    builder.AddNode("Conv", "Conv", {in_qdq, w_qdq, bias_in}, {"Y"}, kOnnxDomain, attrs);
    builder.AddNode("relu_node", "Relu", {"Y"}, {"relu_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "relu_out",
                                                   in_qp.scale, in_qp.zero_point);
  };
}

inline GetTestModelFn BuildConvU8U8S32_RedundantClipFn() {
  const TestInputDef<float> input_def({1, 2, 4, 4}, false,
                                      GetFloatDataInRange(0.0f, 1.0f, 32));
  const TestInputDef<float> weights_def({3, 2, 2, 2}, true,
                                        GetFloatDataInRange(-1.0f, 5.0f, 24));
  const TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));
  return [input_def, weights_def, bias_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    const auto in_qp = GetTestInputQuantParams<uint8_t>(input_def);
    const std::string in_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_input", "input",
                                                        in_qp.scale, in_qp.zero_point,
                                                        /*use_contrib_qdq=*/true);
    MakeTestInput<float>(builder, "weights", weights_def);
    const auto w_qp = GetTestInputQuantParams<uint8_t>(weights_def);
    const std::string w_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_weights", "weights",
                                                       w_qp.scale, w_qp.zero_point,
                                                       /*use_contrib_qdq=*/true);
    const std::string bias_in = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                      in_qp.scale * w_qp.scale,
                                                      /*use_contrib_qdq=*/true);
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    conv_attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    conv_attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    conv_attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    builder.AddNode("Conv", "Conv", {in_qdq, w_qdq, bias_in}, {"conv_out"},
                    kOnnxDomain, conv_attrs);
    builder.MakeScalarInitializer<float>("clip_min", -2.0f);
    builder.MakeScalarInitializer<float>("clip_max", 2.0f);
    builder.AddNode("clip_node", "Clip", {"conv_out", "clip_min", "clip_max"}, {"clip_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "clip_out",
                                                   in_qp.scale, in_qp.zero_point,
                                                   /*use_contrib_qdq=*/true);
  };
}

inline GetTestModelFn BuildConvS8S8S32_PerChannel_ReluFn() {
  const TestInputDef<float> input_def({1, 2, 4, 4}, false,
                                      GetFloatDataInRange(0.0f, 1.0f, 32));
  const TestInputDef<float> weights_def({3, 2, 2, 2}, true,
                                        GetFloatDataInRange(-1.0f, 5.0f, 24));
  const TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));
  return [input_def, weights_def, bias_def](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    const auto in_qp = GetTestInputQuantParams<int8_t>(input_def);
    const std::string in_qdq = AddQDQNodePair<int8_t>(builder, "qdq_input", "input",
                                                       in_qp.scale, in_qp.zero_point);
    std::vector<float> w_scales;
    SetupPerChannelS8Weight(builder, weights_def, /*axis=*/0, /*use_contrib=*/false, w_scales);
    const std::string bias_in = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                      in_qp.scale * w_scales[0]);
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    builder.AddNode("Conv", "Conv", {in_qdq, "weights_dq", bias_in}, {"Y"},
                    kOnnxDomain, attrs);
    builder.AddNode("relu_node", "Relu", {"Y"}, {"relu_out"});
    AddQDQNodePairWithOutputAsGraphOutput<int8_t>(builder, "qdq_out", "relu_out",
                                                  in_qp.scale, in_qp.zero_point);
  };
}

// ---------------------------------------------------------------------------
// BuildConvModelFn — universal builder driven by ConvSpec.
// ---------------------------------------------------------------------------

inline GetTestModelFn BuildConvModelFn(const ConvSpec& s) {
  const std::string op_type(s.op_type);
  const std::string auto_pad(s.auto_pad);
  TestInputDef<float> in_def = ToInputDef(s.input);
  TestInputDef<float> w_def = ToInputDef(s.weights);
  TestInputDef<float> b_def = ToInputDef(s.bias);
  std::vector<int64_t> strides(s.strides.begin(), s.strides.end());
  std::vector<int64_t> pads(s.pads.begin(), s.pads.end());
  std::vector<int64_t> dilations(s.dilations.begin(), s.dilations.end());
  std::vector<int64_t> output_shape(s.output_shape.begin(), s.output_shape.end());
  const int64_t group = s.group;
  const float bsm = s.bias_scale_multiplier;
  const int64_t wqa = s.weight_quant_axis;
  const bool contrib = s.use_contrib_qdq;

  // --- F32 ---
  if (s.quant_mode == ConvQuantMode::None) {
    const std::optional<int64_t> grp = std::make_optional(group);
    return BuildF32ConvFn(op_type, in_def, w_def, b_def, strides, pads, dilations,
                          grp, auto_pad);
  }

  // --- Per-tensor ---
  if (s.quant_mode == ConvQuantMode::PerTensor) {
    if (s.input_type == ConvInputType::U8 && s.weight_type == ConvWeightType::U8) {
      return BuildConvU8U8S32Fn(op_type, in_def, w_def, b_def, strides, pads, dilations,
                                group, auto_pad, bsm, output_shape);
    }
    if (s.input_type == ConvInputType::U16 && s.weight_type == ConvWeightType::U8) {
      return BuildConvU16U8S32Fn(op_type, in_def, w_def, b_def, strides, pads, dilations,
                                  group, auto_pad);
    }
    if (s.input_type == ConvInputType::U16 && s.weight_type == ConvWeightType::U16) {
      // U16/U16 per-tensor, no bias, no contrib QDQ (opset 21 standard)
      return [in_def, w_def, op_type, strides, pads, dilations, auto_pad]
             (ModelTestBuilder& builder) {
        MakeTestInput<float>(builder, "input", in_def);
        const auto in_qp = GetTestInputQuantParams<uint16_t>(in_def);
        const std::string in_qdq = AddQDQNodePair<uint16_t>(builder, "qdq_input", "input",
                                                             in_qp.scale, in_qp.zero_point);
        MakeTestInput<float>(builder, "weights", w_def);
        const auto w_qp = GetTestInputQuantParams<uint16_t>(w_def);
        const std::string w_qdq = AddQDQNodePair<uint16_t>(builder, "qdq_weights", "weights",
                                                             w_qp.scale, w_qp.zero_point);
        std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
        attrs.push_back(builder.MakeStringAttribute("auto_pad", auto_pad));
        if (!pads.empty() && auto_pad == "NOTSET")
          attrs.push_back(builder.MakeIntsAttribute("pads", pads));
        if (!strides.empty())
          attrs.push_back(builder.MakeIntsAttribute("strides", strides));
        if (!dilations.empty())
          attrs.push_back(builder.MakeIntsAttribute("dilations", dilations));
        builder.AddNode("Conv", op_type, {in_qdq, w_qdq}, {"Y"}, kOnnxDomain, attrs);
        AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_out", "Y",
                                                        in_qp.scale, in_qp.zero_point);
      };
    }
    if (s.input_type == ConvInputType::U16 && s.weight_type == ConvWeightType::S16) {
      // U16/S16 per-tensor (contrib QDQ, dynamic weight)
      return [in_def, w_def, b_def, op_type, strides, pads, dilations, auto_pad]
             (ModelTestBuilder& builder) {
        MakeTestInput<float>(builder, "input", in_def);
        const auto in_qp = GetTestInputQuantParams<uint16_t>(in_def);
        const std::string in_qdq = AddQDQNodePair<uint16_t>(builder, "qdq_input", "input",
                                                             in_qp.scale, in_qp.zero_point,
                                                             /*use_ms_domain=*/true);
        MakeTestInput<float>(builder, "weights", w_def);
        const auto w_qp = GetTestInputQuantParams<int16_t>(w_def);
        const std::string w_qdq = AddQDQNodePair<int16_t>(builder, "qdq_weights", "weights",
                                                            w_qp.scale, w_qp.zero_point,
                                                            /*use_ms_domain=*/true);
        std::vector<std::string> conv_inputs{in_qdq, w_qdq};
        if (!b_def.GetShape().empty()) {
          conv_inputs.push_back(MakeTestQDQBiasInput(builder, "bias", b_def,
                                                      in_qp.scale * w_qp.scale,
                                                      /*use_contrib_qdq=*/true));
        }
        std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
        attrs.push_back(builder.MakeStringAttribute("auto_pad", auto_pad));
        if (!pads.empty() && auto_pad == "NOTSET")
          attrs.push_back(builder.MakeIntsAttribute("pads", pads));
        if (!strides.empty())
          attrs.push_back(builder.MakeIntsAttribute("strides", strides));
        if (!dilations.empty())
          attrs.push_back(builder.MakeIntsAttribute("dilations", dilations));
        builder.AddNode("Conv", op_type, conv_inputs, {"Y"}, kOnnxDomain, attrs);
        AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_out", "Y",
                                                        in_qp.scale, in_qp.zero_point,
                                                        /*use_ms_domain=*/true);
      };
    }
  }

  // --- Per-channel ---
  if (s.quant_mode == ConvQuantMode::PerChannel) {
    if (s.weight_type == ConvWeightType::S8) {
      // Input: U8 or U16; Weight: S8 per-channel.
      // contrib flag gates the QDQ domain for all nodes.
      const bool is_u16 = (s.input_type == ConvInputType::U16);
      return [in_def, w_def, b_def, op_type, strides, pads, dilations, group, auto_pad,
              output_shape, wqa, contrib, is_u16, bsm](ModelTestBuilder& builder) {
        MakeTestInput<float>(builder, "input", in_def);
        float in_scale;
        std::string in_qdq;
        uint8_t u8_zp = 0;
        float u8_scale = 0.0f;
        uint16_t u16_zp = 0;
        float u16_scale = 0.0f;
        if (is_u16) {
          const auto in_qp = GetTestInputQuantParams<uint16_t>(in_def);
          in_scale = u16_scale = in_qp.scale;
          u16_zp = in_qp.zero_point;
          in_qdq = AddQDQNodePair<uint16_t>(builder, "qdq_input", "input",
                                             in_qp.scale, in_qp.zero_point,
                                             /*use_ms_domain=*/contrib);
        } else {
          const auto in_qp = GetTestInputQuantParams<uint8_t>(in_def);
          in_scale = u8_scale = in_qp.scale;
          u8_zp = in_qp.zero_point;
          in_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_input", "input",
                                            in_qp.scale, in_qp.zero_point);
        }
        std::vector<float> w_scales;
        SetupPerChannelS8Weight(builder, w_def, wqa, contrib, w_scales);
        std::vector<std::string> conv_inputs{in_qdq, "weights_dq"};
        if (!b_def.GetShape().empty()) {
          conv_inputs.push_back(MakeTestQDQBiasInput(builder, "bias", b_def,
                                                      in_scale * w_scales[0] * bsm,
                                                      contrib));
        }
        std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
        AddConvNodeAttrs(builder, attrs, auto_pad, group, strides, pads, dilations, output_shape);
        builder.AddNode("Conv", op_type, conv_inputs, {"Y"}, kOnnxDomain, attrs);
        if (is_u16) {
          AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_out", "Y",
                                                          u16_scale, u16_zp,
                                                          /*use_ms_domain=*/contrib);
        } else {
          AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "Y",
                                                         u8_scale, u8_zp);
        }
      };
    }

    if (s.weight_type == ConvWeightType::S4) {
      // Input: U16; Weight: S4 per-channel (INT4, opset 21, no contrib).
      // Per-channel S32 bias (scale[i] = in_qp.scale * w_scales[i]).
      // wqa may be negative (e.g. -4 normalises to 0 for 4-D weight).
      const int64_t quant_axis = (wqa >= 0) ? wqa
                                             : wqa + static_cast<int64_t>(s.weights.shape.size());
      return [in_def, w_def, b_def, op_type, strides, pads, dilations, group, auto_pad,
              output_shape, quant_axis, wqa](ModelTestBuilder& builder) {
        MakeTestInput<float>(builder, "input", in_def);
        const auto in_qp = GetTestInputQuantParams<uint16_t>(in_def);
        const std::string in_qdq = AddQDQNodePair<uint16_t>(builder, "qdq_input", "input",
                                                             in_qp.scale, in_qp.zero_point);
        std::vector<float> w_scales;
        SetupPerChannelS4Weight(builder, w_def, quant_axis, wqa, w_scales);
        std::vector<std::string> conv_inputs{in_qdq, "weights_dq"};
        if (!b_def.GetShape().empty()) {
          const int64_t n_ch = static_cast<int64_t>(w_scales.size());
          std::vector<float> b_scales(n_ch);
          std::vector<int32_t> b_zps(n_ch, 0);
          for (int64_t i = 0; i < n_ch; ++i)
            b_scales[i] = in_qp.scale * w_scales[i];
          const auto b_mat = MaterializeWeightDef(b_def);
          const auto& bias_data = b_mat.GetRawData();
          std::vector<int32_t> b_quant(n_ch);
          for (int64_t i = 0; i < n_ch; ++i)
            b_quant[i] = static_cast<int32_t>(std::round(bias_data[i] / b_scales[i]));
          builder.MakeInitializer<int32_t>("bias_quant", {n_ch}, b_quant);
          std::vector<ONNX_NAMESPACE::AttributeProto> b_dq_attrs;
          b_dq_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));
          builder.AddDequantizeLinearNode("BiasDQ", "bias_quant", b_scales, b_zps,
                                          "bias_dq", b_dq_attrs);
          conv_inputs.push_back("bias_dq");
        }
        std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
        AddConvNodeAttrs(builder, attrs, auto_pad, group, strides, pads, dilations, output_shape);
        builder.AddNode("Conv", op_type, conv_inputs, {"Y"}, kOnnxDomain, attrs);
        AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_out", "Y",
                                                        in_qp.scale, in_qp.zero_point);
      };
    }

    if (s.weight_type == ConvWeightType::S16) {
      // Input: U16; Weight: S16 per-channel (contrib QDQ).
      return [in_def, w_def, b_def, op_type, strides, pads, dilations, group, auto_pad,
              output_shape, wqa](ModelTestBuilder& builder) {
        MakeTestInput<float>(builder, "input", in_def);
        const auto in_qp = GetTestInputQuantParams<uint16_t>(in_def);
        const std::string in_qdq = AddQDQNodePair<uint16_t>(builder, "qdq_input", "input",
                                                             in_qp.scale, in_qp.zero_point,
                                                             /*use_ms_domain=*/true);
        std::vector<float> w_scales;
        SetupPerChannelS16Weight(builder, w_def, wqa, w_scales);
        std::vector<std::string> conv_inputs{in_qdq, "weights_dq"};
        if (!b_def.GetShape().empty()) {
          conv_inputs.push_back(MakeTestQDQBiasInput(builder, "bias", b_def,
                                                      in_qp.scale * w_scales[0],
                                                      /*use_contrib_qdq=*/true));
        }
        std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
        AddConvNodeAttrs(builder, attrs, auto_pad, group, strides, pads, dilations, output_shape);
        builder.AddNode("Conv", op_type, conv_inputs, {"Y"}, kOnnxDomain, attrs);
        AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_out", "Y",
                                                        in_qp.scale, in_qp.zero_point,
                                                        /*use_ms_domain=*/true);
      };
    }
  }

  return [](ModelTestBuilder&) { FAIL() << "BuildConvModelFn: unhandled ConvSpec combination"; };
}

// ---------------------------------------------------------------------------
// BuildConvFusionModelFn — dispatcher for ConvFusionSpec (Conv + activation).
// ---------------------------------------------------------------------------

inline GetTestModelFn BuildConvFusionModelFn(const ConvFusionSpec& s) {
  if (s.input_type == ConvFusionInputType::U8) {
    if (s.fusion_type == ConvFusionType::Relu) return BuildConvU8U8S32_ReluFn();
    if (s.fusion_type == ConvFusionType::ClipRedundant) return BuildConvU8U8S32_RedundantClipFn();
  }
  if (s.input_type == ConvFusionInputType::S8) {
    return BuildConvS8S8S32_PerChannel_ReluFn();
  }
  return [](ModelTestBuilder&) { FAIL() << "BuildConvFusionModelFn: unhandled spec"; };
}

// ---------------------------------------------------------------------------
// BuildConvF32ReferenceFn — F32 reference model derived from a QDQ ConvSpec.
// Used by TestQDQModelAccuracy as the float oracle (same topology, no Q/DQ).
// Uses range-based TestInputDefs so ModelTestBuilder's seed=2345 random engine
// generates the same data as BuildConvQDQFn (which also uses seed=2345).
// ---------------------------------------------------------------------------

inline GetTestModelFn BuildConvF32ReferenceFn(const ConvSpec& s) {
  const std::optional<int64_t> grp = std::make_optional(static_cast<int64_t>(s.group));
  return BuildF32ConvFn(std::string(s.op_type),
                        MaterializeWeightDef(ToInputDef(s.input)),
                        MaterializeWeightDef(ToInputDef(s.weights)),
                        MaterializeWeightDef(ToInputDef(s.bias)),
                        std::vector<int64_t>(s.strides.begin(), s.strides.end()),
                        std::vector<int64_t>(s.pads.begin(), s.pads.end()),
                        std::vector<int64_t>(s.dilations.begin(), s.dilations.end()),
                        grp, std::string(s.auto_pad),
                        std::vector<int64_t>(s.output_shape.begin(), s.output_shape.end()));
}

// ---------------------------------------------------------------------------
// BuildConvQDQFn<AType, WType> — QDQ Conv builder returning GetTestQDQModelFn<AType>.
//
// Used by conv_accuracy_test.cc with TestQDQModelAccuracy so that the
// output is compared against a float reference using relative (0.4%) tolerance
// rather than the absolute thresholds needed by RunQnnModelTest.
//
// Per-channel bias fix: each channel i uses bias_scale[i] = in_scale * w_scales[i]
// instead of the single w_scales[0] used in BuildConvModelFn. This matches the
// integration test's BuildQDQPerChannelConvTestCase and eliminates the ~8.0f
// absolute error caused by QNN EP's per-channel requantization of a mis-scaled bias.
// ---------------------------------------------------------------------------

template <typename AType, typename WType>
inline GetTestQDQModelFn<AType> BuildConvQDQFn(const ConvSpec& s) {
  const std::string op_type(s.op_type);
  const std::string auto_pad(s.auto_pad);
  // Use range-based TestInputDefs (not pre-materialized) so that the lambda uses
  // MaterializeWithBuilderRng, which generates random data via builder's seed=2345
  // engine — matching the F32 reference model's MakeTestInput call sequence exactly.
  const TestInputDef<float> in_def = ToInputDef(s.input);
  const TestInputDef<float> w_def  = ToInputDef(s.weights);
  const TestInputDef<float> b_def  = ToInputDef(s.bias);
  const std::vector<int64_t> strides(s.strides.begin(), s.strides.end());
  const std::vector<int64_t> pads(s.pads.begin(), s.pads.end());
  const std::vector<int64_t> dilations(s.dilations.begin(), s.dilations.end());
  const std::vector<int64_t> output_shape(s.output_shape.begin(), s.output_shape.end());
  const int64_t group = s.group;
  const float bsm = s.bias_scale_multiplier;
  const int64_t wqa = s.weight_quant_axis;
  const bool contrib = s.use_contrib_qdq;

  return [op_type, auto_pad, in_def, w_def, b_def, strides, pads, dilations,
          output_shape, group, bsm, wqa, contrib]
         (ModelTestBuilder& builder, std::vector<QuantParams<AType>>& output_qparams) {
    MakeTestInput<float>(builder, "input", MaterializeWeightDef(in_def));
    const auto in_qp = GetTestInputQuantParams<AType>(in_def);
    const std::string in_qdq = AddQDQNodePair<AType>(builder, "qdq_input", "input",
                                                      in_qp.scale, in_qp.zero_point, contrib);
    std::vector<std::string> conv_inputs{in_qdq};

    if constexpr (std::is_same_v<WType, uint8_t>) {
      // Per-tensor U8/U8 or U16/U8
      MakeTestInput<float>(builder, "weights", MaterializeWeightDef(w_def));
      const auto w_qp = GetTestInputQuantParams<uint8_t>(w_def);
      conv_inputs.push_back(AddQDQNodePair<uint8_t>(builder, "qdq_weights", "weights",
                                                     w_qp.scale, w_qp.zero_point, contrib));
      if (!b_def.GetShape().empty())
        conv_inputs.push_back(MakeTestQDQBiasInput(builder, "bias", MaterializeWeightDef(b_def),
                                                    in_qp.scale * w_qp.scale * bsm, contrib));
    } else if constexpr (std::is_same_v<WType, uint16_t>) {
      // Per-tensor U16/U16 (no bias, no contrib in opset-21 standard)
      MakeTestInput<float>(builder, "weights", MaterializeWeightDef(w_def));
      const auto w_qp = GetTestInputQuantParams<uint16_t>(w_def);
      conv_inputs.push_back(AddQDQNodePair<uint16_t>(builder, "qdq_weights", "weights",
                                                      w_qp.scale, w_qp.zero_point));
    } else if constexpr (std::is_same_v<WType, int8_t>) {
      // Per-channel S8 weight (U8/S8, U16/S8, or S8/S8 fusion)
      std::vector<float> w_scales;
      SetupPerChannelS8Weight(builder, MaterializeWeightDef(w_def),
                              wqa, contrib, w_scales);
      conv_inputs.push_back("weights_dq");
      if (!b_def.GetShape().empty()) {
        const int64_t n_ch = static_cast<int64_t>(w_scales.size());
        const auto b_rng = MaterializeWeightDef(b_def);
        const auto& bias_data = b_rng.GetRawData();
        std::vector<float> b_scales(n_ch);
        std::vector<int32_t> b_zps(n_ch, 0);
        std::vector<int32_t> b_quant(n_ch);
        for (int64_t i = 0; i < n_ch; ++i) {
          b_scales[i] = in_qp.scale * w_scales[i] * bsm;
          b_quant[i] = static_cast<int32_t>(std::round(bias_data[i] / b_scales[i]));
        }
        builder.MakeInitializer<int32_t>("bias_quant", {n_ch}, b_quant);
        std::vector<ONNX_NAMESPACE::AttributeProto> b_dq_attrs;
        b_dq_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));
        builder.AddDequantizeLinearNode("BiasDQ", "bias_quant", b_scales, b_zps,
                                        "bias_dq", b_dq_attrs, contrib);
        conv_inputs.push_back("bias_dq");
      }
    } else if constexpr (std::is_same_v<WType, Int4x2>) {
      // Per-channel S4 weight (U8/S4, U16/S4, opset 21)
      const int64_t quant_axis = (wqa >= 0) ? wqa
                                             : wqa + static_cast<int64_t>(w_def.GetShape().size());
      std::vector<float> w_scales;
      SetupPerChannelS4Weight(builder, MaterializeWeightDef(w_def),
                              quant_axis, wqa, w_scales);
      conv_inputs.push_back("weights_dq");
      if (!b_def.GetShape().empty()) {
        const int64_t n_ch = static_cast<int64_t>(w_scales.size());
        const auto b_rng = MaterializeWeightDef(b_def);
        const auto& bias_data = b_rng.GetRawData();
        std::vector<float> b_scales(n_ch);
        std::vector<int32_t> b_zps(n_ch, 0);
        std::vector<int32_t> b_quant(n_ch);
        for (int64_t i = 0; i < n_ch; ++i) {
          b_scales[i] = in_qp.scale * w_scales[i];
          b_quant[i] = static_cast<int32_t>(std::round(bias_data[i] / b_scales[i]));
        }
        builder.MakeInitializer<int32_t>("bias_quant", {n_ch}, b_quant);
        std::vector<ONNX_NAMESPACE::AttributeProto> b_dq_attrs;
        b_dq_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));
        builder.AddDequantizeLinearNode("BiasDQ", "bias_quant", b_scales, b_zps, "bias_dq", b_dq_attrs);
        conv_inputs.push_back("bias_dq");
      }
    } else if constexpr (std::is_same_v<WType, int16_t>) {
      // Per-channel S16 weight (U16/S16, contrib)
      std::vector<float> w_scales;
      SetupPerChannelS16Weight(builder, MaterializeWeightDef(w_def),
                               wqa, w_scales);
      conv_inputs.push_back("weights_dq");
      if (!b_def.GetShape().empty()) {
        const int64_t n_ch = static_cast<int64_t>(w_scales.size());
        const auto b_rng = MaterializeWeightDef(b_def);
        const auto& bias_data = b_rng.GetRawData();
        std::vector<float> b_scales(n_ch);
        std::vector<int32_t> b_zps(n_ch, 0);
        std::vector<int32_t> b_quant(n_ch);
        for (int64_t i = 0; i < n_ch; ++i) {
          b_scales[i] = in_qp.scale * w_scales[i] * bsm;
          b_quant[i] = static_cast<int32_t>(std::round(bias_data[i] / b_scales[i]));
        }
        builder.MakeInitializer<int32_t>("bias_quant", {n_ch}, b_quant);
        std::vector<ONNX_NAMESPACE::AttributeProto> b_dq_attrs;
        b_dq_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));
        builder.AddDequantizeLinearNode("BiasDQ", "bias_quant", b_scales, b_zps,
                                        "bias_dq", b_dq_attrs, /*use_contrib=*/true);
        conv_inputs.push_back("bias_dq");
      }
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    AddConvNodeAttrs(builder, attrs, auto_pad, group, strides, pads, dilations, output_shape);
    builder.AddNode("Conv", op_type, conv_inputs, {"Y"}, kOnnxDomain, attrs);

    // Output Q/DQ: scales come from TestQDQModelAccuracy's f32-output-range computation.
    AddQDQNodePairWithOutputAsGraphOutput<AType>(builder, "qdq_out", "Y",
                                                  output_qparams[0].scale,
                                                  output_qparams[0].zero_point, contrib);
  };
}

// ---------------------------------------------------------------------------
// Fusion QDQ builders — return GetTestQDQModelFn<AType> for TestQDQModelAccuracy.
// Shapes/data identical to the GetTestModelFn counterparts above so the
// f32 reference and QDQ model see the same inputs.
// ---------------------------------------------------------------------------

inline GetTestQDQModelFn<uint8_t> BuildConvU8U8S32_ReluQDQFn() {
  const TestInputDef<float> input_def({1, 2, 4, 4}, false,
                                      GetFloatDataInRange(0.0f, 1.0f, 32));
  const TestInputDef<float> weights_def({3, 2, 2, 2}, true,
                                        GetFloatDataInRange(-1.0f, 5.0f, 24));
  const TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));
  return [input_def, weights_def, bias_def]
         (ModelTestBuilder& builder, std::vector<QuantParams<uint8_t>>& output_qparams) {
    MakeTestInput<float>(builder, "input", input_def);
    const auto in_qp = GetTestInputQuantParams<uint8_t>(input_def);
    const std::string in_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_input", "input",
                                                        in_qp.scale, in_qp.zero_point);
    MakeTestInput<float>(builder, "weights", weights_def);
    const auto w_qp = GetTestInputQuantParams<uint8_t>(weights_def);
    const std::string w_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_weights", "weights",
                                                       w_qp.scale, w_qp.zero_point);
    const std::string bias_in = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                      in_qp.scale * w_qp.scale);
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    builder.AddNode("Conv", "Conv", {in_qdq, w_qdq, bias_in}, {"Y"}, kOnnxDomain, attrs);
    builder.AddNode("relu_node", "Relu", {"Y"}, {"relu_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "relu_out",
                                                   output_qparams[0].scale,
                                                   output_qparams[0].zero_point);
  };
}

inline GetTestQDQModelFn<uint8_t> BuildConvU8U8S32_RedundantClipQDQFn() {
  const TestInputDef<float> input_def({1, 2, 4, 4}, false,
                                      GetFloatDataInRange(0.0f, 1.0f, 32));
  const TestInputDef<float> weights_def({3, 2, 2, 2}, true,
                                        GetFloatDataInRange(-1.0f, 5.0f, 24));
  const TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));
  return [input_def, weights_def, bias_def]
         (ModelTestBuilder& builder, std::vector<QuantParams<uint8_t>>& output_qparams) {
    MakeTestInput<float>(builder, "input", input_def);
    const auto in_qp = GetTestInputQuantParams<uint8_t>(input_def);
    const std::string in_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_input", "input",
                                                        in_qp.scale, in_qp.zero_point,
                                                        /*use_contrib_qdq=*/true);
    MakeTestInput<float>(builder, "weights", weights_def);
    const auto w_qp = GetTestInputQuantParams<uint8_t>(weights_def);
    const std::string w_qdq = AddQDQNodePair<uint8_t>(builder, "qdq_weights", "weights",
                                                       w_qp.scale, w_qp.zero_point,
                                                       /*use_contrib_qdq=*/true);
    const std::string bias_in = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                      in_qp.scale * w_qp.scale,
                                                      /*use_contrib_qdq=*/true);
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    conv_attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    conv_attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    conv_attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    builder.AddNode("qdq_clip_conv", "Conv", {in_qdq, w_qdq, bias_in}, {"conv_out"},
                    kOnnxDomain, conv_attrs);
    builder.MakeScalarInitializer<float>("clip_min", -2.0f);
    builder.MakeScalarInitializer<float>("clip_max", 2.0f);
    builder.AddNode("qdq_clip_act", "Clip", {"conv_out", "clip_min", "clip_max"}, {"clip_out"});
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "clip_out",
                                                   output_qparams[0].scale,
                                                   output_qparams[0].zero_point,
                                                   /*use_contrib_qdq=*/true);
  };
}

inline GetTestQDQModelFn<int8_t> BuildConvS8S8S32_PerChannel_ReluQDQFn() {
  const TestInputDef<float> input_def({1, 2, 4, 4}, false,
                                      GetFloatDataInRange(0.0f, 1.0f, 32));
  const TestInputDef<float> weights_def({3, 2, 2, 2}, true,
                                        GetFloatDataInRange(-1.0f, 5.0f, 24));
  const TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));
  return [input_def, weights_def, bias_def]
         (ModelTestBuilder& builder, std::vector<QuantParams<int8_t>>& output_qparams) {
    MakeTestInput<float>(builder, "input", input_def);
    const auto in_qp = GetTestInputQuantParams<int8_t>(input_def);
    const std::string in_qdq = AddQDQNodePair<int8_t>(builder, "qdq_input", "input",
                                                       in_qp.scale, in_qp.zero_point);
    std::vector<float> w_scales;
    SetupPerChannelS8Weight(builder, weights_def, /*axis=*/0, /*use_contrib=*/false, w_scales);
    // Per-channel bias: scale[i] = in_scale * w_scales[i]
    constexpr int64_t n_ch = 3;
    const auto b_mat = MaterializeWeightDef(bias_def);
    const auto& bias_data = b_mat.GetRawData();
    std::vector<float> b_scales(n_ch);
    std::vector<int32_t> b_zps(n_ch, 0);
    std::vector<int32_t> b_quant(n_ch);
    for (int64_t i = 0; i < n_ch; ++i) {
      b_scales[i] = in_qp.scale * w_scales[i];
      b_quant[i] = static_cast<int32_t>(std::round(bias_data[i] / b_scales[i]));
    }
    builder.MakeInitializer<int32_t>("bias_quant", {n_ch}, b_quant);
    std::vector<ONNX_NAMESPACE::AttributeProto> b_dq_attrs;
    b_dq_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));
    builder.AddDequantizeLinearNode("BiasDQ", "bias_quant", b_scales, b_zps,
                                    "bias_dq", b_dq_attrs);
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    builder.AddNode("Conv", "Conv", {in_qdq, "weights_dq", "bias_dq"}, {"Y"},
                    kOnnxDomain, attrs);
    builder.AddNode("relu_node", "Relu", {"Y"}, {"relu_out"});
    AddQDQNodePairWithOutputAsGraphOutput<int8_t>(builder, "qdq_out", "relu_out",
                                                  output_qparams[0].scale,
                                                  output_qparams[0].zero_point);
  };
}

// F32 reference builders for fusion (no Q/DQ, same shapes/data as QDQ counterparts).
inline GetTestModelFn BuildConvFusionF32ReferenceFn(const ConvFusionSpec& s) {
  const TestInputDef<float> input_def({1, 2, 4, 4}, false,
                                      GetFloatDataInRange(0.0f, 1.0f, 32));
  const TestInputDef<float> weights_def({3, 2, 2, 2}, true,
                                        GetFloatDataInRange(-1.0f, 5.0f, 24));
  const TestInputDef<float> bias_def({3}, true, GetFloatDataInRange(-1.0f, 1.0f, 3));
  const bool has_clip = (s.fusion_type == ConvFusionType::ClipRedundant);
  return [input_def, weights_def, bias_def, has_clip](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "input", input_def);
    MakeTestInput<float>(builder, "weights", weights_def);
    MakeTestInput<float>(builder, "bias", bias_def);
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    attrs.push_back(builder.MakeIntsAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
    attrs.push_back(builder.MakeIntsAttribute("strides", std::vector<int64_t>{1, 1}));
    attrs.push_back(builder.MakeIntsAttribute("dilations", std::vector<int64_t>{1, 1}));
    builder.AddNode("f32_conv", "Conv", {"input", "weights", "bias"}, {"conv_out"},
                    kOnnxDomain, attrs);
    if (has_clip) {
      builder.MakeScalarInitializer<float>("clip_min", -2.0f);
      builder.MakeScalarInitializer<float>("clip_max", 2.0f);
      builder.AddNode("f32_act", "Clip", {"conv_out", "clip_min", "clip_max"}, {"output"});
    } else {
      builder.AddNode("f32_act", "Relu", {"conv_out"}, {"output"});
    }
    builder.MakeOutput("output");
  };
}

// QDQ dispatcher for fusion tests.
template <typename AType>
inline GetTestQDQModelFn<AType> BuildConvFusionQDQFn(const ConvFusionSpec& s) {
  if constexpr (std::is_same_v<AType, uint8_t>) {
    if (s.fusion_type == ConvFusionType::Relu)         return BuildConvU8U8S32_ReluQDQFn();
    if (s.fusion_type == ConvFusionType::ClipRedundant) return BuildConvU8U8S32_RedundantClipQDQFn();
  }
  if constexpr (std::is_same_v<AType, int8_t>) {
    return BuildConvS8S8S32_PerChannel_ReluQDQFn();
  }
  return [](ModelTestBuilder&, std::vector<QuantParams<AType>>&) {
    FAIL() << "BuildConvFusionQDQFn: unhandled ConvFusionSpec";
  };
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
