// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/dq_layernorm_fusion.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kOpLayerNormalization[] = "LayerNormalization";

bool IsFixedPoint8(Qnn_DataType_t dtype) {
  return dtype == QNN_DATATYPE_SFIXED_POINT_8 || dtype == QNN_DATATYPE_UFIXED_POINT_8;
}

// Requantizes a per-tensor-quantized 8-bit static initializer from SFIXED_POINT_8 to
// UFIXED_POINT_8 (or vice versa) via an exact +/-128 zero-point shift. For an 8-bit per-tensor
// quantized tensor, real = scale * (q - offset); shifting both q and offset by the same constant
// leaves `real` unchanged, so relabeling SFIXED_POINT_8 <-> UFIXED_POINT_8 this way is lossless
// (no rounding, no clipping), unlike a generic dequantize-then-requantize round trip.
Ort::Status FlipFixedPoint8Sign(const QnnModelWrapper& qmw, const TensorInfo& info,
                                std::vector<uint8_t>& out_bytes, int32_t& out_zero_point, float& out_scale) {
  RETURN_IF_NOT(info.is_initializer && info.initializer_tensor != nullptr,
                "DQLayerNormFusion: tensor must be a static initializer to resign.");
  RETURN_IF_NOT(info.quant_param.IsPerTensor(/*include_bw*/ true),
                "DQLayerNormFusion: tensor must be per-tensor quantized to resign.");
  RETURN_IF_NOT(IsFixedPoint8(info.qnn_data_type),
                "DQLayerNormFusion: only 8-bit fixed-point tensors can be resigned.");

  float scale = 0.0f;
  int32_t zero_point = 0;
  RETURN_IF_ERROR(info.quant_param.GetPerTensorScaleOffset(scale, zero_point));

  std::vector<uint8_t> raw_bytes;
  RETURN_IF_ERROR(qmw.UnpackInitializerData(info.initializer_tensor, raw_bytes));

  const bool src_is_signed = (info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8);
  const int32_t shift = src_is_signed ? 128 : -128;

  out_bytes.assign(raw_bytes.size(), 0);
  for (size_t i = 0; i < raw_bytes.size(); ++i) {
    const int32_t src_val = src_is_signed
                                ? static_cast<int32_t>(reinterpret_cast<const int8_t*>(raw_bytes.data())[i])
                                : static_cast<int32_t>(raw_bytes[i]);
    out_bytes[i] = static_cast<uint8_t>(src_val + shift);
  }
  out_zero_point = zero_point + shift;
  out_scale = scale;
  return Ort::Status();
}

}  // namespace

// ---------------------------------------------------------------------------
// TryFusion
// ---------------------------------------------------------------------------
std::unique_ptr<IQnnNodeGroup> DQLayerNormFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& layer_norm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(node_to_node_unit);
  ORT_UNUSED_PARAMETER(node_unit_to_qnn_node_group);

  auto reject = [&logger](std::string_view reason) -> std::unique_ptr<IQnnNodeGroup> {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                (std::string("DQLayerNormFusion rejected: ").append(reason)).c_str());
    return nullptr;
  };

  // Only intercept QDQ LayerNormalization groups; a standalone float LayerNorm never has this
  // dtype mismatch to begin with.
  if (layer_norm_node_unit.OpType() != kOpLayerNormalization ||
      layer_norm_node_unit.UnitType() != OrtNodeUnit::Type::QDQGroup) {
    return reject("not a QDQ LayerNormalization group");
  }

  // CPU/GPU EP don't have this HTP-specific validator restriction.
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return reject("not targeting an NPU backend");
  }

  const auto& outputs = layer_norm_node_unit.Outputs();
  if (outputs.size() != 1) {
    return reject("LayerNorm must have exactly one output");
  }

  const auto& inputs = layer_norm_node_unit.Inputs();
  if (inputs.size() < 2) {
    return reject("LayerNorm missing scale input");
  }

  std::vector<uint32_t> x_shape;
  if (!qnn_model_wrapper.GetOnnxShape(inputs[0].shape, x_shape) || x_shape.empty()) {
    return reject("cannot determine static rank of LayerNorm input X");
  }
  const size_t input_rank = x_shape.size();

  // QNN LayerNorm on HTP only supports normalization along the last axis. The default op-builder
  // path enforces this too, but since this fusion claims the whole NodeUnit before that path ever
  // runs, it must re-check the restriction itself.
  {
    OrtNodeAttrHelper attrs(layer_norm_node_unit);
    int64_t axis = attrs.Get("axis", static_cast<int64_t>(-1));
    if (axis < 0) {
      axis += static_cast<int64_t>(input_rank);
    }
    if (axis < 0 || static_cast<size_t>(axis) != input_rank - 1) {
      return reject("QNN LayerNorm on HTP only supports normalization along the last axis");
    }
  }

  TensorInfo x_info{};
  if (!qnn_model_wrapper.GetTensorInfo(inputs[0], x_info).IsOK()) {
    return reject("failed to get TensorInfo for X");
  }
  if (!IsFixedPoint8(x_info.qnn_data_type)) {
    return reject("X is not 8-bit fixed-point; sign-mismatch fixup does not apply");
  }

  TensorInfo scale_info{};
  if (!qnn_model_wrapper.GetTensorInfo(inputs[1], scale_info).IsOK()) {
    return reject("failed to get TensorInfo for scale");
  }

  // Only activate on an unsigned/signed 8-bit mismatch between X and scale -- QNN's HTP LayerNorm
  // has no supported config that mixes UFIXED_POINT_8 with SFIXED_POINT_8. Anything else (scale
  // already matches X's family, or is float/wider fixed-point) is already handled correctly by
  // the default op-builder path.
  if (!IsFixedPoint8(scale_info.qnn_data_type) || scale_info.qnn_data_type == x_info.qnn_data_type) {
    return reject("scale dtype does not need resigning");
  }
  if (!scale_info.is_initializer || scale_info.initializer_tensor == nullptr) {
    return reject("scale is not a static initializer");
  }
  if (!scale_info.quant_param.IsPerTensor(/*include_bw*/ true)) {
    return reject("scale quant params are not per-tensor");
  }

  const bool has_bias = inputs.size() > 2 && inputs[2].Exists();
  if (has_bias) {
    TensorInfo bias_info{};
    if (!qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info).IsOK()) {
      return reject("failed to get TensorInfo for bias");
    }
    // Bias is usually SFIXED_POINT_32, which QNN's LayerNorm already accepts regardless of X's
    // sign and this fusion leaves untouched. Only a mismatched 8-bit bias needs resigning too.
    if (IsFixedPoint8(bias_info.qnn_data_type) && bias_info.qnn_data_type != x_info.qnn_data_type) {
      if (!bias_info.is_initializer || bias_info.initializer_tensor == nullptr) {
        return reject("mismatched 8-bit bias is not a static initializer");
      }
      if (!bias_info.quant_param.IsPerTensor(/*include_bw*/ true)) {
        return reject("mismatched 8-bit bias quant params are not per-tensor");
      }
    }
  }

  // Validate the resigned path before committing to the fusion.
  auto fused = std::unique_ptr<DQLayerNormFusion>(new DQLayerNormFusion(layer_norm_node_unit));
  if (Ort::Status status = fused->CreateOrValidateOnQnn(qnn_model_wrapper, /*validate=*/true);
      !status.IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("DQLayerNormFusion rejected by QNN validate: " + status.GetErrorMessage()).c_str());
    return nullptr;
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "DQLayerNormFusion matched and validated");
  return fused;
}

// ---------------------------------------------------------------------------
// Constructor / IQnnNodeGroup plumbing
// ---------------------------------------------------------------------------
DQLayerNormFusion::DQLayerNormFusion(const OrtNodeUnit& node_unit)
    : node_units_{&node_unit}, node_unit_(&node_unit) {
}

Ort::Status DQLayerNormFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, /*validate=*/true);
}

Ort::Status DQLayerNormFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, /*validate=*/false);
}

gsl::span<const OrtNodeUnit* const> DQLayerNormFusion::GetNodeUnits() const {
  return gsl::make_span(node_units_);
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------
Ort::Status DQLayerNormFusion::CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate) const {
  const OrtNodeUnit& node_unit = *node_unit_;
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();
  const bool has_bias = inputs.size() > 2 && inputs[2].Exists();

  const OrtNodeUnitIODef& x_input = inputs[0];
  const OrtNodeUnitIODef& scale_input = inputs[1];
  const OrtNodeUnitIODef& final_output = outputs[0];

  std::vector<uint32_t> x_shape;
  RETURN_IF_NOT(qmw.GetOnnxShape(x_input.shape, x_shape) && !x_shape.empty(),
                "DQLayerNormFusion: cannot determine static rank of LayerNorm input X.");
  const size_t input_rank = x_shape.size();

  TensorInfo x_info{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(x_input, x_info));

  const std::string node_name = utils::UniqueNameGenerator().New(node_unit);

  QnnTensorWrapper x_tensor;
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(x_input, x_tensor));

  // Scale always gets resigned to match X's family -- TryFusion already checked the precondition.
  TensorInfo scale_info{};
  RETURN_IF_ERROR(qmw.GetTensorInfo(scale_input, scale_info));
  std::vector<uint8_t> scale_bytes;
  float scale_scale = 0.0f;
  int32_t scale_zero_point = 0;
  RETURN_IF_ERROR(FlipFixedPoint8Sign(qmw, scale_info, scale_bytes, scale_zero_point, scale_scale));
  const std::string scale_name = node_name + "_scale_resigned";
  QnnTensorWrapper scale_tensor(scale_name, QNN_TENSOR_TYPE_STATIC, x_info.qnn_data_type,
                                QnnQuantParamsWrapper::PerTensor(scale_scale, scale_zero_point),
                                std::vector<uint32_t>(scale_info.shape), std::move(scale_bytes));

  // Bias: pass through unchanged unless it's also 8-bit and mismatched with X, in which case
  // resign it the same way.
  std::string bias_name;
  QnnTensorWrapper bias_tensor;
  bool bias_is_new_tensor = false;
  if (has_bias) {
    const OrtNodeUnitIODef& bias_input = inputs[2];
    TensorInfo bias_info{};
    RETURN_IF_ERROR(qmw.GetTensorInfo(bias_input, bias_info));
    if (IsFixedPoint8(bias_info.qnn_data_type) && bias_info.qnn_data_type != x_info.qnn_data_type) {
      std::vector<uint8_t> bias_bytes;
      float bias_scale = 0.0f;
      int32_t bias_zero_point = 0;
      RETURN_IF_ERROR(FlipFixedPoint8Sign(qmw, bias_info, bias_bytes, bias_zero_point, bias_scale));
      bias_name = node_name + "_bias_resigned";
      bias_tensor = QnnTensorWrapper(bias_name, QNN_TENSOR_TYPE_STATIC, x_info.qnn_data_type,
                                     QnnQuantParamsWrapper::PerTensor(bias_scale, bias_zero_point),
                                     std::vector<uint32_t>(bias_info.shape), std::move(bias_bytes));
      bias_is_new_tensor = true;
    } else {
      bias_name = bias_input.name;
      RETURN_IF_ERROR(qmw.MakeTensorWrapper(bias_input, bias_tensor));
      bias_is_new_tensor = !qmw.IsQnnTensorWrapperExist(bias_name);
    }
  }

  QnnTensorWrapper output_tensor;
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(final_output, output_tensor));

  OrtNodeAttrHelper node_helper(node_unit);
  const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 per ONNX spec.
  Qnn_Scalar_t epsilon_scalar = QNN_SCALAR_INIT;
  epsilon_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_scalar.floatValue = epsilon;
  QnnParamWrapper epsilon_param(node_unit.Index(), node_name, QNN_OP_LAYER_NORM_PARAM_EPSILON, epsilon_scalar);

  // TryFusion already enforced axis == input_rank - 1 (last axis only).
  std::vector<uint32_t> axes_vec = {static_cast<uint32_t>(input_rank - 1)};
  std::vector<uint32_t> axes_shape{1u};
  QnnParamWrapper axes_param(node_unit.Index(), node_name, QNN_OP_LAYER_NORM_PARAM_AXES,
                             std::move(axes_shape), std::move(axes_vec));

  if (validate) {
    std::vector<Qnn_Tensor_t> qnn_inputs = {x_tensor.GetQnnTensor(), scale_tensor.GetQnnTensor()};
    if (has_bias) {
      qnn_inputs.push_back(bias_tensor.GetQnnTensor());
    }
    return qmw.ValidateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_LAYER_NORM,
                               std::move(qnn_inputs),
                               {output_tensor.GetQnnTensor()},
                               {epsilon_param.GetQnnParam(), axes_param.GetQnnParam()});
  }

  const std::string epsilon_param_name = epsilon_param.GetParamTensorName();
  RETURN_IF_NOT(qmw.AddParamWrapper(std::move(epsilon_param)), "Failed to add epsilon param.");
  const std::string axes_param_name = axes_param.GetParamTensorName();
  RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axes_param)), "Failed to add axes param.");

  if (!qmw.IsQnnTensorWrapperExist(x_input.name)) {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(x_tensor)), "Failed to add LN input tensor.");
  }
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(scale_tensor)), "Failed to add resigned LN scale tensor.");
  if (has_bias && bias_is_new_tensor) {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(bias_tensor)), "Failed to add LN bias tensor.");
  }
  if (!qmw.IsQnnTensorWrapperExist(final_output.name)) {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_tensor)), "Failed to add LN output tensor.");
  }

  std::vector<std::string> input_names = {x_input.name, scale_name};
  if (has_bias) {
    input_names.push_back(bias_name);
  }

  RETURN_IF_NOT(qmw.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_LAYER_NORM,
                                  std::move(input_names),
                                  {final_output.name},
                                  {epsilon_param_name, axes_param_name},
                                  /*do_op_validation=*/false),
                "Failed to create resigned LayerNorm node.");

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
