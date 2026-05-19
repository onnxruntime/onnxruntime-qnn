// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cstring>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class LayerNormalizationOpBuilder : public BaseOpBuilder {
 public:
  LayerNormalizationOpBuilder() : BaseOpBuilder("LayerNormalizationOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LayerNormalizationOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  // True iff `shape` (right-aligned to input_rank) has any non-1 dim at an axis < ln_axis.
  // Such dims fall outside QNN/ONNX LayerNorm's normalized axes [ln_axis, input_rank), so the
  // tensor cannot be fed directly to LayerNorm and must be applied via an outer Mul/Add instead.
  static bool HasNonOneDimBeforeAxis(const std::vector<uint32_t>& shape,
                                     size_t input_rank,
                                     size_t ln_axis) {
    const size_t prefix = input_rank - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
      const size_t aligned_axis = prefix + i;
      if (aligned_axis < ln_axis && shape[i] != 1) {
        return true;
      }
    }
    return false;
  }

  Ort::Status BuildDecomposedLayerNorm(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const std::vector<std::string>& input_names,
                                       std::vector<std::string>&& param_tensor_names,
                                       const std::vector<uint32_t>& x_shape,
                                       Qnn_DataType_t x_qnn_data_type,
                                       size_t ln_axis,
                                       bool externalize_scale,
                                       bool externalize_bias,
                                       bool do_op_validation,
                                       const Ort::Logger& logger) const ORT_MUST_USE_RESULT;
};

Ort::Status LayerNormalizationOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger) const {
  // Also check output type is float for CPU.
  const auto& outputs = node_unit.Outputs();
  RETURN_IF(outputs.size() > 1, "QNN LayerNorm only support 1 output.");

  // QNN Op validation can also do the same work, but the message is not so clear.
  // Explicit check and provide clear message here
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    std::vector<uint32_t> input_shape;
    const auto& inputs = node_unit.Inputs();
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape of input 0");
    const size_t input_rank = input_shape.size();
    int32_t default_axis = -1;
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
    RETURN_IF(static_cast<size_t>(default_axis) != input_rank - 1, "QNN LayerNorm for HTP only support axis with last input dimension");
  }

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Ort::Status LayerNormalizationOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger,
                                                       std::vector<std::string>& input_names,
                                                       bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  const auto input_count = inputs.size();
  constexpr size_t X_IDX = 0;
  constexpr size_t SCALE_IDX = 1;
  constexpr size_t BIAS_IDX = 2;

  // Input[0] (X, required)
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[X_IDX], logger, input_names));

  // Input[1] (scale, required)
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[SCALE_IDX], logger, input_names));

  // Input[2] (bias, optional)
  const bool has_bias_input = input_count > BIAS_IDX && inputs[BIAS_IDX].Exists();
  if (has_bias_input) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[BIAS_IDX], logger, input_names));
  }

#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 17 && QNN_API_VERSION_MINOR <= 20
  if (!has_bias_input && IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    // Bias is implicit. QNN SDK 2.24 to 2.27 (QNN API version 2.17 to 2.20) has a validation bug for
    // implicit bias inputs, so provide an explicit bias of all 0 (quantized int32).
    TensorInfo x_input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[X_IDX], x_input_info));

    TensorInfo scale_input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[SCALE_IDX], scale_input_info));

    if (x_input_info.quant_param.IsPerTensor(/*include_bw*/ true) && scale_input_info.quant_param.IsQuantized()) {
      const std::string bias_name = qnn::utils::UniqueNameGenerator().New(node_unit, "_implicit_bias");
      std::vector<uint32_t> bias_shape = scale_input_info.shape;
      RETURN_IF_ERROR(AddZeroBiasInput(qnn_model_wrapper, x_input_info.quant_param, scale_input_info.quant_param,
                                       std::move(bias_shape), bias_name, logger, input_names));
    }
  }
#endif

  return Ort::Status();
}

Ort::Status LayerNormalizationOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                     const OrtNodeUnit& node_unit,
                                                                     std::vector<std::string>&& input_names,
                                                                     const Ort::Logger& logger,
                                                                     bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 according to ONNX spec.
  Qnn_Scalar_t epsilon_param = QNN_SCALAR_INIT;
  epsilon_param.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_param.floatValue = epsilon;
  QnnParamWrapper epsilon_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_LAYER_NORM_PARAM_EPSILON,
                                        epsilon_param);
  param_tensor_names.push_back(epsilon_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(epsilon_param_wrapper));

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();
  int32_t default_axis = -1;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
  size_t axes_rank = input_rank - static_cast<size_t>(default_axis);
  std::vector<uint32_t> axes(axes_rank, 0);
  std::vector<uint32_t> axes_shape{SafeInt<uint32_t>(axes_rank)};
  axes[0] = static_cast<uint32_t>(default_axis);
  for (size_t i = 1; i < axes.size(); ++i) {
    axes[i] = axes[i - 1] + 1;
  }

  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), QNN_OP_LAYER_NORM_PARAM_AXES,
                             std::move(axes_shape), std::move(axes));
  param_tensor_names.push_back(axes_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axes_param));

  // ONNX LayerNormalization requires scale/B to broadcast to X.shape[axis:]. If the user-provided
  // scale or bias has non-1 dims at axes outside [axis, rank) (e.g. they were folded in by an
  // outer Mul/Add that broadcasts across the batch), QNN can't consume them as LN scale/B.
  // Decide which side(s) to externalize:
  //   - scale misaligned -> externalize scale; bias must also be externalized whenever it exists,
  //     because LN computes (norm*scale + bias). Pulling scale out as a trailing Mul would turn a
  //     legal-shape bias inside LN into bias*scale_external, which both changes the math and
  //     re-introduces the broadcast problem. So scale-out forces bias-out.
  //   - scale legal, bias misaligned -> keep LN(scale), externalize only the Add.
  const auto& inputs = node_unit.Inputs();
  const bool has_bias_input = inputs.size() > 2 && inputs[2].Exists();
  TensorInfo scale_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], scale_info));
  const bool scale_misaligned =
      HasNonOneDimBeforeAxis(scale_info.shape, input_rank, static_cast<size_t>(default_axis));

  bool bias_misaligned = false;
  if (has_bias_input) {
    TensorInfo bias_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));
    bias_misaligned =
        HasNonOneDimBeforeAxis(bias_info.shape, input_rank, static_cast<size_t>(default_axis));
  }

  const bool externalize_scale = scale_misaligned;
  const bool externalize_bias = has_bias_input && (bias_misaligned || scale_misaligned);

  if (externalize_scale || externalize_bias) {
    TensorInfo x_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], x_info));
    return BuildDecomposedLayerNorm(qnn_model_wrapper,
                                    node_unit,
                                    input_names,
                                    std::move(param_tensor_names),
                                    input_shape,
                                    x_info.qnn_data_type,
                                    static_cast<size_t>(default_axis),
                                    externalize_scale,
                                    externalize_bias,
                                    do_op_validation,
                                    logger);
  }

  RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                 std::move(input_names),
                                 std::move(param_tensor_names),
                                 logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Ort::Status();
}

Ort::Status LayerNormalizationOpBuilder::BuildDecomposedLayerNorm(QnnModelWrapper& qnn_model_wrapper,
                                                                  const OrtNodeUnit& node_unit,
                                                                  const std::vector<std::string>& input_names,
                                                                  std::vector<std::string>&& param_tensor_names,
                                                                  const std::vector<uint32_t>& x_shape,
                                                                  Qnn_DataType_t x_qnn_data_type,
                                                                  size_t ln_axis,
                                                                  bool externalize_scale,
                                                                  bool externalize_bias,
                                                                  bool do_op_validation,
                                                                  const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  assert(externalize_scale || externalize_bias);
  // externalize_bias implies a trailing Add fed by input_names[2], so the user must have provided
  // a bias input.
  assert(!externalize_bias || input_names.size() > 2);

  const auto& outputs = node_unit.Outputs();
  const std::string& final_output_name = outputs[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  const Qnn_TensorType_t final_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  TensorInfo final_output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], final_output_info));

  // The LN node's scale input: either the user's (when scale stays inside) or a constant 1.0
  // tensor of shape X.shape[axis:] that we synthesize here.
  std::string ln_scale_name;
  if (!externalize_scale) {
    ln_scale_name = input_names[1];
  } else {
    // Match the synthesized ones tensor to the user-provided scale's dtype + quant params so the
    // LN op sees the type it expects in slot 1. Encoding 1.0 in the user's quant scheme is what
    // makes this an identity scale at runtime.
    TensorInfo user_scale_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], user_scale_info));

    std::vector<uint32_t> norm_shape(x_shape.begin() + ln_axis, x_shape.end());
    size_t num_elems = 1;
    for (uint32_t d : norm_shape) {
      num_elems *= static_cast<size_t>(d);
    }

    const Qnn_DataType_t scale_dtype = user_scale_info.qnn_data_type;
    std::vector<uint8_t> const_buf;

    if (user_scale_info.quant_param.IsQuantized()) {
      // Per-tensor or per-channel quantized: quantize 1.0 using the user scale's params and fill.
      RETURN_IF_NOT(user_scale_info.quant_param.IsPerTensor(/*include_bw*/ true),
                    "LayerNorm scale decomposition: per-channel quantized scale is not supported.");
      const float quant_scale = user_scale_info.quant_param.Get().scaleOffsetEncoding.scale;
      const int32_t quant_offset = user_scale_info.quant_param.Get().scaleOffsetEncoding.offset;
      int quant_one = 0;
      RETURN_IF_ERROR(utils::Quantize(1.0, quant_scale, quant_offset, scale_dtype, quant_one));
      switch (scale_dtype) {
        case QNN_DATATYPE_SFIXED_POINT_8: {
          const_buf.resize(num_elems * sizeof(int8_t));
          std::fill(reinterpret_cast<int8_t*>(const_buf.data()),
                    reinterpret_cast<int8_t*>(const_buf.data()) + num_elems,
                    static_cast<int8_t>(quant_one));
          break;
        }
        case QNN_DATATYPE_UFIXED_POINT_8: {
          const_buf.resize(num_elems * sizeof(uint8_t));
          std::fill(const_buf.begin(), const_buf.end(), static_cast<uint8_t>(quant_one));
          break;
        }
        case QNN_DATATYPE_SFIXED_POINT_16: {
          const_buf.resize(num_elems * sizeof(int16_t));
          std::fill(reinterpret_cast<int16_t*>(const_buf.data()),
                    reinterpret_cast<int16_t*>(const_buf.data()) + num_elems,
                    static_cast<int16_t>(quant_one));
          break;
        }
        case QNN_DATATYPE_UFIXED_POINT_16: {
          const_buf.resize(num_elems * sizeof(uint16_t));
          std::fill(reinterpret_cast<uint16_t*>(const_buf.data()),
                    reinterpret_cast<uint16_t*>(const_buf.data()) + num_elems,
                    static_cast<uint16_t>(quant_one));
          break;
        }
        default:
          return MAKE_EP_FAIL("LayerNorm scale decomposition: unsupported quantized scale dtype.");
      }
    } else {
      switch (scale_dtype) {
        case QNN_DATATYPE_FLOAT_32: {
          const_buf.resize(num_elems * sizeof(float));
          float* p = reinterpret_cast<float*>(const_buf.data());
          std::fill(p, p + num_elems, 1.0f);
          break;
        }
        case QNN_DATATYPE_FLOAT_16: {
          const_buf.resize(num_elems * sizeof(Ort::Float16_t));
          Ort::Float16_t* p = reinterpret_cast<Ort::Float16_t*>(const_buf.data());
          std::fill(p, p + num_elems, static_cast<Ort::Float16_t>(1.0f));
          break;
        }
        default:
          return MAKE_EP_FAIL("LayerNorm scale decomposition: unsupported float scale dtype.");
      }
    }

    ln_scale_name = utils::UniqueNameGenerator().New(node_unit, "_ln_scale_one");
    QnnTensorWrapper scale_one_tensor(ln_scale_name,
                                      QNN_TENSOR_TYPE_STATIC,
                                      scale_dtype,
                                      user_scale_info.quant_param.Copy(),
                                      std::move(norm_shape),
                                      std::move(const_buf));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_one_tensor)),
                  "Failed to add LN identity scale tensor.");
  }

  // The Add (if present) is always last and writes to final_output_name. If only Mul is present,
  // Mul writes to final_output_name. Earlier intermediates (LN, optional Mul-before-Add) are NATIVE
  // tensors of x_shape and x dtype.
  // Intermediate tensors need valid quant encoding when x_qnn_data_type is a quantized type;
  // an empty QnnQuantParamsWrapper would fail QNN validation and at runtime would dequantize as
  // (val - 0) * 0 = 0. Reuse the final output's quant params (Copy() so the move at the end of
  // the function still works for the Add output). Range may be coarser than ideal for the LN
  // output when scale/bias shift it, but it's well-formed.
  const std::string ln_out_name = utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_out");
  QnnTensorWrapper ln_out_tensor(ln_out_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  x_qnn_data_type,
                                  final_output_info.quant_param.Copy(),
                                  std::vector<uint32_t>(x_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(ln_out_tensor)),
                "Failed to add decomposed LN intermediate output tensor.");

  std::vector<std::string> ln_inputs = {input_names[0], ln_scale_name};
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_LAYER_NORM,
                                                std::move(ln_inputs),
                                                {ln_out_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add decomposed LayerNorm node.");

  std::string current = ln_out_name;

  if (externalize_scale) {
    const std::string mul_out_name =
        externalize_bias ? utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_mul_out")
                         : final_output_name;
    QnnTensorWrapper mul_out_tensor(
        mul_out_name,
        externalize_bias ? QNN_TENSOR_TYPE_NATIVE : final_tensor_type,
        externalize_bias ? x_qnn_data_type : final_output_info.qnn_data_type,
        externalize_bias ? final_output_info.quant_param.Copy() : std::move(final_output_info.quant_param),
        externalize_bias ? std::vector<uint32_t>(x_shape) : std::move(final_output_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul_out_tensor)),
                  "Failed to add decomposed Mul output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_mul"),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                  {current, input_names[1]},
                                                  {mul_out_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add decomposed Mul node.");
    current = mul_out_name;
  }

  if (externalize_bias) {
    QnnTensorWrapper add_out_tensor(final_output_name,
                                    final_tensor_type,
                                    final_output_info.qnn_data_type,
                                    std::move(final_output_info.quant_param),
                                    std::move(final_output_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(add_out_tensor)),
                  "Failed to add decomposed Add output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_add"),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_ADD,
                                                  {current, input_names[2]},
                                                  {final_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add decomposed Add node.");
  }

  return Ort::Status();
}

void CreateLayerNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<LayerNormalizationOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
