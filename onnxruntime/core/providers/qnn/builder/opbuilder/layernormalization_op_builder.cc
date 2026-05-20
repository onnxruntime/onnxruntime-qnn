// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cassert>
#include <cmath>
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

  const auto& outputs = node_unit.Outputs();
  const std::string& final_output_name = outputs[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  const Qnn_TensorType_t final_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  TensorInfo final_output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], final_output_info));

  QnnQuantParamsWrapper ln_intermediate_qp = final_output_info.quant_param.Copy();
  QnnQuantParamsWrapper mul_intermediate_qp = final_output_info.quant_param.Copy();

  // Quant params (when exists) for the LN intermediate (and the Mul-before-Add intermediate, when present).
  // The LN op produces normalized values bounded by sqrt(N-1) (where N is the number of elements
  // along the normalized axes), times the user scale if scale stays inside LN. Reusing
  // final_output_info.quant_param here clips whenever the LN-stage range is wider than the final
  // output's range — for instance when |user_scale| < 1 and bias is small, or in any path where
  // an externalized Mul is sandwiched in front of an Add that narrows the range.
  // Derive symmetric ranges per-stage instead, sized to the theoretical bound:
  //   ln_out_range    = sqrt(N-1) * (externalize_scale ? 1 : max|user_scale|)
  //   mul_out_range   = sqrt(N-1) * max|user_scale|       (only relevant when both are externalized)
  if (final_output_info.quant_param.IsPerTensor(/*include_bw*/ true)) {
    size_t norm_count = 1;
    for (size_t i = ln_axis; i < x_shape.size(); ++i) {
      norm_count *= static_cast<size_t>(x_shape[i]);
    }
    const float ln_max_abs_normalized =
        std::sqrt(std::max(static_cast<float>(norm_count) - 1.0f, 1.0f));

    float user_scale_max_abs = 1.0f;
    TensorInfo scale_info_for_range{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], scale_info_for_range));
    if (scale_info_for_range.quant_param.IsPerTensor(/*include_bw*/ true)) {
      float s_qmin = 0.0f;
      float s_qmax = 0.0f;
      RETURN_IF_ERROR(utils::GetQminQmax(scale_info_for_range.qnn_data_type, s_qmin, s_qmax));
      const float s_scale = scale_info_for_range.quant_param.Get().scaleOffsetEncoding.scale;
      const int32_t s_offset = scale_info_for_range.quant_param.Get().scaleOffsetEncoding.offset;
      const double s_min = utils::Dequantize(s_offset, s_scale, s_qmin);
      const double s_max = utils::Dequantize(s_offset, s_scale, s_qmax);
      user_scale_max_abs = static_cast<float>(std::max(std::abs(s_min), std::abs(s_max)));
    }

    auto compute_intermediate_qp = [&](float range_abs, QnnQuantParamsWrapper& out_qp) -> Ort::Status {
      const float k = std::max(range_abs, 1e-4f);
      float interm_scale = 0.0f;
      int32_t interm_offset = 0;
      RETURN_IF_ERROR(utils::GetQuantParams(-k, k, x_qnn_data_type, interm_scale, interm_offset,
                                            /*symmetric*/ false));
      out_qp = QnnQuantParamsWrapper(interm_scale, interm_offset);
      return Ort::Status();
    };

    const float ln_out_range_abs = externalize_scale
                                       ? ln_max_abs_normalized
                                       : ln_max_abs_normalized * user_scale_max_abs;
    const float mul_out_range_abs = ln_max_abs_normalized * user_scale_max_abs;
    RETURN_IF_ERROR(compute_intermediate_qp(ln_out_range_abs, ln_intermediate_qp));
    RETURN_IF_ERROR(compute_intermediate_qp(mul_out_range_abs, mul_intermediate_qp));
  }  // End of re-quantize LN quantparam

  std::string ln_scale_name;
  if (!externalize_scale) {
    ln_scale_name = input_names[1];
  } else {
    // Match the synthesized ones tensor to the user-provided scale's dtype + quant params so the
    // LN op sees the type it expects in slot 1. Encoding 1.0 in the user's quant scheme is what
    // makes this an identity scale at runtime.
    TensorInfo scale_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], scale_info));

    std::vector<uint32_t> norm_shape(x_shape.begin() + ln_axis, x_shape.end());
    size_t num_elems = 1;
    for (uint32_t d : norm_shape) {
      num_elems *= static_cast<size_t>(d);
    }

    const Qnn_DataType_t scale_dtype = scale_info.qnn_data_type;
    std::vector<uint8_t> const_buf;

    if (scale_info.quant_param.IsQuantized()) {
      // Per-tensor or per-channel quantized: quantize 1.0 using the user scale's params and fill.
      RETURN_IF_NOT(scale_info.quant_param.IsPerTensor(/*include_bw*/ true),
                    "LayerNorm scale decomposition: per-channel quantized scale is not supported.");
      const float quant_scale = scale_info.quant_param.Get().scaleOffsetEncoding.scale;
      const int32_t quant_offset = scale_info.quant_param.Get().scaleOffsetEncoding.offset;
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
                                      scale_info.quant_param.Copy(),
                                      std::move(norm_shape),
                                      std::move(const_buf));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_one_tensor)),
                  "Failed to add LN identity scale tensor.");
  }

  // The Add (if present) is always last and writes to final_output_name. If only Mul is present,
  // Mul writes to final_output_name. Earlier intermediates (LN, optional Mul-before-Add) are NATIVE
  // tensors of x_shape and x dtype, with the symmetric range derived above from X's input range.
  const std::string ln_out_name = utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_out");
  QnnTensorWrapper ln_out_tensor(ln_out_name,
                                 QNN_TENSOR_TYPE_NATIVE,
                                 x_qnn_data_type,
                                 ln_intermediate_qp.Copy(),
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
        externalize_bias ? mul_intermediate_qp.Copy() : std::move(final_output_info.quant_param),
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
    const QnnQuantParamsWrapper& add_lhs_qp =
        externalize_scale ? mul_intermediate_qp : ln_intermediate_qp;

    std::string bias_name = input_names[2];
    TensorInfo bias_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[2], bias_info));

    // QDQ pipelines feed LayerNorm's bias as int32 (matching ONNX LN's int32 beta convention).
    // QNN's LayerNorm op accepts that, but ELEMENT_WISE_ADD requires both operands to share a
    // dtype, re-quantize the static int32 bias to match LN output dtype.
    // ONNX requires LN input, scale and bias in same dtype, so when dtype mismatch, it must be a qdq graph.
    if (bias_info.qnn_data_type != x_qnn_data_type) {
      RETURN_IF_NOT(bias_info.is_initializer && bias_info.initializer_tensor != nullptr,
                    "LayerNorm decomposition: externalized bias must be a static initializer.");
      RETURN_IF_NOT(bias_info.quant_param.IsPerTensor(/*include_bw*/ true),
                    "LayerNorm decomposition: externalized bias must be per-tensor quantized.");
      RETURN_IF_NOT(add_lhs_qp.IsPerTensor(/*include_bw*/ true),
                    "LayerNorm decomposition: requantized bias requires per-tensor intermediate quant params.");

      const float src_scale = bias_info.quant_param.Get().scaleOffsetEncoding.scale;
      const int32_t src_offset = bias_info.quant_param.Get().scaleOffsetEncoding.offset;
      const float dst_scale = add_lhs_qp.Get().scaleOffsetEncoding.scale;
      const int32_t dst_offset = add_lhs_qp.Get().scaleOffsetEncoding.offset;

      std::vector<uint8_t> raw_bias;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, raw_bias));
      const size_t bias_elem_bytes = utils::GetElementSizeByType(bias_info.qnn_data_type);
      RETURN_IF_NOT(bias_elem_bytes > 0 && raw_bias.size() % bias_elem_bytes == 0,
                    "LayerNorm decomposition: bias raw data size is not a multiple of element size.");
      const size_t num_bias_elems = raw_bias.size() / bias_elem_bytes;

      auto load_q_value = [&](size_t i) -> int64_t {
        switch (bias_info.qnn_data_type) {
          case QNN_DATATYPE_SFIXED_POINT_32:
          case QNN_DATATYPE_INT_32:
            return static_cast<int64_t>(reinterpret_cast<const int32_t*>(raw_bias.data())[i]);
          case QNN_DATATYPE_SFIXED_POINT_16:
            return static_cast<int64_t>(reinterpret_cast<const int16_t*>(raw_bias.data())[i]);
          case QNN_DATATYPE_UFIXED_POINT_16:
            return static_cast<int64_t>(reinterpret_cast<const uint16_t*>(raw_bias.data())[i]);
          case QNN_DATATYPE_SFIXED_POINT_8:
            return static_cast<int64_t>(reinterpret_cast<const int8_t*>(raw_bias.data())[i]);
          case QNN_DATATYPE_UFIXED_POINT_8:
            return static_cast<int64_t>(reinterpret_cast<const uint8_t*>(raw_bias.data())[i]);
          default:
            return 0;
        }
      };

      std::vector<uint8_t> requant_bias(num_bias_elems * utils::GetElementSizeByType(x_qnn_data_type), 0);
      for (size_t i = 0; i < num_bias_elems; ++i) {
        const double dequant_val = utils::Dequantize(src_offset, src_scale, static_cast<double>(load_q_value(i)));
        int q = 0;
        RETURN_IF_ERROR(utils::Quantize(dequant_val, dst_scale, dst_offset, x_qnn_data_type, q));
        switch (x_qnn_data_type) {
          case QNN_DATATYPE_SFIXED_POINT_8:
            reinterpret_cast<int8_t*>(requant_bias.data())[i] = static_cast<int8_t>(q);
            break;
          case QNN_DATATYPE_UFIXED_POINT_8:
            reinterpret_cast<uint8_t*>(requant_bias.data())[i] = static_cast<uint8_t>(q);
            break;
          case QNN_DATATYPE_SFIXED_POINT_16:
            reinterpret_cast<int16_t*>(requant_bias.data())[i] = static_cast<int16_t>(q);
            break;
          case QNN_DATATYPE_UFIXED_POINT_16:
            reinterpret_cast<uint16_t*>(requant_bias.data())[i] = static_cast<uint16_t>(q);
            break;
          default:
            return MAKE_EP_FAIL("LayerNorm decomposition: unsupported intermediate dtype for bias requant.");
        }
      }

      const std::string requant_bias_name = utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_bias_requant");
      QnnTensorWrapper requant_bias_tensor(requant_bias_name,
                                           QNN_TENSOR_TYPE_STATIC,
                                           x_qnn_data_type,
                                           add_lhs_qp.Copy(),
                                           std::vector<uint32_t>(bias_info.shape),
                                           std::move(requant_bias));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(requant_bias_tensor)),
                    "Failed to add requantized bias tensor.");
      bias_name = requant_bias_name;
    }

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
                                                  {current, bias_name},
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
