// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cassert>
#include <cmath>

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
  static bool HasNonOneDimBeforeAxis(const std::vector<uint32_t>& shape,
                                     size_t input_rank,
                                     size_t ln_axis) {
    assert(shape.size() <= input_rank);
    const size_t prefix = input_rank - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
      const size_t aligned_axis = prefix + i;
      if (aligned_axis < ln_axis && shape[i] != 1) {
        return true;
      }
    }
    return false;
  }

  // Bundles the shape/dtype/axis/policy decisions taken in ProcessAttributesAndOutputs and
  // forwarded into BuildDecomposedLayerNorm. Keeping them in a struct keeps the function signature
  // readable; the fields are all small and copy-cheap. x_shape is stored by value so the plan
  // doesn't dangle if a future caller defers / stores it.
  struct DecomposedLayerNormPlan {
    std::vector<uint32_t> x_shape;
    Qnn_DataType_t x_qnn_data_type;
    size_t ln_axis;
    bool externalize_scale;
    bool externalize_bias;
  };

  Ort::Status BuildDecomposedLayerNorm(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const std::vector<std::string>& input_names,
                                       std::vector<std::string>&& param_tensor_names,
                                       const DecomposedLayerNormPlan& plan,
                                       bool do_op_validation,
                                       const Ort::Logger& logger) const ORT_MUST_USE_RESULT;
};

namespace {

// Write `q` into the i-th element of a typed buffer based on `dtype`. Supports the four
// 8/16-bit signed/unsigned fixed-point dtypes that utils::Quantize can saturate to.
Ort::Status StoreQuantizedFixedPoint(Qnn_DataType_t dtype, uint8_t* dst, size_t i, int q) {
  switch (dtype) {
    case QNN_DATATYPE_SFIXED_POINT_8:
      reinterpret_cast<int8_t*>(dst)[i] = static_cast<int8_t>(q);
      return Ort::Status();
    case QNN_DATATYPE_UFIXED_POINT_8:
      reinterpret_cast<uint8_t*>(dst)[i] = static_cast<uint8_t>(q);
      return Ort::Status();
    case QNN_DATATYPE_SFIXED_POINT_16:
      reinterpret_cast<int16_t*>(dst)[i] = static_cast<int16_t>(q);
      return Ort::Status();
    case QNN_DATATYPE_UFIXED_POINT_16:
      reinterpret_cast<uint16_t*>(dst)[i] = static_cast<uint16_t>(q);
      return Ort::Status();
    default:
      return MAKE_EP_FAIL("Unsupported fixed-point dtype for quantized store.");
  }
}

// Requantize a per-tensor packed static buffer from (src_dtype, src_scale, src_offset) to
// (dst_dtype, dst_scale, dst_offset). Source supports 8/16-bit signed and unsigned fixed-point
// plus 32-bit signed (used for QDQ bias); destination supports 8/16-bit signed and unsigned
// fixed-point (utils::Quantize cannot saturate to 32-bit). Resizes `dst` to the right byte
// length. Caller is expected to have allowlisted `src_dtype` upfront for a clearer diagnostic.
Ort::Status LoadFixedPointAsInt64(Qnn_DataType_t src_dtype, const uint8_t* src, size_t i,
                                  /*out*/ int64_t& value) {
  switch (src_dtype) {
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_INT_32:
      value = reinterpret_cast<const int32_t*>(src)[i];
      return Ort::Status();
    case QNN_DATATYPE_SFIXED_POINT_16:
      value = reinterpret_cast<const int16_t*>(src)[i];
      return Ort::Status();
    case QNN_DATATYPE_UFIXED_POINT_16:
      value = reinterpret_cast<const uint16_t*>(src)[i];
      return Ort::Status();
    case QNN_DATATYPE_SFIXED_POINT_8:
      value = reinterpret_cast<const int8_t*>(src)[i];
      return Ort::Status();
    case QNN_DATATYPE_UFIXED_POINT_8:
      value = reinterpret_cast<const uint8_t*>(src)[i];
      return Ort::Status();
    default:
      return MAKE_EP_FAIL("Requantize: unsupported source fixed-point dtype.");
  }
}

Ort::Status RequantizePerTensorStatic(const std::vector<uint8_t>& src,
                                      Qnn_DataType_t src_dtype,
                                      float src_scale,
                                      int32_t src_offset,
                                      Qnn_DataType_t dst_dtype,
                                      float dst_scale,
                                      int32_t dst_offset,
                                      const Ort::Logger& logger,
                                      std::vector<uint8_t>& dst) {
  const size_t src_elem = utils::GetElementSizeByType(src_dtype);
  RETURN_IF_NOT(src_elem > 0 && src.size() % src_elem == 0,
                "Requantize: source size is not a multiple of element size.");
  const size_t dst_elem = utils::GetElementSizeByType(dst_dtype);
  RETURN_IF_NOT(dst_elem > 0, "Requantize: unsupported destination element size.");

  // utils::Quantize silently saturates. Track per-element clipping and warn once at end so the
  // caller (and downstream debuggers) get a signal that the requantized buffer is lossy beyond
  // normal rounding. Callers can decide whether to tolerate or fall back; this is purely a
  // visibility fix for the silent-saturation failure mode.
  int dst_qmin = 0;
  int dst_qmax = 0;
  RETURN_IF_ERROR(utils::GetQminQmax(dst_dtype, dst_qmin, dst_qmax));
  size_t saturated_count = 0;

  const size_t num = src.size() / src_elem;
  dst.assign(num * dst_elem, 0);
  for (size_t i = 0; i < num; ++i) {
    int64_t loaded = 0;
    RETURN_IF_ERROR(LoadFixedPointAsInt64(src_dtype, src.data(), i, loaded));
    const double dq = utils::Dequantize(src_offset, src_scale, static_cast<double>(loaded));
    const int unclipped =
        static_cast<int>(std::round((dq / static_cast<double>(dst_scale)) - static_cast<double>(dst_offset)));
    if (unclipped < dst_qmin || unclipped > dst_qmax) {
      ++saturated_count;
    }
    int q = 0;
    RETURN_IF_ERROR(utils::Quantize(dq, dst_scale, dst_offset, dst_dtype, q));
    RETURN_IF_ERROR(StoreQuantizedFixedPoint(dst_dtype, dst.data(), i, q));
  }
  if (saturated_count > 0) {
    const std::string msg = "LayerNorm decomposition: requantized static tensor saturated " +
                            std::to_string(saturated_count) + " of " + std::to_string(num) +
                            " elements; downstream output will be silently clipped.";
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, msg.c_str());
  }
  return Ort::Status();
}

}  // namespace

Ort::Status LayerNormalizationOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const Ort::Logger& logger) const {
  // Also check output type is float for CPU.
  const auto& outputs = node_unit.Outputs();
  RETURN_IF(outputs.size() > 1, "QNN LayerNorm only support 1 output.");

  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();

  // Reject scale/bias whose rank exceeds X's rank. ONNX LN requires scale/B to broadcast to
  // X.shape[axis:], so their rank cannot exceed input_rank. Letting such a model through would
  // either underflow the prefix arithmetic in HasNonOneDimBeforeAxis or surface later as an opaque
  // QNN shape-mismatch error; falling back to CPU EP here gives a clean diagnostic instead.
  std::vector<uint32_t> scale_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].shape, scale_shape), "Cannot get shape of input 1 (scale)");
  RETURN_IF(scale_shape.size() > input_rank,
            "QNN LayerNorm: scale rank exceeds input rank; model violates ONNX broadcasting spec.");
  if (inputs.size() > 2 && inputs[2].Exists()) {
    std::vector<uint32_t> bias_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[2].shape, bias_shape), "Cannot get shape of input 2 (bias)");
    RETURN_IF(bias_shape.size() > input_rank,
              "QNN LayerNorm: bias rank exceeds input rank; model violates ONNX broadcasting spec.");
  }

  // QNN Op validation can also do the same work, but the message is not so clear.
  // Explicit check and provide clear message here
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    int32_t ln_axis = -1;
    Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
    RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, ln_axis));
    RETURN_IF(static_cast<size_t>(ln_axis) != input_rank - 1,
              "QNN LayerNorm on HTP only supports normalization along the last axis.");
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
  int32_t ln_axis = -1;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, ln_axis));
  // ProcessAxisAttribute is supposed to normalize ln_axis into [0, input_rank); range-check before
  // the subtract so a malformed axis fails loudly instead of underflowing axes_rank to ~SIZE_MAX.
  RETURN_IF(ln_axis < 0 || static_cast<size_t>(ln_axis) >= input_rank,
            "QNN LayerNorm: axis out of range after normalization.");
  size_t axes_rank = input_rank - static_cast<size_t>(ln_axis);
  std::vector<uint32_t> axes(axes_rank, 0);
  std::vector<uint32_t> axes_shape{SafeInt<uint32_t>(axes_rank)};
  axes[0] = static_cast<uint32_t>(ln_axis);
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
      HasNonOneDimBeforeAxis(scale_info.shape, input_rank, static_cast<size_t>(ln_axis));

  bool bias_misaligned = false;
  if (has_bias_input) {
    TensorInfo bias_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));
    bias_misaligned =
        HasNonOneDimBeforeAxis(bias_info.shape, input_rank, static_cast<size_t>(ln_axis));
  }

  const bool externalize_scale = scale_misaligned;
  const bool externalize_bias = has_bias_input && (bias_misaligned || scale_misaligned);

  if (externalize_scale || externalize_bias) {
    TensorInfo x_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], x_info));
    DecomposedLayerNormPlan plan{
        /*x_shape=*/input_shape,
        /*x_qnn_data_type=*/x_info.qnn_data_type,
        /*ln_axis=*/static_cast<size_t>(ln_axis),
        /*externalize_scale=*/externalize_scale,
        /*externalize_bias=*/externalize_bias,
    };
    return BuildDecomposedLayerNorm(qnn_model_wrapper,
                                    node_unit,
                                    input_names,
                                    std::move(param_tensor_names),
                                    plan,
                                    do_op_validation,
                                    logger);
  }

#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 17 && QNN_API_VERSION_MINOR <= 20
  // Bias is implicit. QNN SDK 2.24 to 2.27 (QNN API version 2.17 to 2.20) has a validation bug for
  // implicit bias inputs, so provide an explicit bias of all 0 (quantized int32). Done here (after
  // the decomposition branch) so the synthesized tensor is never orphaned by the decomposed path,
  // which builds its own LN input list and applies the same workaround independently.
  if (!has_bias_input && IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    TensorInfo x_input_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], x_input_info));
    if (x_input_info.quant_param.IsPerTensor(/*include_bw*/ true) && scale_info.quant_param.IsQuantized()) {
      const std::string bias_name = utils::UniqueNameGenerator().New(node_unit, "_implicit_bias");
      std::vector<uint32_t> bias_shape = scale_info.shape;
      RETURN_IF_ERROR(AddZeroBiasInput(qnn_model_wrapper, x_input_info.quant_param, scale_info.quant_param,
                                       std::move(bias_shape), bias_name, logger, input_names));
    }
  }
#endif

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
                                                                  const DecomposedLayerNormPlan& plan,
                                                                  bool do_op_validation,
                                                                  const Ort::Logger& logger) const {
  const std::vector<uint32_t>& x_shape = plan.x_shape;
  const Qnn_DataType_t x_qnn_data_type = plan.x_qnn_data_type;
  const size_t ln_axis = plan.ln_axis;
  const bool externalize_scale = plan.externalize_scale;
  const bool externalize_bias = plan.externalize_bias;

  const auto& outputs = node_unit.Outputs();
  const std::string& final_output_name = outputs[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  const Qnn_TensorType_t final_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  TensorInfo final_output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], final_output_info));

  // ln_intermediate_qp: LN output. With externalize_scale, LN emits standardized values
  // (|x| <= sqrt(N-1)); use a symmetric range over that bound so the external Mul isn't
  // fed clipped values. Otherwise LN already applies the user scale — reuse final_output's
  // qp, since the sqrt(N-1) bound assumes scale~=1 and loses precision at small scales.
  QnnQuantParamsWrapper ln_intermediate_qp;
  if (externalize_scale && final_output_info.quant_param.IsQuantized()) {
    size_t num_norm_elems = 1;
    for (size_t i = ln_axis; i < x_shape.size(); ++i) {
      num_norm_elems *= static_cast<size_t>(x_shape[i]);
    }
    // Hard bound is sqrt(N-1), but real data sits within ~3-sigma; cap at min(sqrt(N-1), 3.0)
    // so narrow dtypes (uint8: 256 levels) don't waste range on an unused tail. Values past 3.0
    // saturate, but are rare enough not to dominate per-tensor error.
    const float hard_bound = num_norm_elems > 1
                                 ? std::sqrt(static_cast<float>(num_norm_elems - 1))
                                 : 1.0f;
    constexpr float kStatisticalLnBound = 3.0f;
    const float ln_abs_max = std::min(hard_bound, kStatisticalLnBound);
    float ln_scale = 0.0f;
    int32_t ln_offset = 0;
    RETURN_IF_ERROR(utils::GetQuantParams(-ln_abs_max, ln_abs_max, x_qnn_data_type,
                                          ln_scale, ln_offset, /*symmetric=*/false));
    ln_intermediate_qp = QnnQuantParamsWrapper(ln_scale, ln_offset);
  } else {
    // FP path
    ln_intermediate_qp = final_output_info.quant_param.Copy();
  }

  // Fetch scale_info once; used by the synth-ones branch, the SDK 2.17-2.20 implicit-bias guard,
  // and the requantize-scale path. Each of those used to fetch independently.
  TensorInfo scale_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], scale_info));

  std::string ln_scale_name;
  if (!externalize_scale) {
    ln_scale_name = input_names[1];
  } else {
    // Match the synthesized ones tensor to the user-provided scale's dtype + quant params so the
    // LN op sees the type it expects in slot 1. Encoding 1.0 in the user's quant scheme is what
    // makes this an identity scale at runtime.
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
      float quant_scale = 0.0f;
      int32_t quant_offset = 0;
      RETURN_IF_ERROR(scale_info.quant_param.GetPerTensorScaleOffset(quant_scale, quant_offset));
      int quant_one = 0;
      RETURN_IF_ERROR(utils::Quantize(1.0, quant_scale, quant_offset, scale_dtype, quant_one));
      // utils::Quantize silently saturates. If the user's scale can't represent 1.0 within an
      // LSB (e.g. small-amplitude gamma like u8 over [0, 0.005] → quant_one rounds to 51000 and
      // clips to 255 ≈ 0.005), the synthesized "ones" tensor is silently off by orders of
      // magnitude. Reject so the node falls back to CPU.
      const double deq_one = utils::Dequantize(quant_offset, quant_scale, static_cast<double>(quant_one));
      RETURN_IF_NOT(std::abs(deq_one - 1.0) <= static_cast<double>(quant_scale),
                    "LayerNorm scale decomposition: user scale's quantization range cannot represent 1.0; "
                    "synthesized identity scale would saturate.");
      const size_t elem_bytes = utils::GetElementSizeByType(scale_dtype);
      RETURN_IF_NOT(elem_bytes > 0, "LayerNorm scale decomposition: unsupported quantized scale dtype.");
      const_buf.assign(num_elems * elem_bytes, 0);
      for (size_t i = 0; i < num_elems; ++i) {
        RETURN_IF_ERROR(StoreQuantizedFixedPoint(scale_dtype, const_buf.data(), i, quant_one));
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

#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 17 && QNN_API_VERSION_MINOR <= 20
  // Mirror the ProcessInputs workaround: on QNN SDK 2.24-2.27 (API 2.17-2.20), an LN node without
  // an explicit bias intermittently fails graph finalize on NPU. The decomposed path always emits
  // LN with only {X, scale}, so synthesize a zero int32 bias here when the same prerequisites hold.
  if (IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    TensorInfo x_input_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], x_input_info));
    if (x_input_info.quant_param.IsPerTensor(/*include_bw*/ true) && scale_info.quant_param.IsQuantized()) {
      const std::string bias_name = utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_implicit_bias");
      std::vector<uint32_t> bias_shape(x_shape.begin() + ln_axis, x_shape.end());
      RETURN_IF_ERROR(AddZeroBiasInput(qnn_model_wrapper, x_input_info.quant_param, scale_info.quant_param,
                                       std::move(bias_shape), bias_name, logger, ln_inputs));
    }
  }
#endif

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
    // Mul produces z*gamma, whose magnitude is bounded by gamma * sqrt(N-1) — well past
    // ln_intermediate_qp's sqrt(N-1) range whenever |gamma| > 1. final_output_info.quant_param
    // is sized for z*gamma+beta and trivially covers z*gamma. Owned copy: the
    // externalize_bias=false branch moves .quant_param into the Mul output tensor below; the
    // externalize_bias=true branch hands it to the Add output later. Either way, mul_out_qp must
    // not alias storage that gets moved out before the requant predicates / scale reads run.
    const QnnQuantParamsWrapper mul_out_qp = final_output_info.quant_param.Copy();

    // QNN ELEMENT_WISE_MULTIPLY requires both operands to share a dtype. In QDQ pipelines (e.g.
    // A16W8) the LN output adopts X's dtype while the user scale is still in its own (often
    // narrower) dtype. Requantize the static scale initializer to x_qnn_data_type so the Mul
    // sees matching operand dtypes. ONNX LN spec demands X/scale/bias share dtype, so a mismatch
    // here implies a quantized graph where the original initializer is available.
    std::string mul_scale_name = input_names[1];
    if (scale_info.qnn_data_type != x_qnn_data_type) {
      RETURN_IF_NOT(scale_info.is_initializer && scale_info.initializer_tensor != nullptr,
                    "LayerNorm decomposition: externalized scale must be a static initializer.");
      RETURN_IF_NOT(scale_info.quant_param.IsPerTensor(/*include_bw*/ true),
                    "LayerNorm decomposition: externalized scale must be per-tensor quantized.");
      RETURN_IF_NOT(mul_out_qp.IsPerTensor(/*include_bw*/ true),
                    "LayerNorm decomposition: requantized scale requires per-tensor intermediate quant params.");

      float src_scale = 0.0f;
      int32_t src_offset = 0;
      RETURN_IF_ERROR(scale_info.quant_param.GetPerTensorScaleOffset(src_scale, src_offset));
      float dst_scale = 0.0f;
      int32_t dst_offset = 0;
      RETURN_IF_ERROR(mul_out_qp.GetPerTensorScaleOffset(dst_scale, dst_offset));

      switch (scale_info.qnn_data_type) {
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
          break;
        default:
          return MAKE_EP_FAIL("LayerNorm decomposition: unsupported externalized scale dtype.");
      }

      std::vector<uint8_t> raw_scale;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(scale_info.initializer_tensor, raw_scale));

      std::vector<uint8_t> requant_scale;
      RETURN_IF_ERROR(RequantizePerTensorStatic(raw_scale,
                                                scale_info.qnn_data_type,
                                                src_scale, src_offset,
                                                x_qnn_data_type,
                                                dst_scale, dst_offset,
                                                logger,
                                                requant_scale));

      const std::string requant_scale_name = utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_scale_requant");
      QnnTensorWrapper requant_scale_tensor(requant_scale_name,
                                            QNN_TENSOR_TYPE_STATIC,
                                            x_qnn_data_type,
                                            mul_out_qp.Copy(),
                                            std::vector<uint32_t>(scale_info.shape),
                                            std::move(requant_scale));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(requant_scale_tensor)),
                    "Failed to add requantized scale tensor.");
      mul_scale_name = requant_scale_name;
    }  // End of requantize scale

    // Mul output: either an internal handoff to the Add (NATIVE, x_dtype, mul_out_qp, x_shape)
    // or the graph output (final_tensor_type, final dtype/qp/shape). Splitting the two cases
    // makes the move-from-final_output_info contract obvious — the final-output branch is the
    // sole writer, so reads of final_output_info.{quant_param,shape} below it are forbidden.
    std::string mul_out_name;
    if (externalize_bias) {
      mul_out_name = utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_mul_out");
      QnnTensorWrapper mul_out_tensor(mul_out_name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      x_qnn_data_type,
                                      mul_out_qp.Copy(),
                                      std::vector<uint32_t>(x_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul_out_tensor)),
                    "Failed to add decomposed Mul output tensor.");
    } else {
      mul_out_name = final_output_name;
      QnnTensorWrapper mul_out_tensor(mul_out_name,
                                      final_tensor_type,
                                      final_output_info.qnn_data_type,
                                      std::move(final_output_info.quant_param),
                                      std::move(final_output_info.shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul_out_tensor)),
                    "Failed to add decomposed Mul output tensor.");
    }
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_mul"),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                  {current, mul_scale_name},
                                                  {mul_out_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add decomposed Mul node.");
    current = mul_out_name;
  }

  if (externalize_bias) {
    // Bias is the Add RHS; size it to cover beta directly. final_output_info.quant_param is
    // sized for z*gamma+beta and so covers beta alone with room to spare. ln_intermediate_qp
    // is too narrow here (its sqrt(N-1) range was sized for z, not beta — a bias whose magnitude
    // exceeds sqrt(N-1) would silently saturate during requantize).
    std::string bias_name = input_names[2];
    TensorInfo bias_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[2], bias_info));

    // QDQ pipelines feed LayerNorm's bias as int32 to satisfy QNN's quantized-LN bias convention
    // (ONNX LN itself requires B to share X's dtype — the int32 only appears once the QDQ pipeline
    // has rewritten it). QNN's LayerNorm op accepts the int32, but ELEMENT_WISE_ADD requires both
    // operands to share a dtype — requantize the static bias to match LN output dtype. The ONNX-
    // level dtype-match invariant means a mismatch here implies a QDQ graph with the original
    // initializer available.
    if (bias_info.qnn_data_type != x_qnn_data_type) {
      RETURN_IF_NOT(bias_info.is_initializer && bias_info.initializer_tensor != nullptr,
                    "LayerNorm decomposition: externalized bias must be a static initializer.");
      RETURN_IF_NOT(bias_info.quant_param.IsPerTensor(/*include_bw*/ true),
                    "LayerNorm decomposition: externalized bias must be per-tensor quantized.");
      RETURN_IF_NOT(final_output_info.quant_param.IsPerTensor(/*include_bw*/ true),
                    "LayerNorm decomposition: requantized bias requires per-tensor final output quant params.");

      float src_scale = 0.0f;
      int32_t src_offset = 0;
      RETURN_IF_ERROR(bias_info.quant_param.GetPerTensorScaleOffset(src_scale, src_offset));
      float dst_scale = 0.0f;
      int32_t dst_offset = 0;
      RETURN_IF_ERROR(final_output_info.quant_param.GetPerTensorScaleOffset(dst_scale, dst_offset));

      // Allowlist bias dtypes upfront so a future QDQ pipeline emitting an unknown bias dtype
      // fails loudly here, rather than silently producing an all-zero requantized bias via the
      // helper's load default branch.
      switch (bias_info.qnn_data_type) {
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
          break;
        default:
          return MAKE_EP_FAIL("LayerNorm decomposition: unsupported externalized bias dtype.");
      }

      std::vector<uint8_t> raw_bias;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, raw_bias));

      std::vector<uint8_t> requant_bias;
      RETURN_IF_ERROR(RequantizePerTensorStatic(raw_bias,
                                                bias_info.qnn_data_type,
                                                src_scale, src_offset,
                                                x_qnn_data_type,
                                                dst_scale, dst_offset,
                                                logger,
                                                requant_bias));

      const std::string requant_bias_name = utils::UniqueNameGenerator().New(node_unit, "_ln_decomposed_bias_requant");
      QnnTensorWrapper requant_bias_tensor(requant_bias_name,
                                           QNN_TENSOR_TYPE_STATIC,
                                           x_qnn_data_type,
                                           final_output_info.quant_param.Copy(),
                                           std::vector<uint32_t>(bias_info.shape),
                                           std::move(requant_bias));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(requant_bias_tensor)),
                    "Failed to add requantized bias tensor.");
      bias_name = requant_bias_name;
    }  // end of requant bias

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
