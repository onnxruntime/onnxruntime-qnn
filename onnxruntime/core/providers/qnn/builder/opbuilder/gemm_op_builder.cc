// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include <gsl/gsl>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_bq_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Detects a block-quantized Gemm weight B, accounting for transB.
// For transB=0: B is [K, N], scale is [K/block_size, N], blocked on axis 0.
// For transB=1: B is [N, K], scale is [N, K/block_size], blocked on axis 1.
// Both cases: K is the contraction axis (blocked axis). Returns true if blocked weight detected.
bool IsBQGemmWeight(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnitIODef& weight,
                    int64_t trans_b) {
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return false;
  }
  if (!weight.quant_param.has_value() || weight.quant_param->scale == nullptr) {
    return false;
  }
  const auto scale_shape = utils::GetInitializerShape(weight.quant_param->scale, qnn_model_wrapper.GetOrtApi());
  std::vector<uint32_t> weight_shape;
  if (!QnnModelWrapper::GetOnnxShape(weight.shape, weight_shape) || weight_shape.size() != 2) {
    return false;  // BQ only for rank-2 Gemm weight.
  }
  if (scale_shape.size() != 2) {
    return false;
  }
  // For transB=0: B=[K,N], blocked on axis 0 → scale=[num_blocks, N], scale_shape[0] < weight_shape[0].
  // For transB=1: B=[N,K], blocked on axis 1 → scale=[N, num_blocks], scale_shape[1] < weight_shape[1].
  const size_t block_axis = (trans_b == 0) ? 0 : 1;
  return bq::IsBQScale(scale_shape, weight_shape, block_axis);
}

// Float types QNN FullyConnected can run in (see HtpOpDefSupplement, FullyConnected datatypes).
bool IsFloatQnnDataType(Qnn_DataType_t dt) {
  return dt == QNN_DATATYPE_FLOAT_16 || dt == QNN_DATATYPE_FLOAT_32;
}

// Precision a QNN FullyConnected kernel runs at. Ordered so that a plain max()
// picks the precision a Gemm must run at: higher tiers can represent everything a lower tier can
enum class GemmPrecision {
  kInt8,     // (U|S)FIXED_POINT_4/8 — HTP folds sub-byte weights into the 8-bit configuration
  kInt16,    // (U|S)FIXED_POINT_16
  kFloat16,  // FLOAT_16
  kFloat32,  // FLOAT_32
};

// Maps a tensor's data type onto the kernel precision that can hold it. Types with no tier of their
// own (BFLOAT_16, plain ints) are rejected: they only reach here when a Gemm mixes data types, and
// there is no kernel that mixes them with anything else.
Ort::Status GetGemmPrecision(Qnn_DataType_t data_type, /*out*/ GemmPrecision& precision) {
  switch (data_type) {
    case QNN_DATATYPE_UFIXED_POINT_4:
    case QNN_DATATYPE_SFIXED_POINT_4:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      precision = GemmPrecision::kInt8;
      return Ort::Status();
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      precision = GemmPrecision::kInt16;
      return Ort::Status();
    case QNN_DATATYPE_FLOAT_16:
      precision = GemmPrecision::kFloat16;
      return Ort::Status();
    case QNN_DATATYPE_FLOAT_32:
      precision = GemmPrecision::kFloat32;
      return Ort::Status();
    default:
      return MAKE_EP_FAIL("QNN EP: Gemm data type cannot mix with another QNN FullyConnected data type.");
  }
}

// Data type an activation takes at `precision`. Quantized precisions keep `like`'s signedness so a
// widened activation stays unsigned/signed like the one it came from.
Qnn_DataType_t GetActivationDataType(GemmPrecision precision, Qnn_DataType_t like) {
  const bool is_signed = like == QNN_DATATYPE_SFIXED_POINT_4 || like == QNN_DATATYPE_SFIXED_POINT_8 ||
                         like == QNN_DATATYPE_SFIXED_POINT_16;
  switch (precision) {
    case GemmPrecision::kInt8:
      return is_signed ? QNN_DATATYPE_SFIXED_POINT_8 : QNN_DATATYPE_UFIXED_POINT_8;
    case GemmPrecision::kInt16:
      return is_signed ? QNN_DATATYPE_SFIXED_POINT_16 : QNN_DATATYPE_UFIXED_POINT_16;
    case GemmPrecision::kFloat16:
      return QNN_DATATYPE_FLOAT_16;
    case GemmPrecision::kFloat32:
      return QNN_DATATYPE_FLOAT_32;
  }
  return QNN_DATATYPE_UNDEFINED;  // Unreachable; every GemmPrecision is handled above.
}

// A per-tensor quantization encoding.
struct ScaleOffset {
  float scale = 0.0f;
  int32_t offset = 0;
};

// The one decision the whole Gemm translation hangs off: which precision FullyConnected runs at, and
// therefore what has to be inserted around it. ProcessInputs and ProcessAttributesAndOutputs build
// one side each and must agree, so both read this instead of re-deriving.
//
//   A -> [fix up in]? -> [FC at fc_act_data_type] -> [fix up out]? -> Y
//
// HTP forces FC's out[0] to carry its in[0] type, so an activation that does not already sit at
// fc_act_data_type gets a Convert/Quantize/Dequantize on its side. Examples, weight tier in brackets:
//
//   u8  -> u8  [u8 ]  run int8    no fix-up
//   u8  -> u16 [u8 ]  run int16   Convert u8->u16 in
//   u16 -> u8  [u8 ]  run int16                                Convert u16->u8 out
//   u8  -> u8  [s16]  run int16   Convert u8->u16 in           Convert u16->u8 out
//   u8  -> f16 [s8 ]  run fp16    Dequantize in
//   f16 -> u8  [s8 ]  run fp16                                 Quantize out
struct GemmPrecisionPlan {
  TensorInfo in_act = {};   // ONNX input activation, Gemm input[0]
  TensorInfo weight = {};   // ONNX weight, Gemm input[1]
  TensorInfo out_act = {};  // ONNX output activation, Gemm output[0]

  // What FC's in[0] and out[0] both carry. A float FC's weight takes it too (see DequantizeWeight);
  // a quantized FC's weight keeps its own ONNX type.
  Qnn_DataType_t fc_act_data_type = QNN_DATATYPE_UNDEFINED;

  // Encoding every tensor the FC chain produces carries (see AddFcChainLink). Empty when FC runs in
  // float.
  QnnQuantParamsWrapper chain_quant_param;

  bool FixesUpInput() const { return in_act.qnn_data_type != fc_act_data_type; }
  bool FixesUpOutput() const { return out_act.qnn_data_type != fc_act_data_type; }

  // A float FC cannot take a quantized weight, so a quantized one needs a float copy.
  bool DequantizesWeight() const {
    return IsFloatQnnDataType(fc_act_data_type) && weight.quant_param.IsQuantized();
  }
};

// Picks the FC precision (the max over the activations and the weight) and derives everything both
// build phases need from it. All rejections live here, so both phases fail identically.
Ort::Status MakeGemmPrecisionPlan(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                  /*out*/ GemmPrecisionPlan& plan) {
  plan = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], plan.in_act));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], plan.weight));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], plan.out_act));

  const Qnn_DataType_t in_type = plan.in_act.qnn_data_type;
  const Qnn_DataType_t out_type = plan.out_act.qnn_data_type;

  // A Gemm whose activations and weight all share one data type is the common case: FC runs at that
  // type and nothing is inserted. Answered up front so uniform types with no precision tier of their
  // own (int32, int64) keep taking the plain path.
  if (in_type == out_type && plan.weight.qnn_data_type == in_type) {
    plan.fc_act_data_type = in_type;
    plan.chain_quant_param = plan.out_act.quant_param.Copy();
    return Ort::Status();
  }

  GemmPrecision in_precision = GemmPrecision::kInt8;
  GemmPrecision out_precision = GemmPrecision::kInt8;
  GemmPrecision weight_precision = GemmPrecision::kInt8;
  RETURN_IF_ERROR(GetGemmPrecision(in_type, in_precision));
  RETURN_IF_ERROR(GetGemmPrecision(out_type, out_precision));
  RETURN_IF_ERROR(GetGemmPrecision(plan.weight.qnn_data_type, weight_precision));

  // Same precision but a different type (e.g. U8 -> S8) is a pure re-signing. Running at either
  // side's type leaves the other needing a fix-up that changes signedness without changing
  // precision, which is not what a Convert expresses. Reject rather than emit such a graph.
  RETURN_IF(in_precision == out_precision && in_type != out_type,
            "QNN EP: unsupported mixed-precision Gemm - input and output activations run at the same "
            "precision but have different data types.");

  // Run at the highest precision any of the three needs. The weight counts because HTP has no kernel
  // with an 8-bit activation and a 16-bit weight ("16bit Weight must have 16bit Activation"), so such
  // a Gemm must run at int16 with both activations fixed up. The int32 bias is deliberately left out:
  // it is an accumulator type, not a kernel precision.
  const GemmPrecision fc_precision = std::max({in_precision, out_precision, weight_precision});

  // Signedness follows whichever activation already sits at the FC precision. If neither does (the
  // weight forced the precision up) both activations are at the same precision, and so at the same
  // type per the check above, so either one answers.
  const Qnn_DataType_t signedness_like = (out_precision == fc_precision) ? out_type : in_type;
  plan.fc_act_data_type = GetActivationDataType(fc_precision, signedness_like);

  // A float chain carries no encoding, so chain_quant_param stays empty and there is nothing more to
  // derive; DequantizesWeight() covers the weight from here.
  if (IsFloatQnnDataType(plan.fc_act_data_type)) {
    return Ort::Status();
  }

  if (plan.FixesUpOutput()) {
    // Run the chain on the narrow output activation's grid re-expressed at FC's wider bit width, so
    // the trailing Convert only drops bits and never rescales. FC still accumulates the bias at
    // input_activation_scale * weight_scale, so the bias needs no requantization for this.
    ScaleOffset output_encoding = {};
    RETURN_IF_ERROR(plan.out_act.quant_param.GetPerTensorScaleOffset(output_encoding.scale,
                                                                     output_encoding.offset));
    ScaleOffset widened_output_encoding = {};
    RETURN_IF_ERROR(utils::DeriveConvertQuantParams(out_type, plan.fc_act_data_type, output_encoding.offset,
                                                    output_encoding.scale, /*output_symmetric*/ false,
                                                    widened_output_encoding.scale,
                                                    widened_output_encoding.offset));
    plan.chain_quant_param = QnnQuantParamsWrapper::PerTensor(widened_output_encoding.scale,
                                                              widened_output_encoding.offset);
  } else {
    // The chain produces the ONNX output tensor itself, so it carries its encoding.
    plan.chain_quant_param = plan.out_act.quant_param.Copy();
  }

  return Ort::Status();
}

// Emits one link of the FC chain — the ops that turn FC's inputs into the node unit's output value:
//
//   plain             [FC]
//   absorbed reshape  [FC] -> [Reshape]          (MatMulAddFusion's post-Gemm Reshape, see below)
//   split bias        [FC] -> [ElementWiseAdd]   (bias QNN FC cannot fold in, see split_gemm)
//
// Every link's output carries the plan's FC data type and chain encoding, which differ from the ONNX
// output's whenever an output fix-up follows. The node is named after the node unit so the QNN graph
// traces back to the ONNX Gemm.
Ort::Status AddFcChainLink(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                           const GemmPrecisionPlan& plan, const char* qnn_op_type,
                           std::vector<std::string>&& input_names, const std::string& output_name,
                           Qnn_TensorType_t output_tensor_type,
                           const std::vector<uint32_t>& output_shape, bool do_op_validation) {
  QnnTensorWrapper output_tensor(output_name, output_tensor_type, plan.fc_act_data_type,
                                 plan.chain_quant_param.Copy(), std::vector<uint32_t>(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)),
                ("Failed to add Gemm FC-chain tensor " + output_name).c_str());
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, qnn_op_type),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW, qnn_op_type,
                                                std::move(input_names), {output_name}, {}, do_op_validation),
                ("Failed to add Gemm " + std::string(qnn_op_type) + " node.").c_str());
  return Ort::Status();
}

// Requantizes the Gemm's int32 bias (input_names[2]) to `activation_scale`, the scale produced by the
// widening Convert, replacing the tensor in place. The bias was quantized at
// original_input_activation_scale * weight_scale, but HTP folds it into the FC accumulator at
// activation_scale * weight_scale and ignores the declared encoding, so the data itself must move.
Ort::Status RequantizeBias(QnnModelWrapper& qnn_model_wrapper, const GemmPrecisionPlan& plan,
                           float activation_scale,
                           /*in,out*/ std::vector<std::string>& input_names) {
  const auto& bias_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[2]);
  const Qnn_TensorType_t bias_tensor_type = bias_wrapper.GetTensorType();

  // A NATIVE int32 bias is an intermediate from another op, which forces split_gemm: the bias lands
  // on a separate ElementWiseAdd that honors its own encoding rather than being folded into the FC
  // accumulator, so it needs no requantization.
  if (bias_wrapper.GetTensorDataType() != QNN_DATATYPE_SFIXED_POINT_32 ||
      bias_tensor_type == QNN_TENSOR_TYPE_NATIVE) {
    return Ort::Status();
  }

  // Any other non-STATIC bias (i.e. a graph input) *is* handed to FC, which folds it at
  // activation_scale * weight_scale and ignores the declared encoding — but it carries no client
  // buffer to rescale. Reject instead of emitting a silently mis-scaled bias.
  RETURN_IF_NOT(bias_tensor_type == QNN_TENSOR_TYPE_STATIC,
                "QNN EP: mixed-precision (widening) Gemm requires a constant int32 bias; a graph-input "
                "bias cannot be requantized for the converted activation scale.");
  const auto& bias_qp = bias_wrapper.GetQnnQuantParams();
  const auto& bias_dims = bias_wrapper.GetTensorDims();

  std::vector<float> bias_scales;
  RETURN_IF_ERROR(bias_qp.GetScales(bias_scales));
  std::vector<int32_t> bias_offsets(bias_scales.size(), 0);

  std::vector<float> weight_scales;
  RETURN_IF_ERROR(plan.weight.quant_param.GetScales(weight_scales));

  const Qnn_Tensor_t& qnn_bias = bias_wrapper.GetQnnTensor();
  const Qnn_ClientBuffer_t& client_buf = GetQnnTensorClientBuf(qnn_bias);
  std::vector<uint8_t> original_bias_data(
      static_cast<const uint8_t*>(client_buf.data),
      static_cast<const uint8_t*>(client_buf.data) + client_buf.dataSize);

  std::vector<uint8_t> requantized_bias_data;
  std::vector<float> new_bias_scales;
  std::vector<int32_t> new_bias_offsets;
  std::optional<int64_t> axis = bias_qp.IsPerChannel()
                                    ? std::optional<int64_t>(0)
                                    : std::nullopt;

  RETURN_IF_ERROR(utils::RequantizeBiasTensor(
      original_bias_data, bias_dims,
      bias_scales, bias_offsets,
      weight_scales, activation_scale,
      QNN_DATATYPE_SFIXED_POINT_32,
      requantized_bias_data, new_bias_scales, new_bias_offsets,
      axis));

  const std::string new_bias_name = utils::UniqueNameGenerator().New(input_names[2], "_requant");
  QnnQuantParamsWrapper new_bias_qp;
  if (bias_qp.IsPerChannel()) {
    new_bias_qp = QnnQuantParamsWrapper::PerChannel(new_bias_scales, new_bias_offsets, /*axis*/ 0);
  } else {
    new_bias_qp = QnnQuantParamsWrapper::PerTensor(new_bias_scales[0], new_bias_offsets[0]);
  }
  QnnTensorWrapper new_bias_wrapper(new_bias_name, QNN_TENSOR_TYPE_STATIC,
                                    QNN_DATATYPE_SFIXED_POINT_32,
                                    std::move(new_bias_qp),
                                    std::vector<uint32_t>(bias_dims),
                                    std::move(requantized_bias_data));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(new_bias_wrapper)),
                "Failed to add requantized bias tensor.");
  input_names[2] = new_bias_name;
  return Ort::Status();
}

// Re-expresses the input activation in the plan's FC data type, when the ONNX activation is not
// already at it: a Convert to a wider quantized type, or a Dequantize to float. No-op otherwise, so
// callers can invoke it unconditionally.
Ort::Status FixUpInput(QnnModelWrapper& qnn_model_wrapper, const GemmPrecisionPlan& plan,
                       /*in,out*/ std::vector<std::string>& input_names, bool do_op_validation) {
  if (!plan.FixesUpInput()) {
    return Ort::Status();
  }

  // input_names[0] is a Transpose output when transA=1, so its dims are the permuted ONNX dims.
  // Read the shape from the staged wrapper rather than from the ONNX-side plan.in_act.
  std::vector<uint32_t> fixed_up_shape = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]).GetTensorDims();

  if (IsFloatQnnDataType(plan.fc_act_data_type)) {
    const std::string dq_output_name = utils::UniqueNameGenerator().New(input_names[0], "_dequantize");
    RETURN_IF_ERROR(qnn_model_wrapper.AddDequantizeNode(input_names[0], dq_output_name, plan.fc_act_data_type,
                                                        std::move(fixed_up_shape), do_op_validation));
    input_names[0] = dq_output_name;
    return Ort::Status();
  }

  ScaleOffset in_encoding = {};
  RETURN_IF_ERROR(plan.in_act.quant_param.GetPerTensorScaleOffset(in_encoding.scale, in_encoding.offset));

  const std::string convert_output_name =
      utils::UniqueNameGenerator().New(input_names[0], "_convert_to_fc_type");
  RETURN_IF_ERROR(utils::InsertConvertOp(qnn_model_wrapper, input_names[0], convert_output_name,
                                         plan.in_act.qnn_data_type, plan.fc_act_data_type,
                                         in_encoding.offset, in_encoding.scale,
                                         fixed_up_shape, /*output_symmetric*/ false, do_op_validation));
  input_names[0] = convert_output_name;

  if (input_names.size() == 3) {
    // The Convert re-expressed the same float range at FC's wider bit width, so the activation scale
    // changed. FC folds the bias in at that derived scale, so the bias data must follow it. Mirrors
    // InsertConvertOp's own derivation above.
    ScaleOffset converted_in_encoding = {};
    RETURN_IF_ERROR(utils::DeriveConvertQuantParams(plan.in_act.qnn_data_type, plan.fc_act_data_type,
                                                    in_encoding.offset, in_encoding.scale,
                                                    /*output_symmetric*/ false, converted_in_encoding.scale,
                                                    converted_in_encoding.offset));
    RETURN_IF_ERROR(RequantizeBias(qnn_model_wrapper, plan, converted_in_encoding.scale, input_names));
  }
  return Ort::Status();
}

// Replaces the quantized weight in input_names[1] with a statically dequantized float copy at the
// plan's FC data type, so FC can run in float mode. No-op when FC runs quantized.
Ort::Status DequantizeWeight(QnnModelWrapper& qnn_model_wrapper, const GemmPrecisionPlan& plan,
                             /*in,out*/ std::vector<std::string>& input_names) {
  if (!plan.DequantizesWeight()) {
    return Ort::Status();
  }

  const Qnn_DataType_t fc_act_data_type = plan.fc_act_data_type;

  const auto& weight_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]);
  const Qnn_Tensor_t& qnn_weight = weight_wrapper.GetQnnTensor();
  const Qnn_ClientBuffer_t& w_buf = GetQnnTensorClientBuf(qnn_weight);

  // Use wrapper dims (post-transpose) and wrapper quant params (already transposed)
  const auto& w_dims = weight_wrapper.GetTensorDims();
  const auto& w_qp = weight_wrapper.GetQnnQuantParams();
  std::vector<float> weight_scales;
  RETURN_IF_ERROR(w_qp.GetScales(weight_scales));
  std::vector<int32_t> weight_offsets(weight_scales.size(), 0);

  size_t num_elements = 1;
  for (auto d : w_dims) num_elements *= d;

  std::vector<float> fp32_weights(num_elements);
  std::optional<int64_t> axis = w_qp.IsPerChannel() ? std::optional<int64_t>(0) : std::nullopt;
  RETURN_IF_ERROR(utils::DequantizePerChannel(
      gsl::span<const uint8_t>(static_cast<const uint8_t*>(w_buf.data), w_buf.dataSize),
      gsl::span<const uint32_t>(w_dims.data(), w_dims.size()),
      weight_scales, weight_offsets,
      fp32_weights, plan.weight.qnn_data_type, axis));

  // Weights must match FC's float width, so they take the activation type: an fp16 FC's in[1] is
  // fp16, an fp32 FC's is fp32 (see HtpOpDefSupplement, FullyConnected datatypes).
  std::vector<uint8_t> float_bytes;
  if (fc_act_data_type == QNN_DATATYPE_FLOAT_16) {
    float_bytes.resize(num_elements * sizeof(uint16_t));
    auto* fp16_dst = reinterpret_cast<uint16_t*>(float_bytes.data());
    for (size_t i = 0; i < num_elements; ++i) {
      const Ort::Float16_t fp16(fp32_weights[i]);
      memcpy(&fp16_dst[i], &fp16.val, sizeof(uint16_t));
    }
  } else {
    float_bytes.resize(num_elements * sizeof(float));
    memcpy(float_bytes.data(), fp32_weights.data(), float_bytes.size());
  }

  const std::string float_weight_name = utils::UniqueNameGenerator().New(
      input_names[1], fc_act_data_type == QNN_DATATYPE_FLOAT_16 ? "_fp16" : "_fp32");
  QnnTensorWrapper float_weight_tensor(float_weight_name, QNN_TENSOR_TYPE_STATIC,
                                       fc_act_data_type, QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(w_dims.begin(), w_dims.end()),
                                       std::move(float_bytes));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(float_weight_tensor)),
                "Failed to add float weight tensor.");
  input_names[1] = float_weight_name;
  return Ort::Status();
}

// Re-encodes the FC chain's result into the ONNX output activation's type, when FC could not run at
// that type directly. No-op otherwise, so callers can invoke it unconditionally.
Ort::Status FixUpOutput(QnnModelWrapper& qnn_model_wrapper, const GemmPrecisionPlan& plan,
                        const std::string& chain_output_name, const std::string& onnx_output_name,
                        bool do_op_validation) {
  if (!plan.FixesUpOutput()) {
    return Ort::Status();
  }

  // The ONNX output is always the narrower of the two here: had it been float it would have set the FC
  // precision itself and needed no fix-up. So the chain is either quantized (Convert, drop bits) or
  // float (Quantize, back to the ONNX grid).
  const char* qnn_op_type = IsFloatQnnDataType(plan.fc_act_data_type) ? QNN_OP_QUANTIZE : QNN_OP_CONVERT;
  const Qnn_TensorType_t final_tensor_type = qnn_model_wrapper.GetTensorType(onnx_output_name);
  QnnTensorWrapper final_output_tensor(onnx_output_name, final_tensor_type, plan.out_act.qnn_data_type,
                                       plan.out_act.quant_param.Copy(),
                                       std::vector<uint32_t>(plan.out_act.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)),
                "Failed to add mixed-precision Gemm output tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(onnx_output_name, qnn_op_type),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                qnn_op_type,
                                                {chain_output_name},
                                                {onnx_output_name},
                                                {},
                                                do_op_validation),
                "Failed to add mixed-precision Gemm output node.");
  return Ort::Status();
}

}  // namespace

class GemmOpBuilder : public BaseOpBuilder {
 public:
  GemmOpBuilder() : BaseOpBuilder("GemmOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GemmOpBuilder);

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
  Ort::Status ExplictOpCheck(const OrtNodeUnit& node_unit) const;
  // Block-quantized (BW_FLOAT_BLOCK) weight path for Gemm→QNN FullyConnected.
  Ort::Status ProcessInputsForBQGemm(QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnit& node_unit,
                                     int64_t trans_b,
                                     float beta,
                                     const Ort::Logger& logger,
                                     std::vector<std::string>& input_names,
                                     bool do_op_validation) const ORT_MUST_USE_RESULT;

  // Emits every node between the Gemm's inputs and `chain_output_name`: FC alone, FC+Reshape when the
  // QDQ selector absorbed a post-Gemm Reshape, or FC+ElementWiseAdd when `split_gemm` says QNN FC
  // cannot fold the bias in. Every link runs at the FC chain's type and encoding (see AddFcChainLink).
  Ort::Status AddFcChain(QnnModelWrapper& qnn_model_wrapper,
                         const OrtNodeUnit& node_unit,
                         const OrtNodeAttrHelper& node_helper,
                         const GemmPrecisionPlan& plan,
                         std::vector<std::string>&& input_names,
                         const std::string& chain_output_name,
                         bool split_gemm,
                         const Ort::Logger& logger,
                         bool do_op_validation) const ORT_MUST_USE_RESULT;
};

Ort::Status GemmOpBuilder::ExplictOpCheck(const OrtNodeUnit& node_unit) const {
  OrtNodeAttrHelper node_helper(node_unit);
  auto alpha = node_helper.Get("alpha", (float)1.0);
  RETURN_IF(alpha != 1.0, "QNN FullyConnected Op only support alpha=1.0.");
  auto beta = node_helper.Get("beta", (float)1.0);
  RETURN_IF(beta != 1.0 && beta != 0.0, "QNN FullyConnected Op only support beta=1.0 or beta=0.0.");

  // input C shape need to be [M] or [1, M] (skip validation when beta=0.0 — C is ignored)
  if (node_unit.Inputs().size() == 3 && beta != 0.0f) {
    auto& inputB = node_unit.Inputs()[1];
    std::vector<uint32_t> inputB_shape;
    QnnModelWrapper::GetOnnxShape(inputB.shape, inputB_shape);

    auto& inputC = node_unit.Inputs()[2];
    std::vector<uint32_t> inputC_shape;
    QnnModelWrapper::GetOnnxShape(inputC.shape, inputC_shape);

    auto transB = node_helper.Get("transB", static_cast<int64_t>(0));
    auto M = (transB == 0) ? inputB_shape.at(1) : inputB_shape.at(0);
    RETURN_IF(inputC_shape.size() == 0 || (inputC_shape.size() == 1 && inputC_shape.at(0) != M) ||
                  (inputC_shape.size() == 2 && inputC_shape.at(1) != M),
              "QNN FullyConnected Op only support C with shape [N, M].");

    RETURN_IF(inputC_shape.size() == 2 && node_unit.Inputs()[2].quant_param.has_value() && inputC_shape.at(0) != 1,
              "QNN FullyConnected Op only support quantized C with shape [1, M].");
  }

  return Ort::Status();
}

Ort::Status GemmOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_ERROR(ExplictOpCheck(node_unit));
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t trans_b = node_helper.Get("transB", static_cast<int64_t>(0));
  const float beta = node_helper.Get("beta", 1.0f);
  const auto& inputs = node_unit.Inputs();

  // Block-quantized weight: translate to QNN FullyConnected with BW_FLOAT_BLOCK weight.
  if (IsBQGemmWeight(qnn_model_wrapper, inputs[1], trans_b)) {
    return ProcessInputsForBQGemm(qnn_model_wrapper, node_unit, trans_b, beta, logger, input_names,
                                  do_op_validation);
  }

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  // for Input A, B, C: 1 -- need transpose, 0 -- not needed
  std::vector<int64_t> input_trans_flag(3, 0);
  input_trans_flag.at(0) = node_helper.Get("transA", (int64_t)0);
  auto transB = node_helper.Get("transB", (int64_t)0);
  // QNN input_1 [m, n] vs Onnx [n, m]
  input_trans_flag.at(1) = transB == 0 ? 1 : 0;
  for (size_t input_i = 0; input_i < inputs.size(); ++input_i) {
    // beta=0.0: C has no effect on the output — skip it so FC receives only (A, B)
    if (input_i == 2 && beta == 0.0f) {
      continue;
    }

    QnnQuantParamsWrapper quantize_param;
    RETURN_IF_ERROR(quantize_param.Init(qnn_model_wrapper, inputs[input_i]));

    bool is_quantized_tensor = inputs[input_i].quant_param.has_value();
    const auto& input_name = inputs[input_i].name;

    // Only skip if the input tensor has already been added (by producer op) *and* we don't need
    // to transpose it.
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name) && input_trans_flag[input_i] == 0) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_name).c_str());
      input_names.push_back(input_name);
      continue;
    }

    ONNXTensorElementDataType input_type = inputs[input_i].type;
    RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_tensor, input_type, qnn_data_type));

    std::vector<uint32_t> input_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].shape, input_shape), "Cannot get shape");

    std::vector<uint8_t> unpacked_tensor;
    bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
    if (is_constant_input) {
      const auto* input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
      if (1 == input_trans_flag.at(input_i)) {
        RETURN_IF_ERROR(quantize_param.HandleTranspose<size_t>(std::vector<size_t>({1, 0})));
        RETURN_IF_ERROR(
            utils::TwoDimensionTranspose(qnn_model_wrapper, input_shape, input_tensor, unpacked_tensor, logger));
      } else {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_tensor, unpacked_tensor));
      }
    }

    std::string input_tensor_name = input_name;
    if (1 == input_trans_flag.at(input_i) && !is_constant_input) {
      RETURN_IF(quantize_param.IsPerChannel(), "Non-constant Gemm inputs only support per-tensor quantization");

      // Add Transpose node
      std::vector<uint32_t> old_input_shape(input_shape);
      input_shape[0] = old_input_shape[1];
      input_shape[1] = old_input_shape[0];
      const std::string& node_input_name(input_name);
      input_tensor_name = utils::UniqueNameGenerator().New(input_tensor_name, "_transpose");
      std::vector<uint32_t> perm{1, 0};
      RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), node_input_name, input_tensor_name,
                                                         old_input_shape, perm, input_shape,
                                                         qnn_data_type, quantize_param, do_op_validation,
                                                         qnn_model_wrapper.IsGraphInput(node_input_name)));
    }

    // Reshape [1, M] shape Bias.
    if (2 == input_i && 2 == input_shape.size() && input_shape[0] == 1) {
      input_shape[0] = input_shape[1];
      input_shape.resize(1);
    }

    input_names.push_back(input_tensor_name);
    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input_tensor_name);
    QnnTensorWrapper input_tensorwrapper(input_tensor_name, tensor_type, qnn_data_type, std::move(quantize_param),
                                         std::move(input_shape), std::move(unpacked_tensor));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  // Bring FC's inputs to the precision it runs at (see GemmPrecisionPlan); both helpers are no-ops
  // when nothing needs fixing up. The output side is handled by ProcessAttributesAndOutputs.
  GemmPrecisionPlan plan;
  RETURN_IF_ERROR(MakeGemmPrecisionPlan(qnn_model_wrapper, node_unit, plan));
  RETURN_IF_ERROR(DequantizeWeight(qnn_model_wrapper, plan, input_names));
  RETURN_IF_ERROR(FixUpInput(qnn_model_wrapper, plan, input_names, do_op_validation));

  return Ort::Status();
}

Ort::Status GemmOpBuilder::ProcessInputsForBQGemm(QnnModelWrapper& qnn_model_wrapper,
                                                  const OrtNodeUnit& node_unit,
                                                  int64_t trans_b,
                                                  float beta,
                                                  const Ort::Logger& logger,
                                                  std::vector<std::string>& input_names,
                                                  bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  // transA=1 means the ONNX activation is [K, M]; QNN FullyConnected needs [M, K], so we insert a
  // Transpose after the FP16 dequantize.
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t trans_a = node_helper.Get("transA", static_cast<int64_t>(0));

  // Determine weight shape and K, N dimensions.
  // transB=0: B=[K,N], blocked on axis 0; transB=1: B=[N,K], blocked on axis 1.
  TensorInfo weight_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], weight_info));
  RETURN_IF_NOT(weight_info.is_initializer, "QNN EP: BQ Gemm weight must be a constant initializer");
  RETURN_IF_NOT(weight_info.shape.size() == 2, "QNN EP: BQ Gemm weight must be rank-2");

  // QNN FC weight is [N, K]. From ONNX:
  //   transB=0: B=[K,N] → must transpose to [N,K]; K is the blocked axis.
  //   transB=1: B=[N,K] → already [N,K], no transpose needed.
  const int64_t N = (trans_b == 0) ? static_cast<int64_t>(weight_info.shape[1])
                                   : static_cast<int64_t>(weight_info.shape[0]);
  const int64_t K = (trans_b == 0) ? static_cast<int64_t>(weight_info.shape[0])
                                   : static_cast<int64_t>(weight_info.shape[1]);

  //
  // Input A (activation): dequantize INT16→FP16 if needed, then transpose to [M, K] if transA=1.
  // QNN HTP BQ FullyConnected accepts a 2-D [M, K] activation with a 2-D BW_FLOAT_BLOCK weight.
  //
  TensorInfo act_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], act_info));
  RETURN_IF_NOT(act_info.shape.size() == 2,
                "QNN EP: BQ Gemm activation must be rank-2 ([M, K], or [K, M] when transA=1)");

  // Add activation to QNN graph (handles graph-input / already-added cases).
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(inputs[0].name)) {
    QnnTensorWrapper act_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(act_info, inputs[0].name, act_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(act_wrapper)), "Failed to add act tensor.");
  }
  input_names.push_back(inputs[0].name);

  {
    const std::string act_name = input_names[0];
    const auto& act_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(act_name);
    const std::vector<uint32_t> act_shape_2d = act_wrapper.GetTensorDims();

    // BW_FLOAT_BLOCK FC requires FP16 activation; dequantize the INT16 activation to FP16.
    // Reuse the original DequantizeLinear output name for the FP16 tensor so the QNN graph
    // stays aligned with the ONNX graph naming.
    const std::string fp16_name = Ort::ConstNode(&node_unit.GetNode()).GetInputs()[0].GetName();
    RETURN_IF_ERROR(bq::AddInt16ToFp16DequantForActivation(qnn_model_wrapper, act_name,
                                                           fp16_name, do_op_validation, "Gemm"));
    input_names[0] = fp16_name;

    // transA=1: the FP16 activation is [K, M]; transpose to [M, K] for QNN FullyConnected.
    if (trans_a != 0) {
      RETURN_IF_NOT(act_shape_2d.size() == 2, "QNN EP: BQ Gemm transA=1 requires a rank-2 activation");
      const std::vector<uint32_t> transposed_shape = {act_shape_2d[1], act_shape_2d[0]};
      const std::string transposed_name = utils::UniqueNameGenerator().New(fp16_name, "_transpose");
      RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), fp16_name, transposed_name,
                                                         act_shape_2d, /*transpose_perm=*/{1u, 0u},
                                                         transposed_shape, QNN_DATATYPE_FLOAT_16,
                                                         QnnQuantParamsWrapper(), do_op_validation,
                                                         /*is_for_input=*/false, /*is_for_output=*/false));
      input_names[0] = transposed_name;
    }
  }

  //
  // Weight B: orient to [N, K] (QNN FC weight layout), build 2-D BW_FLOAT_BLOCK quant params.
  // QNN HTP BQ FullyConnected accepts a 2-D weight [N, K] with block_sizes={1, block_size}:
  // K is the contraction axis (axis 1), blocked into num_blocks chunks.
  //
  const std::string& weight_name = inputs[1].name;
  const auto scale_shape = utils::GetInitializerShape(inputs[1].quant_param->scale, qnn_model_wrapper.GetOrtApi());
  RETURN_IF_NOT(scale_shape.size() == 2, "QNN EP: BQ Gemm scale must be rank-2");

  // num_blocks is the number of blocks along K.
  const int64_t num_blocks = (trans_b == 0) ? scale_shape[0] : scale_shape[1];
  int64_t block_size = 0;
  RETURN_IF_ERROR(bq::ResolveBlockSize(inputs[1], K, num_blocks, "Gemm", block_size));
  const uint32_t bitwidth = bq::GetBQBitwidth(inputs[1].type);
  RETURN_IF_ERROR(bq::ValidateBQBitwidthAndBlockSize(bitwidth, block_size, "Gemm"));

  // Unpack weight to one byte per element (sub-byte INT2/INT4 expanded to INT8).
  std::vector<uint8_t> unpacked_weight;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(weight_info.initializer_tensor, unpacked_weight));

  // For unsigned types, shift to signed domain.
  const bool is_unsigned = bq::IsUnsignedBQType(inputs[1].type);
  if (is_unsigned) {
    RETURN_IF_ERROR(utils::TransformUnsignedToSignedFixedPoint(unpacked_weight,
                                                               static_cast<int64_t>(bitwidth)));
  }

  // Transpose B to [N, K] if transB=0 (ONNX B is [K, N]).
  if (trans_b == 0) {
    std::vector<uint32_t> kn_shape = {static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
    std::vector<uint8_t> transposed;
    RETURN_IF_ERROR(utils::TwoDimensionTranspose<uint8_t>(unpacked_weight, kn_shape, transposed,
                                                          logger, do_op_validation));
    unpacked_weight = std::move(transposed);
  }
  // transB=1: B=[N,K] already in the right layout.

  // block_sizes for 2-D weight [N, K]: K is axis 1, so block_sizes = {1, block_size}.
  const std::vector<uint32_t> block_size_arr = {1u, static_cast<uint32_t>(block_size)};

  // Scales/offsets must be in [N, num_blocks] order (N-major = row-major for [N,K] weight):
  //   transB=0: ONNX scale=[num_blocks, N] → transpose to [N, num_blocks].
  //   transB=1: ONNX scale=[N, num_blocks] → already correct.
  std::vector<float> onnx_scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(inputs[1].quant_param->scale, onnx_scales));
  RETURN_IF_NOT(static_cast<int64_t>(onnx_scales.size()) == N * num_blocks,
                "QNN EP: BQ Gemm scale size mismatch");

  // Float offsets: unsigned_bias - onnx_zp (matching Conv BQ convention).
  std::vector<float> onnx_offsets;
  RETURN_IF_ERROR(bq::ComputeBQOffsets(qnn_model_wrapper, inputs[1].quant_param->zero_point,
                                       is_unsigned, bitwidth, N * num_blocks, onnx_offsets));

  std::vector<float> scales_qnn, offsets_qnn;
  if (trans_b == 0) {
    // Transpose from [num_blocks, N] to [N, num_blocks].
    const std::vector<uint32_t> transpose_shape = {static_cast<uint32_t>(num_blocks), static_cast<uint32_t>(N)};
    RETURN_IF_ERROR(utils::TwoDimensionTranspose<float>(onnx_scales, transpose_shape, scales_qnn, logger));
    RETURN_IF_ERROR(utils::TwoDimensionTranspose<float>(onnx_offsets, transpose_shape, offsets_qnn, logger));
  } else {
    scales_qnn = std::move(onnx_scales);
    offsets_qnn = std::move(onnx_offsets);
  }

  QnnQuantParamsWrapper bq_quant_params =
      QnnQuantParamsWrapper::BwFloatBlock(gsl::span<const float>(scales_qnn),
                                          gsl::span<const float>(offsets_qnn),
                                          bitwidth,
                                          gsl::span<const uint32_t>(block_size_arr));

  // 2-D weight [N, K] with BW_FLOAT_BLOCK encoding.
  std::vector<uint32_t> weight_shape_2d = {static_cast<uint32_t>(N), static_cast<uint32_t>(K)};
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(weight_name);
  QnnTensorWrapper bq_weight_wrapper(weight_name, tensor_type,
                                     QNN_DATATYPE_SFIXED_POINT_8,
                                     std::move(bq_quant_params),
                                     std::move(weight_shape_2d),
                                     std::move(unpacked_weight));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bq_weight_wrapper)),
                "Failed to add BQ Gemm weight tensor.");
  input_names.push_back(weight_name);

  //
  // Input C (bias): must be an INT32-quantized initializer; dequantize to FP16 for the BQ kernel.
  // If beta=0.0, skip bias (existing convention). Float bias is not yet supported.
  //
  if (inputs.size() == 3 && beta != 0.0f) {
    TensorInfo bias_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));
    RETURN_IF(!bias_info.is_initializer, "QNN EP: BQ Gemm bias must be a constant initializer");

    std::vector<uint32_t> bias_shape = bias_info.shape;
    // Collapse [1, N]→[N] (the existing Gemm convention for bias).
    if (bias_shape.size() == 2 && bias_shape[0] == 1) {
      bias_shape = {bias_shape[1]};
    }

    const std::string fp16_bias_name = utils::UniqueNameGenerator().New(inputs[2].name, "_fp16");
    std::vector<uint8_t> fp16_bias_bytes(static_cast<size_t>(N) * sizeof(uint16_t));

    RETURN_IF_NOT(bias_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_32,
                  "QNN EP: BQ Gemm bias must be INT32-quantized; float bias is not yet supported");
    // Dequantize INT32 bias to FP16 using per-tensor or per-channel scale.
    std::vector<uint8_t> raw_bias_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, raw_bias_bytes));
    std::vector<float> bias_scales;
    if (inputs[2].quant_param.has_value() && inputs[2].quant_param->scale != nullptr) {
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(inputs[2].quant_param->scale, bias_scales));
    }
    // The dequantization below assumes a symmetric (zero-point == 0) bias, which is the convention
    // for INT32 QDQ bias (bias_scale = input_scale * weight_scale, zp = 0). A non-zero zero-point
    // would require subtracting it before scaling; reject it rather than silently mis-dequantizing.
    if (inputs[2].quant_param.has_value() && inputs[2].quant_param->zero_point != nullptr) {
      std::vector<int32_t> bias_zps;
      ONNXTensorElementDataType bias_zp_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(inputs[2].quant_param->zero_point, bias_zps, bias_zp_type));
      for (const int32_t zp : bias_zps) {
        RETURN_IF(zp != 0, "QNN EP: BQ Gemm bias must use zero-point 0 (symmetric); non-zero is not supported");
      }
    }
    RETURN_IF_NOT(raw_bias_bytes.size() == static_cast<size_t>(N) * sizeof(int32_t),
                  "QNN EP: BQ Gemm INT32 bias size mismatch");
    const bool is_per_channel_bias = bias_scales.size() == static_cast<size_t>(N);
    const auto* i32_ptr = reinterpret_cast<const int32_t*>(raw_bias_bytes.data());
    auto* u16_ptr = reinterpret_cast<uint16_t*>(fp16_bias_bytes.data());
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
      const float scale = bias_scales.empty() ? 1.0f : (is_per_channel_bias ? bias_scales[i] : bias_scales[0]);
      const Ort::Float16_t fp16(static_cast<float>(i32_ptr[i]) * scale);
      memcpy(&u16_ptr[i], &fp16.val, sizeof(uint16_t));
    }

    QnnTensorWrapper fp16_bias_wrapper(fp16_bias_name, QNN_TENSOR_TYPE_STATIC,
                                       QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                       std::move(bias_shape),
                                       std::move(fp16_bias_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp16_bias_wrapper)),
                  "Failed to add FP16 bias tensor for BQ Gemm.");
    input_names.push_back(fp16_bias_name);
  }

  return Ort::Status();
}

Ort::Status GemmOpBuilder::AddFcChain(QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& node_unit,
                                      const OrtNodeAttrHelper& node_helper,
                                      const GemmPrecisionPlan& plan,
                                      std::vector<std::string>&& input_names,
                                      const std::string& chain_output_name,
                                      bool split_gemm,
                                      const Ort::Logger& logger,
                                      bool do_op_validation) const {
  // APP_READ only when the chain ends on the node unit's own graph output; a generated name (the
  // intermediate an output fix-up consumes, or a pre-bias FC result) is NATIVE.
  const Qnn_TensorType_t chain_output_tensor_type = qnn_model_wrapper.GetTensorType(chain_output_name);

  // MatMulAddFusion post-Gemm Reshape absorbed by the QDQ selector: FC (rank-2) followed by a QNN
  // Reshape (rank-N). node_unit.Outputs()[0] already reflects the Reshape's rank-N shape and the
  // terminal Q's encoding, so the output fix-up (if any) is appended after the Reshape.
  if (node_unit.GetOutputReshapeNode() != nullptr) {
    // Derive FC's rank-2 output shape [M, N] from the rank-2 input activation [M, K] and the
    // weight [K, N] (transB=0 asserted for MatMulAddFusion pattern).
    RETURN_IF_NOT(plan.in_act.shape.size() == 2,
                  "QNN EP: absorbed-Reshape Gemm input activation must be rank-2 [M, K].");
    RETURN_IF_NOT(plan.weight.shape.size() == 2, "QNN EP: absorbed-Reshape Gemm weight must be rank-2 [K, N].");
    RETURN_IF_NOT(node_helper.Get("transB", static_cast<int64_t>(0)) == 0,
                  "QNN EP: absorbed-Reshape Gemm only supports transB=0.");
    RETURN_IF_NOT(node_helper.Get("transA", static_cast<int64_t>(0)) == 0,
                  "QNN EP: absorbed-Reshape Gemm only supports transA=0.");
    // fc_output_shape = [M, N]: with transA=0 the input activation is [M, K] so shape[0] == M.
    const std::vector<uint32_t> fc_output_shape{plan.in_act.shape[0], plan.weight.shape[1]};

    const std::string fc_output_name = utils::UniqueNameGenerator().New(chain_output_name, "_fc");
    RETURN_IF_ERROR(AddFcChainLink(qnn_model_wrapper, node_unit, plan, QNN_OP_FULLY_CONNECTED,
                                   std::move(input_names), fc_output_name, QNN_TENSOR_TYPE_NATIVE,
                                   fc_output_shape, do_op_validation));
    return AddFcChainLink(qnn_model_wrapper, node_unit, plan, QNN_OP_RESHAPE, {fc_output_name},
                          chain_output_name, chain_output_tensor_type, plan.out_act.shape, do_op_validation);
  }

  if (split_gemm) {
    // The pre-bias FC result carries the FC chain's type and encoding: a quantized intermediate
    // needs a defined encoding, and the trailing ElementWiseAdd re-encodes to the same grid.
    const std::string fc_output_name = utils::UniqueNameGenerator().New(chain_output_name, "_fc");
    RETURN_IF_ERROR(AddFcChainLink(qnn_model_wrapper, node_unit, plan, QNN_OP_FULLY_CONNECTED,
                                   {input_names[0], input_names[1]}, fc_output_name, QNN_TENSOR_TYPE_NATIVE,
                                   plan.out_act.shape, do_op_validation));
    return AddFcChainLink(qnn_model_wrapper, node_unit, plan, QNN_OP_ELEMENT_WISE_ADD,
                          {fc_output_name, input_names[2]}, chain_output_name, chain_output_tensor_type,
                          plan.out_act.shape, do_op_validation);
  }

  if (plan.FixesUpOutput()) {
    // FC feeds the output fix-up, so it emits at the FC chain's type/encoding, not the ONNX output's.
    return AddFcChainLink(qnn_model_wrapper, node_unit, plan, QNN_OP_FULLY_CONNECTED, std::move(input_names),
                          chain_output_name, chain_output_tensor_type, plan.out_act.shape, do_op_validation);
  }

  // Nothing to fix up: the chain is a lone FC emitting the ONNX output tensor as-is.
  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), {}, logger, do_op_validation,
                        GetQnnOpType(node_unit.OpType()));
}

Ort::Status GemmOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  auto beta = node_helper.Get("beta", (float)1.0);
  const int64_t trans_b_out = node_helper.Get("transB", static_cast<int64_t>(0));

  // Detect BQ (BW_FLOAT_BLOCK) Gemm using IsBQGemmWeight, consistent with ProcessInputs detection.
  // BQ Gemm→FC: activation stays 2-D, weight is 2-D [N,K] with BW_FLOAT_BLOCK.
  // FC outputs FP16 → re-quantize to INT16.
  if (IsBQGemmWeight(qnn_model_wrapper, node_unit.Inputs()[1], trans_b_out)) {
    const std::string& org_output_name = node_unit.Outputs()[0].name;
    TensorInfo output_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
    const std::vector<uint32_t>& output_shape = output_info.shape;  // [M, N]
    RETURN_IF_NOT(output_shape.size() == 2, "QNN EP: BQ Gemm output must be rank-2 [M, N]");
    RETURN_IF_NOT(output_info.quant_param.IsQuantized(),
                  "QNN EP: BQ Gemm output must be INT16-quantized; float output is not yet supported");

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
    const Qnn_TensorType_t out_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

    // FullyConnected → 2-D FP16 intermediate tensor, then Quantize to INT16.
    // Reuse the original QuantizeLinear input name for the FP16 tensor so the QNN graph
    // stays aligned with the ONNX graph naming.
    const std::string fc_fp16_out = Ort::ConstNode(&node_unit.GetNode()).GetOutputs()[0].GetName();
    QnnTensorWrapper fp16_wrapper(fc_fp16_out, QNN_TENSOR_TYPE_NATIVE,
                                  QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp16_wrapper)),
                  "Failed to add FP16 BQ Gemm FC output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                                  std::move(input_names), {fc_fp16_out},
                                                  {}, do_op_validation),
                  "Failed to add BQ FullyConnected node.");
    RETURN_IF_ERROR(bq::AddFp16ToInt16QuantizeOutput(qnn_model_wrapper,
                                                     fc_fp16_out, org_output_name,
                                                     out_tensor_type, output_info.qnn_data_type,
                                                     output_info.quant_param.Copy(),
                                                     output_shape, do_op_validation));
    return Ort::Status();
  }

  GemmPrecisionPlan plan;
  RETURN_IF_ERROR(MakeGemmPrecisionPlan(qnn_model_wrapper, node_unit, plan));

  const std::string& org_output_name = node_unit.Outputs()[0].name;

  // With an output fix-up the FC chain feeds it under an intermediate name; without one the chain
  // produces the ONNX output tensor itself.
  const std::string fc_chain_output_name = plan.FixesUpOutput()
                                               ? utils::UniqueNameGenerator().New(org_output_name, "_pre_convert")
                                               : org_output_name;

  // QNN FC folds the bias in itself, except when the bias shape or tensor type rules it out; then the
  // FC chain grows a trailing ElementWiseAdd.
  bool split_gemm = false;
  if (node_unit.Inputs().size() == 3 && beta != 0.0f) {
    auto& input_c = node_unit.Inputs()[2];
    std::vector<uint32_t> input_c_shape;
    QnnModelWrapper::GetOnnxShape(input_c.shape, input_c_shape);

    // Split when input_c has 2d shape and not [1, M]
    split_gemm = (input_c_shape.size() == 2 && input_c_shape.at(0) != 1);

    // Split when bias is an intermediate (NATIVE) tensor produced by another op.
    // ORT's MatMulAddFusion can fuse MatMul+Add->Gemm where the Add's other input
    // is an intermediate tensor (e.g., output of another MatMul). QNN FC requires
    // bias to be either STATIC (constant) or APP_WRITE (graph input), not NATIVE.
    split_gemm = split_gemm || qnn_model_wrapper.GetTensorType(input_c.name) == QNN_TENSOR_TYPE_NATIVE;
  }

  RETURN_IF_ERROR(AddFcChain(qnn_model_wrapper, node_unit, node_helper, plan, std::move(input_names),
                             fc_chain_output_name, split_gemm, logger, do_op_validation));

  return FixUpOutput(qnn_model_wrapper, plan, fc_chain_output_name, org_output_name, do_op_validation);
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GemmOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
