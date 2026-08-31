// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_bq_utils.h"

#include <string>
#include <string_view>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/inlined_containers.h"

namespace onnxruntime {
namespace qnn {

namespace {
// HTP BQ: supported weight bitwidths (2/4/8) mapped to their block_size divisor
// constraint — block_size must be a multiple of the corresponding value (same as
// the MatMulNBits HTP constraints).
const InlinedHashMap<uint32_t, int64_t> kHtpBQBitsAndBlockSizeMultipliers{
    {2, 16}, {4, 8}, {8, 4}};
}  // namespace

namespace bq {
uint32_t GetBQBitwidth(ONNXTensorElementDataType onnx_type) {
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return 8;
    default:
      return 0;
  }
}

bool IsUnsignedBQType(ONNXTensorElementDataType onnx_type) {
  return onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2 ||
         onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 ||
         onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}

Ort::Status ValidateBQBitwidthAndBlockSize(uint32_t bitwidth, int64_t block_size, std::string_view op_tag) {
  auto bq_it = kHtpBQBitsAndBlockSizeMultipliers.find(bitwidth);
  RETURN_IF(bq_it == kHtpBQBitsAndBlockSizeMultipliers.end(),
            ("QNN HTP " + std::string(op_tag) + " BQ: unsupported weight bitwidth=" +
             std::to_string(bitwidth))
                .c_str());
  RETURN_IF(block_size % bq_it->second != 0,
            ("QNN HTP " + std::string(op_tag) + " BQ: block_size=" + std::to_string(block_size) +
             " must be a multiple of " + std::to_string(bq_it->second) +
             " for " + std::to_string(bitwidth) + "-bit weight")
                .c_str());
  return Ort::Status();
}

Ort::Status ResolveBlockSize(const OrtNodeUnitIODef& weight, int64_t contraction_dim,
                             int64_t num_blocks, std::string_view op_tag,
                             /*out*/ int64_t& block_size) {
  RETURN_IF(num_blocks <= 0 || contraction_dim % num_blocks != 0,
            ("QNN EP: BQ " + std::string(op_tag) +
             ": contraction dim must be a positive multiple of num_blocks")
                .c_str());
  block_size = contraction_dim / num_blocks;

  // Since PR307 the DQ/Q "block_size" attribute is surfaced on quant_param->block_size. When
  // present, cross-check it against the value derived from the scale shape; reject a malformed
  // model where the two disagree. When absent, the derived value stands.
  if (weight.quant_param.has_value() && weight.quant_param->block_size.has_value()) {
    const int64_t attr_block_size = weight.quant_param->block_size.value();
    RETURN_IF(attr_block_size != block_size,
              ("QNN EP: BQ " + std::string(op_tag) + ": block_size attribute (" +
               std::to_string(attr_block_size) + ") disagrees with scale-derived block_size (" +
               std::to_string(block_size) + ")")
                  .c_str());
  }
  return Ort::Status();
}

bool IsBQScale(gsl::span<const int64_t> scale_shape,
               gsl::span<const uint32_t> weight_shape,
               size_t block_axis) {
  if (block_axis >= scale_shape.size() || block_axis >= weight_shape.size()) {
    return false;
  }
  const int64_t num_blocks = scale_shape[block_axis];
  const int64_t weight_dim = static_cast<int64_t>(weight_shape[block_axis]);
  if (num_blocks <= 0 || num_blocks >= weight_dim) {
    return false;
  }
  return weight_dim % num_blocks == 0;
}

Ort::Status ComputeBQOffsets(const QnnModelWrapper& qnn_model_wrapper,
                             const OrtValueInfo* zero_point,
                             bool is_unsigned_weight,
                             uint32_t bitwidth,
                             int64_t count,
                             /*out*/ std::vector<float>& offsets) {
  RETURN_IF(count <= 0, "QNN EP: BQ ComputeBQOffsets: count must be positive");
  // Signed:   offset = -onnx_zp
  // Unsigned: offset = (1 << (bits-1)) - onnx_zp   (compensates the unsigned→signed shift)
  const float unsigned_bias = is_unsigned_weight ? static_cast<float>(1u << (bitwidth - 1)) : 0.0f;
  offsets.assign(static_cast<size_t>(count), unsigned_bias);
  if (zero_point != nullptr) {
    std::vector<int32_t> zp_values;
    ONNXTensorElementDataType zp_onnx_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(zero_point, zp_values, zp_onnx_type));
    RETURN_IF_NOT(static_cast<int64_t>(zp_values.size()) == count,
                  "QNN EP: BQ zero_point size mismatch");
    for (size_t idx = 0; idx < zp_values.size(); ++idx) {
      offsets[idx] = unsigned_bias - static_cast<float>(zp_values[idx]);
    }
  }
  return Ort::Status();
}

Ort::Status AddInt16ToFp16DequantForActivation(QnnModelWrapper& qnn_model_wrapper,
                                               const std::string& act_name,
                                               const std::string& fp16_name,
                                               bool do_op_validation,
                                               std::string_view op_tag) {
  const Qnn_DataType_t act_dtype = qnn_model_wrapper.GetQnnTensorWrapper(act_name).GetTensorDataType();
  // The BW_FLOAT_BLOCK kernels compute in FP16. The only activation dtype reaching this path
  // through the QDQ selector is INT16 (SFIXED or UFIXED), so anything else is unexpected.
  RETURN_IF_NOT(act_dtype == QNN_DATATYPE_SFIXED_POINT_16 || act_dtype == QNN_DATATYPE_UFIXED_POINT_16,
                ("QNN EP: BQ " + std::string(op_tag) +
                 " activation must be INT16-quantized for the BW_FLOAT_BLOCK kernel")
                    .c_str());
  const std::vector<uint32_t> act_shape = qnn_model_wrapper.GetQnnTensorWrapper(act_name).GetTensorDims();
  return qnn_model_wrapper.AddDequantizeNode(act_name, fp16_name,
                                             QNN_DATATYPE_FLOAT_16, act_shape, do_op_validation);
}

Ort::Status AddFp16ToInt16QuantizeOutput(QnnModelWrapper& qnn_model_wrapper,
                                         const std::string& fp16_out_name,
                                         const std::string& int16_out_name,
                                         Qnn_TensorType_t int16_tensor_type,
                                         Qnn_DataType_t int16_qnn_data_type,
                                         QnnQuantParamsWrapper int16_quant_param,
                                         std::vector<uint32_t> output_shape,
                                         bool do_op_validation) {
  return qnn_model_wrapper.AddQuantizeNode(fp16_out_name, int16_out_name,
                                           int16_tensor_type, int16_qnn_data_type,
                                           std::move(int16_quant_param),
                                           std::move(output_shape), do_op_validation);
}

Ort::Status RegisterWeightAsConv1x1Filter(QnnModelWrapper& qnn_model_wrapper,
                                          const std::string& weight_name,
                                          const TensorInfo& weight_info,
                                          std::vector<uint8_t> weight_data,
                                          std::vector<std::string>& input_names) {
  RETURN_IF_NOT(weight_info.shape.size() == 2,
                "LPBQ 1x1 filter lowering requires weight to be rank-2 [K, N]");
  RETURN_IF_NOT(weight_info.quant_param.IsLPBQ(),
                "LPBQ 1x1 filter lowering requires LPBQ quant params");

  const uint32_t K = weight_info.shape[0];
  const uint32_t N = weight_info.shape[1];

  std::vector<uint32_t> conv_weight_shape = {1u, 1u, K, N};
  QnnQuantParamsWrapper conv_weight_quant = weight_info.quant_param.Copy();
  RETURN_IF_ERROR(conv_weight_quant.HandleUnsqueeze<uint32_t>(weight_info.shape, conv_weight_shape));

  const std::string conv_weight_name = utils::UniqueNameGenerator().New(weight_name, "_reshape");

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(weight_name);
  QnnTensorWrapper weight_tensorwrapper(conv_weight_name, tensor_type, weight_info.qnn_data_type,
                                        std::move(conv_weight_quant), std::move(conv_weight_shape),
                                        std::move(weight_data));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensorwrapper)),
                "Failed to add LPBQ 1x1 filter weight tensor.");
  input_names.emplace_back(conv_weight_name);
  return Ort::Status();
}

Ort::Status AddConv2DNodeforBQLowering(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       std::vector<std::string>&& input_names,
                                       const std::string& conv2d_output_name,
                                       const std::vector<uint32_t>& conv2d_output_shape,
                                       Qnn_DataType_t conv2d_output_dtype,
                                       const QnnQuantParamsWrapper& conv2d_output_quant_param,
                                       bool is_graph_output,
                                       bool do_op_validation) {
  // Build Conv2D params: stride=[1,1], pad=[[0,0],[0,0]], dilation=[1,1], group=1.
  std::vector<std::string> param_tensor_names;

  QnnParamWrapper stride_param(node_unit.Index(), node_unit.Name(),
                               QNN_OP_CONV_2D_PARAM_STRIDE, {2}, {1u, 1u});
  param_tensor_names.push_back(stride_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(stride_param));

  QnnParamWrapper pad_param(node_unit.Index(), node_unit.Name(),
                            QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, {2, 2}, {0u, 0u, 0u, 0u});
  param_tensor_names.push_back(pad_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(pad_param));

  QnnParamWrapper dilation_param(node_unit.Index(), node_unit.Name(),
                                 QNN_OP_CONV_2D_PARAM_DILATION, {2}, {1u, 1u});
  param_tensor_names.push_back(dilation_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(dilation_param));

  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 1u,
                                         QNN_OP_CONV_2D_PARAM_GROUP, param_tensor_names));

  const Qnn_TensorType_t conv2d_tensor_type = is_graph_output
                                                  ? QNN_TENSOR_TYPE_APP_READ
                                                  : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper conv2d_output_wrapper(conv2d_output_name, conv2d_tensor_type, conv2d_output_dtype,
                                         conv2d_output_quant_param.Copy(),
                                         std::vector<uint32_t>(conv2d_output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(conv2d_output_wrapper)),
                "Failed to add Conv2D output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONV_2D,
                                                std::move(input_names), {conv2d_output_name},
                                                std::move(param_tensor_names), do_op_validation),
                "Failed to add Conv2D node.");

  return Ort::Status();
}

}  // namespace bq
}  // namespace qnn
}  // namespace onnxruntime
