#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_node_group/bqconv_fusion.h"

namespace onnxruntime {
namespace qnn {

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& w_dql_node_unit,
                                    const NodeUnit& conv_node_unit,
                                    bool validate);

std::unique_ptr<IQnnNodeGroup> BlockQuantizedConvFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& conv_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Only HTP supports W4BQ encoding format
  // Looking for a Conv to start search for Conv w/ W4BQ encodings pattern.
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) || conv_node_unit.OpType() != "Conv") {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Get DequantizeLinear on Weight (input 1) of Conv node - this should have W4BQ encoding
  const NodeUnit* p_w_dql_node_unit = GetParentOfInput(graph_viewer,
                                                       conv_node_unit,
                                                       conv_node_unit.Inputs()[1],
                                                       node_to_node_unit,
                                                       node_unit_to_qnn_node_group);
  if (p_w_dql_node_unit == nullptr || p_w_dql_node_unit->OpType() != "DequantizeLinear") {
    return nullptr;
  }

  // Check if the weight DequantizeLinear has W4BQ encoding (block quantization)
  // This means it should have per-channel scales and block quantization parameters
  TensorInfo w_dql_tensor_info = {};
  if (Status status = qnn_model_wrapper.GetTensorInfo(p_w_dql_node_unit->Inputs()[0], w_dql_tensor_info);
      !status.IsOK()) {
    return nullptr;
  }

  // Check if weight input is constant initializer and has block quantization
  if (!w_dql_tensor_info.is_initializer || !w_dql_tensor_info.quant_param.IsPerChannel()) {
    return nullptr;
  }

  // Verify this is actually W4BQ by checking the quantization parameters
  const std::optional<NodeUnitIODef::QuantParam>& w_dql_quant_param = p_w_dql_node_unit->Inputs()[0].quant_param;
  if (!w_dql_quant_param.has_value()) {
    return nullptr;
  }

  // Check if it's 4-bit quantization
  bool is_w4_quantization = false;
  if (w_dql_quant_param->zero_point != nullptr) {
    int32_t elem_data_type = 0;
    if (utils::GetOnnxTensorElemDataType(*w_dql_quant_param->zero_point, elem_data_type).IsOK()) {
      is_w4_quantization = (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_INT4) ||
                          (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4);
    }
  }

  if (!is_w4_quantization) {
    return nullptr;
  }

  // Validate the 2-node pattern on QNN
  if (Status status = CreateOrValidateOnQnn(qnn_model_wrapper,
                                            *p_w_dql_node_unit,
                                            conv_node_unit,
                                            true);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<BlockQuantizedConvFusion>(*p_w_dql_node_unit,
                                                    conv_node_unit);
}

BlockQuantizedConvFusion::BlockQuantizedConvFusion(const NodeUnit& W_DQL_node_unit,
                                                   const NodeUnit& Conv_node_unit)
    : node_units_{&W_DQL_node_unit,
                  &Conv_node_unit} {
}

Status BlockQuantizedConvFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], true);
}

Status BlockQuantizedConvFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], false);
}

gsl::span<const NodeUnit* const> BlockQuantizedConvFusion::GetNodeUnits() const {
  return node_units_;
}

const NodeUnit* BlockQuantizedConvFusion::GetTargetNodeUnit() const {
  return node_units_[1];  // Conv is at index 1: [W_DQL, Conv]
}

Status UnpackWeightTensorDataForConv(const QnnModelWrapper& qnn_model_wrapper,
                                     const onnx::TensorProto* weight_tensor_proto,
                                     std::vector<uint32_t>& weight_shape,
                                     int64_t input_channel_axis,
                                     std::vector<uint8_t>& unpacked_tensor) {
  ORT_RETURN_IF_NOT(weight_tensor_proto != nullptr, "Weight tensor proto is null");

  if (input_channel_axis == 0) {
    // Transpose to keep output_channel at index 0;
    // This is needed for proper BQ encoding where output channels must be at dimension 0
    return utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, *weight_tensor_proto, unpacked_tensor);
  } else {
    // No transpose needed, just unpack the initializer data
    return qnn_model_wrapper.UnpackInitializerData(*weight_tensor_proto, unpacked_tensor);
  }
}

Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& w_dql_node_unit,
                             const NodeUnit& conv_node_unit,
                             bool validate) {
  // Validate node types to ensure graceful fallback to CPU if pattern doesn't match
  ORT_RETURN_IF_NOT(w_dql_node_unit.OpType() == "DequantizeLinear",
                    "Expected DequantizeLinear for weight, got ", w_dql_node_unit.OpType());
  ORT_RETURN_IF_NOT(conv_node_unit.OpType() == "Conv",
                    "Expected Conv operation, got ", conv_node_unit.OpType());

  const NodeUnitIODef& w_dql_input_0_def = w_dql_node_unit.Inputs()[0];
  const NodeUnitIODef& conv_input_0_def = conv_node_unit.Inputs()[0];  // Activation input to Conv
  const NodeUnitIODef& conv_output_def = conv_node_unit.Outputs()[0];  // Conv output

  // Prepare input tensor - use existing tensor from DQ output (already FP16)
  QnnTensorWrapper input_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(conv_input_0_def, input_tensor));

  // Prepare W4BQ (4-bit Block Quantized) Weight from the DequantizeLinear node
  const std::optional<NodeUnitIODef::QuantParam>& w_dql_quant_param = w_dql_input_0_def.quant_param;
  ORT_RETURN_IF_NOT(w_dql_quant_param.has_value(), "Weight DequantizeLinear must have quantization parameters");

  // Extract per-channel scales from the W4BQ DequantizeLinear
  std::vector<float> per_channel_float_scale;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(w_dql_quant_param->scale.Name(), per_channel_float_scale));

  // Get block quantization parameters from the DequantizeLinear node attributes
  NodeAttrHelper w_dql_helper(w_dql_node_unit.GetNode());
  auto block_size = w_dql_helper.Get("block_size", static_cast<int64_t>(32));  // Default block size

  // Check if it's 4-bit quantization
  bool is_int4_type = false;
  if (w_dql_quant_param->zero_point != nullptr) {
    int32_t elem_data_type = 0;
    ORT_RETURN_IF_ERROR(utils::GetOnnxTensorElemDataType(*w_dql_quant_param->zero_point, elem_data_type));
    is_int4_type = (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_INT4) ||
                   (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4);
  }

  // Get weight tensor information
  std::vector<uint32_t> weight_shape;
  std::string weight_tensor_name = w_dql_input_0_def.node_arg.Name();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(w_dql_input_0_def.node_arg, weight_shape), "Failed to get weight shape");

  // For W4BQ, we need per-block integer scales - these should be embedded in the quantization parameters
  // For now, create dummy per-block scales (this would need to be extracted from the actual W4BQ encoding)
  size_t num_blocks = (weight_shape[0] + block_size - 1) / block_size;  // Calculate number of blocks
  std::vector<uint8_t> per_block_int_scale(num_blocks, 1);  // Placeholder - should be extracted from W4BQ data

  std::vector<int32_t> weight_offset(per_channel_float_scale.size(), 0);
  size_t output_channel_axis = 0;

  QnnQuantParamsWrapper weight_qparams(per_channel_float_scale, per_block_int_scale, weight_offset,
                                       output_channel_axis, block_size, is_int4_type);

  // Get the quantized weight data directly from the W4BQ tensor
  std::vector<uint8_t> unpacked_tensor;
  const auto& weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*weight_tensor_proto, unpacked_tensor));

  // Use the appropriate data type based on whether it's 4-bit or 8-bit quantization
  Qnn_DataType_t weight_tensor_data_type = is_int4_type ? QNN_DATATYPE_SFIXED_POINT_4 : QNN_DATATYPE_SFIXED_POINT_8;
  Qnn_TensorType_t weight_tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  QnnTensorWrapper weight_tensor(weight_tensor_name, weight_tensor_type, weight_tensor_data_type,
                                 std::move(weight_qparams), std::move(weight_shape),
                                 std::move(unpacked_tensor));

  // Prepare Bias tensor (if exists)
  QnnTensorWrapper bias_tensor;
  const NodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = conv_node_unit.Inputs().size() == 3 && conv_node_unit.Inputs()[2].node_arg.Exists();
  if (has_bias) {
    bias_def_ptr = &conv_node_unit.Inputs()[2];
    std::vector<uint32_t> bias_shape;
    std::string bias_tensor_name = bias_def_ptr->node_arg.Name();
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(bias_def_ptr->node_arg, bias_shape), "Failed to get bias shape");
    Qnn_DataType_t bias_data_type = QNN_DATATYPE_FLOAT_16;  // Use FP16 for bias to match activation
    Qnn_TensorType_t bias_tensor_type = qnn_model_wrapper.GetTensorType(bias_tensor_name);
    const auto& bias_tensor_proto = qnn_model_wrapper.GetConstantTensor(bias_tensor_name);

    // Read bias tensor buffer
    std::vector<uint8_t> unpacked_bias_tensor;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*bias_tensor_proto, unpacked_bias_tensor));

    QnnQuantParamsWrapper bias_qparams;  // No quantization for FP16 bias

    bias_tensor = QnnTensorWrapper(bias_tensor_name, bias_tensor_type, bias_data_type,
                                   std::move(bias_qparams), std::move(bias_shape),
                                   std::move(unpacked_bias_tensor));
  }

  // Prepare Output tensor - use existing tensor wrapper
  QnnTensorWrapper output_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(conv_output_def, output_tensor));

  // Add tensors to model wrapper
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");
  if (has_bias) {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
  }
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");

  // Use the existing Conv op builder to handle the Conv operation with proper parameters
  const auto* conv_op_builder = qnn::GetOpBuilder("Conv");
  ORT_RETURN_IF_NOT(conv_op_builder != nullptr, "Failed to get Conv OpBuilder");

  // Use the Conv op builder to add the node with proper parameters and validation
  return conv_op_builder->AddToModelBuilder(qnn_model_wrapper, conv_node_unit,
                                           logging::LoggingManager::DefaultLogger(), validate);
}
}  // namespace qnn
}  // namespace onnxruntime
