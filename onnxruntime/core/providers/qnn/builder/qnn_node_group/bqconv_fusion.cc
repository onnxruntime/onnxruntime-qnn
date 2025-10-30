// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

// Helper function to transform quantized data from uint4 to signed int4
static void TransformToSignedFixedPoint4(std::vector<uint8_t>& quant_data, int64_t num_blocks, int64_t block_size) {
  for (int32_t block_idx = 0; block_idx < gsl::narrow_cast<int32_t>(num_blocks); ++block_idx) {
    uint32_t zero_point = 8;  // For symmetric quantization, zero_point is 8 for uint4
    for (int32_t val_idx = 0; val_idx < (gsl::narrow_cast<int32_t>(block_size) / 2); ++val_idx) {
      SafeInt<int32_t> safe_index = block_idx;
      safe_index *= (gsl::narrow_cast<int32_t>(block_size) / 2);
      safe_index += val_idx;

      int32_t index = static_cast<int32_t>(safe_index);
      uint8_t quant_value_4x2 = quant_data[index];

      // Convert from uint4 (0-15) to signed int4 (-8 to 7)
      int8_t quant_upper_value = gsl::narrow_cast<int8_t>(((quant_value_4x2 >> 4) & 0xF) - zero_point);
      int8_t quant_lower_value = gsl::narrow_cast<int8_t>(((quant_value_4x2 >> 0) & 0xF) - zero_point);

      quant_data[index] = ((quant_upper_value & 0xF) << 4) | (quant_lower_value & 0xF);
    }
  }
}

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& w_dql_node_unit,
                                    const NodeUnit& conv_node_unit,
                                    bool validate,
                                    const logging::Logger& logger);
std::unique_ptr<IQnnNodeGroup> BlockQuantizedConvFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& conv_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  LOGS(logger, VERBOSE) << "BlockQuantizedConvFusion::TryFusion: Checking Conv node: " << conv_node_unit.Name();

  // Only HTP supports W4BQ encoding format
  // Looking for a Conv to start search for Conv w/ W4BQ encodings pattern.
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) || conv_node_unit.OpType() != "Conv") {
    LOGS(logger, VERBOSE) << "BlockQuantizedConvFusion::TryFusion: Skipping - not NPU backend or not Conv";
    return nullptr;
  }

  // Only attempt fusion when Conv is in NHWC domain (after layout transformation)
  // First pass: Conv is NCHW - skip fusion, layout transform runs
  // Second pass: Conv is NHWC - attempt fusion with correct NHWC shapes
  if (conv_node_unit.Domain() != kMSInternalNHWCDomain) {
    LOGS(logger, VERBOSE) << "BlockQuantizedConvFusion::TryFusion: Skipping - Conv not in NHWC domain yet (waiting for layout transformation)";
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
    LOGS(logger, VERBOSE) << "BlockQuantizedConvFusion::TryFusion: No DQ node found on weight input";
    return nullptr;
  }

  LOGS(logger, VERBOSE) << "BlockQuantizedConvFusion::TryFusion: Found DQ node on weight: " << p_w_dql_node_unit->Name();

  // Check if the weight DequantizeLinear has W4BQ encoding (block quantization)
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
    LOGS(logger, VERBOSE) << "BlockQuantizedConvFusion::TryFusion: Not a W4 quantization";
    return nullptr;
  }

  LOGS(logger, VERBOSE) << "BlockQuantizedConvFusion::TryFusion: Detected W4BQ pattern, validating...";

  // Validate the 2-node pattern on QNN
  if (Status status = CreateOrValidateOnQnn(qnn_model_wrapper,
                                            *p_w_dql_node_unit,
                                            conv_node_unit,
                                            true,
                                            logger);
      !status.IsOK()) {
    LOGS(logger, WARNING) << "BlockQuantizedConvFusion::TryFusion: Validation failed: " << status.ErrorMessage();
    return nullptr;
  }

  LOGS(logger, INFO) << "BlockQuantizedConvFusion: Successfully fused DQ(" << p_w_dql_node_unit->Name()
                     << ") + Conv(" << conv_node_unit.Name() << ") with W4BQ encoding";

  return std::make_unique<BlockQuantizedConvFusion>(*p_w_dql_node_unit,
                                                    conv_node_unit);
}

BlockQuantizedConvFusion::BlockQuantizedConvFusion(const NodeUnit& W_DQL_node_unit,
                                                   const NodeUnit& Conv_node_unit)
    : node_units_{&W_DQL_node_unit,
                  &Conv_node_unit} {
}

Status BlockQuantizedConvFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], true, logger);
}

Status BlockQuantizedConvFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], false, logger);
}

gsl::span<const NodeUnit* const> BlockQuantizedConvFusion::GetNodeUnits() const {
  return node_units_;
}

const NodeUnit* BlockQuantizedConvFusion::GetTargetNodeUnit() const {
  return node_units_[1];  // Conv is at index 1: [W_DQL, Conv]
}

Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& w_dql_node_unit,
                             const NodeUnit& conv_node_unit,
                             bool validate,
                             const logging::Logger& logger) {
  // Validate node types
  ORT_RETURN_IF_NOT(w_dql_node_unit.OpType() == "DequantizeLinear",
                    "Expected DequantizeLinear for weight, got ", w_dql_node_unit.OpType());
  ORT_RETURN_IF_NOT(conv_node_unit.OpType() == "Conv",
                    "Expected Conv operation, got ", conv_node_unit.OpType());

  LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Starting, validate=" << validate;

  const auto& node_name = utils::GetUniqueName(conv_node_unit);
  const NodeUnitIODef& w_dql_input_0_def = w_dql_node_unit.Inputs()[0];  // Quantized weight
  const NodeUnitIODef& conv_input_0_def = conv_node_unit.Inputs()[0];    // Activation input (float from DQ)
  const NodeUnitIODef& conv_output_def = conv_node_unit.Outputs()[0];    // Conv output

  std::vector<std::string> input_names;  // Track input names for the final Conv node

  // Input 0: Activation tensor with fp32 → fp16 Cast
  {
    const auto& input0_name = conv_input_0_def.node_arg.Name();
    LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Processing activation input: " << input0_name;

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input0_name)) {
      QnnTensorWrapper input_tensorwrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(conv_input_0_def, input_tensorwrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }

    // Add Cast op to convert activation from fp32 to fp16
    const std::string cast_to_fp16_name = utils::GetUniqueName(input0_name, "_cast_fp16");
    std::vector<uint32_t> input0_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(conv_input_0_def.node_arg, input0_shape),
                      "Cannot get activation shape");

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(cast_to_fp16_name)) {
      QnnTensorWrapper cast_fp16_tensorwrapper(cast_to_fp16_name,
                                               QNN_TENSOR_TYPE_NATIVE,
                                               QNN_DATATYPE_FLOAT_16,
                                               QnnQuantParamsWrapper(),
                                               std::vector<uint32_t>(input0_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_fp16_tensorwrapper)),
                        "Failed to add fp16 cast tensor.");
      LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Created fp16 cast tensor: " << cast_to_fp16_name;

      // Create Cast node fp32 → fp16
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
          utils::GetUniqueName(node_name, "_cast_to_fp16"),
          QNN_OP_PACKAGE_NAME_QTI_AISW,
          QNN_OP_CAST,
          {input0_name},
          {cast_to_fp16_name},
          {},
          validate),
          "Failed to create Cast node for fp32→fp16");
      LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Created Cast node fp32→fp16";
    }

    // Use fp16 tensor as Conv input
    input_names.push_back(cast_to_fp16_name);
    LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Activation input processed with fp16 cast";
  }

  // Input 1: W4BQ Weight tensor
  {
    const std::string& input1_name = w_dql_input_0_def.node_arg.Name();
    LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Processing weight input: " << input1_name;

    TensorInfo input_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(w_dql_input_0_def, input_info));
    LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Got weight tensor info, is_initializer=" << input_info.is_initializer
                          << ", qnn_data_type=" << input_info.qnn_data_type;

    // Get W4BQ quantization parameters
    const std::optional<NodeUnitIODef::QuantParam>& w_dql_quant_param = w_dql_input_0_def.quant_param;
    ORT_RETURN_IF_NOT(w_dql_quant_param.has_value(), "Weight DequantizeLinear must have quantization parameters");

    // Get block size from DequantizeLinear attributes
    NodeAttrHelper w_dql_helper(w_dql_node_unit.GetNode());
    auto block_size = w_dql_helper.Get("block_size", static_cast<int64_t>(64));

    // Validate it's 4-bit quantization
    bool is_int4_type = false;
    if (w_dql_quant_param->zero_point != nullptr) {
      int32_t elem_data_type = 0;
      ORT_RETURN_IF_ERROR(utils::GetOnnxTensorElemDataType(*w_dql_quant_param->zero_point, elem_data_type));
      is_int4_type = (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_INT4) ||
                     (elem_data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT4);
    }
    ORT_RETURN_IF_NOT(is_int4_type, "Only 4-bit block quantization is supported");

    // Validate zero points are symmetric (all zeros for W4BQ)
    if (w_dql_quant_param->zero_point != nullptr) {
      std::vector<uint8_t> zero_points_data;
      const auto& zp_tensor_proto = qnn_model_wrapper.GetConstantTensor(w_dql_quant_param->zero_point->Name());
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*zp_tensor_proto, zero_points_data));

      // For W4BQ, zero points should all be 0 (symmetric quantization)
      for (size_t i = 0; i < zero_points_data.size(); i++) {
        ORT_RETURN_IF_NOT(zero_points_data[i] == 0,
                          "Only symmetric quantization (zero_point=0) is supported for W4BQ Conv");
      }
    }

    // Calculate total elements and number of blocks using the raw shape
    size_t total_elements = 1;
    for (size_t i = 0; i < input_info.shape.size(); ++i) {
      total_elements *= input_info.shape[i];
    }
    const int64_t num_blocks = (total_elements + block_size - 1) / block_size;

    // Extract per-block float scales from DQ node
    std::vector<uint8_t> scales_uint8_data;
    const auto& scales_tensor_proto = qnn_model_wrapper.GetConstantTensor(w_dql_quant_param->scale.Name());
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*scales_tensor_proto, scales_uint8_data));

    ORT_RETURN_IF_NOT(scales_uint8_data.size() == static_cast<size_t>(num_blocks * sizeof(float)),
                      "Scale tensor size mismatch");

    float* scales_float_ptr = reinterpret_cast<float*>(scales_uint8_data.data());
    const std::vector<float> per_block_float_scales(scales_float_ptr, scales_float_ptr + num_blocks);

    // Create offsets (all zeros for symmetric quantization
    std::vector<int32_t> per_block_int32_offsets(num_blocks, 0);

    // Determine the actual name (following conv_op_builder.cc pattern)
    std::string actual_name = input_info.is_initializer ? input1_name : utils::GetUniqueName(input1_name, "_transpose");

    // Get the original ONNX weight shape (OIHW format) from the DQL input
    std::vector<uint32_t> original_weight_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(w_dql_input_0_def.node_arg, original_weight_shape),
                      "Cannot get original weight shape");
    LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Original weight shape size=" << original_weight_shape.size();

    // Transform shape from OIHW to HWCN format (following conv_op_builder pattern)
    std::vector<uint32_t> actual_shape(original_weight_shape.size());  // Pre-allocate with correct size
    ORT_RETURN_IF_ERROR(utils::NchwShapeToHwcn<uint32_t>(original_weight_shape, actual_shape));
    LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Transformed to HWCN shape";

    // Process the weight tensor based on whether it's an initializer or not
    std::vector<uint8_t> unpacked_tensor;
    if (input_info.is_initializer) {
      // Get transposed initializer bytes (following conv_op_builder.cc pattern)
      // Step 1: Unpack the weight data
      const auto& weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(input1_name);
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*weight_tensor_proto, unpacked_tensor));

      // Step 2: Transform quantized data from uint4 to signed int4
      TransformToSignedFixedPoint4(unpacked_tensor, num_blocks, block_size);

      // Step 3: Transpose the weight data from OIHW to HWCN (following conv_op_builder pattern)
      std::vector<uint8_t> transposed_tensor;
      transposed_tensor.resize(unpacked_tensor.size());

      // Convert shape to int64_t for transpose function
      std::vector<int64_t> shape_int64;
      shape_int64.resize(original_weight_shape.size());
      std::transform(original_weight_shape.begin(), original_weight_shape.end(), shape_int64.begin(),
                     [](uint32_t dim) -> int64_t { return static_cast<int64_t>(dim); });

      // Transpose using byte size of 1 (packed int4 in uint8)
      const size_t elem_byte_size = 1;
      ORT_RETURN_IF_ERROR(utils::TransposeFromNchwToHwcn(std::move(shape_int64), elem_byte_size,
                                                         gsl::span<const uint8_t>(unpacked_tensor),
                                                         gsl::span<uint8_t>(transposed_tensor),
                                                         false));

      // Step 4: Create BQ quantization params (following matmulnbits pattern)
      // Use the actual weight datatype from the tensor info
      const std::vector<uint32_t> block_sizes = {1, static_cast<uint32_t>(block_size)};
      QnnQuantParamsWrapper weight_qparams = QnnQuantParamsWrapper(
          per_block_float_scales,
          per_block_int32_offsets,
          block_sizes,
          input_info.qnn_data_type);

      // Step 5: Transpose quantization parameter's axis (following conv_op_builder pattern)
      LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Checking if weight qparams is per-channel";
      if (weight_qparams.IsPerChannel()) {
        LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Weight qparams is per-channel, transposing";
        std::vector<size_t> perm_inv(nchw2hwcn_perm.size());
        ORT_RETURN_IF_ERROR(utils::InvertPerm<size_t>(nchw2hwcn_perm, perm_inv));
        ORT_RETURN_IF_ERROR(weight_qparams.HandleTranspose<size_t>(perm_inv));
        LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Weight qparams transposed successfully";
      }

      // Step 6: Create weight tensor with BQ encoding, HWCN shape and transposed data
      LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Creating weight tensor wrapper with datatype: " << input_info.qnn_data_type;
      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(actual_name);
      QnnTensorWrapper input_tensorwrapper(actual_name, tensor_type, input_info.qnn_data_type,
                                           std::move(weight_qparams),
                                           std::move(actual_shape), std::move(transposed_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
      LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Weight tensor added successfully with datatype: " << input_info.qnn_data_type;
    } else {
      // Non-initializer case: Add transpose node above weight input (following conv_op_builder.cc pattern)
      ORT_RETURN_IF(input_info.quant_param.IsPerChannel(),
                    "Non-constant Conv inputs only support per-tensor quantization");
      bool is_graph_input = qnn_model_wrapper.IsGraphInput(input1_name);

      if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input1_name)) {
        QnnTensorWrapper weight_tensor_wrapper;
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(w_dql_input_0_def, weight_tensor_wrapper));
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor_wrapper)), "Failed to add weight tensor.");
      }

      // Create empty quantization params for now (no block quantization)
      // TODO: Re-enable block quantization when QNN SDK supports it for Conv2d
      QnnQuantParamsWrapper weight_qparams;  // Empty quantization params

      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(conv_node_unit.Index(),
                                                                   input1_name,
                                                                   actual_name,
                                                                   input_info.shape,
                                                                   actual_shape,
                                                                   input_info.qnn_data_type,
                                                                   weight_qparams,
                                                                   validate,
                                                                   is_graph_input,
                                                                   false,
                                                                   false));

      // Create weight tensor using original data type and HWCN shape
      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(actual_name);
      QnnTensorWrapper input_tensorwrapper(actual_name, tensor_type, input_info.qnn_data_type,
                                           std::move(weight_qparams),
                                           std::move(actual_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }

    // Add the weight tensor name to input_names
    input_names.push_back(actual_name);
    LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Weight input processed successfully";
  }

  //
  // Input 2: Bias tensor (optional) - Following conv_op_builder pattern
  //
  const bool has_bias_input = conv_node_unit.Inputs().size() == 3;
  if (has_bias_input) {
    const auto& bias_name = conv_node_unit.Inputs()[2].node_arg.Name();

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(bias_name)) {
      QnnTensorWrapper input_tensorwrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(conv_node_unit.Inputs()[2], input_tensorwrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }

    input_names.push_back(bias_name);
  }

  //
  // Output tensor with fp16 → fp32 Cast - Following conv_op_builder pattern
  //
  const auto& outputs = conv_node_unit.Outputs();
  const auto& output_name = outputs[0].node_arg.Name();
  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].node_arg, output_shape), "Cannot get output shape");

  QnnQuantParamsWrapper output_quantize_param;
  ORT_RETURN_IF_ERROR(output_quantize_param.Init(qnn_model_wrapper, outputs[0]));
  bool is_quantized_tensor = outputs[0].quant_param.has_value();

  const auto* type_proto = outputs[0].node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_tensor, type_proto, qnn_data_type));

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // Create intermediate fp16 output tensor for Conv
  const std::string conv_fp16_output_name = utils::GetUniqueName(output_name, "_conv_fp16");
  QnnTensorWrapper conv_fp16_output_tensorwrapper(conv_fp16_output_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  QNN_DATATYPE_FLOAT_16,
                                                  QnnQuantParamsWrapper(),
                                                  std::vector<uint32_t>(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(conv_fp16_output_tensorwrapper)),
                    "Failed to add Conv fp16 output tensor.");
  LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Created Conv fp16 output tensor: " << conv_fp16_output_name;

  //
  // Process Conv attributes (following conv_op_builder pattern)
  //
  NodeAttrHelper conv_helper(conv_node_unit.GetNode());
  std::vector<std::string> param_tensor_names;

  // Get input and weight shapes for attribute processing
  const auto& input_0 = conv_node_unit.Inputs()[0];
  const auto& input_1 = w_dql_node_unit.Inputs()[0];
  std::vector<uint32_t> input_0_shape;  // NHWC after layout transformation
  std::vector<uint32_t> input_1_shape;  // OIHW original
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_0_shape), "Cannot get shape");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_1.node_arg, input_1_shape), "Cannot get shape");

  // Kernel shape parameter
  std::vector<uint32_t> kernel_shape;
  kernel_shape = conv_helper.Get("kernel_shape", kernel_shape);
  if (kernel_shape.empty()) {  // Infer from weight shape
    kernel_shape.assign(input_1_shape.begin() + 2, input_1_shape.end());
  }

  // Dilations parameter
  std::vector<uint32_t> dilations;
  dilations.assign(kernel_shape.size(), 1);
  dilations = conv_helper.Get("dilations", dilations);
  QnnParamWrapper dilation_paramwrapper(conv_node_unit.Index(), conv_node_unit.Name(), QNN_OP_CONV_2D_PARAM_DILATION,
                                        {SafeInt<uint32_t>(dilations.size())}, std::vector<uint32_t>(dilations));
  param_tensor_names.push_back(dilation_paramwrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(dilation_paramwrapper));

  // Strides parameter
  std::vector<uint32_t> strides;
  strides.assign(kernel_shape.size(), 1);
  strides = conv_helper.Get("strides", strides);
  QnnParamWrapper stride_amount_paramwrapper(conv_node_unit.Index(), conv_node_unit.Name(), QNN_OP_CONV_2D_PARAM_STRIDE,
                                             {SafeInt<uint32_t>(strides.size())}, std::vector<uint32_t>(strides));
  param_tensor_names.push_back(stride_amount_paramwrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(stride_amount_paramwrapper));

  // Pads attribute (following conv_op_builder pattern)
  {
    std::vector<uint32_t> pads;
    pads.assign(kernel_shape.size() * 2, 0);
    pads = conv_helper.Get("pads", pads);
    auto auto_pad = conv_helper.Get("auto_pad", std::string("NOTSET"));
    ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER" && auto_pad != "VALID",
                  "QNN Conv operators do not support 'auto_pad' value: ", auto_pad.c_str());

    if (auto_pad != "NOTSET") {
      auto pad_type = StringToAutoPadType(auto_pad);
      // skip N, C, input0 shape NHWC
      std::vector<uint32_t> input_dims(input_0_shape.begin() + 1, input_0_shape.end() - 1);
      std::vector<uint32_t> output_dims(output_shape.begin() + 1, output_shape.end() - 1);

      for (size_t dim = 0; dim < kernel_shape.size(); ++dim) {
        int64_t pad_head = 0, pad_tail = 0;
        ORT_RETURN_IF_ERROR(onnxruntime::ComputePad(input_dims[dim], strides[dim], kernel_shape[dim],
                                                    dilations[dim], pad_type, pad_head, pad_tail));
        pads[dim] = static_cast<uint32_t>(pad_head);
        pads[kernel_shape.size() + dim] = static_cast<uint32_t>(pad_tail);
      }
    }

    // Rearrange pads from ONNX format [x1_begin, x2_begin, x1_end, x2_end]
    // to QNN format [x1_begin, x1_end, x2_begin, x2_end] (following conv_op_builder pattern)
    {
      auto pads_size = pads.size();
      auto middle_pos = pads_size / 2;
      std::vector<uint32_t> first_half(pads.begin(), pads.begin() + middle_pos);
      for (size_t i = 0; i < middle_pos; ++i) {
        pads[2 * i] = first_half[i];
        pads[2 * i + 1] = pads[middle_pos + i];
      }
    }
    uint32_t pad_size = narrow<uint32_t>(pads.size() / 2);
    QnnParamWrapper pad_amount_paramwrapper(conv_node_unit.Index(), conv_node_unit.Name(), QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
                                            {pad_size, 2}, std::move(pads));
    param_tensor_names.push_back(pad_amount_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_paramwrapper));
  }

  // Group parameter
  const uint32_t group = conv_helper.Get("group", static_cast<uint32_t>(1));
  Qnn_Scalar_t group_qnn_scalar = QNN_SCALAR_INIT;
  group_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  group_qnn_scalar.uint32Value = group;
  QnnParamWrapper group_paramwrapper(conv_node_unit.Index(), conv_node_unit.Name(), QNN_OP_CONV_2D_PARAM_GROUP, group_qnn_scalar);
  param_tensor_names.push_back(group_paramwrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(group_paramwrapper));

  // Create the QNN Conv2d node with fp16 output - use validate parameter
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CONV_2D,
                                                    std::move(input_names),
                                                    {conv_fp16_output_name},
                                                    std::move(param_tensor_names),
                                                    validate),
                    "Failed to add Conv node.");
  LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Created Conv2d node with fp16 output";

  // Create final fp32 output tensor
  QnnTensorWrapper output_tensorwrapper(output_name, tensor_type, qnn_data_type,
                                        std::move(output_quantize_param), std::move(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                    "Failed to add final output tensor.");
  LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Created final fp32 output tensor: " << output_name;

  // Add Cast op to convert Conv output from fp16 to fp32
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
      utils::GetUniqueName(node_name, "_cast_to_fp32"),
      QNN_OP_PACKAGE_NAME_QTI_AISW,
      QNN_OP_CAST,
      {conv_fp16_output_name},
      {output_name},
      {},
      validate),
      "Failed to create Cast node for fp16→fp32");
  LOGS(logger, VERBOSE) << "CreateOrValidateOnQnn: Created Cast node fp16→fp32";

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
