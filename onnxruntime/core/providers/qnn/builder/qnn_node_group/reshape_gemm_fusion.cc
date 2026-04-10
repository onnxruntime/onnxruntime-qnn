// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_gemm_fusion.h"

#include <algorithm>
#include <cassert>
#include <gsl/gsl>
#include <limits>
#include <optional>
#include <string>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

// Get the weight's input channel (K dimension) from Gemm weight shape [K, N]
int64_t GetWeightInputChannel(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnitIODef& weight_def) {
  std::vector<uint32_t> weight_shape;
  if (!qnn_model_wrapper.GetOnnxShape(weight_def.shape, weight_shape)) {
    return -1;
  }
  if (weight_shape.size() != 2) {
    return -1;
  }
  return static_cast<int64_t>(weight_shape[0]);
}

// Check if reshape input is reshapable to [batch, n] where n is Gemm's input channel (weight's K dim).
// QNN FullyConnected requires: input shape [n] or Rank >= 2 reshapable to [batch, n].
bool CheckShape(const QnnModelWrapper& qnn_model_wrapper, const OrtNode& reshape_node, int64_t weight_input_channel) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Get reshape node inputs and outputs
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&reshape_node, &num_inputs), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape_node, &num_outputs), ort_api, false);

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  std::vector<const OrtValueInfo*> outputs(num_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&reshape_node, inputs.data(), inputs.size()), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&reshape_node, outputs.data(), outputs.size()), ort_api, false);

  const OrtValueInfo* input_info = inputs[0];
  const OrtValueInfo* output_info = outputs[0];

  // Get type info for input and output
  const OrtTypeInfo* input_type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(input_info, &input_type_info), ort_api, false);
  const OrtTypeInfo* output_type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(output_info, &output_type_info), ort_api, false);

  if (!input_type_info || !output_type_info) {
    return false;
  }

  // Cast to tensor info
  const OrtTensorTypeAndShapeInfo* input_tensor_info = nullptr;
  const OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(output_type_info, &output_tensor_info), ort_api, false);

  if (!input_tensor_info || !output_tensor_info) {
    return false;
  }

  // Get dimensions
  size_t input_dims_count = 0;
  size_t output_dims_count = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(input_tensor_info, &input_dims_count), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(output_tensor_info, &output_dims_count), ort_api, false);

  // Output must be 2D [batch, n]
  if (output_dims_count != 2) {
    return false;
  }

  std::vector<int64_t> input_dims(input_dims_count);
  std::vector<int64_t> output_dims(output_dims_count);

  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensions(input_tensor_info, input_dims.data(), input_dims_count),
                             ort_api,
                             false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensions(output_tensor_info, output_dims.data(), output_dims_count),
                             ort_api,
                             false);

  // Check output's last dim (n) matches weight's input channel
  if (output_dims[1] != weight_input_channel) {
    return false;
  }

  // Check input is reshapable to [batch, n]: total elements must equal batch * n
  int64_t total_input = 1;
  for (size_t i = 0; i < input_dims_count; ++i) {
    if (input_dims[i] <= 0) {
      return false;
    }
    total_input *= input_dims[i];
  }

  int64_t total_output = output_dims[0] * output_dims[1];
  return total_input == total_output;
}

// Get the input Reshape node unit that feeds into the Gemm node
const OrtNodeUnit* GetInputReshapeNodeUnit(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  const std::array<std::string_view, 1> reshape_types = {"Reshape"};
  return GetParentOfType(qnn_model_wrapper, gemm_node_unit, reshape_types,
                         node_to_node_unit, node_unit_to_qnn_node_group);
}

// Get the output Reshape node unit that consumes the Gemm output
const OrtNodeUnit* GetOutputReshapeNodeUnit(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  const std::array<std::string_view, 1> reshape_types = {"Reshape"};
  return GetOnlyChildOfType(qnn_model_wrapper, gemm_node_unit, reshape_types,
                            node_to_node_unit, node_unit_to_qnn_node_group);
}

// Get the second output Reshape node unit that consumes reshape1's output (reshape2)
const OrtNodeUnit* GetOutputReshape2NodeUnit(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  const std::array<std::string_view, 1> reshape_types = {"Reshape"};
  return GetOnlyChildOfType(qnn_model_wrapper, reshape1_node_unit, reshape_types,
                            node_to_node_unit, node_unit_to_qnn_node_group);
}

Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& reshape_node_unit,
                                  const OrtNodeUnit& gemm_node_unit, const Ort::Logger& logger, bool validate) {
  assert(reshape_node_unit.OpType() == "Reshape" && gemm_node_unit.OpType() == "Gemm");
  const auto& node_name = utils::UniqueNameGenerator().New(gemm_node_unit);
  const OrtNodeUnitIODef& input_def = reshape_node_unit.Inputs()[0];
  const OrtNodeUnitIODef& weight_def = gemm_node_unit.Inputs()[1];
  const OrtNodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = gemm_node_unit.Inputs().size() == 3;
  if (has_bias) {
    bias_def_ptr = &gemm_node_unit.Inputs()[2];
  }
  const OrtNodeUnitIODef& output_def = gemm_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper bias_tensor;
  QnnTensorWrapper output_tensor;

  // Create input tensor wrapper
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));

  // Process weight tensor
  std::vector<uint32_t> weight_shape;
  std::vector<uint8_t> unpacked_tensor;
  std::string weight_tensor_name = weight_def.name;

  // Get weight shape and validate
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(weight_def.shape, weight_shape), "Failed to get weight shape");

  // Get tensor type for weight
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  Qnn_DataType_t data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(false, weight_def.type, data_type));

  // Get weight tensor proto and perform 2D transpose
  const auto* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  // Transpose the weight tensor (2D matrix transpose)
  RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, weight_tensor_proto, unpacked_tensor, logger, validate));
  QnnTensorWrapper weight_tensor(weight_tensor_name, tensor_type, data_type, QnnQuantParamsWrapper(),
                                 std::move(weight_shape), std::move(unpacked_tensor));
  if (has_bias) {
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(*bias_def_ptr, bias_tensor));
  }

  // Create output tensor wrapper
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    std::vector<Qnn_Tensor_t> input_tensors = {input_tensor.GetQnnTensor(), weight_tensor.GetQnnTensor()};
    if (has_bias) {
      input_tensors.emplace_back(bias_tensor.GetQnnTensor());
    }

    // Validate the QNN node
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_FULLY_CONNECTED, std::move(input_tensors),
                                                      {output_tensor.GetQnnTensor()}, {}));
  } else {
    // For creation, add all tensor wrappers to the model
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");

    // Add bias tensor if it exists
    if (has_bias) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
    }

    // Add output tensor
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");

    // Create input names vector
    std::vector<std::string> input_names = {input_def.name, weight_tensor_name};
    if (has_bias) {
      input_names.emplace_back(bias_def_ptr->name);
    }

    // Create the QNN node for fully connected operation
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                        std::move(input_names), {output_def.name}, {}, validate),
        "Failed to add fused Gemm node.");
  }

  return Ort::Status();
}

// For ReshapeGemmReshapeReshapeFusion
Ort::Status CreateOrValidateOnQnn4Node(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& input_reshape_node_unit,
                                       const OrtNodeUnit& fc_node_unit,
                                       const OrtNodeUnit& output_reshape1_node_unit,
                                       const OrtNodeUnit& output_reshape2_node_unit,
                                       const Ort::Logger& logger,
                                       bool validate) {
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(output_reshape1_node_unit);

  const auto& fc_node_name = utils::UniqueNameGenerator().New(fc_node_unit);
  const auto& reshape_node_name = utils::UniqueNameGenerator().New(output_reshape2_node_unit);

  // Get input from the input reshape's input (original ND tensor)
  const OrtNodeUnitIODef& input_def = input_reshape_node_unit.Inputs()[0];
  // Get weight from FC's input[1]
  const OrtNodeUnitIODef& weight_def = fc_node_unit.Inputs()[1];
  // Check for bias
  const OrtNodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = fc_node_unit.Inputs().size() > 2;
  if (has_bias) {
    bias_def_ptr = &fc_node_unit.Inputs()[2];
  }
  // FC output: use Gemm's 2D output shape (QNN FC requires 2D output)
  const OrtNodeUnitIODef& fc_output_def = fc_node_unit.Outputs()[0];
  const std::string fc_output_name = fc_node_name + "_fc_out";

  // Reshape2 output: final output from reshape2
  const OrtNodeUnitIODef& final_output_def = output_reshape2_node_unit.Outputs()[0];
  const std::string& final_output_name = final_output_def.name;

  // Create input tensor wrapper
  QnnTensorWrapper input_tensor;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));

  // Process weight tensor - need to transpose for FullyConnected
  std::vector<uint32_t> weight_shape;
  std::vector<uint8_t> unpacked_tensor;
  std::string weight_tensor_name = weight_def.name;

  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(weight_def.shape, weight_shape), "Failed to get weight shape");

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  Qnn_DataType_t weight_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(weight_def.quant_param.has_value(), weight_def.type, weight_data_type));

  // Get weight tensor and transpose (Gemm weight is [K, N], FC expects [N, K])
  const auto* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, weight_tensor_proto,
                                               unpacked_tensor, logger, validate));

  QnnQuantParamsWrapper weight_quant_param;
  RETURN_IF_ERROR(weight_quant_param.Init(qnn_model_wrapper, weight_def));
  RETURN_IF_ERROR(weight_quant_param.HandleTranspose<uint32_t>(std::vector<uint32_t>({1, 0})));

  QnnTensorWrapper weight_tensor(weight_tensor_name, tensor_type, weight_data_type,
                                 std::move(weight_quant_param), std::move(weight_shape),
                                 std::move(unpacked_tensor));

  // Process bias if present
  QnnTensorWrapper bias_tensor;
  if (has_bias) {
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(*bias_def_ptr, bias_tensor));
  }

  // Create FC output tensor
  std::vector<uint32_t> fc_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(fc_output_def.shape, fc_output_shape), "Failed to get FC output shape");

  Qnn_DataType_t fc_output_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(fc_output_def.quant_param.has_value(), fc_output_def.type, fc_output_data_type));

  QnnQuantParamsWrapper fc_output_quant_param;
  RETURN_IF_ERROR(fc_output_quant_param.Init(qnn_model_wrapper, fc_output_def));

  QnnTensorWrapper fc_output_tensor(fc_output_name, QNN_TENSOR_TYPE_NATIVE, fc_output_data_type,
                                    std::move(fc_output_quant_param), std::move(fc_output_shape),
                                    std::vector<uint8_t>());

  // Create Reshape2 output tensor (final output)
  std::vector<uint32_t> final_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(final_output_def.shape, final_output_shape), "Failed to get final output shape");

  Qnn_DataType_t final_output_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(final_output_def.quant_param.has_value(), final_output_def.type, final_output_data_type));

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  Qnn_TensorType_t final_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  QnnQuantParamsWrapper final_output_quant_param;
  RETURN_IF_ERROR(final_output_quant_param.Init(qnn_model_wrapper, final_output_def));

  QnnTensorWrapper final_output_tensor(final_output_name, final_output_tensor_type, final_output_data_type,
                                       std::move(final_output_quant_param), std::move(final_output_shape),
                                       std::vector<uint8_t>());

  if (validate) {
    // Validate FC node
    std::vector<Qnn_Tensor_t> fc_input_tensors = {input_tensor.GetQnnTensor(), weight_tensor.GetQnnTensor()};
    if (has_bias) {
      fc_input_tensors.emplace_back(bias_tensor.GetQnnTensor());
    }
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(fc_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_FULLY_CONNECTED, std::move(fc_input_tensors),
                                                      {fc_output_tensor.GetQnnTensor()}, {}));

    // Validate Reshape node
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(reshape_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_RESHAPE, {fc_output_tensor.GetQnnTensor()},
                                                      {final_output_tensor.GetQnnTensor()}, {}));
  } else {
    // Add tensors to model
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");
    if (has_bias) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
    }
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fc_output_tensor)), "Failed to add FC output");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)), "Failed to add final output");

    // Create FC input names
    std::vector<std::string> fc_input_names = {input_def.name, weight_tensor_name};
    if (has_bias) {
      fc_input_names.emplace_back(bias_def_ptr->name);
    }

    // Create the QNN FullyConnected node (ND input -> 2D output)
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(fc_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                        std::move(fc_input_names), {fc_output_name},
                                        {}, validate),
        "Failed to create FullyConnected node.");

    // Create the QNN Reshape node (2D -> final shape)
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(reshape_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
                                        {fc_output_name}, {final_output_name},
                                        {}, validate),
        "Failed to create Reshape node.");
  }

  return Ort::Status();
}

// For ReshapeGemmReshapeFusion
Ort::Status CreateOrValidateOnQnn3Node(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& input_reshape_node_unit,
                                       const OrtNodeUnit& fc_node_unit,
                                       const OrtNodeUnit& output_reshape_node_unit,
                                       const Ort::Logger& logger,
                                       bool validate) {
  ORT_UNUSED_PARAMETER(logger);

  const auto& fc_node_name = utils::UniqueNameGenerator().New(fc_node_unit);
  const auto& reshape_node_name = utils::UniqueNameGenerator().New(output_reshape_node_unit);

  // Get input from the input reshape's input (original ND tensor)
  const OrtNodeUnitIODef& input_def = input_reshape_node_unit.Inputs()[0];
  // Get weight from FC's input[1]
  const OrtNodeUnitIODef& weight_def = fc_node_unit.Inputs()[1];
  // Check for bias
  const OrtNodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = fc_node_unit.Inputs().size() > 2;
  if (has_bias) {
    bias_def_ptr = &fc_node_unit.Inputs()[2];
  }
  // FC output: intermediate 2D output
  const OrtNodeUnitIODef& fc_output_def = fc_node_unit.Outputs()[0];
  const std::string fc_output_name = fc_node_name + "_fc_out";

  // Reshape output: final output from output_reshape
  const OrtNodeUnitIODef& final_output_def = output_reshape_node_unit.Outputs()[0];
  const std::string& final_output_name = final_output_def.name;

  // Create input tensor wrapper
  QnnTensorWrapper input_tensor;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));

  // Process weight tensor - need to transpose for FullyConnected
  std::vector<uint32_t> weight_shape;
  std::vector<uint8_t> unpacked_tensor;
  std::string weight_tensor_name = weight_def.name;

  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(weight_def.shape, weight_shape), "Failed to get weight shape");

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  Qnn_DataType_t weight_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(weight_def.quant_param.has_value(), weight_def.type, weight_data_type));

  // Get weight tensor and transpose (Gemm weight is [K, N], FC expects [N, K])
  const auto* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, weight_tensor_proto,
                                               unpacked_tensor, logger, validate));

  QnnQuantParamsWrapper weight_quant_param;
  RETURN_IF_ERROR(weight_quant_param.Init(qnn_model_wrapper, weight_def));
  RETURN_IF_ERROR(weight_quant_param.HandleTranspose<uint32_t>(std::vector<uint32_t>({1, 0})));

  QnnTensorWrapper weight_tensor(weight_tensor_name, tensor_type, weight_data_type,
                                 std::move(weight_quant_param), std::move(weight_shape),
                                 std::move(unpacked_tensor));

  // Process bias if present
  QnnTensorWrapper bias_tensor;
  if (has_bias) {
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(*bias_def_ptr, bias_tensor));
  }

  // Create FC output tensor (2D intermediate)
  std::vector<uint32_t> fc_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(fc_output_def.shape, fc_output_shape), "Failed to get FC output shape");

  Qnn_DataType_t fc_output_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(fc_output_def.quant_param.has_value(), fc_output_def.type, fc_output_data_type));

  QnnQuantParamsWrapper fc_output_quant_param;
  RETURN_IF_ERROR(fc_output_quant_param.Init(qnn_model_wrapper, fc_output_def));

  QnnTensorWrapper fc_output_tensor(fc_output_name, QNN_TENSOR_TYPE_NATIVE, fc_output_data_type,
                                    std::move(fc_output_quant_param), std::move(fc_output_shape),
                                    std::vector<uint8_t>());

  // Create Reshape output tensor (final output)
  std::vector<uint32_t> final_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(final_output_def.shape, final_output_shape), "Failed to get final output shape");

  Qnn_DataType_t final_output_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(final_output_def.quant_param.has_value(), final_output_def.type, final_output_data_type));

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
  Qnn_TensorType_t final_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  QnnQuantParamsWrapper final_output_quant_param;
  RETURN_IF_ERROR(final_output_quant_param.Init(qnn_model_wrapper, final_output_def));

  QnnTensorWrapper final_output_tensor(final_output_name, final_output_tensor_type, final_output_data_type,
                                       std::move(final_output_quant_param), std::move(final_output_shape),
                                       std::vector<uint8_t>());

  if (validate) {
    // Validate FC node
    std::vector<Qnn_Tensor_t> fc_input_tensors = {input_tensor.GetQnnTensor(), weight_tensor.GetQnnTensor()};
    if (has_bias) {
      fc_input_tensors.emplace_back(bias_tensor.GetQnnTensor());
    }
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(fc_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_FULLY_CONNECTED, std::move(fc_input_tensors),
                                                      {fc_output_tensor.GetQnnTensor()}, {}));

    // Validate Reshape node
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(reshape_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_RESHAPE, {fc_output_tensor.GetQnnTensor()},
                                                      {final_output_tensor.GetQnnTensor()}, {}));
  } else {
    // Add tensors to model
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");
    if (has_bias) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
    }
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fc_output_tensor)), "Failed to add FC output");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)), "Failed to add final output");

    // Create FC input names
    std::vector<std::string> fc_input_names = {input_def.name, weight_tensor_name};
    if (has_bias) {
      fc_input_names.emplace_back(bias_def_ptr->name);
    }

    // Create the QNN FullyConnected node (ND input -> 2D output)
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(fc_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                        std::move(fc_input_names), {fc_output_name},
                                        {}, validate),
        "Failed to create FullyConnected node.");

    // Create the QNN Reshape node (2D -> final shape)
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(reshape_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
                                        {fc_output_name}, {final_output_name},
                                        {}, validate),
        "Failed to create Reshape node.");
  }

  return Ort::Status();
}

}  // namespace

// ============================================================================
// ReshapeGemmReshapeFusion: 2-node fusion (Reshape -> Gemm)
// ============================================================================
std::unique_ptr<IQnnNodeGroup> ReshapeGemmFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Only handle standalone Gemm nodes (not QDQ-wrapped)
  if (gemm_node_unit.OpType() != "Gemm" || gemm_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Check transA and transB - we only handle the default case (no transpose)
  OrtNodeAttrHelper attr_helper(gemm_node_unit);
  int64_t transA = attr_helper.Get("transA", static_cast<int64_t>(0));
  int64_t transB = attr_helper.Get("transB", static_cast<int64_t>(0));
  if (transA != 0 || transB != 0) {
    return nullptr;
  }

  // Weight must be constant and not quantized (pattern is from MatMul->Add fusion)
  const OrtNodeUnitIODef& weight_input = gemm_node_unit.Inputs()[1];
  if (!qnn_model_wrapper.IsConstantInput(weight_input.name) || weight_input.quant_param.has_value()) {
    return nullptr;
  }

  // Find input Reshape
  const OrtNodeUnit* input_reshape = GetInputReshapeNodeUnit(
      qnn_model_wrapper, gemm_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!input_reshape) {
    return nullptr;
  }

  // Get weight's input channel (K dimension)
  int64_t weight_k = GetWeightInputChannel(qnn_model_wrapper, weight_input);
  if (weight_k <= 0) {
    return nullptr;
  }

  // Validate input reshape pattern (ND -> 2D with last dim = weight's K)
  if (!CheckShape(qnn_model_wrapper, input_reshape->GetNode(), weight_k)) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmFusion>(*input_reshape, gemm_node_unit);
}

ReshapeGemmFusion::ReshapeGemmFusion(const OrtNodeUnit& reshape_node_unit, const OrtNodeUnit& gemm_node_unit)
    : node_units_{&reshape_node_unit, &gemm_node_unit} {
}

Ort::Status ReshapeGemmFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], logger, true);
}

Ort::Status ReshapeGemmFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], logger, false);
}

gsl::span<const OrtNodeUnit* const> ReshapeGemmFusion::GetNodeUnits() const {
  return node_units_;
}

const OrtNodeUnit* ReshapeGemmFusion::GetTargetNodeUnit() const {
  return node_units_[1];
}

// ============================================================================
// ReshapeGemmReshapeFusion: 3-node fusion (Reshape -> Gemm -> Reshape)
// ============================================================================

std::unique_ptr<IQnnNodeGroup> ReshapeGemmReshapeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Only handle standalone Gemm nodes (not QDQ-wrapped)
  if (gemm_node_unit.OpType() != "Gemm" || gemm_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Check transA and transB - we only handle the default case (no transpose)
  OrtNodeAttrHelper attr_helper(gemm_node_unit);
  int64_t transA = attr_helper.Get("transA", static_cast<int64_t>(0));
  int64_t transB = attr_helper.Get("transB", static_cast<int64_t>(0));
  if (transA != 0 || transB != 0) {
    return nullptr;
  }

  // Weight must be constant
  const OrtNodeUnitIODef& weight_input = gemm_node_unit.Inputs()[1];
  if (!qnn_model_wrapper.IsConstantInput(weight_input.name)) {
    return nullptr;
  }

  // Find input Reshape
  const OrtNodeUnit* input_reshape = GetInputReshapeNodeUnit(
      qnn_model_wrapper, gemm_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!input_reshape) {
    return nullptr;
  }

  // Find output Reshape (after Gemm)
  const OrtNodeUnit* output_reshape = GetOutputReshapeNodeUnit(
      qnn_model_wrapper, gemm_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!output_reshape) {
    return nullptr;
  }
  // Get weight's input channel (K dimension)
  int64_t weight_k = GetWeightInputChannel(qnn_model_wrapper, weight_input);
  if (weight_k <= 0) {
    return nullptr;
  }
  // Validate input reshape pattern (ND -> 2D with last dim = weight's K)
  if (!CheckShape(qnn_model_wrapper, input_reshape->GetNode(), weight_k)) {
    return nullptr;
  }
  return std::make_unique<ReshapeGemmReshapeFusion>(*input_reshape, gemm_node_unit, *output_reshape);
}

ReshapeGemmReshapeFusion::ReshapeGemmReshapeFusion(const OrtNodeUnit& input_reshape_node_unit,
                                                   const OrtNodeUnit& gemm_node_unit,
                                                   const OrtNodeUnit& output_reshape_node_unit)
    : node_units_{&input_reshape_node_unit, &gemm_node_unit, &output_reshape_node_unit} {
}

Ort::Status ReshapeGemmReshapeFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn3Node(qmw, *node_units_[0], *node_units_[1], *node_units_[2], logger, true);
}

Ort::Status ReshapeGemmReshapeFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn3Node(qmw, *node_units_[0], *node_units_[1], *node_units_[2], logger, false);
}

gsl::span<const OrtNodeUnit* const> ReshapeGemmReshapeFusion::GetNodeUnits() const {
  return node_units_;
}

const OrtNodeUnit* ReshapeGemmReshapeFusion::GetTargetNodeUnit() const {
  return node_units_[1];  // The Gemm node is the target
}

// ============================================================================
// ReshapeGemmReshapeReshapeFusion: 4-node fusion (Reshape -> Gemm -> Reshape -> Reshape)
// ============================================================================

std::unique_ptr<IQnnNodeGroup> ReshapeGemmReshapeReshapeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Only handle Gemm nodes (MatMul+Add gets fused to Gemm)
  if (gemm_node_unit.OpType() != "Gemm" || gemm_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Check transA and transB - we only handle the default case (no transpose)
  OrtNodeAttrHelper attr_helper(gemm_node_unit);
  int64_t transA = attr_helper.Get("transA", static_cast<int64_t>(0));
  int64_t transB = attr_helper.Get("transB", static_cast<int64_t>(0));
  if (transA != 0 || transB != 0) {
    return nullptr;
  }

  // Weight must be constant
  const OrtNodeUnitIODef& weight_input = gemm_node_unit.Inputs()[1];
  if (!qnn_model_wrapper.IsConstantInput(weight_input.name)) {
    return nullptr;
  }

  // Find input Reshape
  const OrtNodeUnit* input_reshape = GetInputReshapeNodeUnit(
      qnn_model_wrapper, gemm_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!input_reshape) {
    return nullptr;
  }

  // Find output Reshape1 (after Gemm)
  const OrtNodeUnit* output_reshape1 = GetOutputReshapeNodeUnit(
      qnn_model_wrapper, gemm_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!output_reshape1) {
    return nullptr;
  }

  // Find output Reshape2 (after Reshape1)
  const OrtNodeUnit* output_reshape2 = GetOutputReshape2NodeUnit(
      qnn_model_wrapper, *output_reshape1, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!output_reshape2) {
    return nullptr;
  }

  // Get weight's input channel (K dimension)
  int64_t weight_k = GetWeightInputChannel(qnn_model_wrapper, weight_input);
  if (weight_k <= 0) {
    return nullptr;
  }

  // Validate input reshape pattern (ND -> 2D with last dim = weight's K)
  if (!CheckShape(qnn_model_wrapper, input_reshape->GetNode(), weight_k)) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmReshapeReshapeFusion>(*input_reshape, gemm_node_unit, *output_reshape1, *output_reshape2);
}

ReshapeGemmReshapeReshapeFusion::ReshapeGemmReshapeReshapeFusion(const OrtNodeUnit& input_reshape_node_unit,
                                                                 const OrtNodeUnit& gemm_node_unit,
                                                                 const OrtNodeUnit& output_reshape1_node_unit,
                                                                 const OrtNodeUnit& output_reshape2_node_unit)
    : node_units_{&input_reshape_node_unit, &gemm_node_unit, &output_reshape1_node_unit, &output_reshape2_node_unit} {
}

Ort::Status ReshapeGemmReshapeReshapeFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn4Node(qmw, *node_units_[0], *node_units_[1], *node_units_[2], *node_units_[3], logger, true);
}

Ort::Status ReshapeGemmReshapeReshapeFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn4Node(qmw, *node_units_[0], *node_units_[1], *node_units_[2], *node_units_[3], logger, false);
}

gsl::span<const OrtNodeUnit* const> ReshapeGemmReshapeReshapeFusion::GetNodeUnits() const {
  return node_units_;
}

const OrtNodeUnit* ReshapeGemmReshapeReshapeFusion::GetTargetNodeUnit() const {
  return node_units_[1];  // The Gemm node is the target
}

}  // namespace qnn
}  // namespace onnxruntime
