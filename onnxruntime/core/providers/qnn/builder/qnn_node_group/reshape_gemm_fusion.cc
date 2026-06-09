// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_gemm_fusion.h"

#include <algorithm>
#include <cassert>
#include <gsl/gsl>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_def.h"
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

  // QNN HTP FullyConnected only supports input rank <= 4. The fusion passes the input_reshape's
  // original input directly to FC (bypassing the reshape), so reject rank > 4.
  if (input_dims_count > 4) {
    return false;
  }

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
  return GetChildNodeUnitAllowQdq(qnn_model_wrapper, gemm_node_unit, "Reshape",
                                  node_to_node_unit, node_unit_to_qnn_node_group);
}

// Get the second output Reshape node unit that consumes reshape1's output (reshape2)
const OrtNodeUnit* GetOutputReshape2NodeUnit(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  return GetChildNodeUnitAllowQdq(qnn_model_wrapper, reshape1_node_unit, "Reshape",
                                  node_to_node_unit, node_unit_to_qnn_node_group);
}

// Common validation for Gemm node in fusion patterns.
// Returns true if the Gemm node is valid for fusion, false otherwise.
// Checks: GPU backend skip, standalone Gemm, no transpose, constant non-quantized weight.
bool IsValidGemmForFusion(const QnnModelWrapper& qnn_model_wrapper,
                          const OrtNodeUnit& gemm_node_unit,
                          bool check_quantized_weight = true) {
  // Skip fusion for GPU backend
  if (IsGpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return false;
  }

  // Only handle standalone Gemm nodes (not QDQ-wrapped)
  if (gemm_node_unit.OpType() != "Gemm" || gemm_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return false;
  }

  // Check transA and transB - we only handle the default case (no transpose)
  OrtNodeAttrHelper attr_helper(gemm_node_unit);
  int64_t transA = attr_helper.Get("transA", static_cast<int64_t>(0));
  int64_t transB = attr_helper.Get("transB", static_cast<int64_t>(0));
  if (transA != 0 || transB != 0) {
    return false;
  }

  // Only fuse when alpha == 1.0 and beta == 1.0
  float alpha = attr_helper.Get("alpha", 1.0f);
  float beta = attr_helper.Get("beta", 1.0f);
  if (alpha != 1.0f || beta != 1.0f) {
    return false;
  }

  // Weight must be constant
  const OrtNodeUnitIODef& weight_input = gemm_node_unit.Inputs()[1];
  if (!qnn_model_wrapper.IsConstantInput(weight_input.name)) {
    return false;
  }

  // Optionally check that weight is not quantized (pattern is from MatMul->Add fusion)
  if (check_quantized_weight && weight_input.quant_param.has_value()) {
    return false;
  }

  return true;
}

// Common implementation for creating/validating fused FC on QNN.
// Handles 2-node (no output reshape), 3-node (one output reshape), and 4-node (skip reshape1, use reshape2).
Ort::Status CreateOrValidateFusedFCOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& input_reshape_node_unit,
                                         const OrtNodeUnit& gemm_node_unit,
                                         const OrtNodeUnit* output_reshape_node_unit,
                                         const Ort::Logger& logger,
                                         bool validate) {
  const bool has_output_reshape = output_reshape_node_unit != nullptr;

  // Get input/output definitions
  const OrtNodeUnitIODef& input_def = input_reshape_node_unit.Inputs()[0];
  const OrtNodeUnitIODef& weight_def = gemm_node_unit.Inputs()[1];
  const OrtNodeUnitIODef* bias_def_ptr = gemm_node_unit.Inputs().size() > 2 ? &gemm_node_unit.Inputs()[2] : nullptr;
  const bool has_bias = bias_def_ptr != nullptr;

  // FC output is Gemm's output; final output is reshape's output if present
  const OrtNodeUnitIODef& fc_output_def = gemm_node_unit.Outputs()[0];
  const OrtNodeUnitIODef& final_output_def = has_output_reshape ? output_reshape_node_unit->Outputs()[0] : fc_output_def;

  const std::string fc_node_name = utils::UniqueNameGenerator().New(gemm_node_unit);
  const std::string& fc_output_name = has_output_reshape ? fc_node_name + "_fc_out" : fc_output_def.name;
  const std::string& final_output_name = final_output_def.name;

  // Create input tensor wrapper
  QnnTensorWrapper input_tensor;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));

  // Process weight tensor - need to transpose for FullyConnected
  std::vector<uint32_t> weight_shape;
  std::vector<uint8_t> unpacked_tensor;
  std::string weight_tensor_name = weight_def.name;

  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(weight_def.shape, weight_shape), "Failed to get weight shape");

  // Get tensor type for weight
  Qnn_TensorType_t weight_tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  Qnn_DataType_t weight_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(false, weight_def.type, weight_data_type));

  // Get weight tensor proto and perform 2D transpose
  const auto* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, weight_tensor_proto,
                                               unpacked_tensor, logger, validate));

  QnnTensorWrapper weight_tensor(weight_tensor_name, weight_tensor_type, weight_data_type, QnnQuantParamsWrapper(),
                                 std::move(weight_shape), std::move(unpacked_tensor));

  // Process bias if present
  QnnTensorWrapper bias_tensor;
  if (has_bias) {
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(*bias_def_ptr, bias_tensor));
  }

  // Create FC output tensor
  QnnTensorWrapper fc_output_tensor;
  if (!has_output_reshape) {
    // For 2-node case, use MakeTensorWrapper for output
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(fc_output_def, fc_output_tensor));
  } else {
    // For 3/4-node case, create intermediate tensor
    std::vector<uint32_t> fc_output_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(fc_output_def.shape, fc_output_shape), "Failed to get FC output shape");

    Qnn_DataType_t fc_output_data_type = QNN_DATATYPE_FLOAT_32;
    RETURN_IF_ERROR(utils::GetQnnDataType(fc_output_def.quant_param.has_value(), fc_output_def.type, fc_output_data_type));

    QnnQuantParamsWrapper fc_output_quant_param;
    RETURN_IF_ERROR(fc_output_quant_param.Init(qnn_model_wrapper, fc_output_def));

    fc_output_tensor = QnnTensorWrapper(fc_output_name, QNN_TENSOR_TYPE_NATIVE, fc_output_data_type,
                                        std::move(fc_output_quant_param), std::move(fc_output_shape),
                                        std::vector<uint8_t>());
  }

  // Create final output tensor (reshape output) if we have output reshape
  QnnTensorWrapper final_output_tensor;
  std::string reshape_node_name;
  if (has_output_reshape) {
    reshape_node_name = utils::UniqueNameGenerator().New(*output_reshape_node_unit);

    std::vector<uint32_t> final_output_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(final_output_def.shape, final_output_shape),
                  "Failed to get final output shape");

    Qnn_DataType_t final_output_data_type = QNN_DATATYPE_FLOAT_32;
    RETURN_IF_ERROR(utils::GetQnnDataType(final_output_def.quant_param.has_value(), final_output_def.type,
                                          final_output_data_type));

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(final_output_name);
    Qnn_TensorType_t final_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

    QnnQuantParamsWrapper final_output_quant_param;
    RETURN_IF_ERROR(final_output_quant_param.Init(qnn_model_wrapper, final_output_def));

    final_output_tensor = QnnTensorWrapper(final_output_name, final_output_tensor_type, final_output_data_type,
                                           std::move(final_output_quant_param), std::move(final_output_shape),
                                           std::vector<uint8_t>());
  }

  if (validate) {
    // Validate FC node
    std::vector<Qnn_Tensor_t> fc_input_tensors = {input_tensor.GetQnnTensor(), weight_tensor.GetQnnTensor()};
    if (has_bias) {
      fc_input_tensors.emplace_back(bias_tensor.GetQnnTensor());
    }
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(fc_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_FULLY_CONNECTED, std::move(fc_input_tensors),
                                                      {fc_output_tensor.GetQnnTensor()}, {}));

    // Validate Reshape node if present
    if (has_output_reshape) {
      RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(reshape_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_RESHAPE, {fc_output_tensor.GetQnnTensor()},
                                                        {final_output_tensor.GetQnnTensor()}, {}));
    }
  } else {
    // Add tensors to model
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");
    if (has_bias) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
    }
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fc_output_tensor)), "Failed to add FC output");
    if (has_output_reshape) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)), "Failed to add final output");
    }

    // Create FC input names
    std::vector<std::string> fc_input_names = {input_def.name, weight_tensor_name};
    if (has_bias) {
      fc_input_names.emplace_back(bias_def_ptr->name);
    }

    // Create the QNN FullyConnected node
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(fc_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                        std::move(fc_input_names), {fc_output_name},
                                        {}, validate),
        "Failed to create FullyConnected node.");

    // Create the QNN Reshape node if present
    if (has_output_reshape) {
      RETURN_IF_NOT(
          qnn_model_wrapper.CreateQnnNode(reshape_node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESHAPE,
                                          {fc_output_name}, {final_output_name},
                                          {}, validate),
          "Failed to create Reshape node.");
    }
  }

  return Ort::Status();
}

}  // namespace

// ============================================================================
// ReshapeGemmFusionGroup implementation
// ============================================================================

ReshapeGemmFusionGroup::ReshapeGemmFusionGroup(std::vector<const OrtNodeUnit*> node_units)
    : node_units_(std::move(node_units)) {
}

Ort::Status ReshapeGemmFusionGroup::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, logger, true);
}

Ort::Status ReshapeGemmFusionGroup::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qmw, logger, false);
}

gsl::span<const OrtNodeUnit* const> ReshapeGemmFusionGroup::GetNodeUnits() const {
  return node_units_;
}

const OrtNodeUnit* ReshapeGemmFusionGroup::GetTargetNodeUnit() const {
  return node_units_[1];  // The Gemm node is always at index 1
}

Ort::Status ReshapeGemmFusionGroup::CreateOrValidateOnQnn(QnnModelWrapper& qmw, const Ort::Logger& logger,
                                                          bool validate) const {
  const OrtNodeUnit* input_reshape = node_units_[0];
  const OrtNodeUnit* gemm = node_units_[1];

  // Determine output reshape node based on fusion size
  const OrtNodeUnit* output_reshape = nullptr;
  if (node_units_.size() == 3) {
    // 3-node: use the output reshape directly
    output_reshape = node_units_[2];
  } else if (node_units_.size() == 4) {
    // 4-node: skip reshape1 (index 2), use reshape2 (index 3)
    output_reshape = node_units_[3];
  }

  return CreateOrValidateFusedFCOnQnn(qmw, *input_reshape, *gemm, output_reshape, logger, validate);
}

// 2-node fusion: Reshape -> Gemm
std::unique_ptr<IQnnNodeGroup> ReshapeGemmFusionGroup::TryFusion2(
    QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  if (!IsValidGemmForFusion(qnn_model_wrapper, gemm_node_unit)) {
    return nullptr;
  }

  // Find input Reshape
  const OrtNodeUnit* input_reshape = GetInputReshapeNodeUnit(
      qnn_model_wrapper, gemm_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!input_reshape) {
    return nullptr;
  }

  // Get weight's input channel (K dimension)
  const OrtNodeUnitIODef& weight_input = gemm_node_unit.Inputs()[1];
  int64_t weight_k = GetWeightInputChannel(qnn_model_wrapper, weight_input);
  if (weight_k <= 0) {
    return nullptr;
  }

  // Validate input reshape pattern (ND -> 2D with last dim = weight's K)
  if (!CheckShape(qnn_model_wrapper, input_reshape->GetNode(), weight_k)) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmFusionGroup>(
      std::vector<const OrtNodeUnit*>{input_reshape, &gemm_node_unit});
}

// 3-node fusion: Reshape -> Gemm -> Reshape
std::unique_ptr<IQnnNodeGroup> ReshapeGemmFusionGroup::TryFusion3(
    QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  if (!IsValidGemmForFusion(qnn_model_wrapper, gemm_node_unit)) {
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
  const OrtNodeUnitIODef& weight_input = gemm_node_unit.Inputs()[1];
  int64_t weight_k = GetWeightInputChannel(qnn_model_wrapper, weight_input);
  if (weight_k <= 0) {
    return nullptr;
  }

  // Validate input reshape pattern (ND -> 2D with last dim = weight's K)
  if (!CheckShape(qnn_model_wrapper, input_reshape->GetNode(), weight_k)) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmFusionGroup>(
      std::vector<const OrtNodeUnit*>{input_reshape, &gemm_node_unit, output_reshape});
}

// 4-node fusion: Reshape -> Gemm -> Reshape -> Reshape
std::unique_ptr<IQnnNodeGroup> ReshapeGemmFusionGroup::TryFusion4(
    QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  if (!IsValidGemmForFusion(qnn_model_wrapper, gemm_node_unit)) {
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

  // Reshape1's output must not be a graph output (we need Reshape2 to consume it)
  const OrtNodeUnitIODef& reshape1_output = output_reshape1->Outputs()[0];
  if (qnn_model_wrapper.IsGraphOutput(reshape1_output.name)) {
    return nullptr;
  }

  // Find output Reshape2 (after Reshape1)
  const OrtNodeUnit* output_reshape2 = GetOutputReshape2NodeUnit(
      qnn_model_wrapper, *output_reshape1, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!output_reshape2) {
    return nullptr;
  }

  // Get weight's input channel (K dimension)
  const OrtNodeUnitIODef& weight_input = gemm_node_unit.Inputs()[1];
  int64_t weight_k = GetWeightInputChannel(qnn_model_wrapper, weight_input);
  if (weight_k <= 0) {
    return nullptr;
  }

  // Validate input reshape pattern (ND -> 2D with last dim = weight's K)
  if (!CheckShape(qnn_model_wrapper, input_reshape->GetNode(), weight_k)) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmFusionGroup>(
      std::vector<const OrtNodeUnit*>{input_reshape, &gemm_node_unit, output_reshape1, output_reshape2});
}

}  // namespace qnn
}  // namespace onnxruntime
