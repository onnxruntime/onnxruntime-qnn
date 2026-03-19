// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_transpose_rank5.h"

#include <gsl/gsl>
#include <optional>
#include <utility>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include <sstream>

#include "core/common/inlined_containers.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr size_t kRank6 = 6;
constexpr size_t kRank5 = 5;
constexpr const char* kOpTypeReshape = "Reshape";
constexpr const char* kOpTypeTranspose = "Transpose";
constexpr const char* kAttrTransposePerm = "perm";

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

/// @brief Get the shape of a tensor from its OrtValueInfo
std::optional<std::vector<int64_t>> GetTensorShape(const OrtApi& ort_api, const OrtValueInfo* value_info) {
  if (value_info == nullptr) {
    return std::nullopt;
  }

  const OrtTypeInfo* type_info = nullptr;
  if (ort_api.GetValueInfoTypeInfo(value_info, &type_info) != nullptr) {
    return std::nullopt;
  }

  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  if (ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info) != nullptr) {
    return std::nullopt;
  }

  size_t dims_count = 0;
  if (ort_api.GetDimensionsCount(tensor_info, &dims_count) != nullptr) {
    return std::nullopt;
  }

  std::vector<int64_t> dims(dims_count);
  if (ort_api.GetDimensions(tensor_info, dims.data(), dims_count) != nullptr) {
    return std::nullopt;
  }

  return dims;
}

/// @brief Get child NodeUnit of specified type, allowing QDQ-wrapped nodes
const OrtNodeUnit* GetChildNodeUnit(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& parent_node_unit,
    const std::string& child_op_type,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtNode& parent_node = parent_node_unit.GetNode();

  ORT_UNUSED_PARAMETER(logger);
  // For QDQ NodeUnits, we need to look at the Q node's output, not the target node's output
  const OrtNode* search_node = &parent_node;
  if (parent_node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup) {
    const auto& q_nodes = parent_node_unit.GetQNodes();
    if (!q_nodes.empty()) {
      search_node = q_nodes[0];  // Use first Q node
    }
  }

  // Search node must have a single child (1 output edge) and must not produce a graph output
  size_t num_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(search_node, &num_outputs), ort_api, nullptr);
  if (num_outputs != 1) {
    return nullptr;
  }

  std::vector<const OrtValueInfo*> outputs(num_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(search_node, outputs.data(), outputs.size()), ort_api, nullptr);

  const OrtValueInfo* output_info = outputs[0];
  bool is_graph_output = false;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_IsGraphOutput(output_info, &is_graph_output), ort_api, nullptr);
  if (is_graph_output) {
    return nullptr;
  }

  // We should have exactly one consumer
  size_t num_consumers = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueNumConsumers(output_info, &num_consumers), ort_api, nullptr);
  if (num_consumers != 1) {
    return nullptr;
  }

  // Get the consumers of this output
  std::vector<const OrtNode*> consumers(num_consumers);
  std::vector<int64_t> input_indices(num_consumers);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueConsumers(output_info, consumers.data(), input_indices.data(), num_consumers), ort_api, nullptr);

  // Get the child node
  const OrtNode* potential_child = consumers[0];
  if (potential_child == nullptr) {
    return nullptr;
  }

  // If the child is a DequantizeLinear, skip it and look at its child (the target op of the next QDQ group)
  if (Ort::ConstNode(potential_child).GetOperatorType() == "DequantizeLinear") {
    // Get DQ node's output
    size_t dq_num_outputs = 0;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(potential_child, &dq_num_outputs), ort_api, nullptr);
    if (dq_num_outputs != 1) {
      return nullptr;
    }

    std::vector<const OrtValueInfo*> dq_outputs(dq_num_outputs);
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(potential_child, dq_outputs.data(), dq_outputs.size()), ort_api, nullptr);

    const OrtValueInfo* dq_output_info = dq_outputs[0];

    // Get consumers of DQ output
    size_t dq_num_consumers = 0;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueNumConsumers(dq_output_info, &dq_num_consumers), ort_api, nullptr);
    if (dq_num_consumers != 1) {
      return nullptr;
    }

    std::vector<const OrtNode*> dq_consumers(dq_num_consumers);
    std::vector<int64_t> dq_input_indices(dq_num_consumers);
    RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueConsumers(dq_output_info, dq_consumers.data(), dq_input_indices.data(), dq_num_consumers), ort_api, nullptr);

    potential_child = dq_consumers[0];
    if (potential_child == nullptr) {
      return nullptr;
    }
  }

  // Check if this node matches the target type
  if (Ort::ConstNode(potential_child).GetOperatorType() != child_op_type) {
    return nullptr;
  }

  // Get the NodeUnit for the child
  const auto child_node_unit_it = node_to_node_unit.find(potential_child);
  if (child_node_unit_it == node_to_node_unit.end()) {
    return nullptr;
  }

  const OrtNodeUnit* child_node_unit = child_node_unit_it->second;

  // Check if child node has already been handled
  if (node_unit_to_qnn_node_group.count(child_node_unit) != 0) {
    return nullptr;
  }

  return child_node_unit;
}

/// @brief Match the pattern: Reshape -> Transpose -> Reshape with rank-6 intermediate tensors
std::optional<std::array<const OrtNodeUnit*, 3>> MatchRank6ToRank5Pattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit* reshape1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  // Validate first Reshape in pattern - allow both SingleNode and QDQGroup
  if (reshape1->OpType() != kOpTypeReshape) {
    return std::nullopt;
  }

  // Get Transpose child (middle node in pattern) - allow both SingleNode and QDQGroup
  const OrtNodeUnit* transpose = GetChildNodeUnit(
      qnn_model_wrapper, *reshape1, kOpTypeTranspose, node_to_node_unit, node_unit_to_qnn_node_group, logger);
  if (transpose == nullptr) {
    return std::nullopt;
  }

  // Get second Reshape child (last node in pattern) - allow both SingleNode and QDQGroup
  const OrtNodeUnit* reshape2 = GetChildNodeUnit(
      qnn_model_wrapper, *transpose, kOpTypeReshape, node_to_node_unit, node_unit_to_qnn_node_group, logger);
  if (reshape2 == nullptr) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 3>{reshape1, transpose, reshape2};
}

/// @brief Validate the pattern conditions and find the unit dimension index
std::optional<size_t> ValidatePatternConditions(
    const OrtNodeUnit* reshape1,
    const OrtNodeUnit* transpose,
    const OrtNodeUnit* reshape2,
    const QnnModelWrapper& qnn_model_wrapper,
    [[maybe_unused]] const Ort::Logger& logger) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Check if reshape shape inputs are constants
  const OrtNodeUnitIODef& reshape1_input_1 = reshape1->Inputs()[1];
  const OrtNodeUnitIODef& reshape2_input_1 = reshape2->Inputs()[1];

  if (!qnn_model_wrapper.IsConstantInput(reshape1_input_1.name)) {
    return std::nullopt;
  }

  if (!qnn_model_wrapper.IsConstantInput(reshape2_input_1.name)) {
    return std::nullopt;
  }

  // Get tensor shapes
  size_t num_reshape1_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&reshape1->GetNode(), &num_reshape1_inputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> reshape1_inputs(num_reshape1_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&reshape1->GetNode(), reshape1_inputs.data(), reshape1_inputs.size()), ort_api, std::nullopt);

  size_t num_reshape1_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape1->GetNode(), &num_reshape1_outputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> reshape1_outputs(num_reshape1_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&reshape1->GetNode(), reshape1_outputs.data(), reshape1_outputs.size()), ort_api, std::nullopt);

  size_t num_transpose_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&transpose->GetNode(), &num_transpose_outputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> transpose_outputs(num_transpose_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&transpose->GetNode(), transpose_outputs.data(), transpose_outputs.size()), ort_api, std::nullopt);

  size_t num_reshape2_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape2->GetNode(), &num_reshape2_outputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> reshape2_outputs(num_reshape2_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&reshape2->GetNode(), reshape2_outputs.data(), reshape2_outputs.size()), ort_api, std::nullopt);

  auto t0_shape = GetTensorShape(ort_api, reshape1_inputs[0]);
  auto t1_shape = GetTensorShape(ort_api, reshape1_outputs[0]);
  auto t2_shape = GetTensorShape(ort_api, transpose_outputs[0]);
  auto t3_shape = GetTensorShape(ort_api, reshape2_outputs[0]);

  if (!t0_shape.has_value() || !t1_shape.has_value() ||
      !t2_shape.has_value() || !t3_shape.has_value()) {
    return std::nullopt;
  }

  // Condition 1: Rank(t1) == Rank(t2) == 6
  if (t1_shape->size() != kRank6 || t2_shape->size() != kRank6) {
    return std::nullopt;
  }

  const auto& t1_dims = *t1_shape;
  const auto& t2_dims = *t2_shape;

  if (t1_dims.empty() || t2_dims.empty()) {
    return std::nullopt;
  }

  // Condition 2: Find a dimension with value 1 that exists at the same index in both t1 and t2
  std::optional<size_t> unit_dim_index;
  for (size_t i = 0; i < kRank6; ++i) {
    if (t1_dims[i] == 1 && t2_dims[i] == 1) {
      unit_dim_index = i;
      break;
    }
  }

  if (!unit_dim_index.has_value()) {
    return std::nullopt;
  }

  // Condition 3: Transpose must leave the unit dimension in place
  OrtNodeAttrHelper transpose_helper(*transpose);
  std::vector<int64_t> perm = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm.size() != kRank6) {
    return std::nullopt;
  }

  if (perm[unit_dim_index.value()] != static_cast<int64_t>(unit_dim_index.value())) {
    return std::nullopt;
  }

  return unit_dim_index;
}

/// @brief Create or validate the QNN nodes with rank-5 tensors
Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    size_t unit_dim_index,
    bool validate,
    const Ort::Logger& logger) {
  const OrtNodeUnit* reshape1 = node_units[0];
  const OrtNodeUnit* transpose = node_units[1];
  const OrtNodeUnit* reshape2 = node_units[2];

  // Get input and output definitions
  const OrtNodeUnitIODef& reshape1_input = reshape1->Inputs()[0];
  const OrtNodeUnitIODef& reshape1_output = reshape1->Outputs()[0];
  const OrtNodeUnitIODef& transpose_output = transpose->Outputs()[0];
  const OrtNodeUnitIODef& reshape2_output = reshape2->Outputs()[0];

  // Get original shapes
  std::vector<uint32_t> t1_dims;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(reshape1_output.shape, t1_dims),
                ("Cannot get shape for " + reshape1_output.name).c_str());

  std::vector<uint32_t> t2_dims;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(transpose_output.shape, t2_dims),
                ("Cannot get shape for " + transpose_output.name).c_str());

  // Create rank-5 shape for t1 (remove unit dimension at unit_dim_index)
  std::vector<uint32_t> t1_rank5_dims;
  t1_rank5_dims.reserve(kRank5);
  for (size_t i = 0; i < t1_dims.size(); ++i) {
    if (i != unit_dim_index) {
      t1_rank5_dims.push_back(static_cast<uint32_t>(t1_dims[i]));
    }
  }

  // Create rank-5 shape for t2 (remove unit dimension at unit_dim_index)
  std::vector<uint32_t> t2_rank5_dims;
  t2_rank5_dims.reserve(kRank5);
  for (size_t i = 0; i < t2_dims.size(); ++i) {
    if (i != unit_dim_index) {
      t2_rank5_dims.push_back(static_cast<uint32_t>(t2_dims[i]));
    }
  }

  // Get transpose permutation and adjust for rank-5
  OrtNodeAttrHelper transpose_helper(*transpose);
  std::vector<int64_t> perm = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm.size() != kRank6) {
    return Ort::Status("Expected rank-6 permutation", OrtErrorCode::ORT_FAIL);
  }

  // Remove unit dimension and adjust indices
  std::vector<uint32_t> perm_rank5;
  perm_rank5.reserve(kRank5);
  for (size_t i = 0; i < perm.size(); ++i) {
    if (i != unit_dim_index) {
      int64_t perm_val = perm[i];
      // Adjust index: if perm_val > unit_dim_index, subtract 1
      if (perm_val > static_cast<int64_t>(unit_dim_index)) {
        perm_val--;
      }
      perm_rank5.push_back(static_cast<uint32_t>(perm_val));
    }
  }

  // Create Reshape1 input tensor wrapper.
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(reshape1_input.name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + reshape1_input.name).c_str());
  } else {
    QnnTensorWrapper input_tensor_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(reshape1_input, input_tensor_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor_wrapper)),
                  "Failed to add the first Reshape's input tensor.");
  }

  // Create Reshape1 output tensor wrapper.
  TensorInfo reshape1_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(reshape1_output, reshape1_output_info));

  QnnTensorWrapper reshape1_output_tensor_wrapper(reshape1_output.name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  reshape1_output_info.qnn_data_type,
                                                  std::move(reshape1_output_info.quant_param),
                                                  std::move(t1_rank5_dims));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape1_output_tensor_wrapper)),
                "Failed to add the first Reshape's output tensor.");

  // Create Reshape1 node.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(reshape1->Name()),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {reshape1_input.name},
                                                {reshape1_output.name},
                                                {},
                                                validate),
                "Failed to add the first Reshape node.");

  // Create Transpose output tensor wrapper.
  TensorInfo transpose_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(transpose_output, transpose_output_info));

  QnnTensorWrapper transpose_output_tensor_wrapper(transpose_output.name,
                                                   QNN_TENSOR_TYPE_NATIVE,
                                                   transpose_output_info.qnn_data_type,
                                                   std::move(transpose_output_info.quant_param),
                                                   std::move(t2_rank5_dims));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(transpose_output_tensor_wrapper)),
                "Failed to add Transpose's output tensor.");

  // Create Transpose perm parameter wrapper.
  QnnParamWrapper perm_param(transpose->Index(),
                             transpose->Name(),
                             QNN_OP_TRANSPOSE_PARAM_PERM,
                             {static_cast<uint32_t>(perm_rank5.size())},
                             std::move(perm_rank5));
  const std::string param_tensor_name = perm_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(perm_param)), "Failed to add Transpose perm param.");

  // Create Transpose node.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(transpose->Name()),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_TRANSPOSE,
                                                {reshape1_output.name},
                                                {transpose_output.name},
                                                {param_tensor_name},
                                                validate),
                "Failed to add Transpose node.");

  // Create Reshape2 output tensor wrapper.
  TensorInfo reshape2_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(reshape2_output, reshape2_output_info));

  Qnn_TensorType_t reshape2_output_tensor_type = qnn_model_wrapper.IsGraphOutput(reshape2_output.name)
                                                     ? QNN_TENSOR_TYPE_APP_READ
                                                     : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper reshape2_output_tensor_wrapper(reshape2_output.name,
                                                  reshape2_output_tensor_type,
                                                  reshape2_output_info.qnn_data_type,
                                                  std::move(reshape2_output_info.quant_param),
                                                  std::move(reshape2_output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape2_output_tensor_wrapper)),
                "Failed to add the second Reshape's output tensor.");

  // Create Reshape2 node.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(reshape2->Name()),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {transpose_output.name},
                                                {reshape2_output.name},
                                                {},
                                                validate),
                "Failed to add the second Reshape node.");

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> Rank6ToRank5Fusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // Match the pattern
  std::optional<std::array<const OrtNodeUnit*, 3>> pattern = MatchRank6ToRank5Pattern(
      qnn_model_wrapper, &reshape1_node_unit, node_to_node_unit, node_unit_to_qnn_node_group, logger);

  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* reshape1 = pattern->at(0);
  const OrtNodeUnit* transpose = pattern->at(1);
  const OrtNodeUnit* reshape2 = pattern->at(2);

  // Validate pattern conditions and get unit dimension index
  auto unit_dim_index = ValidatePatternConditions(reshape1, transpose, reshape2, qnn_model_wrapper, logger);
  if (!unit_dim_index.has_value()) {
    return nullptr;
  }

  // Validate on QNN
  if (!CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), unit_dim_index.value(), /*validate=*/true, logger).IsOK()) {
    return nullptr;
  }

  return std::make_unique<Rank6ToRank5Fusion>(pattern.value(), unit_dim_index.value());
}

gsl::span<const OrtNodeUnit* const> Rank6ToRank5Fusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status Rank6ToRank5Fusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), unit_dim_index_, /*validate=*/true, logger);
}

Ort::Status Rank6ToRank5Fusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), unit_dim_index_, /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
