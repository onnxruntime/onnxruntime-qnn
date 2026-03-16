// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/utils.h"

#include <gsl/gsl>
#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

const OrtNodeUnit* GetOnlyChildOfType(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                      const OrtNodeUnit& parent_node_unit,
                                      gsl::span<const std::string_view> child_op_types,
                                      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Ort::ConstNode parent_node(&parent_node_unit.GetNode());
  std::vector<Ort::ConstValueInfo> outputs = parent_node.GetOutputs();

  // Parent must have a single child and must not produce a graph output.
  if (outputs.size() != 1) {
    return nullptr;
  }
  for (const Ort::ConstValueInfo& output_info : outputs) {
    if (output_info.IsGraphOutput()) {
      return nullptr;
    }
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = outputs[0].GetConsumers();
  if (consumers.size() != 1 || consumers[0].node == nullptr) {
    return nullptr;
  }

  const Ort::ConstNode child_node = consumers[0].node;
  const std::string& child_type = child_node.GetOperatorType();
  bool is_valid_child_type = false;

  for (const auto& valid_op_type : child_op_types) {
    if (valid_op_type == child_type) {
      is_valid_child_type = true;
      break;
    }
  }

  if (!is_valid_child_type) {
    return nullptr;
  }

  const auto child_node_unit_it = node_unit_map.find(child_node);
  if (child_node_unit_it == node_unit_map.end()) {
    return nullptr;
  }
  const OrtNodeUnit* child_node_unit = child_node_unit_it->second;

  // Check if child node has already been handled. Should not be the case if the calling
  // fusion function has been called in topological order, but check to be safe.
  if (qnn_node_group_map.count(child_node_unit) != 0) {
    return nullptr;
  }

  // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (child_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  return child_node_unit;
}

const OrtNodeUnit* GetChildNodeUnitAllowQdq(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& parent_node_unit,
    const std::string& child_op_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtNode& parent_node = parent_node_unit.GetNode();

  // For QDQ NodeUnits, look at the Q node's output instead of the target node's output.
  const OrtNode* search_node = &parent_node;
  if (parent_node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup) {
    const auto& q_nodes = parent_node_unit.GetQNodes();
    if (!q_nodes.empty()) {
      search_node = q_nodes[0];
    }
  }

  // Search node must have a single child and must not produce a graph output.
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

  // Require exactly one consumer.
  size_t num_consumers = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueNumConsumers(output_info, &num_consumers), ort_api, nullptr);
  if (num_consumers != 1) {
    return nullptr;
  }

  std::vector<const OrtNode*> consumers(num_consumers);
  std::vector<int64_t> input_indices(num_consumers);
  RETURN_DEFAULT_IF_API_FAIL(
      ort_api.ValueInfo_GetValueConsumers(output_info, consumers.data(), input_indices.data(), num_consumers),
      ort_api, nullptr);

  const OrtNode* potential_child = consumers[0];
  if (potential_child == nullptr) {
    return nullptr;
  }

  // If the child is a DequantizeLinear, skip it and look at its child.
  if (Ort::ConstNode(potential_child).GetOperatorType() == DEQUANTIZE_LINEAR) {
    size_t dq_num_outputs = 0;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(potential_child, &dq_num_outputs), ort_api, nullptr);
    if (dq_num_outputs != 1) {
      return nullptr;
    }

    std::vector<const OrtValueInfo*> dq_outputs(dq_num_outputs);
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(potential_child, dq_outputs.data(), dq_outputs.size()),
                               ort_api, nullptr);

    const OrtValueInfo* dq_output_info = dq_outputs[0];
    size_t dq_num_consumers = 0;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueNumConsumers(dq_output_info, &dq_num_consumers), ort_api,
                               nullptr);
    if (dq_num_consumers != 1) {
      return nullptr;
    }

    std::vector<const OrtNode*> dq_consumers(dq_num_consumers);
    std::vector<int64_t> dq_input_indices(dq_num_consumers);
    RETURN_DEFAULT_IF_API_FAIL(
        ort_api.ValueInfo_GetValueConsumers(dq_output_info, dq_consumers.data(), dq_input_indices.data(),
                                            dq_num_consumers),
        ort_api, nullptr);

    potential_child = dq_consumers[0];
    if (potential_child == nullptr) {
      return nullptr;
    }
  }

  if (Ort::ConstNode(potential_child).GetOperatorType() != child_op_type) {
    return nullptr;
  }

  const auto child_node_unit_it = node_unit_map.find(potential_child);
  if (child_node_unit_it == node_unit_map.end()) {
    return nullptr;
  }
  const OrtNodeUnit* child_node_unit = child_node_unit_it->second;

  if (qnn_node_group_map.count(child_node_unit) != 0) {
    return nullptr;
  }

  return child_node_unit;
}

std::optional<std::vector<int64_t>> GetInitializerShape(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnitIODef& shape_input) {
  // 1. Require the shape input (eg: Reshape node input[1]) to be a constant initializer.
  if (!qnn_model_wrapper.IsConstantInput(shape_input.name)) {
    return std::nullopt;
  }

  // 2. Get the initializer tensor and ensure it exists.
  const OrtValueInfo* tensor = qnn_model_wrapper.GetConstantTensor(shape_input.name);
  if (tensor == nullptr) {
    return std::nullopt;
  }

  // 3. Fetch type info for the initializer.
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtTypeInfo* type_info = nullptr;
  if (ort_api.GetValueInfoTypeInfo(tensor, &type_info) != nullptr) {
    return std::nullopt;
  }

  // 4. Cast the type info to tensor-specific info.
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  if (ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info) != nullptr) {
    return std::nullopt;
  }

  // 5. Read the tensor element type.
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  if (ort_api.GetTensorElementType(tensor_info, &elem_type) != nullptr) {
    return std::nullopt;
  }

  // 6. Require a 1-D shape tensor.
  size_t dims_count = 0;
  if (ort_api.GetDimensionsCount(tensor_info, &dims_count) != nullptr || dims_count != 1) {
    return std::nullopt;
  }

  // 7. Get the length of the shape vector.
  std::vector<int64_t> dims(dims_count);
  if (ort_api.GetDimensions(tensor_info, dims.data(), dims_count) != nullptr) {
    return std::nullopt;
  }

  // assume dims to be always positive.
  size_t element_count = static_cast<size_t>(dims[0]);

  // 8. Unpack the raw initializer data.
  std::vector<uint8_t> raw_bytes;
  if (!qnn_model_wrapper.UnpackInitializerData(tensor, raw_bytes).IsOK()) {
    return std::nullopt;
  }

  std::vector<int64_t> values;
  values.reserve(element_count);

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    // 9. Validate the byte size for int64 data.
    if (raw_bytes.size() != element_count * sizeof(int64_t)) {
      return std::nullopt;
    }
    const int64_t* data = reinterpret_cast<const int64_t*>(raw_bytes.data());
    values.assign(data, data + element_count);
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    // 10. Validate the byte size for int32 data.
    if (raw_bytes.size() != element_count * sizeof(int32_t)) {
      return std::nullopt;
    }
    const int32_t* data = reinterpret_cast<const int32_t*>(raw_bytes.data());
    for (size_t i = 0; i < element_count; ++i) {
      values.push_back(static_cast<int64_t>(data[i]));
    }
  } else {
    // 11. Reject unsupported element types.
    return std::nullopt;
  }

  return values;
}

const OrtNodeUnit* GetParentOfType(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                   const OrtNodeUnit& child_node_unit,
                                   gsl::span<const std::string_view> parent_op_types,
                                   const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                   const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Ort::ConstNode child_node(&child_node_unit.GetNode());

  for (const Ort::ConstValueInfo& input_info : child_node.GetInputs()) {
    const Ort::ConstNode parent_node = input_info.GetProducerNode().node;
    if (static_cast<const OrtNode*>(parent_node) == nullptr) {
      continue;
    }
    for (const Ort::ConstValueInfo& parent_output_info : parent_node.GetOutputs()) {
      if (parent_output_info.IsGraphOutput()) {
        // Node is producing a graph output
        return nullptr;
      }
    }

    const std::string parent_type = parent_node.GetOperatorType();
    bool is_valid_parent_type = false;

    for (const auto& valid_op_type : parent_op_types) {
      if (valid_op_type == parent_type) {
        is_valid_parent_type = true;
        break;
      }
    }

    if (!is_valid_parent_type) {
      continue;
    }

    const auto parent_node_unit_it = node_unit_map.find(parent_node);
    if (parent_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const OrtNodeUnit* p_parent_node_unit = parent_node_unit_it->second;

    // Check if parent node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(p_parent_node_unit) != 0) {
      return nullptr;
    }

    // parent must not already be part of a QDQ NodeUnit (i.e., be standalone).
    if (p_parent_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return nullptr;
    }

    return p_parent_node_unit;
  }
  return nullptr;
}

const OrtNodeUnit* GetParentOfInput(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                    const OrtNodeUnit& node_unit,
                                    const OrtNodeUnitIODef& input,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const OrtNode* p_child_node = nullptr;

  for (const OrtNode* node : node_unit.GetAllNodesInGroup()) {
    for (const Ort::ConstValueInfo& input_info : Ort::ConstNode(node).GetInputs()) {
      if (input_info.GetName() == input.name) {
        p_child_node = node;
        break;
      }

      if (p_child_node != nullptr) {
        break;
      }
    }
  }

  if (p_child_node == nullptr) {
    return nullptr;
  }

  const Ort::ConstNode child_node(p_child_node);

  for (const Ort::ConstValueInfo& input_info : child_node.GetInputs()) {
    if (input_info.GetName() != input.name) {
      continue;
    }

    const Ort::ConstNode parent_node = input_info.GetProducerNode().node;
    if (static_cast<const OrtNode*>(parent_node) == nullptr) {
      return nullptr;
    }
    for (const Ort::ConstValueInfo& parent_output_info : parent_node.GetOutputs()) {
      if (parent_output_info.IsGraphOutput()) {
        // Node is producing a graph output
        return nullptr;
      }
    }

    const auto parent_node_unit_it = node_unit_map.find(parent_node);
    if (parent_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const OrtNodeUnit* p_parent_node_unit = parent_node_unit_it->second;

    // Check if parent node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(p_parent_node_unit) != 0) {
      return nullptr;
    }

    return p_parent_node_unit;
  }
  return nullptr;
}

const OrtNodeUnit* GetOnlyChildOfOutput(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                        const OrtNodeUnit& node_unit,
                                        const OrtNodeUnitIODef& output,
                                        const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                        const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const OrtNode* p_parent_node = nullptr;

  for (const OrtNode* node : node_unit.GetAllNodesInGroup()) {
    for (const Ort::ConstValueInfo& output_info : Ort::ConstNode(node).GetOutputs()) {
      if (output_info.GetName() == output.name) {
        p_parent_node = node;
        break;
      }
    }
    // Break the loop if producer node of output is found.
    if (p_parent_node != nullptr) {
      break;
    }
  }

  // Return if the given output tensor is not produced by any node in the given node_unit.
  if (p_parent_node == nullptr) {
    return nullptr;
  }

  const Ort::ConstNode parent_node(p_parent_node);

  for (const Ort::ConstValueInfo& parent_output_info : parent_node.GetOutputs()) {
    if (parent_output_info.IsGraphOutput()) {
      // Node is producing a graph output.
      return nullptr;
    }
  }

  for (const Ort::ConstValueInfo& output_info : parent_node.GetOutputs()) {
    // Check if this is the output we're looking for.
    if (output_info.GetName() != output.name) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = output_info.GetConsumers();
    // Check if there is exactly one child.
    // The returned consumer info should not be nullptr node but check to be safe.
    if (consumers.size() != 1 || consumers[0].node == nullptr) {
      return nullptr;
    }

    const Ort::ConstNode child_node = consumers[0].node;
    const auto child_node_unit_it = node_unit_map.find(child_node);
    if (child_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const OrtNodeUnit* p_child_node_unit = child_node_unit_it->second;

    // Check if child node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(p_child_node_unit) != 0) {
      return nullptr;
    }

    return p_child_node_unit;
  }

  return nullptr;
}

const OrtNodeUnit* GetParentOfInputByName(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                          const OrtNodeUnit& node_unit,
                                          const std::string& input_name,
                                          const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                          const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  // Iterate through all nodes in the group
  for (const OrtNode* node : node_unit.GetAllNodesInGroup()) {
    // Check if this node has the input we're looking for
    for (const Ort::ConstValueInfo& input_info : Ort::ConstNode(node).GetInputs()) {
      if (input_info.GetName() != input_name) {
        continue;
      }

      const Ort::ConstNode parent_node = input_info.GetProducerNode().node;

      if (static_cast<const OrtNode*>(parent_node) == nullptr) {
        // Node is not in this graph
        return nullptr;
      }

      for (const Ort::ConstValueInfo& parent_output_info : parent_node.GetOutputs()) {
        if (parent_output_info.IsGraphOutput()) {
          // Node is producing a graph output
          return nullptr;
        }
      }

      const auto parent_node_unit_it = node_unit_map.find(parent_node);
      if (parent_node_unit_it == node_unit_map.end()) {
        return nullptr;
      }
      const OrtNodeUnit* p_parent_node_unit = parent_node_unit_it->second;

      // Check if parent node has already been handled. Should not be the case if the calling
      // fusion function has been called in topological order, but check to be safe.
      if (qnn_node_group_map.count(p_parent_node_unit) != 0) {
        return nullptr;
      }

      // parent must not already be part of a QDQ NodeUnit (i.e., be standalone).
      if (p_parent_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
        return nullptr;
      }

      return p_parent_node_unit;
    }
  }
  return nullptr;
}

}  // namespace qnn
}  // namespace onnxruntime
