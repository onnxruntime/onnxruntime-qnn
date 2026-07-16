// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gsl/gsl>
#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

std::optional<std::vector<uint32_t>> GetReduceAxes(const QnnModelWrapper& qmw,
                                                   const OrtNodeUnit& node_unit) {
  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  if (!qmw.GetOnnxShape(inputs[0].shape, input_shape)) {
    return std::nullopt;
  }
  const size_t input_rank = input_shape.size();

  std::vector<int64_t> raw_axes;
  OrtNodeAttrHelper node_helper(node_unit);

  const int opset = node_unit.SinceVersion();
  if (opset < 18) {
    // Axes is an attribute.
    raw_axes = node_helper.Get("axes", raw_axes);
  } else if (inputs.size() > 1) {
    // Axes is input[1] initializer.
    const std::string& axes_input_name = inputs[1].name;
    if (!qmw.IsConstantInput(axes_input_name)) {
      return std::nullopt;
    }
    const auto* axes_tensor = qmw.GetConstantTensor(axes_input_name);
    if (!axes_tensor) {
      return std::nullopt;
    }
    std::vector<uint8_t> axes_bytes;
    if (!qmw.UnpackInitializerData(axes_tensor, axes_bytes).IsOK()) {
      return std::nullopt;
    }
    raw_axes.resize(axes_bytes.size() / sizeof(int64_t));
    std::memcpy(raw_axes.data(), axes_bytes.data(), axes_bytes.size());
  }

  // Normalize to positive values.
  std::vector<uint32_t> axes;
  axes.reserve(raw_axes.size());
  for (int64_t ax : raw_axes) {
    int64_t positive_ax = (ax < 0) ? (ax + static_cast<int64_t>(input_rank)) : ax;
    if (positive_ax < 0 || static_cast<size_t>(positive_ax) >= input_rank) {
      return std::nullopt;
    }
    axes.push_back(static_cast<uint32_t>(positive_ax));
  }

  // Sort and deduplicate.
  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

  return axes;
}

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
    const QnnModelWrapper& /*qnn_model_wrapper*/,
    const OrtNodeUnit& parent_node_unit,
    const std::string& child_op_type,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  try {
    const Ort::ConstNode parent_node(&parent_node_unit.GetNode());

    // 1. For QDQ NodeUnits (DQ->op->Q), look at the Q node's output instead of the target node's output.
    const OrtNode* search_node = parent_node;
    if (parent_node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup) {
      const auto& q_nodes = parent_node_unit.GetQNodes();
      if (!q_nodes.empty()) {
        search_node = q_nodes[0];
      }
    }

    // 2. Search node must have a single child and must not produce a graph output.
    const std::vector<Ort::ConstValueInfo> outputs = Ort::ConstNode(search_node).GetOutputs();
    if (outputs.size() != 1 || outputs[0].IsGraphOutput()) {
      return nullptr;
    }

    // 3. Search node must have exactly one consumer.
    const std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = outputs[0].GetConsumers();
    if (consumers.size() != 1 || consumers[0].node == nullptr) {
      return nullptr;
    }

    const OrtNode* potential_child = consumers[0].node;

    // 4. If the child is Q/DQ wrapper(s), skip through them and look at the next math child.
    // Example: DQ -> op -> Q -> DQ -> ...
    while (potential_child != nullptr) {
      const std::string child_op = Ort::ConstNode(potential_child).GetOperatorType();
      if (child_op != QUANTIZE_LINEAR && child_op != DEQUANTIZE_LINEAR) {
        break;
      }

      const std::vector<Ort::ConstValueInfo> qdq_outputs = Ort::ConstNode(potential_child).GetOutputs();
      if (qdq_outputs.size() != 1) {
        return nullptr;
      }

      const std::vector<Ort::ValueInfoConsumerProducerInfo> qdq_consumers = qdq_outputs[0].GetConsumers();
      if (qdq_consumers.size() != 1 || qdq_consumers[0].node == nullptr) {
        return nullptr;
      }

      potential_child = qdq_consumers[0].node;
    }

    // 5. Check if the child node is of the expected type.
    if (Ort::ConstNode(potential_child).GetOperatorType() != child_op_type) {
      return nullptr;
    }

    // 5.1 Check if the child node is not already part of another NodeUnit.
    const auto child_node_unit_it = node_unit_map.find(potential_child);
    if (child_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const OrtNodeUnit* child_node_unit = child_node_unit_it->second;

    if (qnn_node_group_map.count(child_node_unit) != 0) {
      return nullptr;
    }

    return child_node_unit;
  } catch (const Ort::Exception& e) {
    // Treated as "no match" (the fusion matcher bails quietly), but log so a genuine
    // API error is distinguishable from a normal non-match.
    if (OrtLoggingManager::HasDefaultLogger()) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE,
                  (std::string("GetChildNodeUnitAllowQdq: ignoring Ort::Exception during graph walk: ") +
                   e.what())
                      .c_str());
    }
    return nullptr;
  }
}

std::optional<std::vector<int64_t>> GetInitializerDataAsInt64(
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

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t element_count = 0;
  try {
    // 3. Read element type from CXX type wrappers.
    const Ort::ConstValueInfo shape_tensor(tensor);
    const Ort::ConstTensorTypeAndShapeInfo tensor_info = shape_tensor.TypeInfo().GetTensorTypeAndShapeInfo();
    elem_type = tensor_info.GetElementType();

    // 4. Use NodeUnit I/O shape metadata to get the expected number of entries in the shape tensor.
    std::vector<uint32_t> shape_tensor_dims;
    if (!qnn_model_wrapper.GetOnnxShape(shape_input.shape, shape_tensor_dims) || shape_tensor_dims.size() != 1) {
      return std::nullopt;
    }
    element_count = static_cast<size_t>(shape_tensor_dims[0]);
  } catch (const Ort::Exception& e) {
    // Treated as "no match", but log so a genuine API error is not silently swallowed.
    if (OrtLoggingManager::HasDefaultLogger()) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE,
                  (std::string("GetInitializerDataAsInt64: ignoring Ort::Exception while reading shape: ") +
                   e.what())
                      .c_str());
    }
    return std::nullopt;
  }

  // 5. Unpack the raw initializer data.
  std::vector<uint8_t> raw_bytes;
  if (!qnn_model_wrapper.UnpackInitializerData(tensor, raw_bytes).IsOK()) {
    return std::nullopt;
  }

  std::vector<int64_t> values;
  values.reserve(element_count);

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    // 6. Validate the byte size for int64 data.
    if (raw_bytes.size() != element_count * sizeof(int64_t)) {
      return std::nullopt;
    }
    const int64_t* data = reinterpret_cast<const int64_t*>(raw_bytes.data());
    values.assign(data, data + element_count);
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    // 7. Validate the byte size for int32 data.
    if (raw_bytes.size() != element_count * sizeof(int32_t)) {
      return std::nullopt;
    }
    const int32_t* data = reinterpret_cast<const int32_t*>(raw_bytes.data());
    for (size_t i = 0; i < element_count; ++i) {
      values.push_back(static_cast<int64_t>(data[i]));
    }
  } else {
    // 8. Reject unsupported element types.
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
  // Walk every node in the group looking for one that consumes `input_name`, then
  // return the NodeUnit that produces it — subject to the same fusion-safety checks
  // used by the other Get*Parent* helpers (not a graph output, not already fused, standalone).
  for (const OrtNode* node : node_unit.GetAllNodesInGroup()) {
    for (const Ort::ConstValueInfo& input_info : Ort::ConstNode(node).GetInputs()) {
      if (input_info.GetName() != input_name) {
        continue;
      }

      const Ort::ConstNode parent_node = input_info.GetProducerNode().node;

      if (static_cast<const OrtNode*>(parent_node) == nullptr) {
        // input_name is a graph input or initializer — no producer node.
        return nullptr;
      }

      for (const Ort::ConstValueInfo& parent_output_info : parent_node.GetOutputs()) {
        if (parent_output_info.IsGraphOutput()) {
          // Producer also feeds a graph output; fusing it would drop that output.
          return nullptr;
        }
      }

      const auto parent_node_unit_it = node_unit_map.find(parent_node);
      if (parent_node_unit_it == node_unit_map.end()) {
        return nullptr;
      }
      const OrtNodeUnit* p_parent_node_unit = parent_node_unit_it->second;

      // Guard against races when fusion dispatch is not perfectly topological.
      if (qnn_node_group_map.count(p_parent_node_unit) != 0) {
        return nullptr;
      }

      // Only fuse standalone (non-QDQ) nodes — QDQ groups are handled separately.
      if (p_parent_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
        return nullptr;
      }

      return p_parent_node_unit;
    }
  }
  return nullptr;
}

std::optional<float> GetScalarConstantValue(const QnnModelWrapper& qmw,
                                            const std::string& input_name) {
  if (!qmw.IsConstantInput(input_name)) return std::nullopt;
  const OrtValueInfo* vi = qmw.GetConstantTensor(input_name);
  if (!vi) return std::nullopt;
  Ort::ConstValueInfo ort_vi(vi);
  Ort::ConstValue ort_val;
  if (!ort_vi.GetInitializer(ort_val).IsOK()) return std::nullopt;
  auto tensor_info = ort_vi.TypeInfo().GetTensorTypeAndShapeInfo();
  if (tensor_info.GetElementCount() != 1) return std::nullopt;
  const auto elem_type = tensor_info.GetElementType();
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float* data = ort_val.GetTensorData<float>();
    if (!data) return std::nullopt;
    return *data;
  }
  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const Ort::Float16_t* data = ort_val.GetTensorData<Ort::Float16_t>();
    if (!data) return std::nullopt;
    return data->ToFloat();
  }
  return std::nullopt;
}

bool IsScalarConstantApprox(const QnnModelWrapper& qmw,
                            const std::string& input_name,
                            float expected,
                            float tol) {
  const auto val = GetScalarConstantValue(qmw, input_name);
  return val.has_value() && std::abs(*val - expected) <= tol;
}

}  // namespace qnn
}  // namespace onnxruntime
