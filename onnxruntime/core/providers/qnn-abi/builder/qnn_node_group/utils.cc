#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"

#include <gsl/gsl>
#include <string_view>
#include <unordered_map>

#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

const OrtNodeUnit* GetOnlyChildOfType(const QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& parent_node_unit,
                                      gsl::span<const std::string_view> child_op_types,
                                      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  // const OrtGraph& graph = qnn_model_wrapper.GetOrtGraph();
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtNode& parent_node = parent_node_unit.GetNode();

  // Parent must have a single child (1 output edge) and must not produce a graph output.
  size_t num_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&parent_node, &num_outputs), ort_api, nullptr);
  if (num_outputs != 1) {
    return nullptr;
  }
  std::vector<const OrtValueInfo*> outputs(num_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&parent_node, outputs.data(), outputs.size()), ort_api,
                             nullptr);

  // Check if any of the node's outputs are graph outputs
  const OrtValueInfo* output_info = outputs[0];

  bool is_graph_output = false;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_IsGraphOutput(output_info, &is_graph_output), ort_api, nullptr);

  // We should have exactly one consumer
  size_t num_consumers = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueNumConsumers(output_info, &num_consumers), ort_api, nullptr);
  if (num_consumers != 1) {
    return nullptr;
  }

  // Get the consumers of this output
  std::vector<const OrtNode*> consumers(num_consumers);
  std::vector<int64_t> input_indices(num_consumers);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueConsumers(output_info,
                                                                 consumers.data(),
                                                                 input_indices.data(),
                                                                 num_consumers),
                             ort_api, nullptr);

  // Get the child node
  const OrtNode* child_node_ptr = consumers[0];
  if (child_node_ptr == nullptr) {
    return nullptr;
  }

  // Child must be of a valid type.
  const std::string& child_type = Ort::ConstNode(child_node_ptr).GetOperatorType();
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

  const auto child_node_unit_it = node_unit_map.find(child_node_ptr);
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

const OrtNodeUnit* GetParentOfType(const QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnit& child_node_unit,
                                   gsl::span<const std::string_view> parent_op_types,
                                   const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                   const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtNode& child_node = child_node_unit.GetNode();

  // Get the first input of the child node
  size_t num_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&child_node, &num_inputs), ort_api, nullptr);
  if (num_inputs == 0) {
    return nullptr;
  }

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&child_node, inputs.data(), inputs.size()), ort_api, nullptr);

  const OrtValueInfo* input_info = inputs[0];
  const OrtNode* parent_node = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueProducer(input_info, &parent_node, nullptr), ort_api, nullptr);

  if (parent_node == nullptr) {
    return nullptr;
  }

  // Parent must be of a valid type.
  const std::string& parent_type = Ort::ConstNode(parent_node).GetOperatorType();
  bool is_valid_parent_type = false;

  for (const auto& valid_op_type : parent_op_types) {
    if (valid_op_type == parent_type) {
      is_valid_parent_type = true;
      break;
    }
  }

  if (!is_valid_parent_type) {
    return nullptr;
  }

  const auto parent_node_unit_it = node_unit_map.find(parent_node);
  if (parent_node_unit_it == node_unit_map.end()) {
    return nullptr;
  }
  const OrtNodeUnit* parent_node_unit = parent_node_unit_it->second;

  // Check if parent node has already been handled.
  if (qnn_node_group_map.count(parent_node_unit) != 0) {
    return nullptr;
  }

  // Parent must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (parent_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  return parent_node_unit;
}

const OrtNodeUnit* GetParentOfInput(const QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const OrtNodeUnitIODef& input,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const OrtNode* p_child_node = nullptr;

  // Find which node in the NodeUnit consumes the specified input
  // For SingleNode: just the main node
  // For QDQGroup: could be Q node, target node, or DQ node
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Check the main target node
  const OrtNode& target_node = node_unit.GetNode();
  size_t num_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&target_node, &num_inputs), ort_api, nullptr);
  std::vector<const OrtValueInfo*> inputs(num_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&target_node, inputs.data(), inputs.size()), ort_api, nullptr);

  for (const auto* input_info : inputs) {
    const char* input_name = nullptr;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(input_info, &input_name), ort_api, nullptr);
    if (input_name && input.name == input_name) {
      p_child_node = &target_node;
      break;
    }
  }

  // If not found in target node, check DQ nodes (for QDQ groups)
  if (p_child_node == nullptr && node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup) {
    const auto& dq_nodes = node_unit.GetDQNodes();
    for (const auto* dq_node : dq_nodes) {
      RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(dq_node, &num_inputs), ort_api, nullptr);
      inputs.resize(num_inputs);
      RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(dq_node, inputs.data(), inputs.size()), ort_api, nullptr);

      for (const auto* input_info : inputs) {
        const char* input_name = nullptr;
        RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(input_info, &input_name), ort_api, nullptr);
        if (input_name && input.name == input_name) {
          p_child_node = dq_node;
          break;
        }
      }
      if (p_child_node != nullptr) break;
    }
  }

  if (p_child_node == nullptr) {
    return nullptr;
  }

  const OrtNode& child_node = *p_child_node;

  // Get all inputs of the child node and find the one matching our input
  size_t num_child_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&child_node, &num_child_inputs), ort_api, nullptr);

  std::vector<const OrtValueInfo*> child_inputs(num_child_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&child_node, child_inputs.data(), child_inputs.size()), ort_api, nullptr);

  // Find the input that matches our target input and get its producer
  for (const auto* child_input_info : child_inputs) {
    const char* child_input_name = nullptr;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(child_input_info, &child_input_name), ort_api, nullptr);

    if (child_input_name && input.name == child_input_name) {
      // Found the matching input, get its producer
      const OrtNode* parent_node_ptr = nullptr;
      RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueProducer(child_input_info, &parent_node_ptr, nullptr), ort_api, nullptr);

      if (parent_node_ptr == nullptr) {
        // Node is not valid
        return nullptr;
      }

      const OrtNode& parent_node = *parent_node_ptr;

      // Check if parent produces a graph output
      size_t num_outputs = 0;
      RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&parent_node, &num_outputs), ort_api, nullptr);

      std::vector<const OrtValueInfo*> outputs(num_outputs);
      RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&parent_node, outputs.data(), outputs.size()), ort_api, nullptr);

      bool is_graph_output = false;
      for(auto output_info: outputs){
        RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_IsGraphOutput(output_info, &is_graph_output), ort_api, nullptr);
        if (is_graph_output) {
          return nullptr;
        }
      }

      const auto parent_node_unit_it = node_unit_map.find(&parent_node);
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
  }
  return nullptr;
}

const OrtNodeUnit* GetChildOfOutput(const QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const OrtNodeUnitIODef& output,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  const OrtNode* p_parent_node = nullptr;

  // Find which node in the NodeUnit produces the specified output
  // For SingleNode: just the main node
  // For QDQGroup: could be Q node, target node, or DQ node

  // Check the main target node
  const OrtNode& target_node = node_unit.GetNode();
  size_t num_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&target_node, &num_outputs), ort_api, nullptr);
  std::vector<const OrtValueInfo*> outputs(num_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&target_node, outputs.data(), outputs.size()), ort_api, nullptr);

  for (const auto* output_info : outputs) {
    const char* output_name = nullptr;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(output_info, &output_name), ort_api, nullptr);
    if (output_name && output.name == output_name) {
      p_parent_node = &target_node;
      break;
    }
  }

  // If not found in target node, check Q nodes (for QDQ groups)
  if (p_parent_node == nullptr && node_unit.UnitType() == OrtNodeUnit::Type::QDQGroup) {
    const auto& q_nodes = node_unit.GetQNodes();
    for (const auto* q_node : q_nodes) {
      RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(q_node, &num_outputs), ort_api, nullptr);
      outputs.resize(num_outputs);
      RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(q_node, outputs.data(), outputs.size()), ort_api, nullptr);

      for (const auto* output_info : outputs) {
        const char* output_name = nullptr;
        RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(output_info, &output_name), ort_api, nullptr);
        if (output_name && output.name == output_name) {
          p_parent_node = q_node;
          break;
        }
      }
      if (p_parent_node != nullptr) break;
    }
  }

  if (p_parent_node == nullptr) {
    return nullptr;
  }

  const OrtNode& parent_node = *p_parent_node;

  // Search node must have a single child (1 output edge) and must not produce a graph output
  size_t num_parent_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&parent_node, &num_parent_outputs), ort_api, nullptr);

  std::vector<const OrtValueInfo*> parent_outputs(num_parent_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&parent_node, parent_outputs.data(), parent_outputs.size()), ort_api, nullptr);

  bool is_graph_output = false;

  for(auto output_info: parent_outputs){
    RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_IsGraphOutput(output_info, &is_graph_output), ort_api, nullptr);
    if (is_graph_output) {
      return nullptr;
    }
  }

  // Find the specific output we're looking for to get its consumers
  const OrtValueInfo* target_output_info = nullptr;
  for (const auto* output_info : parent_outputs) {
    const char* output_name = nullptr;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(output_info, &output_name), ort_api, nullptr);
    if (output_name && output.name == output_name) {
      target_output_info = output_info;
      break;
    }
  }

  if (target_output_info == nullptr) {
    return nullptr;
  }

  // Get consumers of the output to iterate through
  size_t num_consumers = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueNumConsumers(target_output_info, &num_consumers), ort_api, nullptr);

  std::vector<const OrtNode*> consumers(num_consumers);
  std::vector<int64_t> input_indices(num_consumers);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueConsumers(target_output_info, consumers.data(), input_indices.data(), num_consumers), ort_api, nullptr);

  for (size_t i = 0; i < num_consumers; ++i) {
    const OrtNode* child_node = consumers[i];

    // Check if this consumer corresponds to the output we're looking for
    size_t num_child_inputs = 0;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(child_node, &num_child_inputs), ort_api, nullptr);

    std::vector<const OrtValueInfo*> child_inputs(num_child_inputs);
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(child_node, child_inputs.data(), child_inputs.size()), ort_api, nullptr);

    bool is_matching_output = false;
    for (const auto* child_input : child_inputs) {
      const char* child_input_name = nullptr;
      RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(child_input, &child_input_name), ort_api, nullptr);
      if (child_input_name && output.name == child_input_name) {
        is_matching_output = true;
        break;
      }
    }

    if (!is_matching_output) {
      continue;
    }

    if (child_node == nullptr) {
      // Node is not valid
      return nullptr;
    }

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

}  // namespace qnn
}  // namespace onnxruntime
