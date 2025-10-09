// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/gelu_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (root_input), (final_output), true)
#define CreateOnQnn(qnn_model_wrapper, node_units, root_input, final_output) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (root_input), (final_output), false)

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    gsl::span<const NodeUnit* const> node_units,
                                    const NodeUnitIODef& root_input,
                                    const NodeUnitIODef& final_output,
                                    bool validate);

// Helper function to check if an initializer has the expected value
static bool IsInitializerWithExpectedValue(const GraphViewer& graph_viewer,
                                           const NodeArg& node_arg,
                                           float expected_value,
                                           bool check_opset_13_or_higher) {
  if (!graph_viewer.IsConstantInitializer(node_arg.Name(), check_opset_13_or_higher)) {
    return false;
  }

  const auto* initializer = graph_viewer.GetConstantInitializer(node_arg.Name());
  if (!initializer) {
    return false;
  }

  // Check if it's a scalar or single-element tensor
  const auto& dims = initializer->dims();
  if (dims.size() > 0) {
    int64_t num_elements = 1;
    for (int i = 0; i < dims.size(); ++i) {
      num_elements *= dims[i];
    }
    if (num_elements != 1) {
      return false;
    }
  }

  // Get the value based on data type
  float actual_value = 0.0f;
  if (initializer->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const auto& raw_data = initializer->raw_data();
    if (raw_data.size() == sizeof(float) && reinterpret_cast<uintptr_t>(raw_data.data()) % alignof(float) == 0) {
      actual_value = *reinterpret_cast<const float*>(raw_data.data());
    } else {
      return false;
    }
  } else if (initializer->data_type() == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
    const auto& raw_data = initializer->raw_data();
    if (raw_data.size() == sizeof(double) && reinterpret_cast<uintptr_t>(raw_data.data()) % alignof(double) == 0) {
      double double_value = *reinterpret_cast<const double*>(raw_data.data());
      actual_value = static_cast<float>(double_value);
    } else {
      return false;
    }
  } else {
    return false;
  }

  // Compare with tolerance
  constexpr float epsilon = std::numeric_limits<float>::epsilon();
  return std::abs(actual_value - expected_value) <= epsilon * std::max(1.0f, std::abs(expected_value));
}

std::unique_ptr<IQnnNodeGroup> GeluFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& erf_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Looking for a standalone Erf node.
  if (erf_node_unit.OpType() != "Erf" || erf_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Erf must have a Div parent
  const std::array<std::string_view, 1> div_types = {"Div"};
  const NodeUnit* div_node_unit = GetParentOfType(graph_viewer, erf_node_unit, div_types,
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (div_node_unit == nullptr) {
    return nullptr;
  }

  // Check second input of Div is sqrt(2) or approximated sqrt(2)
  const auto& div_inputs = div_node_unit->Inputs();
  if (div_inputs.size() < 2) {
    return nullptr;
  }

  constexpr float approximated_sqrt_two = 1.4142099618911743f;
  constexpr float sqrt_two = static_cast<float>(M_SQRT2);

  bool is_sqrt2 = IsInitializerWithExpectedValue(graph_viewer, div_inputs[1].node_arg, sqrt_two, true) ||
                  IsInitializerWithExpectedValue(graph_viewer, div_inputs[1].node_arg, approximated_sqrt_two, true);

  if (!is_sqrt2) {
    return nullptr;
  }

  // Erf must have a single Add child
  const std::array<std::string_view, 1> add_types = {"Add"};
  const NodeUnit* add_node_unit = GetOnlyChildOfType(graph_viewer, erf_node_unit, add_types,
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
  if (add_node_unit == nullptr) {
    return nullptr;
  }

  // Check the other input to Add is 1.0f
  const auto& add_inputs = add_node_unit->Inputs();
  if (add_inputs.size() < 2) {
    return nullptr;
  }

  const auto& erf_output_name = erf_node_unit.Outputs()[0].node_arg.Name();
  bool is_erf_first_input = (add_inputs[0].node_arg.Name() == erf_output_name);
  const auto& add_const_input = add_inputs[is_erf_first_input ? 1 : 0];

  if (!IsInitializerWithExpectedValue(graph_viewer, add_const_input.node_arg, 1.0f, true)) {
    return nullptr;
  }

  // Add must have a single Mul child
  const std::array<std::string_view, 1> mul_types = {"Mul"};
  const NodeUnit* mul_node_unit = GetOnlyChildOfType(graph_viewer, *add_node_unit, mul_types,
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
  if (mul_node_unit == nullptr) {
    return nullptr;
  }

  // Now check which pattern we have
  const auto& root_input_name = div_inputs[0].node_arg.Name();
  const auto& mul_inputs = mul_node_unit->Inputs();

  if (mul_inputs.size() < 2) {
    return nullptr;
  }

  // Try to match Pattern 1: root -> Mul(0.5) -> ... -> Mul
  // In this case, one input to the final Mul should be from a Mul(0.5) node
  const NodeUnit* mul2_node_unit = nullptr;

  // Check if either input to mul_node_unit comes from a Mul node
  for (size_t i = 0; i < 2; ++i) {
    const auto& mul_input_name = mul_inputs[i].node_arg.Name();

    // Find the node that produces this input
    for (const auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
      const Node* node = graph_viewer.GetNode(node_index);
      if (node == nullptr) continue;

      // Check if this node's output matches our input
      for (const auto* output_def : node->OutputDefs()) {
        if (output_def && output_def->Name() == mul_input_name) {
          // Found the producer node, check if it's a Mul
          auto it = node_to_node_unit.find(node);
          if (it != node_to_node_unit.end()) {
            const NodeUnit* producer_unit = it->second;
            if (producer_unit->OpType() == "Mul" &&
                producer_unit->UnitType() == NodeUnit::Type::SingleNode &&
                node_unit_to_qnn_node_group.find(producer_unit) == node_unit_to_qnn_node_group.end()) {
              // Check if this Mul has root as input and 0.5 as the other input
              const auto& mul2_inputs = producer_unit->Inputs();
              if (mul2_inputs.size() >= 2) {
                bool has_root_input = (mul2_inputs[0].node_arg.Name() == root_input_name ||
                                       mul2_inputs[1].node_arg.Name() == root_input_name);

                if (has_root_input) {
                  size_t const_idx = (mul2_inputs[0].node_arg.Name() == root_input_name) ? 1 : 0;
                  if (IsInitializerWithExpectedValue(graph_viewer, mul2_inputs[const_idx].node_arg, 0.5f, true)) {
                    mul2_node_unit = producer_unit;
                    break;
                  }
                }
              }
            }
          }
        }
      }
      if (mul2_node_unit != nullptr) break;
    }
    if (mul2_node_unit != nullptr) break;
  }

  std::vector<const NodeUnit*> node_units;
  const NodeUnit* final_mul_node_unit = nullptr;

  if (mul2_node_unit != nullptr) {
    // Pattern 1: root -> Mul(0.5) -> ... -> Mul
    node_units = {div_node_unit, &erf_node_unit, add_node_unit, mul2_node_unit, mul_node_unit};
    final_mul_node_unit = mul_node_unit;
  } else {
    // Try Pattern 2: root -> ... -> Mul -> Mul(0.5)
    // Check if one input to mul_node_unit is root
    bool has_root_input = (mul_inputs[0].node_arg.Name() == root_input_name ||
                           mul_inputs[1].node_arg.Name() == root_input_name);

    if (!has_root_input) {
      return nullptr;
    }

    // mul_node_unit must have a single Mul child with 0.5
    const NodeUnit* mul2_node_unit_pattern2 = GetOnlyChildOfType(graph_viewer, *mul_node_unit, mul_types,
                                                                 node_to_node_unit, node_unit_to_qnn_node_group);
    if (mul2_node_unit_pattern2 == nullptr) {
      return nullptr;
    }

    // Check if this final Mul has 0.5 as one input
    const auto& mul2_inputs = mul2_node_unit_pattern2->Inputs();
    if (mul2_inputs.size() < 2) {
      return nullptr;
    }

    const auto& mul_output_name = mul_node_unit->Outputs()[0].node_arg.Name();
    bool is_mul_first_input = (mul2_inputs[0].node_arg.Name() == mul_output_name);
    size_t const_idx = is_mul_first_input ? 1 : 0;

    if (!IsInitializerWithExpectedValue(graph_viewer, mul2_inputs[const_idx].node_arg, 0.5f, true)) {
      return nullptr;
    }

    // Pattern 2
    node_units = {div_node_unit, &erf_node_unit, add_node_unit, mul_node_unit, mul2_node_unit_pattern2};
    final_mul_node_unit = mul2_node_unit_pattern2;
  }

  // Validate on QNN
  const NodeUnitIODef& root_input = div_inputs[0];
  const NodeUnitIODef& final_output = final_mul_node_unit->Outputs()[0];

  if (Status status = ValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<GeluFusion>(std::move(node_units), div_node_unit);
}

GeluFusion::GeluFusion(std::vector<const NodeUnit*>&& node_units, const NodeUnit* target_node_unit)
    : node_units_(std::move(node_units)), target_node_unit_(target_node_unit) {
}

Status GeluFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const NodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const NodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return ValidateOnQnn(qmw, node_units_, root_input, final_output);
}

Status GeluFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const NodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const NodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return CreateOnQnn(qmw, node_units_, root_input, final_output);
}

gsl::span<const NodeUnit* const> GeluFusion::GetNodeUnits() const {
  return gsl::span<const NodeUnit* const>(node_units_.data(), node_units_.size());
}

const NodeUnit* GeluFusion::GetTargetNodeUnit() const {
  return target_node_unit_;
}

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    gsl::span<const NodeUnit* const> node_units,
                                    const NodeUnitIODef& root_input,
                                    const NodeUnitIODef& final_output,
                                    bool validate) {
  assert(node_units.size() >= 4);
  const auto& node_name = utils::GetUniqueName(*node_units[0]);

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(root_input, input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(final_output, output_tensor));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_GELU,
                                                          {input_tensor.GetQnnTensor()},
                                                          {output_tensor.GetQnnTensor()},
                                                          {}));
  } else {
    // Only add tensor wrappers if they don't already exist
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(root_input.node_arg.Name())) {
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    }
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(final_output.node_arg.Name())) {
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    }
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_GELU,
                                                      {root_input.node_arg.Name()},
                                                      {final_output.node_arg.Name()},
                                                      {},
                                                      validate),
                      "Failed to add fused Gelu node.");
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
