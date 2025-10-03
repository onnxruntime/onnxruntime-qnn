// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/cast_lone_q_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, cast_node_unit, q_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (cast_node_unit), (q_node_unit), true)
#define CreateOnQnn(qnn_model_wrapper, cast_node_unit, q_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (cast_node_unit), (q_node_unit), false)

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& cast_node_unit,
                                         const OrtNodeUnit& q_node_unit,
                                         bool validate) {
  assert(cast_node_unit.OpType() == "Cast" && q_node_unit.OpType() == QUANTIZE_LINEAR);
  const auto& node_name = utils::GetUniqueName(cast_node_unit);
  const OrtNodeUnitIODef& input_def = cast_node_unit.Inputs()[0];
  const OrtNodeUnitIODef& output_def = q_node_unit.Outputs()[0];

  // ProcessInputs - only add input tensor if it doesn't already exist
  const auto& input_name = input_def.name;
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    TensorInfo input_tensor_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_tensor_info));
    QnnTensorWrapper input_tensor_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_tensor_info, input_name, input_tensor_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor_wrapper)),
                  "Failed to add input tensor for QNN Convert node.");
  }

  // ProcessAttributesAndOutputs
  const auto& output_name = output_def.name;
  TensorInfo output_tensor_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_tensor_info));
  QnnTensorWrapper output_tensor_wrapper;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_tensor_info, output_name, output_tensor_wrapper));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                "Failed to add output tensor for QNN Convert node.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_CONVERT,
                                                {input_name},
                                                {output_name},
                                                {},
                                                validate),
                "Failed to add fused Convert node.");

  return Ort::Status();
}

std::unique_ptr<IQnnNodeGroup> CastLoneQFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& cast_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  // Expect that this function is called with a standalone Cast.
  if (cast_node_unit.OpType() != "Cast" || cast_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Cast must have a single Q child (1 output edge) and must not produce a graph output.
  const std::array<std::string_view, 1> child_types = {QUANTIZE_LINEAR};
  const OrtNodeUnit* q_node_unit = GetOnlyChildOfType(qnn_model_wrapper, cast_node_unit, child_types,
                                                      node_to_node_unit, node_unit_to_qnn_node_group);

  if (q_node_unit == nullptr) {
    return nullptr;
  }

  // Cast input must not come from a DQ node (we want to fuse Cast -> Q, not DQ -> Cast -> Q).
  const std::array<std::string_view, 1> parent_types = {DEQUANTIZE_LINEAR};
  const OrtNodeUnit* dq_node_unit = GetParentOfType(qnn_model_wrapper, cast_node_unit, parent_types,
                                                    node_to_node_unit, node_unit_to_qnn_node_group);

  if (dq_node_unit != nullptr) {
    return nullptr;
  }

  // Skip if Cast input is a constant.
  if (qnn_model_wrapper.IsConstantInput(cast_node_unit.Inputs()[0].name)) {
    return nullptr;
  }

  if (Ort::Status status = ValidateOnQnn(qnn_model_wrapper, cast_node_unit, *q_node_unit);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<CastLoneQFusion>(cast_node_unit, *q_node_unit);
}

CastLoneQFusion::CastLoneQFusion(const OrtNodeUnit& cast_node_unit, const OrtNodeUnit& q_node_unit)
    : node_units_{&cast_node_unit, &q_node_unit} {
}

Ort::Status CastLoneQFusion::IsSupported(QnnModelWrapper& qmw,
                                         [[maybe_unused]] const Ort::Logger& logger) const {
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

Ort::Status CastLoneQFusion::AddToModelBuilder(QnnModelWrapper& qmw,
                                               [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

gsl::span<const OrtNodeUnit* const> CastLoneQFusion::GetNodeUnits() const {
  return node_units_;
}

const OrtNodeUnit* CastLoneQFusion::GetTargetNodeUnit() const {
  return node_units_[0];
}

}  // namespace qnn
}  // namespace onnxruntime
