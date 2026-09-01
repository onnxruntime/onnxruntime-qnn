// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/l2_norm_fusion.h"

#include <array>
#include <gsl/gsl>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

constexpr char kOpReduceL2[] = "ReduceL2";
constexpr char kOpAdd[] = "Add";
constexpr char kOpDiv[] = "Div";

// Reads a constant float scalar initializer by name, if present. Returns nullopt otherwise.
std::optional<float> GetConstantFloatScalar(const QnnModelWrapper& qmw, const std::string& input_name) {
  if (!qmw.IsConstantInput(input_name)) {
    return std::nullopt;
  }
  const OrtValueInfo* value_info = qmw.GetConstantTensor(input_name);
  if (!value_info) {
    return std::nullopt;
  }
  Ort::ConstValueInfo ort_value_info(value_info);
  Ort::ConstValue ort_value;
  if (!ort_value_info.GetInitializer(ort_value).IsOK()) {
    return std::nullopt;
  }
  auto type_info = ort_value_info.TypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return std::nullopt;
  }
  if (tensor_info.GetElementCount() != 1) {
    return std::nullopt;
  }
  const float* data = ort_value.GetTensorData<float>();
  if (!data) {
    return std::nullopt;
  }
  return *data;
}

#define ValidateOnQnn(qmw, node_units, root_input, final_output, epsilon, axes) \
  CreateOrValidateOnQnn((qmw), (node_units), (root_input), (final_output), (epsilon), (axes), true)
#define CreateOnQnn(qmw, node_units, root_input, final_output, epsilon, axes) \
  CreateOrValidateOnQnn((qmw), (node_units), (root_input), (final_output), (epsilon), (axes), false)

// Emits (or validates) a single QNN L2Norm op for the fused ReduceL2 -> Add(eps) -> Div pattern.
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef& final_output,
                                         float epsilon,
                                         gsl::span<const uint32_t> axes,
                                         bool validate) {
  const std::string node_name = utils::UniqueNameGenerator().New(*node_units[0]);

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(root_input, input_tensor));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(final_output, output_tensor));

  Qnn_Scalar_t epsilon_scalar = QNN_SCALAR_INIT;
  epsilon_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_scalar.floatValue = epsilon;
  QnnParamWrapper epsilon_param(node_units[0]->Index(), node_units[0]->Name(),
                                QNN_OP_L2_NORM_PARAM_EPSILON, epsilon_scalar);

  std::vector<uint32_t> axes_vec(axes.begin(), axes.end());
  std::vector<uint32_t> axes_shape{static_cast<uint32_t>(axes_vec.size())};
  QnnParamWrapper axes_param(node_units[0]->Index(), node_units[0]->Name(),
                             QNN_OP_L2_NORM_PARAM_AXES,
                             std::move(axes_shape), std::move(axes_vec));

  if (validate) {
    return qmw.ValidateQnnNode(node_name,
                               QNN_OP_PACKAGE_NAME_QTI_AISW,
                               QNN_OP_L2_NORM,
                               {input_tensor.GetQnnTensor()},
                               {output_tensor.GetQnnTensor()},
                               {epsilon_param.GetQnnParam(), axes_param.GetQnnParam()});
  }

  const std::string epsilon_param_name = epsilon_param.GetParamTensorName();
  RETURN_IF_NOT(qmw.AddParamWrapper(std::move(epsilon_param)), "Failed to add epsilon param.");
  const std::string axes_param_name = axes_param.GetParamTensorName();
  RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axes_param)), "Failed to add axes param.");

  if (!qmw.IsQnnTensorWrapperExist(root_input.name)) {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(input_tensor)), "Failed to add input tensor.");
  }
  if (!qmw.IsQnnTensorWrapperExist(final_output.name)) {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_tensor)), "Failed to add output tensor.");
  }

  RETURN_IF_NOT(qmw.CreateQnnNode(node_name,
                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                  QNN_OP_L2_NORM,
                                  {root_input.name},
                                  {final_output.name},
                                  {epsilon_param_name, axes_param_name},
                                  validate),
                "Failed to create fused L2Norm node.");

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> L2NormFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reducel2_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  // Must start with a ReduceL2 SingleNode with keepdims=1 (so the norm broadcasts against x in Div).
  if (reducel2_node_unit.OpType() != kOpReduceL2 ||
      reducel2_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  OrtNodeAttrHelper reducel2_helper(reducel2_node_unit);
  if (reducel2_helper.Get("keepdims", static_cast<int64_t>(1)) != 1) {
    return nullptr;
  }

  // ReduceL2 -> Add (the +epsilon).
  const std::array<std::string_view, 1> add_types{kOpAdd};
  const OrtNodeUnit* add_node_unit = GetOnlyChildOfType(qnn_model_wrapper, reducel2_node_unit, add_types,
                                                        node_to_node_unit, node_unit_to_qnn_node_group);
  if (add_node_unit == nullptr) {
    return nullptr;
  }

  // The Add's other input must be a constant float scalar (the epsilon).
  std::optional<float> epsilon;
  for (const auto& add_in : add_node_unit->Inputs()) {
    if (add_in.name == reducel2_node_unit.Outputs()[0].name) {
      continue;
    }
    epsilon = GetConstantFloatScalar(qnn_model_wrapper, add_in.name);
  }
  if (!epsilon.has_value()) {
    return nullptr;
  }

  // Add -> Div. Div numerator must be the ReduceL2's input; denominator is Add's output.
  const std::array<std::string_view, 1> div_types{kOpDiv};
  const OrtNodeUnit* div_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *add_node_unit, div_types,
                                                        node_to_node_unit, node_unit_to_qnn_node_group);
  if (div_node_unit == nullptr) {
    return nullptr;
  }

  const std::string& reducel2_input_name = reducel2_node_unit.Inputs()[0].name;
  const std::string& add_output_name = add_node_unit->Outputs()[0].name;
  const auto& div_inputs = div_node_unit->Inputs();
  if (div_inputs.size() != 2 ||
      div_inputs[0].name != reducel2_input_name ||
      div_inputs[1].name != add_output_name) {
    return nullptr;
  }

  // Axes from the ReduceL2 (handles opset<18 attr and opset>=18 input forms).
  std::optional<std::vector<uint32_t>> axes = GetReduceAxes(qnn_model_wrapper, reducel2_node_unit);
  if (!axes.has_value() || axes->empty()) {
    return nullptr;
  }

  std::array<const OrtNodeUnit*, 3> fused_units{&reducel2_node_unit, add_node_unit, div_node_unit};
  if (Ort::Status status = CreateOrValidateOnQnn(qnn_model_wrapper, fused_units,
                                                 reducel2_node_unit.Inputs()[0], div_node_unit->Outputs()[0],
                                                 epsilon.value(), axes.value(), /*validate=*/true);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<L2NormFusion>(reducel2_node_unit, *add_node_unit, *div_node_unit);
}

gsl::span<const OrtNodeUnit* const> L2NormFusion::GetNodeUnits() const {
  return node_units_;
}

Ort::Status L2NormFusion::IsSupported(QnnModelWrapper& qnn_model_wrapper,
                                      [[maybe_unused]] const Ort::Logger& logger) const {
  std::optional<std::vector<uint32_t>> axes = GetReduceAxes(qnn_model_wrapper, *node_units_[0]);
  RETURN_IF_NOT(axes.has_value() && !axes->empty(), "L2NormFusion: failed to get ReduceL2 axes.");

  const OrtNodeUnit& add_node_unit = *node_units_[1];
  std::optional<float> epsilon;
  for (const auto& add_in : add_node_unit.Inputs()) {
    if (add_in.name == node_units_[0]->Outputs()[0].name) {
      continue;
    }
    epsilon = GetConstantFloatScalar(qnn_model_wrapper, add_in.name);
  }
  RETURN_IF_NOT(epsilon.has_value(), "L2NormFusion: failed to get epsilon.");

  return ValidateOnQnn(qnn_model_wrapper, node_units_, node_units_[0]->Inputs()[0],
                       node_units_[2]->Outputs()[0], epsilon.value(), axes.value());
}

Ort::Status L2NormFusion::AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                            [[maybe_unused]] const Ort::Logger& logger) const {
  std::optional<std::vector<uint32_t>> axes = GetReduceAxes(qnn_model_wrapper, *node_units_[0]);
  RETURN_IF_NOT(axes.has_value() && !axes->empty(), "L2NormFusion: failed to get ReduceL2 axes.");

  const OrtNodeUnit& add_node_unit = *node_units_[1];
  std::optional<float> epsilon;
  for (const auto& add_in : add_node_unit.Inputs()) {
    if (add_in.name == node_units_[0]->Outputs()[0].name) {
      continue;
    }
    epsilon = GetConstantFloatScalar(qnn_model_wrapper, add_in.name);
  }
  RETURN_IF_NOT(epsilon.has_value(), "L2NormFusion: failed to get epsilon.");

  return CreateOnQnn(qnn_model_wrapper, node_units_, node_units_[0]->Inputs()[0],
                     node_units_[2]->Outputs()[0], epsilon.value(), axes.value());
}

}  // namespace qnn
}  // namespace onnxruntime
