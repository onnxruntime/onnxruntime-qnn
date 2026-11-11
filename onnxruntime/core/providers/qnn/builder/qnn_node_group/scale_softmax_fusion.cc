// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/scale_softmax_fusion.h"

#include <array>
#include <gsl/gsl>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kOpMul[] = "Mul";
constexpr char kOpSoftmax[] = "Softmax";

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, mul_node_unit, softmax_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (mul_node_unit), (softmax_node_unit), true)
#define CreateOnQnn(qnn_model_wrapper, mul_node_unit, softmax_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (mul_node_unit), (softmax_node_unit), false)
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& mul_node_unit,
                                         const OrtNodeUnit& softmax_node_unit, bool validate);

/// @brief Get the index of the single-element scale input in the Mul node.
/// @param mul Multiply node unit
/// @return The scale input index (0 or 1) if an input is single-element (the lower-rank one if both are);
/// otherwise std::nullopt.
std::optional<size_t> GetMulScaleInputIndex(const OrtNodeUnit& mul) {
  const auto& inputs = mul.Inputs();
  if (inputs.size() != 2 || !inputs[0].shape.has_value() || !inputs[1].shape.has_value()) {
    return std::nullopt;
  }

  const std::vector<int64_t>& x_shape = *inputs[0].shape;
  const std::vector<int64_t>& y_shape = *inputs[1].shape;
  const bool is_x_single_element = IsSingleElementShape(x_shape);
  const bool is_y_single_element = IsSingleElementShape(y_shape);
  if (!is_x_single_element && !is_y_single_element) {
    return std::nullopt;
  }

  // If both inputs are single-element (e.g. data [1], scale []), the lower-rank one is the scale.
  // Equal ranks are ambiguous.
  if (is_x_single_element && is_y_single_element && x_shape.size() == y_shape.size()) {
    return std::nullopt;
  }

  const bool x_is_scale = is_x_single_element && (!is_y_single_element || x_shape.size() < y_shape.size());
  const size_t scale_index = x_is_scale ? 0U : 1U;

  // The scale must not have higher rank than the data input. Broadcast would otherwise prepend
  // 1-sized leading dimensions (e.g. Mul([4], [[s]]) -> [1, 4]), so the Mul output shape would
  // differ from the data (fused Softmax input) shape. Rank-0 scales are always safe.
  if (inputs[scale_index].shape->size() > inputs[1U - scale_index].shape->size()) {
    return std::nullopt;
  }

  return scale_index;
}

/// @brief Get the axis for softmax
/// @param mul Multiply node unit
/// @param softmax Softmax node unit
/// @return The axis for softmax
std::optional<uint32_t> GetPositiveSoftmaxAxis(const OrtNodeUnit& mul, const OrtNodeUnit& softmax) {
  OrtNodeAttrHelper softmax_attr_helper(softmax);
  int64_t axis_value = softmax_attr_helper.Get("axis", (int64_t)-1);

  if (axis_value < 0) {
    const std::optional<size_t> scale_index = GetMulScaleInputIndex(mul);
    if (!scale_index.has_value()) {
      return std::nullopt;
    }

    const auto& data_shape = mul.Inputs()[1U - scale_index.value()].shape;
    if (!data_shape.has_value()) {
      return std::nullopt;
    }

    axis_value += static_cast<int64_t>(data_shape->size());
  }

  return static_cast<uint32_t>(axis_value);
}

/// @brief Identify scale input from mul node if present
/// @param qnn_model_wrapper QNN model wrapper
/// @param mul Multiply node unit
/// @return The scale value if found, otherwise std::nullopt
std::optional<float> ExtractScaleValueFromMul(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& mul) {
  const std::optional<size_t> scale_index = GetMulScaleInputIndex(mul);
  if (!scale_index.has_value()) {
    return std::nullopt;
  }

  return GetScalarConstantValue(qnn_model_wrapper, mul.Inputs()[scale_index.value()].name);
}

/// @brief Create or validate the QNN node
/// @param qnn_model_wrapper QNN model wrapper
/// @param node_units The node units containing the softmax and mul nodes
/// @param validate Whether to validate the QNN node
/// @return Status
Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& mul_node_unit,
                                  const OrtNodeUnit& softmax_node_unit,
                                  bool validate) {
  assert(mul_node_unit.OpType() == kOpMul && softmax_node_unit.OpType() == kOpSoftmax);

  std::optional<size_t> input_scale_index = GetMulScaleInputIndex(mul_node_unit);
  RETURN_IF(!input_scale_index.has_value(), "Failed to get scale input index");

  size_t input_other_index = 1U - input_scale_index.value();
  const OrtNodeUnitIODef& mul_input_other = mul_node_unit.Inputs()[input_other_index];
  const OrtNodeUnitIODef& softmax_output = softmax_node_unit.Outputs()[0];

  std::vector<std::string> param_tensor_names;
  {  // axis
    std::optional<uint32_t> axis = GetPositiveSoftmaxAxis(mul_node_unit, softmax_node_unit);
    if (axis.has_value()) {
      RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, softmax_node_unit.Index(),
                                             softmax_node_unit.Name(), axis.value(),
                                             QNN_OP_SOFTMAX_PARAM_AXIS, param_tensor_names));
    }
  }
  {  // beta
    OrtNodeAttrHelper softmax_attr_helper(softmax_node_unit);
    float beta = softmax_attr_helper.Get("beta", 1.0f);
    float scale = ExtractScaleValueFromMul(qnn_model_wrapper, mul_node_unit).value_or(1.0f);
    RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, softmax_node_unit.Index(),
                                        softmax_node_unit.Name(), scale * beta,
                                        QNN_OP_SOFTMAX_PARAM_BETA, param_tensor_names));
  }

  QnnTensorWrapper fused_softmax_input;
  QnnTensorWrapper fused_softmax_output;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(mul_input_other, fused_softmax_input));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(softmax_output, fused_softmax_output));

  const std::string node_name = onnxruntime::qnn::utils::UniqueNameGenerator().New(softmax_node_unit);
  if (validate) {
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_SOFTMAX,
                                                      {fused_softmax_input.GetQnnTensor()},
                                                      {fused_softmax_output.GetQnnTensor()},
                                                      {}));
  } else {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fused_softmax_input)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fused_softmax_output)), "Failed to add output");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_SOFTMAX,
                                                  {mul_input_other.name},
                                                  {softmax_output.name},
                                                  std::move(param_tensor_names),
                                                  validate),
                  ("Failed to add fused " + std::string(kOpSoftmax) + " node.").c_str());
  }

  return Ort::Status();
}

}  // namespace

/// @brief Check whether the Softmax axis is the last input dimension.
/// @param mul Multiply node unit.
/// @param softmax Softmax node unit.
/// @return true if the axis is the last dimension, false if it is valid but non-last,
///         or std::nullopt if the axis/rank cannot be determined.
std::optional<bool> IsSoftmaxAxisLast(const OrtNodeUnit& mul, const OrtNodeUnit& softmax) {
  const std::optional<size_t> scale_input_index = GetMulScaleInputIndex(mul);
  if (!scale_input_index.has_value()) return std::nullopt;

  const size_t data_input_index = 1U - scale_input_index.value();
  const auto& data_shape = mul.Inputs()[data_input_index].shape;
  if (!data_shape.has_value() || data_shape->empty()) {
    return std::nullopt;
  }

  const std::optional<uint32_t> axis = GetPositiveSoftmaxAxis(mul, softmax);
  if (!axis.has_value() || axis.value() >= data_shape->size()) {
    return std::nullopt;
  }

  const uint32_t last_axis = static_cast<uint32_t>(data_shape->size() - 1);

  return axis.value() == last_axis;
}

std::unique_ptr<IQnnNodeGroup> ScaleSoftmaxFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& mul_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  if (mul_node_unit.OpType() != kOpMul || mul_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  // Check if the mul node has a scale input that can fold into the softmax's beta
  std::optional<float> scale = ExtractScaleValueFromMul(qnn_model_wrapper, mul_node_unit);
  if (!scale.has_value()) {
    return nullptr;
  }

  // Mul node must have a single Softmax node as child
  const std::array<std::string_view, 1> child_op_types{kOpSoftmax};
  const OrtNodeUnit* softmax = GetOnlyChildOfType(qnn_model_wrapper, mul_node_unit, child_op_types,
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (softmax == nullptr) {
    return nullptr;
  }

  const std::optional<bool> axis_is_last = IsSoftmaxAxisLast(mul_node_unit, *softmax);

  if (!axis_is_last.has_value() || !axis_is_last.value()) {
    return nullptr;
  }

  if (Ort::Status status = ValidateOnQnn(qnn_model_wrapper, mul_node_unit, *softmax);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<ScaleSoftmaxFusion>(mul_node_unit, *softmax);
}

gsl::span<const OrtNodeUnit* const> ScaleSoftmaxFusion::GetNodeUnits() const {
  return node_units_;
}

Ort::Status ScaleSoftmaxFusion::IsSupported(QnnModelWrapper& qnn_model_wrapper,
                                            [[maybe_unused]] const Ort::Logger& logger) const {
  return ValidateOnQnn(qnn_model_wrapper, *node_units_[0], *node_units_[1]);
}

Ort::Status ScaleSoftmaxFusion::AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                                  [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOnQnn(qnn_model_wrapper, *node_units_[0], *node_units_[1]);
}

}  // namespace qnn
}  // namespace onnxruntime
