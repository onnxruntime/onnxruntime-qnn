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

/// @brief Get the tensor shape info for a Mul input.
/// @return The tensor info, or nullptr if it cannot be queried (e.g. non-tensor input).
const OrtTensorTypeAndShapeInfo* GetMulInputTensorInfo(const OrtValueInfo* value_info, const OrtApi& ort_api) {
  const OrtTypeInfo* type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(value_info, &type_info), ort_api, nullptr);
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info), ort_api, nullptr);
  return tensor_info;
}

/// @brief Check whether a Mul input provably holds exactly one element and therefore acts as a scalar
/// under ONNX (numpy-style) broadcasting: rank-0, or every dimension is 1 (e.g. shape [1]).
/// Symbolic (unknown) dimensions are not provably single-element.
/// @param tensor_info Shape info of the Mul input to inspect
/// @param rank Number of dimensions of the Mul input
/// @param ort_api ORT API interface
/// @return true for single-element, false otherwise or if the shape cannot be queried.
bool IsSingleElementInput(const OrtTensorTypeAndShapeInfo* tensor_info, size_t rank, const OrtApi& ort_api) {
  if (rank == 0) {
    return true;
  }

  std::vector<int64_t> dims(rank, -1);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensions(tensor_info, dims.data(), dims.size()), ort_api, false);
  for (int64_t dim : dims) {
    if (dim != 1) {
      return false;
    }
  }

  return true;
}

/// @brief Get the index of the single-element scale input in the Mul node.
/// @param mul Multiply node unit
/// @param ort_api ORT API interface
/// @return The scale input index (0 or 1) if an input is single-element (the lower-rank one if both are);
/// otherwise std::nullopt.
std::optional<size_t> GetMulScaleInputIndex(const OrtNodeUnit& mul, const OrtApi& ort_api) {
  // Get inputs of mul node
  size_t num_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&mul.GetNode(), &num_inputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> inputs(num_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&mul.GetNode(), inputs.data(), inputs.size()),
                             ort_api,
                             std::nullopt);

  const OrtTensorTypeAndShapeInfo* x_tensor_info = GetMulInputTensorInfo(inputs[0], ort_api);
  const OrtTensorTypeAndShapeInfo* y_tensor_info = GetMulInputTensorInfo(inputs[1], ort_api);
  if (x_tensor_info == nullptr || y_tensor_info == nullptr) {
    return std::nullopt;
  }

  size_t x_rank = 0;
  size_t y_rank = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(x_tensor_info, &x_rank), ort_api, std::nullopt);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(y_tensor_info, &y_rank), ort_api, std::nullopt);

  const bool is_x_single_element = IsSingleElementInput(x_tensor_info, x_rank, ort_api);
  const bool is_y_single_element = IsSingleElementInput(y_tensor_info, y_rank, ort_api);
  if (!is_x_single_element && !is_y_single_element) {
    return std::nullopt;
  }

  // If both inputs are single-element (e.g. data [1], scale []), the lower-rank one is the scale.
  // Equal ranks are ambiguous.
  if (is_x_single_element && is_y_single_element && x_rank == y_rank) {
    return std::nullopt;
  }

  const size_t scale_index = (is_x_single_element && (!is_y_single_element || x_rank < y_rank)) ? 0U : 1U;

  // The scale must not have higher rank than the data input. Broadcast would otherwise prepend
  // 1-sized leading dimensions (e.g. Mul([4], [[s]]) -> [1, 4]), so the Mul output shape would
  // differ from the data (fused Softmax input) shape. Rank-0 scales are always safe.
  const size_t scale_rank = (scale_index == 0) ? x_rank : y_rank;
  const size_t data_rank = (scale_index == 0) ? y_rank : x_rank;
  if (scale_rank > data_rank) {
    return std::nullopt;
  }

  return scale_index;
}

/// @brief Get the axis for softmax
/// @param mul Multiply node unit
/// @param softmax Softmax node unit
/// @param ort_api ORT API interface
/// @return The axis for softmax
std::optional<uint32_t> GetPositiveSoftmaxAxis(const OrtNodeUnit& mul, const OrtNodeUnit& softmax, const OrtApi& ort_api) {
  OrtNodeAttrHelper softmax_attr_helper(softmax);
  int64_t axis_value = softmax_attr_helper.Get("axis", (int64_t)-1);

  if (axis_value < 0) {
    // Get the scale input index
    std::optional<size_t> input_scale_index = GetMulScaleInputIndex(mul, ort_api);
    if (!input_scale_index.has_value()) {
      return std::nullopt;
    }

    size_t input_other_index = 1U - input_scale_index.value();

    // Get the other input's shape
    size_t num_inputs = 0;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&mul.GetNode(), &num_inputs), ort_api, std::nullopt);
    std::vector<const OrtValueInfo*> inputs(num_inputs);
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&mul.GetNode(), inputs.data(), inputs.size()),
                               ort_api,
                               std::nullopt);

    const OrtValueInfo* other_input = inputs[input_other_index];
    const OrtTypeInfo* type_info = nullptr;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(other_input, &type_info), ort_api, std::nullopt);
    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info), ort_api, std::nullopt);

    size_t dims_count = 0;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(tensor_info, &dims_count), ort_api, std::nullopt);

    axis_value += static_cast<int64_t>(dims_count);
  }

  return static_cast<uint32_t>(axis_value);
}

/// @brief Identify scale input from mul node if present
/// @param qnn_model_wrapper QNN model wrapper
/// @param mul Multiply node unit
/// @param ort_api ORT API interface
/// @return The scale input float value if found, otherwise std::nullopt
std::optional<float> ExtractScaleValueFromMul(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& mul, const OrtApi& ort_api) {
  std::optional<size_t> input_scale_index = GetMulScaleInputIndex(mul, ort_api);
  if (!input_scale_index.has_value()) {
    return std::nullopt;
  }

  // Get inputs of mul node
  size_t num_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&mul.GetNode(), &num_inputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> inputs(num_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&mul.GetNode(), inputs.data(), inputs.size()),
                             ort_api,
                             std::nullopt);

  const OrtValueInfo* scale_input = inputs[input_scale_index.value()];

  // Get the name of the scale input
  const char* scale_name = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(scale_input, &scale_name), ort_api, std::nullopt);

  // Check if it's a constant initializer
  if (!qnn_model_wrapper.IsConstantInput(scale_name)) {
    return std::nullopt;
  }

  // Get the constant tensor
  const OrtValueInfo* scale_tensor = qnn_model_wrapper.GetConstantTensor(scale_name);
  if (!scale_tensor) {
    return std::nullopt;
  }

  // Get the value
  const OrtValue* value = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetInitializerValue(scale_tensor, &value), ort_api, std::nullopt);

  // Check if it's a float tensor
  ONNXTensorElementDataType element_type;
  OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetTensorTypeAndShape(value, &tensor_info), ort_api, std::nullopt);
  if (OrtStatus* status = ort_api.GetTensorElementType(tensor_info, &element_type)) {
    ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);
    RETURN_DEFAULT_IF_API_FAIL(status, ort_api, std::nullopt);
  }
  ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return std::nullopt;
  }

  // Get the raw data
  const void* raw_data = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetTensorData(value, &raw_data), ort_api, std::nullopt);

  // Return the float value
  return *static_cast<const float*>(raw_data);
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
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  std::optional<size_t> input_scale_index = GetMulScaleInputIndex(mul_node_unit, ort_api);
  RETURN_IF(!input_scale_index.has_value(), "Failed to get scale input index");

  size_t input_other_index = 1U - input_scale_index.value();
  const OrtNodeUnitIODef& mul_input_other = mul_node_unit.Inputs()[input_other_index];
  const OrtNodeUnitIODef& softmax_output = softmax_node_unit.Outputs()[0];

  std::vector<std::string> param_tensor_names;
  {  // axis
    std::optional<uint32_t> axis = GetPositiveSoftmaxAxis(mul_node_unit, softmax_node_unit, ort_api);
    if (axis.has_value()) {
      RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, softmax_node_unit.Index(),
                                             softmax_node_unit.Name(), axis.value(),
                                             QNN_OP_SOFTMAX_PARAM_AXIS, param_tensor_names));
    }
  }
  {  // beta
    OrtNodeAttrHelper softmax_attr_helper(softmax_node_unit);
    float beta = softmax_attr_helper.Get("beta", 1.0f);
    float scale = ExtractScaleValueFromMul(qnn_model_wrapper, mul_node_unit, ort_api).value_or(1.0f);
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
/// @param qnn_model_wrapper QNN model wrapper used to query tensor metadata.
/// @param mul Multiply node unit.
/// @param softmax Softmax node unit.
/// @param ort_api ORT API interface.
/// @return true if the axis is the last dimension, false if it is valid but non-last,
///         or std::nullopt if the axis/rank cannot be determined.
std::optional<bool> IsSoftmaxAxisLast(QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& mul,
                                      const OrtNodeUnit& softmax,
                                      const OrtApi& ort_api) {
  const std::optional<size_t> scale_input_index = GetMulScaleInputIndex(mul, ort_api);
  if (!scale_input_index.has_value()) return std::nullopt;

  const size_t data_input_index = 1U - scale_input_index.value();
  const OrtNodeUnitIODef& softmax_input = mul.Inputs()[data_input_index];

  TensorInfo input_info = {};
  if (Ort::Status status = qnn_model_wrapper.GetTensorInfo(softmax_input, input_info);
      !status.IsOK() || input_info.shape.empty()) {
    return std::nullopt;
  }

  const std::optional<uint32_t> axis = GetPositiveSoftmaxAxis(mul, softmax, ort_api);
  if (!axis.has_value() || axis.value() >= input_info.shape.size()) {
    return std::nullopt;
  }

  const uint32_t last_axis = static_cast<uint32_t>(input_info.shape.size() - 1);

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

  // Get the OrtApi interface
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Check if the mul node has a scale input that can fold into the softmax's beta
  std::optional<float> scale = ExtractScaleValueFromMul(qnn_model_wrapper, mul_node_unit, ort_api);
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

  const std::optional<bool> axis_is_last =
      IsSoftmaxAxisLast(qnn_model_wrapper, mul_node_unit, *softmax, ort_api);

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
