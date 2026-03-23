// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/layer_norm_fusion.h"

#include <gsl/gsl>
#include <cassert>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations.
#define ValidateOnQnn(qmw, node_units, root_input, gamma_input, beta_input, final_output, epsilon, axes) \
  CreateOrValidateOnQnn((qmw), (node_units), (root_input), (gamma_input), (beta_input), (final_output), (epsilon), (axes), true)
#define CreateOnQnn(qmw, node_units, root_input, gamma_input, beta_input, final_output, epsilon, axes) \
  CreateOrValidateOnQnn((qmw), (node_units), (root_input), (gamma_input), (beta_input), (final_output), (epsilon), (axes), false)

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef& gamma_input,
                                         const OrtNodeUnitIODef& beta_input,
                                         const OrtNodeUnitIODef& final_output,
                                         float epsilon,
                                         gsl::span<const uint32_t> axes,
                                         bool validate);

/// <summary>
/// Reads a float scalar value from a constant initializer.
/// Returns std::nullopt if the input is not a constant float scalar.
/// </summary>
static std::optional<float> GetConstantFloatScalar(const QnnModelWrapper& qmw,
                                                    const OrtApi& ort_api,
                                                    const std::string& input_name) {
  if (!qmw.IsConstantInput(input_name)) {
    return std::nullopt;
  }

  const OrtValueInfo* value_info = qmw.GetConstantTensor(input_name);
  if (!value_info) {
    return std::nullopt;
  }

  const OrtValue* value = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetInitializerValue(value_info, &value), ort_api, std::nullopt);

  OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetTensorTypeAndShape(value, &tensor_info), ort_api, std::nullopt);

  ONNXTensorElementDataType elem_type;
  if (OrtStatus* s = ort_api.GetTensorElementType(tensor_info, &elem_type)) {
    ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);
    RETURN_DEFAULT_IF_API_FAIL(s, ort_api, std::nullopt);
  }
  ort_api.ReleaseTensorTypeAndShapeInfo(tensor_info);

  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return std::nullopt;
  }

  const void* raw_data = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetTensorData(value, &raw_data), ort_api, std::nullopt);

  return *static_cast<const float*>(raw_data);
}

/// <summary>
/// Reads the axes from a ReduceMean node. Handles both opset < 18 (attribute) and opset >= 18 (input).
/// Returns the axes as positive uint32_t values normalized by input rank.
/// Returns std::nullopt on failure.
/// </summary>
static std::optional<std::vector<uint32_t>> GetReduceMeanAxes(const QnnModelWrapper& qmw,
                                                               const OrtNodeUnit& reduce_mean_node_unit) {
  const auto& inputs = reduce_mean_node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  if (!qmw.GetOnnxShape(inputs[0].shape, input_shape)) {
    return std::nullopt;
  }
  const size_t input_rank = input_shape.size();

  std::vector<int64_t> raw_axes;
  OrtNodeAttrHelper node_helper(reduce_mean_node_unit);

  const int opset = reduce_mean_node_unit.SinceVersion();
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

  return axes;
}

std::unique_ptr<IQnnNodeGroup> LayerNormFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reduce_mean_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // Must start with a ReduceMean SingleNode.
  if (reduce_mean_node_unit.OpType() != "ReduceMean" ||
      reduce_mean_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // ---- Step 1: ReduceMean₁ → Sub ----
  // ReduceMean₁ must have exactly one child: Sub.
  const std::array<std::string_view, 1> sub_types{"Sub"};
  const OrtNodeUnit* sub_node_unit = GetOnlyChildOfType(qnn_model_wrapper, reduce_mean_node_unit,
                                                        sub_types, node_to_node_unit,
                                                        node_unit_to_qnn_node_group);
  if (sub_node_unit == nullptr) {
    return nullptr;
  }

  // Verify Sub input[0] is the same tensor as ReduceMean₁ input[0] (both consume x).
  const auto& rm1_inputs = reduce_mean_node_unit.Inputs();
  const auto& sub_inputs = sub_node_unit->Inputs();
  if (rm1_inputs.empty() || sub_inputs.size() < 2) {
    return nullptr;
  }
  if (sub_inputs[0].name != rm1_inputs[0].name) {
    return nullptr;
  }

  // ---- Step 2: Sub → Pow or Transpose (detect pattern) ----
  // Regular pattern: Sub has 2 consumers: Pow and Div.
  // Transpose pattern: Sub has 1 consumer: Transpose. Transpose then has 2 consumers: Pow and Div.
  const auto& sub_outputs = sub_node_unit->Outputs();
  if (sub_outputs.empty()) {
    return nullptr;
  }

  bool has_transpose = false;
  const OrtNodeUnit* transpose_node_unit = nullptr;
  const OrtNodeUnit* pow_node_unit = nullptr;
  // Track which tensor feeds the Div numerator (Sub output for regular, Transpose output for transpose pattern).
  std::string div_numerator_name;

  {
    const Ort::ConstNode sub_node(&sub_node_unit->GetNode());
    const auto sub_node_outputs = sub_node.GetOutputs();
    if (sub_node_outputs.size() != 1) {
      return nullptr;
    }

    const auto consumers = sub_node_outputs[0].GetConsumers();
    if (consumers.size() == 2) {
      // Regular pattern: Sub → {Pow, Div}
      for (const auto& consumer_info : consumers) {
        if (consumer_info.node == nullptr) {
          return nullptr;
        }
        const Ort::ConstNode consumer_node = consumer_info.node;
        const std::string consumer_type = consumer_node.GetOperatorType();

        const auto it = node_to_node_unit.find(consumer_node);
        if (it == node_to_node_unit.end()) {
          return nullptr;
        }
        const OrtNodeUnit* consumer_unit = it->second;

        if (node_unit_to_qnn_node_group.count(consumer_unit) != 0) {
          return nullptr;
        }
        if (consumer_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
          return nullptr;
        }

        if (consumer_type == "Pow") {
          pow_node_unit = consumer_unit;
        }
        // "Div" is the other consumer — handled later
      }
      div_numerator_name = sub_outputs[0].name;
    } else if (consumers.size() == 1) {
      // Transpose pattern: Sub → Transpose → {Pow, Div}
      if (consumers[0].node == nullptr) {
        return nullptr;
      }
      const Ort::ConstNode consumer_node = consumers[0].node;
      if (std::string(consumer_node.GetOperatorType()) != "Transpose") {
        return nullptr;
      }
      const auto it = node_to_node_unit.find(consumer_node);
      if (it == node_to_node_unit.end()) {
        return nullptr;
      }
      transpose_node_unit = it->second;
      if (node_unit_to_qnn_node_group.count(transpose_node_unit) != 0 ||
          transpose_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
        return nullptr;
      }
      has_transpose = true;
      div_numerator_name = transpose_node_unit->Outputs()[0].name;
    } else {
      return nullptr;
    }
  }

  // ---- Step 3: Handle Transpose (Pattern B) ----
  if (has_transpose) {
    if (transpose_node_unit == nullptr) {
      return nullptr;
    }
    // Transpose has 2 consumers: Pow (variance path) and Div (numerator path).
    // Find Pow among Transpose's consumers.
    const Ort::ConstNode transpose_node(&transpose_node_unit->GetNode());
    const auto transpose_outputs = transpose_node.GetOutputs();
    if (transpose_outputs.size() != 1) {
      return nullptr;
    }
    const auto transpose_consumers = transpose_outputs[0].GetConsumers();
    if (transpose_consumers.size() != 2) {
      return nullptr;
    }
    for (const auto& consumer_info : transpose_consumers) {
      if (consumer_info.node == nullptr) {
        return nullptr;
      }
      const Ort::ConstNode consumer_node = consumer_info.node;
      if (std::string(consumer_node.GetOperatorType()) == "Pow") {
        const auto it = node_to_node_unit.find(consumer_node);
        if (it == node_to_node_unit.end()) {
          return nullptr;
        }
        pow_node_unit = it->second;
        if (node_unit_to_qnn_node_group.count(pow_node_unit) != 0 ||
            pow_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
          return nullptr;
        }
        break;
      }
    }
    if (pow_node_unit == nullptr) {
      return nullptr;
    }
  } else {
    // Pattern A: pow_node_unit should have been found above.
    if (pow_node_unit == nullptr) {
      return nullptr;
    }
  }

  // ---- Step 4: Pow(2) → ReduceMean₂ ----
  // Verify Pow exponent is 2.0.
  const auto& pow_inputs = pow_node_unit->Inputs();
  if (pow_inputs.size() < 2) {
    return nullptr;
  }
  const std::optional<float> pow_exp = GetConstantFloatScalar(qnn_model_wrapper, ort_api, pow_inputs[1].name);
  if (!pow_exp.has_value() || pow_exp.value() != 2.0f) {
    return nullptr;
  }

  // Pow → ReduceMean₂ (only child).
  const std::array<std::string_view, 1> rm_types{"ReduceMean"};
  const OrtNodeUnit* reduce_mean2_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *pow_node_unit,
                                                                  rm_types, node_to_node_unit,
                                                                  node_unit_to_qnn_node_group);
  if (reduce_mean2_node_unit == nullptr) {
    return nullptr;
  }

  // ---- Step 5: Verify axes match between ReduceMean₁ and ReduceMean₂ ----
  auto axes1_opt = GetReduceMeanAxes(qnn_model_wrapper, reduce_mean_node_unit);
  auto axes2_opt = GetReduceMeanAxes(qnn_model_wrapper, *reduce_mean2_node_unit);
  if (!axes1_opt.has_value() || !axes2_opt.has_value()) {
    return nullptr;
  }
  if (axes1_opt.value() != axes2_opt.value()) {
    return nullptr;
  }

  // Compute QNN axes.
  std::vector<uint32_t> input_shape;
  if (!qnn_model_wrapper.GetOnnxShape(rm1_inputs[0].shape, input_shape)) {
    return nullptr;
  }
  const size_t input_rank = input_shape.size();

  std::vector<uint32_t> qnn_axes;
  if (has_transpose) {
    // Pattern B: Transpose absorbed, QNN always sees axis = last dim.
    qnn_axes = {static_cast<uint32_t>(input_rank - 1)};
  } else {
    // Pattern A: axes must include the last dimension for HTP.
    const auto& axes = axes1_opt.value();
    if (axes.empty()) {
      return nullptr;
    }
    // Check that axes end at last dim (required by QNN HTP LayerNorm).
    if (axes.back() != static_cast<uint32_t>(input_rank - 1)) {
      return nullptr;
    }
    qnn_axes = axes;
  }

  // ---- Step 6: ReduceMean₂ → Add(ε) ----
  const std::array<std::string_view, 1> add_types{"Add"};
  const OrtNodeUnit* add_eps_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *reduce_mean2_node_unit,
                                                             add_types, node_to_node_unit,
                                                             node_unit_to_qnn_node_group);
  if (add_eps_node_unit == nullptr) {
    return nullptr;
  }

  // Extract epsilon from Add's constant input.
  const auto& add_eps_inputs = add_eps_node_unit->Inputs();
  if (add_eps_inputs.size() < 2) {
    return nullptr;
  }
  std::optional<float> epsilon_opt;
  for (const auto& inp : add_eps_inputs) {
    auto val = GetConstantFloatScalar(qnn_model_wrapper, ort_api, inp.name);
    if (val.has_value()) {
      epsilon_opt = val;
      break;
    }
  }
  if (!epsilon_opt.has_value()) {
    return nullptr;
  }
  const float epsilon = epsilon_opt.value();

  // ---- Step 7: Add(ε) → Sqrt ----
  const std::array<std::string_view, 1> sqrt_types{"Sqrt"};
  const OrtNodeUnit* sqrt_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *add_eps_node_unit,
                                                          sqrt_types, node_to_node_unit,
                                                          node_unit_to_qnn_node_group);
  if (sqrt_node_unit == nullptr) {
    return nullptr;
  }

  // ---- Step 8: Sqrt → Div ----
  const std::array<std::string_view, 1> div_types{"Div"};
  const OrtNodeUnit* div_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *sqrt_node_unit,
                                                         div_types, node_to_node_unit,
                                                         node_unit_to_qnn_node_group);
  if (div_node_unit == nullptr) {
    return nullptr;
  }

  // Verify Div input[0] is the numerator (Sub output for regular, Transpose output for transpose pattern).
  const auto& div_inputs = div_node_unit->Inputs();
  if (div_inputs.size() < 2) {
    return nullptr;
  }
  if (div_inputs[0].name != div_numerator_name) {
    return nullptr;
  }

  // ---- Step 9: Div → Mul(γ) ----
  const std::array<std::string_view, 1> mul_types{"Mul"};
  const OrtNodeUnit* mul_gamma_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *div_node_unit,
                                                               mul_types, node_to_node_unit,
                                                               node_unit_to_qnn_node_group);
  if (mul_gamma_node_unit == nullptr) {
    return nullptr;
  }

  // Find gamma: the constant input to Mul.
  const auto& mul_inputs = mul_gamma_node_unit->Inputs();
  if (mul_inputs.size() < 2) {
    return nullptr;
  }
  const OrtNodeUnitIODef* gamma_input_def = nullptr;
  for (const auto& inp : mul_inputs) {
    if (qnn_model_wrapper.IsConstantInput(inp.name)) {
      gamma_input_def = &inp;
      break;
    }
  }
  if (gamma_input_def == nullptr) {
    return nullptr;
  }

  // ---- Step 10: Mul(γ) → Add(β) ----
  const OrtNodeUnit* add_beta_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *mul_gamma_node_unit,
                                                              add_types, node_to_node_unit,
                                                              node_unit_to_qnn_node_group);
  if (add_beta_node_unit == nullptr) {
    return nullptr;
  }

  // Find beta: the constant input to the final Add.
  const auto& add_beta_inputs = add_beta_node_unit->Inputs();
  if (add_beta_inputs.size() < 2) {
    return nullptr;
  }
  const OrtNodeUnitIODef* beta_input_def = nullptr;
  for (const auto& inp : add_beta_inputs) {
    if (qnn_model_wrapper.IsConstantInput(inp.name)) {
      beta_input_def = &inp;
      break;
    }
  }
  if (beta_input_def == nullptr) {
    return nullptr;
  }

  // ---- Step 11: Collect node units and validate on QNN ----
  std::vector<const OrtNodeUnit*> node_units;
  if (has_transpose) {
    node_units = {&reduce_mean_node_unit, sub_node_unit, transpose_node_unit,
                  pow_node_unit, reduce_mean2_node_unit, add_eps_node_unit,
                  sqrt_node_unit, div_node_unit, mul_gamma_node_unit, add_beta_node_unit};
  } else {
    node_units = {&reduce_mean_node_unit, sub_node_unit,
                  pow_node_unit, reduce_mean2_node_unit, add_eps_node_unit,
                  sqrt_node_unit, div_node_unit, mul_gamma_node_unit, add_beta_node_unit};
  }

  const OrtNodeUnitIODef& root_input = rm1_inputs[0];
  const OrtNodeUnitIODef& final_output = add_beta_node_unit->Outputs()[0];

  std::string shape_str;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) shape_str += "x";
    shape_str += std::to_string(input_shape[i]);
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              ("LayerNormFusion: Fusing LayerNorm pattern (" +
               std::string(has_transpose ? "Transpose-wrapped" : "Regular") +
               ") starting at node: " + reduce_mean_node_unit.Name() +
               ", input_shape=" + shape_str +
               ", epsilon=" + std::to_string(epsilon) +
               ", axes=[" + std::to_string(qnn_axes[0]) + "]" +
               ", gamma=" + gamma_input_def->name +
               ", beta=" + beta_input_def->name).c_str());

  if (auto status = ValidateOnQnn(qnn_model_wrapper, node_units, root_input,
                                  *gamma_input_def, *beta_input_def, final_output,
                                  epsilon, qnn_axes);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<LayerNormFusion>(std::move(node_units), &reduce_mean_node_unit,
                                           epsilon, std::move(qnn_axes), has_transpose);
}

LayerNormFusion::LayerNormFusion(std::vector<const OrtNodeUnit*>&& node_units,
                                 const OrtNodeUnit* target_node_unit,
                                 float epsilon,
                                 std::vector<uint32_t> axes,
                                 bool has_transpose)
    : node_units_(std::move(node_units)),
      target_node_unit_(target_node_unit),
      epsilon_(epsilon),
      axes_(std::move(axes)),
      has_transpose_(has_transpose) {
}

Ort::Status LayerNormFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const OrtNodeUnit& rm1 = *node_units_[0];
  const OrtNodeUnit& add_beta = *node_units_.back();
  const OrtNodeUnit& mul_gamma = *node_units_[node_units_.size() - 2];

  const OrtNodeUnitIODef* gamma_input_def = nullptr;
  for (const auto& inp : mul_gamma.Inputs()) {
    if (qmw.IsConstantInput(inp.name)) {
      gamma_input_def = &inp;
      break;
    }
  }
  const OrtNodeUnitIODef* beta_input_def = nullptr;
  for (const auto& inp : add_beta.Inputs()) {
    if (qmw.IsConstantInput(inp.name)) {
      beta_input_def = &inp;
      break;
    }
  }
  if (!gamma_input_def || !beta_input_def) {
    return MAKE_EP_FAIL("LayerNormFusion: cannot find gamma or beta inputs.");
  }

  return ValidateOnQnn(qmw, node_units_, rm1.Inputs()[0],
                       *gamma_input_def, *beta_input_def,
                       add_beta.Outputs()[0], epsilon_, axes_);
}

Ort::Status LayerNormFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  const OrtNodeUnit& rm1 = *node_units_[0];
  const OrtNodeUnit& add_beta = *node_units_.back();
  const OrtNodeUnit& mul_gamma = *node_units_[node_units_.size() - 2];

  const OrtNodeUnitIODef* gamma_input_def = nullptr;
  for (const auto& inp : mul_gamma.Inputs()) {
    if (qmw.IsConstantInput(inp.name)) {
      gamma_input_def = &inp;
      break;
    }
  }
  const OrtNodeUnitIODef* beta_input_def = nullptr;
  for (const auto& inp : add_beta.Inputs()) {
    if (qmw.IsConstantInput(inp.name)) {
      beta_input_def = &inp;
      break;
    }
  }
  if (!gamma_input_def || !beta_input_def) {
    return MAKE_EP_FAIL("LayerNormFusion: cannot find gamma or beta inputs.");
  }

  auto status = CreateOnQnn(qmw, node_units_, rm1.Inputs()[0],
                            *gamma_input_def, *beta_input_def,
                            add_beta.Outputs()[0], epsilon_, axes_);
  if (status.IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
                ("LayerNormFusion: Successfully fused LayerNorm (" +
                 std::string(has_transpose_ ? "Transpose-wrapped" : "Regular") +
                 ") node: " + rm1.Name()).c_str());
  }
  return status;
}

gsl::span<const OrtNodeUnit* const> LayerNormFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>(node_units_.data(), node_units_.size());
}

const OrtNodeUnit* LayerNormFusion::GetTargetNodeUnit() const {
  return target_node_unit_;
}

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef& gamma_input,
                                         const OrtNodeUnitIODef& beta_input,
                                         const OrtNodeUnitIODef& final_output,
                                         float epsilon,
                                         gsl::span<const uint32_t> axes,
                                         bool validate) {
  assert(!node_units.empty());
  const std::string node_name = utils::GetUniqueName(*node_units[0]);

  // Build input/output tensor wrappers.
  QnnTensorWrapper input_tensor;
  QnnTensorWrapper gamma_tensor;
  QnnTensorWrapper beta_tensor;
  QnnTensorWrapper output_tensor;

  RETURN_IF_ERROR(qmw.MakeTensorWrapper(root_input, input_tensor));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(gamma_input, gamma_tensor));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(beta_input, beta_tensor));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(final_output, output_tensor));

  if (validate) {
    // For validation, build Qnn_Param_t structs directly and pass to ValidateQnnNode.
    Qnn_Scalar_t epsilon_scalar = QNN_SCALAR_INIT;
    epsilon_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    epsilon_scalar.floatValue = epsilon;
    QnnParamWrapper epsilon_param(node_units[0]->Index(), node_units[0]->Name(),
                                  QNN_OP_LAYER_NORM_PARAM_EPSILON, epsilon_scalar);

    std::vector<uint32_t> axes_vec(axes.begin(), axes.end());
    std::vector<uint32_t> axes_shape{static_cast<uint32_t>(axes_vec.size())};
    QnnParamWrapper axes_param(node_units[0]->Index(), node_units[0]->Name(),
                               QNN_OP_LAYER_NORM_PARAM_AXES,
                               std::move(axes_shape), std::move(axes_vec));

    RETURN_IF_ERROR(qmw.ValidateQnnNode(node_name,
                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                        QNN_OP_LAYER_NORM,
                                        {input_tensor.GetQnnTensor(),
                                         gamma_tensor.GetQnnTensor(),
                                         beta_tensor.GetQnnTensor()},
                                        {output_tensor.GetQnnTensor()},
                                        {epsilon_param.GetQnnParam(), axes_param.GetQnnParam()}));
  } else {
    // For creation, register params by name then create the node.
    Qnn_Scalar_t epsilon_scalar = QNN_SCALAR_INIT;
    epsilon_scalar.dataType = QNN_DATATYPE_FLOAT_32;
    epsilon_scalar.floatValue = epsilon;
    QnnParamWrapper epsilon_param(node_units[0]->Index(), node_units[0]->Name(),
                                  QNN_OP_LAYER_NORM_PARAM_EPSILON, epsilon_scalar);
    const std::string epsilon_param_name = epsilon_param.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(epsilon_param)), "Failed to add epsilon param.");

    std::vector<uint32_t> axes_vec(axes.begin(), axes.end());
    std::vector<uint32_t> axes_shape{static_cast<uint32_t>(axes_vec.size())};
    QnnParamWrapper axes_param(node_units[0]->Index(), node_units[0]->Name(),
                               QNN_OP_LAYER_NORM_PARAM_AXES,
                               std::move(axes_shape), std::move(axes_vec));
    const std::string axes_param_name = axes_param.GetParamTensorName();
    RETURN_IF_NOT(qmw.AddParamWrapper(std::move(axes_param)), "Failed to add axes param.");

    if (!qmw.IsQnnTensorWrapperExist(root_input.name)) {
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(input_tensor)), "Failed to add input tensor.");
    }
    if (!qmw.IsQnnTensorWrapperExist(gamma_input.name)) {
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(gamma_tensor)), "Failed to add gamma tensor.");
    }
    if (!qmw.IsQnnTensorWrapperExist(beta_input.name)) {
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(beta_tensor)), "Failed to add beta tensor.");
    }
    if (!qmw.IsQnnTensorWrapperExist(final_output.name)) {
      RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_tensor)), "Failed to add output tensor.");
    }

    RETURN_IF_NOT(qmw.CreateQnnNode(node_name,
                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                    QNN_OP_LAYER_NORM,
                                    {root_input.name, gamma_input.name, beta_input.name},
                                    {final_output.name},
                                    {epsilon_param_name, axes_param_name},
                                    validate),
                  "Failed to create fused LayerNorm node.");
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
