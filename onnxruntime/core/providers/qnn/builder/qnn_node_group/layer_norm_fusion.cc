// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/layer_norm_fusion.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <gsl/gsl>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

#define ValidateOnQnn(qmw, node_units, root_input, gamma_input, beta_input, final_output, epsilon, axes) \
  CreateOrValidateOnQnn((qmw), (node_units), (root_input), (gamma_input), (beta_input), (final_output), (epsilon), (axes), true)
#define CreateOnQnn(qmw, node_units, root_input, gamma_input, beta_input, final_output, epsilon, axes) \
  CreateOrValidateOnQnn((qmw), (node_units), (root_input), (gamma_input), (beta_input), (final_output), (epsilon), (axes), false)

// gamma_input / beta_input may be nullptr: a ones / zeros static tensor is synthesized instead.
static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef* gamma_input,
                                         const OrtNodeUnitIODef* beta_input,
                                         const OrtNodeUnitIODef& final_output,
                                         float epsilon,
                                         gsl::span<const uint32_t> axes,
                                         bool validate);

// Scalar constants (Pow exponent, epsilon) are read with GetScalarConstantValue (utils.h), which
// accepts both float32 and float16 initializers. fp16 models (e.g. exported with
// keep_io_types / float16 conversion) store these constants as FLOAT16.

// Returns the first constant input of `node_unit`, or nullptr if none.
static const OrtNodeUnitIODef* FindConstantInput(const QnnModelWrapper& qmw, const OrtNodeUnit& node_unit) {
  for (const auto& inp : node_unit.Inputs()) {
    if (qmw.IsConstantInput(inp.name)) {
      return &inp;
    }
  }
  return nullptr;
}

std::unique_ptr<IQnnNodeGroup> LayerNormFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reduce_mean_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // Must start with a ReduceMean SingleNode with keepdims=1.
  // keepdims=1 is required so that the mean output has the same rank as x,
  // allowing Sub(x, mean) to broadcast correctly.
  if (reduce_mean_node_unit.OpType() != "ReduceMean" ||
      reduce_mean_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }
  {
    OrtNodeAttrHelper rm1_helper(reduce_mean_node_unit);
    if (rm1_helper.Get("keepdims", static_cast<int64_t>(1)) != 1) {
      return nullptr;
    }
  }

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

  // Sub must have exactly 2 consumers: Pow and Div.
  const auto& sub_outputs = sub_node_unit->Outputs();
  if (sub_outputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* pow_node_unit = nullptr;
  const OrtNodeUnit* div_from_sub_unit = nullptr;
  {
    const Ort::ConstNode sub_node(&sub_node_unit->GetNode());
    const auto sub_node_outputs = sub_node.GetOutputs();
    if (sub_node_outputs.size() != 1) {
      return nullptr;
    }
    const auto consumers = sub_node_outputs[0].GetConsumers();
    if (consumers.size() != 2) {
      return nullptr;
    }
    for (const auto& consumer_info : consumers) {
      if (consumer_info.node == nullptr) {
        return nullptr;
      }
      const Ort::ConstNode consumer_node = consumer_info.node;
      const auto it = node_to_node_unit.find(consumer_node);
      if (it == node_to_node_unit.end()) {
        return nullptr;
      }
      const OrtNodeUnit* consumer_unit = it->second;
      if (node_unit_to_qnn_node_group.count(consumer_unit) != 0 ||
          consumer_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
        return nullptr;
      }
      const std::string op_type = consumer_node.GetOperatorType();
      if (op_type == "Pow") {
        pow_node_unit = consumer_unit;
      } else if (op_type == "Div") {
        div_from_sub_unit = consumer_unit;
      }
    }
  }
  if (pow_node_unit == nullptr || div_from_sub_unit == nullptr) {
    return nullptr;
  }

  // Pow exponent must be 2.0.
  const auto& pow_inputs = pow_node_unit->Inputs();
  if (pow_inputs.size() < 2) {
    return nullptr;
  }

  const std::optional<float> pow_exp = GetScalarConstantValue(qnn_model_wrapper, pow_inputs[1].name);
  if (!pow_exp.has_value() || pow_exp.value() != 2.0f) {
    return nullptr;
  }

  // Pow → ReduceMean₂
  const std::array<std::string_view, 1> rm_types{"ReduceMean"};
  const OrtNodeUnit* reduce_mean2_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *pow_node_unit,
                                                                 rm_types, node_to_node_unit,
                                                                 node_unit_to_qnn_node_group);
  if (reduce_mean2_node_unit == nullptr) {
    return nullptr;
  }
  {
    OrtNodeAttrHelper rm2_helper(*reduce_mean2_node_unit);
    if (rm2_helper.Get("keepdims", static_cast<int64_t>(1)) != 1) {
      return nullptr;
    }
  }

  // Both ReduceMeans must have the same axes, pointing to the last dimension.
  auto axes1_opt = GetReduceAxes(qnn_model_wrapper, reduce_mean_node_unit);
  auto axes2_opt = GetReduceAxes(qnn_model_wrapper, *reduce_mean2_node_unit);
  if (!axes1_opt.has_value() || !axes2_opt.has_value()) {
    return nullptr;
  }

  // GetReduceAxes already returns sorted, deduplicated axes — compare directly.
  const auto& axes1 = axes1_opt.value();
  const auto& axes2 = axes2_opt.value();

  if (axes1 != axes2) {
    return nullptr;
  }
  std::vector<uint32_t> input_shape;
  if (!qnn_model_wrapper.GetOnnxShape(rm1_inputs[0].shape, input_shape)) {
    return nullptr;
  }
  const size_t input_rank = input_shape.size();

  // Axes must be a contiguous trailing suffix ending at the last dimension.
  // e.g. {1,2} on rank-3 is valid; {0,2} or {0,1} on rank-3 are not.
  if (axes1.empty() || axes1.back() != static_cast<uint32_t>(input_rank - 1)) {
    return nullptr;
  }
  {
    const uint32_t M = static_cast<uint32_t>(axes1.size());
    for (uint32_t i = 0; i < M; ++i) {
      if (axes1[i] != static_cast<uint32_t>(input_rank) - M + i) {
        return nullptr;
      }
    }
  }

  // ReduceMean₂ → Add(ε)
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
    auto val = GetScalarConstantValue(qnn_model_wrapper, inp.name);
    if (val.has_value()) {
      epsilon_opt = val;
      break;
    }
  }
  if (!epsilon_opt.has_value()) {
    return nullptr;
  }
  const float epsilon = epsilon_opt.value();

  // Add(ε) → Sqrt
  const std::array<std::string_view, 1> sqrt_types{"Sqrt"};
  const OrtNodeUnit* sqrt_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *add_eps_node_unit,
                                                         sqrt_types, node_to_node_unit,
                                                         node_unit_to_qnn_node_group);
  if (sqrt_node_unit == nullptr) {
    return nullptr;
  }

  // Sqrt → Div
  const std::array<std::string_view, 1> div_types{"Div"};
  const OrtNodeUnit* div_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *sqrt_node_unit,
                                                        div_types, node_to_node_unit,
                                                        node_unit_to_qnn_node_group);
  if (div_node_unit == nullptr || div_node_unit != div_from_sub_unit) {
    return nullptr;
  }
  const auto& div_inputs = div_node_unit->Inputs();
  if (div_inputs.size() < 2 || div_inputs[0].name != sub_outputs[0].name) {
    return nullptr;
  }

  // ---- Optional affine tail: Div → Mul(γ) → Add(β) ----
  // Try the longest fusable tail first, then fall back to shorter ones. Each candidate is
  // validated on QNN (which also checks gamma/beta shapes), so an unsupported gamma/beta
  // simply degrades to a shorter fusion instead of rejecting LayerNorm altogether.
  //   1. Full:       LayerNorm(x, γ, β)       — Div has a single Mul child with a constant,
  //                                             which has a single Add child with a constant.
  //   2. Gamma only: LayerNorm(x, γ, zeros)   — Add(β) missing / not fusable.
  //   3. Core only:  LayerNorm(x, ones, zeros) — Div output shared by several consumers
  //                                             (e.g. multi-head norm) or Mul(γ) not fusable.
  const std::array<std::string_view, 1> mul_types{"Mul"};
  const OrtNodeUnit* mul_gamma_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *div_node_unit,
                                                              mul_types, node_to_node_unit,
                                                              node_unit_to_qnn_node_group);
  const OrtNodeUnitIODef* gamma_input_def =
      mul_gamma_node_unit ? FindConstantInput(qnn_model_wrapper, *mul_gamma_node_unit) : nullptr;
  if (gamma_input_def == nullptr) {
    mul_gamma_node_unit = nullptr;
  }

  const OrtNodeUnit* add_beta_node_unit = nullptr;
  const OrtNodeUnitIODef* beta_input_def = nullptr;
  if (mul_gamma_node_unit != nullptr) {
    add_beta_node_unit = GetOnlyChildOfType(qnn_model_wrapper, *mul_gamma_node_unit,
                                            add_types, node_to_node_unit,
                                            node_unit_to_qnn_node_group);
    beta_input_def = add_beta_node_unit ? FindConstantInput(qnn_model_wrapper, *add_beta_node_unit) : nullptr;
    if (beta_input_def == nullptr) {
      add_beta_node_unit = nullptr;
    }
  }

  const std::vector<const OrtNodeUnit*> core_units = {
      &reduce_mean_node_unit, sub_node_unit, pow_node_unit, reduce_mean2_node_unit,
      add_eps_node_unit, sqrt_node_unit, div_node_unit};
  const OrtNodeUnitIODef& root_input = rm1_inputs[0];

  struct Candidate {
    const OrtNodeUnit* mul_gamma;
    const OrtNodeUnitIODef* gamma;
    const OrtNodeUnit* add_beta;
    const OrtNodeUnitIODef* beta;
    const char* kind;
  };
  std::vector<Candidate> candidates;
  if (add_beta_node_unit != nullptr) {
    candidates.push_back({mul_gamma_node_unit, gamma_input_def, add_beta_node_unit, beta_input_def, "full"});
  }
  if (mul_gamma_node_unit != nullptr) {
    candidates.push_back({mul_gamma_node_unit, gamma_input_def, nullptr, nullptr, "gamma-only"});
  }
  candidates.push_back({nullptr, nullptr, nullptr, nullptr, "core-only"});

  std::string shape_str;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (i > 0) shape_str += "x";
    shape_str += std::to_string(input_shape[i]);
  }
  std::string axes_str;
  for (size_t i = 0; i < axes1.size(); ++i) {
    if (i > 0) axes_str += ",";
    axes_str += std::to_string(axes1[i]);
  }

  for (const Candidate& c : candidates) {
    std::vector<const OrtNodeUnit*> node_units = core_units;
    if (c.mul_gamma) node_units.push_back(c.mul_gamma);
    if (c.add_beta) node_units.push_back(c.add_beta);
    const OrtNodeUnitIODef& final_output = node_units.back()->Outputs()[0];

    if (auto status = ValidateOnQnn(qnn_model_wrapper, node_units, root_input,
                                    c.gamma, c.beta, final_output, epsilon, axes1);
        !status.IsOK()) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                  ("LayerNormFusion: " + std::string(c.kind) + " fusion rejected at " +
                   reduce_mean_node_unit.Name() + ": " + status.GetErrorMessage())
                      .c_str());
      continue;
    }

    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("LayerNormFusion: Fusing " + std::string(c.kind) + " LayerNorm starting at node: " +
                 reduce_mean_node_unit.Name() +
                 ", input_shape=" + shape_str +
                 ", epsilon=" + std::to_string(epsilon) +
                 ", axes=[" + axes_str + "]" +
                 ", gamma=" + (c.gamma ? c.gamma->name : std::string("<ones>")) +
                 ", beta=" + (c.beta ? c.beta->name : std::string("<zeros>")))
                    .c_str());

    return std::make_unique<LayerNormFusion>(std::move(node_units), &reduce_mean_node_unit,
                                             epsilon, axes1,
                                             c.gamma ? c.gamma->name : std::string(),
                                             c.beta ? c.beta->name : std::string());
  }

  return nullptr;
}

LayerNormFusion::LayerNormFusion(std::vector<const OrtNodeUnit*>&& node_units,
                                 const OrtNodeUnit* target_node_unit,
                                 float epsilon,
                                 std::vector<uint32_t> axes,
                                 std::string gamma_input_name,
                                 std::string beta_input_name)
    : node_units_(std::move(node_units)),
      target_node_unit_(target_node_unit),
      epsilon_(epsilon),
      axes_(std::move(axes)),
      gamma_input_name_(std::move(gamma_input_name)),
      beta_input_name_(std::move(beta_input_name)) {
}

// Finds the OrtNodeUnitIODef with the given name among the inputs of any fused node unit.
// An empty name means "not present" (synthesized ones/zeros) and yields nullptr with ok=true.
static const OrtNodeUnitIODef* FindInputByName(gsl::span<const OrtNodeUnit* const> node_units,
                                               const std::string& name, bool& ok) {
  ok = true;
  if (name.empty()) {
    return nullptr;
  }
  for (const OrtNodeUnit* nu : node_units) {
    for (const auto& inp : nu->Inputs()) {
      if (inp.name == name) {
        return &inp;
      }
    }
  }
  ok = false;
  return nullptr;
}

Ort::Status LayerNormFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  bool gamma_ok = false, beta_ok = false;
  const OrtNodeUnitIODef* gamma_def = FindInputByName(node_units_, gamma_input_name_, gamma_ok);
  const OrtNodeUnitIODef* beta_def = FindInputByName(node_units_, beta_input_name_, beta_ok);
  if (!gamma_ok || !beta_ok) {
    return MAKE_EP_FAIL("LayerNormFusion: cannot find gamma or beta inputs.");
  }

  return ValidateOnQnn(qmw, node_units_, node_units_[0]->Inputs()[0],
                       gamma_def, beta_def,
                       node_units_.back()->Outputs()[0], epsilon_, axes_);
}

Ort::Status LayerNormFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  bool gamma_ok = false, beta_ok = false;
  const OrtNodeUnitIODef* gamma_def = FindInputByName(node_units_, gamma_input_name_, gamma_ok);
  const OrtNodeUnitIODef* beta_def = FindInputByName(node_units_, beta_input_name_, beta_ok);
  if (!gamma_ok || !beta_ok) {
    return MAKE_EP_FAIL("LayerNormFusion: cannot find gamma or beta inputs.");
  }

  const OrtNodeUnit& rm1 = *node_units_[0];
  auto status = CreateOnQnn(qmw, node_units_, rm1.Inputs()[0],
                            gamma_def, beta_def,
                            node_units_.back()->Outputs()[0], epsilon_, axes_);
  if (status.IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
                ("LayerNormFusion: Successfully fused LayerNorm node: " + rm1.Name()).c_str());
  }
  return status;
}

gsl::span<const OrtNodeUnit* const> LayerNormFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>(node_units_.data(), node_units_.size());
}

const OrtNodeUnit* LayerNormFusion::GetTargetNodeUnit() const {
  return target_node_unit_;
}

// Validates that a gamma/beta shape is compatible with LayerNorm over the given axes,
// then squeezes it to 1D. Rules:
//   - All non-normalized dims must be 1 (no scaling along non-normalized axes).
//   - All normalized dims must match the input shape at those axes.
// Returns an error status if the shape is invalid.
static Ort::Status ValidateAndSqueezeScaleShape(
    const std::vector<uint32_t>& scale_shape,
    const std::vector<uint32_t>& input_shape,
    gsl::span<const uint32_t> axes,
    /*out*/ std::vector<uint32_t>& squeezed_shape) {
  // Normalize axes: sort and deduplicate.
  std::vector<uint32_t> normalized_axes(axes.begin(), axes.end());
  std::sort(normalized_axes.begin(), normalized_axes.end());
  normalized_axes.erase(std::unique(normalized_axes.begin(), normalized_axes.end()),
                        normalized_axes.end());

  // If already the right rank, verify each dim matches the input at the corresponding axis.
  if (scale_shape.size() == normalized_axes.size()) {
    for (size_t i = 0; i < normalized_axes.size(); ++i) {
      if (scale_shape[i] != input_shape[normalized_axes[i]]) {
        return MAKE_EP_FAIL("LayerNormFusion: scale shape dimension mismatch at normalized axis.");
      }
    }
    squeezed_shape = scale_shape;
    return Ort::Status();
  }

  // Must have same rank as input to check per-dim semantics.
  if (scale_shape.size() != input_shape.size()) {
    return MAKE_EP_FAIL("LayerNormFusion: scale shape rank must match input rank or normalized axes count.");
  }

  std::unordered_set<uint32_t> axes_set(normalized_axes.begin(), normalized_axes.end());
  squeezed_shape.clear();
  squeezed_shape.reserve(normalized_axes.size());

  for (size_t i = 0; i < scale_shape.size(); ++i) {
    if (axes_set.count(static_cast<uint32_t>(i))) {
      // Normalized axis: dim must match input.
      if (scale_shape[i] != input_shape[i]) {
        return MAKE_EP_FAIL("LayerNormFusion: scale shape dimension mismatch at normalized axis.");
      }
      squeezed_shape.push_back(scale_shape[i]);
    } else {
      // Non-normalized axis: dim must be 1.
      if (scale_shape[i] != 1) {
        return MAKE_EP_FAIL("LayerNormFusion: scale shape must be 1 at non-normalized axes.");
      }
    }
  }

  return Ort::Status();
}

// Builds a static tensor filled with `value`, shaped like the normalized axes of the input, using
// the input's float data type. Used when gamma (ones) or beta (zeros) is not part of the fusion.
static Ort::Status MakeConstantScaleTensor(const std::string& name,
                                           Qnn_DataType_t data_type,
                                           const std::vector<uint32_t>& input_shape,
                                           gsl::span<const uint32_t> axes,
                                           float value,
                                           QnnTensorWrapper& tensor) {
  std::vector<uint32_t> shape;
  shape.reserve(axes.size());
  size_t num_elems = 1;
  for (uint32_t ax : axes) {
    shape.push_back(input_shape[ax]);
    num_elems *= input_shape[ax];
  }

  std::vector<uint8_t> bytes;
  if (data_type == QNN_DATATYPE_FLOAT_32) {
    std::vector<float> vals(num_elems, value);
    bytes.resize(num_elems * sizeof(float));
    std::memcpy(bytes.data(), vals.data(), bytes.size());
  } else if (data_type == QNN_DATATYPE_FLOAT_16) {
    const Ort::Float16_t v(value);
    bytes.resize(num_elems * sizeof(uint16_t));
    for (size_t i = 0; i < num_elems; ++i) {
      std::memcpy(bytes.data() + i * sizeof(uint16_t), &v.val, sizeof(uint16_t));
    }
  } else {
    return MAKE_EP_FAIL("LayerNormFusion: synthesized gamma/beta only supported for float32/float16 input.");
  }

  tensor = QnnTensorWrapper(name, QNN_TENSOR_TYPE_STATIC, data_type, QnnQuantParamsWrapper(),
                            std::move(shape), std::move(bytes));
  return Ort::Status();
}

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef* gamma_input,
                                         const OrtNodeUnitIODef* beta_input,
                                         const OrtNodeUnitIODef& final_output,
                                         float epsilon,
                                         gsl::span<const uint32_t> axes,
                                         bool validate) {
  const std::string node_name = utils::UniqueNameGenerator().New(*node_units[0]);

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper gamma_tensor;
  QnnTensorWrapper beta_tensor;
  QnnTensorWrapper output_tensor;

  RETURN_IF_ERROR(qmw.MakeTensorWrapper(root_input, input_tensor));
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(final_output, output_tensor));

  // Gamma and beta must only scale along the normalized axes.
  // Non-normalized dims must be 1; normalized dims must match the input.
  // The validated shape is then squeezed to 1D for QNN.
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qmw.GetOnnxShape(root_input.shape, input_shape), "Cannot get input shape.");

  // Synthesized gamma/beta: unquantized float of the same type as the input.
  const Qnn_DataType_t input_dtype = input_tensor.GetTensorDataType();
  const bool needs_synth = (gamma_input == nullptr || beta_input == nullptr);
  RETURN_IF(needs_synth && input_tensor.GetQnnQuantParams().IsQuantized(),
            "LayerNormFusion: partial (no gamma/beta) fusion not supported for quantized input.");

  std::string gamma_name;
  if (gamma_input != nullptr) {
    TensorInfo gamma_info = {};
    RETURN_IF_ERROR(qmw.GetTensorInfo(*gamma_input, gamma_info));
    std::vector<uint32_t> squeezed_gamma;
    RETURN_IF_ERROR(ValidateAndSqueezeScaleShape(gamma_info.shape, input_shape, axes, squeezed_gamma));
    gamma_info.shape = std::move(squeezed_gamma);
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(gamma_info, gamma_input->name, gamma_tensor));
    gamma_name = gamma_input->name;
  } else {
    gamma_name = utils::UniqueNameGenerator().New(node_name, "_gamma_ones");
    RETURN_IF_ERROR(MakeConstantScaleTensor(gamma_name, input_dtype, input_shape, axes, 1.0f, gamma_tensor));
  }

  std::string beta_name;
  if (beta_input != nullptr) {
    TensorInfo beta_info = {};
    RETURN_IF_ERROR(qmw.GetTensorInfo(*beta_input, beta_info));
    std::vector<uint32_t> squeezed_beta;
    RETURN_IF_ERROR(ValidateAndSqueezeScaleShape(beta_info.shape, input_shape, axes, squeezed_beta));
    beta_info.shape = std::move(squeezed_beta);
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(beta_info, beta_input->name, beta_tensor));
    beta_name = beta_input->name;
  } else {
    beta_name = utils::UniqueNameGenerator().New(node_name, "_beta_zeros");
    RETURN_IF_ERROR(MakeConstantScaleTensor(beta_name, input_dtype, input_shape, axes, 0.0f, beta_tensor));
  }

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

  if (validate) {
    return qmw.ValidateQnnNode(node_name,
                               QNN_OP_PACKAGE_NAME_QTI_AISW,
                               QNN_OP_LAYER_NORM,
                               {input_tensor.GetQnnTensor(),
                                gamma_tensor.GetQnnTensor(),
                                beta_tensor.GetQnnTensor()},
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

  if (!qmw.IsQnnTensorWrapperExist(gamma_name)) {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(gamma_tensor)), "Failed to add gamma tensor.");
  }

  if (!qmw.IsQnnTensorWrapperExist(beta_name)) {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(beta_tensor)), "Failed to add beta tensor.");
  }
  if (!qmw.IsQnnTensorWrapperExist(final_output.name)) {
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_tensor)), "Failed to add output tensor.");
  }

  RETURN_IF_NOT(qmw.CreateQnnNode(node_name,
                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                  QNN_OP_LAYER_NORM,
                                  {root_input.name, gamma_name, beta_name},
                                  {final_output.name},
                                  {epsilon_param_name, axes_param_name},
                                  validate),
                "Failed to create fused LayerNorm node.");

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
