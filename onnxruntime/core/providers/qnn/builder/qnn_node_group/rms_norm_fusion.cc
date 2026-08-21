// Copyright (c) Qualcomm Technologies, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/rms_norm_fusion.h"

#include <gsl/gsl>
#include <cassert>
#include <cmath>
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
#define ValidateOnQnn(qnn_model_wrapper, node_units) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), true)
#define CreateOnQnn(qnn_model_wrapper, node_units) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), false)

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const std::vector<const NodeUnit*>& node_units,
                                    bool validate);

// Helper: check if a node's input is a scalar constant (initializer or Constant op)
static bool IsSmallConstantInput(const GraphViewer& graph_viewer, const Node& node, size_t input_idx) {
  if (input_idx >= node.InputDefs().size()) return false;
  const auto* input_arg = node.InputDefs()[input_idx];
  if (!input_arg || !input_arg->Exists()) return false;

  // Case 1: initializer tensor
  const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
  if (graph_viewer.GetInitializedTensor(input_arg->Name(), tensor) && tensor) {
    int64_t num_elements = 1;
    for (int i = 0; i < tensor->dims_size(); ++i) num_elements *= tensor->dims(i);
    return num_elements == 1;
  }

  // Case 2: produced by a Constant node
  const Node* producer = graph_viewer.GetProducerNode(input_arg->Name());
  if (producer && producer->OpType() == "Constant") {
    const auto* value_attr = producer->GetAttributes().count("value")
                                 ? &producer->GetAttributes().at("value")
                                 : nullptr;
    if (value_attr && value_attr->has_t()) {
      const auto& t = value_attr->t();
      int64_t num_elements = 1;
      for (auto dim : t.dims()) num_elements *= dim;
      return num_elements <= 1;  // scalar or 1-element
    }
  }
  return false;
}

// Helper: get epsilon value from a constant input (initializer or Constant op)
static float GetEpsilonValue(const GraphViewer& graph_viewer, const Node& node, size_t input_idx) {
  const auto* input_arg = node.InputDefs()[input_idx];

  // Case 1: initializer
  const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
  if (graph_viewer.GetInitializedTensor(input_arg->Name(), tensor) && tensor) {
    if (tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      if (tensor->float_data_size() > 0) return tensor->float_data(0);
      if (tensor->has_raw_data() && tensor->raw_data().size() >= sizeof(float)) {
        float val;
        memcpy(&val, tensor->raw_data().data(), sizeof(float));
        return val;
      }
    }
  }

  // Case 2: Constant node
  const Node* producer = graph_viewer.GetProducerNode(input_arg->Name());
  if (producer && producer->OpType() == "Constant") {
    const auto* value_attr = producer->GetAttributes().count("value")
                                 ? &producer->GetAttributes().at("value")
                                 : nullptr;
    if (value_attr && value_attr->has_t()) {
      const auto& t = value_attr->t();
      if (t.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        if (t.float_data_size() > 0) return t.float_data(0);
        if (t.has_raw_data() && t.raw_data().size() >= sizeof(float)) {
          float val;
          memcpy(&val, t.raw_data().data(), sizeof(float));
          return val;
        }
      }
    }
  }
  return 1e-5f;
}

std::unique_ptr<IQnnNodeGroup> RmsNormFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& mul_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  // ============================================================================
  // Pattern: Mul(x,x) -> ReduceMean -> Add(eps) -> Sqrt -> Div(x,sqrt) -> Mul(gamma) [-> Add(beta)]
  //
  // Entry point: a Mul node where both inputs are the SAME tensor (x * x).
  // ============================================================================

  // Step 1: Verify this is a Mul with both inputs being the same (x * x)
  if (mul_node_unit.OpType() != "Mul" || mul_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  const Node& mul_node = mul_node_unit.GetNode();
  const auto& mul_inputs = mul_node.InputDefs();
  if (mul_inputs.size() < 2) return nullptr;

  // Both inputs must be the same tensor (x * x)
  const std::string& x_input_name = mul_inputs[0]->Name();
  if (mul_inputs[1]->Name() != x_input_name) {
    return nullptr;  // Not x * x pattern
  }

  LOGS(logger, VERBOSE) << "[RmsNormFusion] Step 1 passed: Mul(x,x) found at " << mul_node_unit.Name()
                        << " x=" << x_input_name;

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Step 2: Mul(x,x) -> ReduceMean
  const std::array<std::string_view, 1> reduce_types = {"ReduceMean"};
  const NodeUnit* reduce_node_unit = GetOnlyChildOfType(graph_viewer, mul_node_unit, reduce_types,
                                                        node_to_node_unit, node_unit_to_qnn_node_group);
  if (!reduce_node_unit) return nullptr;

  // Step 3: ReduceMean -> Add(epsilon)
  const std::array<std::string_view, 1> add_types = {"Add"};
  const NodeUnit* add_eps_node_unit = GetOnlyChildOfType(graph_viewer, *reduce_node_unit, add_types,
                                                         node_to_node_unit, node_unit_to_qnn_node_group);
  if (!add_eps_node_unit) return nullptr;

  // Validate: one input of Add must be a small constant (epsilon)
  const Node& add_eps_node = add_eps_node_unit->GetNode();
  bool has_epsilon = IsSmallConstantInput(graph_viewer, add_eps_node, 0) ||
                     IsSmallConstantInput(graph_viewer, add_eps_node, 1);
  if (!has_epsilon) return nullptr;

  // Step 4: Add(eps) -> Sqrt
  const std::array<std::string_view, 1> sqrt_types = {"Sqrt"};
  const NodeUnit* sqrt_node_unit = GetOnlyChildOfType(graph_viewer, *add_eps_node_unit, sqrt_types,
                                                      node_to_node_unit, node_unit_to_qnn_node_group);
  if (!sqrt_node_unit) return nullptr;

  // Step 5: Sqrt -> Div(x, sqrt_result)
  const std::array<std::string_view, 1> div_types = {"Div"};
  const NodeUnit* div_node_unit = GetOnlyChildOfType(graph_viewer, *sqrt_node_unit, div_types,
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
  if (!div_node_unit) return nullptr;

  // Validate: Div's first input must be the original x (not the sqrt output)
  const Node& div_node = div_node_unit->GetNode();
  if (div_node.InputDefs()[0]->Name() != x_input_name) {
    // x must be the numerator of the division
    return nullptr;
  }

  // Step 6: Div -> Mul(gamma) — affine scaling with weight
  const std::array<std::string_view, 1> mul_types = {"Mul"};
  const NodeUnit* gamma_mul_node_unit = GetOnlyChildOfType(graph_viewer, *div_node_unit, mul_types,
                                                           node_to_node_unit, node_unit_to_qnn_node_group);
  if (!gamma_mul_node_unit) return nullptr;

  // Validate: one input of gamma Mul must be a static weight (gamma)
  const Node& gamma_mul_node = gamma_mul_node_unit->GetNode();
  bool has_gamma_weight = false;
  for (size_t i = 0; i < gamma_mul_node.InputDefs().size(); ++i) {
    const auto* arg = gamma_mul_node.InputDefs()[i];
    if (arg && arg->Exists()) {
      const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
      if (graph_viewer.GetInitializedTensor(arg->Name(), tensor)) {
        has_gamma_weight = true;
        break;
      }
    }
  }
  if (!has_gamma_weight) return nullptr;

  // Step 7 (optional): Mul(gamma) -> Add(beta) — optional bias
  const NodeUnit* beta_add_node_unit = nullptr;
  {
    const std::array<std::string_view, 1> add_beta_types = {"Add"};
    const NodeUnit* candidate = GetOnlyChildOfType(graph_viewer, *gamma_mul_node_unit, add_beta_types,
                                                   node_to_node_unit, node_unit_to_qnn_node_group);
    if (candidate) {
      const Node& candidate_node = candidate->GetNode();
      // Check that one input is a static tensor (beta weight)
      for (size_t i = 0; i < candidate_node.InputDefs().size(); ++i) {
        const auto* arg = candidate_node.InputDefs()[i];
        if (arg && arg->Exists()) {
          const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
          if (graph_viewer.GetInitializedTensor(arg->Name(), tensor)) {
            beta_add_node_unit = candidate;
            break;
          }
        }
      }
    }
  }

  // Step 8 (optional): trailing Cast — if the last node (beta Add or gamma Mul) has a single
  // Cast child, include it. The Cast output is the RmsNorm output tensor which typically
  // has a quantization encoding in the JSON (e.g., /rms_norm_0/Cast_1_output_0).
  const NodeUnit* last_pattern_node = beta_add_node_unit ? beta_add_node_unit : gamma_mul_node_unit;
  const NodeUnit* trailing_cast_node_unit = nullptr;
  {
    const std::array<std::string_view, 1> cast_types = {"Cast"};
    const NodeUnit* candidate = GetOnlyChildOfType(graph_viewer, *last_pattern_node, cast_types,
                                                   node_to_node_unit, node_unit_to_qnn_node_group);
    if (candidate && candidate->UnitType() == NodeUnit::Type::SingleNode) {
      trailing_cast_node_unit = candidate;
    }
  }

  // Also check for leading Cast — if the Mul(x,x) input comes from a Cast
  const NodeUnit* leading_cast_node_unit = nullptr;
  {
    const std::array<std::string_view, 1> cast_types = {"Cast"};
    const NodeUnit* candidate = GetParentOfType(graph_viewer, mul_node_unit, cast_types,
                                                node_to_node_unit, node_unit_to_qnn_node_group);
    if (candidate && candidate->UnitType() == NodeUnit::Type::SingleNode) {
      // The Cast output feeds Mul(x,x) (twice — both inputs) and Div(x, sqrt) = 3 edges.
      // Just verify the Cast is the direct parent of our Mul's input.
      leading_cast_node_unit = candidate;
    }
  }

  // Collect all node units in the fusion
  std::vector<const NodeUnit*> fused_nodes;
  if (leading_cast_node_unit) {
    fused_nodes.push_back(leading_cast_node_unit);  // Leading Cast (optional)
  }
  fused_nodes.push_back(&mul_node_unit);       // Mul(x, x)
  fused_nodes.push_back(reduce_node_unit);     // ReduceMean
  fused_nodes.push_back(add_eps_node_unit);    // Add(epsilon)
  fused_nodes.push_back(sqrt_node_unit);       // Sqrt
  fused_nodes.push_back(div_node_unit);        // Div(x, sqrt)
  fused_nodes.push_back(gamma_mul_node_unit);  // Mul(gamma)
  if (beta_add_node_unit) {
    fused_nodes.push_back(beta_add_node_unit);  // Add(beta) optional
  }
  if (trailing_cast_node_unit) {
    fused_nodes.push_back(trailing_cast_node_unit);  // Trailing Cast (optional)
  }

  LOGS(logger, VERBOSE) << "[RmsNormFusion] Pattern matched starting at: " << mul_node_unit.Name()
                        << " (" << fused_nodes.size() << " nodes"
                        << ", leadCast=" << (leading_cast_node_unit ? "yes" : "no")
                        << ", trailCast=" << (trailing_cast_node_unit ? "yes" : "no") << ")";

  // Validate on QNN
  auto fusion = std::make_unique<RmsNormFusion>(std::move(fused_nodes));
  if (Status status = fusion->IsSupported(qnn_model_wrapper, logger); !status.IsOK()) {
    LOGS(logger, VERBOSE) << "[RmsNormFusion] Not supported on QNN: " << status.ErrorMessage();
    return nullptr;
  }

  return fusion;
}

RmsNormFusion::RmsNormFusion(std::vector<const NodeUnit*> node_units)
    : node_units_(std::move(node_units)) {}

Status RmsNormFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, node_units_);
}

Status RmsNormFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, node_units_);
}

gsl::span<const NodeUnit* const> RmsNormFusion::GetNodeUnits() const {
  return node_units_;
}

const NodeUnit* RmsNormFusion::GetTargetNodeUnit() const {
  // The Div node is where input paths converge (x from original, sqrt from the mean path)
  // Find it by type since indices shift with leading Cast
  for (const auto* nu : node_units_) {
    if (nu->OpType() == "Div") return nu;
  }
  return node_units_[0];  // fallback
}

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const std::vector<const NodeUnit*>& node_units,
                                    bool validate) {
  // Detect leading/trailing Cast by checking first and last node types
  size_t idx = 0;
  const bool has_leading_cast = (node_units[0]->OpType() == "Cast");
  if (has_leading_cast) idx++;  // skip leading Cast for core pattern

  const NodeUnit& mul_xx = *node_units[idx++];      // Mul(x, x)
  const NodeUnit& reduce_mean = *node_units[idx++]; // ReduceMean
  const NodeUnit& add_eps = *node_units[idx++];     // Add(eps)
  idx++;                                             // Sqrt (skip, not needed below)
  const NodeUnit& div = *node_units[idx++];         // Div
  const NodeUnit& gamma_mul = *node_units[idx++];   // Mul(gamma)

  const NodeUnit* beta_add = nullptr;
  if (idx < node_units.size() && node_units[idx]->OpType() == "Add") {
    beta_add = node_units[idx++];
  }

  const bool has_trailing_cast = (idx < node_units.size() && node_units[idx]->OpType() == "Cast");
  const NodeUnit* trailing_cast = has_trailing_cast ? node_units[idx] : nullptr;

  // Get the unique name for the fused node
  const auto& node_name = utils::GetUniqueName(div);

  // Input: if leading Cast, use Cast's input; otherwise Mul's input
  const NodeUnitIODef& x_input_def = has_leading_cast
      ? node_units[0]->Inputs()[0]   // Leading Cast's input (original tensor)
      : mul_xx.Inputs()[0];          // Mul's input

  // Output: if trailing Cast, use Cast's output; otherwise last core node's output
  const NodeUnit& output_node = trailing_cast ? *trailing_cast
                                : (beta_add ? *beta_add : gamma_mul);
  const NodeUnitIODef& output_def = output_node.Outputs()[0];

  // Gamma weight: the static input to Mul(gamma)
  const Node& gamma_node = gamma_mul.GetNode();
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  size_t gamma_idx = 0;
  for (size_t i = 0; i < gamma_node.InputDefs().size(); ++i) {
    const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
    if (graph_viewer.GetInitializedTensor(gamma_node.InputDefs()[i]->Name(), tensor)) {
      gamma_idx = i;
      break;
    }
  }
  const NodeUnitIODef& gamma_input_def = gamma_mul.Inputs()[gamma_idx];

  // Epsilon value
  const Node& eps_node = add_eps.GetNode();
  size_t eps_idx = IsSmallConstantInput(graph_viewer, eps_node, 0) ? 0 : 1;
  float epsilon = GetEpsilonValue(graph_viewer, eps_node, eps_idx);

  // Axes: from ReduceMean
  NodeAttrHelper reduce_attr(reduce_mean);
  auto axes = reduce_attr.Get("axes", std::vector<int64_t>{-1});

  // Build QNN tensors
  QnnTensorWrapper input_tensor;
  QnnTensorWrapper gamma_tensor;
  QnnTensorWrapper output_tensor;

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(x_input_def, input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(gamma_input_def, gamma_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  // Build params: epsilon and axes
  Qnn_Scalar_t epsilon_param = QNN_SCALAR_INIT;
  epsilon_param.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_param.floatValue = epsilon;

  // Convert axes to uint32_t for QNN
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(x_input_def.node_arg, input_shape),
                    "Cannot get shape of RmsNorm input");
  const size_t input_rank = input_shape.size();

  std::vector<uint32_t> qnn_axes;
  for (auto axis : axes) {
    int32_t resolved = static_cast<int32_t>(axis < 0 ? axis + static_cast<int64_t>(input_rank) : axis);
    qnn_axes.push_back(static_cast<uint32_t>(resolved));
  }
  std::vector<uint32_t> axes_shape{static_cast<uint32_t>(qnn_axes.size())};

  if (validate) {
    // Build param wrappers for validation
    QnnParamWrapper eps_pw(div.Index(), node_name, QNN_OP_RMS_NORM_PARAM_EPSILON, epsilon_param);
    QnnParamWrapper axes_pw(div.Index(), node_name, QNN_OP_RMS_NORM_PARAM_AXES,
                            std::vector<uint32_t>(axes_shape), std::vector<uint32_t>(qnn_axes));

    std::vector<Qnn_Tensor_t> inputs;
    inputs.push_back(input_tensor.GetQnnTensor());
    inputs.push_back(gamma_tensor.GetQnnTensor());

    // QNN RmsNorm requires 3 inputs: x, gamma, beta.
    // Add beta from the model or create a dummy zero-filled beta.
    QnnTensorWrapper beta_tensor_wrapper;
    if (beta_add) {
      const Node& beta_node = beta_add->GetNode();
      size_t beta_idx = 0;
      for (size_t i = 0; i < beta_node.InputDefs().size(); ++i) {
        const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
        if (graph_viewer.GetInitializedTensor(beta_node.InputDefs()[i]->Name(), tensor)) {
          beta_idx = i;
          break;
        }
      }
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(beta_add->Inputs()[beta_idx], beta_tensor_wrapper));
    } else {
      // Create dummy zero beta with same shape as gamma
      TensorInfo gamma_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(gamma_input_def, gamma_info));
      std::vector<uint32_t> beta_shape = gamma_info.shape;
      Qnn_DataType_t beta_dtype = gamma_info.qnn_data_type;
      size_t beta_bytes = utils::GetQnnTensorDataSizeInBytes(beta_shape, beta_dtype);
      std::vector<uint8_t> beta_data(beta_bytes, 0);
      const std::string beta_name = node_name + "_beta_dummy";
      beta_tensor_wrapper = QnnTensorWrapper(beta_name, QNN_TENSOR_TYPE_STATIC,
                                             beta_dtype, QnnQuantParamsWrapper(),
                                             std::move(beta_shape), std::move(beta_data));
    }
    inputs.push_back(beta_tensor_wrapper.GetQnnTensor());

    std::vector<Qnn_Param_t> params = {eps_pw.GetQnnParam(), axes_pw.GetQnnParam()};

    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_RMS_NORM,
                                                          std::move(inputs),
                                                          {output_tensor.GetQnnTensor()},
                                                          std::move(params)));
  } else {
    // Add tensors
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gamma_tensor)), "Failed to add gamma");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");

    // QNN RmsNorm requires 3 inputs: x, gamma, beta. Always add beta.
    std::vector<std::string> input_names = {x_input_def.node_arg.Name(), gamma_input_def.node_arg.Name()};
    if (beta_add) {
      const Node& beta_node = beta_add->GetNode();
      size_t beta_idx = 0;
      for (size_t i = 0; i < beta_node.InputDefs().size(); ++i) {
        const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
        if (graph_viewer.GetInitializedTensor(beta_node.InputDefs()[i]->Name(), tensor)) {
          beta_idx = i;
          break;
        }
      }
      QnnTensorWrapper beta_tensor;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(beta_add->Inputs()[beta_idx], beta_tensor));
      input_names.push_back(beta_add->Inputs()[beta_idx].node_arg.Name());
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(beta_tensor)), "Failed to add beta");
    } else {
      // Create dummy zero beta with same shape as gamma
      TensorInfo gamma_info = {};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(gamma_input_def, gamma_info));
      std::vector<uint32_t> beta_shape = gamma_info.shape;
      Qnn_DataType_t beta_dtype = gamma_info.qnn_data_type;
      size_t beta_bytes = utils::GetQnnTensorDataSizeInBytes(beta_shape, beta_dtype);
      std::vector<uint8_t> beta_data(beta_bytes, 0);
      const std::string beta_name = node_name + "_beta_dummy";
      QnnTensorWrapper beta_tw(beta_name, QNN_TENSOR_TYPE_STATIC,
                               beta_dtype, QnnQuantParamsWrapper(),
                               std::move(beta_shape), std::move(beta_data));
      input_names.push_back(beta_name);
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(beta_tw)), "Failed to add dummy beta");
    }

    // Add params
    QnnParamWrapper eps_pw(div.Index(), node_name, QNN_OP_RMS_NORM_PARAM_EPSILON, epsilon_param);
    QnnParamWrapper axes_pw(div.Index(), node_name, QNN_OP_RMS_NORM_PARAM_AXES,
                            std::move(axes_shape), std::move(qnn_axes));
    std::vector<std::string> param_names = {eps_pw.GetParamTensorName(), axes_pw.GetParamTensorName()};
    qnn_model_wrapper.AddParamWrapper(std::move(eps_pw));
    qnn_model_wrapper.AddParamWrapper(std::move(axes_pw));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_RMS_NORM,
                                                      std::move(input_names),
                                                      {output_def.node_arg.Name()},
                                                      std::move(param_names),
                                                      validate),
                      "Failed to add fused RmsNorm node.");
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
