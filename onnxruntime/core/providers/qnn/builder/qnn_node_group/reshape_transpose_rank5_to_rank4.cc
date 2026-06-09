// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_transpose_rank5_to_rank4.h"

#include <gsl/gsl>
#include <optional>
#include <utility>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/common/inlined_containers.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr size_t kRank5 = 5;
constexpr size_t kRank4 = 4;
constexpr const char* kOpTypeReshape = "Reshape";
constexpr const char* kOpTypeTranspose = "Transpose";
constexpr const char* kAttrTransposePerm = "perm";

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

/// @brief Get the shape of a tensor from its OrtValueInfo
std::optional<std::vector<int64_t>> GetTensorShape(const OrtApi& ort_api, const OrtValueInfo* value_info) {
  if (value_info == nullptr) {
    return std::nullopt;
  }

  const OrtTypeInfo* type_info = nullptr;
  if (ort_api.GetValueInfoTypeInfo(value_info, &type_info) != nullptr) {
    return std::nullopt;
  }

  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  if (ort_api.CastTypeInfoToTensorInfo(type_info, &tensor_info) != nullptr) {
    return std::nullopt;
  }

  size_t dims_count = 0;
  if (ort_api.GetDimensionsCount(tensor_info, &dims_count) != nullptr) {
    return std::nullopt;
  }

  std::vector<int64_t> dims(dims_count);
  if (ort_api.GetDimensions(tensor_info, dims.data(), dims_count) != nullptr) {
    return std::nullopt;
  }

  return dims;
}

std::optional<std::array<const OrtNodeUnit*, 3>> MatchRank5ToRank4Pattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit* reshape1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  if (reshape1->OpType() != kOpTypeReshape) {
    return std::nullopt;
  }

  const OrtNodeUnit* transpose = GetChildNodeUnitAllowQdq(
      qnn_model_wrapper, *reshape1, kOpTypeTranspose, node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose == nullptr) {
    return std::nullopt;
  }

  const OrtNodeUnit* reshape2 = GetChildNodeUnitAllowQdq(
      qnn_model_wrapper, *transpose, kOpTypeReshape, node_to_node_unit, node_unit_to_qnn_node_group);
  if (reshape2 == nullptr) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 3>{reshape1, transpose, reshape2};
}

std::optional<size_t> FindAdjacentMergeIndex(const std::vector<int64_t>& perm_rank5) {
  for (size_t p = 0; p + 1 < perm_rank5.size(); ++p) {
    if (perm_rank5[p + 1] == perm_rank5[p] + 1) {
      return p;
    }
  }
  return std::nullopt;
}

std::optional<size_t> ValidatePatternConditions(
    const OrtNodeUnit* reshape1,
    const OrtNodeUnit* transpose,
    const OrtNodeUnit* reshape2,
    const QnnModelWrapper& qnn_model_wrapper,
    [[maybe_unused]] const Ort::Logger& logger) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Check if reshape shape inputs are constants
  const OrtNodeUnitIODef& reshape1_input_1 = reshape1->Inputs()[1];
  const OrtNodeUnitIODef& reshape2_input_1 = reshape2->Inputs()[1];

  if (!qnn_model_wrapper.IsConstantInput(reshape1_input_1.name)) {
    return std::nullopt;
  }

  if (!qnn_model_wrapper.IsConstantInput(reshape2_input_1.name)) {
    return std::nullopt;
  }

  // Get tensor shapes
  size_t num_reshape1_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&reshape1->GetNode(), &num_reshape1_inputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> reshape1_inputs(num_reshape1_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&reshape1->GetNode(), reshape1_inputs.data(), reshape1_inputs.size()), ort_api, std::nullopt);

  size_t num_reshape1_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape1->GetNode(), &num_reshape1_outputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> reshape1_outputs(num_reshape1_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&reshape1->GetNode(), reshape1_outputs.data(), reshape1_outputs.size()), ort_api, std::nullopt);

  size_t num_transpose_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&transpose->GetNode(), &num_transpose_outputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> transpose_outputs(num_transpose_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&transpose->GetNode(), transpose_outputs.data(), transpose_outputs.size()), ort_api, std::nullopt);

  size_t num_reshape2_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape2->GetNode(), &num_reshape2_outputs), ort_api, std::nullopt);
  std::vector<const OrtValueInfo*> reshape2_outputs(num_reshape2_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&reshape2->GetNode(), reshape2_outputs.data(), reshape2_outputs.size()), ort_api, std::nullopt);

  auto t0_shape = GetTensorShape(ort_api, reshape1_inputs[0]);
  auto t1_shape = GetTensorShape(ort_api, reshape1_outputs[0]);
  auto t2_shape = GetTensorShape(ort_api, transpose_outputs[0]);
  auto t3_shape = GetTensorShape(ort_api, reshape2_outputs[0]);

  if (!t0_shape.has_value() || !t1_shape.has_value() ||
      !t2_shape.has_value() || !t3_shape.has_value()) {
    return std::nullopt;
  }

  // Condition 1: Rank(t1) == Rank(t2) == 5
  if (t1_shape->size() != kRank5 || t2_shape->size() != kRank5) {
    return std::nullopt;
  }

  // Condition 2: Transpose perm must be rank-5.
  OrtNodeAttrHelper transpose_helper(*transpose);
  std::vector<int64_t> perm = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm.size() != kRank5) {
    return std::nullopt;
  }

  // Condition 3: There must be an adjacent pair (p, p+1) in the rank-5 perm whose values are
  // consecutive (perm[p+1] == perm[p] + 1), so the two input dims they reference can be merged.
  std::optional<size_t> merge_perm_index = FindAdjacentMergeIndex(perm);
  if (!merge_perm_index.has_value()) {
    return std::nullopt;
  }

  return merge_perm_index.value();
}

/// @brief Create or validate the QNN nodes with rank-4 tensors
Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    size_t merge_perm_index,
    bool validate,
    const Ort::Logger& logger) {
  const OrtNodeUnit* reshape1 = node_units[0];
  const OrtNodeUnit* transpose = node_units[1];
  const OrtNodeUnit* reshape2 = node_units[2];

  // Get input and output definitions
  const OrtNodeUnitIODef& reshape1_input = reshape1->Inputs()[0];
  const OrtNodeUnitIODef& reshape1_output = reshape1->Outputs()[0];
  const OrtNodeUnitIODef& transpose_output = transpose->Outputs()[0];
  const OrtNodeUnitIODef& reshape2_output = reshape2->Outputs()[0];

  // Get original rank-5 shapes
  std::vector<uint32_t> t1_rank5_dims;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(reshape1_output.shape, t1_rank5_dims),
                ("Cannot get shape for " + reshape1_output.name).c_str());

  std::vector<uint32_t> t2_rank5_dims;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(transpose_output.shape, t2_rank5_dims),
                ("Cannot get shape for " + transpose_output.name).c_str());

  // Get the rank-5 perm.
  OrtNodeAttrHelper transpose_helper(*transpose);
  std::vector<int64_t> perm_rank5 = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm_rank5.size() != kRank5 || merge_perm_index + 1 >= perm_rank5.size()) {
    return Ort::Status("Invalid rank-5 perm or merge index", OrtErrorCode::ORT_FAIL);
  }

  // Step 1: merge dims at input indices perm_rank5[merge_perm_index] and perm_rank5[merge_perm_index+1]
  // (which are consecutive input indices because perm[p+1] == perm[p] + 1).
  const int64_t merge_input_idx_a = perm_rank5[merge_perm_index];
  const int64_t merge_input_idx_b = perm_rank5[merge_perm_index + 1];
  if (merge_input_idx_b != merge_input_idx_a + 1) {
    return Ort::Status("Merge indices must be consecutive in input space", OrtErrorCode::ORT_FAIL);
  }

  // Build the rank-4 t1 shape by merging t1_rank5[merge_input_idx_a] and t1_rank5[merge_input_idx_b].
  std::vector<uint32_t> t1_rank4_dims;
  t1_rank4_dims.reserve(kRank4);
  for (size_t i = 0; i < t1_rank5_dims.size(); ++i) {
    if (static_cast<int64_t>(i) == merge_input_idx_a) {
      t1_rank4_dims.push_back(t1_rank5_dims[i] * t1_rank5_dims[i + 1]);
    } else if (static_cast<int64_t>(i) == merge_input_idx_b) {
      continue;
    } else {
      t1_rank4_dims.push_back(t1_rank5_dims[i]);
    }
  }

  // Build the rank-4 perm by removing position (merge_perm_index + 1) and shifting any value
  // > merge_input_idx_b down by one (since input index merge_input_idx_b is gone).
  std::vector<uint32_t> perm_rank4;
  perm_rank4.reserve(kRank4);
  for (size_t i = 0; i < perm_rank5.size(); ++i) {
    if (i == merge_perm_index + 1) {
      continue;
    }
    int64_t v = perm_rank5[i];
    if (v > merge_input_idx_b) {
      v--;
    }
    perm_rank4.push_back(static_cast<uint32_t>(v));
  }

  // Build the rank-4 t2 shape by applying perm_rank4 to t1_rank4_dims.
  std::vector<uint32_t> t2_rank4_dims;
  t2_rank4_dims.reserve(kRank4);
  for (uint32_t p : perm_rank4) {
    t2_rank4_dims.push_back(t1_rank4_dims[p]);
  }

  // Create Reshape1 input tensor wrapper.
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(reshape1_input.name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + reshape1_input.name).c_str());
  } else {
    QnnTensorWrapper input_tensor_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(reshape1_input, input_tensor_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor_wrapper)),
                  "Failed to add the first Reshape's input tensor.");
  }

  // Create Reshape1 output tensor wrapper (rank-4).
  TensorInfo reshape1_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(reshape1_output, reshape1_output_info));

  QnnTensorWrapper reshape1_output_tensor_wrapper(reshape1_output.name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  reshape1_output_info.qnn_data_type,
                                                  std::move(reshape1_output_info.quant_param),
                                                  std::move(t1_rank4_dims));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape1_output_tensor_wrapper)),
                "Failed to add the first Reshape's output tensor.");

  // Create Reshape1 node.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(reshape1->Name()),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {reshape1_input.name},
                                                {reshape1_output.name},
                                                {},
                                                validate),
                "Failed to add the first Reshape node.");

  // Create Transpose output tensor wrapper (rank-4).
  TensorInfo transpose_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(transpose_output, transpose_output_info));

  QnnTensorWrapper transpose_output_tensor_wrapper(transpose_output.name,
                                                   QNN_TENSOR_TYPE_NATIVE,
                                                   transpose_output_info.qnn_data_type,
                                                   std::move(transpose_output_info.quant_param),
                                                   std::move(t2_rank4_dims));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(transpose_output_tensor_wrapper)),
                "Failed to add Transpose's output tensor.");

  // Create Transpose perm parameter wrapper.
  QnnParamWrapper perm_param(transpose->Index(),
                             transpose->Name(),
                             QNN_OP_TRANSPOSE_PARAM_PERM,
                             {static_cast<uint32_t>(perm_rank4.size())},
                             std::move(perm_rank4));
  const std::string param_tensor_name = perm_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(perm_param)), "Failed to add Transpose perm param.");

  // Create Transpose node.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(transpose->Name()),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_TRANSPOSE,
                                                {reshape1_output.name},
                                                {transpose_output.name},
                                                {param_tensor_name},
                                                validate),
                "Failed to add Transpose node.");

  // Create Reshape2 output tensor wrapper (original rank).
  TensorInfo reshape2_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(reshape2_output, reshape2_output_info));

  Qnn_TensorType_t reshape2_output_tensor_type = qnn_model_wrapper.IsGraphOutput(reshape2_output.name)
                                                     ? QNN_TENSOR_TYPE_APP_READ
                                                     : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper reshape2_output_tensor_wrapper(reshape2_output.name,
                                                  reshape2_output_tensor_type,
                                                  reshape2_output_info.qnn_data_type,
                                                  std::move(reshape2_output_info.quant_param),
                                                  std::move(reshape2_output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape2_output_tensor_wrapper)),
                "Failed to add the second Reshape's output tensor.");

  // Create Reshape2 node.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(reshape2->Name()),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {transpose_output.name},
                                                {reshape2_output.name},
                                                {},
                                                validate),
                "Failed to add the second Reshape node.");

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> Rank5ToRank4Fusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  std::optional<std::array<const OrtNodeUnit*, 3>> pattern = MatchRank5ToRank4Pattern(
      qnn_model_wrapper, &reshape1_node_unit, node_to_node_unit, node_unit_to_qnn_node_group, logger);

  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* reshape1 = pattern->at(0);
  const OrtNodeUnit* transpose = pattern->at(1);
  const OrtNodeUnit* reshape2 = pattern->at(2);

  auto merge_perm_index = ValidatePatternConditions(reshape1, transpose, reshape2, qnn_model_wrapper, logger);
  if (!merge_perm_index.has_value()) {
    return nullptr;
  }

  if (!CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), merge_perm_index.value(),
                             /*validate=*/true, logger)
           .IsOK()) {
    return nullptr;
  }

  return std::make_unique<Rank5ToRank4Fusion>(pattern.value(), merge_perm_index.value());
}

gsl::span<const OrtNodeUnit* const> Rank5ToRank4Fusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status Rank5ToRank4Fusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), merge_perm_index_,
                               /*validate=*/true, logger);
}

Ort::Status Rank5ToRank4Fusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), merge_perm_index_,
                               /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
