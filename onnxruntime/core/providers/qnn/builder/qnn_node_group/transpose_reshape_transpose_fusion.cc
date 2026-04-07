// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/transpose_reshape_transpose_fusion.h"

#include <array>
#include <gsl/gsl>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kAttrTransposePerm[] = "perm";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpReshape[] = "Reshape";

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

/// @brief Get transpose permutation attribute
std::optional<std::vector<int64_t>> GetTransposePerm(const OrtNodeUnit& transpose) {
  if (transpose.OpType() != kOpTranspose) {
    return std::nullopt;
  }
  OrtNodeAttrHelper helper(transpose);
  return helper.Get(kAttrTransposePerm, std::vector<int64_t>());
}

/// @brief Match pattern: Transpose -> Reshape -> Transpose
std::optional<std::array<const OrtNodeUnit*, 3>> MatchTransposeReshapeTransposePattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit* transpose1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  // Validate first node is Transpose
  if (transpose1->OpType() != kOpTranspose) {
    return std::nullopt;
  }

  // Only handle SingleNode type (not QDQ groups for now)
  if (transpose1->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  // Get Reshape child
  const std::array<std::string_view, 1> reshape_types{kOpReshape};
  const OrtNodeUnit* reshape = GetOnlyChildOfType(qnn_model_wrapper, *transpose1, reshape_types,
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
  if (reshape == nullptr || reshape->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  // Get second Transpose child
  const std::array<std::string_view, 1> transpose_types{kOpTranspose};
  const OrtNodeUnit* transpose2 = GetOnlyChildOfType(qnn_model_wrapper, *reshape, transpose_types,
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose2 == nullptr || transpose2->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 3>{transpose1, reshape, transpose2};
}

/// @brief Check if the combined Transpose->Reshape->Transpose is equivalent to a single Reshape.
/// This is true when original dimensions appear in their natural order in the output (only merged, not reordered).
/// Fusable example:
///  Input: [2, 3, 4] (dims: A=2, B=3, C=4)
///
///  Input [2,3,4]
///      │
///      ▼ Transpose perm1=[1,2,0]
///  [3,4,2]  (B,C,A)
///      │
///      ▼ Reshape to [12,2]  (merge B*C=12, keep A=2)
///  [12,2]
///      │
///      ▼ Transpose perm2=[1,0]
///  [2,12]  (A, B*C)
/// @param input_shape Shape of the input tensor.
/// @param perm1 Permutation of the first Transpose.
/// @param reshape_shape Target shape of the Reshape (output of Reshape).
/// @param perm2 Permutation of the second Transpose.
/// @param[out] fused_shape The equivalent reshape shape if fusion is valid.
/// @return true if fusion is valid, false otherwise.
bool CanFuseToReshape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& perm1,
    const std::vector<int64_t>& intermediate_shape,  // output of first transpose (before reshape)
    const std::vector<int64_t>& reshape_shape,       // output of reshape
    const std::vector<int64_t>& perm2,
    const std::vector<int64_t>& output_shape,  // final output shape
    std::vector<int64_t>& fused_shape) {
  const size_t input_rank = input_shape.size();
  const size_t output_rank = output_shape.size();

  // Basic validation
  if (perm1.size() != input_rank) {
    return false;
  }
  if (perm2.size() != reshape_shape.size()) {
    return false;
  }

  // Analyze reshape: determine how dimensions are merged
  // Build mapping: reshape_output_dim -> list of input dims (indices into intermediate_shape)
  // E.g. [2, 3, 4] -> Reshape -> [2*3, 4]. reshape_mapping = [[0, 1], [2]]
  std::vector<std::vector<size_t>> reshape_mapping(reshape_shape.size());

  size_t intermediate_idx = 0;
  for (size_t out_idx = 0; out_idx < reshape_shape.size(); ++out_idx) {
    int64_t target_size = reshape_shape[out_idx];
    int64_t accumulated_size = 1;

    while (intermediate_idx < intermediate_shape.size() && accumulated_size < target_size) {
      reshape_mapping[out_idx].push_back(intermediate_idx);
      accumulated_size *= intermediate_shape[intermediate_idx];
      intermediate_idx++;
    }

    // Handle case where dimensions match exactly
    if (accumulated_size == target_size) {
      // Good, this output dim is complete
    } else if (intermediate_idx < intermediate_shape.size() &&
               accumulated_size * intermediate_shape[intermediate_idx] == target_size) {
      // Need one more dimension
      reshape_mapping[out_idx].push_back(intermediate_idx);
      intermediate_idx++;
    } else {
      // Reshape is not a simple merge - can't fuse
      return false;
    }

    // If reshape_mapping is empty for this output, add current intermediate index
    if (reshape_mapping[out_idx].empty() && intermediate_idx < intermediate_shape.size()) {
      reshape_mapping[out_idx].push_back(intermediate_idx);
      intermediate_idx++;
    }
  }

  // Map to original input dimensions by applying perm2 then perm1.
  // final_mapping[i] = [perm1[d] for d in reshape_mapping[perm2[i]]]
  // E.g. perm2=[1,0], reshape_mapping=[[0,1], [2]], perm1=[1,2,0]
  // ->
  //      final_mapping[0] = [perm1[d] for d in reshape_mapping[1]] = [perm1[2]] = [0]
  //      final_mapping[1] = [perm1[d] for d in reshape_mapping[0]] = [perm1[0], perm1[1]] = [1, 2]
  std::vector<std::vector<size_t>> final_mapping(output_rank);
  for (size_t i = 0; i < output_rank; ++i) {
    size_t src_idx = static_cast<size_t>(perm2[i]);
    if (src_idx >= reshape_mapping.size()) {
      return false;
    }
    for (size_t intermediate_dim : reshape_mapping[src_idx]) {
      if (intermediate_dim < perm1.size()) {
        final_mapping[i].push_back(static_cast<size_t>(perm1[intermediate_dim]));
      }
    }
  }

  // Check if final_mapping represents a valid "reshape-only" transformation
  // Valid if: original dimensions appear in strictly increasing order (0, 1, 2, ...)
  // The order within each group matters for data layout, so we must NOT sort.
  // E.g., [1, 2] means dim 1 has higher stride than dim 2 (correct for merge)
  //       [2, 1] means dim 2 has higher stride than dim 1 (different data layout)
  size_t expected_next_dim = 0;
  for (size_t i = 0; i < output_rank; ++i) {
    for (size_t orig_dim : final_mapping[i]) {
      if (orig_dim != expected_next_dim) {
        // Dimensions are reordered, can't fuse to reshape
        return false;
      }
      expected_next_dim++;
    }
  }

  // All input dimensions should be accounted for
  if (expected_next_dim != input_rank) {
    return false;
  }

  // Compute fused output shape by merging original dimensions according to final_mapping
  fused_shape.clear();
  fused_shape.reserve(output_rank);
  for (size_t i = 0; i < output_rank; ++i) {
    int64_t dim_size = 1;
    for (size_t orig_dim : final_mapping[i]) {
      dim_size *= input_shape[orig_dim];
    }
    fused_shape.push_back(dim_size);
  }

  // Verify the fused shape matches the expected output shape
  if (fused_shape.size() != output_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < fused_shape.size(); ++i) {
    if (fused_shape[i] != output_shape[i]) {
      return false;
    }
  }

  return true;
}

/// @brief Create or validate the fused Reshape node on QNN
Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    const std::vector<int64_t>& fused_shape,
    bool validate,
    const Ort::Logger& logger) {
  const OrtNodeUnit* transpose1 = node_units[0];
  const OrtNodeUnit* transpose2 = node_units[2];

  // Get input from the first Transpose
  const OrtNodeUnitIODef& input_def = transpose1->Inputs()[0];
  // Get output from the second Transpose
  const OrtNodeUnitIODef& output_def = transpose2->Outputs()[0];

  // Create input tensor wrapper
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_def.name)) {
    QnnTensorWrapper input_tensor_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor_wrapper)),
                  "Failed to add input tensor for TransposeReshapeTransposeFusion.");
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_def.name).c_str());
  }

  // Create output tensor wrapper with fused shape
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));

  std::vector<uint32_t> output_shape_u32;
  output_shape_u32.reserve(fused_shape.size());
  for (int64_t dim : fused_shape) {
    output_shape_u32.push_back(static_cast<uint32_t>(dim));
  }

  Qnn_TensorType_t output_tensor_type = qnn_model_wrapper.IsGraphOutput(output_def.name)
                                            ? QNN_TENSOR_TYPE_APP_READ
                                            : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper output_tensor_wrapper(output_def.name,
                                         output_tensor_type,
                                         output_info.qnn_data_type,
                                         std::move(output_info.quant_param),
                                         std::move(output_shape_u32));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                "Failed to add output tensor for TransposeReshapeTransposeFusion.");

  // Create the fused Reshape node
  const auto& node_name = utils::UniqueNameGenerator().New(*transpose1);
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {input_def.name},
                                                {output_def.name},
                                                {},
                                                validate),
                "Failed to add fused Reshape node for TransposeReshapeTransposeFusion.");

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> TransposeReshapeTransposeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& transpose1_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // Match the pattern: Transpose -> Reshape -> Transpose
  std::optional<std::array<const OrtNodeUnit*, 3>> pattern = MatchTransposeReshapeTransposePattern(
      qnn_model_wrapper, &transpose1_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);

  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* transpose1 = pattern->at(0);
  const OrtNodeUnit* reshape = pattern->at(1);
  const OrtNodeUnit* transpose2 = pattern->at(2);

  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Get input shape of first Transpose
  size_t num_t1_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&transpose1->GetNode(), &num_t1_inputs), ort_api, nullptr);
  std::vector<const OrtValueInfo*> t1_inputs(num_t1_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&transpose1->GetNode(), t1_inputs.data(), t1_inputs.size()),
                             ort_api, nullptr);

  auto input_shape = GetTensorShape(ort_api, t1_inputs[0]);
  if (!input_shape.has_value()) {
    return nullptr;
  }

  // Get output shape of first Transpose (= input to Reshape)
  size_t num_t1_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&transpose1->GetNode(), &num_t1_outputs), ort_api, nullptr);
  std::vector<const OrtValueInfo*> t1_outputs(num_t1_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&transpose1->GetNode(), t1_outputs.data(), t1_outputs.size()),
                             ort_api, nullptr);

  auto intermediate_shape = GetTensorShape(ort_api, t1_outputs[0]);
  if (!intermediate_shape.has_value()) {
    return nullptr;
  }

  // Get output shape of Reshape
  size_t num_reshape_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape->GetNode(), &num_reshape_outputs), ort_api, nullptr);
  std::vector<const OrtValueInfo*> reshape_outputs(num_reshape_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&reshape->GetNode(), reshape_outputs.data(), reshape_outputs.size()),
                             ort_api, nullptr);

  auto reshape_shape = GetTensorShape(ort_api, reshape_outputs[0]);
  if (!reshape_shape.has_value()) {
    return nullptr;
  }

  // Get output shape of second Transpose (final output)
  size_t num_t2_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&transpose2->GetNode(), &num_t2_outputs), ort_api, nullptr);
  std::vector<const OrtValueInfo*> t2_outputs(num_t2_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&transpose2->GetNode(), t2_outputs.data(), t2_outputs.size()),
                             ort_api, nullptr);

  auto output_shape = GetTensorShape(ort_api, t2_outputs[0]);
  if (!output_shape.has_value()) {
    return nullptr;
  }

  // Get permutations
  auto perm1 = GetTransposePerm(*transpose1);
  auto perm2 = GetTransposePerm(*transpose2);

  if (!perm1.has_value() || !perm2.has_value()) {
    return nullptr;
  }

  // Check if the pattern can be fused to a single Reshape
  std::vector<int64_t> fused_shape;
  if (!CanFuseToReshape(*input_shape, *perm1, *intermediate_shape, *reshape_shape, *perm2, *output_shape, fused_shape)) {
    return nullptr;
  }

  // Validate on QNN
  Ort::Status status = CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), fused_shape, /*validate=*/true, logger);
  if (!status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<TransposeReshapeTransposeFusion>(pattern.value(), std::move(fused_shape));
}

gsl::span<const OrtNodeUnit* const> TransposeReshapeTransposeFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status TransposeReshapeTransposeFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), fused_output_shape_, /*validate=*/true, logger);
}

Ort::Status TransposeReshapeTransposeFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), fused_output_shape_, /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
