// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

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
#include "core/providers/qnn/builder/qnn_node_group/transpose_reshape_transpose_fusion.h"
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

/// @brief Get transpose permutation attribute
std::optional<std::vector<int64_t>> GetTransposePerm(const OrtNodeUnit& transpose) {
  if (transpose.OpType() != kOpTranspose) {
    return std::nullopt;
  }
  OrtNodeAttrHelper helper(transpose);
  return helper.Get(kAttrTransposePerm, std::vector<int64_t>());
}

// Match pattern: Transpose -> Reshape -> Transpose
std::optional<std::array<const OrtNodeUnit*, 3>> MatchTransposeReshapeTransposePattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit* transpose1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  // Validate first node is Transpose
  if (transpose1->OpType() != kOpTranspose) {
    return std::nullopt;
  }

  // Get Reshape child (allow QDQ nodes in between)
  const OrtNodeUnit* reshape = GetChildNodeUnitAllowQdq(qnn_model_wrapper, *transpose1, kOpReshape,
                                                        node_to_node_unit, node_unit_to_qnn_node_group);
  if (reshape == nullptr) {
    return std::nullopt;
  }

  // Get second Transpose child (allow QDQ nodes in between)
  const OrtNodeUnit* transpose2 = GetChildNodeUnitAllowQdq(qnn_model_wrapper, *reshape, kOpTranspose,
                                                           node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose2 == nullptr) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 3>{transpose1, reshape, transpose2};
}

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
    if (accumulated_size != target_size) {
      // Reshape is not a simple merge - can't fuse
      return false;
    }

    // If reshape_mapping is empty (target_size == 1), the corresponding intermediate dimension must also be 1
    if (reshape_mapping[out_idx].empty()) {
      if (intermediate_idx >= intermediate_shape.size() ||
          intermediate_shape[intermediate_idx] != 1) {
        return false;
      }
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
    for (size_t intermediate_dim : reshape_mapping[src_idx]) {
      final_mapping[i].push_back(static_cast<size_t>(perm1[intermediate_dim]));
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

  // Get input shape of first Transpose
  const auto& input_shape = transpose1->Inputs()[0].shape;
  if (!input_shape.has_value()) {
    return nullptr;
  }

  // Get output shape of first Transpose (= input to Reshape)
  const auto& intermediate_shape = transpose1->Outputs()[0].shape;
  if (!intermediate_shape.has_value()) {
    return nullptr;
  }

  // Get output shape of Reshape
  const auto& reshape_shape = reshape->Outputs()[0].shape;
  if (!reshape_shape.has_value()) {
    return nullptr;
  }

  // Get output shape of second Transpose (final output)
  const auto& output_shape = transpose2->Outputs()[0].shape;
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
