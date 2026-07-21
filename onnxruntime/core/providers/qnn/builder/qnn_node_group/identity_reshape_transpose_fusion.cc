// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/identity_reshape_transpose_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/common/inlined_containers.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr const char* kOpTypeReshape = "Reshape";
constexpr const char* kOpTypeTranspose = "Transpose";
constexpr const char* kAttrTransposePerm = "perm";

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

/// @brief Match [Reshape -> Transpose] starting from a Reshape NodeUnit.
std::optional<std::array<const OrtNodeUnit*, 2>> MatchReshapeTransposePattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit* reshape,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  if (reshape->OpType() != kOpTypeReshape) {
    return std::nullopt;
  }

  const OrtNodeUnit* transpose = GetChildNodeUnitAllowQdq(
      qnn_model_wrapper, *reshape, kOpTypeTranspose, node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose == nullptr) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 2>{reshape, transpose};
}

/// @brief Return true if applying `perm` to a tensor of shape `t1_dims` preserves the
/// physical memory order of elements. Equivalently: the non-unit axes of t1 appear in
/// the same relative order in the transposed output.
bool IsTransposeMemoryPreserving(const std::vector<uint32_t>& t1_dims,
                                 const std::vector<int64_t>& perm) {
  if (t1_dims.size() != perm.size()) {
    return false;
  }
  int64_t last_pos_in_output = -1;
  for (size_t axis = 0; axis < t1_dims.size(); ++axis) {
    if (t1_dims[axis] <= 1) {
      continue;
    }
    // Find the output position `k` such that perm[k] == axis.
    auto it = std::find(perm.begin(), perm.end(), static_cast<int64_t>(axis));
    if (it == perm.end()) {
      return false;
    }
    const int64_t k = std::distance(perm.begin(), it);
    if (k <= last_pos_in_output) {
      return false;
    }
    last_pos_in_output = k;
  }
  return true;
}

/// @brief Emit (validate=true) or add (validate=false) a single identity Reshape
/// replacing the Reshape and Transpose in the graph. Reshape with matching input/output
/// shapes is pure metadata on QNN backends (no data movement), so this eliminates the
/// ~8 MB memory shuffle that the original Reshape+Transpose pair performed.
Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    bool validate,
    const Ort::Logger& logger) {
  const OrtNodeUnit* reshape = node_units[0];
  const OrtNodeUnit* transpose = node_units[1];

  const OrtNodeUnitIODef& reshape_input = reshape->Inputs()[0];
  const OrtNodeUnitIODef& transpose_output = transpose->Outputs()[0];

  std::vector<uint32_t> t0_dims;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(reshape_input.shape, t0_dims),
                ("Cannot get shape for " + reshape_input.name).c_str());

  // Reshape input tensor wrapper.
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(reshape_input.name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("Tensor already added, skip it: " + reshape_input.name).c_str());
  } else {
    QnnTensorWrapper input_tensor_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(reshape_input, input_tensor_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor_wrapper)),
                  "Failed to add the Reshape's input tensor.");
  }

  // Fused-Reshape output tensor wrapper. Use the Transpose's original output name so
  // downstream consumers are unaffected. Shape equals t0 (identity), which we already
  // validated matches t2.
  TensorInfo transpose_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(transpose_output, transpose_output_info));

  const Qnn_TensorType_t out_tensor_type = qnn_model_wrapper.IsGraphOutput(transpose_output.name)
                                               ? QNN_TENSOR_TYPE_APP_READ
                                               : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper fused_output_tensor_wrapper(transpose_output.name,
                                               out_tensor_type,
                                               transpose_output_info.qnn_data_type,
                                               std::move(transpose_output_info.quant_param),
                                               std::vector<uint32_t>(t0_dims));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fused_output_tensor_wrapper)),
                "Failed to add the fused Reshape's output tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(reshape->Name()),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_RESHAPE,
                                                {reshape_input.name},
                                                {transpose_output.name},
                                                {},
                                                validate),
                "Failed to add fused identity Reshape node.");

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> IdentityReshapeTransposeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  auto pattern = MatchReshapeTransposePattern(
      qnn_model_wrapper, &reshape_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* reshape = pattern->at(0);
  const OrtNodeUnit* transpose = pattern->at(1);

  // Reshape's shape input (Inputs()[1]) must be a constant initializer so the inferred
  // output shape is a runtime invariant. Without this, the fusion could be incorrect at
  // runtime if the shape input changes.
  const auto& reshape_inputs = reshape->Inputs();
  if (reshape_inputs.size() < 2 || !qnn_model_wrapper.IsConstantInput(reshape_inputs[1].name)) {
    return nullptr;
  }

  // Retrieve the three shapes: t0 (Reshape input), t1 (Reshape output = Transpose input),
  // and t2 (Transpose output). All three must be statically known.
  std::vector<uint32_t> t0_dims;
  std::vector<uint32_t> t1_dims;
  std::vector<uint32_t> t2_dims;
  if (!qnn_model_wrapper.GetOnnxShape(reshape->Inputs()[0].shape, t0_dims) ||
      !qnn_model_wrapper.GetOnnxShape(reshape->Outputs()[0].shape, t1_dims) ||
      !qnn_model_wrapper.GetOnnxShape(transpose->Outputs()[0].shape, t2_dims)) {
    return nullptr;
  }

  // Condition 1: t0 and t2 must have the same shape (same rank, same dims).
  if (t0_dims != t2_dims) {
    return nullptr;
  }

  // Condition 2: Transpose must preserve memory order relative to t1.
  OrtNodeAttrHelper transpose_helper(*transpose);
  std::vector<int64_t> perm = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm.size() != t1_dims.size()) {
    return nullptr;
  }
  if (!IsTransposeMemoryPreserving(t1_dims, perm)) {
    return nullptr;
  }

  // Ask QNN whether it accepts the fused identity Transpose we intend to emit.
  if (!CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), /*validate=*/true, logger).IsOK()) {
    return nullptr;
  }

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
              ("IdentityReshapeTransposeFusion matched: " + reshape->Name() + " -> " + transpose->Name()).c_str());
  return std::make_unique<IdentityReshapeTransposeFusion>(pattern.value());
}

gsl::span<const OrtNodeUnit* const> IdentityReshapeTransposeFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status IdentityReshapeTransposeFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), /*validate=*/true, logger);
}

Ort::Status IdentityReshapeTransposeFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
