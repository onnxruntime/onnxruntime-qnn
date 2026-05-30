// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/transpose_gather_transpose_fusion.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <gsl/gsl>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/normalize_indices_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kAttrTransposePerm[] = "perm";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpGather[] = "Gather";

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

std::optional<std::vector<int64_t>> GetTransposePerm(const OrtNodeUnit& transpose) {
  if (transpose.OpType() != kOpTranspose) {
    return std::nullopt;
  }
  OrtNodeAttrHelper helper(transpose);
  return helper.Get(kAttrTransposePerm, std::vector<int64_t>());
}

// Match the pattern: Transpose -> Gather -> Transpose, all SingleNode, no QDQ wrapping.
std::optional<std::array<const OrtNodeUnit*, 3>> MatchPattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& transpose1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  if (transpose1.OpType() != kOpTranspose ||
      transpose1.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  const std::array<std::string_view, 1> gather_types{kOpGather};
  const OrtNodeUnit* gather = GetOnlyChildOfType(qnn_model_wrapper, transpose1, gather_types,
                                                 node_to_node_unit, node_unit_to_qnn_node_group);
  if (gather == nullptr || gather->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  const std::array<std::string_view, 1> transpose_types{kOpTranspose};
  const OrtNodeUnit* transpose2 = GetOnlyChildOfType(qnn_model_wrapper, *gather, transpose_types,
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose2 == nullptr || transpose2->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 3>{&transpose1, gather, transpose2};
}

// Validate that perm2 cancels (perm1 + Gather) so that the whole pattern is equivalent to
// Gather(x, indices, axis=perm1[gather_axis]).
bool IsCancelingPair(const std::vector<int64_t>& perm1,
                     const std::vector<int64_t>& perm2,
                     int64_t gather_axis,
                     int64_t indices_rank) {
  const int64_t n = static_cast<int64_t>(perm1.size());
  if (n <= 0) return false;
  if (gather_axis < 0 || gather_axis >= n) return false;
  if (indices_rank < 0) return false;
  const int64_t out_rank = n - 1 + indices_rank;
  if (static_cast<int64_t>(perm2.size()) != out_rank) return false;

  // Sanity check: perm1 is a permutation of [0, n).
  std::vector<bool> seen(static_cast<size_t>(n), false);
  for (int64_t v : perm1) {
    if (v < 0 || v >= n || seen[static_cast<size_t>(v)]) return false;
    seen[static_cast<size_t>(v)] = true;
  }

  const int64_t fused_axis = perm1[static_cast<size_t>(gather_axis)];

  // Build the inverse mapping: for each desired source position in the final
  // output, find the index in g that holds it.
  //
  //   data_axis_to_g[d] = index in g that holds data axis d
  //   indices_dim_to_g[k] = A + k
  std::vector<int64_t> data_axis_to_g(static_cast<size_t>(n), -1);
  for (int64_t i = 0; i < n; ++i) {
    if (i == gather_axis) continue;
    const int64_t g_pos = (i < gather_axis) ? i : (i + indices_rank - 1);
    data_axis_to_g[static_cast<size_t>(perm1[static_cast<size_t>(i)])] = g_pos;
  }

  // Walk the desired output dim by dim and check that perm2 picks the right g index.
  int64_t out_idx = 0;
  for (int64_t d = 0; d < fused_axis; ++d, ++out_idx) {
    if (perm2[static_cast<size_t>(out_idx)] != data_axis_to_g[static_cast<size_t>(d)]) return false;
  }
  for (int64_t k = 0; k < indices_rank; ++k, ++out_idx) {
    if (perm2[static_cast<size_t>(out_idx)] != gather_axis + k) return false;
  }
  for (int64_t d = fused_axis + 1; d < n; ++d, ++out_idx) {
    if (perm2[static_cast<size_t>(out_idx)] != data_axis_to_g[static_cast<size_t>(d)]) return false;
  }

  // If every position matched, perm2 was forced to be a valid permutation of [0, out_rank):
  // each expected value is unique (data axes via the perm1-injective data_axis_to_g, indices
  // via gather_axis + k for k in [0, indices_rank)) and they cover [0, out_rank) exactly.
  return out_idx == out_rank;
}

Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                  gsl::span<const OrtNodeUnit* const> node_units,
                                  int32_t fused_axis,
                                  bool validate,
                                  const Ort::Logger& logger) {
  const OrtNodeUnit* transpose1 = node_units[0];
  const OrtNodeUnit* gather = node_units[1];
  const OrtNodeUnit* transpose2 = node_units[2];

  const OrtNodeUnitIODef& x_def = transpose1->Inputs()[0];   // input to transpose1 (= input to fused Gather)
  const OrtNodeUnitIODef& idx_def = gather->Inputs()[1];     // gather indices
  const OrtNodeUnitIODef& y_def = transpose2->Outputs()[0];  // final output

  // Resolve the axis dim from the original (pre-transpose1) input shape.
  std::vector<uint32_t> x_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(x_def.shape, x_shape),
                "Failed to read input shape for TransposeGatherTransposeFusion.");
  RETURN_IF_NOT(fused_axis >= 0 && static_cast<size_t>(fused_axis) < x_shape.size(),
                "Fused axis out of range for TransposeGatherTransposeFusion.");
  const int64_t axis_dim = static_cast<int64_t>(x_shape[static_cast<size_t>(fused_axis)]);

  // Output info (final fused Gather output uses transpose2's output shape/quant).
  TensorInfo y_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(y_def, y_info));

  // Build the fused Gather inputs: x_def passed through unchanged; indices statically prepared.
  // 1. x tensor: reuse existing wrapper if present, otherwise create from the def.
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(x_def.name)) {
    QnnTensorWrapper x_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(x_def, x_wrapper));
    if (!validate) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(x_wrapper)),
                    "Failed to add input tensor for TransposeGatherTransposeFusion.");
    }
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("Tensor already added, skip it: " + x_def.name).c_str());
  }

  // 2. Indices tensor: must be a constant initializer. Convert int64 -> int32 if needed
  //    and resolve negative indices against the axis dim of x_def.
  TensorInfo idx_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(idx_def, idx_info));
  RETURN_IF_NOT(idx_info.is_initializer,
                "TransposeGatherTransposeFusion requires constant Gather indices.");

  std::vector<uint8_t> onnx_idx_bytes;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(idx_info.initializer_tensor, onnx_idx_bytes));

  std::vector<uint8_t> qnn_idx_bytes;
  Qnn_DataType_t idx_qnn_type = idx_info.qnn_data_type;
  if (idx_qnn_type == QNN_DATATYPE_INT_64) {
    RETURN_IF_NOT((utils::MakeStaticIndicesPositiveAndValidate<int64_t, int32_t>(
                      onnx_idx_bytes, axis_dim, qnn_idx_bytes, /*has_negative_indices=*/nullptr)),
                  "TransposeGatherTransposeFusion: failed to convert int64 indices to int32.");
    idx_qnn_type = QNN_DATATYPE_INT_32;
  } else if (idx_qnn_type == QNN_DATATYPE_INT_32) {
    RETURN_IF_NOT((utils::MakeStaticIndicesPositiveAndValidate<int32_t, int32_t>(
                      onnx_idx_bytes, axis_dim, qnn_idx_bytes, /*has_negative_indices=*/nullptr)),
                  "TransposeGatherTransposeFusion: indices out of range.");
  } else {
    return MAKE_EP_FAIL("TransposeGatherTransposeFusion: unsupported indices data type.");
  }

  // Use a unique indices tensor name so the static buffer attaches cleanly to the fused Gather
  // and does not collide with the original ONNX initializer reference.
  const std::string fused_idx_name = utils::UniqueNameGenerator().New(*gather, "_tgt_idx");
  if (!validate) {
    QnnTensorWrapper idx_wrapper(fused_idx_name,
                                 QNN_TENSOR_TYPE_STATIC,
                                 idx_qnn_type,
                                 QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(idx_info.shape),
                                 std::move(qnn_idx_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(idx_wrapper)),
                  "Failed to add indices tensor for TransposeGatherTransposeFusion.");
  }

  // 3. Output tensor: shape from transpose2's output.
  Qnn_TensorType_t out_tensor_type = qnn_model_wrapper.IsGraphOutput(y_def.name)
                                         ? QNN_TENSOR_TYPE_APP_READ
                                         : QNN_TENSOR_TYPE_NATIVE;
  if (!validate) {
    QnnTensorWrapper y_wrapper(y_def.name,
                               out_tensor_type,
                               y_info.qnn_data_type,
                               y_info.quant_param.Copy(),
                               std::vector<uint32_t>(y_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(y_wrapper)),
                  "Failed to add output tensor for TransposeGatherTransposeFusion.");
  }

  // 4. Axis param.
  Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
  axis_scalar.dataType = QNN_DATATYPE_INT_32;
  axis_scalar.int32Value = fused_axis;
  QnnParamWrapper axis_param(gather->Index(), gather->Name(), QNN_OP_GATHER_PARAM_AXIS, axis_scalar);
  std::string axis_param_name = axis_param.GetParamTensorName();

  if (validate) {
    // Build transient wrappers for validation only.
    QnnTensorWrapper x_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(x_def, x_wrapper));
    QnnTensorWrapper idx_wrapper(fused_idx_name,
                                 QNN_TENSOR_TYPE_STATIC,
                                 idx_qnn_type,
                                 QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(idx_info.shape),
                                 std::move(qnn_idx_bytes));
    QnnTensorWrapper y_wrapper(y_def.name,
                               out_tensor_type,
                               y_info.qnn_data_type,
                               y_info.quant_param.Copy(),
                               std::vector<uint32_t>(y_info.shape));
    return qnn_model_wrapper.ValidateQnnNode(
        utils::UniqueNameGenerator().New(*gather),
        QNN_OP_PACKAGE_NAME_QTI_AISW,
        QNN_OP_GATHER,
        {x_wrapper.GetQnnTensor(), idx_wrapper.GetQnnTensor()},
        {y_wrapper.GetQnnTensor()},
        {axis_param.GetQnnParam()});
  }

  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(*gather),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_GATHER,
                                                {x_def.name, fused_idx_name},
                                                {y_def.name},
                                                {axis_param_name},
                                                /*do_op_validation*/ false),
                "Failed to add fused Gather node for TransposeGatherTransposeFusion.");

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> TransposeGatherTransposeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& transpose1_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  std::optional<std::array<const OrtNodeUnit*, 3>> pattern = MatchPattern(
      qnn_model_wrapper, transpose1_node_unit, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* transpose1 = pattern->at(0);
  const OrtNodeUnit* gather = pattern->at(1);
  const OrtNodeUnit* transpose2 = pattern->at(2);

  // Both Transposes must have a perm attribute.
  std::optional<std::vector<int64_t>> perm1 = GetTransposePerm(*transpose1);
  std::optional<std::vector<int64_t>> perm2 = GetTransposePerm(*transpose2);
  if (!perm1.has_value() || !perm2.has_value() || perm1->empty() || perm2->empty()) {
    return nullptr;
  }

  // Gather must have axis attribute and a constant indices initializer.
  if (gather->Inputs().size() < 2) return nullptr;
  const OrtNodeUnitIODef& idx_def = gather->Inputs()[1];
  if (!qnn_model_wrapper.IsConstantInput(idx_def.name)) {
    return nullptr;
  }
  std::vector<uint32_t> idx_shape;
  if (!qnn_model_wrapper.GetOnnxShape(idx_def.shape, idx_shape)) {
    return nullptr;
  }

  const OrtNodeUnitIODef& x_def = transpose1->Inputs()[0];
  std::vector<uint32_t> x_shape;
  if (!qnn_model_wrapper.GetOnnxShape(x_def.shape, x_shape)) {
    return nullptr;
  }
  if (x_shape.size() != perm1->size()) {
    return nullptr;
  }

  // Resolve gather axis (relative to transpose1 output, which has rank == perm1->size()).
  OrtNodeAttrHelper gather_attrs(*gather);
  int64_t gather_axis = gather_attrs.Get("axis", static_cast<int64_t>(0));
  if (gather_axis < 0) gather_axis += static_cast<int64_t>(perm1->size());
  if (gather_axis < 0 || static_cast<size_t>(gather_axis) >= perm1->size()) {
    return nullptr;
  }

  if (!IsCancelingPair(*perm1, *perm2, gather_axis, static_cast<int64_t>(idx_shape.size()))) {
    return nullptr;
  }

  const int32_t fused_axis = static_cast<int32_t>((*perm1)[static_cast<size_t>(gather_axis)]);

  // Validate on QNN.
  Ort::Status status = CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), fused_axis,
                                             /*validate=*/true, logger);
  if (!status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<TransposeGatherTransposeFusion>(pattern.value(), fused_axis);
}

gsl::span<const OrtNodeUnit* const> TransposeGatherTransposeFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status TransposeGatherTransposeFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), fused_axis_, /*validate=*/true, logger);
}

Ort::Status TransposeGatherTransposeFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), fused_axis_, /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
