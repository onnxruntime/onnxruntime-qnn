// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_transpose_fusion.h"

#include <gsl/gsl>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

// Derive the permutation a Reshape performs on its input when the Reshape is
// structurally equivalent to a Transpose. Non-1 dims must appear in the same relative
// order in the output; size-1 dims may move freely. Fills `perm` on success so that
// reshape_output_shape[i] == reshape_input_shape[perm[i]].
bool DeriveReshapeAsPerm(const std::vector<int64_t>& input_shape,
                         const std::vector<int64_t>& output_shape,
                         std::vector<int64_t>& perm) {
  if (input_shape.size() != output_shape.size()) {
    return false;
  }
  const size_t rank = input_shape.size();
  for (size_t i = 0; i < rank; ++i) {
    if (input_shape[i] < 0 || output_shape[i] < 0) {
      return false;
    }
  }

  perm.assign(rank, -1);
  std::vector<int64_t> input_dims = input_shape;
  for (size_t i = 0; i < rank; ++i) {
    const int64_t target = output_shape[i];
    size_t cur = 0;
    while (cur < rank && input_dims[cur] != target) {
      if (input_dims[cur] != -1 && input_dims[cur] != 1 && target != 1) {
        return false;
      }
      ++cur;
    }
    if (cur == rank) {
      return false;
    }
    perm[i] = static_cast<int64_t>(cur);
    input_dims[cur] = -1;
  }
  return true;
}

// Try to compose the Reshape-as-perm with the Transpose's perm. Returns true on success
// and fills `fused_perm` with reshape_perm[transpose_perm[i]]. Returns false if the
// Reshape is not Transpose-equivalent or shapes/perms are invalid.
bool ComputeFusedPerm(const OrtNodeUnit& reshape_node_unit,
                      const OrtNodeUnit& transpose_node_unit,
                      std::vector<int64_t>& fused_perm) {
  const OrtNodeUnitIODef& reshape_input = reshape_node_unit.Inputs()[0];
  const OrtNodeUnitIODef& reshape_output = reshape_node_unit.Outputs()[0];
  if (!reshape_input.shape.has_value() || !reshape_output.shape.has_value()) {
    return false;
  }

  std::vector<int64_t> reshape_perm;
  if (!DeriveReshapeAsPerm(*reshape_input.shape, *reshape_output.shape, reshape_perm)) {
    return false;
  }
  const size_t rank = reshape_perm.size();

  OrtNodeAttrHelper transpose_helper(transpose_node_unit);
  std::vector<int64_t> transpose_perm = transpose_helper.Get("perm", std::vector<int64_t>{});
  if (transpose_perm.empty()) {
    // ONNX default for a missing perm is reverse-of-input-rank.
    transpose_perm.resize(rank);
    for (size_t i = 0; i < rank; ++i) {
      transpose_perm[i] = static_cast<int64_t>(rank - 1 - i);
    }
  }
  if (transpose_perm.size() != rank) {
    return false;
  }

  fused_perm.assign(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t t = transpose_perm[i];
    if (t < 0 || static_cast<size_t>(t) >= rank) {
      return false;
    }
    fused_perm[i] = reshape_perm[static_cast<size_t>(t)];
  }
  return true;
}

bool IsIdentityPerm(const std::vector<int64_t>& perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& reshape_node_unit,
                                  const OrtNodeUnit& transpose_node_unit,
                                  const std::vector<int64_t>& fused_perm,
                                  bool validate,
                                  const Ort::Logger& logger) {
  const OrtNodeUnitIODef& reshape_input = reshape_node_unit.Inputs()[0];
  const OrtNodeUnitIODef& transpose_output = transpose_node_unit.Outputs()[0];

  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(reshape_input.name)) {
    QnnTensorWrapper input_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(reshape_input, input_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_wrapper)),
                  "[ReshapeTransposeFusion] Failed to add input tensor wrapper");
  }

  if (IsIdentityPerm(fused_perm)) {
    // Identity composed perm => Reshape input and Transpose output shapes coincide.
    // Emit a noop Reshape so downstream consumers see the expected tensor name.
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("[ReshapeTransposeFusion] Emitting noop Reshape " + reshape_input.name +
                 " -> " + transpose_output.name)
                    .c_str());
    return qnn_model_wrapper.AddNoopReshapeNode(reshape_node_unit.Name(),
                                                reshape_input.name,
                                                transpose_output,
                                                validate);
  }

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(reshape_input.shape, input_shape),
                ("[ReshapeTransposeFusion] Failed to get input shape for " + reshape_input.name).c_str());
  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(transpose_output.shape, output_shape),
                ("[ReshapeTransposeFusion] Failed to get output shape for " + transpose_output.name).c_str());

  std::vector<uint32_t> perm_u32;
  perm_u32.reserve(fused_perm.size());
  for (int64_t p : fused_perm) {
    perm_u32.push_back(static_cast<uint32_t>(p));
  }

  Qnn_DataType_t data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(transpose_output.quant_param.has_value(),
                                        transpose_output.type, data_type));

  QnnTensorWrapper output_wrapper;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(transpose_output, output_wrapper));
  QnnQuantParamsWrapper quant_param = output_wrapper.GetQnnQuantParams();

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              ("[ReshapeTransposeFusion] Emitting single Transpose " + reshape_input.name +
               " -> " + transpose_output.name)
                  .c_str());

  return qnn_model_wrapper.AddTransposeNode(transpose_node_unit.Index(),
                                            reshape_input.name,
                                            transpose_output.name,
                                            input_shape,
                                            perm_u32,
                                            output_shape,
                                            data_type,
                                            quant_param,
                                            validate,
                                            qnn_model_wrapper.IsGraphInput(reshape_input.name),
                                            qnn_model_wrapper.IsGraphOutput(transpose_output.name));
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ReshapeTransposeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  if (reshape_node_unit.OpType() != "Reshape") {
    return nullptr;
  }

  const OrtNodeUnit* transpose_node_unit = GetChildNodeUnitAllowQdq(
      qnn_model_wrapper, reshape_node_unit, "Transpose",
      node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose_node_unit == nullptr) {
    return nullptr;
  }

  std::vector<int64_t> fused_perm;
  if (!ComputeFusedPerm(reshape_node_unit, *transpose_node_unit, fused_perm)) {
    return nullptr;
  }

  // Commit only if QNN can build the collapsed op.
  if (!CreateOrValidateOnQnn(qnn_model_wrapper, reshape_node_unit, *transpose_node_unit,
                             fused_perm, /*validate=*/true, logger)
           .IsOK()) {
    return nullptr;
  }

  const char* kind = IsIdentityPerm(fused_perm) ? "noop Reshape" : "single Transpose";
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
              ("[ReshapeTransposeFusion] Collapsing Reshape (" + reshape_node_unit.Name() +
               ") -> Transpose (" + transpose_node_unit->Name() + ") to " + kind)
                  .c_str());
  return std::make_unique<ReshapeTransposeFusion>(reshape_node_unit, *transpose_node_unit,
                                                  std::move(fused_perm));
}

gsl::span<const OrtNodeUnit* const> ReshapeTransposeFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status ReshapeTransposeFusion::IsSupported(QnnModelWrapper& qnn_model_wrapper,
                                                const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, *node_units_[0], *node_units_[1],
                               fused_perm_, /*validate=*/true, logger);
}

Ort::Status ReshapeTransposeFusion::AddToModelBuilder(QnnModelWrapper& qnn_model_wrapper,
                                                      const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, *node_units_[0], *node_units_[1],
                               fused_perm_, /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
