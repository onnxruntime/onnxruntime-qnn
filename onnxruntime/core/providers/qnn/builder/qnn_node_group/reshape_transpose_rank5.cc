// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_transpose_rank5.h"

#include <gsl/gsl>
#include <cstdint>
#include <optional>
#include <utility>
#include <string>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include <sstream>

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

constexpr size_t kRank6 = 6;
constexpr size_t kRank5 = 5;
constexpr const char* kOpTypeReshape = "Reshape";
constexpr const char* kOpTypeTranspose = "Transpose";
constexpr const char* kAttrTransposePerm = "perm";

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

// Validated rank reduction for a matched triple.
struct Reduction {
  RankReduction kind;
  size_t index;  // Squeeze only; always 0 for MergeLeading.
};

/// @brief Match the pattern: Reshape -> Transpose -> Reshape with rank-6 intermediate tensors
std::optional<std::array<const OrtNodeUnit*, 3>> MatchRank6ToRank5Pattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit* reshape1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  // Validate first Reshape in pattern - allow both SingleNode and QDQGroup
  if (reshape1->OpType() != kOpTypeReshape) {
    return std::nullopt;
  }

  // Get Transpose child (middle node in pattern) - allow both SingleNode and QDQGroup
  const OrtNodeUnit* transpose = GetChildNodeUnitAllowQdq(
      qnn_model_wrapper, *reshape1, kOpTypeTranspose, node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose == nullptr) {
    return std::nullopt;
  }

  // Get second Reshape child (last node in pattern) - allow both SingleNode and QDQGroup
  const OrtNodeUnit* reshape2 = GetChildNodeUnitAllowQdq(
      qnn_model_wrapper, *transpose, kOpTypeReshape, node_to_node_unit, node_unit_to_qnn_node_group);
  if (reshape2 == nullptr) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 3>{reshape1, transpose, reshape2};
}

/// @brief Validate the pattern conditions and compute the rank reduction.
/// Squeeze is preferred: the first fixed unit dim wins, mirroring tetracode's
/// DropBatch-before-MergeDims order.
///
/// Both reductions are pure view operations, so validity depends only on which axes
/// survive and never on the permutation's semantics; the perm is read purely to build
/// the rank-5 equivalent. SpaceToDepth permutations are deliberately not excluded here.
/// Priority comes from fusions-table order -- SpaceToDepthFusion is tried first -- so it
/// claims the patterns it can represent, and this fusion picks up the ones it declines.
/// It declines bare RTR before Layout Transform runs (spacetodepth_fusion.cc), which is
/// exactly the case that previously fell back to the CPU.
std::optional<Reduction> ValidatePatternConditions(
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

  // Condition 1: Rank(t1) == Rank(t2) == 6
  if (t1_shape->size() != kRank6 || t2_shape->size() != kRank6) {
    return std::nullopt;
  }

  const auto& t1_dims = *t1_shape;
  const auto& t2_dims = *t2_shape;

  if (t1_dims.empty() || t2_dims.empty()) {
    return std::nullopt;
  }

  // Condition 2: transpose perm, needed by both modes below.
  OrtNodeAttrHelper transpose_helper(*transpose);
  std::vector<int64_t> perm = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm.size() != kRank6) {
    return std::nullopt;
  }

  // Squeeze the first unit dimension left in place by the transpose.
  for (size_t i = 0; i < kRank6; ++i) {
    if (t1_dims[i] == 1 && t2_dims[i] == 1 && perm[i] == static_cast<int64_t>(i)) {
      return Reduction{RankReduction::Squeeze, i};
    }
  }

  // Otherwise merge the first two input dims when they travel together. With
  // perm[0] == 0, t1_dims[0] != 1 here (a unit dim 0 would have matched above).
  if (perm[0] == 0 && perm[1] == 1) {
    return Reduction{RankReduction::MergeLeading, 0};
  }

  return std::nullopt;
}

/// @brief Create or validate the QNN nodes with rank-5 tensors
Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    Reduction reduction,
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

  // Get original shapes
  std::vector<uint32_t> t1_dims;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(reshape1_output.shape, t1_dims),
                ("Cannot get shape for " + reshape1_output.name).c_str());

  std::vector<uint32_t> t2_dims;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(transpose_output.shape, t2_dims),
                ("Cannot get shape for " + transpose_output.name).c_str());

  if (t1_dims.size() != kRank6 || t2_dims.size() != kRank6) {
    return Ort::Status("Expected rank-6 intermediate tensors", OrtErrorCode::ORT_FAIL);
  }

  std::vector<uint32_t> t1_rank5_dims;
  t1_rank5_dims.reserve(kRank5);
  std::vector<uint32_t> t2_rank5_dims;
  t2_rank5_dims.reserve(kRank5);
  std::vector<uint32_t> perm_rank5;
  perm_rank5.reserve(kRank5);

  // Get transpose permutation (re-read here; the graph is unchanged since validation).
  OrtNodeAttrHelper transpose_helper(*transpose);
  std::vector<int64_t> perm = transpose_helper.Get(kAttrTransposePerm, std::vector<int64_t>{});
  if (perm.size() != kRank6) {
    return Ort::Status("Expected rank-6 permutation", OrtErrorCode::ORT_FAIL);
  }

  if (reduction.kind == RankReduction::MergeLeading) {
    // Dims 0 and 1 travel together (perm[0] == 0 && perm[1] == 1, validated), so
    // output position 1 disappears into position 0.
    const uint64_t merged_t1 = static_cast<uint64_t>(t1_dims[0]) * t1_dims[1];
    const uint64_t merged_t2 = static_cast<uint64_t>(t2_dims[0]) * t2_dims[1];
    if (merged_t1 > UINT32_MAX || merged_t2 > UINT32_MAX) {
      return Ort::Status("Merged dimension overflows uint32", OrtErrorCode::ORT_FAIL);
    }
    t1_rank5_dims.push_back(static_cast<uint32_t>(merged_t1));
    t1_rank5_dims.insert(t1_rank5_dims.end(), t1_dims.begin() + 2, t1_dims.end());
    t2_rank5_dims.push_back(static_cast<uint32_t>(merged_t2));
    t2_rank5_dims.insert(t2_rank5_dims.end(), t2_dims.begin() + 2, t2_dims.end());
    perm_rank5.push_back(0);
    for (size_t i = 2; i < perm.size(); ++i) {
      // perm[i] >= 2 here (values 0 and 1 sit at positions 0 and 1).
      perm_rank5.push_back(static_cast<uint32_t>(perm[i] - 1));
    }
  } else {
    const size_t unit_dim_index = reduction.index;
    // Create rank-5 shape for t1 (remove unit dimension at unit_dim_index)
    for (size_t i = 0; i < t1_dims.size(); ++i) {
      if (i != unit_dim_index) {
        t1_rank5_dims.push_back(static_cast<uint32_t>(t1_dims[i]));
      }
    }

    // Create rank-5 shape for t2 (remove unit dimension at unit_dim_index)
    for (size_t i = 0; i < t2_dims.size(); ++i) {
      if (i != unit_dim_index) {
        t2_rank5_dims.push_back(static_cast<uint32_t>(t2_dims[i]));
      }
    }

    // Remove unit dimension and adjust indices
    for (size_t i = 0; i < perm.size(); ++i) {
      if (i != unit_dim_index) {
        int64_t perm_val = perm[i];
        // Adjust index: if perm_val > unit_dim_index, subtract 1
        if (perm_val > static_cast<int64_t>(unit_dim_index)) {
          perm_val--;
        }
        perm_rank5.push_back(static_cast<uint32_t>(perm_val));
      }
    }
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

  // Create Reshape1 output tensor wrapper.
  TensorInfo reshape1_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(reshape1_output, reshape1_output_info));

  QnnTensorWrapper reshape1_output_tensor_wrapper(reshape1_output.name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  reshape1_output_info.qnn_data_type,
                                                  std::move(reshape1_output_info.quant_param),
                                                  std::move(t1_rank5_dims));
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

  // Create Transpose output tensor wrapper.
  TensorInfo transpose_output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(transpose_output, transpose_output_info));

  QnnTensorWrapper transpose_output_tensor_wrapper(transpose_output.name,
                                                   QNN_TENSOR_TYPE_NATIVE,
                                                   transpose_output_info.qnn_data_type,
                                                   std::move(transpose_output_info.quant_param),
                                                   std::move(t2_rank5_dims));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(transpose_output_tensor_wrapper)),
                "Failed to add Transpose's output tensor.");

  // Create Transpose perm parameter wrapper.
  QnnParamWrapper perm_param(transpose->Index(),
                             transpose->Name(),
                             QNN_OP_TRANSPOSE_PARAM_PERM,
                             {static_cast<uint32_t>(perm_rank5.size())},
                             std::move(perm_rank5));
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

  // Create Reshape2 output tensor wrapper.
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

std::unique_ptr<IQnnNodeGroup> Rank6ToRank5Fusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // Match the pattern
  std::optional<std::array<const OrtNodeUnit*, 3>> pattern = MatchRank6ToRank5Pattern(
      qnn_model_wrapper, &reshape1_node_unit, node_to_node_unit, node_unit_to_qnn_node_group, logger);

  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* reshape1 = pattern->at(0);
  const OrtNodeUnit* transpose = pattern->at(1);
  const OrtNodeUnit* reshape2 = pattern->at(2);

  // Validate pattern conditions and compute the rank reduction
  auto reduction = ValidatePatternConditions(reshape1, transpose, reshape2, qnn_model_wrapper, logger);
  if (!reduction.has_value()) {
    return nullptr;
  }

  // Validate on QNN
  if (!CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), reduction.value(), /*validate=*/true, logger).IsOK()) {
    return nullptr;
  }

  return std::make_unique<Rank6ToRank5Fusion>(pattern.value(), reduction->kind,
                                              reduction->index);
}

gsl::span<const OrtNodeUnit* const> Rank6ToRank5Fusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status Rank6ToRank5Fusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  const Reduction reduction{kind_, index_};
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), reduction, /*validate=*/true, logger);
}

Ort::Status Rank6ToRank5Fusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  const Reduction reduction{kind_, index_};
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), reduction, /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
