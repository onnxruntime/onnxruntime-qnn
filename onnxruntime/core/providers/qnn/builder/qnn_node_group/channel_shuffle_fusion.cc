// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/channel_shuffle_fusion.h"

#include <array>
#include <gsl/gsl>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
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
constexpr char kOpChannelShuffle[] = "ChannelShuffle";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpReshape[] = "Reshape";

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

std::optional<std::vector<int64_t>> GetTransposePerm(const OrtNodeUnit& transpose) {
  if (transpose.OpType() != kOpTranspose) {
    return std::nullopt;
  }
  OrtNodeAttrHelper helper(transpose);
  std::vector<int64_t> perm;
  return helper.Get(kAttrTransposePerm, std::vector<int64_t>());
}

std::vector<int64_t> InvertTransposePerm(gsl::span<const int64_t> perm) {
  const size_t perm_size = perm.size();
  std::vector<int64_t> perm_inverse(perm_size);
  for (size_t i = 0; i < perm_size; ++i) {
    size_t j = gsl::narrow_cast<size_t>(perm[i]);
    perm_inverse[j] = gsl::narrow_cast<int64_t>(i);
  }
  return perm_inverse;
}

bool IsCancelingTransposePermPair(
    std::optional<gsl::span<const int64_t>> perm1,
    std::optional<gsl::span<const int64_t>> perm2) {
  if (!perm1.has_value() || !perm2.has_value()) {
    return false;
  }
  if (perm1->size() != perm2->size()) {
    return false;
  }
  std::vector<int64_t> perm1_inverted_vector = InvertTransposePerm(*perm1);
  auto perm1_inverted = gsl::make_span<const int64_t>(
      perm1_inverted_vector.data(), perm1_inverted_vector.size());
  if (perm1_inverted != perm2.value()) {
    return false;
  }
  return true;
}

// Normalize a potentially collapsed rank-N intermediate reshape output shape back to rank-N+1.
// ORT's TransposeOptimizer may collapse the leading unit batch dimension (N=1) in the
// post-Layout-Transform pass, dropping rank by 1. Re-inserting a unit dim at index 0
// restores the expected shape. Returns false if the shape cannot be interpreted as a valid
// channel-split, and sets was_collapsed=true if a unit dim was actually re-inserted.
bool NormalizeChannelShuffleIntermediateShape(const std::vector<uint32_t>& input_dims,
                                              std::vector<uint32_t>& reshape1_output_dims,
                                              bool& was_collapsed) {
  was_collapsed = false;
  const size_t in_rank = input_dims.size();
  const size_t out_rank = reshape1_output_dims.size();
  // Normal case: output rank is input rank + 1.
  if (out_rank == in_rank + 1) {
    return true;
  }
  // Collapsed case: output rank == input rank and the leading unit batch was removed.
  // The layout-specific channel split is validated after restoring the batch dimension.
  if (out_rank == in_rank && input_dims[0] == 1) {
    reshape1_output_dims.insert(reshape1_output_dims.begin(), 1);
    was_collapsed = true;
    return true;
  }
  return false;
}

struct ChannelShuffleCoreInfo {
  uint32_t num_groups = 0;
};

std::optional<ChannelShuffleCoreInfo> GetChannelShuffleCoreInfo(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1,
    const OrtNodeUnit& transpose,
    const OrtNodeUnit& reshape2) {
  std::vector<uint32_t> input_shape;
  std::vector<uint32_t> intermediate_shape;
  std::vector<uint32_t> output_shape;
  if (!qnn_model_wrapper.GetOnnxShape(reshape1.Inputs()[0].shape, input_shape) || input_shape.size() < 3 ||
      !qnn_model_wrapper.GetOnnxShape(reshape1.Outputs()[0].shape, intermediate_shape) ||
      !qnn_model_wrapper.GetOnnxShape(reshape2.Outputs()[0].shape, output_shape) ||
      output_shape.size() != input_shape.size()) {
    return std::nullopt;
  }

  bool was_collapsed = false;
  if (!NormalizeChannelShuffleIntermediateShape(input_shape, intermediate_shape, was_collapsed) ||
      intermediate_shape.size() != input_shape.size() + 1 ||
      intermediate_shape[0] != input_shape[0]) {
    return std::nullopt;
  }

  std::optional<std::vector<int64_t>> raw_perm = GetTransposePerm(transpose);
  if (!raw_perm.has_value()) {
    return std::nullopt;
  }
  std::vector<int64_t> perm = std::move(raw_perm.value());
  if (was_collapsed) {
    if (perm.size() + 1 != intermediate_shape.size()) {
      return std::nullopt;
    }
    std::vector<int64_t> lifted_perm;
    lifted_perm.reserve(intermediate_shape.size());
    lifted_perm.push_back(0);
    for (int64_t axis : perm) {
      lifted_perm.push_back(axis + 1);
    }
    perm = std::move(lifted_perm);
  }
  std::vector<int64_t> nchw_perm(intermediate_shape.size());
  std::iota(nchw_perm.begin(), nchw_perm.end(), 0);
  std::swap(nchw_perm[1], nchw_perm[2]);

  // When the leading NHWC->NCHW transpose is absorbed, the middle transpose both
  // swaps the split channel dimensions and moves them ahead of the spatial dimensions:
  // [N, spatial..., G, C/G] -> [N, C/G, G, spatial...].
  std::vector<int64_t> nhwc_to_nchw_perm;
  nhwc_to_nchw_perm.reserve(intermediate_shape.size());
  nhwc_to_nchw_perm.push_back(0);
  nhwc_to_nchw_perm.push_back(gsl::narrow_cast<int64_t>(intermediate_shape.size() - 1));
  nhwc_to_nchw_perm.push_back(gsl::narrow_cast<int64_t>(intermediate_shape.size() - 2));
  for (size_t axis = 1; axis + 2 < intermediate_shape.size(); ++axis) {
    nhwc_to_nchw_perm.push_back(gsl::narrow_cast<int64_t>(axis));
  }

  if (perm == nchw_perm) {
    if (output_shape != input_shape || input_shape[1] != intermediate_shape[1] * intermediate_shape[2] ||
        !std::equal(input_shape.begin() + 2, input_shape.end(), intermediate_shape.begin() + 3)) {
      return std::nullopt;
    }
    return ChannelShuffleCoreInfo{intermediate_shape[1]};
  }

  if (perm == nhwc_to_nchw_perm) {
    const size_t split_axis = intermediate_shape.size() - 2;
    std::vector<uint32_t> expected_nchw_output;
    expected_nchw_output.reserve(input_shape.size());
    expected_nchw_output.push_back(input_shape[0]);
    expected_nchw_output.push_back(input_shape.back());
    expected_nchw_output.insert(expected_nchw_output.end(), input_shape.begin() + 1, input_shape.end() - 1);
    if (input_shape.back() != intermediate_shape[split_axis] * intermediate_shape[split_axis + 1] ||
        !std::equal(input_shape.begin() + 1, input_shape.end() - 1, intermediate_shape.begin() + 1) ||
        output_shape != expected_nchw_output) {
      return std::nullopt;
    }
    return ChannelShuffleCoreInfo{intermediate_shape[split_axis]};
  }

  return std::nullopt;
}

/// @brief Match pattern: Transpose -> ChannelShuffle (Reshape -> Transpose -> Reshape) -> Transpose
/// E.g.,: T(perm=[0, 2, 1, 3]) -> R(N, G, C/G, H, W) -> T(perm=[0, 1, 3, 2, 4]) -> R(N, C, H, W) -> T(perm=[0, 2, 1, 3])
/// @param qnn_model_wrapper QNN model wrapper.
/// @param transpose_head The first transpose node starting the pattern.
/// @param node_to_node_unit Maps a Node to a NodeUnit.
/// @param node_unit_to_qnn_node_group Maps a NodeUnit to a IQnnNodeGroup.
/// @return The matched pattern as an array of NodeUnits if found, otherwise std::nullopt.
/// @note This is ChannelShuffle with transpose wraps commonly seen ORT post partitioning.
std::optional<std::array<const OrtNodeUnit*, 5>> MatchChannelShufflePattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit* transpose_head,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  // Helper function to get a single child of a specific type
  auto GetChildOfType = [&](const OrtNodeUnit& node, std::string_view expect_type) -> const OrtNodeUnit* {
    const std::array<std::string_view, 1> child_op_types{expect_type};
    const OrtNodeUnit* child = GetOnlyChildOfType(qnn_model_wrapper, node, child_op_types,
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
    if (child == nullptr) {
      return nullptr;
    }
    if (child->OpType() != expect_type) {
      return nullptr;
    }
    if (child->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return nullptr;
    }
    return child;
  };

  if (transpose_head->OpType() != kOpTranspose) {
    return std::nullopt;
  }
  if (transpose_head->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }
  const OrtNodeUnit* reshape1 = GetChildOfType(*transpose_head, kOpReshape);
  if (reshape1 == nullptr) {
    return std::nullopt;
  }
  const OrtNodeUnit* transpose = GetChildOfType(*reshape1, kOpTranspose);
  if (transpose == nullptr) {
    return std::nullopt;
  }
  const OrtNodeUnit* reshape2 = GetChildOfType(*transpose, kOpReshape);
  if (reshape2 == nullptr) {
    return std::nullopt;
  }
  const OrtNodeUnit* transpose_tail = GetChildOfType(*reshape2, kOpTranspose);
  if (transpose_tail == nullptr) {
    return std::nullopt;
  }
  return std::array<const OrtNodeUnit*, 5>{transpose_head, reshape1, transpose, reshape2, transpose_tail};
}

/// @brief Create or validate the QNN node of type ChannelShuffle.
/// @param qnn_model_wrapper QNN model wrapper
/// @param node_units The node units containing the nodes in pattern
/// @param validate Whether to validate the QNN node
/// @return Status
Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                  gsl::span<const OrtNodeUnit* const> node_units,
                                  bool validate) {
  // node_units layout:
  //   Full 5-node: [T_head, Reshape1, T_mid, Reshape2, T_tail]
  //   4-node (no head): a fresh 4-element span [Reshape1, T_mid, Reshape2, T_tail]
  //                     (index 0 = Reshape1)
  //
  // Detect the variant by checking if the first node is a Transpose (5-node) or Reshape (4-node).
  const bool has_head_transpose = (node_units[0]->OpType() == kOpTranspose);
  const OrtNodeUnit* transpose_head = has_head_transpose ? node_units[0] : nullptr;
  const OrtNodeUnit* reshape1 = has_head_transpose ? node_units[1] : node_units[0];
  const OrtNodeUnit* transpose = has_head_transpose ? node_units[2] : node_units[1];
  const OrtNodeUnit* reshape2 = has_head_transpose ? node_units[3] : node_units[2];
  const OrtNodeUnit* transpose_tail = has_head_transpose ? node_units[4] : node_units[3];
  // IO boundaries: T_head's input (if present) else Reshape1's input; T_tail's output.
  const OrtNodeUnitIODef& cs_input_def = has_head_transpose ? transpose_head->Inputs()[0]
                                                            : reshape1->Inputs()[0];
  const OrtNodeUnitIODef& cs_output_def = transpose_tail->Outputs()[0];

  std::vector<std::string> param_tensor_names;
  std::vector<Qnn_Param_t> param_tensors;
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Get input shape to determine channel axis.
  // Use the channel shuffle's input node (T_head if present, else Reshape1).
  {
    const OrtNodeUnit* input_node = has_head_transpose ? transpose_head : reshape1;
    size_t num_input_node_inputs = 0;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumInputs(&input_node->GetNode(), &num_input_node_inputs));
    std::vector<const OrtValueInfo*> input_node_inputs(num_input_node_inputs);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetInputs(&input_node->GetNode(),
                                                      input_node_inputs.data(),
                                                      input_node_inputs.size()));
    const OrtValueInfo* input_info = input_node_inputs[0];
    const OrtTypeInfo* input_type_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(input_info, &input_type_info));
    const OrtTensorTypeAndShapeInfo* input_tensor_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info));

    // Get dimensions count
    size_t input_dims_count = 0;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensionsCount(input_tensor_info, &input_dims_count));

    // Set channel axis parameter (last axis for NHWC format)
    const uint32_t channel_axis = static_cast<uint32_t>(input_dims_count - 1);
    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, transpose_tail->Index(),
                                           transpose_tail->Name(), channel_axis,
                                           QNN_OP_CHANNEL_SHUFFLE_PARAM_AXIS, param_tensor_names));
  }

  const auto core_info = GetChannelShuffleCoreInfo(qnn_model_wrapper, *reshape1, *transpose, *reshape2);
  RETURN_IF_NOT(core_info.has_value(), "ChannelShuffleFusion: invalid reshape/transpose core signature.");
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, transpose_tail->Index(),
                                         transpose_tail->Name(), core_info->num_groups,
                                         QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS, param_tensor_names));

  // Create tensor wrappers for input and output
  QnnTensorWrapper channel_shuffle_input;
  QnnTensorWrapper channel_shuffle_output;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(cs_input_def, channel_shuffle_input));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(cs_output_def, channel_shuffle_output));

  // Note: Skipped QNN validation API due to its inconsistent behavior than creation API. Re-enable it when fixed.
  if (!validate) {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(channel_shuffle_input)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(channel_shuffle_output)), "Failed to add output");
    const std::string_view node_name_suffix = has_head_transpose ? "" : "_from_reshape";
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(onnxruntime::qnn::utils::UniqueNameGenerator().New(
                                                      *transpose_tail, node_name_suffix),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_CHANNEL_SHUFFLE,
                                                  {cs_input_def.name},
                                                  {cs_output_def.name},
                                                  std::move(param_tensor_names),
                                                  validate),
                  ("Failed to add fused " + std::string(kOpChannelShuffle) + " node.").c_str());
  }

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ChannelShuffleFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& transpose_head,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  std::optional<std::array<const OrtNodeUnit*, 5>> pattern = MatchChannelShufflePattern(
      qnn_model_wrapper, &transpose_head, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!pattern.has_value()) {
    return nullptr;
  }
  const OrtNodeUnit* reshape1 = pattern->at(1);
  const OrtNodeUnit* transpose = pattern->at(2);
  const OrtNodeUnit* reshape2 = pattern->at(3);
  const OrtNodeUnit* transpose_tail = pattern->at(4);
  if (!GetChannelShuffleCoreInfo(qnn_model_wrapper, *reshape1, *transpose, *reshape2).has_value()) {
    return nullptr;
  }

  // Check if the first and last transpose is a canceling transpose pair
  std::optional<std::vector<int64_t>> perm_head = GetTransposePerm(transpose_head);
  if (!perm_head.has_value()) {
    return nullptr;
  }
  std::optional<std::vector<int64_t>> perm_tail = GetTransposePerm(*transpose_tail);
  if (!perm_tail.has_value()) {
    return nullptr;
  }

  auto perm_head_span = gsl::make_span<const int64_t>(perm_head.value().data(), perm_head.value().size());
  auto perm_tail_span = gsl::make_span<const int64_t>(perm_tail.value().data(), perm_tail.value().size());

  if (!IsCancelingTransposePermPair(perm_head_span, perm_tail_span)) {
    return nullptr;
  }

  if (!CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), /*validate=*/true).IsOK()) {
    return nullptr;
  }
  return std::make_unique<ChannelShuffleFusion>(pattern.value());
}

std::unique_ptr<IQnnNodeGroup> ChannelShuffleFusion::TryFusionFromReshape(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  // Bare RTR pattern: Reshape1 -> T_mid -> Reshape2 -> T_tail
  // Used when ORT 1.29's TransposeOptimizer absorbed the leading Transpose into Reshape1's shape.
  // Only try this in the post-Layout-Transform pass where the absorption happens.
  if (!qnn_model_wrapper.IsPostLayoutTransform()) {
    return nullptr;
  }
  if (reshape1_node_unit.OpType() != kOpReshape) {
    return nullptr;
  }

  auto GetChildOfTypeCS = [&](const OrtNodeUnit& node, std::string_view expect_type) -> const OrtNodeUnit* {
    const std::array<std::string_view, 1> child_op_types{expect_type};
    const OrtNodeUnit* child = GetOnlyChildOfType(qnn_model_wrapper, node, child_op_types,
                                                  node_to_node_unit, node_unit_to_qnn_node_group);
    if (child == nullptr || child->OpType() != expect_type) {
      return nullptr;
    }
    if (child->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return nullptr;
    }
    return child;
  };

  const OrtNodeUnit* transpose_mid = GetChildOfTypeCS(reshape1_node_unit, kOpTranspose);
  if (transpose_mid == nullptr) return nullptr;

  const OrtNodeUnit* reshape2 = GetChildOfTypeCS(*transpose_mid, kOpReshape);
  if (reshape2 == nullptr) return nullptr;

  const OrtNodeUnit* transpose_tail = GetChildOfTypeCS(*reshape2, kOpTranspose);
  if (transpose_tail == nullptr) return nullptr;

  // Validate the complete ChannelShuffle core before claiming the Reshape. In particular,
  // use the middle transpose permutation to identify NCHW vs NHWC and then require the
  // corresponding channel split, spatial dimensions, and Reshape2 output shape.
  if (!GetChannelShuffleCoreInfo(qnn_model_wrapper, reshape1_node_unit, *transpose_mid, *reshape2).has_value()) {
    return nullptr;
  }
  const std::optional<std::vector<int64_t>> tail_perm = GetTransposePerm(*transpose_tail);
  constexpr std::array<int64_t, 4> kNchwToNhwcPerm{0, 2, 3, 1};
  if (!tail_perm.has_value() || tail_perm->size() != kNchwToNhwcPerm.size() ||
      !std::equal(tail_perm->begin(), tail_perm->end(), kNchwToNhwcPerm.begin())) {
    return nullptr;
  }

  // Validate shape: Reshape1 input must equal T_tail output (ChannelShuffle is shape-preserving at group boundaries).
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  size_t r1_in_count = 0, t_tail_out_count = 0;
  std::vector<const OrtValueInfo*> r1_ins, t_tail_outs;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&reshape1_node_unit.GetNode(), &r1_in_count), ort_api, nullptr);
  r1_ins.resize(r1_in_count);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&reshape1_node_unit.GetNode(), r1_ins.data(), r1_in_count), ort_api, nullptr);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&transpose_tail->GetNode(), &t_tail_out_count), ort_api, nullptr);
  t_tail_outs.resize(t_tail_out_count);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&transpose_tail->GetNode(), t_tail_outs.data(), t_tail_out_count), ort_api, nullptr);

  const OrtTypeInfo* r1_in_ti = nullptr;
  const OrtTensorTypeAndShapeInfo* r1_in_tsi = nullptr;
  const OrtTypeInfo* t_tail_out_ti = nullptr;
  const OrtTensorTypeAndShapeInfo* t_tail_out_tsi = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(r1_ins[0], &r1_in_ti), ort_api, nullptr);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(r1_in_ti, &r1_in_tsi), ort_api, nullptr);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(t_tail_outs[0], &t_tail_out_ti), ort_api, nullptr);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(t_tail_out_ti, &t_tail_out_tsi), ort_api, nullptr);

  size_t r1_in_rank = 0, t_tail_out_rank = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(r1_in_tsi, &r1_in_rank), ort_api, nullptr);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(t_tail_out_tsi, &t_tail_out_rank), ort_api, nullptr);

  // ChannelShuffle is shape-preserving; input and output must have same rank.
  if (r1_in_rank != t_tail_out_rank || r1_in_rank < 2) {
    return nullptr;
  }

  std::vector<int64_t> r1_in_dims(r1_in_rank), t_tail_out_dims(t_tail_out_rank);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensions(r1_in_tsi, r1_in_dims.data(), r1_in_rank), ort_api, nullptr);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensions(t_tail_out_tsi, t_tail_out_dims.data(), t_tail_out_rank), ort_api, nullptr);

  if (!std::equal(r1_in_dims.begin(), r1_in_dims.end(), t_tail_out_dims.begin())) {
    return nullptr;
  }

  // Build 4-node pattern array: [Reshape1, T_mid, Reshape2, T_tail]
  std::array<const OrtNodeUnit*, 4> four_node_pattern{&reshape1_node_unit, transpose_mid, reshape2, transpose_tail};
  gsl::span<const OrtNodeUnit* const> four_span{four_node_pattern.data(), four_node_pattern.size()};

  if (!CreateOrValidateOnQnn(qnn_model_wrapper, four_span, /*validate=*/true).IsOK()) {
    return nullptr;
  }
  return std::make_unique<ChannelShuffleFusion>(four_span, ChannelShuffleFusion::NoHeadTransposeTag{});
}

gsl::span<const OrtNodeUnit* const> ChannelShuffleFusion::GetNodeUnits() const {
  // For the no-head-transpose variant, node_units_[0] is nullptr; skip it.
  size_t start = has_head_transpose_ ? 0 : 1;
  return gsl::span<const OrtNodeUnit* const>{node_units_.data() + start, node_units_.size() - start};
}

Ort::Status ChannelShuffleFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), /*validate=*/true);
}

Ort::Status ChannelShuffleFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), /*validate=*/false);
}

}  // namespace qnn
}  // namespace onnxruntime
