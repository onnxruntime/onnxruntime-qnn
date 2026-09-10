// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/channel_shuffle_fusion.h"

#include <array>
#include <gsl/gsl>
#include <memory>
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
bool NormalizeChannelShuffleIntermediateShape(const std::vector<int64_t>& input_dims,
                                              std::vector<int64_t>& reshape1_output_dims,
                                              bool& was_collapsed) {
  was_collapsed = false;
  const size_t in_rank = input_dims.size();
  const size_t out_rank = reshape1_output_dims.size();
  // Normal case: output rank is input rank + 1.
  if (out_rank == in_rank + 1) {
    return true;
  }
  // Collapsed case: output rank == input rank, input[0]==1, and input[1] == out[0]*out[1].
  if (out_rank == in_rank && in_rank >= 2 && input_dims[0] == 1 &&
      out_rank >= 2 && input_dims[1] == reshape1_output_dims[0] * reshape1_output_dims[1]) {
    reshape1_output_dims.insert(reshape1_output_dims.begin(), 1);
    was_collapsed = true;
    return true;
  }
  return false;
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
  // node_units layout (always 5 elements in the backing array, accessed via the outer ChannelShuffleFusion):
  //   Full 5-node: [T_head, Reshape1, T_mid, Reshape2, T_tail]
  //   4-node (no head): node_units passed here starts at index 1: [Reshape1, T_mid, Reshape2, T_tail]
  //
  // Detect the variant by checking if the first node is a Transpose (5-node) or Reshape (4-node).
  const bool has_head_transpose = (node_units[0]->OpType() == kOpTranspose);
  const OrtNodeUnit* transpose_head = has_head_transpose ? node_units[0] : nullptr;
  const OrtNodeUnit* reshape1      = has_head_transpose ? node_units[1] : node_units[0];
  const OrtNodeUnit* transpose_tail = has_head_transpose ? node_units[4] : node_units[3];
  // IO boundaries: T_head's input (if present) else Reshape1's input; T_tail's output.
  const OrtNodeUnitIODef& cs_input_def  = has_head_transpose ? transpose_head->Inputs()[0]
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

  // Extract number of groups from reshape1 output shape
  {
    // Get reshape1 input shape (needed to detect the collapsed unit-batch case)
    // reshape1 is already determined above as the Reshape starting the RTR pattern.
    size_t num_reshape1_inputs = 0;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumInputs(&reshape1->GetNode(), &num_reshape1_inputs));
    std::vector<const OrtValueInfo*> reshape1_inputs_list(num_reshape1_inputs);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetInputs(&reshape1->GetNode(),
                                                      reshape1_inputs_list.data(),
                                                      reshape1_inputs_list.size()));
    const OrtTypeInfo* r1_in_type_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(reshape1_inputs_list[0], &r1_in_type_info));
    const OrtTensorTypeAndShapeInfo* r1_in_tensor_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(r1_in_type_info, &r1_in_tensor_info));
    size_t r1_in_dims_count = 0;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensionsCount(r1_in_tensor_info, &r1_in_dims_count));
    std::vector<int64_t> r1_in_dims(r1_in_dims_count);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensions(r1_in_tensor_info, r1_in_dims.data(), r1_in_dims_count));

    // Get reshape1 output shape
    size_t num_reshape1_outputs = 0;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumOutputs(&reshape1->GetNode(), &num_reshape1_outputs));
    std::vector<const OrtValueInfo*> reshape1_outputs(num_reshape1_outputs);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetOutputs(&reshape1->GetNode(),
                                                       reshape1_outputs.data(),
                                                       reshape1_outputs.size()));
    const OrtValueInfo* reshape1_output_info = reshape1_outputs[0];
    const OrtTypeInfo* reshape1_output_type_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(reshape1_output_info, &reshape1_output_type_info));
    const OrtTensorTypeAndShapeInfo* reshape1_output_tensor_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(reshape1_output_type_info,
                                                                &reshape1_output_tensor_info));

    // Get dimensions
    size_t reshape1_output_dims_count = 0;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensionsCount(reshape1_output_tensor_info, &reshape1_output_dims_count));
    std::vector<int64_t> reshape1_output_dims(reshape1_output_dims_count);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensions(reshape1_output_tensor_info,
                                                     reshape1_output_dims.data(),
                                                     reshape1_output_dims_count));

    // Normalize the intermediate shape to handle ORT 1.29 unit-batch-dim collapse.
    bool was_collapsed = false;
    if (!NormalizeChannelShuffleIntermediateShape(r1_in_dims, reshape1_output_dims, was_collapsed)) {
      // NCHW normalization failed; try NHWC detection.
      // In NHWC, input is {N,H,W,C} and output is {N,H,W,G,C/G} where C at index 3 → G+C/G.
      // Check if input[3] == output[3] * output[4] (NHWC groups split).
      if (r1_in_dims.size() == 4 && reshape1_output_dims.size() == 5 &&
          r1_in_dims[0] == reshape1_output_dims[0] &&    // N matches
          r1_in_dims[3] == reshape1_output_dims[3] * reshape1_output_dims[4]) {  // C = G * C/G
        // NHWC format: num_groups is at output index 3.
        RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, transpose_tail->Index(),
                                               transpose_tail->Name(),
                                               static_cast<uint32_t>(reshape1_output_dims[3]),
                                               QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS, param_tensor_names));
        // Done with num_groups extraction for NHWC.
      } else {
        RETURN_IF_NOT(false, "ChannelShuffleFusion: unexpected reshape1 output shape during CreateOrValidate.");
      }
    } else {
      // NormalizeChannelShuffleIntermediateShape succeeded.
      // For NCHW input {N,C,H,W}: num_groups = output[1] = G.
      // For NHWC input {N,H,W,C}: num_groups = output[3] = G (after {N,H,W,G,C/G} split).
      // Detect format: NCHW has input[1]==C matched by output[1]*output[2]; NHWC has input[3]==C.
      int64_t num_groups_val = 0;
      if (!r1_in_dims.empty() && !reshape1_output_dims.empty() &&
          r1_in_dims.size() >= 4 && reshape1_output_dims.size() >= 3 &&
          r1_in_dims[3] == reshape1_output_dims[reshape1_output_dims.size() - 2] *
                           reshape1_output_dims[reshape1_output_dims.size() - 1]) {
        // NHWC: last two dims of output are G and C/G; G is at output[-2].
        num_groups_val = reshape1_output_dims[reshape1_output_dims.size() - 2];
      } else {
        // NCHW: G is at output[1].
        num_groups_val = reshape1_output_dims[1];
      }
      fprintf(stderr, "CSDBG: NCHW/else branch, num_groups=%ld\n", (long)num_groups_val);
      // NCHW/normalized: num_groups is G; detect NHWC vs NCHW to find correct G index.
      // NHWC output {N,H,W,G,C/G}: C = G × C/G, so input[3] == output[-2] × output[-1] → G = output[-2].
      // NCHW output {N,G,C/G,H,W}: G is at output[1].
      RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, transpose_tail->Index(),
                                             transpose_tail->Name(),
                                             static_cast<uint32_t>(num_groups_val),
                                             QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS, param_tensor_names));
    }
  }

  // Create tensor wrappers for input and output
  QnnTensorWrapper channel_shuffle_input;
  QnnTensorWrapper channel_shuffle_output;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(cs_input_def, channel_shuffle_input));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(cs_output_def, channel_shuffle_output));

  // Note: Skipped QNN validation API due to its inconsistent behavior than creation API. Re-enable it when fixed.
  if (!validate) {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(channel_shuffle_input)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(channel_shuffle_output)), "Failed to add output");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(onnxruntime::qnn::utils::UniqueNameGenerator().New(*transpose_tail),
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
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Input shape to reshape1 must equal output shape of reshape2; and has rank > 2
  // Get reshape1 input shape
  size_t num_reshape1_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&reshape1->GetNode(), &num_reshape1_inputs),
                             ort_api,
                             nullptr);
  std::vector<const OrtValueInfo*> reshape1_inputs(num_reshape1_inputs);
  RETURN_DEFAULT_IF_API_FAIL(
      ort_api.Node_GetInputs(&reshape1->GetNode(), reshape1_inputs.data(), reshape1_inputs.size()),
      ort_api,
      nullptr);
  const OrtValueInfo* reshape1_input_info = reshape1_inputs[0];
  const OrtTypeInfo* reshape1_input_type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(reshape1_input_info, &reshape1_input_type_info),
                             ort_api,
                             nullptr);
  const OrtTensorTypeAndShapeInfo* reshape1_input_tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(reshape1_input_type_info, &reshape1_input_tensor_info),
                             ort_api,
                             nullptr);

  // Get reshape2 output shape
  size_t num_reshape2_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape2->GetNode(), &num_reshape2_outputs),
                             ort_api,
                             nullptr);
  std::vector<const OrtValueInfo*> reshape2_outputs(num_reshape2_outputs);
  RETURN_DEFAULT_IF_API_FAIL(
      ort_api.Node_GetOutputs(&reshape2->GetNode(), reshape2_outputs.data(), reshape2_outputs.size()),
      ort_api, nullptr);
  const OrtValueInfo* reshape2_output_info = reshape2_outputs[0];
  const OrtTypeInfo* reshape2_output_type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(reshape2_output_info, &reshape2_output_type_info),
                             ort_api,
                             nullptr);
  const OrtTensorTypeAndShapeInfo* reshape2_output_tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(
      ort_api.CastTypeInfoToTensorInfo(reshape2_output_type_info, &reshape2_output_tensor_info),
      ort_api,
      nullptr);

  // Compare dimensions
  size_t reshape1_input_dims_count = 0;
  size_t reshape2_output_dims_count = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(reshape1_input_tensor_info, &reshape1_input_dims_count),
                             ort_api,
                             nullptr);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(reshape2_output_tensor_info, &reshape2_output_dims_count),
                             ort_api,
                             nullptr);

  if (reshape1_input_dims_count != reshape2_output_dims_count) {
    return nullptr;
  }

  std::vector<int64_t> reshape1_input_dims(reshape1_input_dims_count);
  std::vector<int64_t> reshape2_output_dims(reshape2_output_dims_count);
  RETURN_DEFAULT_IF_API_FAIL(
      ort_api.GetDimensions(reshape1_input_tensor_info, reshape1_input_dims.data(), reshape1_input_dims_count),
      ort_api,
      nullptr);
  RETURN_DEFAULT_IF_API_FAIL(
      ort_api.GetDimensions(reshape2_output_tensor_info, reshape2_output_dims.data(), reshape2_output_dims_count),
      ort_api,
      nullptr);

  // Check if reshape1 input dims equal reshape2 output dims
  if (!std::equal(reshape1_input_dims.begin(), reshape1_input_dims.end(), reshape2_output_dims.begin())) {
    return nullptr;
  }

  // Get reshape1 output shape
  size_t num_reshape1_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape1->GetNode(), &num_reshape1_outputs),
                             ort_api,
                             nullptr);
  std::vector<const OrtValueInfo*> reshape1_outputs(num_reshape1_outputs);
  RETURN_DEFAULT_IF_API_FAIL(
      ort_api.Node_GetOutputs(&reshape1->GetNode(), reshape1_outputs.data(), reshape1_outputs.size()),
      ort_api,
      nullptr);
  const OrtValueInfo* reshape1_output_info = reshape1_outputs[0];
  const OrtTypeInfo* reshape1_output_type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(reshape1_output_info, &reshape1_output_type_info),
                             ort_api,
                             nullptr);
  const OrtTensorTypeAndShapeInfo* reshape1_output_tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(reshape1_output_type_info, &reshape1_output_tensor_info),
                             ort_api,
                             nullptr);

  size_t reshape1_output_dims_count = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(reshape1_output_tensor_info, &reshape1_output_dims_count),
                             ort_api,
                             nullptr);
  std::vector<int64_t> reshape1_output_dims(reshape1_output_dims_count);
  RETURN_DEFAULT_IF_API_FAIL(
      ort_api.GetDimensions(reshape1_output_tensor_info, reshape1_output_dims.data(), reshape1_output_dims_count),
      ort_api,
      nullptr);

  // Intermediate shape must split channels in groups only.
  // Normalize to handle ORT 1.29's unit-batch-dim collapse in post-Layout-Transform.
  bool was_collapsed = false;
  if (!NormalizeChannelShuffleIntermediateShape(reshape1_input_dims, reshape1_output_dims, was_collapsed)) {
    return nullptr;
  }
  reshape1_output_dims_count = reshape1_output_dims.size();

  if (reshape1_input_dims[0] != reshape1_output_dims[0]) {
    return nullptr;
  }
  if (reshape1_output_dims_count < 3) {
    return nullptr;
  }
  if (reshape1_input_dims[1] != (reshape1_output_dims[1] * reshape1_output_dims[2])) {
    return nullptr;
  }
  if (reshape1_output_dims_count != reshape1_input_dims_count + 1) {
    return nullptr;
  }
  size_t remaining_dims = reshape1_input_dims_count - 2;
  if (reshape1_output_dims_count < remaining_dims + 3) {
    return nullptr;
  }
  for (size_t i = 0; i < remaining_dims; ++i) {
    if (reshape1_input_dims[i + 2] != reshape1_output_dims[i + 3]) {
      return nullptr;
    }
  }

  // Intermediate transpose must only permute channels
  std::optional<std::vector<int64_t>> perm = GetTransposePerm(*transpose);
  if (!perm.has_value()) {
    return nullptr;
  }
  // If ORT collapsed the unit batch dim, the mid-transpose perm is also rank-reduced.
  // Lift it back to match the normalized reshape1 output rank by prepending 0 and shifting.
  std::vector<int64_t> perm_to_check = perm.value();
  if (was_collapsed && perm_to_check.size() + 1 == reshape1_output_dims_count) {
    std::vector<int64_t> lifted;
    lifted.reserve(reshape1_output_dims_count);
    lifted.push_back(0);
    for (int64_t p : perm_to_check) {
      lifted.push_back(p + 1);
    }
    perm_to_check = std::move(lifted);
  }
  std::swap(perm_to_check[1], perm_to_check[2]);
  std::vector<int64_t> perm_expected(perm_to_check.size());
  for (size_t i = 0; i < perm_expected.size(); ++i) {
    perm_expected[i] = static_cast<int64_t>(i);
  }
  if (perm_to_check != perm_expected) {
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

  const OrtTypeInfo* r1_in_ti = nullptr; const OrtTensorTypeAndShapeInfo* r1_in_tsi = nullptr;
  const OrtTypeInfo* t_tail_out_ti = nullptr; const OrtTensorTypeAndShapeInfo* t_tail_out_tsi = nullptr;
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
  return std::make_unique<ChannelShuffleFusion>(four_span, false /*no_head_transpose_tag*/);
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
