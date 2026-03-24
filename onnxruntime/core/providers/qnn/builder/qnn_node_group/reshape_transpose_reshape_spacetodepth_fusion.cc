// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/reshape_transpose_reshape_spacetodepth_fusion.h"

#include <gsl/gsl>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kAttrTransposePerm[] = "perm";
constexpr char kOpReshape[] = "Reshape";
constexpr char kOpTranspose[] = "Transpose";
constexpr size_t kRank4 = 4;
constexpr size_t kRank6 = 6;

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

struct SpaceToDepthPattern {
  std::array<const OrtNodeUnit*, 5> node_units{};
  size_t node_count = 0;
  size_t reshape1_index = 0;
  size_t transpose_index = 1;
  size_t reshape2_index = 2;
};

struct PatternIndices {
  size_t reshape1_index = 0;
  size_t transpose_index = 1;
  size_t reshape2_index = 2;
  bool has_head_transpose = false;
  bool has_tail_transpose = false;
};

PatternIndices GetPatternIndices(gsl::span<const OrtNodeUnit* const> node_units) {
  PatternIndices indices;
  // T(NHWC->NCHW) + RTR + T(NCHW->NHWC) pattern has 5 nodes.
  // T(NHWC->NCHW) + RTR and RTR + T(NCHW->NHWC) patterns have 4 nodes.
  indices.has_head_transpose = node_units.size() >= 4 &&
                               node_units[0] != nullptr &&
                               node_units[0]->OpType() == kOpTranspose;
  indices.reshape1_index = indices.has_head_transpose ? 1 : 0;
  indices.transpose_index = indices.reshape1_index + 1;
  indices.reshape2_index = indices.transpose_index + 1;
  indices.has_tail_transpose = node_units.size() > (indices.reshape2_index + 1) &&
                               node_units[indices.reshape2_index + 1] != nullptr &&
                               node_units[indices.reshape2_index + 1]->OpType() == kOpTranspose;
  return indices;
}

std::optional<std::vector<int64_t>> GetTransposePerm(const OrtNodeUnit& transpose) {
  if (transpose.OpType() != kOpTranspose) {
    return std::nullopt;
  }

  OrtNodeAttrHelper attrs(transpose);
  return attrs.Get(kAttrTransposePerm, std::vector<int64_t>{});
}

bool IsNhwcNchwTransposePair(gsl::span<const int64_t> head_perm,
                             gsl::span<const int64_t> tail_perm) {
  if (head_perm.size() != tail_perm.size()) {
    return false;
  }

  // Inverse of head_perm should match tail_perm.
  std::vector<int64_t> head_inverted(head_perm.size(), -1);
  for (size_t i = 0; i < head_perm.size(); ++i) {
    const int64_t p = head_perm[i];
    if (p < 0 || p >= static_cast<int64_t>(head_perm.size())) {
      return false;
    }
    // inv[head_perm[i]] = i
    head_inverted[gsl::narrow_cast<size_t>(p)] = gsl::narrow_cast<int64_t>(i);
  }

  return std::equal(head_inverted.begin(), head_inverted.end(), tail_perm.begin());
}

const OrtNodeUnit& GetReshape2Node(gsl::span<const OrtNodeUnit* const> node_units) {
  const PatternIndices indices = GetPatternIndices(node_units);
  return *(node_units[indices.reshape2_index]);
}

// QNN SpaceToDepth needs NHWC layout, so for T(NHWC->NCHW) + RTR mark the input from the head transpose as start of pattern.
const OrtNodeUnitIODef& GetPatternInputDef(gsl::span<const OrtNodeUnit* const> node_units) {
  const PatternIndices indices = GetPatternIndices(node_units);
  const OrtNodeUnit* input_owner = indices.has_head_transpose ? node_units[0] : node_units[indices.reshape1_index];
  return input_owner->Inputs()[0];
}

// for RTR + T(NCHW->NHWC) mark the output of tail transpose as end of pattern, to keep NHWC at group boundaries.
const OrtNodeUnitIODef& GetPatternOutputDef(gsl::span<const OrtNodeUnit* const> node_units) {
  const PatternIndices indices = GetPatternIndices(node_units);
  const OrtNodeUnit* output_owner = indices.has_tail_transpose
                                        ? node_units[indices.reshape2_index + 1]
                                        : node_units[indices.reshape2_index];
  return output_owner->Outputs()[0];
}

bool IsNchwToNhwcPerm(gsl::span<const int64_t> perm) {
  static constexpr std::array<int64_t, 4> kPermNchwToNhwc = {0, 2, 3, 1};
  return perm.size() == kPermNchwToNhwc.size() &&
         std::equal(perm.begin(), perm.end(), kPermNchwToNhwc.begin());
}

bool IsNhwcToNchwPerm(gsl::span<const int64_t> perm) {
  static constexpr std::array<int64_t, 4> kPermNhwcToNchw = {0, 3, 1, 2};
  return perm.size() == kPermNhwcToNchw.size() &&
         std::equal(perm.begin(), perm.end(), kPermNhwcToNchw.begin());
}

// Fast structural gate to distinguish SpaceToDepth-like RTR from generic RTR.
bool HasSpaceToDepthCoreSignature(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1,
    const OrtNodeUnit& transpose,
    const OrtNodeUnit& reshape2) {
  // input shape needs to be 4D with positive dims (non dynamic).
  std::vector<uint32_t> input_shape;
  if (!qnn_model_wrapper.GetOnnxShape(reshape1.Inputs()[0].shape, input_shape) || input_shape.size() != kRank4) {
    return false;
  }

  for (uint32_t dim : input_shape) {
    if (dim <= 0) {
      return false;
    }
  }

  // check ranks of Reshape 'shape' to be 6 and 4.
  auto shape_6d = GetInitializerShape(qnn_model_wrapper, reshape1.Inputs()[1]);
  auto shape_4d = GetInitializerShape(qnn_model_wrapper, reshape2.Inputs()[1]);
  if (!shape_6d.has_value() || !shape_4d.has_value()) {
    return false;
  }

  if (shape_6d->size() != kRank6 || shape_4d->size() != kRank4) {
    return false;
  }

  // check Reshape 'shape' dimensions to be positive.
  for (int64_t v : *shape_6d) {
    if (v <= 0) {
      return false;
    }
  }

  for (int64_t v : *shape_4d) {
    if (v <= 0) {
      return false;
    }
  }

  // [N, C, H, W]
  const int64_t n = static_cast<int64_t>(input_shape[0]);
  const int64_t c = static_cast<int64_t>(input_shape[1]);
  const int64_t h = static_cast<int64_t>(input_shape[2]);
  const int64_t w = static_cast<int64_t>(input_shape[3]);

  // [N, C, H / b0, b0, W / b1, b1]
  const int64_t r_n = (*shape_6d)[0];
  const int64_t r_c = (*shape_6d)[1];
  const int64_t h_div = (*shape_6d)[2];
  const int64_t b0 = (*shape_6d)[3];
  const int64_t w_div = (*shape_6d)[4];
  const int64_t b1 = (*shape_6d)[5];

  // r_ are expected to match input [N,C]
  if (r_n != n || r_c != c || b0 < 1 || b1 < 1) {
    return false;
  }

  // b0,b1 are expected block sizes that divide H and W respectively.
  if (h % b0 != 0 || w % b1 != 0) {
    return false;
  }
  if (h_div != h / b0 || w_div != w / b1) {
    return false;
  }

  // [N, C * b0 * b1, H / b0, W / b1]
  const int64_t expected_c = c * b0 * b1;
  const std::array<int64_t, 4> expected_shape_4d = {n, expected_c, h / b0, w / b1};
  if (!std::equal(shape_4d->begin(), shape_4d->end(), expected_shape_4d.begin())) {
    return false;
  }

  // check transpose perm to be either {0,3,5,1,2,4} or {0,1,3,5,2,4}.
  OrtNodeAttrHelper transpose_attrs(transpose);
  std::vector<int64_t> perm = transpose_attrs.Get(kAttrTransposePerm, std::vector<int64_t>{});
  const std::array<int64_t, 6> perm_dcr = {0, 3, 5, 1, 2, 4};
  const std::array<int64_t, 6> perm_crd = {0, 1, 3, 5, 2, 4};

  return perm.size() == kRank6 &&
         (std::equal(perm.begin(), perm.end(), perm_dcr.begin()) ||
          std::equal(perm.begin(), perm.end(), perm_crd.begin()));
}

std::optional<SpaceToDepthPattern> MatchPattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  // 1. Validate the starting node op type.
  if (reshape1.OpType() != kOpReshape) {
    return std::nullopt;
  }

  // 2. Find the Transpose child of Reshape1.
  const OrtNodeUnit* transpose = GetChildNodeUnitAllowQdq(qnn_model_wrapper, reshape1, kOpTranspose,
                                                          node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose == nullptr) {
    return std::nullopt;
  }

  // 3. Find the final Reshape2 child of Transpose.
  const OrtNodeUnit* reshape2 = GetChildNodeUnitAllowQdq(qnn_model_wrapper, *transpose, kOpReshape,
                                                         node_to_node_unit, node_unit_to_qnn_node_group);
  if (reshape2 == nullptr) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                "SpaceToDepthFusion: no trailing Reshape2 child found after Transpose.");
    return std::nullopt;
  }

  // 3.1 Fast signature check for SpaceToDepth core RTR decomposition.
  if (!HasSpaceToDepthCoreSignature(qnn_model_wrapper, reshape1, *transpose, *reshape2)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "SpaceToDepthFusion: not a S2D RTR pattern.");
    return std::nullopt;
  }

  // 4. Find optional head and tail Transposes around the Reshape-Transpose-Reshape core pattern.
  SpaceToDepthPattern core;
  core.node_units = {&reshape1, transpose, reshape2, nullptr, nullptr};
  core.node_count = 3;
  core.reshape1_index = 0;
  core.transpose_index = 1;
  core.reshape2_index = 2;

  // 4.1 Check for optional head transpose before Reshape1.
  const OrtNodeUnit* transpose_head =
      GetParentOfInput(qnn_model_wrapper,
                       reshape1,
                       reshape1.Inputs()[0],
                       node_to_node_unit,
                       node_unit_to_qnn_node_group);
  if (transpose_head != nullptr && transpose_head->OpType() == kOpTranspose) {
    const OrtNodeUnit* reshape1_from_head = GetChildNodeUnitAllowQdq(qnn_model_wrapper,
                                                                     *transpose_head,
                                                                     kOpReshape,
                                                                     node_to_node_unit,
                                                                     node_unit_to_qnn_node_group);
    if (reshape1_from_head == &reshape1) {
      std::optional<std::vector<int64_t>> head_perm = GetTransposePerm(*transpose_head);
      if (head_perm.has_value()) {
        // 4.1.1 If head transpose is NHWC->NCHW, then mark it as head of SpaceToDepth pattern.
        const auto head_perm_span = gsl::make_span<const int64_t>(head_perm->data(), head_perm->size());
        if (IsNhwcToNchwPerm(head_perm_span)) {
          core.node_units = {transpose_head, &reshape1, transpose, reshape2, nullptr};
          core.node_count = 4;
          core.reshape1_index = 1;
          core.transpose_index = 2;
          core.reshape2_index = 3;
        }
      }
    }
  }

  // 4.2 Check for optional tail transpose after Reshape2.
  const OrtNodeUnit* transpose_tail = GetChildNodeUnitAllowQdq(qnn_model_wrapper,
                                                               *reshape2,
                                                               kOpTranspose,
                                                               node_to_node_unit,
                                                               node_unit_to_qnn_node_group);
  if (transpose_tail != nullptr && transpose_tail->OpType() == kOpTranspose) {
    std::optional<std::vector<int64_t>> tail_perm = GetTransposePerm(*transpose_tail);
    if (tail_perm.has_value()) {
      const auto tail_perm_span = gsl::make_span<const int64_t>(tail_perm->data(), tail_perm->size());
      if (IsNchwToNhwcPerm(tail_perm_span)) {
        const bool has_head = core.node_count == 4 && core.node_units[0] != nullptr &&
                              core.node_units[0]->OpType() == kOpTranspose;
        if (has_head) {
          // It is a 5-node pattern with both head and tail transposes.
          std::optional<std::vector<int64_t>> head_perm = GetTransposePerm(*core.node_units[0]);

          if (head_perm.has_value()) {
            auto head_perm_span = gsl::make_span<const int64_t>(head_perm->data(), head_perm->size());
            // 4.2.1 If head and tail transposes needs to be NHWC<->NCHW pair
            if (IsNhwcNchwTransposePair(head_perm_span, tail_perm_span)) {
              core.node_units = {core.node_units[0], &reshape1, transpose, reshape2, transpose_tail};
              core.node_count = 5;
              core.reshape1_index = 1;
              core.transpose_index = 2;
              core.reshape2_index = 3;
            }
          }
        } else {
          // It is a 4-node pattern with only tail transpose.
          core.node_units = {&reshape1, transpose, reshape2, transpose_tail, nullptr};
          core.node_count = 4;
          core.reshape1_index = 0;
          core.transpose_index = 1;
          core.reshape2_index = 2;
        }
      }
    }
  }

  // After (3) we know it is S2D RTR pattern but we Skip RTR-only pattern to avoid fusion in 1st get_capability call,
  // as it results in redundant cancelling Transpose operators added into QnnModelWrapper and gets into DLC.
  // Fusion needs to happen only after "Layout Transformer" pass, so the Supported forms are strictly:
  // a) T(NHWC->NCHW) + RTR + T(NCHW->NHWC)
  // b) T(NHWC->NCHW) + RTR
  // c) RTR + T(NCHW->NHWC)
  if (core.node_count == 3) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "SpaceToDepthFusion: skip RTR-only pattern.");
    return std::nullopt;
  }

  return core;
}

bool ValidateAndComputeParams(
    const OrtNodeUnit& reshape1,
    const OrtNodeUnit& transpose,
    const OrtNodeUnit& reshape2,
    const QnnModelWrapper& qnn_model_wrapper,
    uint32_t& block_height,
    uint32_t& block_width,
    uint32_t& mode,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(reshape2);

  // Core structural validation is already done in HasSpaceToDepthCoreSignature from MatchPattern.
  // Here we only extract params needed for QNN op creation and backend-specific guards.

  // 1. Read shape_6d to obtain block sizes.
  auto shape_6d = GetInitializerShape(qnn_model_wrapper, reshape1.Inputs()[1]);
  if (!shape_6d.has_value() || shape_6d->size() != kRank6) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "SpaceToDepthFusion: reshape1 shape initializer missing/invalid.");
    return false;
  }

  // [N, C, H / b0, b0, W / b1, b1]
  const int64_t b0 = (*shape_6d)[3];
  const int64_t b1 = (*shape_6d)[5];

  if (b0 < 1 || b1 < 1) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "SpaceToDepthFusion: invalid block size.");
    return false;
  }

  // 2. explicit uint32 range checks for final QNN param conversion.
  if (b0 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      b1 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "SpaceToDepthFusion: block size overflows uint32.");
    return false;
  }

  // 3. Validate transpose permutation and resolve mode (DCR / CRD).
  OrtNodeAttrHelper transpose_attrs(transpose);
  std::vector<int64_t> perm = transpose_attrs.Get(kAttrTransposePerm, std::vector<int64_t>{});
  const std::array<int64_t, 6> perm_dcr = {0, 3, 5, 1, 2, 4};
  const std::array<int64_t, 6> perm_crd = {0, 1, 3, 5, 2, 4};

  if (perm.size() != kRank6) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "SpaceToDepthFusion: perm rank is not 6.");
    return false;
  }

  if (std::equal(perm.begin(), perm.end(), perm_dcr.begin())) {
    mode = QNN_OP_SPACE_TO_DEPTH_MODE_DCR;
  } else if (std::equal(perm.begin(), perm.end(), perm_crd.begin())) {
    mode = QNN_OP_SPACE_TO_DEPTH_MODE_CRD;
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "SpaceToDepthFusion: perm is not DCR/CRD.");
    return false;
  }

  /*
   * TODO(AISW-175353): Remove these backend-specific fusion guards once the
   * SpaceToDepth kernel limitations are fixed.
   * Tracking issue: https://jira-dc.qualcomm.com/jira/browse/AISW-175353
   */
  // 4. Backend-specific constraints for known kernel limitations.
  const QnnBackendType backend_type = qnn_model_wrapper.GetQnnBackendType();

  if (IsCpuBackend(backend_type) && b0 != b1) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                "SpaceToDepthFusion: skip fusion on CPU for unequal block sizes.");
    return false;
  }
  // ============ Backend-specific constraints end =============.

  block_height = static_cast<uint32_t>(b0);
  block_width = static_cast<uint32_t>(b1);
  return true;
}

Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    uint32_t block_height,
    uint32_t block_width,
    uint32_t mode,
    const Ort::Logger& logger,
    bool validate,
    std::optional<bool> use_nhwc_fallback,
    std::optional<bool>* use_nhwc_fallback_out) {
  const PatternIndices pattern_indices = GetPatternIndices(node_units);
  // RTR + T(NCHW->NHWC) ==> NHWC->NCHW + S2D
  const bool need_pre_transpose = !pattern_indices.has_head_transpose;
  // NHWC->NCHW + RTR ==> S2D + NHWC->NCHW
  const bool need_post_transpose = !pattern_indices.has_tail_transpose;

  // 1) Common setup: resolve pattern boundary IO tensors.
  const OrtNodeUnit& reshape2 = GetReshape2Node(node_units);

  const OrtNodeUnitIODef& input_def = GetPatternInputDef(node_units);
  const OrtNodeUnitIODef& output_def = GetPatternOutputDef(node_units);

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  // 2) Common setup: build SpaceToDepth node name and params.
  const std::string node_name = utils::GetUniqueName(reshape2, "_fused_spacetodepth");

  const uint32_t block_size_h = block_height;
  const uint32_t block_size_w = block_width;

  // 2.1) Mode param.
  Qnn_Scalar_t mode_scalar = QNN_SCALAR_INIT;
  mode_scalar.dataType = QNN_DATATYPE_UINT_32;
  mode_scalar.uint32Value = mode;
  QnnParamWrapper mode_param(reshape2.Index(), reshape2.Name(),
                             QNN_OP_SPACE_TO_DEPTH_PARAM_MODE, mode_scalar);

  // ============ Helper: Validation of SpaceToDepth node with given block param.
  auto validate_with_block_param = [&](QnnParamWrapper& block_param,
                                       const Qnn_Tensor_t& input,
                                       const Qnn_Tensor_t& output) -> Ort::Status {
    std::vector<Qnn_Param_t> params;
    params.push_back(block_param.GetQnnParam());
    params.push_back(mode_param.GetQnnParam());
    return qnn_model_wrapper.ValidateQnnNode(node_name,
                                             QNN_OP_PACKAGE_NAME_QTI_AISW,
                                             QNN_OP_SPACE_TO_DEPTH,
                                             {input},
                                             {output},
                                             std::move(params));
  };

  // ============ Helper: Create direct SpaceToDepth node (no extra pre/post transpose nodes).
  auto create_with_block_param = [&](QnnParamWrapper& block_param) -> Ort::Status {
    // A) Add input tensor.
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_def.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add s2d input");
    } else {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                  ("Tensor already added, skip it: " + input_def.name).c_str());
    }

    // B) Add output tensor.
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(output_def.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add s2d output");
    }

    // C) Add block-size and mode params.
    std::vector<std::string> param_tensor_names;
    const std::string block_param_name = block_param.GetParamTensorName();
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(block_param)), "Failed to add s2d blocksize param");
    param_tensor_names.push_back(block_param_name);

    const std::string mode_param_name = mode_param.GetParamTensorName();
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(mode_param)), "Failed to add s2d mode param");
    param_tensor_names.push_back(mode_param_name);

    // D) Create SpaceToDepth node.
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SPACE_TO_DEPTH,
                                                  {input_def.name}, {output_def.name},
                                                  std::move(param_tensor_names), validate),
                  "Failed to add fused SpaceToDepth node.");

    return Ort::Status();
  };

  auto build_tensor_block_param = [&]() -> QnnParamWrapper {
    std::vector<uint32_t> block_shape{2};
    std::vector<uint32_t> block_data{block_size_h, block_size_w};
    return QnnParamWrapper(reshape2.Index(), reshape2.Name(),
                           QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE,
                           std::move(block_shape), std::move(block_data));
  };

  // 3) ============= Validation path =============
  if (validate) {
    auto block_param = build_tensor_block_param();

    // 3.1) Fully wrapped pattern (NHWC->NCHW + RTR + NCHW->NHWC = S2D).
    if (!need_pre_transpose && !need_post_transpose) {
      Ort::Status status = validate_with_block_param(block_param,
                                                     input_tensor.GetQnnTensor(),
                                                     output_tensor.GetQnnTensor());
      if (use_nhwc_fallback_out) {
        *use_nhwc_fallback_out = false;
      }
      return status;
    }

    // 3.2) Wrapped-NHWC validation: gather IO shape + tensor metadata.
    // Two cases: a) T(NHWC->NCHW) + RTR ==> S2D + T(NHWC->NCHW), so input is from head transpose
    //               and output is Reshape2 + Post Transpose.
    //            b) RTR + T(NHWC->NCHW) ==> T(NCHW->NHWC) + S2D, so input is from Pre Transpose + Reshape1
    //               and output is from tail transpose.
    std::vector<uint32_t> input_shape;
    std::vector<uint32_t> output_shape;
    if (!qnn_model_wrapper.GetOnnxShape(input_def.shape, input_shape) ||
        !qnn_model_wrapper.GetOnnxShape(output_def.shape, output_shape) ||
        input_shape.size() != kRank4 || output_shape.size() != kRank4) {
      return MAKE_EP_FAIL("Failed to get rank-4 input/output shapes for NHWC SpaceToDepth.");
    }

    TensorInfo input_info = {};
    TensorInfo output_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_info));
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));

    const std::string base_name = utils::GetUniqueName(reshape2, "_spacetodepth_nhwc");
    const std::string pre_node_name = base_name + "_pre";
    const std::string post_node_name = base_name + "_post";
    const std::string nhwc_in_name = base_name + "_in";
    const std::string nhwc_out_name = base_name + "_out";

    std::optional<QnnTensorWrapper> nhwc_input_tensor;
    std::optional<QnnTensorWrapper> nhwc_output_tensor;

    // 3.3) If needed, validate pre-transpose (NCHW -> NHWC).
    if (need_pre_transpose) {
      std::vector<uint32_t> nhwc_input_shape = {input_shape[0], input_shape[2], input_shape[3], input_shape[1]};
      nhwc_input_tensor.emplace(nhwc_in_name,
                                QNN_TENSOR_TYPE_NATIVE,
                                input_info.qnn_data_type,
                                input_info.quant_param.Copy(),
                                std::move(nhwc_input_shape));

      std::vector<uint32_t> pre_perm = {0, 2, 3, 1};
      QnnParamWrapper pre_perm_param(reshape2.Index(), pre_node_name,
                                     QNN_OP_TRANSPOSE_PARAM_PERM,
                                     {static_cast<uint32_t>(pre_perm.size())},
                                     std::move(pre_perm));

      Ort::Status pre_status = qnn_model_wrapper.ValidateQnnNode(pre_node_name,
                                                                 QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                 QNN_OP_TRANSPOSE,
                                                                 {input_tensor.GetQnnTensor()},
                                                                 {nhwc_input_tensor->GetQnnTensor()},
                                                                 {pre_perm_param.GetQnnParam()});
      if (!pre_status.IsOK()) {
        return pre_status;
      }
    }

    // 3.4) If needed, prepare NHWC output tensor for post-transpose.
    if (need_post_transpose) {
      std::vector<uint32_t> nhwc_output_shape = {output_shape[0], output_shape[2], output_shape[3], output_shape[1]};
      nhwc_output_tensor.emplace(nhwc_out_name,
                                 QNN_TENSOR_TYPE_NATIVE,
                                 output_info.qnn_data_type,
                                 output_info.quant_param.Copy(),
                                 std::move(nhwc_output_shape));
    }

    // 3.5) Validate SpaceToDepth core with wrapped NHWC IO.
    const Qnn_Tensor_t& s2d_input = need_pre_transpose ? nhwc_input_tensor->GetQnnTensor()
                                                       : input_tensor.GetQnnTensor();
    const Qnn_Tensor_t& s2d_output = need_post_transpose ? nhwc_output_tensor->GetQnnTensor()
                                                         : output_tensor.GetQnnTensor();

    Ort::Status nhwc_status = validate_with_block_param(block_param, s2d_input, s2d_output);
    if (!nhwc_status.IsOK()) {
      return nhwc_status;
    }

    // 3.6) If needed, validate post-transpose (NHWC -> NCHW).
    if (need_post_transpose) {
      std::vector<uint32_t> post_perm = {0, 3, 1, 2};
      QnnParamWrapper post_perm_param(reshape2.Index(), post_node_name,
                                      QNN_OP_TRANSPOSE_PARAM_PERM,
                                      {static_cast<uint32_t>(post_perm.size())},
                                      std::move(post_perm));
      Ort::Status post_status = qnn_model_wrapper.ValidateQnnNode(post_node_name,
                                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                                  QNN_OP_TRANSPOSE,
                                                                  {nhwc_output_tensor->GetQnnTensor()},
                                                                  {output_tensor.GetQnnTensor()},
                                                                  {post_perm_param.GetQnnParam()});
      if (!post_status.IsOK()) {
        return post_status;
      }
    }

    if (use_nhwc_fallback_out) {
      *use_nhwc_fallback_out = true;
    }
    return nhwc_status;
  }

  if (!use_nhwc_fallback.has_value()) {
    use_nhwc_fallback = false;
  }

  // 4) ============= Creation path: build QNN nodes =============
  const bool use_wrapped_nhwc = need_pre_transpose || need_post_transpose || use_nhwc_fallback.value();

  if (!use_wrapped_nhwc) {
    auto block_param = build_tensor_block_param();
    return create_with_block_param(block_param);
  }

  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_def.name)) {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
  }
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(output_def.name)) {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
  }

  // 4.1) Wrapped-NHWC creation: gather tensor metadata and ONNX shapes.
  TensorInfo input_info = {};
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def, input_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_def, output_info));

  std::vector<uint32_t> input_shape;
  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_def.shape, input_shape), "Failed to get input shape.");
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output_def.shape, output_shape), "Failed to get output shape.");

  const std::string base_name = utils::GetUniqueName(reshape2, "_spacetodepth_nhwc");
  const std::string pre_node_name = base_name + "_pre";
  const std::string post_node_name = base_name + "_post";
  const std::string nhwc_in_name = base_name + "_in";
  const std::string nhwc_out_name = base_name + "_out";

  std::string s2d_input_name = input_def.name;
  std::string s2d_output_name = output_def.name;

  // 4.2) If needed, create pre-transpose and route S2D input through NHWC tensor.
  if (need_pre_transpose) {
    std::vector<uint32_t> nhwc_input_shape = {input_shape[0], input_shape[2], input_shape[3], input_shape[1]};
    QnnTensorWrapper nhwc_input_tensor(nhwc_in_name,
                                       QNN_TENSOR_TYPE_NATIVE,
                                       input_info.qnn_data_type,
                                       input_info.quant_param.Copy(),
                                       std::move(nhwc_input_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(nhwc_input_tensor)),
                  "Failed to add NHWC input tensor.");

    std::vector<uint32_t> pre_perm = {0, 2, 3, 1};
    QnnParamWrapper pre_perm_param(reshape2.Index(), pre_node_name,
                                   QNN_OP_TRANSPOSE_PARAM_PERM,
                                   {static_cast<uint32_t>(pre_perm.size())},
                                   std::move(pre_perm));
    const std::string pre_perm_name = pre_perm_param.GetParamTensorName();
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(pre_perm_param)),
                  "Failed to add pre-transpose perm param.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(pre_node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_TRANSPOSE,
                                                  {input_def.name}, {nhwc_in_name},
                                                  {pre_perm_name}, validate),
                  "Failed to add pre-transpose node.");
    s2d_input_name = nhwc_in_name;
  }

  // 4.3) If needed, create NHWC output tensor for post-transpose.
  if (need_post_transpose) {
    std::vector<uint32_t> nhwc_output_shape = {output_shape[0], output_shape[2], output_shape[3], output_shape[1]};
    QnnTensorWrapper nhwc_output_tensor(nhwc_out_name,
                                        QNN_TENSOR_TYPE_NATIVE,
                                        output_info.qnn_data_type,
                                        output_info.quant_param.Copy(),
                                        std::move(nhwc_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(nhwc_output_tensor)),
                  "Failed to add NHWC output tensor.");
    s2d_output_name = nhwc_out_name;
  }

  // 4.4) Create wrapped SpaceToDepth core node.
  std::vector<std::string> s2d_params;
  auto block_param = build_tensor_block_param();
  const std::string block_name = block_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(block_param)),
                "Failed to add blocksize param.");
  s2d_params.push_back(block_name);

  const std::string mode_param_name = mode_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(mode_param)), "Failed to add mode param");
  s2d_params.push_back(mode_param_name);

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(base_name + "_s2d",
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_SPACE_TO_DEPTH,
                                                {s2d_input_name}, {s2d_output_name},
                                                std::move(s2d_params), validate),
                "Failed to add NHWC SpaceToDepth node.");

  // 4.5) If needed, create post-transpose to restore NCHW boundary.
  if (need_post_transpose) {
    std::vector<uint32_t> post_perm = {0, 3, 1, 2};
    QnnParamWrapper post_perm_param(reshape2.Index(), post_node_name,
                                    QNN_OP_TRANSPOSE_PARAM_PERM,
                                    {static_cast<uint32_t>(post_perm.size())},
                                    std::move(post_perm));
    const std::string post_perm_name = post_perm_param.GetParamTensorName();
    RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(post_perm_param)),
                  "Failed to add post-transpose perm param.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(post_node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_TRANSPOSE,
                                                  {nhwc_out_name}, {output_def.name},
                                                  {post_perm_name}, validate),
                  "Failed to add post-transpose node.");
  }

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ReshapeTransposeReshapeSpaceToDepthFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtNode& start_node = reshape_node_unit.GetNode();

  // 1. Skip fusion if the pattern Start Node is graph output.
  size_t num_outputs = 0;
  if (ort_api.Node_GetNumOutputs(&start_node, &num_outputs) == nullptr && num_outputs > 0) {
    std::vector<const OrtValueInfo*> outputs(num_outputs);
    if (ort_api.Node_GetOutputs(&start_node, outputs.data(), outputs.size()) == nullptr) {
      bool is_graph_output = false;
      if (ort_api.ValueInfo_IsGraphOutput(outputs[0], &is_graph_output) == nullptr && is_graph_output) {
        return nullptr;
      }
    }
  }

  // 2. Match Pattern
  auto pattern = MatchPattern(qnn_model_wrapper, reshape_node_unit,
                              node_to_node_unit, node_unit_to_qnn_node_group, logger);
  if (!pattern.has_value()) {
    return nullptr;
  }

  // 3. Get pattern node units.
  const OrtNodeUnit* reshape1 = pattern->node_units[pattern->reshape1_index];
  const OrtNodeUnit* transpose = pattern->node_units[pattern->transpose_index];
  const OrtNodeUnit* reshape2 = pattern->node_units[pattern->reshape2_index];

  // 4. Compute block h,w and mode params.
  uint32_t block_height = 0;
  uint32_t block_width = 0;
  uint32_t mode = 0;
  if (!ValidateAndComputeParams(*reshape1, *transpose, *reshape2, qnn_model_wrapper,
                                block_height, block_width, mode, logger)) {
    return nullptr;
  }

  // 5. Validate on QNN.
  std::optional<bool> use_nhwc_fallback;
  gsl::span<const OrtNodeUnit* const> pattern_span(pattern->node_units.data(), pattern->node_count);
  Ort::Status validate_status = CreateOrValidateOnQnn(qnn_model_wrapper, pattern_span,
                                                      block_height, block_width, mode,
                                                      logger, true, std::nullopt, &use_nhwc_fallback);
  if (!validate_status.IsOK()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("SpaceToDepthFusion: QNN validation failed: " + validate_status.GetErrorMessage()).c_str());
    return nullptr;
  }

  // 6. Create and return the fusion group.
  return std::make_unique<ReshapeTransposeReshapeSpaceToDepthFusion>(pattern_span,
                                                                     block_height, block_width, mode,
                                                                     use_nhwc_fallback.value_or(false));
}

gsl::span<const OrtNodeUnit* const> ReshapeTransposeReshapeSpaceToDepthFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status ReshapeTransposeReshapeSpaceToDepthFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), block_height_, block_width_, mode_, logger, true,
                               std::nullopt, nullptr);
}

Ort::Status ReshapeTransposeReshapeSpaceToDepthFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), block_height_, block_width_, mode_, logger, false,
                               use_nhwc_fallback_, nullptr);
}

}  // namespace qnn
}  // namespace onnxruntime
