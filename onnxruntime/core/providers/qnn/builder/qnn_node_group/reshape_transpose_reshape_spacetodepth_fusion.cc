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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

std::optional<std::array<const OrtNodeUnit*, 3>> MatchPattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape1,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group) {
  // 1. Validate the starting Reshape node unit type.
  if (reshape1.OpType() != kOpReshape ||
      (reshape1.UnitType() != OrtNodeUnit::Type::SingleNode &&
       reshape1.UnitType() != OrtNodeUnit::Type::QDQGroup)) {
    return std::nullopt;
  }

  // 2. Find the Transpose child.
  const OrtNodeUnit* transpose = GetChildNodeUnitAllowQdq(qnn_model_wrapper, reshape1, kOpTranspose,
                                                          node_to_node_unit, node_unit_to_qnn_node_group);
  if (transpose == nullptr) {
    return std::nullopt;
  }

  // 3. Find the final Reshape child.
  const OrtNodeUnit* reshape2 = GetChildNodeUnitAllowQdq(qnn_model_wrapper, *transpose, kOpReshape,
                                                         node_to_node_unit, node_unit_to_qnn_node_group);
  if (reshape2 == nullptr) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 3>{&reshape1, transpose, reshape2};
}

bool ValidateAndComputeParams(
    const OrtNodeUnit& reshape1,
    const OrtNodeUnit& transpose,
    const OrtNodeUnit& reshape2,
    const QnnModelWrapper& qnn_model_wrapper,
  uint32_t& block_height,
  uint32_t& block_width,
    uint32_t& mode) {
  // 1. Validate the 4D input shape is static.
  std::vector<uint32_t> input_shape;
  if (!qnn_model_wrapper.GetOnnxShape(reshape1.Inputs()[0].shape, input_shape)) {
    return false;
  }
  if (input_shape.size() != kRank4) {
    return false;
  }
  for (uint32_t dim : input_shape) {
    // dynamic dimensions not supported.
    if (dim <= 0) {
      return false;
    }
  }

  // 2. Read constant reshape shapes.
  auto shape_6d = GetInitializerShape(qnn_model_wrapper, reshape1.Inputs()[1]);
  auto shape_4d = GetInitializerShape(qnn_model_wrapper, reshape2.Inputs()[1]);
  if (!shape_6d.has_value() || !shape_4d.has_value()) {
    return false;
  }
  if (shape_6d->size() != kRank6 || shape_4d->size() != kRank4) {
    return false;
  }

  // 3. Require positive reshape dims.
  for (int64_t v : *shape_6d) {
    // dynamic dimensions not supported.
    if (v <= 0) {
      return false;
    }
  }
  for (int64_t v : *shape_4d) {
    // dynamic dimensions not supported.
    if (v <= 0) {
      return false;
    }
  }

  // 4. Validate reshape1 target layout and block sizes.
  const int64_t n = static_cast<int64_t>(input_shape[0]);
  const int64_t c = static_cast<int64_t>(input_shape[1]);
  const int64_t h = static_cast<int64_t>(input_shape[2]);
  const int64_t w = static_cast<int64_t>(input_shape[3]);

  const int64_t r_n = (*shape_6d)[0];
  const int64_t r_c = (*shape_6d)[1];
  const int64_t h_div = (*shape_6d)[2];
  const int64_t b0 = (*shape_6d)[3];
  const int64_t w_div = (*shape_6d)[4];
  const int64_t b1 = (*shape_6d)[5];

  if (r_n != n || r_c != c) {
    return false;
  }

  // 5. Validate block sizes and divisibility.
  if (b0 < 1 || b1 < 1) {
    return false;
  }

  if (b0 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      b1 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  if (h % b0 != 0 || w % b1 != 0) {
    return false;
  }

  if (h_div != h / b0 || w_div != w / b1) {
    return false;
  }

  // 6. Validate expected output channel size.
  int64_t channel_multiplier = 0;
  try {
    channel_multiplier = SafeInt<int64_t>(b0) * SafeInt<int64_t>(b1);
  } catch (const SafeIntException&) {
    return false;
  }

  int64_t expected_c = 0;
  try {
    expected_c = SafeInt<int64_t>(c) * SafeInt<int64_t>(channel_multiplier);
  } catch (const SafeIntException&) {
    return false;
  }

  const std::array<int64_t, 4> expected_shape_4d = {n, expected_c, h / b0, w / b1};
  if (!std::equal(shape_4d->begin(), shape_4d->end(), expected_shape_4d.begin())) {
    return false;
  }

  // 7. Validate transpose permutation and resolve mode (DCR / CRD).
  OrtNodeAttrHelper transpose_attrs(transpose);
  std::vector<int64_t> perm = transpose_attrs.Get(kAttrTransposePerm, std::vector<int64_t>{});
  const std::array<int64_t, 6> perm_dcr = {0, 3, 5, 1, 2, 4};
  const std::array<int64_t, 6> perm_crd = {0, 1, 3, 5, 2, 4};

  if (perm.size() != kRank6) {
    return false;
  }

  if (std::equal(perm.begin(), perm.end(), perm_dcr.begin())) {
    mode = QNN_OP_SPACE_TO_DEPTH_MODE_DCR;
  } else if (std::equal(perm.begin(), perm.end(), perm_crd.begin())) {
    mode = QNN_OP_SPACE_TO_DEPTH_MODE_CRD;
  } else {
    return false;
  }

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
    bool validate) {
  // 1. Prepare tensor wrappers for SpaceToDepth node input/output.
  const OrtNodeUnit* reshape1 = node_units[0];
  const OrtNodeUnit* reshape2 = node_units[2];

  const OrtNodeUnitIODef& input_def = reshape1->Inputs()[0];
  const OrtNodeUnitIODef& output_def = reshape2->Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  // 2. Build SpaceToDepth parameters.
  const std::string node_name = utils::GetUniqueName(*reshape2, "_spacetodepth");

  // 2.1 block_height and block_width params.
  std::vector<uint32_t> block_shape{2};
  std::vector<uint32_t> block_data{block_height, block_width};
  QnnParamWrapper block_param(reshape2->Index(), reshape2->Name(),
                              QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE,
                              std::move(block_shape), std::move(block_data));

  // 2.2 mode param.
  Qnn_Scalar_t mode_scalar = QNN_SCALAR_INIT;
  mode_scalar.dataType = QNN_DATATYPE_UINT_32;
  mode_scalar.uint32Value = mode;
  QnnParamWrapper mode_param(reshape2->Index(), reshape2->Name(),
                             QNN_OP_SPACE_TO_DEPTH_PARAM_MODE, mode_scalar);

  // 3. Validate the SpaceToDepth QNN node.
  if (validate) {
    std::vector<Qnn_Param_t> params;
    params.push_back(block_param.GetQnnParam());
    params.push_back(mode_param.GetQnnParam());
    return qnn_model_wrapper.ValidateQnnNode(node_name,
                                             QNN_OP_PACKAGE_NAME_QTI_AISW,
                                             QNN_OP_SPACE_TO_DEPTH,
                                             {input_tensor.GetQnnTensor()},
                                             {output_tensor.GetQnnTensor()},
                                             std::move(params));
  }

  // 4.1 Add SpaceToDepth node Input tensors to the model wrapper.
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_def.name)) {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("Tensor already added, skip it: " + input_def.name).c_str());
  }

  // 4.2 Add SpaceToDepth node Output tensor to the model wrapper.
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(output_def.name)) {
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
  }

  // 4.3 Add the SpaceToDepth node BlockSize param to the model wrapper.
  std::vector<std::string> param_tensor_names;
  const std::string block_param_name = block_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(block_param)), "Failed to add blocksize param");
  param_tensor_names.push_back(block_param_name);

  // 4.4 Add the SpaceToDepth node Mode param to the model wrapper.
  const std::string mode_param_name = mode_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(mode_param)), "Failed to add mode param");
  param_tensor_names.push_back(mode_param_name);

  // 5. Create the SpaceToDepth node on QNN.
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SPACE_TO_DEPTH,
                                                {input_def.name}, {output_def.name},
                                                std::move(param_tensor_names), validate),
                "Failed to add fused SpaceToDepth node.");

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ReshapeTransposeReshapeSpaceToDepthFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& reshape_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  auto pattern = MatchPattern(qnn_model_wrapper, reshape_node_unit,
                              node_to_node_unit, node_unit_to_qnn_node_group);
  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* reshape1 = pattern->at(0);
  const OrtNodeUnit* transpose = pattern->at(1);
  const OrtNodeUnit* reshape2 = pattern->at(2);

  uint32_t block_height = 0;
  uint32_t block_width = 0;
  uint32_t mode = 0;
  if (!ValidateAndComputeParams(*reshape1, *transpose, *reshape2, qnn_model_wrapper,
                                block_height, block_width, mode)) {
    return nullptr;
  }

  if (!CreateOrValidateOnQnn(qnn_model_wrapper, pattern.value(), block_height, block_width,
                             mode, logger, true).IsOK()) {
    return nullptr;
  }

  return std::make_unique<ReshapeTransposeReshapeSpaceToDepthFusion>(pattern.value(),
                                                                     block_height, block_width, mode);
}

gsl::span<const OrtNodeUnit* const> ReshapeTransposeReshapeSpaceToDepthFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status ReshapeTransposeReshapeSpaceToDepthFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), block_height_, block_width_,
                               mode_, logger, true);
}

Ort::Status ReshapeTransposeReshapeSpaceToDepthFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), block_height_, block_width_,
                               mode_, logger, false);
}

}  // namespace qnn
}  // namespace onnxruntime
