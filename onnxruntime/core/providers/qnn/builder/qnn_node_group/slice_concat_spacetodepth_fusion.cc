// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/slice_concat_spacetodepth_fusion.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr size_t kNchwRank = 4;
constexpr int64_t kChannelAxis = 1;
constexpr int64_t kHeightAxis = 2;
constexpr int64_t kWidthAxis = 3;
constexpr int64_t kSliceStepTwo = 2;

using NodeToUnitMap = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using UnitToGroupMap = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

struct SlicePhase {
  int64_t height_offset = -1;
  int64_t width_offset = -1;
};

// Canonical ONNX SpaceToDepth (DCR) block order.
constexpr std::array<SlicePhase, 4> kCanonicalDcrPhases = {
    SlicePhase{0, 0}, SlicePhase{0, 1}, SlicePhase{1, 0}, SlicePhase{1, 1}};

[[nodiscard]] bool IsSliceUnit(const OrtNodeUnit* node_unit) {
  return node_unit != nullptr && node_unit->OpType() == "Slice";
}

[[nodiscard]] bool IsConcatUnit(const OrtNodeUnit* node_unit) {
  return node_unit != nullptr && node_unit->OpType() == "Concat";
}

// Slice starts/ends/axes/steps are inputs of the Slice target node itself. For
// QDQ-wrapped Slices, unit Inputs() are the outer (quantized) tensors, so the
// parameter names must come from the target node.
[[nodiscard]] std::vector<std::string> GetSliceTargetInputNames(const OrtNodeUnit& slice_unit) {
  std::vector<std::string> names;
  for (const Ort::ConstValueInfo& input_info : Ort::ConstNode(&slice_unit.GetNode()).GetInputs()) {
    names.emplace_back(input_info.GetName());
  }
  return names;
}

// Slice indices use int64 or int32 depending on exporter; accept both.
[[nodiscard]] std::optional<std::vector<int64_t>> ReadSliceIndexInitializer(
    const QnnModelWrapper& model_wrapper, const std::vector<std::string>& target_inputs, size_t input_index) {
  if (input_index >= target_inputs.size()) {
    return std::nullopt;
  }
  const std::string& initializer_name = target_inputs[input_index];
  if (initializer_name.empty() || !model_wrapper.IsConstantInput(initializer_name)) {
    return std::nullopt;
  }
  const OrtValueInfo* constant = model_wrapper.GetConstantTensor(initializer_name);
  if (constant == nullptr) {
    return std::nullopt;
  }
  Ort::ConstValueInfo constant_info(constant);
  Ort::ConstValue constant_value;
  if (!constant_info.GetInitializer(constant_value).IsOK()) {
    return std::nullopt;
  }
  const auto tensor_info = constant_info.TypeInfo().GetTensorTypeAndShapeInfo();
  const size_t element_count = tensor_info.GetElementCount();
  if (element_count == 0 || element_count > 4) {
    return std::nullopt;
  }
  const auto element_type = tensor_info.GetElementType();
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* data = constant_value.GetTensorData<int64_t>();
    if (data == nullptr) {
      return std::nullopt;
    }
    return std::vector<int64_t>(data, data + element_count);
  }
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* data = constant_value.GetTensorData<int32_t>();
    if (data == nullptr) {
      return std::nullopt;
    }
    return std::vector<int64_t>(data, data + element_count);
  }
  return std::nullopt;
}

struct SliceTileSpec {
  // Start offset along an axis cut with step 2; -1 when the axis is untouched.
  int64_t height_start = -1;
  int64_t width_start = -1;
};

// Parse one Slice into the (H, W) tile offsets it selects. Accepts single-axis
// slices (cascaded tiling) and multi-axis slices (parallel tiling);
// starts/ends/axes/steps may be int32 or int64. Exact coverage is proven by
// output shapes at the call site — ends may exceed the dim (ONNX clamps), so
// ends are only sanity-read here, never trusted for coverage.
[[nodiscard]] std::optional<SliceTileSpec> ParseSliceTile(
    const QnnModelWrapper& model_wrapper, const OrtNodeUnit& slice_unit) {
  if (!IsSliceUnit(&slice_unit) || slice_unit.SinceVersion() < 10 || slice_unit.Inputs().empty()) {
    return std::nullopt;
  }
  // Parameter initializers hang off the target node even when the Slice is QDQ-wrapped.
  const std::vector<std::string> target_inputs = GetSliceTargetInputNames(slice_unit);
  if (target_inputs.size() <= 3) {
    return std::nullopt;
  }
  std::vector<uint32_t> data_shape;
  if (!QnnModelWrapper::GetOnnxShape(slice_unit.Inputs()[0].shape, data_shape) ||
      data_shape.size() != kNchwRank) {
    return std::nullopt;
  }
  const auto starts = ReadSliceIndexInitializer(model_wrapper, target_inputs, 1);
  const auto ends = ReadSliceIndexInitializer(model_wrapper, target_inputs, 2);
  const auto axes = ReadSliceIndexInitializer(model_wrapper, target_inputs, 3);
  if (!starts.has_value() || !ends.has_value() || !axes.has_value() ||
      starts->size() != axes->size() || ends->size() != axes->size()) {
    return std::nullopt;
  }
  std::vector<int64_t> steps(axes->size(), 1);
  if (target_inputs.size() > 4) {
    const auto steps_init = ReadSliceIndexInitializer(model_wrapper, target_inputs, 4);
    if (!steps_init.has_value() || steps_init->size() != axes->size()) {
      return std::nullopt;
    }
    steps = std::move(*steps_init);
  }
  SliceTileSpec spec;
  std::array<bool, kNchwRank> axis_seen{false, false, false, false};
  for (size_t i = 0; i < axes->size(); ++i) {
    int64_t axis = (*axes)[i] < 0 ? (*axes)[i] + static_cast<int64_t>(kNchwRank) : (*axes)[i];
    if (axis < 0 || axis >= static_cast<int64_t>(kNchwRank) || axis_seen[static_cast<size_t>(axis)]) {
      return std::nullopt;
    }
    axis_seen[static_cast<size_t>(axis)] = true;
    if (steps[i] != 1 && steps[i] != kSliceStepTwo) {
      return std::nullopt;
    }
    if (steps[i] != kSliceStepTwo) {
      continue;
    }
    if (axis != kHeightAxis && axis != kWidthAxis) {
      return std::nullopt;
    }
    const int64_t dim = static_cast<int64_t>(data_shape[static_cast<size_t>(axis)]);
    const int64_t start = (*starts)[i] < 0 ? (*starts)[i] + dim : (*starts)[i];
    if (start != 0 && start != 1) {
      return std::nullopt;
    }
    if (axis == kHeightAxis) {
      spec.height_start = start;
    } else {
      spec.width_start = start;
    }
  }
  return spec;
}

[[nodiscard]] bool HasExpectedShape(const std::vector<uint32_t>& actual,
                                    uint32_t batch, uint32_t channels, uint32_t height, uint32_t width) {
  return actual.size() == kNchwRank && actual[0] == batch && actual[1] == channels &&
         actual[2] == height && actual[3] == width;
}

// QDQGroups merge DQ/Q into the Slice/Concat unit, so the Concat parent is the Slice
// group directly. A lone DQ/Q singleton between them carries an unchecked scale.
[[nodiscard]] const OrtNodeUnit* GetParentSliceOrNull(
    const QnnModelWrapper& model_wrapper, const OrtNodeUnit& child_unit,
    const OrtNodeUnitIODef& child_input, const NodeToUnitMap& node_to_unit,
    const UnitToGroupMap& unit_to_group) {
  const OrtNodeUnit* parent =
      GetParentOfInput(model_wrapper, child_unit, child_input, node_to_unit, unit_to_group);
  return IsSliceUnit(parent) ? parent : nullptr;
}

// Each fused intermediate must have no consumers outside the group; otherwise lowering
// to S2D would silently drop that edge's tensor. Consumers resolve through NodeUnits:
// in QDQ graphs the direct edge target is a DQ node owned by the consumer's group.
[[nodiscard]] bool HasExactlyUnitConsumers(const OrtNodeUnit& producer_unit,
                                           const std::vector<const OrtNodeUnit*>& expected_consumer_units,
                                           const NodeToUnitMap& node_to_unit) {
  // Group output leaves via the Q node for QDQ groups, via the target otherwise.
  const OrtNode* output_producer = &producer_unit.GetNode();
  if (producer_unit.UnitType() == OrtNodeUnit::Type::QDQGroup && !producer_unit.GetQNodes().empty()) {
    output_producer = producer_unit.GetQNodes()[0];
  }
  const Ort::ConstNode producer_node(output_producer);
  const std::vector<Ort::ConstValueInfo> producer_outputs = producer_node.GetOutputs();
  if (producer_outputs.size() != 1 || producer_outputs[0].IsGraphOutput()) {
    return false;
  }
  const std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = producer_outputs[0].GetConsumers();
  if (consumers.size() != expected_consumer_units.size()) {
    return false;
  }
  for (const OrtNodeUnit* expected_unit : expected_consumer_units) {
    const bool found = std::any_of(consumers.begin(), consumers.end(),
                                   [&node_to_unit, expected_unit](
                                       const Ort::ValueInfoConsumerProducerInfo& consumer) {
                                     if (consumer.node == nullptr) {
                                       return false;
                                     }
                                     const auto it = node_to_unit.find(consumer.node);
                                     return it != node_to_unit.end() && it->second == expected_unit;
                                   });
    if (!found) {
      return false;
    }
  }
  return true;
}

struct ConcatInputWalk {
  SlicePhase phase;
  // Slice units on the path from the shared root tensor to this Concat input,
  // root-most last. One entry for parallel tiling, two for cascaded tiling.
  std::vector<const OrtNodeUnit*> path_units;
  std::string root_tensor_name;
};

// Resolve the (h, w) phase of one Concat input by walking up through 1-2 Slice
// units to the shared root tensor. Parallel tiling (one multi-axis Slice) and
// cascaded tiling (two single-axis Slices) compose to the same phase.
[[nodiscard]] std::optional<ConcatInputWalk> ResolveConcatInputPhase(
    const QnnModelWrapper& model_wrapper, const OrtNodeUnit& concat_unit, size_t input_index,
    const NodeToUnitMap& node_to_unit, const UnitToGroupMap& unit_to_group) {
  const OrtNodeUnit* slice = GetParentSliceOrNull(model_wrapper, concat_unit,
                                                  concat_unit.Inputs()[input_index],
                                                  node_to_unit, unit_to_group);
  if (slice == nullptr) {
    return std::nullopt;
  }
  const auto spec = ParseSliceTile(model_wrapper, *slice);
  if (!spec.has_value()) {
    return std::nullopt;
  }
  const bool cuts_h = spec->height_start >= 0;
  const bool cuts_w = spec->width_start >= 0;
  if (!cuts_h && !cuts_w) {
    return std::nullopt;  // Full passthrough, not a tile.
  }
  ConcatInputWalk walk;
  walk.path_units.push_back(slice);
  int64_t height_offset = spec->height_start;
  int64_t width_offset = spec->width_start;
  const OrtNodeUnit* top_slice = slice;
  if ((cuts_h && !cuts_w) || (!cuts_h && cuts_w)) {
    // Cascaded tiling: the complementary axis comes from a parent Slice.
    const OrtNodeUnit* parent =
        GetParentSliceOrNull(model_wrapper, *slice, slice->Inputs()[0], node_to_unit, unit_to_group);
    if (parent == nullptr) {
      return std::nullopt;
    }
    const auto parent_spec = ParseSliceTile(model_wrapper, *parent);
    if (!parent_spec.has_value()) {
      return std::nullopt;
    }
    const bool parent_cuts_h = parent_spec->height_start >= 0;
    const bool parent_cuts_w = parent_spec->width_start >= 0;
    if (cuts_h) {
      if (parent_cuts_h || !parent_cuts_w) {
        return std::nullopt;
      }
      width_offset = parent_spec->width_start;
    } else {
      if (parent_cuts_w || !parent_cuts_h) {
        return std::nullopt;
      }
      height_offset = parent_spec->height_start;
    }
    walk.path_units.push_back(parent);
    top_slice = parent;
  }
  if (top_slice->Inputs().empty()) {
    return std::nullopt;
  }
  walk.phase = SlicePhase{height_offset, width_offset};
  walk.root_tensor_name = top_slice->Inputs()[0].name;
  return walk;
}

[[nodiscard]] bool TryGetDcrPhasePermutation(const std::array<SlicePhase, 4>& actual_phases,
                                             std::array<int64_t, 4>& permutation) {
  std::array<bool, 4> used{false, false, false, false};
  for (size_t i = 0; i < actual_phases.size(); ++i) {
    bool matched = false;
    for (size_t j = 0; j < kCanonicalDcrPhases.size(); ++j) {
      if (!used[j] && actual_phases[i].height_offset == kCanonicalDcrPhases[j].height_offset &&
          actual_phases[i].width_offset == kCanonicalDcrPhases[j].width_offset) {
        permutation[i] = static_cast<int64_t>(j);
        used[j] = true;
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

struct PerTensorQuant {
  bool quantized = false;
  float scale = 1.0f;
  int32_t offset = 0;
  uint32_t qnn_dtype = 0;
};

[[nodiscard]] std::optional<PerTensorQuant> GetBoundaryQuant(
    const QnnModelWrapper& model_wrapper, const OrtNodeUnitIODef& tensor_def) {
  TensorInfo tensor_info = {};
  if (!model_wrapper.GetTensorInfo(tensor_def, tensor_info).IsOK()) {
    return std::nullopt;
  }
  PerTensorQuant quant;
  quant.qnn_dtype = tensor_info.qnn_data_type;
  quant.quantized = tensor_info.quant_param.IsQuantized();
  if (quant.quantized) {
    if (!tensor_info.quant_param.IsPerTensor(/*include_bw*/ true)) {
      return std::nullopt;
    }
    if (!tensor_info.quant_param.GetPerTensorScaleOffset(quant.scale, quant.offset).IsOK()) {
      return std::nullopt;
    }
  }
  return quant;
}

[[nodiscard]] bool HasSameQuant(const PerTensorQuant& lhs, const PerTensorQuant& rhs) {
  if (lhs.quantized != rhs.quantized || lhs.qnn_dtype != rhs.qnn_dtype) {
    return false;
  }
  return !lhs.quantized || (lhs.scale == rhs.scale && lhs.offset == rhs.offset);
}

// Channel indices restoring Concat block order from canonical DCR output.
[[nodiscard]] std::vector<int32_t> BuildDcrGatherIndices(const std::array<int64_t, 4>& dcr_permutation,
                                                         uint32_t channel_count) {
  std::vector<int32_t> gather_indices;
  gather_indices.reserve(static_cast<size_t>(4 * channel_count));
  for (const int64_t source_block : dcr_permutation) {
    for (uint32_t channel = 0; channel < channel_count; ++channel) {
      gather_indices.push_back(static_cast<int32_t>(source_block * channel_count + channel));
    }
  }
  return gather_indices;
}

Ort::Status AddLayoutTranspose(QnnModelWrapper& model_wrapper, const OrtNodeUnit& concat_unit,
                               const std::string& transpose_name, const std::string& input_name,
                               const std::string& output_name, std::vector<uint32_t> perm, bool validate,
                               std::vector<Qnn_Tensor_t> validated_inputs,
                               std::vector<Qnn_Tensor_t> validated_outputs) {
  QnnParamWrapper perm_param(concat_unit.Index(), transpose_name, QNN_OP_TRANSPOSE_PARAM_PERM,
                             {static_cast<uint32_t>(perm.size())}, std::move(perm));
  if (validate) {
    std::vector<Qnn_Param_t> params{perm_param.GetQnnParam()};
    return model_wrapper.ValidateQnnNode(transpose_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                         std::move(validated_inputs), std::move(validated_outputs),
                                         std::move(params));
  }
  const std::string perm_param_name = perm_param.GetParamTensorName();
  RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(perm_param)), "Failed to add transpose perm.");
  RETURN_IF_NOT(model_wrapper.CreateQnnNode(transpose_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_TRANSPOSE,
                                            {input_name}, {output_name}, {perm_param_name},
                                            /*validate*/ false),
                "Failed to add layout transpose.");
  return Ort::Status();
}

// Lowering shared by IsSupported (validate=true) and AddToModelBuilder (validate=false):
// PreT(NCHW->NHWC) + S2D(DCR) + [Gather channel reorder] + PostT(NHWC->NCHW).
Ort::Status CreateOrValidateTiledGraph(QnnModelWrapper& model_wrapper, const OrtNodeUnit& common_input_owner,
                                       const OrtNodeUnit& concat_unit,
                                       const std::array<int64_t, 4>& dcr_permutation, uint32_t channel_count,
                                       bool validate, const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  const OrtNodeUnitIODef& tiled_input_def = common_input_owner.Inputs()[0];
  const OrtNodeUnitIODef& tiled_output_def = concat_unit.Outputs()[0];

  QnnTensorWrapper tiled_input_tensor, tiled_output_tensor;
  RETURN_IF_ERROR(model_wrapper.MakeTensorWrapper(tiled_input_def, tiled_input_tensor));
  RETURN_IF_ERROR(model_wrapper.MakeTensorWrapper(tiled_output_def, tiled_output_tensor));

  TensorInfo tiled_input_info = {}, tiled_output_info = {};
  RETURN_IF_ERROR(model_wrapper.GetTensorInfo(tiled_input_def, tiled_input_info));
  RETURN_IF_ERROR(model_wrapper.GetTensorInfo(tiled_output_def, tiled_output_info));

  std::vector<uint32_t> input_shape, output_shape;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(tiled_input_def.shape, input_shape),
                "SliceConcatS2D: bad input shape.");
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(tiled_output_def.shape, output_shape),
                "SliceConcatS2D: bad output shape.");

  const std::string base_name = utils::UniqueNameGenerator().New(concat_unit, "_tiled_s2d");
  const std::string nhwc_input_name = base_name + "_in";
  const std::string nhwc_output_name = base_name + "_s2d_out";
  const std::string pre_name = base_name + "_pre";
  const std::string s2d_name = base_name + "_s2d";
  const std::string post_name = base_name + "_post";

  QnnTensorWrapper nhwc_input_tensor(
      nhwc_input_name, QNN_TENSOR_TYPE_NATIVE, tiled_input_info.qnn_data_type,
      tiled_input_info.quant_param.Copy(),
      std::vector<uint32_t>{input_shape[0], input_shape[2], input_shape[3], input_shape[1]});
  QnnTensorWrapper nhwc_output_tensor(
      nhwc_output_name, QNN_TENSOR_TYPE_NATIVE, tiled_output_info.qnn_data_type,
      tiled_output_info.quant_param.Copy(),
      std::vector<uint32_t>{output_shape[0], output_shape[2], output_shape[3], output_shape[1]});

  const bool needs_gather = dcr_permutation != std::array<int64_t, 4>{0, 1, 2, 3};
  // Gather runs on NCHW channels after PostT (mirrors upstream S2D-then-Gather order).
  // Without a Gather, PostT writes the group output directly; the intermediate must not
  // be re-declared under that name, or the real (possibly APP_READ) tensor is shadowed.
  const std::string nchw_s2d_name = needs_gather ? base_name + "_gather_in" : tiled_output_def.name;

  std::optional<QnnTensorWrapper> nchw_s2d_tensor;
  if (needs_gather) {
    nchw_s2d_tensor.emplace(nchw_s2d_name, QNN_TENSOR_TYPE_NATIVE, tiled_output_info.qnn_data_type,
                            tiled_output_info.quant_param.Copy(), std::vector<uint32_t>(output_shape));
  }

  std::vector<int32_t> gather_indices;
  if (needs_gather) {
    gather_indices = BuildDcrGatherIndices(dcr_permutation, channel_count);
  }
  std::vector<uint8_t> gather_indices_bytes;
  if (needs_gather) {
    gather_indices_bytes.resize(gather_indices.size() * sizeof(int32_t));
    std::memcpy(gather_indices_bytes.data(), gather_indices.data(), gather_indices_bytes.size());
  }
  // Static int32 indices (QNN Gather has no int64 static path); built only when reordering.
  std::optional<QnnTensorWrapper> gather_indices_tensor;
  if (needs_gather) {
    gather_indices_tensor.emplace(
        base_name + "_gather_idx", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, QnnQuantParamsWrapper(),
        std::vector<uint32_t>{static_cast<uint32_t>(gather_indices.size())}, std::move(gather_indices_bytes));
  }

  Qnn_Scalar_t gather_axis_scalar = QNN_SCALAR_INIT;
  gather_axis_scalar.dataType = QNN_DATATYPE_INT_32;
  gather_axis_scalar.int32Value = 1;
  QnnParamWrapper gather_axis_param(concat_unit.Index(), base_name + "_gather",
                                    QNN_OP_GATHER_PARAM_AXIS, gather_axis_scalar);

  std::vector<uint32_t> block_shape{2};
  std::vector<uint32_t> block_data{SliceConcatSpaceToDepthFusion::kBlockHeight,
                                   SliceConcatSpaceToDepthFusion::kBlockWidth};
  QnnParamWrapper block_param(concat_unit.Index(), s2d_name, QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE,
                              std::move(block_shape), std::move(block_data));
  Qnn_Scalar_t mode_scalar = QNN_SCALAR_INIT;
  mode_scalar.dataType = QNN_DATATYPE_UINT_32;
  mode_scalar.uint32Value = QNN_OP_SPACE_TO_DEPTH_MODE_DCR;
  QnnParamWrapper mode_param(concat_unit.Index(), s2d_name, QNN_OP_SPACE_TO_DEPTH_PARAM_MODE, mode_scalar);

  if (validate) {
    RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, concat_unit, pre_name, tiled_input_def.name,
                                       nhwc_input_name, {0, 2, 3, 1}, /*validate*/ true,
                                       {tiled_input_tensor.GetQnnTensor()},
                                       {nhwc_input_tensor.GetQnnTensor()}));
    {
      std::vector<Qnn_Param_t> params{block_param.GetQnnParam(), mode_param.GetQnnParam()};
      RETURN_IF_ERROR(model_wrapper.ValidateQnnNode(s2d_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_SPACE_TO_DEPTH,
                                                    {nhwc_input_tensor.GetQnnTensor()},
                                                    {nhwc_output_tensor.GetQnnTensor()}, std::move(params)));
    }
    RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, concat_unit, post_name, nhwc_output_name,
                                       nchw_s2d_name, {0, 3, 1, 2}, /*validate*/ true,
                                       {nhwc_output_tensor.GetQnnTensor()},
                                       {needs_gather ? nchw_s2d_tensor->GetQnnTensor()
                                                     : tiled_output_tensor.GetQnnTensor()}));
    if (needs_gather) {
      std::vector<Qnn_Param_t> params{gather_axis_param.GetQnnParam()};
      RETURN_IF_ERROR(model_wrapper.ValidateQnnNode(base_name + "_gather", QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_GATHER,
                                                    {nchw_s2d_tensor->GetQnnTensor(),
                                                     gather_indices_tensor->GetQnnTensor()},
                                                    {tiled_output_tensor.GetQnnTensor()}, std::move(params)));
    }
    return Ort::Status();
  }

  auto add_tensor_once = [&](QnnTensorWrapper&& tensor, const std::string& tensor_name,
                             const char* error_message) -> Ort::Status {
    if (model_wrapper.IsQnnTensorWrapperExist(tensor_name)) {
      return Ort::Status();
    }
    RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(tensor)), error_message);
    return Ort::Status();
  };
  RETURN_IF_ERROR(add_tensor_once(std::move(tiled_input_tensor), tiled_input_def.name, "Bad tiled input."));
  RETURN_IF_ERROR(add_tensor_once(std::move(tiled_output_tensor), tiled_output_def.name, "Bad tiled output."));
  RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(nhwc_input_tensor)), "Bad NHWC input.");
  RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(nhwc_output_tensor)), "Bad NHWC S2D output.");
  if (needs_gather) {
    RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(*nchw_s2d_tensor)), "Bad NCHW S2D output.");
    RETURN_IF_NOT(model_wrapper.AddTensorWrapper(std::move(*gather_indices_tensor)), "Bad Gather indices.");
  }

  RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, concat_unit, pre_name, tiled_input_def.name,
                                     nhwc_input_name, {0, 2, 3, 1}, /*validate*/ false, {}, {}));
  {
    const std::string block_name = block_param.GetParamTensorName();
    const std::string mode_name = mode_param.GetParamTensorName();
    RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(block_param)), "Failed to add S2D block param.");
    RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(mode_param)), "Failed to add S2D mode param.");
    RETURN_IF_NOT(model_wrapper.CreateQnnNode(s2d_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_SPACE_TO_DEPTH,
                                              {nhwc_input_name}, {nhwc_output_name}, {block_name, mode_name},
                                              /*validate*/ false),
                  "Failed to add SpaceToDepth.");
  }
  RETURN_IF_ERROR(AddLayoutTranspose(model_wrapper, concat_unit, post_name, nhwc_output_name, nchw_s2d_name,
                                     {0, 3, 1, 2}, /*validate*/ false, {}, {}));

  if (needs_gather) {
    const std::string gather_name = base_name + "_gather";
    const std::string axis_name = gather_axis_param.GetParamTensorName();
    RETURN_IF_NOT(model_wrapper.AddParamWrapper(std::move(gather_axis_param)), "Failed to add Gather axis.");
    RETURN_IF_NOT(model_wrapper.CreateQnnNode(gather_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_GATHER,
                                              {nchw_s2d_name, base_name + "_gather_idx"},
                                              {tiled_output_def.name}, {axis_name}, /*validate*/ false),
                  "Failed to add channel Gather.");
  }
  return Ort::Status();
}

}  // namespace

gsl::span<const OrtNodeUnit* const> SliceConcatSpaceToDepthFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

std::unique_ptr<IQnnNodeGroup> SliceConcatSpaceToDepthFusion::TryFusion(
    QnnModelWrapper& model_wrapper, const OrtNodeUnit& concat_unit, const NodeToUnitMap& node_to_unit,
    const UnitToGroupMap& unit_to_group, const Ort::Logger& logger) {
  if (!IsConcatUnit(&concat_unit) || concat_unit.Inputs().size() != 4) {
    return nullptr;
  }
  if (OrtNodeAttrHelper(concat_unit).Get("axis", static_cast<int64_t>(-1)) != kChannelAxis) {
    return nullptr;
  }
  if (model_wrapper.GetQnnBackendType() != QnnBackendType::HTP &&
      model_wrapper.GetQnnBackendType() != QnnBackendType::HTP_FP16) {
    return nullptr;
  }

  // Resolve each Concat input to its (h, w) phase; parallel tiling (one
  // multi-axis Slice) and cascaded tiling (two single-axis Slices) compose to
  // the same walk. All walks must share one root tensor.
  std::array<ConcatInputWalk, 4> walks{};
  for (size_t i = 0; i < walks.size(); ++i) {
    auto walk = ResolveConcatInputPhase(model_wrapper, concat_unit, i, node_to_unit, unit_to_group);
    if (!walk.has_value() || walk->root_tensor_name.empty()) {
      return nullptr;
    }
    walks[i] = std::move(*walk);
  }
  const std::string& root_tensor_name = walks[0].root_tensor_name;
  for (size_t i = 1; i < walks.size(); ++i) {
    if (walks[i].root_tensor_name != root_tensor_name) {
      return nullptr;
    }
  }

  // Every intermediate tensor must be consumed only inside the pattern;
  // otherwise lowering to S2D would silently drop that edge's tensor.
  // Consumers resolve through NodeUnits: a DQ node belongs to its group.
  std::unordered_map<const OrtNodeUnit*, std::vector<const OrtNodeUnit*>> expected_consumers;
  for (const auto& walk : walks) {
    for (size_t p = 0; p < walk.path_units.size(); ++p) {
      const OrtNodeUnit* expected =
          (p == 0) ? static_cast<const OrtNodeUnit*>(&concat_unit) : walk.path_units[p - 1];
      expected_consumers[walk.path_units[p]].push_back(expected);
    }
  }
  std::vector<const OrtNodeUnit*> pattern_slices;
  pattern_slices.reserve(7);
  for (const auto& entry : expected_consumers) {
    if (!HasExactlyUnitConsumers(*entry.first, entry.second, node_to_unit)) {
      return nullptr;
    }
    pattern_slices.push_back(entry.first);
  }

  // Uniform shape proof, independent of tiling form: the root is NCHW with even
  // spatial dims, every Concat input is exactly one tile, Concat stacks 4 tiles.
  std::vector<uint32_t> input_shape, concat_shape;
  const OrtNodeUnit& root_owner = *walks[0].path_units.back();
  if (root_owner.Inputs().empty() ||
      !QnnModelWrapper::GetOnnxShape(root_owner.Inputs()[0].shape, input_shape) ||
      !QnnModelWrapper::GetOnnxShape(concat_unit.Outputs()[0].shape, concat_shape)) {
    return nullptr;
  }
  if (input_shape.size() != kNchwRank || input_shape[0] == 0 || input_shape[1] == 0 ||
      input_shape[2] % 2 != 0 || input_shape[3] % 2 != 0 || input_shape[2] == 0 || input_shape[3] == 0) {
    return nullptr;
  }
  const uint32_t batch = input_shape[0];
  const uint32_t channels = input_shape[1];
  const uint32_t height = input_shape[2];
  const uint32_t width = input_shape[3];
  for (size_t i = 0; i < walks.size(); ++i) {
    std::vector<uint32_t> tile_shape;
    if (!QnnModelWrapper::GetOnnxShape(concat_unit.Inputs()[i].shape, tile_shape) ||
        !HasExpectedShape(tile_shape, batch, channels, height / 2, width / 2)) {
      return nullptr;
    }
  }
  if (!HasExpectedShape(concat_shape, batch, 4 * channels, height / 2, width / 2)) {
    return nullptr;
  }

  // Effective (h, w) phase per Concat input; any permutation of the four is accepted,
  // with the Gather restoring exact Concat order downstream.
  std::array<SlicePhase, 4> actual_phases{};
  for (size_t i = 0; i < walks.size(); ++i) {
    actual_phases[i] = walks[i].phase;
  }
  std::array<int64_t, 4> dcr_permutation{0, 1, 2, 3};
  if (!TryGetDcrPhasePermutation(actual_phases, dcr_permutation)) {
    return nullptr;
  }

  // Every boundary tensor shares one per-tensor quant; S2D+Gather only rearrange,
  // so any intermediate requant would be skipped and change numerics. Both sides of
  // each edge are checked: in QDQ graphs a Slice output def and the consuming def
  // carry their own DQ encodings, so an input-side requant is invisible in the
  // producer's def alone.
  std::vector<const OrtNodeUnitIODef*> boundary_defs;
  boundary_defs.reserve(2 * pattern_slices.size() + 6);
  boundary_defs.push_back(&root_owner.Inputs()[0]);
  for (const OrtNodeUnit* slice_unit : pattern_slices) {
    if (slice_unit->Inputs().empty() || slice_unit->Outputs().empty()) {
      return nullptr;
    }
    boundary_defs.push_back(&slice_unit->Inputs()[0]);
    boundary_defs.push_back(&slice_unit->Outputs()[0]);
  }
  for (const OrtNodeUnitIODef& concat_input : concat_unit.Inputs()) {
    boundary_defs.push_back(&concat_input);
  }
  boundary_defs.push_back(&concat_unit.Outputs()[0]);
  // Float S2D-DCR is inaccurate on HTP (AISW-175353; upstream float-DCR tests disabled),
  // so fuse quantized graphs only. Revisit when the kernel is fixed.
  std::optional<PerTensorQuant> reference_quant;
  for (const OrtNodeUnitIODef* boundary_def : boundary_defs) {
    const auto boundary_quant = GetBoundaryQuant(model_wrapper, *boundary_def);
    if (!boundary_quant.has_value()) {
      return nullptr;
    }
    if (!reference_quant.has_value()) {
      reference_quant = boundary_quant;
    } else if (!HasSameQuant(*reference_quant, *boundary_quant)) {
      return nullptr;
    }
  }

  if (!reference_quant.has_value() || !reference_quant->quantized) {
    return nullptr;
  }

  // Validate on the backend before claiming the NodeUnits: a rejection here must leave
  // the Slices and Concat free to be offloaded individually, and anything that slips
  // past Phase 1 would abort the whole partition's Compile.
  if (!CreateOrValidateTiledGraph(model_wrapper, root_owner, concat_unit, dcr_permutation, channels,
                                  /*validate*/ true, logger)
           .IsOK()) {
    return nullptr;
  }

  pattern_slices.push_back(&concat_unit);
  return std::make_unique<SliceConcatSpaceToDepthFusion>(
      gsl::make_span<const OrtNodeUnit* const>(pattern_slices.data(), pattern_slices.size()),
      root_owner, dcr_permutation, channels);
}

Ort::Status SliceConcatSpaceToDepthFusion::IsSupported(QnnModelWrapper& model_wrapper,
                                                       const Ort::Logger& logger) const {
  if (model_wrapper.GetQnnBackendType() != QnnBackendType::HTP &&
      model_wrapper.GetQnnBackendType() != QnnBackendType::HTP_FP16) {
    return MAKE_EP_FAIL("SliceConcatS2D: HTP only.");
  }
  if (node_units_.size() < kMinGroupSize || !IsConcatUnit(concat_node_unit_) ||
      root_input_owner_ == nullptr) {
    return MAKE_EP_FAIL("SliceConcatS2D: expected >= 5 units with Concat target.");
  }
  std::vector<uint32_t> input_shape, output_shape;
  if (!QnnModelWrapper::GetOnnxShape(root_input_owner_->Inputs()[0].shape, input_shape) ||
      !QnnModelWrapper::GetOnnxShape(concat_node_unit_->Outputs()[0].shape, output_shape)) {
    return MAKE_EP_FAIL("SliceConcatS2D: unresolved shapes.");
  }
  if (input_shape.size() != kNchwRank || output_shape.size() != kNchwRank ||
      std::any_of(input_shape.begin(), input_shape.end(), [](uint32_t dim) { return dim == 0; }) ||
      std::any_of(output_shape.begin(), output_shape.end(), [](uint32_t dim) { return dim == 0; })) {
    return MAKE_EP_FAIL("SliceConcatS2D: rank-4 non-zero shapes only.");
  }
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateTiledGraph(model_wrapper, *root_input_owner_, *concat_node_unit_, phase_permutation_,
                                    channel_count_, /*validate*/ true, logger);
}

Ort::Status SliceConcatSpaceToDepthFusion::AddToModelBuilder(QnnModelWrapper& model_wrapper,
                                                             const Ort::Logger& logger) const {
  return CreateOrValidateTiledGraph(model_wrapper, *root_input_owner_, *concat_node_unit_, phase_permutation_,
                                    channel_count_, /*validate*/ false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
