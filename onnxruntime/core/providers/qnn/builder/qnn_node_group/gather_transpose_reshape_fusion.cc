// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_node_group/gather_transpose_reshape_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace {

constexpr char kOpGather[] = "Gather";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpReshape[] = "Reshape";

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, node_units, index_pattern, idx0, idx1, perm_4d) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (index_pattern), (idx0), (idx1), (perm_4d), true)
#define CreateOnQnn(qnn_model_wrapper, node_units, index_pattern, idx0, idx1, perm_4d) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (index_pattern), (idx0), (idx1), (perm_4d), false)

static Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    GatherIndicesPattern index_pattern,
    int64_t idx0,
    int64_t idx1,
    const std::vector<uint32_t>& perm_4d,
    bool validate);

// Returns true if indices[i,j] == i*cols + j for all i,j.
static bool IsRowMajorPattern(const int64_t* data, int64_t rows, int64_t cols) {
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      if (data[i * cols + j] != i * cols + j) return false;
  return true;
}

// Returns true if indices[i,j] == j*rows + i for all i,j.
static bool IsColMajorPattern(const int64_t* data, int64_t rows, int64_t cols) {
  for (int64_t i = 0; i < rows; ++i)
    for (int64_t j = 0; j < cols; ++j)
      if (data[i * cols + j] != j * rows + i) return false;
  return true;
}

static std::optional<GatherIndicesPattern> DetectIndicesPattern(
    const std::vector<uint8_t>& raw_bytes,
    ONNXTensorElementDataType elem_type,
    int64_t rows,
    int64_t cols) {
  const int64_t total = rows * cols;
  std::vector<int64_t> data(static_cast<size_t>(total));

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    if (raw_bytes.size() != static_cast<size_t>(total) * sizeof(int64_t)) return std::nullopt;
    const int64_t* src = reinterpret_cast<const int64_t*>(raw_bytes.data());
    std::copy(src, src + total, data.begin());
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    if (raw_bytes.size() != static_cast<size_t>(total) * sizeof(int32_t)) return std::nullopt;
    const int32_t* src = reinterpret_cast<const int32_t*>(raw_bytes.data());
    for (int64_t k = 0; k < total; ++k) data[static_cast<size_t>(k)] = src[k];
  } else {
    return std::nullopt;
  }

  if (IsRowMajorPattern(data.data(), rows, cols)) return GatherIndicesPattern::kRowMajor;
  if (IsColMajorPattern(data.data(), rows, cols)) return GatherIndicesPattern::kColMajor;
  return std::nullopt;
}

// Matches Gather -> Transpose -> Reshape starting from a Gather node unit.
static std::optional<std::array<const OrtNodeUnit*, 3>> MatchPattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& gather_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  if (gather_node_unit.OpType() != kOpGather ||
      gather_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  const std::array<std::string_view, 1> transpose_types{kOpTranspose};
  const OrtNodeUnit* transpose = GetOnlyChildOfType(qnn_model_wrapper, gather_node_unit,
                                                    transpose_types, node_to_node_unit,
                                                    node_unit_to_qnn_node_group);
  if (transpose == nullptr) return std::nullopt;

  const std::array<std::string_view, 1> reshape_types{kOpReshape};
  const OrtNodeUnit* reshape = GetOnlyChildOfType(qnn_model_wrapper, *transpose,
                                                  reshape_types, node_to_node_unit,
                                                  node_unit_to_qnn_node_group);
  if (reshape == nullptr) return std::nullopt;

  return std::array<const OrtNodeUnit*, 3>{&gather_node_unit, transpose, reshape};
}

// Validates all fusion preconditions and computes the adjusted 4D permutation.
static bool ValidateAndComputeFusionParams(
    const OrtNodeUnit& gather,
    const OrtNodeUnit& transpose,
    const OrtNodeUnit& reshape,
    const QnnModelWrapper& qnn_model_wrapper,
    GatherIndicesPattern& index_pattern,
    int64_t& idx0,
    int64_t& idx1,
    std::vector<uint32_t>& perm_4d) {
  // Input must be rank-5 with fully static shape.
  const OrtNodeUnitIODef& x_def = gather.Inputs()[0];
  const std::vector<int64_t>& x_shape = x_def.shape;
  if (x_shape.size() != 5) return false;
  for (int64_t d : x_shape)
    if (d <= 0) return false;

  // Gather axis must be 4 (last dim of rank-5).
  OrtNodeAttrHelper gather_attrs(gather);
  int64_t axis = gather_attrs.Get("axis", int64_t(0));
  if (axis < 0) axis += static_cast<int64_t>(x_shape.size());
  if (axis != 4) return false;

  // Indices must be rank-2 and constant.
  if (gather.Inputs().size() < 2) return false;
  const OrtNodeUnitIODef& idx_def = gather.Inputs()[1];
  if (idx_def.shape.size() != 2 || idx_def.shape[0] <= 0 || idx_def.shape[1] <= 0) return false;
  if (!qnn_model_wrapper.IsConstantInput(idx_def.name)) return false;
  idx0 = idx_def.shape[0];
  idx1 = idx_def.shape[1];

  // Indices must cover exactly the gathered dimension.
  if (x_shape[4] != idx0 * idx1) return false;

  // Gather output must be rank-6.
  if (gather.Outputs().empty() || gather.Outputs()[0].shape.size() != 6) return false;

  // Detect row-major / col-major index pattern.
  const OrtValueInfo* idx_vi = qnn_model_wrapper.GetConstantTensor(idx_def.name);
  if (idx_vi == nullptr) return false;
  std::vector<uint8_t> idx_bytes;
  if (!qnn_model_wrapper.UnpackInitializerData(idx_vi, idx_bytes).IsOK()) return false;
  auto detected = DetectIndicesPattern(idx_bytes, idx_def.type, idx0, idx1);
  if (!detected.has_value()) return false;
  index_pattern = *detected;

  // Transpose perm must be length-6, perm[0:3]==[0,1,2], perm[3:6] a permutation of {3,4,5}.
  OrtNodeAttrHelper transpose_attrs(transpose);
  std::vector<int64_t> perm = transpose_attrs.Get("perm", std::vector<int64_t>{});
  if (perm.size() != 6 || perm[0] != 0 || perm[1] != 1 || perm[2] != 2) return false;
  std::unordered_set<int64_t> tail(perm.begin() + 3, perm.end());
  if (tail != std::unordered_set<int64_t>{3, 4, 5}) return false;

  // Transpose output must be rank-6.
  if (transpose.Outputs().empty() || transpose.Outputs()[0].shape.size() != 6) return false;

  // Reshape output must be rank < 6.
  if (reshape.Outputs().empty() || reshape.Outputs()[0].shape.size() >= 6) return false;

  // Compute 4D permutation: dims 0,1,2 merge to 0; old dim k -> new dim k-2 for k in {3,4,5}.
  perm_4d.resize(4);
  perm_4d[0] = 0;
  for (size_t k = 1; k < 4; ++k)
    perm_4d[k] = static_cast<uint32_t>(perm[k + 2] - 2);

  return true;
}

static Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    GatherIndicesPattern index_pattern,
    int64_t idx0,
    int64_t idx1,
    const std::vector<uint32_t>& perm_4d,
    bool validate) {
  const OrtNodeUnit* gather = node_units[0];
  const OrtNodeUnit* transpose = node_units[1];
  const OrtNodeUnit* reshape = node_units[2];

  const OrtNodeUnitIODef& gather_input = gather->Inputs()[0];
  const OrtNodeUnitIODef& reshape_output = reshape->Outputs()[0];

  Qnn_DataType_t data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(gather_input.quant_param.has_value(), gather_input.type, data_type));

  QnnTensorWrapper input_wrapper;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(gather_input, input_wrapper));
  QnnQuantParamsWrapper quant_params = input_wrapper.GetQnnQuantParams();

  std::vector<uint32_t> x_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_input.shape, x_shape), "Failed to get input shape");
  RETURN_IF_NOT(x_shape.size() == 5, "Expected rank-5 input");

  const uint32_t d0 = x_shape[0], d1 = x_shape[1], d2 = x_shape[2], d3 = x_shape[3];
  RETURN_IF_NOT(d0 <= UINT32_MAX / d1 && d0 * d1 <= UINT32_MAX / d2, "Merged batch dimension overflows uint32");
  const uint32_t merged = d0 * d1 * d2;
  const uint32_t u_idx0 = static_cast<uint32_t>(idx0);
  const uint32_t u_idx1 = static_cast<uint32_t>(idx1);

  // For col-major, d4 splits as [idx1, idx0]; for row-major as [idx0, idx1].
  const uint32_t factor0 = (index_pattern == GatherIndicesPattern::kColMajor) ? u_idx1 : u_idx0;
  const uint32_t factor1 = (index_pattern == GatherIndicesPattern::kColMajor) ? u_idx0 : u_idx1;

  const std::string r1_out = utils::GetUniqueName(*gather, "_gtr_r1_out");
  const std::string col_t_out = utils::GetUniqueName(*gather, "_gtr_col_t_out");
  const std::string main_t_out = utils::GetUniqueName(*gather, "_gtr_main_t_out");

  // Reshape rank-5 input to rank-4: [d0,d1,d2,d3,d4] -> [merged, d3, factor0, factor1]
  const std::vector<uint32_t> r1_shape = {merged, d3, factor0, factor1};
  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
      gather_input.name, r1_out, x_shape, r1_shape, data_type, quant_params, validate,
      qnn_model_wrapper.IsGraphInput(gather_input.name), false));

  std::string main_t_in = r1_out;
  const std::vector<uint32_t> normalised = {merged, d3, u_idx0, u_idx1};

  if (index_pattern == GatherIndicesPattern::kColMajor) {
    // Swap last two dims to normalise to [merged, d3, idx0, idx1].
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        gather->Index(), r1_out, col_t_out,
        r1_shape, {0u, 1u, 3u, 2u}, normalised,
        data_type, quant_params, validate, false, false));
    main_t_in = col_t_out;
  }

  std::vector<uint32_t> main_t_shape(4);
  for (size_t k = 0; k < 4; ++k) main_t_shape[k] = normalised[perm_4d[k]];

  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
      transpose->Index(), main_t_in, main_t_out,
      normalised, perm_4d, main_t_shape,
      data_type, quant_params, validate, false, false));

  std::vector<uint32_t> out_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(reshape_output.shape, out_shape), "Failed to get output shape");

  Qnn_DataType_t out_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(reshape_output.quant_param.has_value(), reshape_output.type, out_type));

  QnnTensorWrapper output_wrapper;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(reshape_output, output_wrapper));

  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
      main_t_out, reshape_output.name, main_t_shape, out_shape,
      out_type, output_wrapper.GetQnnQuantParams(), validate,
      false, qnn_model_wrapper.IsGraphOutput(reshape_output.name)));

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> GatherTransposeReshapeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& gather_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  auto pattern = MatchPattern(qnn_model_wrapper, gather_node_unit,
                              node_to_node_unit, node_unit_to_qnn_node_group);
  if (!pattern.has_value()) return nullptr;

  GatherIndicesPattern index_pattern = GatherIndicesPattern::kRowMajor;
  int64_t idx0 = 0, idx1 = 0;
  std::vector<uint32_t> perm_4d;

  if (!ValidateAndComputeFusionParams(*pattern->at(0), *pattern->at(1), *pattern->at(2),
                                      qnn_model_wrapper, index_pattern, idx0, idx1, perm_4d)) {
    return nullptr;
  }

  if (!ValidateOnQnn(qnn_model_wrapper, *pattern, index_pattern, idx0, idx1, perm_4d).IsOK()) {
    return nullptr;
  }

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              ("GatherTransposeReshapeFusion: fusing Gather+Transpose+Reshape for node '" +
               gather_node_unit.Name() + "'")
                  .c_str());

  return std::make_unique<GatherTransposeReshapeFusion>(
      *pattern, index_pattern, idx0, idx1, std::move(perm_4d));
}

gsl::span<const OrtNodeUnit* const> GatherTransposeReshapeFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status GatherTransposeReshapeFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const Ort::Logger& logger) const {
  return ValidateOnQnn(qnn_model_wrapper, GetNodeUnits(), index_pattern_, idx0_, idx1_, new_perm_4d_);
}

Ort::Status GatherTransposeReshapeFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, const Ort::Logger& logger) const {
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              ("GatherTransposeReshapeFusion: building fused ops for node '" +
               node_units_[0]->Name() + "'")
                  .c_str());
  return CreateOnQnn(qnn_model_wrapper, GetNodeUnits(), index_pattern_, idx0_, idx1_, new_perm_4d_);
}

}  // namespace qnn
}  // namespace onnxruntime
