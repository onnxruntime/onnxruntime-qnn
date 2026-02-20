// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

constexpr size_t kRank4 = 4;
constexpr size_t kRank5 = 5;
constexpr size_t kRank6 = 6;
constexpr int64_t kGatherAxis = 4;
constexpr const char* kOpTypeGather = "Gather";
constexpr const char* kOpTypeTranspose = "Transpose";
constexpr const char* kOpTypeReshape = "Reshape";
constexpr const char* kAttrAxis = "axis";
constexpr const char* kAttrPerm = "perm";

using MapNodeToNodeUnit = std::unordered_map<const OrtNode*, const OrtNodeUnit*>;
using MapNodeUnitToGroup = std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>;

// ---------------------------------------------------------------------------
// Index-pattern detection helpers
// ---------------------------------------------------------------------------

/// @brief Check whether the int64 indices array follows the row-major pattern:
///        indices[i, j] == i * idx1 + j  for all i in [0, idx0), j in [0, idx1).
static bool IsRowMajorPattern(const int64_t* data, int64_t idx0, int64_t idx1) {
  for (int64_t i = 0; i < idx0; ++i) {
    for (int64_t j = 0; j < idx1; ++j) {
      if (data[i * idx1 + j] != i * idx1 + j) {
        return false;
      }
    }
  }
  return true;
}

/// @brief Check whether the int64 indices array follows the col-major pattern:
///        indices[i, j] == j * idx0 + i  for all i in [0, idx0), j in [0, idx1).
static bool IsColMajorPattern(const int64_t* data, int64_t idx0, int64_t idx1) {
  for (int64_t i = 0; i < idx0; ++i) {
    for (int64_t j = 0; j < idx1; ++j) {
      if (data[i * idx1 + j] != j * idx0 + i) {
        return false;
      }
    }
  }
  return true;
}

/// @brief Detect the index pattern from a raw byte buffer.
///        Supports int32 and int64 element types.
///        Returns std::nullopt if the pattern is neither row-major nor col-major.
static std::optional<GatherIndicesPattern> DetectIndicesPattern(
    const std::vector<uint8_t>& indices_bytes,
    ONNXTensorElementDataType elem_type,
    int64_t idx0,
    int64_t idx1) {
  const int64_t total = idx0 * idx1;

  // Convert to a temporary int64 buffer for uniform comparison.
  std::vector<int64_t> indices_i64(static_cast<size_t>(total));

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    if (indices_bytes.size() != static_cast<size_t>(total) * sizeof(int64_t)) {
      return std::nullopt;
    }
    const int64_t* src = reinterpret_cast<const int64_t*>(indices_bytes.data());
    std::copy(src, src + total, indices_i64.begin());
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    if (indices_bytes.size() != static_cast<size_t>(total) * sizeof(int32_t)) {
      return std::nullopt;
    }
    const int32_t* src = reinterpret_cast<const int32_t*>(indices_bytes.data());
    for (int64_t k = 0; k < total; ++k) {
      indices_i64[static_cast<size_t>(k)] = static_cast<int64_t>(src[k]);
    }
  } else {
    return std::nullopt;  // Unsupported index dtype
  }

  const int64_t* data = indices_i64.data();
  if (IsRowMajorPattern(data, idx0, idx1)) {
    return GatherIndicesPattern::kRowMajor;
  }
  if (IsColMajorPattern(data, idx0, idx1)) {
    return GatherIndicesPattern::kColMajor;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Pattern matching
// ---------------------------------------------------------------------------

/// @brief Starting from a Gather NodeUnit, attempt to match the full
///        Gather -> Transpose -> Reshape chain.
///        Returns {gather, transpose, reshape} or std::nullopt.
static std::optional<std::array<const OrtNodeUnit*, 3>> MatchPattern(
    const QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& gather_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    [[maybe_unused]] const Ort::Logger& logger) {
  if (gather_node_unit.OpType() != kOpTypeGather) {
    return std::nullopt;
  }

  // Gather must be a standalone (non-QDQ) node for this fusion.
  if (gather_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return std::nullopt;
  }

  // Find the single Transpose consumer of the Gather output.
  const std::array<std::string_view, 1> transpose_types{kOpTypeTranspose};
  const OrtNodeUnit* transpose_node_unit = GetOnlyChildOfType(
      qnn_model_wrapper,
      gather_node_unit,
      transpose_types,
      node_to_node_unit,
      node_unit_to_qnn_node_group);
  if (transpose_node_unit == nullptr) {
    return std::nullopt;
  }

  // Find the single Reshape consumer of the Transpose output.
  const std::array<std::string_view, 1> reshape_types{kOpTypeReshape};
  const OrtNodeUnit* reshape_node_unit = GetOnlyChildOfType(
      qnn_model_wrapper,
      *transpose_node_unit,
      reshape_types,
      node_to_node_unit,
      node_unit_to_qnn_node_group);
  if (reshape_node_unit == nullptr) {
    return std::nullopt;
  }

  return std::array<const OrtNodeUnit*, 3>{&gather_node_unit, transpose_node_unit, reshape_node_unit};
}

// ---------------------------------------------------------------------------
// Condition validation
// ---------------------------------------------------------------------------

/// @brief Validate all pattern conditions and compute the fusion parameters.
///        On success, populates index_pattern, idx0, idx1, and new_perm_4d.
static bool ValidateAndComputeFusionParams(
    const OrtNodeUnit& gather_node_unit,
    const OrtNodeUnit& transpose_node_unit,
    const OrtNodeUnit& reshape_node_unit,
    const QnnModelWrapper& qnn_model_wrapper,
    [[maybe_unused]] const Ort::Logger& logger,
    /*out*/ GatherIndicesPattern& index_pattern,
    /*out*/ int64_t& idx0,
    /*out*/ int64_t& idx1,
    /*out*/ std::vector<uint32_t>& new_perm_4d) {
  // ------------------------------------------------------------------
  // 1. Gather input must be rank-5.
  // ------------------------------------------------------------------
  const OrtNodeUnitIODef& x_def = gather_node_unit.Inputs()[0];
  const std::vector<int64_t>& x_shape = x_def.shape;
  if (x_shape.size() != kRank5) {
    return false;
  }
  for (int64_t d : x_shape) {
    if (d <= 0) return false;  // Require fully static shape
  }

  // ------------------------------------------------------------------
  // 2. Gather axis must be 4 (last dimension of rank-5 input).
  // ------------------------------------------------------------------
  OrtNodeAttrHelper gather_helper(gather_node_unit);
  int64_t axis = gather_helper.Get(kAttrAxis, int64_t(0));
  if (axis < 0) {
    axis += static_cast<int64_t>(x_shape.size());
  }
  if (axis != kGatherAxis) {
    return false;
  }

  const int64_t d4 = x_shape[4];  // Size of the gathered dimension

  // ------------------------------------------------------------------
  // 3. Gather indices must be rank-2 and constant.
  // ------------------------------------------------------------------
  if (gather_node_unit.Inputs().size() < 2) {
    return false;
  }
  const OrtNodeUnitIODef& indices_def = gather_node_unit.Inputs()[1];
  const std::vector<int64_t>& indices_shape = indices_def.shape;
  if (indices_shape.size() != 2) {
    return false;
  }
  if (indices_shape[0] <= 0 || indices_shape[1] <= 0) {
    return false;
  }
  idx0 = indices_shape[0];
  idx1 = indices_shape[1];

  if (!qnn_model_wrapper.IsConstantInput(indices_def.name)) {
    return false;
  }

  // ------------------------------------------------------------------
  // 4. d4 == idx0 * idx1 (indices cover exactly the gathered dimension).
  // ------------------------------------------------------------------
  if (d4 != idx0 * idx1) {
    return false;
  }

  // ------------------------------------------------------------------
  // 5. Gather output must be rank-6.
  // ------------------------------------------------------------------
  if (gather_node_unit.Outputs().empty()) {
    return false;
  }
  const OrtNodeUnitIODef& gather_out_def = gather_node_unit.Outputs()[0];
  if (gather_out_def.shape.size() != kRank6) {
    return false;
  }

  // ------------------------------------------------------------------
  // 6. Read indices data and detect row-major / col-major pattern.
  // ------------------------------------------------------------------
  const OrtValueInfo* indices_vi = qnn_model_wrapper.GetConstantTensor(indices_def.name);
  if (indices_vi == nullptr) {
    return false;
  }

  std::vector<uint8_t> indices_bytes;
  if (!qnn_model_wrapper.UnpackInitializerData(indices_vi, indices_bytes).IsOK()) {
    return false;
  }

  std::optional<GatherIndicesPattern> detected = DetectIndicesPattern(
      indices_bytes, indices_def.type, idx0, idx1);
  if (!detected.has_value()) {
    return false;
  }
  index_pattern = detected.value();

  // ------------------------------------------------------------------
  // 7. Transpose perm must have length 6, perm[0:3] == [0,1,2], and
  //    perm[3:6] must be a permutation of {3,4,5}.
  //    This guarantees that dims 0,1,2 stay together (so they can be
  //    merged) and that new_perm_4d is a valid 4-element permutation.
  // ------------------------------------------------------------------
  OrtNodeAttrHelper transpose_helper(transpose_node_unit);
  std::vector<int64_t> perm = transpose_helper.Get(kAttrPerm, std::vector<int64_t>{});
  if (perm.size() != kRank6) {
    return false;
  }
  if (perm[0] != 0 || perm[1] != 1 || perm[2] != 2) {
    return false;
  }

  // Verify perm[3:6] is exactly {3,4,5} in some order.
  std::unordered_set<int64_t> tail_set(perm.begin() + 3, perm.end());
  if (tail_set != std::unordered_set<int64_t>{3, 4, 5}) {
    return false;
  }

  // ------------------------------------------------------------------
  // 8. Transpose output must be rank-6.
  // ------------------------------------------------------------------
  if (transpose_node_unit.Outputs().empty()) {
    return false;
  }
  if (transpose_node_unit.Outputs()[0].shape.size() != kRank6) {
    return false;
  }

  // ------------------------------------------------------------------
  // 9. Reshape output must be rank < 6.
  // ------------------------------------------------------------------
  if (reshape_node_unit.Outputs().empty()) {
    return false;
  }
  const std::vector<int64_t>& reshape_out_shape = reshape_node_unit.Outputs()[0].shape;
  if (reshape_out_shape.size() >= kRank6) {
    return false;
  }

  // ------------------------------------------------------------------
  // 10. Compute the adjusted 4D permutation.
  //
  //     The original 6D perm operates on [d0,d1,d2,d3,idx0,idx1].
  //     After merging dims 0,1,2 into a single "merged" dim, the 4D
  //     tensor is [merged, d3, idx0, idx1].
  //
  //     Mapping from old 6D dim index to new 4D dim index:
  //       {0,1,2} -> 0  (all part of merged)
  //       3       -> 1
  //       4       -> 2
  //       5       -> 3
  //
  //     Since perm[0:3] == [0,1,2] (verified above), the merged dim
  //     always stays at position 0 in the output.  The remaining three
  //     output positions come from perm[3], perm[4], perm[5], each of
  //     which is in {3,4,5}, so the mapping is simply: new = old - 2.
  // ------------------------------------------------------------------
  new_perm_4d.resize(kRank4);
  new_perm_4d[0] = 0;
  for (size_t k = 1; k < kRank4; ++k) {
    // perm[k+2] is in {3,4,5}; subtract 2 to get {1,2,3}.
    new_perm_4d[k] = static_cast<uint32_t>(perm[k + 2] - 2);
  }

  return true;
}

// ---------------------------------------------------------------------------
// QNN node construction
// ---------------------------------------------------------------------------

/// @brief Build (or validate) the fused sequence of QNN nodes.
static Ort::Status CreateOrValidateOnQnn(
    QnnModelWrapper& qnn_model_wrapper,
    gsl::span<const OrtNodeUnit* const> node_units,
    GatherIndicesPattern index_pattern,
    int64_t idx0,
    int64_t idx1,
    const std::vector<uint32_t>& new_perm_4d,
    bool validate,
    [[maybe_unused]] const Ort::Logger& logger) {
  const OrtNodeUnit* gather = node_units[0];
  const OrtNodeUnit* transpose = node_units[1];
  const OrtNodeUnit* reshape = node_units[2];

  const OrtNodeUnitIODef& gather_input = gather->Inputs()[0];
  const OrtNodeUnitIODef& reshape_output = reshape->Outputs()[0];

  Qnn_DataType_t data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(
      gather_input.quant_param.has_value(), gather_input.type, data_type));

  QnnTensorWrapper gather_input_wrapper;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(gather_input, gather_input_wrapper));
  QnnQuantParamsWrapper quant_params = gather_input_wrapper.GetQnnQuantParams();

  std::vector<uint32_t> x_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_input.shape, x_shape),
                "GatherTransposeReshapeFusion: failed to get gather input shape");
  RETURN_IF_NOT(x_shape.size() == kRank5,
                "GatherTransposeReshapeFusion: gather input must be rank-5");

  const uint32_t d0 = x_shape[0];
  const uint32_t d1 = x_shape[1];
  const uint32_t d2 = x_shape[2];
  const uint32_t d3 = x_shape[3];
  const uint32_t merged = d0 * d1 * d2;
  const uint32_t u_idx0 = static_cast<uint32_t>(idx0);
  const uint32_t u_idx1 = static_cast<uint32_t>(idx1);

  const uint32_t factor0 = (index_pattern == GatherIndicesPattern::kColMajor) ? u_idx1 : u_idx0;
  const uint32_t factor1 = (index_pattern == GatherIndicesPattern::kColMajor) ? u_idx0 : u_idx1;

  const std::string base = utils::GetUniqueName(*gather);
  const std::string reshape1_out_name = base + "_gtr_r1_out";
  const std::string col_t_out_name = base + "_gtr_col_t_out";
  const std::string main_t_out_name = base + "_gtr_main_t_out";

  const std::vector<uint32_t> reshape1_out_shape = {merged, d3, factor0, factor1};

  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
      gather_input.name,
      reshape1_out_name,
      x_shape,
      reshape1_out_shape,
      data_type,
      quant_params,
      validate,
      /*is_for_input=*/qnn_model_wrapper.IsGraphInput(gather_input.name),
      /*is_for_output=*/false));

  std::string main_t_in_name = reshape1_out_name;
  const std::vector<uint32_t> normalised_shape = {merged, d3, u_idx0, u_idx1};

  if (index_pattern == GatherIndicesPattern::kColMajor) {
    const std::vector<uint32_t> swap_perm = {0u, 1u, 3u, 2u};
    // Input shape is [merged, d3, idx1, idx0]; output is [merged, d3, idx0, idx1].
    RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
        gather->Index(),
        reshape1_out_name,
        col_t_out_name,
        reshape1_out_shape,  // [merged, d3, idx1, idx0]
        swap_perm,
        normalised_shape,  // [merged, d3, idx0, idx1]
        data_type,
        quant_params,
        validate,
        /*is_for_input=*/false,
        /*is_for_output=*/false));
    main_t_in_name = col_t_out_name;
  }

  std::vector<uint32_t> main_t_out_shape(kRank4);
  for (size_t k = 0; k < kRank4; ++k) {
    main_t_out_shape[k] = normalised_shape[new_perm_4d[k]];
  }

  RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(
      transpose->Index(),
      main_t_in_name,
      main_t_out_name,
      normalised_shape,
      new_perm_4d,
      main_t_out_shape,
      data_type,
      quant_params,
      validate,
      /*is_for_input=*/false,
      /*is_for_output=*/false));

  std::vector<uint32_t> final_output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(reshape_output.shape, final_output_shape),
                "GatherTransposeReshapeFusion: failed to get reshape output shape");

  Qnn_DataType_t output_data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(
      reshape_output.quant_param.has_value(), reshape_output.type, output_data_type));

  QnnTensorWrapper reshape_output_wrapper;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(reshape_output, reshape_output_wrapper));

  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
      main_t_out_name,
      reshape_output.name,
      main_t_out_shape,
      final_output_shape,
      output_data_type,
      reshape_output_wrapper.GetQnnQuantParams(),
      validate,
      /*is_for_input=*/false,
      /*is_for_output=*/qnn_model_wrapper.IsGraphOutput(reshape_output.name)));

  return Ort::Status();
}

}  // namespace

// ---------------------------------------------------------------------------
// IQnnNodeGroup interface implementation
// ---------------------------------------------------------------------------

/*static*/
std::unique_ptr<IQnnNodeGroup> GatherTransposeReshapeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& gather_node_unit,
    const MapNodeToNodeUnit& node_to_node_unit,
    const MapNodeUnitToGroup& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // Match the structural pattern.
  std::optional<std::array<const OrtNodeUnit*, 3>> pattern = MatchPattern(
      qnn_model_wrapper, gather_node_unit,
      node_to_node_unit, node_unit_to_qnn_node_group, logger);
  if (!pattern.has_value()) {
    return nullptr;
  }

  const OrtNodeUnit* gather = pattern->at(0);
  const OrtNodeUnit* transpose = pattern->at(1);
  const OrtNodeUnit* reshape = pattern->at(2);

  // Validate conditions and compute fusion parameters.
  GatherIndicesPattern index_pattern = GatherIndicesPattern::kRowMajor;
  int64_t idx0 = 0;
  int64_t idx1 = 0;
  std::vector<uint32_t> new_perm_4d;

  if (!ValidateAndComputeFusionParams(
          *gather, *transpose, *reshape,
          qnn_model_wrapper, logger,
          index_pattern, idx0, idx1, new_perm_4d)) {
    return nullptr;
  }

  // Validate that the fused sequence is supported by QNN.
  if (!CreateOrValidateOnQnn(
           qnn_model_wrapper, pattern.value(),
           index_pattern, idx0, idx1, new_perm_4d,
           /*validate=*/true, logger)
           .IsOK()) {
    return nullptr;
  }

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              ("GatherTransposeReshapeFusion: fusing Gather+Transpose+Reshape "
               "(rank-6 tensors) -> rank-4 ops for Gather node '" +
               gather_node_unit.Name() + "'")
                  .c_str());

  return std::make_unique<GatherTransposeReshapeFusion>(
      pattern.value(), index_pattern, idx0, idx1, std::move(new_perm_4d));
}

gsl::span<const OrtNodeUnit* const> GatherTransposeReshapeFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>{node_units_.data(), node_units_.size()};
}

Ort::Status GatherTransposeReshapeFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper,
    [[maybe_unused]] const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(
      qnn_model_wrapper, GetNodeUnits(),
      index_pattern_, idx0_, idx1_, new_perm_4d_,
      /*validate=*/true, logger);
}

Ort::Status GatherTransposeReshapeFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper,
    const Ort::Logger& logger) const {
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
              ("GatherTransposeReshapeFusion: building fused rank-4 ops "
               "(replacing rank-6 Gather+Transpose+Reshape) for Gather node '" +
               node_units_[0]->Name() + "'")
                  .c_str());
  return CreateOrValidateOnQnn(
      qnn_model_wrapper, GetNodeUnits(),
      index_pattern_, idx0_, idx1_, new_perm_4d_,
      /*validate=*/false, logger);
}

}  // namespace qnn
}  // namespace onnxruntime
