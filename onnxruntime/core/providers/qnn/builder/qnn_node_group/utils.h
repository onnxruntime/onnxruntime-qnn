// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
constexpr const char* QUANTIZE_LINEAR = "QuantizeLinear";
constexpr const char* DEQUANTIZE_LINEAR = "DequantizeLinear";
constexpr size_t QDQ_MAX_NUM_INPUTS = 3;
constexpr size_t QDQ_SCALE_INPUT_IDX = 1;
constexpr size_t QDQ_ZERO_POINT_INPUT_IDX = 2;

/// <summary>
/// Extracts and normalizes axes from a Reduce operator (ReduceMean, ReduceSum, etc.).
/// Handles both attribute-based axes (opset < 18) and input-based axes (opset >= 18).
/// Returns normalized positive axes, sorted and deduplicated.
/// </summary>
/// <param name="qnn_model_wrapper">QnnModelWrapper containing the OrtGraph and OrtApi</param>
/// <param name="node_unit">The Reduce operator node unit</param>
/// <returns>Normalized axes as uint32_t vector, or std::nullopt if extraction fails</returns>
std::optional<std::vector<uint32_t>> GetReduceAxes(const QnnModelWrapper& qnn_model_wrapper,
                                                   const OrtNodeUnit& node_unit);

/// <summary>
/// Utility function to get a child NodeUnit. The returned NodeUnit must be the parent's only child, must be
/// of the expected type, and must not be a part of another IQnnNodeGroup.
/// </summary>
/// <param name="qnn_model_wrapper">QnnModelWrapper containing the OrtGraph and OrtApi</param>
/// <param name="parent_node_unit">Parent NodeUnit</param>
/// <param name="child_op_types">Valid child types</param>
/// <param name="node_unit_map">Maps a Node to its NodeUnit</param>
/// <param name="node_unit_to_qnn_node_group">Maps a NodeUnit to its IQnnNodeGroup.
/// Used to check that the child has not already been added to another IQnnNodeGroup.</param>
/// <returns></returns>
const OrtNodeUnit* GetOnlyChildOfType(const QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& parent_node_unit,
                                      gsl::span<const std::string_view> child_op_types,
                                      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group);

/// <summary>
/// Utility function to get a child NodeUnit, allowing QDQ-wrapped nodes. The returned NodeUnit must be the parent's
/// only child, must be of the expected type, and must not be a part of another IQnnNodeGroup.
/// </summary>
const OrtNodeUnit* GetChildNodeUnitAllowQdq(const QnnModelWrapper& qnn_model_wrapper,
                                            const OrtNodeUnit& parent_node_unit,
                                            const std::string& child_op_type,
                                            const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                            const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group);

const OrtNodeUnit* GetParentOfType(const QnnModelWrapper& qnn_model_wrapper,
                                   const OrtNodeUnit& child_node_unit,
                                   gsl::span<const std::string_view> parent_op_types,
                                   const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                   const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group);

const OrtNodeUnit* GetParentOfInput(const QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnit& node_unit,
                                    const OrtNodeUnitIODef& input,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map);

const OrtNodeUnit* GetOnlyChildOfOutput(const QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const OrtNodeUnitIODef& output,
                                        const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                        const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map);

const OrtNodeUnit* GetParentOfInputByName(const QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const std::string& input_name,
                                          const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                          const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map);

/// <summary>
/// Reads a scalar constant initializer by tensor name and returns its value as float.
/// Supports float32 and float16 element types. Returns nullopt if the input is not a
/// constant, not a scalar (element count != 1), or has an unsupported element type.
/// </summary>
std::optional<float> GetScalarConstantValue(const QnnModelWrapper& qnn_model_wrapper,
                                            const std::string& input_name);

/// <summary>
/// Returns true if the named input is a scalar constant approximately equal to `expected`.
/// Tolerance is applied as abs(val - expected) <= tol.
/// Supports float32 and float16 element types.
/// tol defaults to 1e-3f (~2% relative error for the GELU cubic coefficient 0.044715).
/// </summary>
bool IsScalarConstantApprox(const QnnModelWrapper& qnn_model_wrapper,
                            const std::string& input_name,
                            float expected,
                            float tol = 1e-3f);

/// <summary>
/// Returns true when a Reshape from `input_shape` to `output_shape` is structurally
/// equivalent to a Transpose: both shapes have the same rank, all dims are static and
/// non-negative, and the sequence of non-1 dims is identical (same values in the same
/// order). Size-1 dims are free to appear anywhere.
///
/// Defined inline so consumers (including out-of-EP unit tests) do not need to link
/// against the QNN EP module to use it.
/// </summary>
inline bool IsReshapePermutable(const std::vector<int64_t>& input_shape,
                                const std::vector<int64_t>& output_shape) {
  if (input_shape.size() != output_shape.size()) {
    return false;
  }
  const auto is_negative = [](int64_t d) { return d < 0; };
  if (std::any_of(input_shape.begin(), input_shape.end(), is_negative) ||
      std::any_of(output_shape.begin(), output_shape.end(), is_negative)) {
    return false;
  }

  const auto is_non_one = [](int64_t d) { return d != 1; };
  std::vector<int64_t> input_non_one;
  std::vector<int64_t> output_non_one;
  std::copy_if(input_shape.begin(), input_shape.end(), std::back_inserter(input_non_one), is_non_one);
  std::copy_if(output_shape.begin(), output_shape.end(), std::back_inserter(output_non_one), is_non_one);
  return input_non_one == output_non_one;
}

/// <summary>
/// Precondition: IsReshapePermutable(input_shape, output_shape) is true. Fills `perm` so
/// that output_shape[i] == input_shape[perm[i]]. Non-1 dims are matched left-to-right
/// (preserving their relative order); size-1 dims greedily take the next unused slot.
///
/// Defined inline so consumers (including out-of-EP unit tests) do not need to link
/// against the QNN EP module to use it.
/// </summary>
inline void ComputeReshapePerm(const std::vector<int64_t>& input_shape,
                               const std::vector<int64_t>& output_shape,
                               std::vector<int64_t>& perm) {
  const size_t rank = input_shape.size();
  perm.assign(rank, -1);
  std::vector<bool> used(rank, false);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t target = output_shape[i];
    for (size_t j = 0; j < rank; ++j) {
      if (!used[j] && input_shape[j] == target) {
        perm[i] = static_cast<int64_t>(j);
        used[j] = true;
        break;
      }
    }
  }
}

}  // namespace qnn
}  // namespace onnxruntime
