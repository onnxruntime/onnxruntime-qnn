// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Implements the IQnnNodeGroup interface for the ONNX QLinearMatMul operator.
///
/// QLinearMatMul carries quantization parameters as explicit inputs (a_scale, a_zero_point,
/// b_scale, b_zero_point, y_scale, y_zero_point). QNN encodes quant params as tensor metadata,
/// so this node group reads the scale/zp initializers, builds QnnQuantParamsWrapper objects,
/// and attaches them to the QNN tensors for A, B, and Y. The remaining shape handling
/// (rank-1 reshapes, FullyConnected dispatch) mirrors the float MatMul path.
///
/// Mapped QNN ops:
///   QNN_OP_MAT_MUL         -- general batched matrix multiply
///   QNN_OP_FULLY_CONNECTED -- used when B is a rank-2 static initializer (or rank-1)
/// </summary>
class QLinearMatMulNodeGroup : public IQnnNodeGroup {
 public:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(QLinearMatMulNodeGroup);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_unit_; }
  std::string_view Type() const override { return "QLinearMatMulNodeGroup"; }

  /// <summary>
  /// Tries to claim the given SingleNode QLinearMatMul node unit. Returns a valid
  /// IQnnNodeGroup on success, nullptr otherwise (unsupported shape, dynamic quant params, etc.).
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  explicit QLinearMatMulNodeGroup(const OrtNodeUnit& node_unit);

  Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate,
                                    const Ort::Logger& logger) const;

  const OrtNodeUnit* node_unit_;
};

}  // namespace qnn
}  // namespace onnxruntime
