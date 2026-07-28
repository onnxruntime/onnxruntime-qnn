// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// ReciprocalMulFusion: Fuses SingleNode Reciprocal->Mul into ElementWiseDivide.

#pragma once

#include <array>
#include <memory>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// Fuses Reciprocal->Mul into ElementWiseDivide (SingleNode only to preserve quantization).
class ReciprocalMulFusion : public IQnnNodeGroup {
 public:
  ReciprocalMulFusion(const OrtNodeUnit& reciprocal_node_unit, const OrtNodeUnit& mul_node_unit,
                      bool recip_is_mul_input0);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ReciprocalMulFusion);

  // IQnnNodeGroup interface
  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "ReciprocalMulFusion"; }

  // Factory
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reciprocal_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::array<const OrtNodeUnit*, 2> node_units_;  // [0]=Reciprocal, [1]=Mul
  bool recip_is_mul_input0_{false};               // Which Mul input slot carries Reciprocal output.
};

}  // namespace qnn
}  // namespace onnxruntime
