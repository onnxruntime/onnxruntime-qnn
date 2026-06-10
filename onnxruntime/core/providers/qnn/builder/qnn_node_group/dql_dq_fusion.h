// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <gsl/gsl>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Fuses the DynamicQuantizeLinear + DequantizeLinear pair into a QNN identity Transpose.
///
/// Pattern (ONNX):
///   x (float32) --> DynamicQuantizeLinear --> (y_uint8, y_scale, y_zp)
///                   ALL 3 outputs exclusively --> DequantizeLinear --> x_approx (float32)
///
/// Condition: all three DQL outputs must feed exclusively into the same DequantizeLinear
/// (including y_zp — DQ must not omit zero_point, since DQL's y_zp is dynamic and typically
/// non-zero; allowing DQ to default to zero_point=0 would break the identity round-trip), and
/// the DequantizeLinear output must be float32.  This is the "fake-quantize" identity pattern:
/// the round-trip DQL -> DQ leaves x_approx numerically close to x (quantized then restored to
/// float), so the entire sub-graph can be bypassed on QNN without meaningful accuracy loss.
///
/// Rewrite (QNN):
///   x --> Transpose(identity permutation) --> x_approx
///
/// QNN has no native Identity op; an identity-permutation Transpose compiles to zero cycles on
/// HTP.
/// </summary>
class DqlDqFusion : public IQnnNodeGroup {
 public:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(DqlDqFusion);

  static constexpr std::string_view kType = "DqlDqFusion";

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return dql_; }
  std::string_view Type() const override { return kType; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& dql_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  DqlDqFusion(const OrtNodeUnit* dql, const OrtNodeUnit* dq,
              std::string float_input_name, std::string float_output_name);

  Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate) const;

  const OrtNodeUnit* dql_;
  std::array<const OrtNodeUnit*, 2> node_units_;
  std::string float_input_name_;   // DQL's input[0]: the original float activation
  std::string float_output_name_;  // DQ's output[0]: passed through to downstream ops
};

}  // namespace qnn
}  // namespace onnxruntime
