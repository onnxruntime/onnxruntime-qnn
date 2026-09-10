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
/// Fixes an unsigned/signed 8-bit mismatch between a QDQ LayerNormalization's X input and its
/// static scale (and, when also 8-bit and mismatched, its static bias) so the node can run as a
/// native QNN LayerNorm on HTP.
///
/// QNN's HTP LayerNorm validator only supports these INT8 combinations:
///   UFIXED_POINT_8 / UFIXED_POINT_8 / UFIXED_POINT_8  -> UFIXED_POINT_8
///   UFIXED_POINT_8 / UFIXED_POINT_8 / SFIXED_POINT_32 -> UFIXED_POINT_8
///   SFIXED_POINT_8 / SFIXED_POINT_8 / SFIXED_POINT_32 -> SFIXED_POINT_8
/// None of them mix an UFIXED_POINT_8 operand with a SFIXED_POINT_8 operand. Standard QDQ
/// quantization tooling commonly produces exactly that mix -- an asymmetric unsigned X (uint8)
/// with a symmetric signed scale (int8) -- which fails QNN validation and falls back to CPU EP
/// for the whole node, even though the underlying math is fully representable in 8 bits.
///
/// This fusion requantizes the mismatched static operand(s) to match X's family via an exact,
/// lossless zero-point shift (+/-128): for a per-tensor 8-bit quantized tensor, flipping between
/// SFIXED_POINT_8 and UFIXED_POINT_8 while shifting the zero-point by the same amount preserves
/// the dequantized value exactly (no rounding, no clipping) -- it is a pure bit-pattern relabeling.
///
///   x_q --> [DQ] --> LayerNormalization(x, scale_signed_int8, [bias]) --> [Q] --> y_q
///                                        |          |
///                                        v          v
///                              scale_resigned   bias_resigned (if also 8-bit and mismatched)
/// </summary>
class DQLayerNormFusion : public IQnnNodeGroup {
 public:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(DQLayerNormFusion);

  // Used as Type().
  static constexpr std::string_view kType = "DQLayerNormFusion";

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return node_unit_; }
  std::string_view Type() const override { return kType; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& layer_norm_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  explicit DQLayerNormFusion(const OrtNodeUnit& node_unit);

  Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate) const;

  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* node_unit_;
};

}  // namespace qnn
}  // namespace onnxruntime
