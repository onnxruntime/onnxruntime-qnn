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
/// Fuses the dynamic-quantize MatMulInteger pattern into a float QNN MatMul.
///
/// Pattern (ONNX):
///   x --> DynamicQuantizeLinear --> (a_q, a_scale, a_zp)
///         a_q, B, a_zp, B_zp           --> MatMulInteger --> Cast(FLOAT)
///         a_scale, B_scale_init        --> parallel_Mul
///         Cast.out, parallel_Mul.out   --> requant_Mul
///         requant_Mul.out, bias_init   --> Add   (optional)
///
/// Rewrite (QNN). The pre-DQL float input feeds MatMul's first input; the int8 / uint8 weight
/// is either dequantized by a QNN Dequantize op (per-tensor B_scale) or pre-dequantized to
/// float offline and emitted as a STATIC float weight (per-channel B_scale: QNN's Dequantize
/// does not accept per-channel quantized inputs). MatMul is layout-agnostic, so no Transpose
/// ops are inserted around it.
///
///   x ---------------------------------+
///                                      |   (input[0] of MatMul)
///                                      v
///   B --> [Dequantize(B_scale,B_zp)] --> MatMul --> [Add(bias)] --> out
///
/// DynamicQuantizeLinear is mandatory in the pattern. When N MatMulIntegers share one DQL the
/// first sibling fusion claims DQL on behalf of all of them; subsequent siblings detect the
/// existing claim and skip the double-claim.
/// </summary>
class DQMatMulIntegerFusion : public IQnnNodeGroup {
 public:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(DQMatMulIntegerFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return matmul_integer_; }
  std::string_view Type() const override { return "DQMatMulIntegerFusion"; }

  /// <summary>
  /// Tries to match the DQL/MatMulInteger/Cast/Mul/Mul/[Add] pattern starting at a
  /// MatMulInteger. Returns the fused IQnnNodeGroup on success, nullptr otherwise.
  /// </summary>
  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& matmul_integer_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  // Aggregates everything TryFusion needs to hand off to the constructor.
  struct Pattern {
    const OrtNodeUnit* dql;  // nullptr if a sibling fusion already claimed it
    const OrtNodeUnit* matmul_integer;
    const OrtNodeUnit* cast;
    const OrtNodeUnit* parallel_mul;
    const OrtNodeUnit* requant_mul;
    const OrtNodeUnit* add_bias;            // nullptr if no trailing Add
    std::string float_input_name;           // pre-DQL float input feeds MatMul's input[0]
    const OrtNodeUnitIODef* b_scale_iodef;  // B_scale initializer (from parallel_Mul)
    std::string terminator_output_name;
    std::string bias_name;  // empty if no trailing Add
    bool has_b_zp;          // true if MatMulInteger has a B_zp input
  };

  explicit DQMatMulIntegerFusion(Pattern pattern);

  Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate) const;

  std::vector<const OrtNodeUnit*> node_units_;  // nodes claimed by this fusion (for ORT bookkeeping)
  const OrtNodeUnit* matmul_integer_;           // target node, also used for attrs/inputs
  const OrtNodeUnit* requant_mul_;              // used to read terminator output shape
  const OrtNodeUnit* add_bias_;                 // nullptr if no trailing Add

  std::string float_input_name_;
  const OrtNodeUnitIODef* b_scale_iodef_;
  std::string terminator_output_name_;
  std::string bias_name_;
  bool has_b_zp_;
};

}  // namespace qnn
}  // namespace onnxruntime
