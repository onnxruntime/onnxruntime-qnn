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
/// Fuses the dynamic-quantize ConvInteger pattern into a float QNN Conv2d.
///
/// Pattern (ONNX):
///   x --> DynamicQuantizeLinear --> (a_q, a_scale, a_zp)
///         a_q, B_int8, a_zp, B_zp_int8 --> ConvInteger --> Cast(FLOAT)
///         a_scale, B_scale_init        --> parallel_Mul
///         Cast.out, parallel_Mul.out   --> requant_Mul
///         requant_Mul.out, bias_init   --> Add   (optional)
///
/// Rewrite (QNN). The pre-DQL float input feeds Conv2d's activation; the int8 weight is
/// transposed NCHW->HWCN and either dequantized by a QNN Dequantize op (per-tensor B_scale)
/// or pre-dequantized to float offline and emitted as a STATIC float weight (per-channel
/// B_scale: QNN's Dequantize does not accept per-channel quantized inputs).
///
///   x --> Transpose(NCHW->NHWC) -----------------+
///                                                |   (activation input to Conv2d)
///                                                v
///   B --> (NCHW->HWCN bytes) --> [Dequantize] --> Conv2d --> Transpose(NHWC->NCHW) --> [Add(bias)] --> out
///
/// DynamicQuantizeLinear is mandatory in the pattern. When N ConvIntegers share one DQL the
/// first sibling fusion claims DQL on behalf of all of them; subsequent siblings detect the
/// existing claim and skip the double-claim.
/// </summary>
class DQConvIntegerFusion : public IQnnNodeGroup {
 public:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(DQConvIntegerFusion);

  // Used as Type() and to detect sibling claims on a shared DQL (see TryFusion).
  static constexpr std::string_view kType = "DQConvIntegerFusion";

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override { return conv_integer_; }
  std::string_view Type() const override { return kType; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& conv_integer_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  struct Pattern {
    const OrtNodeUnit* dql;  // nullptr if a sibling fusion already claimed it
    const OrtNodeUnit* conv_integer;
    const OrtNodeUnit* cast;
    const OrtNodeUnit* parallel_mul;
    const OrtNodeUnit* requant_mul;
    const OrtNodeUnit* add_bias;            // nullptr if no trailing Add
    std::string float_input_name;           // pre-DQL float input feeds Conv's activation
    const OrtNodeUnitIODef* b_scale_iodef;  // B_scale initializer (from parallel_Mul)
    std::string terminator_output_name;
    bool has_b_zp;
  };

  explicit DQConvIntegerFusion(Pattern pattern);

  Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qmw, bool validate) const;

  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* conv_integer_;
  const OrtNodeUnit* requant_mul_;
  const OrtNodeUnit* add_bias_;  // nullptr if no trailing Add

  std::string float_input_name_;
  const OrtNodeUnitIODef* b_scale_iodef_;
  std::string terminator_output_name_;
  bool has_b_zp_;
};

}  // namespace qnn
}  // namespace onnxruntime
