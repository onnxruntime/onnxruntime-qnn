// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModelWrapper;

/// <summary>
/// Fuses a decomposed LayerNorm pattern into a single QNN LayerNorm operator.
///
///                    +--------------------------------------------+
///                    |                                            |
///                    v                                            |
///   [x] --> ReduceMean --> Sub --> Pow(2) --> ReduceMean --> Add(eps) --> Sqrt --> Div --> Mul(gamma) --> Add(beta) ==>
///                                  |                                               ^
///                                  |                                               |
///                                  +-----------------------------------------------+
///
/// All NodeUnits must be of type SingleNode.
/// </summary>
class LayerNormFusion : public IQnnNodeGroup {
 public:
  LayerNormFusion(std::vector<const OrtNodeUnit*>&& node_units,
                  const OrtNodeUnit* target_node_unit,
                  float epsilon,
                  std::vector<uint32_t> axes,
                  std::string gamma_input_name,
                  std::string beta_input_name);
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(LayerNormFusion);

  Ort::Status IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  Ort::Status AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const override;
  gsl::span<const OrtNodeUnit* const> GetNodeUnits() const override;
  const OrtNodeUnit* GetTargetNodeUnit() const override;
  std::string_view Type() const override { return "LayerNormFusion"; }

  static std::unique_ptr<IQnnNodeGroup> TryFusion(
      QnnModelWrapper& qnn_model_wrapper,
      const OrtNodeUnit& reduce_mean_node_unit,
      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
      const Ort::Logger& logger);

 private:
  std::vector<const OrtNodeUnit*> node_units_;
  const OrtNodeUnit* target_node_unit_;
  float epsilon_;
  std::vector<uint32_t> axes_;
  std::string gamma_input_name_;
  std::string beta_input_name_;
};

}  // namespace qnn
}  // namespace onnxruntime
