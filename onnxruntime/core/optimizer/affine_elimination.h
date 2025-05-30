// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class AffineElimination

Rewrite rule that eliminates Affine nodes if alpha = 1.0f and beta = 0.0f

It is attempted to be triggered only on nodes with op type "Affine".
*/
class AffineElimination : public RewriteRule {
 public:
  AffineElimination() noexcept : RewriteRule("AffineElimination") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Affine"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
