// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class WhereDummyDq

    Rewrite rule that insert a dummy DQ on Where node's initializer input
    when Where node has one DQ and one one scalar initializer input
*/
class WhereDummyDq : public RewriteRule {
 public:
  WhereDummyDq() noexcept : RewriteRule("WhereDummyDq") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Where"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
};
}  // namespace onnxruntime