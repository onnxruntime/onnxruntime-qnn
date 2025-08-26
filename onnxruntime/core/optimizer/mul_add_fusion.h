// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once


#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MulAddFusion

Rewrite rule that fuses two Mul+Add nodes to a single Batchnorm node.

It is attempted to be triggered only on nodes with op type "Mul".
*/
class MulAddFusion : public GraphTransformer {
 public:
  MulAddFusion() noexcept : GraphTransformer("MulAddFusion") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const;
  Status FuseMulAdd(Node& node, Graph& graph, bool& modified, const logging::Logger& logger) const;
};

}  // namespace onnxruntime
