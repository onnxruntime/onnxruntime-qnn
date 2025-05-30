// Copyright (c) Qualcomm Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/affine_elimination.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

/**
  Case to eliminate Affine node with alpha=1.0f and beta=0.0f when
  - the input nodearg has only one consumer, which is the Affine itself
  - the input def is not a graph output

  For examples:

  OK to eliminate:

    Affine output is another node, and the Affine is the only consumer of X
      X ---> Affine ---> Y where Y could be graph output

    Affine input arg is not shared with other output arg of X
      + (arg0) ---> Affine0 ---> Z
      |
      X (arg1) ---> Affine1 ---> Y

  Not OK to eliminate:

    Affine input arg, i.e., arg0, is also an input arg of other Affine
      + (arg0) ---> Affine0 ---> Z
      |
      X (arg0) ---> Affine1 ---> Y

    Affine input def, i.e., def0, is also a graph output
      + (def0) ---> Z where Z is graph output
      |
      X (def0/arg0) ---> Affine ---> Y
 */
Status AffineElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  if (!graph.NodeProducesGraphOutput(node)) {
    if (graph_utils::RemoveNode(graph, node)) {
      rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
    }
  } else {
    // keep a reference of output def to the graph output
    NodeArg* output = node.MutableOutputDefs()[0];
    const Node* p_input_node = graph_utils::GetInputNode(node, 0);
    // get mutable input node
    Node& input_node = *graph.GetNode(p_input_node->Index());
    int output_idx = graph_utils::GetNodeOutputIndexFromOutputName(input_node, node.MutableInputDefs()[0]->Name());
    // remove Affine node and its input edge
    graph.RemoveNode(node.Index());
    // update input node's output def to the graph output
    input_node.MutableOutputDefs()[output_idx] = output;
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }
  return Status::OK();
}

bool AffineElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  // Affine with alpha = 1.0f and beta = 0.0f is an Identity
  if (!optimizer_utils::IsAttributeWithExpectedValue(node, "alpha", static_cast<float>(1.0f)) ||
      !optimizer_utils::IsAttributeWithExpectedValue(node, "beta", static_cast<float>(0.0f))) {
    return false;
  }

  if (graph_utils::CanRemoveNode(graph, node, logger)) {
    return true;
  }

  bool node_output_is_graph_output = graph.NodeProducesGraphOutput(node);

  // relax the condition if Affine is connecting to graph output
  if (node.GetOutputEdgesCount() != 0 ||
      node.OutputDefs().size() != 1 ||
      !node_output_is_graph_output) {
    return false;
  }

  const Node* p_input_node = graph_utils::GetInputNode(node, 0);
  if (p_input_node == nullptr) {
    return false;
  }
  if (p_input_node->OpType() == "YieldOp" && node_output_is_graph_output) {
    return false;
  }

  // skip if the src arg is also a graph output
  int src_arg_index = graph_utils::GetNodeOutputIndexFromOutputName(*p_input_node, node.InputDefs()[0]->Name());
  if (graph.IsOutput(p_input_node->OutputDefs()[src_arg_index]))
    return false;

  // count how many consumers are sharing the same src arg
  int count = 0;
  for (auto it = p_input_node->OutputEdgesBegin(), end = p_input_node->OutputEdgesEnd(); it != end; ++it) {
    if (it->GetSrcArgIndex() == src_arg_index) {
      count++;
    }
  }
  // condition not met if there are more than 1 consumer for the same src arg
  if (count > 1)
    return false;

  return true;
}

}  // namespace onnxruntime
