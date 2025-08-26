// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/mul_add_fusion.h"
#include "core/optimizer/utils.h"
#include "core/common/logging/logging.h"
#include "core/framework/data_types.h"
#include "core/framework/tensorprotoutils.h" // For utilities like TensorProtoToMLFloat16 etc.

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

bool MulAddFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  LOGS(logger, VERBOSE) << "Start SatisfyCondition";
  auto& mul_node = node;
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
      mul_node.GetOutputEdgesCount() != 1) {
    return false;
  }
  const auto& add_node = *mul_node.OutputNodesBegin();
  // Make sure the two nodes do not span execution providers.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7}) ||
      (add_node.GetExecutionProviderType() != mul_node.GetExecutionProviderType())) {
    return false;
  }
  // Pattern: Input -> Mul -> Add
  if (mul_node.InputDefs().size() != 2 || add_node.InputDefs().size() != 2) {
    return false;
  }

  // Get the second input of Mul (scale) and Add (bias)
  // These must be initializer tensors (constants)
  if (!graph_utils::NodeArgIsConstant(graph, *mul_node.InputDefs()[1]) ||
      !graph_utils::NodeArgIsConstant(graph, *add_node.InputDefs()[1])) {
    return false;
  }

  return true;
}

Status MulAddFusion::FuseMulAdd(Node& node, Graph& graph, bool& modified, const logging::Logger& logger) const {
  LOGS(logger, VERBOSE) << "Start FuseMulAdd";
  auto& mul_node = node;
  Node& add_node = *graph.GetNode(mul_node.OutputNodesBegin()->Index());
  // Before layout transform, channel is the 1st dimension
  int64_t num_channel = mul_node.InputDefs()[0]->Shape()->dim(1).dim_value();

  // Process scale and bias. Should be {num_channel}
  const auto* scale_tensor_proto = graph_utils::GetConstantInitializer(graph, mul_node.InputDefs()[1]->Name());
  const auto* bias_tensor_proto = graph_utils::GetConstantInitializer(graph, add_node.InputDefs()[1]->Name());
  ORT_ENFORCE(scale_tensor_proto);
  ORT_ENFORCE(bias_tensor_proto);
  ONNX_NAMESPACE::TensorProto reshaped_scale_proto = *scale_tensor_proto;
  ONNX_NAMESPACE::TensorProto reshaped_bias_tensor_proto = *bias_tensor_proto;
  reshaped_scale_proto.clear_dims();
  reshaped_scale_proto.set_name(scale_tensor_proto->name()+ "_reshaped");
  reshaped_scale_proto.add_dims(num_channel);
  reshaped_bias_tensor_proto.clear_dims();
  reshaped_bias_tensor_proto.set_name(bias_tensor_proto->name()+ "_reshaped");
  reshaped_bias_tensor_proto.add_dims(num_channel);
  NodeArg& reshaped_scale_node_arg = graph_utils::AddInitializer(graph, reshaped_scale_proto);
  NodeArg& reshaped_bias_node_arg = graph_utils::AddInitializer(graph, reshaped_bias_tensor_proto);

  // Initializer scale_init{graph, *scale_tensor_proto, graph.ModelPath()};
  // Initializer bias_init{graph, *bias_tensor_proto, graph.ModelPath()};
  Initializer mean_init(
    static_cast<ONNX_NAMESPACE::TensorProto_DataType>(mul_node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type()),
    graph.GenerateNodeArgName(mul_node.Name() + "_mul_add_fusion_mean"),
    gsl::span<const int64_t>({num_channel})
  );
  ONNX_NAMESPACE::TensorProto mean_tensor_proto;
  mean_init.ToProto(mean_tensor_proto);
  NodeArg& mean_init_node_arg = graph_utils::AddInitializer(graph, mean_tensor_proto);

  Initializer var_init(
    static_cast<ONNX_NAMESPACE::TensorProto_DataType>(mul_node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type()),
    graph.GenerateNodeArgName(add_node.Name() + "_mul_add_fusion_var"),
    gsl::span<const int64_t>({num_channel})
  );
  var_init.add(1);
  ONNX_NAMESPACE::TensorProto var_tensor_proto;
  var_init.ToProto(var_tensor_proto);
  NodeArg& var_init_node_arg = graph_utils::AddInitializer(graph, var_tensor_proto);


  // TODO: Finish the batchnorm node
  Node& bn_node = graph.AddNode(
    graph.GenerateNodeName(mul_node.Name() + "/MulAddFusion"), 
    "BatchNormalization",
    "fused Mul and Add",
    gsl::span<NodeArg* const>({
      mul_node.MutableInputDefs()[0],
      &reshaped_scale_node_arg,
      &reshaped_bias_node_arg,
      &mean_init_node_arg,
      &var_init_node_arg}),
    gsl::span<NodeArg* const>({
      add_node.MutableOutputDefs()[0]}),
      nullptr,
      kOnnxDomainAlias
  );
  bn_node.SetExecutionProviderType(mul_node.GetExecutionProviderType());
  constexpr float eps = 0.0f;
  constexpr float momentum = 0.0f;
  // constexpr int training_mode = 0;
  bn_node.SetSinceVersion(9);
  bn_node.AddAttribute("epsilon", eps);
  bn_node.AddAttribute("momentum", momentum);
  // bn_node.AddAttribute("training_mode", static_cast<int64_t>(training_mode));
  LOGS(logger, VERBOSE) << "bn_node.SinceVersion()" << bn_node.SinceVersion();
  LOGS(logger, VERBOSE) << "bn_node.Domain()" << bn_node.Domain();

  auto mul_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(mul_node);
  auto add_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(add_node);
  // TODO: Handle the case that constant is not on the 0'th position
  LOGS(logger, VERBOSE) << "AddEdge mul_input_edges[0]";
  graph.AddEdge(mul_input_edges[0].src_node, bn_node.Index(), mul_input_edges[0].src_arg_index, 0);
  // LOGS(logger, VERBOSE) << "AddEdge mul_input_edges[1]";
  // graph.AddEdge(mul_input_edges[1].src_node, bn_node.Index(), mul_input_edges[1].src_arg_index, 1);
  // LOGS(logger, VERBOSE) << "AddEdge add_input_edges[1]";
  // graph.AddEdge(add_input_edges[1].src_node, bn_node.Index(), add_input_edges[1].src_arg_index, 2);

  graph_utils::GraphEdge::RemoveGraphEdges(graph, mul_input_edges);
  graph_utils::GraphEdge::RemoveGraphEdges(graph, add_input_edges);
  graph_utils::RemoveNodeOutputEdges(graph, add_node);
  graph_utils::ReplaceDownstreamNodeInput(graph, add_node, 0, bn_node, 0);
  graph.RemoveNode(mul_node.Index());
  graph.RemoveNode(add_node.Index());

  modified = true;
  return Status::OK();
}

Status MulAddFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  const GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (this->SatisfyCondition(graph, node, logger)) {
      ORT_RETURN_IF_ERROR(this->FuseMulAdd(node, graph, modified, logger));
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
