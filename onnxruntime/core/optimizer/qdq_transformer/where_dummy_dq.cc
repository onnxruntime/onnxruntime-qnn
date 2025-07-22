// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/where_dummy_dq.h"

#include "core/framework/tensorprotoutils.h"
#include "core/common/common.h"
#include "core/util/qmath.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {
bool WhereDummyDq::SatisfyCondition(const Graph& graph, const Node& node) const {
  if (!(node.OpType() == "Where")) {
    return false;
  }
  const auto& where_inputs = node.InputDefs();
  const Node* parent_node_1 = graph.GetProducerNode(where_inputs[1]->Name());
  const Node* parent_node_2 = graph.GetProducerNode(where_inputs[2]->Name());

  bool is_p1_dq = (parent_node_1 && parent_node_1->OpType() == QDQ::DQOpName);
  bool is_p2_dq = (parent_node_2 && parent_node_2->OpType() == QDQ::DQOpName);

  // WhereDummyDq focus on WhereOp with one DQ input and one scalar initializer input
  if (is_p1_dq && !parent_node_2) {
    return (where_inputs[2]->Shape()->dim_size() == 0);
  }
  if (!parent_node_1 && is_p2_dq) {
    return (where_inputs[1]->Shape()->dim_size() == 0);
  }
  return false;
}

Status WhereDummyDq::InsertDummyDQ(Node& node, Graph& graph, bool& modified) const {
  const auto& where_inputs = node.InputDefs();
  const Node* parent_node_1 = graph.GetProducerNode(where_inputs[1]->Name());
  const Node* parent_node_2 = graph.GetProducerNode(where_inputs[2]->Name());

  // With SatisfyCondition, we must have one DQ and one initializer
  const Node* dq_node = parent_node_1 ? parent_node_1 : parent_node_2;
  int const_idx = parent_node_1 ? 2 : 1;

  // Dummy data initializer.
  const ONNX_NAMESPACE::TensorProto* dq_node_scale_proto = nullptr;
  graph.GetInitializedTensor(dq_node->InputDefs()[1]->Name(), dq_node_scale_proto);
  const ONNX_NAMESPACE::TensorProto* dq_node_zp_proto = nullptr;
  graph.GetInitializedTensor(dq_node->InputDefs()[2]->Name(), dq_node_zp_proto);

  ONNX_NAMESPACE::TensorProto dummy_data_proto;
  int dummy_data = 1;
  dummy_data_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_data"));
  // Set data type to the one of const_node dq's zp dtype
  dummy_data_proto.set_data_type(dq_node_zp_proto->data_type());
  dummy_data_proto.add_int32_data(dummy_data);
  NodeArg& dummy_data_arg = graph_utils::AddInitializerWithExternalData(graph, dummy_data_proto);

  // Dummy scale initializer.
  const ONNX_NAMESPACE::TensorProto* const_node_data_proto = nullptr;
  graph.GetInitializedTensor(where_inputs[const_idx]->Name(), const_node_data_proto);

  ONNX_NAMESPACE::TensorProto dummy_scale_proto;
  // Set scale to the original value
  Initializer initializer(graph, *const_node_data_proto, graph.ModelPath());
  // Use float to represent the original data value
  const float* where_const_data = initializer.data<float>();
  float scale = *where_const_data;
  dummy_scale_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_scale"));
  dummy_scale_proto.set_data_type(dq_node_scale_proto->data_type());
  dummy_scale_proto.add_float_data(scale);
  NodeArg& dummy_scale_arg = graph_utils::AddInitializerWithExternalData(graph, dummy_scale_proto);

  // Dummy zero point initializer.
  int zp = 0;
  ONNX_NAMESPACE::TensorProto dummy_zp_proto;
  dummy_zp_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_zp"));
  dummy_zp_proto.set_data_type(dq_node_zp_proto->data_type());
  dummy_zp_proto.add_int32_data(static_cast<int32_t>(zp));
  NodeArg& dummy_zp_arg = graph_utils::AddInitializerWithExternalData(graph, dummy_zp_proto);

  ONNX_NAMESPACE::TypeProto dummy_dq_type_proto = utils::TypeProtoFromTensorProto(*const_node_data_proto);
  dummy_dq_type_proto.mutable_tensor_type()->set_elem_type(const_node_data_proto->data_type());
  NodeArg& dummy_dq_arg =
      graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_dummy_dq"), &dummy_dq_type_proto);
  Node& dummy_dq_node =
      graph.AddNode(
          graph.GenerateNodeArgName(node.Name() + "_dummy_dq"),
          QDQ::DQOpName,
          "DeQuantizeLinear from WhereDummyDq GraphTransformer",
          {&dummy_data_arg, &dummy_scale_arg, &dummy_zp_arg},
          {&dummy_dq_arg},
          nullptr,
          dq_node->Domain());

  node.MutableInputDefs()[const_idx] = &dummy_dq_arg;
  if (graph.GetConsumerNodes(where_inputs[const_idx]->Name()).size() == 0) {
    graph.RemoveInitializedTensor(where_inputs[const_idx]->Name());
  }
  graph.AddEdge(dummy_dq_node.Index(), node.Index(), 0, const_idx);
  modified = true;

  return Status::OK();
}

Status WhereDummyDq::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  const GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (this->SatisfyCondition(graph, node)) {
      ORT_RETURN_IF_ERROR(WhereDummyDq::InsertDummyDQ(node, graph, modified));
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime