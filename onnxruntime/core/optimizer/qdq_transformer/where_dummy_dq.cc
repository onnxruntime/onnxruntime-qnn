// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/where_dummy_dq.h"

#include "core/common/common.h"
#include "core/util/qmath.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {
    bool WhereDummyDq::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
        return true;
    }

    Status WhereDummyDq::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
        std::cout << "Doing WhereDummyDq" << std::endl;
        auto& where_node = node;
        const auto& where_inputs = where_node.InputDefs();
        const NodeArg* where_in0 = where_inputs[0];
        const NodeArg* where_in1 = where_inputs[1];
        const NodeArg* where_in2 = where_inputs[2];
        const Node* parent_node_0 = graph.GetProducerNode(where_in0->Name());
        const Node* parent_node_1 = graph.GetProducerNode(where_in1->Name());
        const Node* parent_node_2 = graph.GetProducerNode(where_in2->Name());
        bool is_with_qdq = false;
        if ((parent_node_0 && parent_node_0->OpType() == QDQ::DQOpName) ||
            (parent_node_1 && parent_node_1->OpType() == QDQ::DQOpName) ||
            (parent_node_2 && parent_node_2->OpType() == QDQ::DQOpName)) {
            is_with_qdq = true;
        }
        if (!is_with_qdq) {
            return Status::OK();
        }
        if (!parent_node_1) {
            std::cout << "insert DQ for parent_node_1" << std::endl;
            ONNX_NAMESPACE::TensorProto dummy_data_proto;
            int dummy_data = 1;
            dummy_data_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_data"));
            // TODO: Set data type to the one of parent_node_2
            dummy_data_proto.set_data_type(onnx::TensorProto_DataType_UINT16);
            dummy_data_proto.add_int32_data(dummy_data);
            NodeArg& dummy_data_arg = graph_utils::AddInitializerWithExternalData(graph, dummy_data_proto);
            
            // Dummy scale initializer.
            ONNX_NAMESPACE::TensorProto dummy_scale_proto;
            // TODO: Set scale to the original value
            float scale = 1.0;
            dummy_scale_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_scale"));
            dummy_scale_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            dummy_scale_proto.add_float_data(scale);
            NodeArg& dummy_scale_arg = graph_utils::AddInitializerWithExternalData(graph, dummy_scale_proto);

            std::cout << "scale: " << scale << std::endl;

            // Dummy zero point initializer.
            int zp = 0;
            ONNX_NAMESPACE::TensorProto dummy_zp_proto;
            dummy_zp_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_zp"));
            dummy_zp_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT16);
            dummy_zp_proto.add_int32_data(static_cast<int32_t>(zp));
            NodeArg& dummy_zp_arg = graph_utils::AddInitializerWithExternalData(graph, dummy_zp_proto);

            std::cout << "zp: " << zp << std::endl;

            ONNX_NAMESPACE::TypeProto dummy_dq_type_proto;
            dummy_dq_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            *dummy_dq_type_proto.mutable_tensor_type()->mutable_shape() = *where_in1->Shape();
            NodeArg& dummy_dq_arg =
                graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_dummy_dq"), &dummy_dq_type_proto);
            Node& dummy_dq_node =
                graph.AddNode(
                    graph.GenerateNodeArgName(node.Name() + "_dummy_dq"),
                    QDQ::DQOpName,
                    "Dummy DQ node",
                    {&dummy_data_arg, &dummy_scale_arg, &dummy_zp_arg},
                    {&dummy_dq_arg},
                    nullptr,
                    "com.microsoft"
                );
            //TODO: Use the domain of another dq node
            where_node.MutableInputDefs()[1] = &dummy_dq_arg;
            // TODO: Whether to remove original constant
            graph.AddEdge(dummy_dq_node.Index(), where_node.Index(), 0, 1);
        }
        if (!parent_node_2) {
            // TODO: insert DQ for parent_node_2
            std::cout << "insert DQ for parent_node_2" << std::endl;
        }
        return Status::OK();
    }
}