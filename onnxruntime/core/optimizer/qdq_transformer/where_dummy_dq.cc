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
    bool WhereDummyDq::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
        // TODO: Finish the SatisfyCondition
        return true;
    }

    Status WhereDummyDq::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
        std::cout << "Doing WhereDummyDq" << std::endl;
        auto& where_node = node;
        const auto& where_inputs = where_node.InputDefs();
        const Node* parent_node_1 = graph.GetProducerNode(where_inputs[1]->Name());
        const Node* parent_node_2 = graph.GetProducerNode(where_inputs[2]->Name());
        bool is_with_qdq = false;

        const Node* dq_node = nullptr;
        int const_idx = -1;
        if (parent_node_1 && parent_node_1->OpType() == QDQ::DQOpName && !parent_node_2) {
            dq_node = parent_node_1;
            const_idx = 2;
        } else if (parent_node_2 && parent_node_2->OpType() == QDQ::DQOpName && !parent_node_1) {
            dq_node = parent_node_2;
            const_idx = 1;
        } else {
            return Status::OK();
        }

        const ONNX_NAMESPACE::TensorProto* const_node_data_proto = nullptr;
        graph.GetInitializedTensor(where_inputs[const_idx]->Name(), const_node_data_proto);
        const ONNX_NAMESPACE::TensorProto* dq_node_scale_proto = nullptr;
        graph.GetInitializedTensor(dq_node->InputDefs()[2]->Name(), dq_node_scale_proto);

        std::cout << "insert DQ for dq_node" << std::endl;
        ONNX_NAMESPACE::TensorProto dummy_data_proto;
        int dummy_data = 1;
        dummy_data_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_data"));
        // Set data type to the one of const_node dq's zp dtype
        dummy_data_proto.set_data_type(dq_node_scale_proto->data_type());
        dummy_data_proto.add_int32_data(dummy_data);
        NodeArg& dummy_data_arg = graph_utils::AddInitializerWithExternalData(graph, dummy_data_proto);
        
        // Dummy scale initializer.
        ONNX_NAMESPACE::TensorProto dummy_scale_proto;
        // Set scale to the original value
        Initializer initializer(graph, *const_node_data_proto, graph.ModelPath());
        const float* where_const_data = initializer.data<float>();
        float scale = *where_const_data;
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

        ONNX_NAMESPACE::TypeProto dummy_dq_type_proto = utils::TypeProtoFromTensorProto(*const_node_data_proto);
        dummy_dq_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        // *dummy_dq_type_proto.mutable_tensor_type()->mutable_shape() = *where_in1->Shape();
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
                dq_node->Domain()
            );

        std::cout << "Finish create dummy dq node" << std::endl;

        graph.RemoveInitializedTensor(where_inputs[const_idx]->Name());
        where_node.MutableInputDefs()[const_idx] = &dummy_dq_arg;
        // TODO: Whether to remove original constant
        graph.AddEdge(dummy_dq_node.Index(), where_node.Index(), 0, const_idx);

        return Status::OK();
    }
}