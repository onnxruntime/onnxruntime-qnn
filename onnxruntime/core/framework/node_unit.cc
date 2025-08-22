// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "node_unit.h"
#include <utility>
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

namespace {

enum class QLinearOpType : uint8_t {
  Unknown,  // Unknown or not a linear quantized op
  DequantizeLinear,
  QuantizeLinear,
  QLinearConv,
  QLinearMatMul,
  QLinearAdd,
  QLinearSigmoid,
  QLinearAveragePool,
  QLinearMul,
  QLinearReduceMean,
  QLinearConcat,
  QLinearGlobalAveragePool,
  QLinearLeakyRelu,
};

QLinearOpType GetQLinearOpType(const onnxruntime::Node& node) {
  const auto& op_type = node.OpType();
  if (op_type == "DequantizeLinear")
    return QLinearOpType::DequantizeLinear;
  else if (op_type == "QuantizeLinear")
    return QLinearOpType::QuantizeLinear;
  else if (op_type == "QLinearConv")
    return QLinearOpType::QLinearConv;
  else if (op_type == "QLinearMatMul")
    return QLinearOpType::QLinearMatMul;
  else if (op_type == "QLinearAdd")
    return QLinearOpType::QLinearAdd;
  else if (op_type == "QLinearSigmoid")
    return QLinearOpType::QLinearSigmoid;
  else if (op_type == "QLinearAveragePool")
    return QLinearOpType::QLinearAveragePool;
  else if (op_type == "QLinearMul")
    return QLinearOpType::QLinearMul;
  else if (op_type == "QLinearReduceMean")
    return QLinearOpType::QLinearReduceMean;
  else if (op_type == "QLinearConcat")
    return QLinearOpType::QLinearConcat;
  else if (op_type == "QLinearGlobalAveragePool")
    return QLinearOpType::QLinearGlobalAveragePool;
  else if (op_type == "QLinearLeakyRelu")
    return QLinearOpType::QLinearLeakyRelu;

  return QLinearOpType::Unknown;
}

// Ops have 1 input
bool IsUnaryQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearSigmoid ||
         type == QLinearOpType::QLinearAveragePool ||
         type == QLinearOpType::QLinearGlobalAveragePool ||
         type == QLinearOpType::QLinearLeakyRelu ||
         type == QLinearOpType::QLinearReduceMean;
}

// Ops have 2 inputs
bool IsBinaryQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearConv ||
         type == QLinearOpType::QLinearMatMul ||
         type == QLinearOpType::QLinearAdd ||
         type == QLinearOpType::QLinearMul;
}

// Ops have 1 or more inputs
bool IsVariadicQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearConcat;
}

const std::vector<const Node*> GetQDQIONodes(const GraphViewer& graph_viewer,
                                             const QDQ::NodeGroup& node_group, bool is_input) {
  std::vector<const Node*> io_nodes;
  const auto& src_nodes = is_input ? node_group.dq_nodes : node_group.q_nodes;
  io_nodes.reserve(src_nodes.size());
  for (const auto& node_idx : src_nodes) {
    io_nodes.push_back(graph_viewer.GetNode(node_idx));
  }

  return io_nodes;
}

// Get the input or output NodeUnitIODef(s) for the given QDQ NodeGroup
std::vector<NodeUnitIODef> GetQDQIODefs(const Node& target_node, const QDQ::NodeGroup& node_group, bool is_input) {
  const auto& dq_or_q_nodes = is_input ? node_group.dq_nodes : node_group.q_nodes;
  const auto target_node_io_defs = is_input ? target_node.InputDefs() : target_node.OutputDefs();
  const size_t target_node_io_defs_size = target_node_io_defs.size();

  // Find all the quantized IO defs and indices (for the input/output of the target node)
  std::unordered_map<size_t, NodeUnitIODef> quantized_io_defs;
  quantized_io_defs.reserve(target_node_io_defs_size);

  auto cur = is_input ? target_node.InputEdgesBegin() : target_node.OutputEdgesBegin();
  auto end = is_input ? target_node.InputEdgesEnd() : target_node.OutputEdgesEnd();

  for (; cur != end; ++cur) {
    const Node& node = cur->GetNode();

    // If we can find the node index in the dq or q nodes this is a quantized input/output
    if (std::find(dq_or_q_nodes.cbegin(), dq_or_q_nodes.cend(), node.Index()) != dq_or_q_nodes.cend()) {
      const auto node_inputs = node.InputDefs();
      const auto& node_attrs = node.GetAttributes();

      // Get the Q or DQ axis attribute if available.
      std::optional<int64_t> axis;
      if (auto entry = node_attrs.find("axis"); entry != node_attrs.end()) {
        axis = entry->second.i();
      }

      // quantization scale and zp are always the input[1, 2]
      NodeUnitIODef::QuantParam quant_param{*node_inputs[1], node_inputs.size() == 3 ? node_inputs[2] : nullptr, axis};

      if (is_input) {
        // DQ is input to the target node, use the DstArgIndex
        auto idx = cur->GetDstArgIndex();
        // This is a DQ node, we are using x, x_scale, x_zp (input[0, 1, 2])

        // Debug: Check if this is weight-related DQ resolution
        std::string dq_output_name = node.OutputDefs()[0]->Name();
        std::string dq_input_name = node_inputs[0]->Name();
        if (dq_output_name.find("weight") != std::string::npos ||
            dq_input_name.find("weight") != std::string::npos ||
            dq_output_name.find("conv") != std::string::npos ||
            dq_input_name.find("conv") != std::string::npos) {
          std::cout << "NodeUnit: *** WEIGHT-RELATED DQ INPUT RESOLUTION ***" << std::endl;
          std::cout << "NodeUnit: Target node: " << target_node.Name() << " (type: " << target_node.OpType() << ")" << std::endl;
          std::cout << "NodeUnit: DQ node: " << node.Name() << " (type: " << node.OpType() << ")" << std::endl;
          std::cout << "NodeUnit: DQ output (what Conv sees in ONNX): '" << dq_output_name << "'" << std::endl;
          std::cout << "NodeUnit: DQ input (what NodeUnit returns): '" << dq_input_name << "'" << std::endl;
          std::cout << "NodeUnit: Input index: " << idx << std::endl;
          std::cout << "NodeUnit: *** TRANSPARENT DQ RESOLUTION APPLIED ***" << std::endl;
        }

        quantized_io_defs.insert({idx, NodeUnitIODef{*node_inputs[0], quant_param}});
      } else {
        // Q is output of the target node, use the SrcArgIndex
        auto idx = cur->GetSrcArgIndex();
        // This is a Q node, we are using y (output[0]), y_scale, y_zp (input[1, 2])
        const auto node_outputs = node.OutputDefs();
        quantized_io_defs.insert({idx, NodeUnitIODef{*node_outputs[0], quant_param}});
      }
    }
  }

  // Construct the IODefs for this QDQ NodeGroup
  std::vector<NodeUnitIODef> io_defs;
  io_defs.reserve(target_node_io_defs_size);
  for (size_t i = 0; i < target_node_io_defs_size; i++) {
    // If we can find the NodeUnitIODef for this index, this is a quantized input/output
    if (quantized_io_defs.find(i) != quantized_io_defs.cend()) {
      io_defs.push_back(std::move(quantized_io_defs.at(i)));

      // Debug: Log the resolved input transformation
      const auto& resolved_io_def = io_defs.back();
      std::string resolved_name = resolved_io_def.node_arg.Name();
      std::string original_name = target_node_io_defs[i]->Name();
      if (resolved_name != original_name &&
          (resolved_name.find("weight") != std::string::npos ||
           resolved_name.find("conv") != std::string::npos ||
           original_name.find("weight") != std::string::npos ||
           original_name.find("conv") != std::string::npos)) {
        std::cout << "NodeUnit: *** FINAL INPUT TRANSFORMATION ***" << std::endl;
        std::cout << "NodeUnit: Target node: " << target_node.Name() << " input[" << i << "]" << std::endl;
        std::cout << "NodeUnit: Original ONNX input: '" << original_name << "'" << std::endl;
        std::cout << "NodeUnit: Resolved NodeUnit input: '" << resolved_name << "'" << std::endl;
        std::cout << "NodeUnit: *** WEIGHT INPUT SUCCESSFULLY RESOLVED ***" << std::endl;
      }
    } else {
      // This is a regular input
      io_defs.push_back({*target_node_io_defs[i], std::nullopt});
    }
  }

  return io_defs;
}

}  // namespace

Status QDQ::NodeGroup::CanCreateNodeGroup(const GraphViewer& graph_viewer,
                                          const Node& target_node,
                                          const Node* redundant_clip_node,
                                          gsl::span<const Node* const> dq_nodes,
                                          gsl::span<const Node* const> q_nodes) {

  // Debug: Log QDQ group validation attempt
  std::cout << "NodeUnit: *** QDQ GROUP VALIDATION ATTEMPT ***" << std::endl;
  std::cout << "NodeUnit: Target node: " << target_node.Name() << " (type: " << target_node.OpType() << ")" << std::endl;
  std::cout << "NodeUnit: DQ nodes count: " << dq_nodes.size() << std::endl;
  for (size_t i = 0; i < dq_nodes.size(); ++i) {
    if (dq_nodes[i]) {
      std::cout << "NodeUnit: DQ[" << i << "]: " << dq_nodes[i]->Name() << " (type: " << dq_nodes[i]->OpType() << ")" << std::endl;
    }
  }
  std::cout << "NodeUnit: Q nodes count: " << q_nodes.size() << std::endl;
  for (size_t i = 0; i < q_nodes.size(); ++i) {
    if (q_nodes[i]) {
      std::cout << "NodeUnit: Q[" << i << "]: " << q_nodes[i]->Name() << " (type: " << q_nodes[i]->OpType() << ")" << std::endl;
    }
  }
  std::cout << "NodeUnit: Redundant clip node: " << (redundant_clip_node ? redundant_clip_node->Name() : "null") << std::endl;

  // Within a QDQ node group, a target node input is the only consumer of each DQ.
  // This should have been ensured by the EnsureUniqueDQForNodeUnit graph transformer, but other graph modifications
  // may have happened since. Verify that this is still true.
  for (const auto* dq_node : dq_nodes) {
    const bool dq_produces_graph_output = graph_viewer.NodeProducesGraphOutput(*dq_node);
    std::cout << "NodeUnit: Checking DQ node: " << dq_node->Name() << std::endl;
    std::cout << "NodeUnit: DQ produces graph output: " << (dq_produces_graph_output ? "YES" : "NO") << std::endl;

    ORT_RETURN_IF(dq_produces_graph_output,
                  "QDQ node group cannot have DQ node that produces a graph output. DQ node: ", dq_node->Name(),
                  ", target node: ", target_node.Name());

    const bool dq_has_single_output_edge_to_target =
        dq_node->GetOutputEdgesCount() == 1 &&
        dq_node->OutputEdgesBegin()->GetNode().Index() == target_node.Index();

    std::cout << "NodeUnit: DQ output edges count: " << dq_node->GetOutputEdgesCount() << std::endl;
    if (dq_node->GetOutputEdgesCount() > 0) {
      std::cout << "NodeUnit: DQ first output edge to: " << dq_node->OutputEdgesBegin()->GetNode().Name()
                << " (target: " << target_node.Name() << ")" << std::endl;
    }
    std::cout << "NodeUnit: DQ has single output edge to target: " << (dq_has_single_output_edge_to_target ? "YES" : "NO") << std::endl;

    ORT_RETURN_IF_NOT(dq_has_single_output_edge_to_target,
                      "QDQ node group cannot have DQ that doesn't have a single output edge to the target node. "
                      "DQ node: ",
                      dq_node->Name(), ", target node: ", target_node.Name());
  }

  // If redundant_clip_node is present, currently we require target node has only one output edge, which is connected to
  // the redundant_clip_node. The redundant_clip_node's output is consumed by the Q node that can be fused with itself.
  if (redundant_clip_node) {
    ORT_RETURN_IF_NOT(!graph_viewer.NodeProducesGraphOutput(target_node) && target_node.GetOutputEdgesCount() == 1 &&
                          target_node.OutputEdgesBegin()->GetNode().Index() == redundant_clip_node->Index(),
                      "QDQ node group cannot have target node with more than one output edge if there is redunant clip "
                      "node. target node: ",
                      target_node.Name());
    ORT_RETURN_IF_NOT(
        !graph_viewer.NodeProducesGraphOutput(*redundant_clip_node) && q_nodes.size() == 1 &&
            redundant_clip_node->GetOutputEdgesCount() == 1 &&
            redundant_clip_node->OutputEdgesBegin()->GetNode().Index() == q_nodes[0]->Index(),
        "QDQ node group cannot have redudant clip node that doesn't have a single output edge to a Q node. "
        "redundant clip node: ",
        redundant_clip_node->Name());
    return Status::OK();
  }

  // an output from the target node can have either Q consumers or direct consumers. it cannot have both.
  // this must be checked on a per output basis.
  // e.g. TopK produces values and indices. The indices output won't be quantized, so even if we replace the TopK QDQ
  // node group with a quantized TopK, an int64_t indices value will be produced and can provide a graph output.
  if (!q_nodes.empty()) {
    std::cout << "NodeUnit: *** VALIDATING TARGET NODE OUTPUTS ***" << std::endl;
    std::cout << "NodeUnit: Target node has " << target_node.GetOutputEdgesCount() << " output edges" << std::endl;

    auto cur_edge = target_node.OutputEdgesBegin();
    auto end_edge = target_node.OutputEdgesEnd();
    std::vector<const Node*> output_consumers(target_node.OutputDefs().size(), nullptr);

    for (; cur_edge != end_edge; ++cur_edge) {
      auto output_idx = cur_edge->GetSrcArgIndex();
      const Node& this_consumer = cur_edge->GetNode();
      const Node* existing_consumer = output_consumers[output_idx];

      std::cout << "NodeUnit: Output[" << output_idx << "] consumer: " << this_consumer.Name()
                << " (type: " << this_consumer.OpType() << ")" << std::endl;

      if (existing_consumer != nullptr) {
        // another edge for this output. either both are Q or both are not.
        bool valid = true;
        if (existing_consumer->OpType() == "QuantizeLinear") {
          valid = this_consumer.OpType() == "QuantizeLinear";
        } else {
          valid = this_consumer.OpType() != "QuantizeLinear";
        }

        std::cout << "NodeUnit: Multiple consumers for output[" << output_idx << "]: "
                  << existing_consumer->Name() << " and " << this_consumer.Name() << std::endl;
        std::cout << "NodeUnit: Validation result: " << (valid ? "PASS" : "FAIL") << std::endl;

        ORT_RETURN_IF_NOT(valid,
                          "QDQ node group cannot have an output from the target node being consumed by a Q node and "
                          "a non-Q node. target node: ",
                          target_node.Name());
      } else {
        output_consumers[output_idx] = &this_consumer;
      }
    }

        const auto& graph_outputs = graph_viewer.GetOutputs();
    std::cout << "NodeUnit: *** VALIDATING GRAPH OUTPUTS ***" << std::endl;
    std::cout << "NodeUnit: Graph has " << graph_outputs.size() << " outputs" << std::endl;

    for (size_t idx = 0, end = output_consumers.size(); idx < end; ++idx) {
      // any output with a Q cannot be a graph output as it will disappear if the QDQ node unit is converted to
      // a quantized op.
      if (output_consumers[idx] != nullptr && output_consumers[idx]->OpType() == "QuantizeLinear") {
        const auto& output_name = target_node.OutputDefs()[idx]->Name();
        bool is_graph_output = std::any_of(graph_outputs.begin(), graph_outputs.end(),
                                         [&output_name](const NodeArg* node_arg) {
                                           return node_arg->Name() == output_name;
                                         });

        std::cout << "NodeUnit: Output[" << idx << "] '" << output_name << "' consumed by Q node: "
                  << output_consumers[idx]->Name() << std::endl;
        std::cout << "NodeUnit: Output[" << idx << "] is graph output: " << (is_graph_output ? "YES" : "NO") << std::endl;

        ORT_RETURN_IF(is_graph_output,
                      "QDQ node group cannot have an output from the target node that is consumed by a Q node and "
                      "a graph output. target node: ",
                      target_node.Name(), " output idx:", idx);
      }
    }
  }

  std::cout << "NodeUnit: *** QDQ GROUP VALIDATION SUCCESSFUL ***" << std::endl;
  return Status::OK();
}

NodeUnit::NodeUnit(const Node& node)
    : target_node_(node),
      redundant_clip_node_(nullptr),
      type_(Type::SingleNode),
      input_edge_count_(node.GetInputEdgesCount()) {
  InitForSingleNode();
}

NodeUnit::NodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& node_group)
    : dq_nodes_{GetQDQIONodes(graph_viewer, node_group, true /* is_input */)},
      target_node_(*graph_viewer.GetNode(node_group.target_node)),
      redundant_clip_node_(node_group.redundant_clip_node.has_value()
                               ? graph_viewer.GetNode(node_group.redundant_clip_node.value())
                               : nullptr),
      q_nodes_{GetQDQIONodes(graph_viewer, node_group, false /* is_input */)},
      type_(Type::QDQGroup),
      inputs_{GetQDQIODefs(target_node_, node_group, true /* is_input */)},
      outputs_{GetQDQIODefs((redundant_clip_node_ ? *redundant_clip_node_ : target_node_), node_group,
                            false /* is_input */)} {

  // Debug: Log QDQ NodeGroup creation
  std::cout << "NodeUnit: Creating QDQ NodeGroup for target node: " << target_node_.Name()
            << " (type: " << target_node_.OpType() << ")" << std::endl;
  std::cout << "NodeUnit: DQ nodes count: " << dq_nodes_.size() << std::endl;
  for (size_t i = 0; i < dq_nodes_.size(); ++i) {
    if (dq_nodes_[i]) {
      std::cout << "NodeUnit: DQ[" << i << "]: " << dq_nodes_[i]->Name() << std::endl;
    }
  }
  std::cout << "NodeUnit: Q nodes count: " << q_nodes_.size() << std::endl;
  for (size_t i = 0; i < q_nodes_.size(); ++i) {
    if (q_nodes_[i]) {
      std::cout << "NodeUnit: Q[" << i << "]: " << q_nodes_[i]->Name() << std::endl;
    }
  }
  std::cout << "NodeUnit: *** ATTEMPTING QDQ GROUP VALIDATION ***" << std::endl;
  Status validation_status = QDQ::NodeGroup::CanCreateNodeGroup(graph_viewer, target_node_, redundant_clip_node_, dq_nodes_, q_nodes_);
  if (!validation_status.IsOK()) {
    std::cout << "NodeUnit: *** QDQ GROUP VALIDATION FAILED ***" << std::endl;
    std::cout << "NodeUnit: Error: " << validation_status.ErrorMessage() << std::endl;
    ORT_THROW_IF_ERROR(validation_status);
  } else {
    std::cout << "NodeUnit: *** QDQ GROUP VALIDATION PASSED ***" << std::endl;
  }

  input_edge_count_ = std::accumulate(dq_nodes_.cbegin(), dq_nodes_.cend(), size_t(0),
                                      [](size_t acc, const Node* node) { return acc + node->GetInputEdgesCount(); });

  // add edges for inputs that are not from DQ nodes. there is one edge to each DQ node.
  // other inputs could come from initializers or graph inputs (no edges) or other nodes (edge).
  input_edge_count_ += target_node_.GetInputEdgesCount() - dq_nodes_.size();

  // create output edges. each target node output either goes to Q node/s or non-Q node/s.
  // ValidateNodeGroupQDQNodes ensures this.
  // If redundant clip node is present, the target node has only one output edge to the redundant clip node.
  const Node& output_producer = redundant_clip_node_ ? *redundant_clip_node_ : target_node_;
  auto cur_edge = output_producer.OutputEdgesBegin();
  auto end_edge = output_producer.OutputEdgesEnd();
  for (; cur_edge != end_edge; ++cur_edge) {
    const Node& node = cur_edge->GetNode();

    // if node is in q_nodes we hide the Q node.
    if (std::find(q_nodes_.cbegin(), q_nodes_.cend(), &node) != q_nodes_.cend()) {
      auto src_idx = cur_edge->GetSrcArgIndex();
      auto q_cur_edge = node.OutputEdgesBegin();
      auto q_end_edge = node.OutputEdgesEnd();
      for (; q_cur_edge != q_end_edge; ++q_cur_edge) {
        output_edges_.insert(Node::EdgeEnd{q_cur_edge->GetNode(), src_idx, q_cur_edge->GetDstArgIndex()});
      }
    } else {
      // non-Q node, or Q node that isn't in the QDQ node group (unexpected but may be possible). add as-is.
      output_edges_.insert(*cur_edge);
    }
  }
}

NodeUnit::NodeUnit(gsl::span<const Node* const> dq_nodes, const Node& target_node, const Node* redundant_clip_node,
                   gsl::span<const Node* const> q_nodes, Type unit_type,
                   gsl::span<const NodeUnitIODef> inputs, gsl::span<const NodeUnitIODef> outputs,
                   size_t input_edge_count, Node::EdgeSet output_edges)
    : dq_nodes_(dq_nodes.begin(), dq_nodes.end()),
      target_node_(target_node),
      redundant_clip_node_(redundant_clip_node),
      q_nodes_(q_nodes.begin(), q_nodes.end()),
      type_(unit_type),
      inputs_(inputs.begin(), inputs.end()),
      outputs_(outputs.begin(), outputs.end()),
      input_edge_count_(input_edge_count),
      output_edges_(std::move(output_edges)) {
}

const std::string& NodeUnit::Domain() const noexcept { return target_node_.Domain(); }
const std::string& NodeUnit::OpType() const noexcept { return target_node_.OpType(); }
const std::string& NodeUnit::Name() const noexcept { return target_node_.Name(); }
int NodeUnit::SinceVersion() const noexcept { return target_node_.SinceVersion(); }
NodeIndex NodeUnit::Index() const noexcept { return target_node_.Index(); }
const std::filesystem::path& NodeUnit::ModelPath() const noexcept { return target_node_.ModelPath(); }
ProviderType NodeUnit::GetExecutionProviderType() const noexcept { return target_node_.GetExecutionProviderType(); }

void NodeUnit::InitForSingleNode() {
  const auto& input_defs = target_node_.InputDefs();
  const auto& output_defs = target_node_.OutputDefs();
  const auto& node_attrs = target_node_.GetAttributes();
  auto qlinear_type = GetQLinearOpType(target_node_);
  if (qlinear_type == QLinearOpType::Unknown) {
    // Not a Qlinear op, add all inputs / outputs
    auto add_all_io = [](std::vector<NodeUnitIODef>& defs,
                         const ConstPointerContainer<std::vector<NodeArg*>>& node_defs) {
      defs.reserve(node_defs.size());

      for (const auto def : node_defs) {
        defs.push_back(NodeUnitIODef{*def, std::nullopt});
      }
    };

    add_all_io(inputs_, input_defs);
    add_all_io(outputs_, output_defs);
  } else if (IsUnaryQLinearOp(qlinear_type)) {
    // Unary QLinear Op has 5 inputs
    // x, x_scale, x_zp, y_scale, y_zp (optional)
    inputs_.push_back(NodeUnitIODef{*input_defs[0], NodeUnitIODef::QuantParam{*input_defs[1], input_defs[2]}});
    outputs_.push_back(NodeUnitIODef{*output_defs[0],
                                     NodeUnitIODef::QuantParam{*input_defs[3],
                                                               input_defs.size() > 4 ? input_defs[4] : nullptr}});

  } else if (IsBinaryQLinearOp(qlinear_type)) {
    // Binary QLinear Op has 9 inputs
    // x1, x1_scale, x1_zp, x2/w, x2_scale, x2_zp, y_scale , y_zp, B
    inputs_.push_back(NodeUnitIODef{*input_defs[0], NodeUnitIODef::QuantParam{*input_defs[1], input_defs[2]}});
    inputs_.push_back(NodeUnitIODef{*input_defs[3], NodeUnitIODef::QuantParam{*input_defs[4], input_defs[5]}});

    if (input_defs.size() == 9) {                                      // has Bias
      inputs_.push_back(NodeUnitIODef{*input_defs[8], std::nullopt});  // for Bias the scale and zp are optional
    }

    outputs_.push_back(NodeUnitIODef{*output_defs[0], NodeUnitIODef::QuantParam{*input_defs[6], input_defs[7]}});

  } else if (qlinear_type == QLinearOpType::DequantizeLinear) {
    // DequantizeLinear has 3 inputs
    // x, x_scale, x_zp
    // output is not quantized

    // Get the DQ axis attribute if available.
    std::optional<int64_t> axis;
    if (auto entry = node_attrs.find("axis"); entry != node_attrs.end()) {
      axis = entry->second.i();
    }

    inputs_.push_back(NodeUnitIODef{*input_defs[0],
                                    NodeUnitIODef::QuantParam{*input_defs[1],
                                                              input_defs.size() == 3 ? input_defs[2] : nullptr,
                                                              axis}});
    outputs_.push_back(NodeUnitIODef{*output_defs[0], std::nullopt});

  } else if (qlinear_type == QLinearOpType::QuantizeLinear) {
    // QuantizeLinear the input is not quantized and has 3 inputs
    // x, y_scale, y_zp (optional)
    // The output is quantized

    // Get the Q axis attribute if available.
    std::optional<int64_t> axis;
    if (auto entry = node_attrs.find("axis"); entry != node_attrs.end()) {
      axis = entry->second.i();
    }

    inputs_.push_back(NodeUnitIODef{*input_defs[0], std::nullopt});
    outputs_.push_back(NodeUnitIODef{*output_defs[0],
                                     NodeUnitIODef::QuantParam{*input_defs[1],
                                                               input_defs.size() == 3 ? input_defs[2] : nullptr,
                                                               axis}});
  } else if (IsVariadicQLinearOp(qlinear_type)) {
    size_t input_num = (input_defs.size() - 2) / 3;
    for (size_t i = 0; i < input_num; i++) {
      inputs_.push_back(NodeUnitIODef{*input_defs[3 * i + 2], NodeUnitIODef::QuantParam{*input_defs[3 * i + 3],
                                                                                        input_defs[3 * i + 4]}});
    }
    outputs_.push_back(NodeUnitIODef{*output_defs[0], NodeUnitIODef::QuantParam{*input_defs[0], input_defs[1]}});
  } else {
    ORT_THROW("The QLinear op [", static_cast<uint8_t>(qlinear_type), "] is not supported");
  }
}

Node::EdgeConstIterator NodeUnit::OutputEdgesBegin() const {
  return (type_ == Type::SingleNode) ? target_node_.OutputEdgesBegin() : output_edges_.begin();
}

Node::EdgeConstIterator NodeUnit::OutputEdgesEnd() const {
  return (type_ == Type::SingleNode) ? target_node_.OutputEdgesEnd() : output_edges_.end();
}

std::vector<const Node*> NodeUnit::GetAllNodesInGroup() const noexcept {
  std::vector<const Node*> all_nodes = dq_nodes_;
  all_nodes.push_back(&target_node_);
  if (redundant_clip_node_) {
    all_nodes.push_back(redundant_clip_node_);
  }
  all_nodes.reserve(all_nodes.size() + q_nodes_.size());
  for (auto& n : q_nodes_)
    all_nodes.push_back(n);
  return all_nodes;
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
