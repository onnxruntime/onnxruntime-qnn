// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <algorithm>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

// RAII guard for QnnModelWrapper::Push/PopBranchGraphScope.
class BranchGraphScopeGuard {
 public:
  BranchGraphScopeGuard(QnnModelWrapper& qmw, const OrtGraph* branch) : qmw_(qmw) {
    qmw_.PushBranchGraphScope(branch);
  }
  ~BranchGraphScopeGuard() { qmw_.PopBranchGraphScope(); }
  BranchGraphScopeGuard(const BranchGraphScopeGuard&) = delete;
  BranchGraphScopeGuard& operator=(const BranchGraphScopeGuard&) = delete;

 private:
  QnnModelWrapper& qmw_;
};

// Build a minimal OrtNodeUnitIODef from a plain OrtValueInfo*.
// Returns non-OK status if the value info cannot be read.
Ort::Status MakeIODefFromValueInfo(const OrtApi& ort_api,
                                   const OrtValueInfo* vi,
                                   OrtNodeUnitIODef& def) {
  const char* name = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoName(vi, &name));
  def.name = name ? name : "";

  const OrtTypeInfo* type_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(vi, &type_info));

  const OrtTensorTypeAndShapeInfo* tinfo = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tinfo));

  ONNXTensorElementDataType onnx_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(tinfo, &onnx_dtype));
  def.type = onnx_dtype;

  size_t ndim = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensionsCount(tinfo, &ndim));
  if (ndim > 0) {
    std::vector<int64_t> dims(ndim);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensions(tinfo, dims.data(), ndim));
    def.shape = std::move(dims);
  } else {
    def.shape = std::vector<int64_t>{};
  }

  return Ort::Status();
}

// Returns the branch GraphProto for the named attribute ("then_branch" or "else_branch").
Ort::Status GetBranchGraph(const OrtApi& ort_api,
                           const OrtNode& if_node,
                           const std::string& attr_name,
                           const OrtGraph*& out_graph) {
  out_graph = nullptr;
  size_t num_subgraphs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumSubgraphs(&if_node, &num_subgraphs));
  RETURN_IF_NOT(num_subgraphs > 0, "If node has no subgraphs.");

  std::vector<const OrtGraph*> subgraphs(num_subgraphs);
  std::vector<const char*> names(num_subgraphs);
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetSubgraphs(&if_node, subgraphs.data(), num_subgraphs, names.data()));

  for (size_t i = 0; i < num_subgraphs; ++i) {
    if (names[i] && std::string(names[i]) == attr_name) {
      out_graph = subgraphs[i];
      return Ort::Status();
    }
  }
  return MAKE_EP_FAIL(("If node missing subgraph attribute: " + attr_name).c_str());
}

// Validate and retrieve a branch's single output: value-info pointer, name, dtype, shape.
Ort::Status GetBranchSingleOutput(const OrtApi& ort_api,
                                  const OrtGraph* graph,
                                  const OrtValueInfo*& out_vi,
                                  std::string& out_name,
                                  ONNXTensorElementDataType& out_type,
                                  std::vector<int64_t>& out_shape) {
  size_t n = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNumOutputs(graph, &n));
  RETURN_IF_NOT(n == 1, "If branches must produce exactly 1 output.");

  std::vector<const OrtValueInfo*> outs(n);
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetOutputs(graph, outs.data(), n));
  out_vi = outs[0];

  const char* name = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoName(out_vi, &name));
  out_name = name ? name : "";

  const OrtTypeInfo* type_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(out_vi, &type_info));

  ONNXType value_type = ONNX_TYPE_UNKNOWN;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetOnnxTypeFromTypeInfo(type_info, &value_type));
  RETURN_IF_NOT(value_type == ONNX_TYPE_TENSOR,
                "If branch output must be a plain tensor (sequence/optional not supported).");

  const OrtTensorTypeAndShapeInfo* tinfo = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tinfo));
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(tinfo, &out_type));

  size_t ndim = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensionsCount(tinfo, &ndim));
  out_shape.resize(ndim);
  if (ndim > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensions(tinfo, out_shape.data(), ndim));
  }
  return Ort::Status();
}

// Translate all nodes of a branch into the QnnModelWrapper.
// Pushes the branch graph scope (so branch initializers resolve as STATIC) and pops on exit.
Ort::Status TranslateBranch(QnnModelWrapper& qmw,
                            const Ort::Logger& logger,
                            const OrtGraph* branch,
                            bool do_op_validation) {
  const OrtApi& ort_api = qmw.GetOrtApi();

  size_t num_nodes = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNumNodes(branch, &num_nodes));
  std::vector<const OrtNode*> nodes(num_nodes);
  if (num_nodes > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNodes(branch, nodes.data(), num_nodes));
  }

  BranchGraphScopeGuard scope_guard(qmw, branch);

  for (const OrtNode* node : nodes) {
    OrtNodeUnit unit(node, ort_api);
    const auto* builder = qnn::GetOpBuilder(unit.OpType());
    if (builder == nullptr) {
      return MAKE_EP_FAIL(("If branch op `" + unit.OpType() +
                           "` is not supported by QNN EP.")
                              .c_str());
    }
    Ort::Status s = do_op_validation
                        ? builder->IsOpSupported(qmw, unit, logger)
                        : builder->AddToModelBuilder(qmw, unit, logger, false);
    if (!s.IsOK()) {
      return s;
    }
  }

  return Ort::Status();
}

// If a branch's declared output is not yet registered after TranslateBranch (e.g., the branch
// has zero compute nodes because all Constants were folded into branch initializers), register
// it now from the branch initializer table using the value-info already retrieved by
// GetBranchSingleOutput.
Ort::Status EnsureBranchOutputRegistered(QnnModelWrapper& qmw,
                                         const OrtGraph* branch,
                                         const OrtValueInfo* output_vi,
                                         const std::string& output_name) {
  if (qmw.IsQnnTensorWrapperExist(output_name)) {
    return Ort::Status();
  }

  const OrtApi& ort_api = qmw.GetOrtApi();
  OrtNodeUnitIODef io_def;
  RETURN_IF_ERROR(MakeIODefFromValueInfo(ort_api, output_vi, io_def));

  QnnTensorWrapper wrapper;
  {
    BranchGraphScopeGuard scope_guard(qmw, branch);
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(io_def, wrapper));
  }
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(wrapper)),
                ("Failed to register branch output tensor: " + output_name).c_str());
  return Ort::Status();
}

}  // namespace

class IfOpBuilder : public BaseOpBuilder {
 public:
  IfOpBuilder() : BaseOpBuilder("IfOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IfOpBuilder);

 protected:
  // Registers explicit input (cond) AND implicit inputs (outer-scope tensors consumed
  // by branch subgraphs) as QNN tensors.
  Ort::Status ProcessInputs(QnnModelWrapper& qmw,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override;

  // Validates branch structure, translates both branches, and emits the final
  // QNN_OP_ELEMENT_WISE_SELECT node.
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qmw,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override;
};

Ort::Status IfOpBuilder::ProcessInputs(QnnModelWrapper& qmw,
                                       const OrtNodeUnit& node_unit,
                                       const Ort::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const {
  // Register the explicit condition input through the standard pipeline.
  RETURN_IF_ERROR(BaseOpBuilder::ProcessInputs(qmw, node_unit, logger, input_names, do_op_validation));

  // Register implicit inputs (outer-scope tensors consumed by branch subgraphs).
  // These are NOT direct inputs to the If node, so ORT's partitioner doesn't surface
  // them as partition I/O. We register them here so branch op-builders can resolve them.
  const OrtApi& ort_api = qmw.GetOrtApi();
  const OrtNode& if_node = node_unit.GetNode();

  size_t num_implicit = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumImplicitInputs(&if_node, &num_implicit));
  if (num_implicit == 0) return Ort::Status();

  std::vector<const OrtValueInfo*> implicit_inputs(num_implicit);
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetImplicitInputs(&if_node,
                                                            implicit_inputs.data(),
                                                            num_implicit));

  for (const OrtValueInfo* vi : implicit_inputs) {
    OrtNodeUnitIODef io_def;
    RETURN_IF_ERROR(MakeIODefFromValueInfo(ort_api, vi, io_def));
    if (io_def.name.empty() || qmw.IsQnnTensorWrapperExist(io_def.name)) {
      continue;
    }
    QnnTensorWrapper tensor_wrapper;
    RETURN_IF_ERROR(qmw.MakeTensorWrapper(io_def, tensor_wrapper));
    RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(tensor_wrapper)),
                  ("Failed to register implicit input: " + io_def.name).c_str());
  }

  return Ort::Status();
}

Ort::Status IfOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qmw,
                                                     const OrtNodeUnit& node_unit,
                                                     std::vector<std::string>&& input_names,
                                                     const Ort::Logger& logger,
                                                     bool do_op_validation) const {
  const OrtApi& ort_api = qmw.GetOrtApi();
  const OrtNode& if_node = node_unit.GetNode();

  // ONNX If spec: cond is a scalar boolean.
  RETURN_IF_NOT(!node_unit.Inputs().empty(), "If node missing cond input.");
  const auto& cond_def = node_unit.Inputs()[0];
  RETURN_IF_NOT(cond_def.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
                "If cond input must have BOOL dtype.");
  std::vector<uint32_t> cond_shape;
  RETURN_IF_NOT(qmw.GetOnnxShape(cond_def.shape, cond_shape), "Cannot get cond shape.");
  bool cond_is_scalar = cond_shape.empty() ||
                        std::all_of(cond_shape.begin(), cond_shape.end(),
                                    [](uint32_t d) { return d == 1u; });
  RETURN_IF_NOT(cond_is_scalar, "If cond input must be a scalar (single-element) tensor.");

  // Retrieve branch graphs.
  const OrtGraph* then_graph = nullptr;
  const OrtGraph* else_graph = nullptr;
  RETURN_IF_ERROR(GetBranchGraph(ort_api, if_node, "then_branch", then_graph));
  RETURN_IF_ERROR(GetBranchGraph(ort_api, if_node, "else_branch", else_graph));

  // Validate branch outputs: single output, matching shape + dtype, no name collision.
  const OrtValueInfo* then_vi = nullptr;
  const OrtValueInfo* else_vi = nullptr;
  std::string then_name, else_name;
  ONNXTensorElementDataType then_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType else_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<int64_t> then_shape, else_shape;
  RETURN_IF_ERROR(GetBranchSingleOutput(ort_api, then_graph, then_vi, then_name, then_type, then_shape));
  RETURN_IF_ERROR(GetBranchSingleOutput(ort_api, else_graph, else_vi, else_name, else_type, else_shape));
  RETURN_IF_NOT(then_type == else_type, "If branches must have identical output dtype.");
  RETURN_IF_NOT(then_shape == else_shape, "If branches must have identical output shape.");

  RETURN_IF_NOT(!node_unit.Outputs().empty(), "If node must declare an output.");
  const std::string& if_out_name = node_unit.Outputs()[0].name;
  RETURN_IF_NOT(then_name != if_out_name && else_name != if_out_name,
                "If branch terminus name collides with If output name; rename not supported.");

  // Translate both branches.
  RETURN_IF_ERROR(TranslateBranch(qmw, logger, then_graph, do_op_validation));
  RETURN_IF_ERROR(TranslateBranch(qmw, logger, else_graph, do_op_validation));

  // For purely constant branches (0 compute nodes after ORT folding), the branch's declared
  // output never gets registered by an op-builder. Register it here from the branch initializer.
  RETURN_IF_ERROR(EnsureBranchOutputRegistered(qmw, then_graph, then_vi, then_name));
  RETURN_IF_ERROR(EnsureBranchOutputRegistered(qmw, else_graph, else_vi, else_name));

  // Register the If output tensor.
  QnnTensorWrapper output_wrapper;
  RETURN_IF_ERROR(qmw.MakeTensorWrapper(node_unit.Outputs()[0], output_wrapper));
  RETURN_IF_NOT(qmw.AddTensorWrapper(std::move(output_wrapper)),
                "Failed to register If output tensor.");

  // Emit ElementWiseSelect(cond, then_out, else_out) -> if_out.
  const std::string cond_name = input_names.empty() ? node_unit.Inputs()[0].name
                                                    : input_names[0];
  RETURN_IF_NOT(qmw.CreateQnnNode(node_unit.Name() + "_select",
                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                  QNN_OP_ELEMENT_WISE_SELECT,
                                  {cond_name, then_name, else_name},
                                  {if_out_name},
                                  {},
                                  do_op_validation),
                "Failed to create ElementWiseSelect for If.");

  return Ort::Status();
}

void CreateIfOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<IfOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
