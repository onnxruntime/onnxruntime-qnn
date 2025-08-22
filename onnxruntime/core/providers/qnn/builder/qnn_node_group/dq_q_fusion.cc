#include "core/providers/qnn/builder/qnn_node_group/dq_q_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <utility>
#include <iostream>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, dq_node_unit, q_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (dq_node_unit), (q_node_unit), true)
#define CreateOnQnn(qnn_model_wrapper, dq_node_unit, q_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (dq_node_unit), (q_node_unit), false)
static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& dq_node_unit,
                                    const NodeUnit& q_node_unit, bool validate);
static bool IsDQQConversion(const GraphViewer& graph_viewer, const Node& dq_node, const Node& q_node);

std::unique_ptr<IQnnNodeGroup> DQQFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& dq_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  // Expect that this function is called with a standalone DQ.
  if (dq_node_unit.OpType() != DEQUANTIZE_LINEAR || dq_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const Node& dq_node = dq_node_unit.GetNode();

  // Check if this is a weight-related DQ node
  bool is_weight_related = false;
  if (dq_node.InputDefs().size() > 0) {
    std::string dq_input_name = dq_node.InputDefs()[0]->Name();
    if (dq_input_name.find("weight") != std::string::npos ||
        dq_input_name.find("conv") != std::string::npos) {
      is_weight_related = true;
      std::cout << "DQQFusion: *** WEIGHT-RELATED DQ NODE DETECTED ***" << std::endl;
      std::cout << "DQQFusion: DQ node: " << dq_node.Name() << std::endl;
      std::cout << "DQQFusion: DQ input: " << dq_input_name << std::endl;
      std::cout << "DQQFusion: DQ output: " << dq_node.OutputDefs()[0]->Name() << std::endl;
    }
  }

  // DQ must have a single Q child (1 output edge) and must not produce a graph output.
  const std::array<std::string_view, 1> child_types = {QUANTIZE_LINEAR};
  const NodeUnit* q_node_unit = GetOnlyChildOfType(graph_viewer, dq_node_unit, child_types,
                                                   node_to_node_unit, node_unit_to_qnn_node_group);

  if (q_node_unit == nullptr) {
    if (is_weight_related) {
      std::cout << "DQQFusion: No Q child found for weight-related DQ node" << std::endl;
    }
    return nullptr;
  }

  if (is_weight_related) {
    std::cout << "DQQFusion: Found Q child: " << q_node_unit->GetNode().Name() << std::endl;
    std::cout << "DQQFusion: Q output: " << q_node_unit->GetNode().OutputDefs()[0]->Name() << std::endl;
  }

  // DQ and Q must have equal scale type and different zp type.
  if (!IsDQQConversion(graph_viewer, dq_node, q_node_unit->GetNode())) {
    if (is_weight_related) {
      std::cout << "DQQFusion: DQ->Q conversion not supported for weight-related nodes" << std::endl;
    }
    return nullptr;
  }

  if (Status status = ValidateOnQnn(qnn_model_wrapper, dq_node_unit, *q_node_unit);
      !status.IsOK()) {
    if (is_weight_related) {
      std::cout << "DQQFusion: QNN validation failed for weight-related DQ->Q sequence" << std::endl;
    }
    return nullptr;
  }

  if (is_weight_related) {
    std::cout << "DQQFusion: *** CREATING WEIGHT-RELATED DQ->Q FUSION ***" << std::endl;
    std::cout << "DQQFusion: Will fuse DQ(" << dq_node.Name() << ") -> Q(" << q_node_unit->GetNode().Name() << ") into Convert op" << std::endl;
  }

  return std::make_unique<DQQFusion>(dq_node_unit, *q_node_unit);
}

DQQFusion::DQQFusion(const NodeUnit& dq_node_unit, const NodeUnit& q_node_unit)
    : node_units_{&dq_node_unit, &q_node_unit} {
}

Status DQQFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

Status DQQFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  // Check if this is a weight-related fusion
  if (node_units_[0]->Inputs().size() > 0) {
    std::string dq_input_name = node_units_[0]->Inputs()[0].node_arg.Name();
    if (dq_input_name.find("weight") != std::string::npos ||
        dq_input_name.find("conv") != std::string::npos) {
      std::cout << "DQQFusion: *** APPLYING WEIGHT-RELATED DQ->Q FUSION ***" << std::endl;
      std::cout << "DQQFusion: DQ input: " << dq_input_name << std::endl;
      std::cout << "DQQFusion: DQ output: " << node_units_[0]->Outputs()[0].node_arg.Name() << std::endl;
      std::cout << "DQQFusion: Q output: " << node_units_[1]->Outputs()[0].node_arg.Name() << std::endl;
      std::cout << "DQQFusion: Creating Convert op from " << dq_input_name << " to " << node_units_[1]->Outputs()[0].node_arg.Name() << std::endl;
    }
  }

  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

gsl::span<const NodeUnit* const> DQQFusion::GetNodeUnits() const {
  return node_units_;
}

const NodeUnit* DQQFusion::GetTargetNodeUnit() const {
  return node_units_[0];
}

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& dq_node_unit,
                                    const NodeUnit& q_node_unit,
                                    bool validate) {
  assert(dq_node_unit.OpType() == DEQUANTIZE_LINEAR && q_node_unit.OpType() == QUANTIZE_LINEAR);
  const auto& node_name = utils::GetUniqueName(dq_node_unit);
  const NodeUnitIODef& input_def = dq_node_unit.Inputs()[0];
  const NodeUnitIODef& output_def = q_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_CONVERT,
                                                          {input_tensor.GetQnnTensor()},
                                                          {output_tensor.GetQnnTensor()},
                                                          {}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_CONVERT,
                                                      {input_def.node_arg.Name()},
                                                      {output_def.node_arg.Name()},
                                                      {},
                                                      validate),
                      "Failed to add fused Convert node.");
  }

  return Status::OK();
}

static bool IsDQQConversion(const GraphViewer& graph_viewer, const Node& dq_node, const Node& q_node) {
  ConstPointerContainer<std::vector<NodeArg*>> dq_input_defs = dq_node.InputDefs();
  ConstPointerContainer<std::vector<NodeArg*>> q_input_defs = q_node.InputDefs();

  auto is_scalar_shape = [](const NodeArg& input_arg) -> bool {
    auto shape = input_arg.Shape();
    if (shape == nullptr) {
      return false;
    }

    auto dim_size = shape->dim_size();
    return dim_size == 0 || (dim_size == 1 && shape->dim(0).has_dim_value() && shape->dim(0).dim_value() == 1);
  };

  // Q/DQ contains optional input is not supported
  // non-scalar Q/DQ scale and zero point needs are not supported
  if (dq_input_defs.size() != QDQ_MAX_NUM_INPUTS ||
      q_input_defs.size() != QDQ_MAX_NUM_INPUTS ||
      !is_scalar_shape(*q_input_defs[QDQ_SCALE_INPUT_IDX]) ||
      !is_scalar_shape(*q_input_defs[QDQ_ZERO_POINT_INPUT_IDX]) ||
      !is_scalar_shape(*dq_input_defs[QDQ_SCALE_INPUT_IDX]) ||
      !is_scalar_shape(*dq_input_defs[QDQ_ZERO_POINT_INPUT_IDX])) {
    return false;
  }

  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
      graph_viewer.GetConstantInitializer(dq_input_defs[QDQ_SCALE_INPUT_IDX]->Name());
  const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
      graph_viewer.GetConstantInitializer(q_input_defs[QDQ_SCALE_INPUT_IDX]->Name());
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
      graph_viewer.GetConstantInitializer(dq_input_defs[QDQ_ZERO_POINT_INPUT_IDX]->Name());
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
      graph_viewer.GetConstantInitializer(q_input_defs[QDQ_ZERO_POINT_INPUT_IDX]->Name());
  if (nullptr == q_zp_tensor_proto ||
      nullptr == dq_zp_tensor_proto ||
      nullptr == q_scale_tensor_proto ||
      nullptr == dq_scale_tensor_proto) {
    return false;
  }

  // All TensorProtos must have a data type
  if (!q_zp_tensor_proto->has_data_type() || !dq_zp_tensor_proto->has_data_type() ||
      !q_scale_tensor_proto->has_data_type() || !dq_scale_tensor_proto->has_data_type()) {
    return false;
  }

  // For scale, ensure that the Q/DQ have same scale type.
  //
  // For zero-point: we previously only fused (DQ -> Q) into a Convert op if the quantization types differed.
  // However, a single Convert op is faster than (DQ -> Q), so we should always fuse regardless of the zero-point type.
  return (dq_scale_tensor_proto->data_type() == q_scale_tensor_proto->data_type());
}

}  // namespace qnn
}  // namespace onnxruntime
