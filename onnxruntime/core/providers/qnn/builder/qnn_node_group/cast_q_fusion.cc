// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/cast_q_fusion.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

constexpr char kOpCast[] = "Cast";
constexpr char kOpQuantize[] = "QuantizeLinear";

std::unique_ptr<IQnnNodeGroup> CastQuantizeFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& cast_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    [[maybe_unused]] const logging::Logger& logger) {
    if (cast_node_unit.OpType() != kOpCast || cast_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
        return nullptr;
    }

    // Cast node must have a single QuantizeLinear node as child
    const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
    const std::array<std::string_view, 1> child_op_types{kOpQuantize};
    const NodeUnit* quantizeLinear = GetOnlyChildOfType(
        graph_viewer, cast_node_unit, child_op_types,
        node_to_node_unit, node_unit_to_qnn_node_group);

    if (quantizeLinear == nullptr) {
        return nullptr;
    }

    std::array<const NodeUnit*, 2> node_unit_array{&cast_node_unit, quantizeLinear};
    auto node_units = gsl::make_span<const NodeUnit*>(node_unit_array.data(), 2);
    // if (CreateOrValidateOnQnn(&qnn_model_wrapper, node_units, /*validate=*/true) != Status::OK()) {
    //     return nullptr;
    // }
    return std::make_unique<CastQuantizeFusion>(node_units);
}

gsl::span<const NodeUnit* const> CastQuantizeFusion::GetNodeUnits() const {
  return gsl::span<const NodeUnit* const>{node_units_.data(), node_units_.size()};
}

Status CastQuantizeFusion::IsSupported(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  return Status::OK();
}

Status CastQuantizeFusion::AddToModelBuilder(
    QnnModelWrapper& qnn_model_wrapper, [[maybe_unused]] const logging::Logger& logger) const {
  gsl::span<const NodeUnit* const> node_units = GetNodeUnits();
  const NodeUnit* cast = node_units[0];
  const NodeUnit* quantizeLinear = node_units[1];

  // const auto* cast_builder = qnn::GetOpBuilder("Cast");
  {
    //ProcessInputs
    const auto& input_name = cast->Inputs()[0].node_arg.Name();
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
      // TODO: Skip FPtoBoolCast
      // TODO: Skip Constant cast
      std::vector<uint8_t> unpacked_tensor;
      std::vector<uint32_t> input_shape;
      ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(cast->Inputs()[0].node_arg, input_shape),
                        "Cannot get shape for QNN Cast node's input.");
      Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
      const auto* type_proto = cast->Inputs()[0].node_arg.TypeAsProto();
      ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false,  // Do not try to get the quantized type. HTP cast supports normal types.
                                                type_proto,
                                                qnn_data_type));
      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input_name);
      QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, QnnQuantParamsWrapper(),
                                           std::move(input_shape), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                        "Failed to add input tensor for QNN Cast node.");
    }
    // TODO: Skip Int64 cases
  }
  //ProcessInt64Tensors
  {
    //ProcessAttributesAndOutputs
    const auto* type_proto = quantizeLinear->Outputs()[0].node_arg.TypeAsProto();
    Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
    LOGS(logger, VERBOSE) << "Process output with Quantize";
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(true,  // Do not try to get the quantized type. HTP cast supports normal types.
                                              type_proto,
                                              qnn_data_type));
    std::vector<uint32_t> output_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(
      quantizeLinear->Outputs()[0].node_arg, output_shape), "Cannot get shape for QNN Cast node's output.");
    // TODO: Deal with scale offset
    QnnTensorWrapper output_tensorwrapper(quantizeLinear->Outputs()[0].node_arg.Name(),
                                          QNN_TENSOR_TYPE_NATIVE, // Cast is followed by QuantizeLinear
                                          qnn_data_type,
                                          QnnQuantParamsWrapper(0.00010681315325200558, 0),
                                          std::move(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                      "Failed to add output tensor for QNN Cast node.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast->Name(),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CAST,
                                                    {cast->Inputs()[0].node_arg.Name()},
                                                    {quantizeLinear->Outputs()[0].node_arg.Name()},
                                                    {},
                                                    false),
                      "Failed to add fused " + std::string(kOpCast) + " node.");
  }
  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime