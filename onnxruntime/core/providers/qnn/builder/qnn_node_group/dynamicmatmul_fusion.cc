#include "core/providers/qnn/builder/qnn_node_group/dynamicmatmul_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& dynamicquantizelinear_node_unit,
                                    const NodeUnit& matmulinteger_node_unit,
                                    const NodeUnit& cast_node_unit,
                                    const NodeUnit& mulright_node_unit,
                                    const NodeUnit& mul_node_unit,
                                    const NodeUnit& add_node_unit,
                                    bool validate);

std::unique_ptr<IQnnNodeGroup> DynamicMatMulFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& matmulinteger_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);

  // Looking for a standalone MatMulInteger to start the sequence.
  if (matmulinteger_node_unit.OpType() != "MatMulInteger" ||
      matmulinteger_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Get DynamicQuantizeLinear NodeUnit
  const std::array<std::string_view, 1> matmulinteger_parent_types = {"DynamicQuantizeLinear"};
  const NodeUnit* p_dynamicquantizelinear_node_unit = GetParentOfType(graph_viewer,
                                                                      matmulinteger_node_unit,
                                                                      matmulinteger_parent_types,
                                                                      node_to_node_unit,
                                                                      node_unit_to_qnn_node_group);
  if (p_dynamicquantizelinear_node_unit == nullptr) {
    return nullptr;
  }

  // Get Cast NodeUnit
  const std::array<std::string_view, 1> matmulinteger_child_types = {"Cast"};
  const NodeUnit* p_cast_node_unit = GetOnlyChildOfType(graph_viewer,
                                                        matmulinteger_node_unit,
                                                        matmulinteger_child_types,
                                                        node_to_node_unit,
                                                        node_unit_to_qnn_node_group);
  if (p_cast_node_unit == nullptr) {
    return nullptr;
  }

  // Get Right Mul NodeUnit
  const std::array<std::string_view, 1> cast_child_types = {"Mul"};
  const NodeUnit* p_mul_node_unit = GetOnlyChildOfType(graph_viewer,
                                                       *p_cast_node_unit,
                                                       cast_child_types,
                                                       node_to_node_unit,
                                                       node_unit_to_qnn_node_group);
  if (p_mul_node_unit == nullptr) {
    return nullptr;
  }

  // Get Mul NodeUnit
  const std::array<std::string_view, 1> mul_parent_types = {"Mul"};
  const NodeUnit* p_mul_right_node_unit = GetParentOfType(graph_viewer,
                                                          *p_mul_node_unit,
                                                          mul_parent_types,
                                                          node_to_node_unit,
                                                          node_unit_to_qnn_node_group);
  if (p_mul_right_node_unit == nullptr) {
    return nullptr;
  }

  // Get Add NodeUnit
  const std::array<std::string_view, 1> mul_child_types = {"Add"};
  const NodeUnit* p_add_node_unit = GetOnlyChildOfType(graph_viewer,
                                                       *p_mul_node_unit,
                                                       mul_child_types,
                                                       node_to_node_unit,
                                                       node_unit_to_qnn_node_group);
  if (p_add_node_unit == nullptr) {
    return nullptr;
  }

  if (Status status = CreateOrValidateOnQnn(qnn_model_wrapper,
                                            *p_dynamicquantizelinear_node_unit,
                                            matmulinteger_node_unit,
                                            *p_cast_node_unit,
                                            *p_mul_right_node_unit,
                                            *p_mul_node_unit,
                                            *p_add_node_unit,
                                            true);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<DynamicMatMulFusion>(*p_dynamicquantizelinear_node_unit,
                                               matmulinteger_node_unit,
                                               *p_cast_node_unit,
                                               *p_mul_right_node_unit,
                                               *p_mul_node_unit,
                                               *p_add_node_unit);
}

DynamicMatMulFusion::DynamicMatMulFusion(const NodeUnit& DynamicQuantizeLinear_node_unit,
                                         const NodeUnit& MatMulInteger_node_unit,
                                         const NodeUnit& Cast_node_unit,
                                         const NodeUnit& MulRight_node_unit,
                                         const NodeUnit& Mul_node_unit,
                                         const NodeUnit& Add_node_unit)
    : node_units_{&DynamicQuantizeLinear_node_unit,
                  &MatMulInteger_node_unit,
                  &Cast_node_unit,
                  &MulRight_node_unit,
                  &Mul_node_unit,
                  &Add_node_unit} {
}

Status DynamicMatMulFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], *node_units_[3], *node_units_[4], *node_units_[5], true);
}

Status DynamicMatMulFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], *node_units_[3], *node_units_[4], *node_units_[5], false);
}

gsl::span<const NodeUnit* const> DynamicMatMulFusion::GetNodeUnits() const {
  return node_units_;
}

const NodeUnit* DynamicMatMulFusion::GetTargetNodeUnit() const {
  return node_units_[1];
}

Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& dynamicquantizelinear_node_unit,
                             const NodeUnit& matmulinteger_node_unit,
                             const NodeUnit& cast_node_unit,
                             const NodeUnit& mulright_node_unit,
                             const NodeUnit& mul_node_unit,
                             const NodeUnit& add_node_unit,
                             bool validate) {
  assert(dynamicquantizelinear_node_unit.OpType() == "DynamicQuantizeLinear" &&
         matmulinteger_node_unit.OpType() == "MatMulInteger" &&
         cast_node_unit.OpType() == "Cast" &&
         mulright_node_unit.OpType() == "Mul" &&
         mul_node_unit.OpType() == "Mul" &&
         add_node_unit.OpType() == "Add");
  const auto& node_name = utils::GetNodeName(matmulinteger_node_unit);
  const NodeUnitIODef& input_1_def = dynamicquantizelinear_node_unit.Inputs()[0];
  const NodeUnitIODef& input_2_def = matmulinteger_node_unit.Inputs()[1];
  const NodeUnitIODef& input_3_def = add_node_unit.Inputs()[1];
  const NodeUnitIODef& output_def = add_node_unit.Outputs()[0];

  QnnTensorWrapper input_1_tensor;
  // QnnTensorWrapper input_2_tensor;
  QnnTensorWrapper input_3_tensor;
  QnnTensorWrapper output_tensor;

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_1_def, input_1_tensor));

  // Prepare FXP Weight
  TensorInfo input2_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_2_def, input2_info));
  const std::string& input2_name = input_2_def.node_arg.Name();
  std::vector<uint8_t> initializer_data;
  if (input2_info.is_initializer) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input2_info.initializer_tensor, initializer_data));
  }

  std::vector<uint32_t> op_shape = {
      input2_info.shape[0],
      input2_info.shape[1]};

  QnnQuantParamsWrapper weight_qparams;
  // get weight_scale value from input[1] of mul_right
  TensorInfo w_scale_info = {};
  const NodeUnitIODef& w_scale_def = mulright_node_unit.Inputs()[1];
  std::vector<uint8_t> w_scale_initializer_data;
  float weight_scale = 0.0f;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(w_scale_def, w_scale_info));
  if (w_scale_info.is_initializer) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*w_scale_info.initializer_tensor, w_scale_initializer_data));
    const float32_t* tensor_data = reinterpret_cast<const float32_t*>(w_scale_initializer_data.data());
    weight_scale = static_cast<float32_t>(*tensor_data);
  }

  // get weight_offset value from input[3] of matmulinteger
  TensorInfo w_offset_info = {};
  const NodeUnitIODef& w_offset_def = matmulinteger_node_unit.Inputs()[3];
  std::vector<uint8_t> w_offset_initializer_data;
  int32_t weight_offset = 0;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(w_offset_def, w_offset_info));
  if (w_offset_info.is_initializer) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*w_offset_info.initializer_tensor, w_offset_initializer_data));
    const uint8_t* tensor_data = reinterpret_cast<const uint8_t*>(w_offset_initializer_data.data());
    weight_offset = -(static_cast<int32_t>(*tensor_data));
  }

  weight_qparams = QnnQuantParamsWrapper(weight_scale, weight_offset);

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input2_name);
  QnnTensorWrapper input_2_tensor(input2_name, tensor_type, QNN_DATATYPE_SFIXED_POINT_8,
                                  std::move(weight_qparams), std::move(op_shape),
                                  std::move(initializer_data));

  // ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_2_def, input_2_tensor));

  // Prepare Bias and outputs
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_3_def, input_3_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_MAT_MUL,
                                                          {input_1_tensor.GetQnnTensor(), input_2_tensor.GetQnnTensor(), input_3_tensor.GetQnnTensor()},
                                                          {output_tensor.GetQnnTensor()},
                                                          {}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_1_tensor)), "Failed to add input1");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_2_tensor)), "Failed to add input2");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_3_tensor)), "Failed to add input3");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_MAT_MUL,
                                                      {input_1_def.node_arg.Name(), input_2_def.node_arg.Name(), input_3_def.node_arg.Name()},
                                                      {output_def.node_arg.Name()},
                                                      {},
                                                      validate),
                      "Failed to Fuse DynamicMatMulInteger Sequence into MatMul node.");
  }

  return Status();
}
}  // namespace qnn
}  // namespace onnxruntime