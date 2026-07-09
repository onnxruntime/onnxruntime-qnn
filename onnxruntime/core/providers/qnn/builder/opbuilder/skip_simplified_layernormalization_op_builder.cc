// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Handles com.microsoft.SkipSimplifiedLayerNormalization.
//
// Inputs:  [0] input, [1] skip, [2] gamma, [3] bias (optional)
// Outputs: [0] output (required); training outputs [1],[2],[3] are not supported.
//
// Decomposed as:
//   sum = Add(input, skip)
//   sum = Add(sum, bias)   [if bias present]
//   output = RMSNorm(sum, gamma, epsilon)
class SkipSimplifiedLayerNormOpBuilder : public BaseOpBuilder {
 public:
  SkipSimplifiedLayerNormOpBuilder() : BaseOpBuilder("SkipSimplifiedLayerNormOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SkipSimplifiedLayerNormOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status EmitAddNode(QnnModelWrapper& qnn_model_wrapper,
                          const OrtNodeUnit& node_unit,
                          const std::string& lhs_name,
                          const std::string& rhs_name,
                          const std::string& output_name,
                          const TensorInfo& output_info,
                          bool do_op_validation) const ORT_MUST_USE_RESULT;
};

Ort::Status SkipSimplifiedLayerNormOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            const Ort::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  RETURN_IF(outputs.size() > 1,
            "QNN EP SkipSimplifiedLayerNorm: only single output (Y) is supported.");

  constexpr size_t INPUT_IDX = 0;
  constexpr size_t SKIP_IDX = 1;
  constexpr size_t GAMMA_IDX = 2;

  RETURN_IF(inputs.size() < 3, "QNN EP SkipSimplifiedLayerNorm requires input, skip, and gamma.");
  RETURN_IF_NOT(inputs[INPUT_IDX].Exists(), "QNN EP SkipSimplifiedLayerNorm: input[0] must be present.");
  RETURN_IF_NOT(inputs[SKIP_IDX].Exists(), "QNN EP SkipSimplifiedLayerNorm: input[1] (skip) must be present.");
  RETURN_IF_NOT(inputs[GAMMA_IDX].Exists(), "QNN EP SkipSimplifiedLayerNorm: input[2] (gamma) must be present.");

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[INPUT_IDX].shape, input_shape),
                "Cannot get shape of input 0");
  RETURN_IF(input_shape.size() > 3, "QNN EP SkipSimplifiedLayerNorm only supports input rank <= 3 (2D or 3D)");

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].shape, output_shape),
                "Cannot get shape of output 0");
  RETURN_IF(output_shape.size() > 3, "QNN EP SkipSimplifiedLayerNorm only supports output rank <= 3 (2D or 3D)");

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Ort::Status SkipSimplifiedLayerNormOpBuilder::EmitAddNode(QnnModelWrapper& qnn_model_wrapper,
                                                          const OrtNodeUnit& node_unit,
                                                          const std::string& lhs_name,
                                                          const std::string& rhs_name,
                                                          const std::string& output_name,
                                                          const TensorInfo& output_info,
                                                          bool do_op_validation) const {
  QnnTensorWrapper out_tensor(output_name, QNN_TENSOR_TYPE_NATIVE,
                              output_info.qnn_data_type, output_info.quant_param.Copy(),
                              std::vector<uint32_t>(output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(out_tensor)),
                "Failed to add SkipSimplifiedLayerNorm Add output tensor");

  Qnn_Scalar_t op_scalar = QNN_SCALAR_INIT;
  op_scalar.dataType = QNN_DATATYPE_UINT_32;
  op_scalar.uint32Value = QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD;
  // Suffix output_name to keep param names unique across the two Add nodes from the same ONNX node.
  QnnParamWrapper op_param(node_unit.Index(), node_unit.Name() + "_" + output_name,
                           QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION, op_scalar);
  const std::string param_name = op_param.GetParamTensorName();
  RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(op_param)),
                "Failed to add SkipSimplifiedLayerNorm Add op param");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                    utils::UniqueNameGenerator().New(node_unit, "_add_" + output_name),
                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                    QNN_OP_ELEMENT_WISE_BINARY,
                    {lhs_name, rhs_name},
                    {output_name},
                    {param_name},
                    do_op_validation),
                "Failed to create SkipSimplifiedLayerNorm Add QNN node");

  return Ort::Status();
}

Ort::Status SkipSimplifiedLayerNormOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            const Ort::Logger& logger,
                                                            std::vector<std::string>& input_names,
                                                            bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  constexpr size_t INPUT_IDX = 0;
  constexpr size_t SKIP_IDX = 1;
  constexpr size_t GAMMA_IDX = 2;
  constexpr size_t BIAS_IDX = 3;

  const bool has_bias = inputs.size() > BIAS_IDX && inputs[BIAS_IDX].Exists();

  std::vector<std::string> add_inputs;
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[INPUT_IDX], logger, add_inputs));
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[SKIP_IDX], logger, add_inputs));

  // Intermediate tensor inherits shape/dtype/quant from input[0].
  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[INPUT_IDX], input_info));

  const std::string sum_name = node_unit.Name() + "_skip_sum";
  RETURN_IF_ERROR(EmitAddNode(qnn_model_wrapper, node_unit,
                              add_inputs[0], add_inputs[1], sum_name, input_info, do_op_validation));

  std::string final_sum_name = sum_name;

  if (has_bias) {
    std::vector<std::string> bias_names;
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[BIAS_IDX], logger, bias_names));

    const std::string sum_bias_name = node_unit.Name() + "_skip_bias_sum";
    RETURN_IF_ERROR(EmitAddNode(qnn_model_wrapper, node_unit,
                                sum_name, bias_names[0], sum_bias_name, input_info, do_op_validation));
    final_sum_name = sum_bias_name;
  }

  // input_names fed to QNN_OP_RMS_NORM: [sum, gamma]
  input_names.push_back(final_sum_name);
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[GAMMA_IDX], logger, input_names));

#if !defined(QNN_SDK_VERSION_MINOR) || (QNN_SDK_VERSION_MAJOR == 2 && QNN_SDK_VERSION_MINOR < 49)
  if (IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    TensorInfo scale_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[GAMMA_IDX], scale_info));

    Qnn_DataType_t beta_data_type = QNN_DATATYPE_UFIXED_POINT_8;
    if (scale_info.qnn_data_type == QNN_DATATYPE_FLOAT_32 ||
        scale_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) {
      beta_data_type = scale_info.qnn_data_type;
    }

    QnnQuantParamsWrapper beta_quant_param;
    if (scale_info.quant_param.IsQuantized()) {
      beta_quant_param = QnnQuantParamsWrapper::PerTensor(1.0f, 0);
    }

    const size_t beta_size = utils::GetQnnTensorDataSizeInBytes(scale_info.shape, beta_data_type);
    std::vector<uint8_t> beta_data(beta_size, 0);
    const std::string beta_name = node_unit.Name() + "_beta_dummy";
    QnnTensorWrapper beta_tensor(beta_name, QNN_TENSOR_TYPE_STATIC, beta_data_type,
                                 std::move(beta_quant_param), std::move(scale_info.shape),
                                 std::move(beta_data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(beta_tensor)),
                  "Failed to add dummy beta tensor for SkipSimplifiedLayerNorm.");
    input_names.push_back(beta_name);
  }
#endif

  return Ort::Status();
}

Ort::Status SkipSimplifiedLayerNormOpBuilder::ProcessAttributesAndOutputs(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    std::vector<std::string>&& input_names,
    const Ort::Logger& logger,
    bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                      epsilon, QNN_OP_RMS_NORM_PARAM_EPSILON, param_tensor_names));

  // No axis attribute — always normalizes over the last dimension.
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape),
                "Cannot get shape of input 0");
  const uint32_t last_axis = static_cast<uint32_t>(input_shape.size() - 1);

  std::vector<uint32_t> axes = {last_axis};
  std::vector<uint32_t> axes_shape = {1};
  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), QNN_OP_RMS_NORM_PARAM_AXES,
                             std::move(axes_shape), std::move(axes));
  param_tensor_names.push_back(axes_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axes_param));

  RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                 std::move(input_names),
                                 std::move(param_tensor_names),
                                 logger,
                                 do_op_validation,
                                 QNN_OP_RMS_NORM));
  return Ort::Status();
}

void CreateSkipSimplifiedLayerNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SkipSimplifiedLayerNormOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
