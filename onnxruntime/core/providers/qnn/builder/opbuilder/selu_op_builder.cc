// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// ONNX SeLU has no native QNN op. It is lowered into two QNN ops:
//   1. ElementwiseNeuron (ELU operation, alpha attribute)  -> elu_output
//   2. ElementwiseBinary (Multiply, gamma constant tensor) -> final output
class SeluOpBuilder : public BaseOpBuilder {
 public:
  SeluOpBuilder() : BaseOpBuilder("SeluOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SeluOpBuilder);

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status SeluOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Processing Selu operator: " + node_unit.Name()).c_str());

  const std::string& input_name = input_names[0];
  const std::string& output_name = node_unit.Outputs()[0].name;

  OrtNodeAttrHelper node_helper(node_unit);
  // Use ONNX opset 22 precise defaults (opset 1 uses the same values rounded to fewer digits).
  const float alpha = node_helper.Get("alpha", 1.67326319217681884765625f);
  const float gamma = node_helper.Get("gamma", 1.05070102214813232421875f);

  TensorInfo input_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));

  // -----------------------------------------------------------------------
  // Step 1: ELU — applies alpha*exp(x)-alpha for x<=0
  // -----------------------------------------------------------------------
  const std::string elu_output_name = utils::UniqueNameGenerator().New(node_unit.Name() + "_elu_out");

  QnnTensorWrapper elu_output_wrapper(elu_output_name,
                                      QNN_TENSOR_TYPE_NATIVE,
                                      input_info.qnn_data_type,
                                      QnnQuantParamsWrapper(),
                                      std::vector<uint32_t>(input_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(elu_output_wrapper)),
                "Selu: failed to add intermediate ELU output tensor.");

  std::vector<std::string> elu_param_names;

  // alpha param
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), alpha,
                                      QNN_OP_ELU_PARAM_ALPHA, elu_param_names));

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit.Name() + "_elu"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELU,
                                                {input_name},
                                                {elu_output_name},
                                                std::move(elu_param_names),
                                                do_op_validation),
                "Selu: failed to create ELU node.");

  // -----------------------------------------------------------------------
  // Step 2: gamma constant tensor (scalar, matches input data type)
  // -----------------------------------------------------------------------
  const std::string gamma_tensor_name = utils::UniqueNameGenerator().New(node_unit.Name() + "_gamma");
  std::vector<uint8_t> gamma_data;

  if (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) {
    gamma_data.resize(sizeof(Ort::Float16_t));
    Ort::Float16_t gamma_fp16(gamma);
    memcpy(gamma_data.data(), &gamma_fp16.val, sizeof(Ort::Float16_t));
  } else if (input_info.qnn_data_type == QNN_DATATYPE_BFLOAT_16) {
    gamma_data.resize(sizeof(Ort::BFloat16_t));
    Ort::BFloat16_t gamma_bf16(gamma);
    memcpy(gamma_data.data(), &gamma_bf16.val, sizeof(Ort::BFloat16_t));
  } else {
    gamma_data.resize(sizeof(float));
    memcpy(gamma_data.data(), &gamma, sizeof(float));
  }

  QnnTensorWrapper gamma_wrapper(gamma_tensor_name,
                                 QNN_TENSOR_TYPE_STATIC,
                                 input_info.qnn_data_type,
                                 QnnQuantParamsWrapper(),
                                 {1},
                                 std::move(gamma_data));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(gamma_wrapper)),
                "Selu: failed to add gamma constant tensor.");

  // -----------------------------------------------------------------------
  // Step 3: Multiply (gamma * elu_output) → final output
  // -----------------------------------------------------------------------
  const Qnn_TensorType_t output_tensor_type =
      qnn_model_wrapper.IsGraphOutput(output_name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  QnnTensorWrapper output_wrapper(output_name,
                                  output_tensor_type,
                                  input_info.qnn_data_type,
                                  input_info.quant_param.Copy(),
                                  std::vector<uint32_t>(input_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                "Selu: failed to add output tensor.");

  std::string gamma_mul_name = utils::UniqueNameGenerator().New(node_unit.Name() + "_gamma_mul");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(gamma_mul_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                {gamma_tensor_name, elu_output_name},
                                                {output_name},
                                                {},
                                                do_op_validation),
                "Selu: failed to create Multiply node.");

  return Ort::Status();
}

void CreateSeluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SeluOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
