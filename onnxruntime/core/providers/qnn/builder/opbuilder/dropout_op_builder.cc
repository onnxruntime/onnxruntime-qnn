// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// ONNX Dropout has no native QNN op. In inference mode (training_mode=false),
// it is a pure identity on the data input. Output[0] is mapped to a QNN
// Transpose with identity permutation (same approach as IdentityOpBuilder).
// The optional mask output[1] is filled with a constant all-ones bool tensor.
class DropoutOpBuilder : public BaseOpBuilder {
 public:
  DropoutOpBuilder() : BaseOpBuilder("DropoutOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DropoutOpBuilder);

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

  Ort::Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const Ort::Logger& logger,
                                       const std::vector<std::string>& input_names,
                                       size_t output_index,
                                       Qnn_DataType_t qnn_data_type,
                                       QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const;
};

Ort::Status DropoutOpBuilder::ExplicitOpCheck(QnnModelWrapper& qnn_model_wrapper,
                                               const OrtNodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();

  // training_mode is input[2] (opset >= 12). If present and non-empty, it must be a
  // constant false — QNN EP supports inference mode only.
  if (inputs.size() > 2 && inputs[2].Exists()) {
    const std::string& training_mode_name = inputs[2].name;
    RETURN_IF(!qnn_model_wrapper.IsConstantInput(training_mode_name),
              "Dropout: dynamic training_mode is not supported.");

    TensorInfo tm_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], tm_info));
    std::vector<uint8_t> tm_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(tm_info.initializer_tensor, tm_bytes));
    RETURN_IF(!tm_bytes.empty() && tm_bytes[0] != 0,
              "Dropout: training_mode=true is not supported.");
  }

  return Ort::Status();
}

Ort::Status DropoutOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const Ort::Logger& logger,
                                             std::vector<std::string>& input_names,
                                             bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_ERROR(ExplicitOpCheck(qnn_model_wrapper, node_unit));
  }

  // Only process data (input[0]). ratio and training_mode are ignored in inference mode.
  return ProcessInput(qnn_model_wrapper, node_unit.Inputs()[0], logger, input_names);
}

Ort::Status DropoutOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                           const OrtNodeUnit& node_unit,
                                                           std::vector<std::string>&& input_names,
                                                           const Ort::Logger& logger,
                                                           bool do_op_validation) const {
  // output[0]: identity via Transpose with identity permutation.
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape), "Cannot get shape");
  uint32_t rank = static_cast<uint32_t>(input_shape.size());

  std::vector<uint32_t> perm_data(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    perm_data[i] = i;
  }

  QnnParamWrapper perm_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                             std::vector<uint32_t>{rank}, std::move(perm_data));
  std::vector<std::string> param_tensor_names;
  param_tensor_names.push_back(perm_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(perm_param));

  RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names),
                                 std::move(param_tensor_names), logger, do_op_validation,
                                 QNN_OP_TRANSPOSE));

  // output[1]: optional mask — constant all-ones bool tensor matching the input shape.
  const auto& outputs = node_unit.Outputs();
  if (outputs.size() > 1 && outputs[1].Exists()) {
    const std::string& mask_name = outputs[1].name;
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(mask_name);
    const Qnn_TensorType_t mask_tensor_type =
        is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

    size_t num_elements = 1;
    for (uint32_t dim : input_shape) {
      num_elements *= dim;
    }

    std::vector<uint8_t> mask_data(num_elements, 1u);
    QnnTensorWrapper mask_wrapper(mask_name,
                                  mask_tensor_type,
                                  QNN_DATATYPE_BOOL_8,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(input_shape),
                                  std::move(mask_data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mask_wrapper)),
                  "Dropout: failed to add mask output tensor.");
  }

  return Ort::Status();
}

Ort::Status DropoutOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        const Ort::Logger& logger,
                                                        const std::vector<std::string>& input_names,
                                                        size_t output_index,
                                                        Qnn_DataType_t qnn_data_type,
                                                        QnnQuantParamsWrapper& quant_param) const {
  // output[0] is a pass-through — copy input quant params so scale/offset are preserved.
  if (output_index == 0 && quant_param.IsPerTensor()) {
    return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                    0, output_index, qnn_data_type, quant_param);
  }
  return Ort::Status();
}

void CreateDropoutOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<DropoutOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
