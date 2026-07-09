// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class NeuronOpBuilder : public BaseOpBuilder {
 public:
  NeuronOpBuilder() : BaseOpBuilder("NeuronOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NeuronOpBuilder);

 protected:
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
};

// Limit to float type for now
static Ort::Status ProcessNodeAttribute(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const std::string& onnx_attr_key,
                                        const std::string& qnn_param_key,
                                        std::vector<std::string>& param_tensor_names,
                                        const float default_value = 1.0f) {
  OrtNodeAttrHelper node_helper(node_unit);
  float attr_value = node_helper.Get(onnx_attr_key, default_value);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), attr_value,
                                      qnn_param_key, param_tensor_names));

  return Ort::Status();
}

Ort::Status NeuronOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& logger,
                                                         bool do_op_validation) const {
  if (input_names.empty()) {
    return Ort::Status();
  }

  const std::string& op_type = node_unit.OpType();

  if (do_op_validation) {
    const auto qnn_backend_type = qnn_model_wrapper.GetQnnBackendType();

    if (op_type == "Softplus" && qnn_backend_type != QnnBackendType::CPU) {
      TensorInfo input_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
      RETURN_IF(input_info.shape.size() > 4,
                "QNN EP does not support Softplus with input rank > 4.");
    }

#if QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 21 && QNN_API_VERSION_MINOR <= 23
    // Skip QNN validation for Tanh with uint16 (quantized) output.
    // This gets around a Tanh QNN validation bug in QNN SDK 2.28.0 - 2.30.0.
    // The QNN documentation states that the output scale and offset for ufixed_point_16 should be
    // (1/32768) and -32768, respectively. However, the QNN validator incorrectly rejects these values.
    if (op_type == "Tanh") {
      TensorInfo output_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
      if (output_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
        ORT_CXX_LOG(logger,
                    ORT_LOGGING_LEVEL_INFO,
                    ("Skipping QNN validation for Tanh node '" + node_unit.Name() +
                     "' with quantized uint16 output.")
                        .c_str());
        return Ort::Status();
      }
    }
#endif
  }

  std::vector<std::string> param_tensor_names;

  // Every op handled by this builder lowers to QNN_OP_ELEMENT_WISE_NEURON and differs only by the
  // OPERATION scalar (plus, for a few ops, extra alpha/beta/threshold params handled below).
  static const std::unordered_map<std::string, uint32_t> neuron_op_to_operation = {
      {"Elu", QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU},
      {"Gelu", QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU},
      {"Relu", QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU},
      {"Sigmoid", QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID},
      {"Tanh", QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH},
      {"Softplus", QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS},
      {"HardSwish", QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH},
      {"HardSigmoid", QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID},
  };

  auto neuron_it = neuron_op_to_operation.find(op_type);
  RETURN_IF(neuron_it == neuron_op_to_operation.end(),
            ("QNN EP: NeuronOpBuilder received an unsupported op type: " + op_type).c_str());
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(),
                                         static_cast<uint32_t>(neuron_it->second),
                                         QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION, param_tensor_names));

  // Op-specific extra params. Gelu/Relu/Sigmoid/Tanh/HardSwish need nothing beyond OPERATION.
  if (op_type == "Elu") {
    RETURN_IF_ERROR(ProcessNodeAttribute(qnn_model_wrapper, node_unit, "alpha",
                                         QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA, param_tensor_names));
  } else if (op_type == "Softplus") {
    // ONNX Softplus has no attributes; set QNN defaults (beta=1, threshold=20).
    RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 1.0f,
                                        QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA, param_tensor_names));
    RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 20.0f,
                                        QNN_OP_ELEMENT_WISE_NEURON_PARAM_THRESHOLD, param_tensor_names));
  } else if (op_type == "HardSigmoid") {
    RETURN_IF_ERROR(ProcessNodeAttribute(qnn_model_wrapper, node_unit, "alpha",
                                         QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA,
                                         param_tensor_names, 0.2f));
    RETURN_IF_ERROR(ProcessNodeAttribute(qnn_model_wrapper, node_unit, "beta",
                                         QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA,
                                         param_tensor_names, 0.5f));
  }

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(op_type));
}

/**
 * Overrides offset and scale quantization parameters for operators (e.g., Sigmoid or Tanh) that require
 * specific values. Returns true if the quantization parameters were overridden.
 *
 * \param op_type The ONNX operator type.
 * \param qnn_data_type The QNN tensor data type.
 * \param quant_params Output scale/offset parameter that may be overridden.
 * \return True if the offset and scale were overridden.
 */
static bool OverrideQuantParams(const std::string& op_type, Qnn_DataType_t qnn_data_type,
                                Qnn_ScaleOffset_t& quant_params) {
  const int32_t orig_offset = quant_params.offset;
  const float orig_scale = quant_params.scale;

  if (op_type == "Sigmoid" || op_type == "HardSigmoid") {
    switch (qnn_data_type) {
      case QNN_DATATYPE_UFIXED_POINT_16:
        quant_params.offset = 0;
        quant_params.scale = 1.0f / 65536.0f;
        break;
      case QNN_DATATYPE_SFIXED_POINT_16:
        quant_params.offset = 0;
        quant_params.scale = 1.0f / 32768.0f;
        break;
      default:
        break;  // Do nothing.
    }
  }

  if (op_type == "Tanh") {
    switch (qnn_data_type) {
      case QNN_DATATYPE_UFIXED_POINT_16:
        quant_params.offset = -32768;
        quant_params.scale = 1.0f / 32768.0f;
        break;
      case QNN_DATATYPE_SFIXED_POINT_16:
        quant_params.offset = 0;
        quant_params.scale = 1.0f / 32768.0f;
        break;
      default:
        break;  // Do nothing.
    }
  }

  return quant_params.offset != orig_offset || quant_params.scale != orig_scale;
}

Ort::Status NeuronOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      const Ort::Logger& logger,
                                                      const std::vector<std::string>& input_names,
                                                      size_t output_index,
                                                      Qnn_DataType_t qnn_data_type,
                                                      QnnQuantParamsWrapper& quant_param) const {
  ORT_UNUSED_PARAMETER(input_names);
  const std::string& op_type = node_unit.OpType();

  // Override output quantization parameters for uint16 QDQ Sigmoid or Tanh.
  // QNN requires 16-bit QDQ Sigmoid and Tanh to use specific output scale and zero-point values
  // regardless of floating-point range.
  if (op_type == "Sigmoid" || op_type == "Tanh" || op_type == "HardSigmoid") {
    const auto& outputs = node_unit.Outputs();
    RETURN_IF_NOT(output_index < outputs.size(),
                  ("Invalid output index in OverrideOutputQuantParam for op " + op_type).c_str());

    const auto& output = node_unit.Outputs()[output_index];
    const std::string& output_name = output.name;

    if (quant_param.IsPerTensor(/*include_bw*/ false)) {
      if (OverrideQuantParams(op_type, qnn_data_type, quant_param.Get().scaleOffsetEncoding)) {
        const int32_t offset = quant_param.Get().scaleOffsetEncoding.offset;
        const float scale = quant_param.Get().scaleOffsetEncoding.scale;

        std::ostringstream oss;
        oss << "QNN requires that 16-bit quantized " << op_type
            << " operators use offset/scale values "
            << "of <" << offset << ", " << scale
            << ">. QNN EP will override the original values for output " << output_name;
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());
        RETURN_IF(qnn_model_wrapper.IsQnnTensorWrapperExist(output_name),
                  ("QNN EP is unable to override output quantization parameters for " + op_type +
                   " operator. Node name: " + node_unit.Name() + ", output name: " + output_name)
                      .c_str());
      }
    }
  }

  return Ort::Status();
}

void CreateNeuronOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<NeuronOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
