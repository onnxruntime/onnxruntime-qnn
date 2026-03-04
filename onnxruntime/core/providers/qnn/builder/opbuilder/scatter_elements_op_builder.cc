// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <utility>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class ScatterElementsOpBuilder : public BaseOpBuilder {
 public:
  ScatterElementsOpBuilder() : BaseOpBuilder("ScatterElementsOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScatterElementsOpBuilder);

 protected:
  Ort::Status ProcessInput(QnnModelWrapper& qnn_model_wrapper,
                           const OrtNodeUnitIODef& input,
                           const Ort::Logger& logger,
                           std::vector<std::string>& input_names,
                           bool allow_int_32) const;

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
  static constexpr std::array<std::string_view, 4> scatterelements_supported_reduction = {"none", "add", "mul", "max"};
};

Ort::Status ScatterElementsOpBuilder::ProcessInput(QnnModelWrapper& qnn_model_wrapper,
                                                   const OrtNodeUnitIODef& input,
                                                   const Ort::Logger& logger,
                                                   std::vector<std::string>& input_names,
                                                   bool allow_int_32) const {
  const std::string& input_name = input.name;

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    const auto& qnn_tensor = qnn_model_wrapper.GetQnnTensorWrapper(input_name);
    Qnn_DataType_t tensor_type = qnn_tensor.GetTensorDataType();
    // Input tensor exist and supported.
    if (tensor_type != QNN_DATATYPE_INT_64 && (tensor_type != QNN_DATATYPE_INT_32 || allow_int_32)) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_name).c_str());
      input_names.push_back(input_name);
      return Ort::Status();
    }
  }

  const std::string& tensor_name = input.name;

  TensorInfo tensor_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input, tensor_info));

  std::vector<uint8_t> unpacked_tensor;
  if (tensor_info.is_initializer) {
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(tensor_info.initializer_tensor, unpacked_tensor));
  }

  Qnn_DataType_t data_type = tensor_info.qnn_data_type;

  // QNN doesn't support int64_t. Data and updates in ScatterElements doesn't support int32_t.

  // Casting initializers to int32_t if allow_int32, otherwise casts to float32.
  Qnn_DataType_t casted_data_type = allow_int_32 ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_FLOAT_32;
  bool initializer_OOB = false;  // flag for initalizer has out of boundary after convertion.
  if (unpacked_tensor.size() && (data_type == QNN_DATATYPE_INT_64 || (data_type == QNN_DATATYPE_INT_32 && !allow_int_32))) {
    const size_t num_elems = (data_type == QNN_DATATYPE_INT_64) ? unpacked_tensor.size() / sizeof(int64_t) : unpacked_tensor.size() / sizeof(int32_t);
    std::vector<uint8_t> cast_data;
    cast_data.resize(num_elems * (allow_int_32 ? sizeof(int32_t) : sizeof(float)));

    // Cast int 64 to float 32/int 32
    if (data_type == QNN_DATATYPE_INT_64) {
      gsl::span<int64_t> origin_values{reinterpret_cast<int64_t*>(unpacked_tensor.data()), num_elems};

      if (allow_int_32) {
        gsl::span<int32_t> new_values(reinterpret_cast<int32_t*>(cast_data.data()), num_elems);
        for (size_t i = 0; i < num_elems; i++) {
          int64_t v_ = origin_values[i];
          if (!initializer_OOB &&
              (v_ < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
               v_ > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))) {
            initializer_OOB = true;
          }

          new_values[i] = static_cast<int32_t>(origin_values[i]);
          ;
        }
      } else {
        gsl::span<float> new_values(reinterpret_cast<float*>(cast_data.data()), num_elems);
        for (size_t i = 0; i < num_elems; i++) {
          int64_t v_ = origin_values[i];
          if (!initializer_OOB &&
              (v_ < static_cast<int64_t>(std::numeric_limits<float>::min()) ||
               v_ > static_cast<int64_t>(std::numeric_limits<float>::max()))) {
            initializer_OOB = true;
          }

          new_values[i] = static_cast<float>(origin_values[i]);
        }
      }
    }

    if (initializer_OOB) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Initializer value out of boundary: " + input_name).c_str());
    }

    // Cast int 32 to float
    if (data_type == QNN_DATATYPE_INT_32 && !allow_int_32) {
      gsl::span<int32_t> origin_values{reinterpret_cast<int32_t*>(unpacked_tensor.data()), num_elems};
      gsl::span<float> new_values(reinterpret_cast<float*>(cast_data.data()), num_elems);
      for (size_t i = 0; i < num_elems; i++) {
        new_values[i] = static_cast<float>(origin_values[i]);
      }
    }

    // Only when there is an unsupported initializer dtype, data_type is updated to casted_data_type.
    data_type = casted_data_type;
    unpacked_tensor = std::move(cast_data);
  }

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(tensor_name);
  Qnn_TensorMemType_t mem_type = QNN_TENSORMEMTYPE_RAW;
  QnnTensorWrapper tensor_wrapper = QnnTensorWrapper(tensor_name,
                                                     tensor_type,
                                                     data_type,
                                                     std::move(tensor_info.quant_param),
                                                     std::vector<uint32_t>(tensor_info.shape),
                                                     std::move(unpacked_tensor),
                                                     mem_type);

  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)), "Failed to add tensor.");

  // Cast to int 32 or float when not an initializer.
  // Here checks if the tensor created is supported, so using data_type on L141 is correct.
  if (!tensor_info.is_initializer && (data_type == QNN_DATATYPE_INT_64 || (data_type == QNN_DATATYPE_INT_32 && !allow_int_32))) {
    std::string cast_32_name = utils::GetUniqueName(tensor_name, allow_int_32 ? "_Cast_int_32" : "_Cast_fp_32");
    std::string cast_32_output_name = utils::GetUniqueName(tensor_name, allow_int_32 ? "_Cast_int_32" : "_Cast_fp_32");
    QnnTensorWrapper cast_fp_32(cast_32_output_name,
                                QNN_TENSOR_TYPE_NATIVE,
                                allow_int_32 ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_FLOAT_32,
                                QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(tensor_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_fp_32)),
                  "Failed to add output tensor for QNN Cast node.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_32_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_CAST,
                                                  {tensor_name},
                                                  {cast_32_output_name},
                                                  {},
                                                  false),
                  "Failed to create QNN Cast node.");
    input_names.push_back(cast_32_output_name);
  } else {
    input_names.push_back(input_name);
  }

  return Ort::Status();
}

Ort::Status ScatterElementsOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const OrtNodeUnit& node_unit,
                                                    const Ort::Logger& logger,
                                                    std::vector<std::string>& input_names,
                                                    bool do_op_validation) const {
  if (do_op_validation) {
    // QNN ScatterElements doesn't support MIN reduction
    OrtNodeAttrHelper node_helper(node_unit);
    std::string reduction = node_helper.Get("reduction", "none");
    RETURN_IF_NOT(utils::ArrayHasString(scatterelements_supported_reduction, reduction),
                  ("ScatterElements does not support reduction " + reduction).c_str());
  }
  const auto& inputs = node_unit.Inputs();
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    // indices: input[1] allows int32.
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names, input_i == 1));
  }

  return Ort::Status();
}

// Process Reduction attribute of ScatterElements op
Ort::Status ProcessReductionAttribute(QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& node_unit,
                                      std::vector<std::string>& param_tensor_names) {
  OrtNodeAttrHelper node_helper(node_unit);
  std::string reduction = node_helper.Get("reduction", "none");
  Qnn_Scalar_t reduction_qnn_scalar = QNN_SCALAR_INIT;
  reduction_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
  if ("none" == reduction) {
    reduction_qnn_scalar.uint32Value = QNN_OP_SCATTER_ELEMENTS_REDUCTION_NONE;
  } else if ("add" == reduction) {
    reduction_qnn_scalar.uint32Value = QNN_OP_SCATTER_ELEMENTS_REDUCTION_ADD;
  } else if ("mul" == reduction) {
    reduction_qnn_scalar.uint32Value = QNN_OP_SCATTER_ELEMENTS_REDUCTION_MUL;
  } else if ("max" == reduction) {
    reduction_qnn_scalar.uint32Value = QNN_OP_SCATTER_ELEMENTS_REDUCTION_MAX;
  } else {
    return MAKE_EP_FAIL("ScatterElements support only reduction:{none, add, mul, max}.");
  }
  QnnParamWrapper reduction_param(node_unit.Index(), node_unit.Name(), QNN_OP_SCATTER_ELEMENTS_PARAM_REDUCTION,
                                  reduction_qnn_scalar);
  param_tensor_names.push_back(reduction_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(reduction_param));

  return Ort::Status();
}

Ort::Status ScatterElementsOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                                                  std::vector<std::string>&& input_names, const Ort::Logger& logger,
                                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  // Process axis attribute
  int32_t default_axis = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, default_axis));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(), QNN_OP_SCATTER_ELEMENTS_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  // Process reduction attribute
  RETURN_IF_ERROR(ProcessReductionAttribute(qnn_model_wrapper, node_unit, param_tensor_names));

  // Create ScatterElements -> fp32 -> Optional(Cast -> int32) -> Optional(Cast -> int 64)
  // Check if we need to add a cast node for int64
  const auto& outputs = node_unit.Outputs();
  const auto& output_name = outputs[0].name;
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));
  // Qnn_DataType_t supported_qnn_data_type = GetSupportedOutputDataType(0, output_info.qnn_data_type);
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  // Cast to int64 when model output is int64
  bool need_int64_cast = false;
  if (is_graph_output) {
    if (output_info.qnn_data_type == QNN_DATATYPE_INT_64 || output_info.qnn_data_type == QNN_DATATYPE_UINT_64) {
      need_int64_cast = true;
    }
  }

  bool need_int32_cast = false;
  if (output_info.qnn_data_type == QNN_DATATYPE_INT_32 || output_info.qnn_data_type == QNN_DATATYPE_UINT_32) {
    need_int32_cast = true;
  }

  const std::string scatter_elements_output_name = (!need_int64_cast && !need_int32_cast) ? output_name : utils::GetUniqueName(output_name, "_cast");
  QnnTensorWrapper scatter_elements_output(scatter_elements_output_name,
                                           (is_graph_output && !need_int64_cast && !need_int32_cast) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                           (need_int64_cast || need_int32_cast) ? QNN_DATATYPE_FLOAT_32 : output_info.qnn_data_type,
                                           QnnQuantParamsWrapper(output_info.quant_param),
                                           std::vector<uint32_t>(output_info.shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scatter_elements_output)), "Failed to add tensor.");

  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_SCATTER_ELEMENTS,
                                                std::move(input_names),
                                                {scatter_elements_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add ScatterElements node.");

  // Cast from fp32/int 32 to int 64
  if (need_int64_cast || need_int32_cast) {
    // Cast from fp32 to int 32
    std::string cast_int_32_name = utils::GetUniqueName(node_unit, "_Cast_int_32");
    std::string cast_int_32_output_name = need_int64_cast ? utils::GetUniqueName(node_unit, "_Cast_int_32_output") : output_name;
    QnnTensorWrapper cast_int_32(cast_int_32_output_name,
                                 (is_graph_output && !need_int64_cast) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                 QNN_DATATYPE_INT_32,
                                 QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(output_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_int_32)),
                  "Failed to add output tensor for QNN Cast node.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_int_32_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_CAST,
                                                  {scatter_elements_output_name},
                                                  {cast_int_32_output_name},
                                                  {},
                                                  false),
                  "Failed to create QNN Cast node.");

    if (need_int64_cast) {
      // Cast from int 32 to int 64
      std::string cast_int_64_name = utils::GetUniqueName(node_unit, "_Cast_int_64");
      QnnTensorWrapper cast_int_64(output_name,
                                   is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                   QNN_DATATYPE_INT_64,
                                   QnnQuantParamsWrapper(),
                                   std::vector<uint32_t>(output_info.shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_int_64)),
                    "Failed to add output tensor for QNN Cast node.");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_int_64_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_CAST,
                                                    {cast_int_32_output_name},
                                                    {output_name},
                                                    {},
                                                    false),
                    "Failed to create QNN Cast node.");
    }
  }
  return Ort::Status();
}

void CreateScatterElementsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ScatterElementsOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime