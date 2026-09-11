// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Handles both ONNX Cast and CastLike. CastLike's second input only conveys the target dtype
// (already resolved into node_unit.Outputs()[0].type by ONNX type inference), so it is never
// added to the QNN graph.
//
// FP -> Bool lowering:
//   * HTP: emit Sign -> Abs -> Cast so the Cast input is normalised to {0.0, 1.0} to comply v79+.
//   * Other backends: emit NotEqual(x, 0.f).
class CastOpBuilder : public BaseOpBuilder {
 public:
  CastOpBuilder() : BaseOpBuilder("CastOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CastOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  // True when the ONNX Cast/CastLike is (fp32|fp16) -> bool. Independent of backend.
  bool IsFpToBoolCast(const OrtNodeUnit& node_unit) const;

  // True when the fp -> bool Cast should be lowered as Sign -> Abs -> Cast on HTP.
  // Assumes IsFpToBoolCast(node_unit) is already true; only checks the backend.
  bool UseSignAbsCastForHtp(const QnnModelWrapper& qnn_model_wrapper) const;

  Ort::Status ProcessExtraInputForNotEqual(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           std::vector<std::string>& input_names,
                                           const Ort::Logger& logger) const;

  // Emits Sign -> Abs -> Cast(BOOL) for node_unit.Inputs()[0] -> output_name. Caller must
  // have already added the input and output tensor wrappers.
  Ort::Status EmitSignAbsCastDecomposition(QnnModelWrapper& qnn_model_wrapper,
                                           const OrtNodeUnit& node_unit,
                                           const std::string& output_name,
                                           bool do_op_validation) const ORT_MUST_USE_RESULT;
};

bool CastOpBuilder::IsFpToBoolCast(const OrtNodeUnit& node_unit) const {
  ONNXTensorElementDataType input_type = node_unit.Inputs()[0].type;
  ONNXTensorElementDataType output_type = node_unit.Outputs()[0].type;

  Qnn_DataType_t input_qnn_dtype = QNN_DATATYPE_UNDEFINED;
  Qnn_DataType_t output_qnn_dtype = QNN_DATATYPE_UNDEFINED;

  if (!utils::GetQnnDataType(false, input_type, input_qnn_dtype).IsOK() ||
      !utils::GetQnnDataType(false, output_type, output_qnn_dtype).IsOK()) {
    return false;
  }

  return ((input_qnn_dtype == QNN_DATATYPE_FLOAT_16 || input_qnn_dtype == QNN_DATATYPE_FLOAT_32) &&
          output_qnn_dtype == QNN_DATATYPE_BOOL_8);
}

bool CastOpBuilder::UseSignAbsCastForHtp(const QnnModelWrapper& qnn_model_wrapper) const {
  const QnnBackendType be = qnn_model_wrapper.GetQnnBackendType();
  return (be == QnnBackendType::HTP || be == QnnBackendType::SERIALIZER);
}

Ort::Status CastOpBuilder::ProcessExtraInputForNotEqual(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>& input_names,
                                                        const Ort::Logger& logger) const {
  const auto& input = node_unit.Inputs()[0];
  if (input.quant_param.has_value()) {
    return Ort::Status();
  }

  // Build additional static input with value 0.
  const std::string& input_name = utils::UniqueNameGenerator().New(node_unit, "_notequal_zero");

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  ONNXTensorElementDataType input_type = input.type;
  RETURN_IF_ERROR(utils::GetQnnDataType(false, input_type, qnn_data_type));

  QnnTensorWrapper input_tensor_wrapper(input_name,
                                        QNN_TENSOR_TYPE_STATIC,
                                        qnn_data_type,
                                        QnnQuantParamsWrapper(),
                                        std::vector<uint32_t>{1},
                                        std::vector<uint8_t>(utils::GetElementSizeByType(qnn_data_type), 0));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor_wrapper)),
                "Failed to add additional input tensor for QNN Cast node that will be replaced by NotEqual.");
  input_names.push_back(input_name);

  ORT_CXX_LOG(logger,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("FP-to-Bool Cast node " + node_unit.Name() + " is replaced by NotEqual.").c_str());
  return Ort::Status();
}

Ort::Status CastOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  // 1. Validate input count. Cast has 1 input; CastLike has 2 (second only conveys dtype).
  const auto& inputs = node_unit.Inputs();
  RETURN_IF_NOT(inputs.size() == 1 || inputs.size() == 2,
                "QNN Cast node must have 1 input (Cast) or 2 inputs (CastLike).");
  const auto& input = inputs[0];

  // 2. FP64 is unsupported in QNN graph IO.
  const auto& input_name = input.name;
  RETURN_IF(qnn_model_wrapper.IsGraphInput(input_name) && input.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
            "Unsupported FP64 data type in graph IO.");

  // 3. Reuse existing tensor wrapper if this input was already added by an upstream op.
  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_name).c_str());
    input_names.push_back(input_name);
    if (IsFpToBoolCast(node_unit) && !UseSignAbsCastForHtp(qnn_model_wrapper)) {
      return ProcessExtraInputForNotEqual(qnn_model_wrapper, node_unit, input_names, logger);
    }
    return Ort::Status();
  }

  // 4. Unpack initializer data if the input is a constant.
  std::vector<uint8_t> unpacked_tensor;
  bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
  if (is_constant_input) {
    const auto* input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_tensor, unpacked_tensor));
  }

  // 5. Resolve tensor metadata (type, shape, dtype).
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input_name);
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.shape, input_shape),
                "Cannot get shape for QNN Cast node's input.");

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  ONNXTensorElementDataType input_type = input.type;

  RETURN_IF_ERROR(utils::GetQnnDataType(false,  // Do not try to get the quantized type. HTP cast supports normal types.
                                        input_type,
                                        qnn_data_type));

  // 6. Register the input tensor wrapper.
  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::move(input_shape), std::move(unpacked_tensor));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                "Failed to add input tensor for QNN Cast node.");
  input_names.push_back(input_name);

  // 7. Append the zero-tensor input for the NotEqual lowering when applicable.
  // FP -> bool on non-HTP backends needs the zero constant for the NotEqual lowering.
  if (IsFpToBoolCast(node_unit) && !UseSignAbsCastForHtp(qnn_model_wrapper)) {
    return ProcessExtraInputForNotEqual(qnn_model_wrapper, node_unit, input_names, logger);
  }
  return Ort::Status();
}

Ort::Status CastOpBuilder::EmitSignAbsCastDecomposition(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        const std::string& output_name,
                                                        bool do_op_validation) const {
  // 1. Derive input name, dtype, and shape from the Cast's ONNX input.
  const auto& input = node_unit.Inputs()[0];
  const std::string& input_name = input.name;

  Qnn_DataType_t input_qnn_dtype = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(false, input.type, input_qnn_dtype));

  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.shape, input_shape),
                "Cannot get shape for FP-to-Bool Cast input.");

  // 2. Reserve deterministic intermediate names so DLC inspection and context cache lookups stay stable.
  const std::string sign_out_name = utils::UniqueNameGenerator().New(node_unit, "_signabs_sign");
  const std::string abs_out_name = utils::UniqueNameGenerator().New(node_unit, "_signabs_abs");

  // 3. Register the two NATIVE intermediates with the same shape/dtype as the Cast input.
  QnnTensorWrapper sign_out_wrapper(sign_out_name,
                                    QNN_TENSOR_TYPE_NATIVE,
                                    input_qnn_dtype,
                                    QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(input_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(sign_out_wrapper)),
                "Failed to add Sign intermediate tensor.");

  QnnTensorWrapper abs_out_wrapper(abs_out_name,
                                   QNN_TENSOR_TYPE_NATIVE,
                                   input_qnn_dtype,
                                   QnnQuantParamsWrapper(),
                                   std::vector<uint32_t>(input_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(abs_out_wrapper)),
                "Failed to add Abs intermediate tensor.");

  // 4. Sign(x) -> sign_out. Sign(x) is {-1, 0, +1}.
  const std::string sign_node_name = utils::UniqueNameGenerator().New(node_unit, "_signabs_sign_node");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(sign_node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_SIGN,
                                                {input_name},
                                                {sign_out_name},
                                                {},
                                                do_op_validation),
                "Failed to create Sign node.");

  // 5. Abs(sign_out) -> abs_out. Collapses {-1, 0, +1} to {0, 1}.
  const std::string abs_node_name = utils::UniqueNameGenerator().New(node_unit, "_signabs_abs_node");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(abs_node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_ABS,
                                                {sign_out_name},
                                                {abs_out_name},
                                                {},
                                                do_op_validation),
                "Failed to create Abs node.");

  // 6. Cast(abs_out, to=BOOL) -> original ONNX Cast output.
  const std::string cast_node_name = utils::UniqueNameGenerator().New(node_unit);
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_CAST,
                                                {abs_out_name},
                                                {output_name},
                                                {},
                                                do_op_validation),
                "Failed to create Cast node.");
  return Ort::Status();
}

Ort::Status CastOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  // 1. Validate output arity.
  const auto& outputs = node_unit.Outputs();
  RETURN_IF_NOT(outputs.size() == 1, "QNN Cast node must have a single output.");
  const auto& output = outputs[0];
  const auto& output_name = output.name;

  // 2. Resolve output dtype and shape.
  ONNXTensorElementDataType output_type = output.type;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  RETURN_IF_ERROR(utils::GetQnnDataType(false,  // Do not try to get the quantized type. HTP cast supports normal types.
                                        output_type,
                                        qnn_data_type));

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.shape, output_shape),
                "Cannot get shape for QNN Cast node's output.");
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  // 3. Narrow int64 -> int32 for internal tensors; substitute fp32 for internal fp64.
  const Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  if (qnn_data_type == QNN_DATATYPE_INT_64 && tensor_type == QNN_TENSOR_TYPE_NATIVE) {
    qnn_data_type = QNN_DATATYPE_INT_32;
  } else if (qnn_data_type == QNN_DATATYPE_FLOAT_64) {
    RETURN_IF(is_graph_output, "Unsupported FP64 data type in graph IO.");
    qnn_data_type = QNN_DATATYPE_FLOAT_32;
  }

  // 4. Register the output tensor wrapper.
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        qnn_data_type,
                                        QnnQuantParamsWrapper(),
                                        std::move(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                "Failed to add output tensor for QNN Cast node.");

  // 5. FP -> bool: Sign -> Abs -> Cast on HTP, NotEqual(x, 0.f) elsewhere.
  //    Non-fp -> bool: regular one-to-one Cast.
  if (IsFpToBoolCast(node_unit)) {
    if (UseSignAbsCastForHtp(qnn_model_wrapper)) {
      RETURN_IF_NOT(input_names.size() == 1, "FP-to-Bool Cast decomposition expects exactly one input.");
      return EmitSignAbsCastDecomposition(qnn_model_wrapper, node_unit, output_name, do_op_validation);
    }
    // NotEqual(x, 0.f) — the zero constant was already appended in ProcessInputs.
    const std::string notequal_node_name = utils::UniqueNameGenerator().New(node_unit);
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(notequal_node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_NOT_EQUAL,
                                                  std::move(input_names),
                                                  {output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to create NotEqual node.");
    return Ort::Status();
  }

  // 6. Non-fp -> bool: regular one-to-one Cast.
  const std::string cast_node_name = utils::UniqueNameGenerator().New(node_unit);
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                GetQnnOpType(node_unit.OpType()),
                                                std::move(input_names),
                                                {output_name},
                                                {},
                                                do_op_validation),
                "Failed to create Cast node.");

  return Ort::Status();
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<CastOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
