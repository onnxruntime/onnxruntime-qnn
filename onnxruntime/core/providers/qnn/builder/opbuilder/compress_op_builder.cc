// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// GPU-only fixed-shape Compress lowering.
//
// ONNX Compress normally packs selected values and therefore has a dynamic
// output length. This lowering is intentionally enabled only when the ONNX
// output shape is the same as the input shape. It replaces unselected rows
// with zero, preserving the static tensor shape required by a QNN context.
class CompressOpBuilder : public BaseOpBuilder {
 public:
  CompressOpBuilder() : BaseOpBuilder("CompressOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CompressOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

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
};

Ort::Status CompressOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const Ort::Logger& logger) const {
  RETURN_IF_NOT(IsGpuBackend(qnn_model_wrapper.GetQnnBackendType()),
                "Compress: GPU fixed-shape lowering only.");
  const auto& inputs = node_unit.Inputs();
  RETURN_IF_NOT(inputs.size() == 2, "Compress: expected input and condition.");

  TensorInfo data_info{};
  TensorInfo condition_info{};
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], data_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], condition_info));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  OrtNodeAttrHelper helper(node_unit);
  const int64_t axis = helper.Get("axis", static_cast<int64_t>(0));
  RETURN_IF_NOT(axis == 0, "Compress GPU lowering supports axis=0 only.");
  RETURN_IF_NOT(data_info.shape.size() == 2u && condition_info.shape.size() == 1u,
                "Compress GPU lowering supports rank-2 input and rank-1 condition only.");
  RETURN_IF_NOT(data_info.shape == output_info.shape,
                "Compress GPU lowering requires fixed output shape equal to input shape.");
  RETURN_IF_NOT(condition_info.shape[0] == data_info.shape[0],
                "Compress GPU lowering condition length must equal input dimension 0.");
  RETURN_IF_NOT((data_info.qnn_data_type == QNN_DATATYPE_FLOAT_16 ||
                 data_info.qnn_data_type == QNN_DATATYPE_FLOAT_32) &&
                    output_info.qnn_data_type == data_info.qnn_data_type,
                "Compress GPU lowering supports matching FLOAT16/FLOAT32 input and output.");
  RETURN_IF_NOT(condition_info.qnn_data_type == QNN_DATATYPE_BOOL_8,
                "Compress GPU lowering supports BOOL condition only.");
  return BaseOpBuilder::IsOpSupported(qnn_model_wrapper, node_unit, logger);
}

Ort::Status CompressOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const Ort::Logger& logger,
                                             std::vector<std::string>& input_names,
                                             bool /*do_op_validation*/) const {
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, node_unit.Inputs()[0], logger, input_names));
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, node_unit.Inputs()[1], logger, input_names));
  return Ort::Status();
}

Ort::Status CompressOpBuilder::ProcessAttributesAndOutputs(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& node_unit,
    std::vector<std::string>&& input_names,
    const Ort::Logger& /*logger*/,
    bool do_op_validation) const {
  const OrtNodeUnitIODef& output_tensor = node_unit.Outputs()[0];
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_tensor, output_info));
  const auto& data_dims = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]).GetTensorDims();
  const std::string condition_expanded_name =
      utils::UniqueNameGenerator().New(input_names[1], "_compress_expand");
  const std::vector<uint32_t> condition_shape{data_dims[0], 1u};
  RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input_names[1],
                                                   condition_expanded_name,
                                                   {data_dims[0]},
                                                   condition_shape,
                                                   QNN_DATATYPE_BOOL_8,
                                                   QnnQuantParamsWrapper(),
                                                   do_op_validation,
                                                   false,
                                                   false));

  const std::string zero_name = utils::UniqueNameGenerator().New(node_unit.Name(), "_compress_zero");
  const size_t zero_size = output_info.qnn_data_type == QNN_DATATYPE_FLOAT_16 ? sizeof(uint16_t) : sizeof(float);
  QnnTensorWrapper zero_tensor(zero_name,
                               QNN_TENSOR_TYPE_STATIC,
                               output_info.qnn_data_type,
                               QnnQuantParamsWrapper(),
                               std::vector<uint32_t>{1u},
                               std::vector<uint8_t>(zero_size, 0u));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(zero_tensor)),
                "Compress GPU lowering failed to add zero tensor.");

  QnnTensorWrapper output_wrapper;
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_tensor, output_wrapper));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_wrapper)),
                "Compress GPU lowering failed to add output tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_compress_select"),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_ELEMENT_WISE_SELECT,
                                                {condition_expanded_name, input_names[0], zero_name},
                                                {output_tensor.name},
                                                {},
                                                do_op_validation),
                "Compress GPU lowering failed to add ElementWiseSelect node.");
  return Ort::Status();
}

void CreateCompressOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<CompressOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
