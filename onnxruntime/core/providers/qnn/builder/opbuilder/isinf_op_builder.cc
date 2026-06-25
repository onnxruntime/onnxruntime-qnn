// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class IsInfOpBuilder : public BaseOpBuilder {
 public:
  IsInfOpBuilder() : BaseOpBuilder("IsInfOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IsInfOpBuilder);

  // Override IsOpSupported to decline IsInf on the NPU/HTP backend (see implementation).
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

Ort::Status IsInfOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const Ort::Logger& logger) const {
  // IsInf is declined on the NPU/HTP backend. The QnnHtp op-config validator accepts the node
  // (so it would be claimed during partitioning), but graphFinalize then fails with error 1002
  // (QNN_COMMON_ERROR_MEM_ALLOC): the only registered HTP IsInf kernel is fp16 crouton-tiled
  // (IsInf.*@CB.Ce.t*2) with no TCM/flat fallback, so finalize cannot allocate it and the whole
  // QNN subgraph fails to compile at session creation. Declining here lets ORT fall IsInf back to
  // the CPU EP instead of failing to load any model that contains it.
  RETURN_IF(IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()),
            "QNN EP does not support IsInf on the HTP backend "
            "(graphFinalize fails with error 1002 / QNN_COMMON_ERROR_MEM_ALLOC).");
  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Ort::Status IsInfOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const Ort::Logger& logger,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  if (do_op_validation) {
    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
    Qnn_DataType_t target_tensor_type = input_info.qnn_data_type;

    RETURN_IF((QNN_DATATYPE_FLOAT_32 != target_tensor_type && QNN_DATATYPE_FLOAT_16 != target_tensor_type),
              "QNN IsInf Op supports only float32 and float16 input tensors.");
  }
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Ort::Status();
}

Ort::Status IsInfOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        const Ort::Logger& logger,
                                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  // ONNX IsInf detect_positive / detect_negative default to 1 (detect both).
  OrtNodeAttrHelper node_helper(node_unit);
  const bool detect_positive = node_helper.Get("detect_positive", static_cast<int64_t>(1)) != 0;
  const bool detect_negative = node_helper.Get("detect_negative", static_cast<int64_t>(1)) != 0;

  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper,
                                     node_unit.Index(),
                                     node_unit.Name(),
                                     detect_positive,
                                     QNN_OP_IS_INF_PARAM_DETECT_POSITIVE,
                                     param_tensor_names));
  RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper,
                                     node_unit.Index(),
                                     node_unit.Name(),
                                     detect_negative,
                                     QNN_OP_IS_INF_PARAM_DETECT_NEGATIVE,
                                     param_tensor_names));

  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  if (do_op_validation) {
    RETURN_IF(QNN_DATATYPE_BOOL_8 != output_info.qnn_data_type, "QNN IsInf Op supports only bool8 output tensor.");
  }
  const std::string& org_output_name = node_unit.Outputs()[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  std::vector<uint32_t> output_shape = output_info.shape;
  const std::string isinf_node_name = utils::UniqueNameGenerator().New(node_unit, "_IsInf");

  QnnTensorWrapper isinf_output(org_output_name,
                                is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                output_info.qnn_data_type,
                                output_info.quant_param.Copy(),
                                std::move(output_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(isinf_output)), "Failed to add tensor.");
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(isinf_node_name,
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_IS_INF,
                                                {input_names[0]},
                                                {org_output_name},
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to create QNN IsInf node.");

  return Ort::Status();
}

void CreateIsInfOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<IsInfOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
