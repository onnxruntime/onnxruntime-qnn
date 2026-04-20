// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Maps ONNX Identity to QNN Transpose with identity permutation [0, 1, ..., rank-1].
// QNN has no native Identity op. Same-shape Reshape is rejected by HTP graph compose,
// but identity-perm Transpose is a well-supported no-op pattern on all QNN backends.
class IdentityOpBuilder : public BaseOpBuilder {
 public:
  IdentityOpBuilder() : BaseOpBuilder("IdentityOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IdentityOpBuilder);

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Ort::Status IdentityOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                           const OrtNodeUnit& node_unit,
                                                           std::vector<std::string>&& input_names,
                                                           const Ort::Logger& logger,
                                                           bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  if (input_names.size() < 1) {
    return Ort::Status();
  }

  // Build identity permutation from input rank
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape), "Cannot get shape");
  uint32_t rank = static_cast<uint32_t>(input_shape.size());

  std::vector<uint32_t> perm_data(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    perm_data[i] = i;
  }

  std::vector<uint32_t> perm_shape{rank};
  QnnParamWrapper perm_param(node_unit.Index(), node_unit.Name(), QNN_OP_TRANSPOSE_PARAM_PERM,
                             std::move(perm_shape), std::move(perm_data));
  std::vector<std::string> param_tensor_names;
  param_tensor_names.push_back(perm_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(perm_param));

  // Output tensor copies input data type and quantization
  const auto& output_name = node_unit.Outputs()[0].name;
  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  const QnnTensorWrapper& input_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[0]);

  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        input_tensor_wrapper.GetTensorDataType(),
                                        input_tensor_wrapper.GetQnnQuantParams().Copy(),
                                        std::vector<uint32_t>(input_shape));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");

  std::vector<std::string> output_names{output_name};
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                QNN_OP_TRANSPOSE,
                                                std::move(input_names),
                                                std::move(output_names),
                                                std::move(param_tensor_names),
                                                do_op_validation),
                "Failed to add node.");

  return Ort::Status();
}

void CreateIdentityOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<IdentityOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
