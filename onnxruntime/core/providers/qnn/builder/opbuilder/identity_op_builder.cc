// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Maps ONNX Identity to QNN Transpose with identity permutation [0, 1, ..., rank-1].
// QNN has no native Identity op and rejects same-shape Reshape during HTP graph compose.
// Identity-perm Transpose is optimized to zero cycles on HTP.
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

  Ort::Status OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const Ort::Logger& logger,
                                       const std::vector<std::string>& input_names,
                                       size_t output_index,
                                       Qnn_DataType_t qnn_data_type,
                                       QnnQuantParamsWrapper& quant_param) const override ORT_MUST_USE_RESULT;
};

Ort::Status IdentityOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                           const OrtNodeUnit& node_unit,
                                                           std::vector<std::string>&& input_names,
                                                           const Ort::Logger& logger,
                                                           bool do_op_validation) const {
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

  return ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names),
                        std::move(param_tensor_names), logger, do_op_validation, QNN_OP_TRANSPOSE);
}

Ort::Status IdentityOpBuilder::OverrideOutputQuantParam(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        const Ort::Logger& logger,
                                                        const std::vector<std::string>& input_names,
                                                        size_t output_index,
                                                        Qnn_DataType_t qnn_data_type,
                                                        QnnQuantParamsWrapper& quant_param) const {
  if (!quant_param.IsPerTensor()) {
    return Ort::Status();
  }

  return SetOutputQParamEqualToInputIfNearlyEqual(qnn_model_wrapper, node_unit, logger, input_names,
                                                  0, output_index, qnn_data_type, quant_param);
}

void CreateIdentityOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<IdentityOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
