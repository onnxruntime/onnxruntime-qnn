// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/normalize_indices_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {
constexpr std::array<std::string_view, 3> kSupportedReductions = {"none", "add", "mul"};
}  // namespace

class ScatterNDOpBuilder : public BaseOpBuilder {
 public:
  ScatterNDOpBuilder() : BaseOpBuilder("ScatterNDOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScatterNDOpBuilder);

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

Ort::Status ScatterNDOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                              const OrtNodeUnit& node_unit,
                                              const Ort::Logger& logger,
                                              std::vector<std::string>& input_names,
                                              bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  RETURN_IF(inputs.size() != 3, "QNN EP: ScatterND operator must have three inputs.");

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  // QNN rejects negative/INT_64 indices; rewrite statics to keep the node on QNN.
  TensorInfo data_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], data_info));
  RETURN_IF_ERROR(utils::NormalizeIndicesForScatterND(
      qnn_model_wrapper, inputs[1], data_info.shape,
      logger, input_names, do_op_validation));

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, input_names));
  return Ort::Status();
}

Ort::Status ScatterNDOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                            const OrtNodeUnit& node_unit,
                                                            std::vector<std::string>&& input_names,
                                                            const Ort::Logger& logger,
                                                            bool do_op_validation) const {
  if (input_names.empty()) {
    return Ort::Status();
  }

  if (do_op_validation) {
    // TODO: Remove once QNN CPU supports ScatterND.
    RETURN_IF(qnn_model_wrapper.GetQnnBackendType() == QnnBackendType::CPU,
              "QNN EP does not support ScatterND op on CPU backend. Falling back to ORT CPU.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const std::string reduction = node_helper.Get("reduction", "none");
  RETURN_IF_NOT(utils::ArrayHasString(kSupportedReductions, reduction),
                ("ScatterND does not support reduction " + reduction).c_str());

  Qnn_Scalar_t reduction_scalar = QNN_SCALAR_INIT;
  reduction_scalar.dataType = QNN_DATATYPE_UINT_32;
  if (reduction == "none") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_NONE;
  } else if (reduction == "add") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_ADD;
  } else if (reduction == "mul") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_MUL;
  } else {
    return MAKE_EP_FAIL(("Unexpected ScatterND reduction: " + reduction).c_str());
  }

  QnnParamWrapper reduction_param(node_unit.Index(), node_unit.Name(),
                                  QNN_OP_SCATTER_ND_PARAM_REDUCTION, reduction_scalar);
  std::vector<std::string> param_tensor_names = {reduction_param.GetParamTensorName()};
  qnn_model_wrapper.AddParamWrapper(std::move(reduction_param));

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateScatterNDOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ScatterNDOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
