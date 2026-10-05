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

// Bounds each tuple column by the matching `data_shape[c]`.
Ort::Status ProcessScatterNDIndices(QnnModelWrapper& qnn_model_wrapper,
                                    const OrtNodeUnitIODef& indices_input,
                                    const std::vector<uint32_t>& data_shape,
                                    const Ort::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) {
  std::string indices_tensor_name = indices_input.name;

  TensorInfo indices_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  // ONNX ScatterND rank>=1 is not enforced by shape inference; rely on a well-formed graph.
  const uint32_t index_tuple_size = indices_info.shape.back();

  const auto axis_dim_for_element = [index_tuple_size, &data_shape](size_t element_index) -> int64_t {
    const size_t col = element_index % static_cast<size_t>(index_tuple_size);
    return static_cast<int64_t>(data_shape[col]);
  };

  std::vector<uint8_t> qnn_indices_bytes;
  bool has_negative_indices = false;

  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(indices_info.initializer_tensor,
                                                            onnx_indices_bytes));

    // ONNX ScatterND `indices` is hard-typed tensor(int64) (unlike ScatterElements).
    RETURN_IF_NOT(utils::NormalizeIndicesBytes<int64_t>(onnx_indices_bytes, axis_dim_for_element,
                                                        qnn_indices_bytes, has_negative_indices),
                  "QNN does not support out-of-range index values for ScatterND.");
    indices_info.qnn_data_type = QNN_DATATYPE_INT_32;

    // Rename so a sibling op reusing the same ONNX initializer under a different
    // axis bound cannot alias our rewritten copy.
    indices_tensor_name = indices_tensor_name + "_qnn_idx";
  }

  return utils::AddNormalizedIndicesTensor(qnn_model_wrapper, std::move(indices_info),
                                           indices_tensor_name, std::move(qnn_indices_bytes),
                                           logger, input_names, do_op_validation);
}

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
  RETURN_IF_ERROR(ProcessScatterNDIndices(qnn_model_wrapper, inputs[1], data_info.shape,
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

  uint32_t reduction_value = QNN_OP_SCATTER_ND_REDUCTION_NONE;
  if (reduction == "none") {
    reduction_value = QNN_OP_SCATTER_ND_REDUCTION_NONE;
  } else if (reduction == "add") {
    reduction_value = QNN_OP_SCATTER_ND_REDUCTION_ADD;
  } else if (reduction == "mul") {
    reduction_value = QNN_OP_SCATTER_ND_REDUCTION_MUL;
  } else {
    return MAKE_EP_FAIL(("Unexpected ScatterND reduction: " + reduction).c_str());
  }

  std::vector<std::string> param_tensor_names;
  RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), reduction_value,
                                         QNN_OP_SCATTER_ND_PARAM_REDUCTION, param_tensor_names));

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
