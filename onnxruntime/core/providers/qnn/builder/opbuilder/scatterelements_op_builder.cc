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

constexpr std::array<std::string_view, 4> kSupportedReductions = {"none", "add", "mul", "max"};

Ort::Status ProcessScatterElementsIndices(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnitIODef& indices_input,
                                          const std::vector<uint32_t>& data_shape,
                                          int64_t axis,
                                          const Ort::Logger& logger,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) {
  std::string indices_tensor_name = indices_input.name;

  TensorInfo indices_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(indices_input, indices_info));

  const int64_t axis_dim = static_cast<int64_t>(data_shape[static_cast<size_t>(axis)]);
  // Uniform bound -- element_index unused (contrast with ScatterND's per-column closure).
  const auto axis_dim_for_element = [axis_dim](size_t /*element_index*/) -> int64_t {
    return axis_dim;
  };

  std::vector<uint8_t> qnn_indices_bytes;
  bool has_negative_indices = false;

  if (indices_info.is_initializer) {
    std::vector<uint8_t> onnx_indices_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(indices_info.initializer_tensor,
                                                            onnx_indices_bytes));

    // `Tind` is {int32, int64}; reject anything else rather than reinterpret bytes.
    if (indices_info.qnn_data_type == QNN_DATATYPE_INT_64) {
      RETURN_IF_NOT(utils::NormalizeIndicesBytes<int64_t>(onnx_indices_bytes, axis_dim_for_element,
                                                          qnn_indices_bytes, has_negative_indices),
                    "QNN does not support out-of-range index values for ScatterElements.");
      indices_info.qnn_data_type = QNN_DATATYPE_INT_32;
    } else if (indices_info.qnn_data_type == QNN_DATATYPE_INT_32) {
      RETURN_IF_NOT(utils::NormalizeIndicesBytes<int32_t>(onnx_indices_bytes, axis_dim_for_element,
                                                          qnn_indices_bytes, has_negative_indices),
                    "QNN does not support out-of-range index values for ScatterElements.");
      if (!has_negative_indices) {
        // No per-axis remapping happened; reuse the original bytes unchanged.
        qnn_indices_bytes = std::move(onnx_indices_bytes);
      }
    } else {
      return MAKE_EP_FAIL("ScatterElements indices must be INT_32 or INT_64.");
    }

    if (has_negative_indices) {
      // Remapped bytes depend on axis_dim; rename so siblings under a different axis can't alias.
      indices_tensor_name = utils::UniqueNameGenerator().New(indices_tensor_name, "_qnn_idx");
    }
  }

  return utils::AddNormalizedIndicesTensor(qnn_model_wrapper, std::move(indices_info),
                                           indices_tensor_name, std::move(qnn_indices_bytes),
                                           logger, input_names, do_op_validation);
}

}  // namespace

// Op builder for ONNX ScatterElements (https://onnx.ai/onnx/operators/onnx__ScatterElements.html).
// ONNX allows negative and INT_64 indices; QNN accepts only non-negative INT_32. Static
// indices are normalized at partition time; dynamic INT_64 indices get a runtime Cast.
class ScatterElementsOpBuilder : public BaseOpBuilder {
 public:
  ScatterElementsOpBuilder() : BaseOpBuilder("ScatterElementsOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScatterElementsOpBuilder);

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

Ort::Status ScatterElementsOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const OrtNodeUnit& node_unit,
                                                    const Ort::Logger& logger,
                                                    std::vector<std::string>& input_names,
                                                    bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  RETURN_IF(inputs.size() != 3, "QNN EP: ScatterElements operator must have three inputs.");

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  // QNN rejects negative indices; rewrite statics to keep the node on QNN.
  TensorInfo data_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], data_info));

  OrtNodeAttrHelper node_helper(node_unit);
  int64_t axis = node_helper.Get("axis", static_cast<int64_t>(0));
  const int64_t rank = static_cast<int64_t>(data_info.shape.size());
  if (axis < 0) {
    axis += rank;
  }
  RETURN_IF_NOT(axis >= 0 && axis < rank, "ScatterElements axis out of range.");

  RETURN_IF_ERROR(ProcessScatterElementsIndices(qnn_model_wrapper, inputs[1], data_info.shape, axis,
                                                logger, input_names, do_op_validation));

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, input_names));
  return Ort::Status();
}

Ort::Status ScatterElementsOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                  const OrtNodeUnit& node_unit,
                                                                  std::vector<std::string>&& input_names,
                                                                  const Ort::Logger& logger,
                                                                  bool do_op_validation) const {
  if (input_names.empty()) {
    return Ort::Status();
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const std::string reduction = node_helper.Get("reduction", "none");
  RETURN_IF_NOT(utils::ArrayHasString(kSupportedReductions, reduction),
                ("ScatterElements does not support reduction " + reduction).c_str());

  std::vector<std::string> param_tensor_names;

  int32_t axis_value = 0;
  Qnn_Scalar_t axis_qnn_scalar = QNN_SCALAR_INIT;
  RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, axis_qnn_scalar, axis_value));
  QnnParamWrapper axis_param(node_unit.Index(), node_unit.Name(),
                             QNN_OP_SCATTER_ELEMENTS_PARAM_AXIS, axis_qnn_scalar);
  param_tensor_names.push_back(axis_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axis_param));

  Qnn_Scalar_t reduction_scalar = QNN_SCALAR_INIT;
  reduction_scalar.dataType = QNN_DATATYPE_UINT_32;
  if (reduction == "none") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ELEMENTS_REDUCTION_NONE;
  } else if (reduction == "add") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ELEMENTS_REDUCTION_ADD;
  } else if (reduction == "mul") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ELEMENTS_REDUCTION_MUL;
  } else if (reduction == "max") {
    reduction_scalar.uint32Value = QNN_OP_SCATTER_ELEMENTS_REDUCTION_MAX;
  } else {
    return MAKE_EP_FAIL(("Unexpected ScatterElements reduction: " + reduction).c_str());
  }

  QnnParamWrapper reduction_param(node_unit.Index(), node_unit.Name(),
                                  QNN_OP_SCATTER_ELEMENTS_PARAM_REDUCTION, reduction_scalar);
  param_tensor_names.push_back(reduction_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(reduction_param));

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        std::move(param_tensor_names),
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateScatterElementsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ScatterElementsOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
