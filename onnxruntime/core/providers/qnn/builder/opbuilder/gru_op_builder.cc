// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <cstring>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/rnn_op_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// NOTE: The ONNX->QNN decomposition (gate slicing, time-step unroll, bidirectional Concat)
// is shared with the qti_aisw "StatefulGru" builder via rnn_details::AddUnidirectionGRU in
// rnn_op_utils.cc. Both builders delegate to that function — changes there affect both.
class GRUOpBuilder : public BaseOpBuilder {
 public:
  GRUOpBuilder() : BaseOpBuilder("GRUOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GRUOpBuilder);

 protected:
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

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
  Ort::Status AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                                 const OrtNodeUnit& node_unit,
                                 const std::string& direction,
                                 const std::vector<std::string>& input_names,
                                 const Ort::Logger& logger,
                                 const bool& do_op_validation,
                                 const bool& is_bidirection,
                                 std::vector<std::string>& uni_gru_output_names) const;
};

Ort::Status GRUOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  if (node_unit.Inputs().size() > 4 && node_unit.Inputs()[4].Exists()) {
    TensorInfo tensor_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[4], tensor_info));
    RETURN_IF_NOT(tensor_info.is_initializer, "QNN EP: dynamic sequence_length is not supported.");
    std::vector<uint8_t> sequence_lens_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(tensor_info.initializer_tensor, sequence_lens_bytes));
    const size_t num_elems = sequence_lens_bytes.size() / sizeof(int32_t);
    // Copy into an aligned int32_t buffer; sequence_lens_bytes.data() is only 1-byte aligned, so a
    // reinterpret_cast<const int32_t*> + deref would be undefined behavior.
    std::vector<int32_t> sequence_lens(num_elems);
    if (num_elems > 0) {
      std::memcpy(sequence_lens.data(), sequence_lens_bytes.data(), num_elems * sizeof(int32_t));
    }
    RETURN_IF(std::any_of(sequence_lens.begin(), sequence_lens.end(),
                          [&sequence_lens](int32_t i) { return i != sequence_lens[0]; }),
              "QNN EP: Only support GRU with same sequence length.");
  }

  OrtNodeAttrHelper node_helper(node_unit);
  RETURN_IF(node_helper.Get("layout", static_cast<int64_t>(0)) != 0,
            "QNN EP doesn't support layout=1 for GRU (ORT CPU EP cannot provide a reference for accuracy validation).");
  const float clip = node_helper.Get("clip", 0.0f);
  RETURN_IF(clip != 0,
            "QNN EP: GRU 'clip' (gate-input activation clamp) has no equivalent QNN GRU "
            "parameter and cannot be expressed in the QNN graph.");
  const std::vector<std::string> activations = node_helper.Get("activations", std::vector<std::string>{});
  RETURN_IF((activations.size() >= 2 && (activations[0] != "sigmoid" || activations[1] != "tanh")) ||
                (activations.size() == 4 && (activations[2] != "sigmoid" || activations[3] != "tanh")),
            "QNN EP doesn't support non-default activations for GRU.");
  return Ort::Status();
}

Ort::Status GRUOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  const auto& onnx_inputs = node_unit.Inputs();
  for (size_t i = 0; i < onnx_inputs.size(); i++) {
    if (onnx_inputs[i].Exists()) {
      RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, onnx_inputs[i], logger, input_names));
    } else {
      input_names.emplace_back("");
    }
  }
  return Ort::Status();
}

Ort::Status GRUOpBuilder::AddUnidirectionGRU(QnnModelWrapper& qnn_model_wrapper,
                                             const OrtNodeUnit& node_unit,
                                             const std::string& direction,
                                             const std::vector<std::string>& input_names,
                                             const Ort::Logger& logger,
                                             const bool& do_op_validation,
                                             const bool& is_bidirection,
                                             std::vector<std::string>& uni_gru_output_names) const {
  return rnn_details::AddUnidirectionGRU(qnn_model_wrapper, node_unit, direction, input_names,
                                         logger, do_op_validation, is_bidirection,
                                         14, rnn_details::kNoReset(),
                                         uni_gru_output_names);
}

Ort::Status GRUOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      std::vector<std::string>&& input_names,
                                                      const Ort::Logger& logger,
                                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  OrtNodeAttrHelper node_helper(node_unit);
  std::string direction = node_helper.Get("direction", "forward");
  RETURN_IF_NOT(inputs.size() >= 3 && inputs.size() <= 6, "GRU should receive inputs ranging from 3 to 6!");

  if (direction == "bidirectional") {
    std::vector<std::string> fwd_out, rev_out;
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, "forward", input_names, logger, do_op_validation, true, fwd_out));
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, "reverse", input_names, logger, do_op_validation, true, rev_out));
    for (size_t i = 0; i < 2; i++) {
      TensorInfo output_info = {};
      if (node_unit.Outputs().size() > i && node_unit.Outputs()[i].Exists()) {
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[i], output_info));
        std::string name = node_unit.Outputs()[i].name;
        std::vector<std::string> cp;
        uint32_t concat_axis = 0;
        RETURN_IF_ERROR(rnn_details::DeriveNumDirectionsConcatAxis(output_info.shape, concat_axis));
        RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), name,
                                               concat_axis,
                                               QNN_OP_CONCAT_PARAM_AXIS, cp));
        Qnn_TensorType_t tt = qnn_model_wrapper.IsGraphOutput(name) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
        QnnTensorWrapper tw(name, tt, output_info.qnn_data_type, output_info.quant_param.Copy(),
                            std::vector<uint32_t>(output_info.shape));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tw)), "Failed to add Concat output.");
        RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_CONCAT),
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_CONCAT,
                                                      {fwd_out[i], rev_out[i]}, {name}, std::move(cp), do_op_validation),
                      "Failed to create Concat node.");
      }
    }
  } else {
    std::vector<std::string> uni_out;
    RETURN_IF_ERROR(AddUnidirectionGRU(qnn_model_wrapper, node_unit, direction, input_names, logger, do_op_validation, false, uni_out));
  }
  return Ort::Status();
}

void CreateGRUOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GRUOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
