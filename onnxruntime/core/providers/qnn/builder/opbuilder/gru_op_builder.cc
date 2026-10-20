// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/rnn_op_builder_base.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

// NOTE: The ONNX->QNN decomposition (gate slicing, time-step unroll, bidirectional Concat)
// is shared with the qti_aisw "StatefulGru" builder via GruBaseOpBuilder and
// rnn_utils::AddUnidirectionGRU. Changes in that shared path affect both builders.
class GRUOpBuilder : public GruBaseOpBuilder {
 public:
  GRUOpBuilder() : GruBaseOpBuilder("GRUOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GRUOpBuilder);

 protected:
  const char* OpDisplayName() const override { return "GRU"; }
  size_t MaxOnnxInputCount() const override { return 6; }
  const char* FpFallbackInputSuffix() const override { return "_gru_to_f32"; }
  bool AllowsExplicitZeroClip() const override { return false; }

  Ort::Status ValidateAdditionalSupport(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  rnn_utils::ResetInput GetResetInput(const OrtNodeUnit& node_unit,
                                      const std::vector<std::string>& input_names) const override;

  size_t GetQnnGruInputCount(const rnn_utils::ResetInput& reset) const override;
};

Ort::Status GRUOpBuilder::ValidateAdditionalSupport(QnnModelWrapper& qnn_model_wrapper,
                                                    const OrtNodeUnit& node_unit,
                                                    const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(qnn_model_wrapper);
  ORT_UNUSED_PARAMETER(logger);
  OrtNodeAttrHelper node_helper(node_unit);
  RETURN_IF(node_helper.Get("layout", static_cast<int64_t>(0)) != 0,
            "QNN EP doesn't support layout=1 for GRU (ORT CPU EP cannot provide a reference for accuracy validation).");
  const std::vector<std::string> activations = node_helper.Get("activations", std::vector<std::string>{});
  RETURN_IF((activations.size() >= 2 && (activations[0] != "sigmoid" || activations[1] != "tanh")) ||
                (activations.size() == 4 && (activations[2] != "sigmoid" || activations[3] != "tanh")),
            "QNN EP doesn't support non-default activations for GRU.");
  return Ort::Status();
}

rnn_utils::ResetInput GRUOpBuilder::GetResetInput(const OrtNodeUnit& node_unit,
                                                  const std::vector<std::string>& input_names) const {
  ORT_UNUSED_PARAMETER(node_unit);
  ORT_UNUSED_PARAMETER(input_names);
  return rnn_utils::NoQnnResetInput();
}

size_t GRUOpBuilder::GetQnnGruInputCount(const rnn_utils::ResetInput& reset) const {
  ORT_UNUSED_PARAMETER(reset);
  return 14;
}

void CreateGRUOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GRUOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
