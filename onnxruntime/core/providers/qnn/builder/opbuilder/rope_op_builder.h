// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class RopeOpBuilder : public BaseOpBuilder {
 public:
  RopeOpBuilder() : BaseOpBuilder("RopeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RopeOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ValidateInputShapes(QnnModelWrapper& qnn_model_wrapper,
                                  const OrtNodeUnit& node_unit,
                                  const Ort::Logger& logger) const ORT_MUST_USE_RESULT;

  Ort::Status DecomposeRotaryEmbedding(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       std::vector<std::string>&& input_names,
                                       const Ort::Logger& logger,
                                       bool do_op_validation) const ORT_MUST_USE_RESULT;
};

}  // namespace qnn
}  // namespace onnxruntime
