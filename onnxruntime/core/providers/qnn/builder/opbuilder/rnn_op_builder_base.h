// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/rnn_op_utils.h"

namespace onnxruntime {
namespace qnn {

class GruBaseOpBuilder : public BaseOpBuilder {
 public:
  explicit GruBaseOpBuilder(const std::string& op_builder_type) : BaseOpBuilder(op_builder_type) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GruBaseOpBuilder);

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

  virtual const char* OpDisplayName() const = 0;
  virtual size_t MaxOnnxInputCount() const = 0;
  virtual const char* FpFallbackInputSuffix() const = 0;
  // ONNX GRU historically rejects an explicitly supplied clip, including clip=0. StatefulGru
  // follows the block-op contract, where an explicit zero is equivalent to the default.
  virtual bool AllowsExplicitZeroClip() const { return true; }

  virtual Ort::Status ValidateAdditionalSupport(QnnModelWrapper&,
                                                const OrtNodeUnit&,
                                                const Ort::Logger&) const {
    return Ort::Status();
  }

  virtual rnn_utils::ResetInput GetResetInput(const OrtNodeUnit& node_unit,
                                              const std::vector<std::string>& input_names) const = 0;
  virtual size_t GetQnnGruInputCount(const rnn_utils::ResetInput& reset) const = 0;
};

class LstmBaseOpBuilder : public BaseOpBuilder {
 public:
  explicit LstmBaseOpBuilder(const std::string& op_builder_type) : BaseOpBuilder(op_builder_type) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LstmBaseOpBuilder);

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

  virtual const char* OpDisplayName() const = 0;
  virtual size_t MaxOnnxInputCount() const = 0;

  virtual Ort::Status ValidateInputShapeSupport(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger) const;

  virtual Ort::Status ValidateAdditionalSupport(QnnModelWrapper&,
                                                const OrtNodeUnit&,
                                                const Ort::Logger&) const {
    return Ort::Status();
  }

  virtual rnn_utils::ResetInput GetResetInput(const OrtNodeUnit& node_unit,
                                              const std::vector<std::string>& input_names) const = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
