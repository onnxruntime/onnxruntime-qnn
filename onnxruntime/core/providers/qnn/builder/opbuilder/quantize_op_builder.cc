// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/opbuilder/qdq_constant_folding.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

// Handles ONNX QuantizeLinear and DequantizeLinear. Q/DQ have no QNN params, so
// only inputs & outputs are processed (plus a compile-time constant-fold short-circuit).
class QuantizeOpBuilder : public BaseOpBuilder {
 public:
  QuantizeOpBuilder() : BaseOpBuilder("QuantizeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QuantizeOpBuilder);

 protected:
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ValidateQdqNode(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const;
};

Ort::Status QuantizeOpBuilder::ValidateQdqNode(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const {
  const std::string& op_type = node_unit.OpType();

  if (op_type == "DequantizeLinear") {
    bool is_per_chan_quant = false;
    int64_t quant_axis = 0;
    RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(node_unit.Inputs()[0], is_per_chan_quant, quant_axis));
    // Per-channel standalone DQ is allowed only if the input is a compile-time constant;
    const bool is_input_const = qnn_model_wrapper.IsEffectivelyConstantInput(node_unit.Inputs()[0].name);
    RETURN_IF(is_per_chan_quant && !is_input_const,
              "QNN EP does not support a standalone DQ op with per-channel quantization");

    if (qnn_model_wrapper.GetModelSettings().offload_graph_io_quantization &&
        qnn_model_wrapper.IsGraphOutput(node_unit.Outputs()[0].name)) {
      // Only register the override for the first DQ node that consumes this graph output.
      // If another DQ node already maps to the same external name, skip registration so
      // that the second output becomes a separate APP_READ tensor instead of creating
      // two APP_READ tensors with the same external name (which reduces the composed
      // QNN graph's input count and causes a null slot in qnn_tensor_infos at runtime).
      if (!qnn_model_wrapper.IsExternalOverrideTarget(node_unit.Outputs()[0].name)) {
        // The tensor name override is used to align the output name of DLC produced by IRBackend
        // with the output name of original onnx graph for better consistency.
        qnn_model_wrapper.SetTensorNameOverride(/*internal=*/node_unit.Inputs()[0].name,
                                                /*external=*/node_unit.Outputs()[0].name);
      }
      return MAKE_EP_FAIL("QNN EP is configured to not take DQ nodes that generate a graph output.");
    }
  }

  if (op_type == "QuantizeLinear") {
    bool is_per_chan_quant = false;
    int64_t quant_axis = 0;
    RETURN_IF_ERROR(qnn_model_wrapper.IsPerChannelQuantized(node_unit.Outputs()[0], is_per_chan_quant, quant_axis));
    // Per-channel standalone Q is allowed only if the input is a compile-time constant;
    const bool is_input_const = qnn_model_wrapper.IsEffectivelyConstantInput(node_unit.Inputs()[0].name);
    RETURN_IF(is_per_chan_quant && !is_input_const,
              "QNN EP does not support a standalone Q op with per-channel quantization");

    if (qnn_model_wrapper.GetModelSettings().offload_graph_io_quantization &&
        qnn_model_wrapper.IsGraphInput(node_unit.Inputs()[0].name)) {
      // Only register the override for the first Q node that consumes this graph input.
      // If another Q node already maps to the same external name, skip registration so
      // that the second input becomes a separate APP_WRITE tensor instead of creating
      // two APP_WRITE tensors with the same external name (which reduces the composed
      // QNN graph's input count and causes a null slot in qnn_tensor_infos at runtime).
      if (!qnn_model_wrapper.IsExternalOverrideTarget(node_unit.Inputs()[0].name)) {
        // The tensor name override is used to align the input name of DLC produced by IRBackend
        // with the input name of original onnx graph for better consistency.
        qnn_model_wrapper.SetTensorNameOverride(/*internal=*/node_unit.Outputs()[0].name,
                                                /*external=*/node_unit.Inputs()[0].name);
      }
      return MAKE_EP_FAIL("QNN EP is configured to not take Q nodes that consume a graph input.");
    }
  }

  return Ort::Status();
}

Ort::Status QuantizeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                           const OrtNodeUnit& node_unit,
                                                           std::vector<std::string>&& input_names,
                                                           const Ort::Logger& logger,
                                                           bool do_op_validation) const {
  if (input_names.empty()) {
    return Ort::Status();
  }

  if (do_op_validation) {
    RETURN_IF_ERROR(ValidateQdqNode(qnn_model_wrapper, node_unit));
  }

  // Emit a STATIC tensor instead of an APP_WRITE input for standalone Q/DQ on constant inputs.
  if (CanFoldConstantQdq(qnn_model_wrapper, node_unit)) {
    Ort::Status fold_status = TryFoldConstantQDQ(qnn_model_wrapper, node_unit);
    if (fold_status.IsOK()) {
      return Ort::Status();
    }
  }

  return ProcessOutputs(qnn_model_wrapper, node_unit,
                        std::move(input_names),
                        /*param_tensor_names=*/{},
                        logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
}

void CreateQuantizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<QuantizeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
