// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

class SizeOpBuilder : public BaseOpBuilder {
 public:
  SizeOpBuilder() : BaseOpBuilder("SizeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SizeOpBuilder);

 protected:
  // Override IsOpSupported to skip ProcessDataTypes — Size accepts any input element type.
  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Always register the input tensor in the QNN tensor map.
  // For constant initializers this registers as QNN_TENSOR_TYPE_STATIC (no consumer required).
  // For folded constants (NATIVE tensors from a prior constant-fold) the tensor is
  // already registered; ProcessInput detects the duplicate and skips re-registration.
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  // Core constant-folding logic: compute the scalar element count and register it as a
  // QNN_TENSOR_TYPE_STATIC tensor (or emit a Cast node when the output is a graph output).
  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation = false) const override ORT_MUST_USE_RESULT;
};

Ort::Status SizeOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger) const {
  const auto& input_def = node_unit.Inputs()[0];

  // Only accept constant initializers and previously constant-folded tensors.
  // Live graph inputs (APP_WRITE) are rejected here; without a downstream QNN consumer
  // node the graph would fail with QNN error 6004 at inference time.
  const std::string& input_name = input_def.name;
  RETURN_IF_NOT(qnn_model_wrapper.IsEffectivelyConstantInput(input_name),
                "QNN EP Size op: input must be a constant initializer or folded constant.");

  // Require a fully static input shape. GetOnnxShape returns false if any dim is dynamic.
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(input_def.shape, input_shape),
                "QNN EP: Size op requires a fully static input shape (no dynamic dimensions).");
  // Skip ProcessDataTypes: Size accepts any ONNX element type and always outputs int64.
  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, /*do_op_validation=*/true);
}

Ort::Status SizeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool /*do_op_validation*/) const {
  // Always register the input tensor so SetupQnnInputOutput can find it.
  // For constant initializers this becomes QNN_TENSOR_TYPE_STATIC (no consumer required).
  const auto& input_0 = node_unit.Inputs()[0];
  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, input_0, logger, input_names));
  return Ort::Status();
}

Ort::Status SizeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(input_names);
  ORT_UNUSED_PARAMETER(logger);

  // Step 1: Get the fully-static input shape.
  // GetOnnxShape returns an empty vector for 0-D (scalar) inputs; product of {} = 1 (correct).
  const auto& input_def = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(QnnModelWrapper::GetOnnxShape(input_def.shape, input_shape),
                "QNN EP: Size op requires a fully static input shape.");

  // Step 2: Compute total element count.
  int64_t size_value = 1;
  for (uint32_t dim : input_shape) {
    size_value *= static_cast<int64_t>(dim);
  }

  // Step 3: Serialize as int32 (QNN represents int64 values as int32 internally).
  RETURN_IF(size_value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            "QNN EP: Size value exceeds int32 range.");
  int32_t v32 = static_cast<int32_t>(size_value);
  std::vector<uint8_t> data(sizeof(int32_t));
  memcpy(data.data(), &v32, sizeof(int32_t));

  // QNN scalar shape is represented as {1} (rank-1, single element).
  const std::vector<uint32_t> scalar_shape = {1};

  const std::string& out_name = node_unit.Outputs()[0].name;

  if (!qnn_model_wrapper.IsGraphOutput(out_name)) {
    // Internal tensor: register as a folded static constant (int32 scalar).
    QnnTensorWrapper tensor_wrapper(out_name,
                                    QNN_TENSOR_TYPE_STATIC,
                                    QNN_DATATYPE_INT_32,
                                    QnnQuantParamsWrapper(),
                                    std::vector<uint32_t>(scalar_shape),
                                    std::move(data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)),
                  "QNN EP: Failed to add Size output static tensor.");
    qnn_model_wrapper.MarkTensorAsFoldedConstant(out_name);
  } else {
    // Graph output: must be APP_READ and int64 (per ONNX spec).
    // QNN has no native int64 storage, so use an int32 static intermediate and cast to int64.
    const std::string i32_name = out_name + "_size_i32";
    QnnTensorWrapper i32_wrapper(i32_name,
                                 QNN_TENSOR_TYPE_STATIC,
                                 QNN_DATATYPE_INT_32,
                                 QnnQuantParamsWrapper(),
                                 std::vector<uint32_t>(scalar_shape),
                                 std::move(data));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(i32_wrapper)),
                  "QNN EP: Failed to add Size int32 intermediate tensor.");

    const std::string cast_node_name = out_name + "_size_cast";
    RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(cast_node_name,
                                                  i32_name,
                                                  out_name,
                                                  QNN_TENSOR_TYPE_APP_READ,
                                                  QNN_DATATYPE_INT_64,
                                                  QnnQuantParamsWrapper(),
                                                  std::vector<uint32_t>(scalar_shape),
                                                  do_op_validation));
  }

  return Ort::Status();
}

void CreateSizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SizeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
