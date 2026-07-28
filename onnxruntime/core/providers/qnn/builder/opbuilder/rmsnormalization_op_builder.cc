// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <optional>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// QNN's RmsNorm OpDef requires gamma (and beta) to be rank size(axes), whereas ONNX
// RMSNormalization lets `scale` be any shape unidirectionally broadcastable to X. The two
// only disagree by leading 1-dims, e.g. scale [1, 1, C] against X [1, S, C] with axes [2].
// Returns the number of leading 1-dims to drop to reach `target_rank`, or std::nullopt if
// the shape cannot be reconciled (a non-1 dim would have to be dropped).
std::optional<size_t> GetNumLeadingOnesToSqueeze(const std::vector<uint32_t>& shape, size_t target_rank) {
  if (shape.size() <= target_rank) {
    return 0;  // Already at or below the required rank; nothing to squeeze.
  }

  const size_t num_leading = shape.size() - target_rank;
  for (size_t i = 0; i < num_leading; ++i) {
    if (shape[i] != 1) {
      return std::nullopt;
    }
  }
  return num_leading;
}

}  // namespace

class RMSNormalizationOpBuilder : public BaseOpBuilder {
 public:
  RMSNormalizationOpBuilder() : BaseOpBuilder("RMSNormalizationOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RMSNormalizationOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override final ORT_MUST_USE_RESULT;

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

Ort::Status RMSNormalizationOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     const Ort::Logger& logger) const {
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  // Reject if the optional inv_std_var output is requested (SimplifiedLayerNormalization).
  // QNN RMSNorm only produces a single output (Y).
  RETURN_IF(outputs.size() > 1,
            "QNN RMSNorm only supports 1 output; "
            "SimplifiedLayerNormalization inv_std_var output is not supported.");

  // Validate scale input is present
  constexpr size_t SCALE_IDX = 1;
  const bool has_scale_input = inputs.size() > SCALE_IDX && inputs[SCALE_IDX].Exists();
  RETURN_IF_NOT(has_scale_input, "QNN EP requires scale input for RMSNorm operator");

  // Validate input and output rank constraints
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();
  RETURN_IF(input_rank > 4, "QNN RMSNorm only supports input rank <= 4");

  std::vector<uint32_t> output_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].shape, output_shape), "Cannot get shape of output 0");
  const size_t output_rank = output_shape.size();
  RETURN_IF(output_rank > 4, "QNN RMSNorm only supports output rank <= 4");

  int32_t axis = 0;
  RETURN_IF_ERROR(GetCanonicalizedAxisAttribute(qnn_model_wrapper, node_unit, "axis", -1, axis));

  // Additional constraints for NPU backend
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    RETURN_IF(static_cast<size_t>(axis) != input_rank - 1,
              "QNN RMSNorm for NPU backend only supports axis with last input dimension");
  }

  // QNN's RmsNorm requires rank(gamma) == size(axes). ProcessInputs squeezes leading 1-dims off
  // the scale to satisfy that, so reject here only when the shape genuinely cannot be reconciled.
  // Without this the node is claimed and then fails deep inside QNN op validation, which surfaces
  // as an opaque error code plus a graph split rather than a clean CPU fallback.
  const size_t axes_rank = input_rank - static_cast<size_t>(axis);

  std::vector<uint32_t> scale_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[SCALE_IDX].shape, scale_shape),
                "Cannot get shape of input 1 (scale)");
  RETURN_IF_NOT(GetNumLeadingOnesToSqueeze(scale_shape, axes_rank).has_value(),
                "QNN RMSNorm requires the scale rank to equal the number of normalized axes; "
                "this scale has non-1 leading dimensions and cannot be squeezed to match.");

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Ort::Status RMSNormalizationOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                     const OrtNodeUnit& node_unit,
                                                     const Ort::Logger& logger,
                                                     std::vector<std::string>& input_names,
                                                     bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  constexpr size_t X_IDX = 0;
  constexpr size_t SCALE_IDX = 1;

  RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[X_IDX], logger, input_names));

  TensorInfo scale_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[SCALE_IDX], scale_info));

  // QNN's RmsNorm requires rank(gamma) == size(axes), while ONNX allows any scale shape
  // unidirectionally broadcastable to X (e.g. [1, 1, C] for X [1, S, C]). Squeeze the leading
  // 1-dims so the tensor QNN sees matches the OpDef. IsOpSupported already rejected any shape
  // that cannot be reconciled this way.
  std::vector<uint32_t> x_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[X_IDX].shape, x_shape), "Cannot get shape of input 0");
  int32_t axis = 0;
  RETURN_IF_ERROR(GetCanonicalizedAxisAttribute(qnn_model_wrapper, node_unit, "axis", -1, axis));
  const size_t axes_rank = x_shape.size() - static_cast<size_t>(axis);

  const std::optional<size_t> num_leading_ones = GetNumLeadingOnesToSqueeze(scale_info.shape, axes_rank);
  RETURN_IF_NOT(num_leading_ones.has_value(),
                "QNN RMSNorm requires the scale rank to equal the number of normalized axes.");

  if (*num_leading_ones > 0) {
    const std::vector<uint32_t> squeezed_shape(scale_info.shape.begin() + *num_leading_ones,
                                               scale_info.shape.end());
    const std::string& orig_scale_name = inputs[SCALE_IDX].name;

    // Squeezing shifts a per-channel quantization axis, and there is no helper to remap it for a
    // rank reduction. This combination (per-channel gamma carrying leading 1-dims) has no known
    // producer, so reject it rather than silently emitting a misaligned axis.
    RETURN_IF(scale_info.quant_param.IsPerChannel() || scale_info.quant_param.IsLPBQ(),
              "QNN RMSNorm does not support per-channel/LPBQ quantization on a scale that requires "
              "squeezing to match the number of normalized axes");

    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("RMSNorm node " + node_unit.Name() + ": squeezing scale `" + orig_scale_name +
                 "` to rank " + std::to_string(squeezed_shape.size()) + " to match QNN RmsNorm's OpDef.")
                    .c_str());

    // Emit under a derived name so a scale shared with another consumer keeps its original rank
    // for that consumer.
    const std::string squeezed_scale_name = utils::UniqueNameGenerator().New(orig_scale_name, "_squeeze");

    if (scale_info.is_initializer) {
      // A static scale needs no Reshape node: dropping leading 1-dims does not change the element
      // layout, so only the declared dims change.
      std::vector<uint8_t> scale_bytes;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(scale_info.initializer_tensor, scale_bytes));

      QnnTensorWrapper scale_tensor_wrapper(squeezed_scale_name,
                                            QNN_TENSOR_TYPE_STATIC,
                                            scale_info.qnn_data_type,
                                            scale_info.quant_param.Copy(),
                                            std::vector<uint32_t>(squeezed_shape),
                                            std::move(scale_bytes));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(scale_tensor_wrapper)),
                    "Failed to add squeezed scale tensor for QNN RMSNorm node.");
    } else {
      // A dynamic scale is produced by another node, so its rank can only be changed in-graph.
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(orig_scale_name,
                                                       squeezed_scale_name,
                                                       scale_info.shape,
                                                       squeezed_shape,
                                                       scale_info.qnn_data_type,
                                                       scale_info.quant_param,
                                                       do_op_validation,
                                                       qnn_model_wrapper.IsGraphInput(orig_scale_name)));
    }

    input_names.push_back(squeezed_scale_name);
    scale_info.shape = squeezed_shape;
  } else {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[SCALE_IDX], logger, input_names));
  }

#if !defined(QNN_SDK_VERSION_MINOR) || (QNN_SDK_VERSION_MAJOR == 2 && QNN_SDK_VERSION_MINOR < 49)
  // QNN SDK < 2.49 requires an explicit beta/bias input for QNN_OP_RMS_NORM on NPU.
  // SDK 2.49+ accepts beta as optional, so the dummy tensor is only needed for older SDKs.
  // Note: SDK 2.47 and 2.48 share the same QNN API version (2.36), so QNN_SDK_VERSION_MINOR
  // derived from CMake is used here instead of QNN_API_VERSION_MINOR.
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("RMSNorm node " + node_unit.Name() + ": adding dummy beta tensor (SDK < 2.49).").c_str());

    // scale_info.shape is the post-squeeze shape, so beta inherits the OpDef-conformant rank.
    std::vector<uint32_t> beta_shape = scale_info.shape;

    // Match beta datatype to scale for float types, use UFIXED_POINT_8 for INT types
    Qnn_DataType_t beta_data_type = QNN_DATATYPE_UFIXED_POINT_8;
    if (scale_info.qnn_data_type == QNN_DATATYPE_FLOAT_32 ||
        scale_info.qnn_data_type == QNN_DATATYPE_FLOAT_16) {
      beta_data_type = scale_info.qnn_data_type;
    }

    // Use appropriate quantization parameters for zero values
    QnnQuantParamsWrapper beta_quant_param;
    if (scale_info.quant_param.IsQuantized()) {
      float quant_scale = 1.0f;
      int32_t zero_point = 0;
      beta_quant_param = QnnQuantParamsWrapper::PerTensor(quant_scale, zero_point);
    }

    const size_t beta_size_in_bytes = utils::GetQnnTensorDataSizeInBytes(beta_shape, beta_data_type);
    std::vector<uint8_t> beta_data(beta_size_in_bytes, 0);
    const std::string beta_tensor_name = node_unit.Name() + "_beta_dummy";
    QnnTensorWrapper beta_tensor_wrapper(beta_tensor_name,
                                         QNN_TENSOR_TYPE_STATIC,
                                         beta_data_type,
                                         std::move(beta_quant_param),
                                         std::move(beta_shape),
                                         std::move(beta_data));

    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(beta_tensor_wrapper)),
                  "Failed to add dummy beta tensor for QNN RMSNorm node.");
    input_names.push_back(beta_tensor_name);
  }
#else
  if (IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("RMSNorm node " + node_unit.Name() + ": skipping dummy beta tensor (SDK >= 2.49).").c_str());
  }
#endif  // !defined(QNN_SDK_VERSION_MINOR) || (QNN_SDK_VERSION_MAJOR == 2 && QNN_SDK_VERSION_MINOR < 49)

  return Ort::Status();
}

Ort::Status RMSNormalizationOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                                   const OrtNodeUnit& node_unit,
                                                                   std::vector<std::string>&& input_names,
                                                                   const Ort::Logger& logger,
                                                                   bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  // Process epsilon attribute
  const float epsilon = node_helper.Get("epsilon", 1e-05f);
  RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), epsilon,
                                      QNN_OP_RMS_NORM_PARAM_EPSILON, param_tensor_names));

  // Process axis attribute and create axes parameter
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape), "Cannot get shape of Input 0");
  const size_t input_rank = input_shape.size();
  int32_t axis = -1;
  RETURN_IF_ERROR(GetCanonicalizedAxisAttribute(qnn_model_wrapper, node_unit, "axis", -1, axis));
  size_t axes_rank = input_rank - static_cast<size_t>(axis);
  std::vector<uint32_t> axes(axes_rank, 0);
  std::vector<uint32_t> axes_shape{SafeInt<uint32_t>(axes_rank)};
  axes[0] = static_cast<uint32_t>(axis);
  for (size_t i = 1; i < axes.size(); ++i) {
    axes[i] = axes[i - 1] + 1;
  }

  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), QNN_OP_RMS_NORM_PARAM_AXES,
                             std::move(axes_shape), std::move(axes));
  param_tensor_names.push_back(axes_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axes_param));

  RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                 std::move(input_names),
                                 std::move(param_tensor_names),
                                 logger,
                                 do_op_validation,
                                 GetQnnOpType(node_unit.OpType())));
  return Ort::Status();
}

void CreateRMSNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<RMSNormalizationOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
