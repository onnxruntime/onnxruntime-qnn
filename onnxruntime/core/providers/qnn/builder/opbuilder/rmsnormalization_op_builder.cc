// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
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
// RMSNormalization lets `scale` be any shape unidirectionally broadcastable to X, e.g. scale
// [1, 1, C] against X [1, S, C]. Returns `shape` with leading 1-dims dropped to reach
// `target_rank`, or std::nullopt if a non-1 dim would have to be dropped. A shape already at or
// below `target_rank` is returned unchanged, leaving QNN to validate it as before.
std::optional<std::vector<uint32_t>> TrySqueezeLeadingOnesTo(const std::vector<uint32_t>& shape,
                                                             size_t target_rank) {
  if (shape.size() <= target_rank) {
    return shape;
  }

  const size_t num_leading_dims = shape.size() - target_rank;
  if (std::any_of(shape.begin(), shape.begin() + num_leading_dims, [](uint32_t dim) { return dim != 1; })) {
    return std::nullopt;
  }
  return std::vector<uint32_t>(shape.begin() + num_leading_dims, shape.end());
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

 private:
  // Number of trailing axes X is normalized over, i.e. size(axes) in QNN's RmsNorm OpDef.
  Ort::Status GetAxesRank(const QnnModelWrapper& qnn_model_wrapper,
                          const OrtNodeUnit& node_unit,
                          size_t& axes_rank) const ORT_MUST_USE_RESULT;
};

Ort::Status RMSNormalizationOpBuilder::GetAxesRank(const QnnModelWrapper& qnn_model_wrapper,
                                                   const OrtNodeUnit& node_unit,
                                                   size_t& axes_rank) const {
  std::vector<uint32_t> input_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(node_unit.Inputs()[0].shape, input_shape),
                "Cannot get shape of input 0");
  int32_t axis = 0;
  RETURN_IF_ERROR(GetCanonicalizedAxisAttribute(qnn_model_wrapper, node_unit, "axis", -1, axis));
  axes_rank = input_shape.size() - static_cast<size_t>(axis);
  return Ort::Status();
}

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

  size_t axes_rank = 0;
  RETURN_IF_ERROR(GetAxesRank(qnn_model_wrapper, node_unit, axes_rank));

  // Additional constraints for NPU backend
  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (is_npu_backend) {
    RETURN_IF(axes_rank != 1,
              "QNN RMSNorm for NPU backend only supports axis with last input dimension");
  }

  // A scale rank that ProcessInputs cannot reconcile is rejected there, which propagates out of this
  // call and leaves the node unclaimed.
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

  // Squeeze the scale's leading 1-dims so it satisfies QNN's rank(gamma) == size(axes).
  size_t axes_rank = 0;
  RETURN_IF_ERROR(GetAxesRank(qnn_model_wrapper, node_unit, axes_rank));

  const std::optional<std::vector<uint32_t>> squeezed_shape =
      TrySqueezeLeadingOnesTo(scale_info.shape, axes_rank);
  RETURN_IF_NOT(squeezed_shape.has_value(),
                "QNN RMSNorm requires the scale rank to equal the number of normalized axes; this scale has "
                "non-1 leading dimensions and cannot be squeezed to match.");

  if (squeezed_shape->size() == scale_info.shape.size()) {
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[SCALE_IDX], logger, input_names));
  } else {
    const std::string& scale_name = inputs[SCALE_IDX].name;

    // Any axis-bearing or per-block encoding describes the pre-squeeze rank, and there is no helper to
    // remap one for a rank reduction. Such a gamma has no known producer, so reject it rather than
    // silently emitting a misaligned encoding.
    RETURN_IF(scale_info.quant_param.IsQuantized() && !scale_info.quant_param.IsPerTensor(/*include_bw=*/true),
              "QNN RMSNorm only supports per-tensor quantization on a scale that requires squeezing to match "
              "the number of normalized axes");

    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("RMSNorm node " + node_unit.Name() + ": squeezing scale `" + scale_name + "` to rank " +
                 std::to_string(squeezed_shape->size()) + " to match QNN RmsNorm's OpDef.")
                    .c_str());

    // Derived name so a scale shared with another consumer keeps its original rank there.
    const std::string squeezed_scale_name = utils::UniqueNameGenerator().New(scale_name, "_squeeze");

    if (scale_info.is_initializer) {
      // Dropping leading 1-dims does not change a static tensor's element layout.
      std::vector<uint8_t> scale_bytes;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(scale_info.initializer_tensor, scale_bytes));

      QnnTensorWrapper squeezed_scale(squeezed_scale_name,
                                      QNN_TENSOR_TYPE_STATIC,
                                      scale_info.qnn_data_type,
                                      scale_info.quant_param.Copy(),
                                      std::vector<uint32_t>(*squeezed_shape),
                                      std::move(scale_bytes));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(squeezed_scale)),
                    "Failed to add squeezed scale tensor for QNN RMSNorm node.");
    } else {
      // A computed scale can only be reshaped in-graph.
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(scale_name,
                                                       squeezed_scale_name,
                                                       scale_info.shape,
                                                       *squeezed_shape,
                                                       scale_info.qnn_data_type,
                                                       scale_info.quant_param,
                                                       do_op_validation,
                                                       qnn_model_wrapper.IsGraphInput(scale_name)));
    }

    input_names.push_back(squeezed_scale_name);
    scale_info.shape = *squeezed_shape;
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
  size_t axes_rank = 0;
  RETURN_IF_ERROR(GetAxesRank(qnn_model_wrapper, node_unit, axes_rank));
  std::vector<uint32_t> axes(axes_rank, 0);
  std::vector<uint32_t> axes_shape{SafeInt<uint32_t>(axes_rank)};
  axes[0] = static_cast<uint32_t>(input_shape.size() - axes_rank);
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
