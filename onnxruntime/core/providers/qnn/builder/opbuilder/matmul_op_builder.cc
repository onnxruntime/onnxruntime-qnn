// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <limits>
#include <numeric>

#include <gsl/gsl>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_bq_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {
// Detects a block-quantized MatMul weight (ONNX MatMul input[1]).
// Accepts weight rank 2–4: shape [..., K, N] where any leading dims beyond K/N must equal 1
// (i.e. reshapeable to [1, 1, K, N]). Per ONNX opset 21 the scale has the same rank as the
// weight with the contraction axis (K, at rank-2) dimension smaller: scale_shape[rank-2] <
// weight_shape[rank-2]. Only meaningful on the NPU backend.
bool IsBQWeight(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnitIODef& weight) {
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return false;
  }
  if (!weight.quant_param.has_value() || weight.quant_param->scale == nullptr) {
    return false;
  }
  const auto scale_shape = utils::GetInitializerShape(weight.quant_param->scale, qnn_model_wrapper.GetOrtApi());
  std::vector<uint32_t> weight_shape;
  if (!QnnModelWrapper::GetOnnxShape(weight.shape, weight_shape)) {
    return false;
  }
  const size_t rank = weight_shape.size();
  if (rank < 2 || rank > 4) {
    return false;  // BQ supports weight rank 2–4 (reshapeable to [1,1,K,N]).
  }
  // All leading dims (beyond the last two: K and N) must be 1.
  for (size_t i = 0; i + 2 < rank; ++i) {
    if (weight_shape[i] != 1) {
      return false;
    }
  }
  if (scale_shape.size() != rank) {
    return false;  // Scale must have the same rank as the weight.
  }
  // All leading dims of the scale must also be 1.
  for (size_t i = 0; i + 2 < rank; ++i) {
    if (scale_shape[i] != 1) {
      return false;
    }
  }
  // Blocked axis is rank-2 (K dimension). scale_shape[rank-2] < weight_shape[rank-2].
  const size_t k_axis = rank - 2;
  return bq::IsBQScale(scale_shape, weight_shape, k_axis);
}

// Flattens the leading dims of `shape` (all but the last `n_trailing` dims) into a single
// uint32_t batch value. Defaults to n_trailing=1 (i.e. all dims except the last one).
Ort::Status FlattenLeadingDims(const std::vector<uint32_t>& shape, uint32_t& batch,
                               size_t n_trailing = 1) {
  RETURN_IF(shape.size() < n_trailing, "FlattenLeadingDims: n_trailing exceeds shape rank.");
  const int64_t batch_i64 = std::accumulate(shape.begin(), shape.end() - static_cast<ptrdiff_t>(n_trailing),
                                            static_cast<int64_t>(1), std::multiplies<int64_t>());
  RETURN_IF(batch_i64 <= 0 ||
                batch_i64 > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
            "MatMul: flattened batch dimension product overflows uint32_t.");
  batch = static_cast<uint32_t>(batch_i64);
  return Ort::Status();
}

// Determines which QNN op to lower the ONNX MatMul to.
// Sets use_fully_connected=true  -> lower to QNN FullyConnected
// Sets use_conv2d=true           -> lower to QNN Conv2D (LPBQ path)
// Both false                     -> lower to QNN MatMul
Ort::Status CheckInputs(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnitIODef& input_def_0,
                        const OrtNodeUnitIODef& input_def_1, TensorInfo& input_info_0, TensorInfo& input_info_1,
                        bool& use_fully_connected, bool& use_conv2d) {
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def_0, input_info_0));
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_def_1, input_info_1));

  // LPBQ weights require Conv2D lowering (1x1 filters) on NPU backends.
  // QNN Conv2D supports LPBQ (blockwise expansion) on the filter channel axis (axis 3 in HWCN format).
  // This check must come before the FullyConnected logic so that LPBQ weights are never routed to FC.
  // input_0 must be rank >= 2 so that we can identify the M (width) and K (channel) dimensions.
  use_conv2d = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) &&
               input_info_1.quant_param.IsLPBQ() &&
               input_info_1.shape.size() == 2 &&
               input_info_1.is_initializer &&
               input_info_0.shape.size() >= 2 &&
               utils::IsQuant16bit(input_info_0.qnn_data_type);

#if QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR <= 20
  // Validation crashes if use QNN FullyConnected in QNN SDK versions 2.26 - 2.27
  // Just use QNN MatMul for these older QNN SDK versions.
  use_fully_connected = false;
#else
  // Use FullyConnected if 2nd input is a rank 2 initializer or a rank 1 tensor.
  // FullyConnected cannot pass the Op validation if keep_dims is true, so if input_0 is per-channel quantized tensor
  // with rank > 2, it's not easy to set the quantization parameters for the output reshaped rank 2 tensor.
  // In this case, we will not use FullyConnected.
  use_fully_connected =
      (input_info_1.shape.size() == 2 && input_info_1.is_initializer) || input_info_1.shape.size() == 1;
  use_fully_connected =
      use_fully_connected && !(input_info_0.quant_param.IsPerChannel() && input_info_0.shape.size() > 2);
  // Don't use FullyConnected if both inputs are dynamic and uint16 (quantized)
  use_fully_connected = use_fully_connected && !(utils::IsQuant16bit(input_info_0.qnn_data_type) &&
                                                 !input_info_0.is_initializer &&
                                                 utils::IsQuant16bit(input_info_1.qnn_data_type) &&
                                                 !input_info_1.is_initializer);
  // Don't use FullyConnected for LPBQ weights
  use_fully_connected = use_fully_connected && !use_conv2d;
#endif
  return Ort::Status();
}

// Process input[0] for ONNX MatMul lowered to QNN MatMul, FullyConnected, or Conv2D.
// A Reshape node (or reshaped initializer) is inserted when any of the following is true:
//   1. is_rank1:          input is rank-1 (reshaped to [1, K] for MatMul/FC compatibility).
//   2. shape_mismatch:    target_shape is provided and differs from the current shape
//                         (used by the Conv2D path to produce 4D NHWC layout).
//   3. use_fully_connected && rank > 2: leading dims are flattened to a single batch dim
//                         so QNN FullyConnected receives a 2D input.
// Note: target_shape and use_fully_connected are mutually exclusive - the Conv2D path
// always passes target_shape and never sets use_fully_connected.
Ort::Status ProcessInput0(QnnModelWrapper& qnn_model_wrapper,
                          const TensorInfo& input_0_info,
                          const std::string& original_input_0_name,
                          std::vector<std::string>& input_names,
                          const Ort::Logger& logger,
                          bool do_op_validation,
                          bool use_fully_connected,
                          const std::vector<uint32_t>* target_shape = nullptr) {
  // use_fully_connected and target_shape (conv2d path) are mutually exclusive
  assert(!(use_fully_connected && target_shape != nullptr));
  const bool is_rank1 = input_0_info.shape.size() == 1;
  const bool shape_mismatch = (target_shape != nullptr && input_0_info.shape != *target_shape);
  const bool reshape_input_0 = is_rank1 || shape_mismatch || (use_fully_connected && input_0_info.shape.size() > 2);
  std::string actual_input_0_name = original_input_0_name;

  if (reshape_input_0) {
    actual_input_0_name = utils::UniqueNameGenerator().New(original_input_0_name, "_reshape");
    std::vector<uint32_t> reshape_target;
    if (shape_mismatch) {
      reshape_target = *target_shape;
    } else if (is_rank1) {
      reshape_target = {1, input_0_info.shape[0]};
    } else {
      uint32_t batch = 0;
      RETURN_IF_ERROR(FlattenLeadingDims(input_0_info.shape, batch));
      reshape_target = {batch, input_0_info.shape.back()};
    }
    QnnQuantParamsWrapper quant_param_reshaped = input_0_info.quant_param.Copy();
    if (is_rank1 || shape_mismatch) {
      RETURN_IF_ERROR(quant_param_reshaped.HandleUnsqueeze<uint32_t>(input_0_info.shape, reshape_target));
    }

    // If input_0 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_0_info.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_0_info.initializer_tensor, unpacked_tensor));
      QnnTensorWrapper input_tensorwrapper(actual_input_0_name, QNN_TENSOR_TYPE_STATIC, input_0_info.qnn_data_type,
                                           std::move(quant_param_reshaped), std::move(reshape_target), std::move(unpacked_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(original_input_0_name, actual_input_0_name,
                                                       input_0_info.shape, reshape_target,
                                                       input_0_info.qnn_data_type, input_0_info.quant_param,
                                                       quant_param_reshaped, do_op_validation,
                                                       qnn_model_wrapper.IsGraphInput(original_input_0_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(actual_input_0_name)) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + actual_input_0_name).c_str());
    } else {
      QnnTensorWrapper input_0_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_0_info, actual_input_0_name, input_0_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_0_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(actual_input_0_name);

  return Ort::Status();
}
}  // namespace

/**
 * An ONNX MatMul can be translated to either a QNN MatMul, a QNN FullyConnected, or a QNN Conv2D.
 * ONNX's MatMul supports inputs of rank 1, but neither QNN's MatMul nor FullyConnected support two rank 1 inputs.
 * So, we need to add Reshape Ops if necessary.
 * In two cases, FullyConnected (input_1's shape is [n, k]) is used instead of MatMul without extra Transpose Op:
 * 1. input_1 is a rank 2 initializer.
 * 2. input_1 is a rank 1 tensor.
 * For LPBQ-quantized weights on NPU backends, Conv2D with 1x1 filters is used instead of MatMul.
 */
class MatMulOpBuilder : public BaseOpBuilder {
 public:
  MatMulOpBuilder() : BaseOpBuilder("MatMulOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names, const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ProcessInputsForQnnMatMul(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const TensorInfo& input_info_0,
                                        const TensorInfo& input_info_1,
                                        const Ort::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const ORT_MUST_USE_RESULT;
  Ort::Status ProcessInputsForQnnFullyConnected(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const TensorInfo& input_info_0,
                                                const TensorInfo& input_info_1,
                                                const Ort::Logger& logger,
                                                std::vector<std::string>& input_names,
                                                bool do_op_validation) const ORT_MUST_USE_RESULT;
  Ort::Status ProcessInputsForQnnConv2D(QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnit& node_unit,
                                        const TensorInfo& input_info_0,
                                        const TensorInfo& input_info_1,
                                        const Ort::Logger& logger,
                                        std::vector<std::string>& input_names,
                                        bool do_op_validation) const ORT_MUST_USE_RESULT;
  // Block-quantized (BW_FLOAT_BLOCK) weight path. Translates to a QNN MatMul whose weight carries a
  // per-block float scale; activation is dequantized to FP16 and the FP16 output is re-quantized to INT16.
  Ort::Status ProcessInputsForBQMatMul(QnnModelWrapper& qnn_model_wrapper,
                                       const OrtNodeUnit& node_unit,
                                       const Ort::Logger& logger,
                                       std::vector<std::string>& input_names,
                                       bool do_op_validation) const ORT_MUST_USE_RESULT;
};

// Process operator inputs. Dispatches to other processing functions depending on whether we're
// translating an ONNX MatMul to a QNN MatMul, a QNN FullyConnected, or a QNN Conv2D.
Ort::Status MatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit,
                                           const Ort::Logger& logger, std::vector<std::string>& input_names,
                                           bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  bool use_fully_connected = false;
  bool use_conv2d = false;
  RETURN_IF_ERROR(
      CheckInputs(qnn_model_wrapper, inputs[0], inputs[1], input_info_0, input_info_1,
                  use_fully_connected, use_conv2d));

  // Block-quantized weight: translate to a QNN MatMul with a BW_FLOAT_BLOCK weight.
  if (IsBQWeight(qnn_model_wrapper, inputs[1]) && !input_info_1.quant_param.IsLPBQ()) {
    return ProcessInputsForBQMatMul(qnn_model_wrapper, node_unit, logger, input_names, do_op_validation);
  }

  if (use_conv2d) {
    return ProcessInputsForQnnConv2D(qnn_model_wrapper,
                                     node_unit,
                                     input_info_0,
                                     input_info_1,
                                     logger,
                                     input_names,
                                     do_op_validation);
  } else if (use_fully_connected) {
    return ProcessInputsForQnnFullyConnected(qnn_model_wrapper,
                                             node_unit,
                                             input_info_0,
                                             input_info_1,
                                             logger,
                                             input_names,
                                             do_op_validation);
  } else {
    return ProcessInputsForQnnMatMul(qnn_model_wrapper,
                                     node_unit,
                                     input_info_0,
                                     input_info_1,
                                     logger,
                                     input_names,
                                     do_op_validation);
  }
}

Ort::Status MatMulOpBuilder::ProcessInputsForQnnMatMul(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const TensorInfo& input_info_0,
                                                       const TensorInfo& input_info_1,
                                                       const Ort::Logger& logger,
                                                       std::vector<std::string>& input_names,
                                                       bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const bool reshape_input_1 = input_info_1.shape.size() == 1;

  const std::string& org_input_0_name = inputs[0].name;
  RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, org_input_0_name, input_names,
                                logger, do_op_validation, /*use_fully_connected=*/false));

  // Process input 1.
  const std::string& org_input_1_name = inputs[1].name;
  std::string input_1_name = org_input_1_name;
  if (reshape_input_1) {
    // Input[1] is a rank 1 tensor that needs to be reshaped.
    std::vector<uint32_t> shape_2d;
    QnnQuantParamsWrapper quant_param_2d = input_info_1.quant_param.Copy();
    input_1_name = utils::UniqueNameGenerator().New(org_input_1_name, "_reshape");
    shape_2d = {input_info_1.shape[0], 1};
    RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_info_1.shape, shape_2d));

    // If input_1 is initializer, unpack it and add the tensor with new quantization parameter and shape.
    // Otherwise, add a Reshape node.
    if (input_info_1.is_initializer) {
      std::vector<uint8_t> unpacked_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_info_1.initializer_tensor, unpacked_tensor));

      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
      QnnTensorWrapper input_tensorwrapper(input_1_name, tensor_type, input_info_1.qnn_data_type,
                                           std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(org_input_1_name, input_1_name, input_info_1.shape, shape_2d,
                                                       input_info_1.qnn_data_type, input_info_1.quant_param,
                                                       quant_param_2d, do_op_validation,
                                                       qnn_model_wrapper.IsGraphInput(org_input_1_name), false));
    }
  } else {
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_1_name)) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_1_name).c_str());
    } else {
      QnnTensorWrapper input_1_tensor;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(inputs[1], input_1_tensor));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_1_tensor)), "Failed to add tensor.");
    }
  }
  input_names.emplace_back(input_1_name);

  // Inserts a QNN Convert op before uint16 input[1] to avoid QNN HTP validation failure.
  //
  // QNN graph that fails validation:
  //     input_0_uint16 ---> MatMul ---> output_uint16
  //                         ^
  //                         |
  //     input_1_uint16 -----+
  //
  // For dynamic weights, QNN graph that passes validation:
  //     input_0_uint16 ---------------------------> MatMul ---> output_uint16
  //                                                   ^
  //                                                   |
  //     input_1_uint16_asym --> Convert(uint16_sym) --+
  //
  // For static weights, QNN graph that passes validation:
  //     input_0_uint16 ---------------------> MatMul ---> output_uint16
  //                                             ^
  //                                             |
  //     input_1_uint16 --> Convert(int16_sym) --+
  if (!input_info_0.is_initializer &&
      input_info_0.qnn_data_type == input_info_1.qnn_data_type &&
      input_info_0.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
    RETURN_IF_NOT(input_info_1.quant_param.IsPerTensor(),
                  "MatMul's activation inputs only support per-tensor quantization");
    const Qnn_QuantizeParams_t& quant_param = input_info_1.quant_param.Get();
    // insert Convert op after input1
    std::string convert_input_name = input_names.back();
    input_names.pop_back();
    const std::string convert_output_name = utils::UniqueNameGenerator().New(convert_input_name, "_convert");
    std::vector<uint32_t> input_1_shape = input_info_1.shape;
    if (reshape_input_1) {
      input_1_shape = {input_info_1.shape[0], 1};
    }
    if (!input_info_1.is_initializer) {
      // QNN offsets negate ONNX zero points, so symmetric uint16 uses -32768.
      constexpr int32_t kSymmetricU16Offset = -32768;
      if (quant_param.scaleOffsetEncoding.offset != kSymmetricU16Offset) {
        RETURN_IF_ERROR(utils::InsertConvertOp(qnn_model_wrapper,
                                               convert_input_name,
                                               convert_output_name,
                                               input_info_1.qnn_data_type,
                                               QNN_DATATYPE_UFIXED_POINT_16,
                                               quant_param.scaleOffsetEncoding.offset,
                                               quant_param.scaleOffsetEncoding.scale,
                                               input_1_shape,
                                               true,  // symmetric
                                               do_op_validation));
        input_names.push_back(convert_output_name);
      } else {
        input_names.push_back(convert_input_name);
      }
    } else {
      RETURN_IF_ERROR(utils::InsertConvertOp(qnn_model_wrapper,
                                             convert_input_name,
                                             convert_output_name,
                                             input_info_1.qnn_data_type,
                                             QNN_DATATYPE_SFIXED_POINT_16,
                                             quant_param.scaleOffsetEncoding.offset,
                                             quant_param.scaleOffsetEncoding.scale,
                                             input_1_shape,
                                             true,  // symmetric
                                             do_op_validation));
      input_names.push_back(convert_output_name);
    }
  }
  return Ort::Status();
}

Ort::Status MatMulOpBuilder::ProcessInputsForQnnFullyConnected(QnnModelWrapper& qnn_model_wrapper,
                                                               const OrtNodeUnit& node_unit,
                                                               const TensorInfo& input_info_0,
                                                               const TensorInfo& input_info_1,
                                                               const Ort::Logger& logger,
                                                               std::vector<std::string>& input_names,
                                                               bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const bool reshape_input_1 = input_info_1.shape.size() == 1;

  const std::string& org_input_0_name = inputs[0].name;
  RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, org_input_0_name, input_names,
                                logger, do_op_validation, /*use_fully_connected=*/true));

  // Process input 1.
  const std::string& org_input_1_name = inputs[1].name;
  std::string input_1_name = org_input_1_name;
  std::vector<uint32_t> shape_2d;
  QnnQuantParamsWrapper quant_param_2d = input_info_1.quant_param.Copy();
  if (reshape_input_1) {
    // Input[1] is a rank 1 tensor that needs to be reshaped.
    input_1_name = utils::UniqueNameGenerator().New(org_input_1_name, "_reshape");

    // FullyConnected requires input_1's shape to be [n, k].
    shape_2d = {1, input_info_1.shape[0]};
    RETURN_IF_ERROR(quant_param_2d.HandleUnsqueeze<uint32_t>(input_info_1.shape, shape_2d));
  } else {
    assert(input_info_1.shape.size() == 2);
    input_1_name = utils::UniqueNameGenerator().New(org_input_1_name, "_transpose");
    shape_2d = {input_info_1.shape[1], input_info_1.shape[0]};
    RETURN_IF_ERROR(quant_param_2d.HandleTranspose<uint32_t>(std::vector<uint32_t>({1, 0})));
  }

  // If input_1 is initializer, unpack it and add the tensor with new quantization parameter and shape.
  // Otherwise, add a Reshape node.
  if (input_info_1.is_initializer) {
    std::vector<uint8_t> unpacked_tensor;
    if (!reshape_input_1) {
      // 2D initializer should be transposed to [n, k].
      std::vector<uint32_t> original_shape_copy = input_info_1.shape;
      RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper,
                                                   original_shape_copy,  // Will be modified to new shape (unnecessary)
                                                   input_info_1.initializer_tensor,
                                                   unpacked_tensor,
                                                   logger,
                                                   do_op_validation));
    } else {
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_info_1.initializer_tensor, unpacked_tensor));
    }

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
    QnnTensorWrapper input_tensorwrapper(input_1_name, tensor_type, input_info_1.qnn_data_type,
                                         std::move(quant_param_2d), std::move(shape_2d), std::move(unpacked_tensor));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  } else {
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(org_input_1_name, input_1_name, input_info_1.shape, shape_2d,
                                                     input_info_1.qnn_data_type, input_info_1.quant_param,
                                                     quant_param_2d, do_op_validation,
                                                     qnn_model_wrapper.IsGraphInput(org_input_1_name), false));
  }
  input_names.emplace_back(input_1_name);

  // Workaround that inserts a QNN Convert op before input[1] (converts from quantized uint16 to signed symmetric int16)
  // to avoid a QNN validation failure.
  //
  // QNN graph WITHOUT workaround (fails validation):
  //     input_0_uint16 ---> FC ---> output_uint16
  //                         ^
  //                         |
  //     input_1_uint16 -----+
  //
  // QNN graph WITH workaround (passes validation):
  //     input_0_uint16 ----------------------> FC ---> output_uint16
  //                                            ^
  //                                            |
  //     input_1_uint16 --> Convert(to int16) --+

  std::string weight_input_name = input_names.back();
  const auto& weight_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(weight_input_name);

  if (weight_tensor_wrapper.GetTensorDataType() == QNN_DATATYPE_UFIXED_POINT_16) {
    const auto& quant_param_wrapper = weight_tensor_wrapper.GetQnnQuantParams();
    const Qnn_QuantizeParams_t& quant_param = quant_param_wrapper.Get();
    const auto& transformed_input1_shape = weight_tensor_wrapper.GetTensorDims();

    RETURN_IF_NOT(quant_param_wrapper.IsPerTensor(),
                  "FC's INT16 weight inputs only support INT16 per-tensor quantization");

    // Pop Conv weight. Insert Convert op after Weight
    input_names.pop_back();
    std::string convert_output_name = utils::UniqueNameGenerator().New(weight_input_name, "_convert");

    RETURN_IF_ERROR(utils::InsertConvertOp(qnn_model_wrapper,
                                           weight_input_name,
                                           convert_output_name,
                                           QNN_DATATYPE_UFIXED_POINT_16,
                                           QNN_DATATYPE_SFIXED_POINT_16,
                                           quant_param.scaleOffsetEncoding.offset,
                                           quant_param.scaleOffsetEncoding.scale,
                                           transformed_input1_shape,
                                           true,  // Symmetric
                                           do_op_validation));
    input_names.push_back(convert_output_name);
  }
  return Ort::Status();
}

// Lowers an ONNX MatMul with LPBQ-quantized weight to a QNN Conv2D with 1x1 filters.
//
// Activation reshape:
//   2D: [M, K]       -> [1, 1, M, K]  (NHWC)
//   3D: [B, M, K]    -> [B, 1, M, K]  (NHWC)
//   4D: [B, C, M, K] -> no change
// Weight unsqueeze:
//   [K, N] -> [1, 1, K, N]  (HWCN; LPBQ axis 1->3)
// Conv2D params: stride=[1,1], pad_amount=[[0,0],[0,0]].
Ort::Status MatMulOpBuilder::ProcessInputsForQnnConv2D(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       const TensorInfo& input_info_0,
                                                       const TensorInfo& input_info_1,
                                                       const Ort::Logger& logger,
                                                       std::vector<std::string>& input_names,
                                                       bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const std::string& org_input_0_name = inputs[0].name;
  const std::string& org_input_1_name = inputs[1].name;

  // Reshape input[0] to 4D NHWC based on input rank.
  //   2D: [M, K]       -> [1, 1, M, K]
  //   3D: [B, M, K]    -> [B, 1, M, K]
  //   4D: [B, C, M, K] -> no change
  {
    const auto& shape = input_info_0.shape;
    std::vector<uint32_t> conv_input_shape;
    if (shape.size() == 2) {
      conv_input_shape = {1, 1, shape[0], shape[1]};
    } else if (shape.size() == 3) {
      conv_input_shape = {shape[0], 1, shape[1], shape[2]};
    } else if (shape.size() == 4) {
      conv_input_shape = shape;
    } else {
      return MAKE_EP_FAIL("LPBQ Conv2D lowering only supports activation rank 2, 3, or 4.");
    }
    RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, org_input_0_name, input_names,
                                  logger, do_op_validation, /*use_fully_connected=*/false, &conv_input_shape));
  }

  // Unsqueeze input[1] [K, N] -> [1, 1, K, N] (HWCN)
  // The LPBQ encoding axis is on the N (output-channel) dimension:
  // Axis 1 in [K, N]  ->  axis 3 in [1, 1, K, N]
  {
    // CheckInputs already checks the rank for input[1] for Conv2D. This check is just for any future refactoring guard.
    RETURN_IF_NOT(input_info_1.shape.size() == 2, "LPBQ Conv2D lowering requires weight to be rank-2 [K, N]");
    const uint32_t K = input_info_1.shape[0];
    const uint32_t N = input_info_1.shape[1];

    std::vector<uint32_t> conv_weight_shape = {1, 1, K, N};
    QnnQuantParamsWrapper conv_weight_quant = input_info_1.quant_param.Copy();
    RETURN_IF_ERROR(conv_weight_quant.HandleUnsqueeze<uint32_t>(input_info_1.shape, conv_weight_shape));

    const std::string conv_weight_name = utils::UniqueNameGenerator().New(org_input_1_name, "_reshape");

    RETURN_IF_NOT(input_info_1.is_initializer, "LPBQ Conv2D lowering requires weight to be static initializer");
    std::vector<uint8_t> unpacked_tensor;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_info_1.initializer_tensor, unpacked_tensor));

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(org_input_1_name);
    QnnTensorWrapper weight_tensorwrapper(conv_weight_name, tensor_type, input_info_1.qnn_data_type,
                                          std::move(conv_weight_quant), std::move(conv_weight_shape),
                                          std::move(unpacked_tensor));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensorwrapper)),
                  "Failed to add Conv2D weight tensor.");
    input_names.emplace_back(conv_weight_name);
  }

#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 16 && QNN_API_VERSION_MINOR <= 18)
  if (IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    // Bias is implicit. QNN SDK 2.23/2.24/2.25 (QNN API version 2.16/2.17/2.18) has a validation bug for
    // implicit bias inputs, so provide an explicit bias of all 0 (quantized int32).

    if (input_info_0.quant_param.IsPerTensor(/*include_bw*/ true) && input_info_1.quant_param.IsQuantized()) {
      const std::string bias_name = qnn::utils::UniqueNameGenerator().New(node_unit, "_implicit_bias");
      std::vector<uint32_t> bias_shape = {input_info_1.shape[0]};
      RETURN_IF_ERROR(AddZeroBiasInput(qnn_model_wrapper, input_info_0.quant_param, input_info_1.quant_param,
                                       std::move(bias_shape), bias_name, logger, input_names));
    }
  }
#endif

  return Ort::Status();
}

Ort::Status MatMulOpBuilder::ProcessInputsForBQMatMul(QnnModelWrapper& qnn_model_wrapper,
                                                      const OrtNodeUnit& node_unit,
                                                      const Ort::Logger& logger,
                                                      std::vector<std::string>& input_names,
                                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  // Weight may be rank 2–4 with shape [..., K, N] (leading dims are 1).
  // K and N are always the last two dimensions; the weight is registered as [1, 1, K, N] in QNN.
  TensorInfo input_info_1{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input_info_1));
  RETURN_IF_NOT(input_info_1.is_initializer, "QNN EP: BQ MatMul weight must be a constant initializer");
  const size_t w_rank = input_info_1.shape.size();
  RETURN_IF_NOT(w_rank >= 2 && w_rank <= 4,
                "QNN EP: BQ MatMul weight must be rank 2, 3, or 4 with shape [..., K, N]");
  const int64_t K = static_cast<int64_t>(input_info_1.shape[w_rank - 2]);
  const int64_t N = static_cast<int64_t>(input_info_1.shape[w_rank - 1]);

  //
  // Input 0: activation. BW_FLOAT_BLOCK MatMul computes in FP16, so an INT16 activation must be
  // dequantized to FP16 first (mirrors the Conv BQ activation path). QNN HTP additionally requires the
  // activation to be 4-D, so the [..., M, K] activation is reshaped to [batch, 1, M, K], where batch is
  // the product of all leading dims. Any rank >= 2 reshapes cleanly; the output is reshaped back the
  // same way.
  //
  TensorInfo input_info_0{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info_0));
  RETURN_IF_NOT(input_info_0.shape.size() >= 2,
                "QNN EP: BQ MatMul activation must be rank >= 2 so it can be reshaped to 4-D [batch, 1, M, K]");
  RETURN_IF_ERROR(ProcessInput0(qnn_model_wrapper, input_info_0, inputs[0].name, input_names, logger,
                                do_op_validation, /*use_fully_connected=*/false));
  {
    const std::string act_name = input_names[0];
    const auto& act_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(act_name);
    std::vector<uint32_t> act_shape = act_wrapper.GetTensorDims();

    // BW_FLOAT_BLOCK MatMul requires FP16 activation; dequantize the INT16 activation to FP16.
    // Reuse the original DequantizeLinear output name for the FP16 tensor so the QNN graph
    // stays aligned with the ONNX graph naming.
    const std::string fp16_name = Ort::ConstNode(&node_unit.GetNode()).GetInputs()[0].GetName();
    RETURN_IF_ERROR(bq::AddInt16ToFp16DequantForActivation(qnn_model_wrapper, act_name,
                                                           fp16_name, do_op_validation, "MatMul"));

    // Reshape the FP16 activation [..., M, K] to 4-D [batch, 1, M, K] for the QNN HTP BQ MatMul.
    const uint32_t k_dim = act_shape.back();
    const uint32_t m_dim = act_shape[act_shape.size() - 2];
    uint32_t batch = 0;
    RETURN_IF_ERROR(FlattenLeadingDims(act_shape, batch, /*n_trailing=*/2));
    const std::vector<uint32_t> act_shape_4d = {batch, 1u, m_dim, k_dim};
    const std::string act_4d_name = utils::UniqueNameGenerator().New(fp16_name, "_reshape_4d");
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(fp16_name, act_4d_name, act_shape, act_shape_4d,
                                                     QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                                     do_op_validation,
                                                     /*is_for_input=*/false, /*is_for_output=*/false));
    input_names[0] = act_4d_name;
  }

  //
  // Input 1: weight. Build QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK quant params.
  // The weight is always registered as 4-D [1, 1, K, N] in QNN regardless of the ONNX rank.
  //
  const std::string& input1_name = inputs[1].name;

  // Scale shape mirrors weight rank: [..., num_blocks, N] — blocked axis is rank-2.
  const auto scale_shape = utils::GetInitializerShape(inputs[1].quant_param->scale, qnn_model_wrapper.GetOrtApi());
  RETURN_IF_NOT(scale_shape.size() == w_rank,
                "QNN EP: BQ MatMul scale rank must match weight rank");
  const int64_t num_blocks = scale_shape[w_rank - 2];
  int64_t block_size = 0;
  RETURN_IF_ERROR(bq::ResolveBlockSize(inputs[1], K, num_blocks, "MatMul", block_size));
  const uint32_t bitwidth = bq::GetBQBitwidth(inputs[1].type);
  RETURN_IF_ERROR(bq::ValidateBQBitwidthAndBlockSize(bitwidth, block_size, "MatMul"));

  // Unpack the weight to one byte per element (sub-byte INT2/INT4 expanded to INT8).
  std::vector<uint8_t> unpacked_tensor;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_info_1.initializer_tensor, unpacked_tensor));

  // For unsigned types (UINT2/UINT4/UINT8), shift weight data to the signed domain.
  const bool is_unsigned_weight = bq::IsUnsignedBQType(inputs[1].type);
  if (is_unsigned_weight) {
    RETURN_IF_ERROR(utils::TransformUnsignedToSignedFixedPoint(unpacked_tensor,
                                                               static_cast<int64_t>(bitwidth)));
  }

  // QNN HTP requires a BQ MatMul to be expressed with 4-D activation, 4-D weight, and a 4-D blockSize.
  // The weight [..., K, N] is always registered as [1, 1, K, N]; with transpose_in1 = 0 the
  // contraction axis K is axis 2, so blockSize is {1, 1, block_size, 1}.
  const std::vector<uint32_t> block_size_arr = {1u, 1u, static_cast<uint32_t>(block_size), 1u};

  // ONNX per-block float scales are laid out [..., num_blocks, N] (block-major along K-axis).
  // QNN expects the scale/offset array output-channel-major: [N, num_blocks].
  std::vector<float> onnx_scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(inputs[1].quant_param->scale, onnx_scales));
  RETURN_IF_NOT(static_cast<int64_t>(onnx_scales.size()) == num_blocks * N,
                "QNN EP: BQ MatMul scale size mismatch");

  // Float offsets in ONNX [num_blocks, N] order before transpose.
  std::vector<float> onnx_offsets;
  RETURN_IF_ERROR(bq::ComputeBQOffsets(qnn_model_wrapper, inputs[1].quant_param->zero_point,
                                       is_unsigned_weight, bitwidth, num_blocks * N, onnx_offsets));

  // Transpose scales/offsets [num_blocks, N] → [N, num_blocks].
  const std::vector<uint32_t> transpose_shape = {static_cast<uint32_t>(num_blocks), static_cast<uint32_t>(N)};
  std::vector<float> scales_qnn, offsets_qnn;
  RETURN_IF_ERROR(utils::TwoDimensionTranspose<float>(onnx_scales, transpose_shape, scales_qnn, logger));
  RETURN_IF_ERROR(utils::TwoDimensionTranspose<float>(onnx_offsets, transpose_shape, offsets_qnn, logger));

  QnnQuantParamsWrapper bq_quant_params = QnnQuantParamsWrapper::BwFloatBlock(gsl::span<const float>(scales_qnn),
                                                                              gsl::span<const float>(offsets_qnn),
                                                                              bitwidth,
                                                                              gsl::span<const uint32_t>(block_size_arr));

  // Always use SFIXED_POINT_8: unsigned types are pre-converted by TransformUnsignedToSignedFixedPoint.
  // Weight is reshaped to 4-D [1, 1, K, N] to satisfy the QNN HTP BQ MatMul requirement.
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input1_name);
  std::vector<uint32_t> weight_shape = {1u, 1u, static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
  QnnTensorWrapper bq_weight_wrapper(input1_name, tensor_type,
                                     QNN_DATATYPE_SFIXED_POINT_8,
                                     std::move(bq_quant_params),
                                     std::move(weight_shape),
                                     std::move(unpacked_tensor));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bq_weight_wrapper)),
                "Failed to add BQ MatMul weight tensor.");
  input_names.push_back(input1_name);

  return Ort::Status();
}

Ort::Status MatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                         const OrtNodeUnit& node_unit,
                                                         std::vector<std::string>&& input_names,
                                                         const Ort::Logger& /*logger*/, bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  TensorInfo input_info_0{};
  TensorInfo input_info_1{};
  bool use_fully_connected = false;
  bool use_conv2d = false;
  std::string qnn_op_type;
  RETURN_IF_ERROR(
      CheckInputs(qnn_model_wrapper, inputs[0], inputs[1], input_info_0, input_info_1,
                  use_fully_connected, use_conv2d));

  // A block-quantized weight is always emitted as a QNN MatMul (see ProcessInputsForBQMatMul), even
  // when CheckInputs would otherwise route a rank-2 initializer weight to FullyConnected. Force the
  // MatMul path here so the output handling matches how the inputs were built.
  if (IsBQWeight(qnn_model_wrapper, inputs[1])) {
    use_fully_connected = false;
  }

  bool reshape_input_0 = input_info_0.shape.size() == 1;
  bool reshape_input_1 = input_info_1.shape.size() == 1;
  bool reshape_output = reshape_input_0 || reshape_input_1 || (use_fully_connected && input_info_0.shape.size() > 2) ||
                        (use_conv2d && input_info_0.shape.size() < 4);

  std::vector<std::string> param_tensor_names;
  if (use_conv2d) {
    qnn_op_type = QNN_OP_CONV_2D;
    // Conv2D params: stride=[1,1], pad_amount=[[0,0],[0,0]], dilation=[1,1], group=1
    QnnParamWrapper stride_param(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_STRIDE, {2}, {1, 1});
    param_tensor_names.push_back(stride_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(stride_param));

    QnnParamWrapper pad_param(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, {2, 2}, {0, 0, 0, 0});
    param_tensor_names.push_back(pad_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(pad_param));

    QnnParamWrapper dilation_param(node_unit.Index(), node_unit.Name(), QNN_OP_CONV_2D_PARAM_DILATION, {2}, {1, 1});
    param_tensor_names.push_back(dilation_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(dilation_param));

    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), 1,
                                           QNN_OP_CONV_2D_PARAM_GROUP, param_tensor_names));
  } else if (use_fully_connected) {
    qnn_op_type = QNN_OP_FULLY_CONNECTED;
  } else {
    qnn_op_type = QNN_OP_MAT_MUL;
    RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), false,
                                       QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, param_tensor_names));
    RETURN_IF_ERROR(AddQnnScalar<bool>(qnn_model_wrapper, node_unit.Index(), node_unit.Name(), false,
                                       QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, param_tensor_names));
  }

  const std::string& org_output_name = node_unit.Outputs()[0].name;
  std::string op_output_name = org_output_name;
  TensorInfo output_info{};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
  std::vector<uint32_t> op_output_shape = output_info.shape;
  QnnQuantParamsWrapper op_output_quant_param = output_info.quant_param.Copy();
  if (reshape_output) {
    op_output_name = utils::UniqueNameGenerator().New(org_output_name, "_reshape");
    if (use_conv2d) {
      op_output_shape.insert(op_output_shape.end() - 2, 1);
      if (op_output_shape.size() < 4)
        op_output_shape.insert(op_output_shape.begin(), 1);
      RETURN_IF_ERROR(op_output_quant_param.HandleUnsqueeze<uint32_t>(output_info.shape, op_output_shape));
    } else if (use_fully_connected && input_info_0.shape.size() > 2) {
      uint32_t batch = 0;
      RETURN_IF_ERROR(FlattenLeadingDims(input_info_0.shape, batch));
      op_output_shape = {batch, reshape_input_1 ? 1 : input_info_1.shape.back()};
      RETURN_IF(op_output_quant_param.IsPerChannel(), "QNN FC output does not support per-channel quant.");
    } else {
      // If both inputs are 1D tensors, the output shape is [1] instead of scalar. So if both inputs are 1D tensors,
      // we only need to add one "1" to the op_output_shape.
      if (reshape_input_1) {
        op_output_shape.emplace_back(1);
      } else if (reshape_input_0) {
        op_output_shape.insert(op_output_shape.end() - 1, 1);
      }
      RETURN_IF_ERROR(op_output_quant_param.HandleUnsqueeze<uint32_t>(output_info.shape, op_output_shape));
    }
  }

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
  const bool is_op_output_graph_output = is_graph_output && !reshape_output;
  Qnn_TensorType_t op_output_tensor_type =
      is_op_output_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // Detect a BQ (BW_FLOAT_BLOCK) MatMul from the weight tensor's quant encoding. input_names[1] is
  // the weight (BQ MatMul always has exactly 2 inputs and is never reshaped). A BQ MatMul computes
  // in FP16, so it must output FP16 and then re-quantize to the INT16 the downstream QDQ expects.
  bool is_bq_matmul = false;
  if (!use_fully_connected && input_names.size() > 1 &&
      qnn_model_wrapper.IsQnnTensorWrapperExist(input_names[1])) {
    is_bq_matmul = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]).GetQnnQuantParams().IsBlockQuantized();
  }

  if (is_bq_matmul) {
    // The QNN HTP BQ MatMul runs on 4-D tensors and outputs FP16.
    // Pipeline: MatMul (4-D FP16 [batch,1,M,N]) → Reshape (to ONNX [...,M,N] FP16)
    //           → Quantize (FP16 → INT16)
    RETURN_IF_NOT(output_info.quant_param.IsQuantized(),
                  "QNN EP: BQ MatMul output must be INT16-quantized; float output is not yet supported");
    const uint32_t n_dim = op_output_shape.back();
    const uint32_t m_dim = op_output_shape[op_output_shape.size() - 2];
    uint32_t batch = 0;
    RETURN_IF_ERROR(FlattenLeadingDims(op_output_shape, batch, /*n_trailing=*/2));
    const std::vector<uint32_t> matmul_out_shape_4d = {batch, 1u, m_dim, n_dim};

    const std::string matmul_4d_out = utils::UniqueNameGenerator().New(op_output_name, "_matmul_4d");
    QnnTensorWrapper matmul_4d_wrapper(matmul_4d_out, QNN_TENSOR_TYPE_NATIVE,
                                       QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                       std::vector<uint32_t>(matmul_out_shape_4d));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(matmul_4d_wrapper)),
                  "Failed to add 4-D FP16 BQ MatMul output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_MAT_MUL,
                                                  std::move(input_names), {matmul_4d_out},
                                                  std::move(param_tensor_names), do_op_validation),
                  "Failed to add BQ MatMul node.");

    // Reshape 4-D FP16 [batch,1,M,N] back to the ONNX FP16 output shape [...,M,N].
    // Reuse the original QuantizeLinear node's input name to keep the QNN graph aligned
    // with the ONNX graph naming.
    const std::string matmul_fp16_out = Ort::ConstNode(&node_unit.GetNode()).GetOutputs()[0].GetName();
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(matmul_4d_out, matmul_fp16_out, matmul_out_shape_4d,
                                                     op_output_shape, QNN_DATATYPE_FLOAT_16,
                                                     QnnQuantParamsWrapper(), do_op_validation,
                                                     /*is_for_input=*/false, /*is_for_output=*/false));

    RETURN_IF_ERROR(bq::AddFp16ToInt16QuantizeOutput(qnn_model_wrapper,
                                                     matmul_fp16_out, op_output_name,
                                                     op_output_tensor_type, output_info.qnn_data_type,
                                                     op_output_quant_param.Copy(),
                                                     op_output_shape, do_op_validation));
  } else {
    QnnTensorWrapper op_output_tensor_wrapper(op_output_name, op_output_tensor_type, output_info.qnn_data_type,
                                              op_output_quant_param.Copy(), std::vector<uint32_t>(op_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(op_output_tensor_wrapper)),
                  "Failed to add output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, qnn_op_type,
                                                  std::move(input_names), {op_output_name},
                                                  std::move(param_tensor_names), do_op_validation),
                  "Failed to add fused Matmul node.");
  }

  if (reshape_output) {
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(
        op_output_name, org_output_name, op_output_shape, output_info.shape, output_info.qnn_data_type,
        op_output_quant_param, output_info.quant_param, do_op_validation, false, is_graph_output));
  }

  return Ort::Status();
}

void CreateMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
