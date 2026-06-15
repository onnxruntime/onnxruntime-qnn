// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>

#include <gsl/gsl>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

// HTP BQ Gemm/FC: supported weight bitwidths and their block_size divisor constraints.
const std::unordered_map<uint32_t, int64_t> kHtpGemmBQBitsAndBlockSizeMultipliers{
    {2, 16}, {4, 8}, {8, 4}};

// Returns BQ weight bitwidth (2/4/8) from an ONNX element data type, or 0 if unsupported.
uint32_t GetBQBitwidth(ONNXTensorElementDataType onnx_type) {
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return 8;
    default:
      return 0;
  }
}

// Detects a block-quantized Gemm weight B, accounting for transB.
// For transB=0: B is [K, N], scale is [K/block_size, N], blocked on axis 0.
// For transB=1: B is [N, K], scale is [N, K/block_size], blocked on axis 1.
// Both cases: K is the contraction axis (blocked axis). Returns true if blocked weight detected.
bool IsBQGemmWeight(const QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnitIODef& weight,
                    int64_t trans_b) {
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    return false;
  }
  if (!weight.quant_param.has_value() || weight.quant_param->scale == nullptr) {
    return false;
  }
  const auto scale_shape = utils::GetInitializerShape(weight.quant_param->scale, qnn_model_wrapper.GetOrtApi());
  std::vector<uint32_t> weight_shape;
  if (!QnnModelWrapper::GetOnnxShape(weight.shape, weight_shape) || weight_shape.size() != 2) {
    return false;  // BQ only for rank-2 Gemm weight.
  }
  if (scale_shape.size() != 2) {
    return false;
  }
  // For transB=0: B=[K,N], blocked on axis 0 → scale=[num_blocks, N], scale_shape[0] < weight_shape[0].
  // For transB=1: B=[N,K], blocked on axis 1 → scale=[N, num_blocks], scale_shape[1] < weight_shape[1].
  const int blocked_axis = (trans_b == 0) ? 0 : 1;
  if (scale_shape[blocked_axis] >= static_cast<int64_t>(weight_shape[blocked_axis])) {
    return false;
  }
  const int64_t num_blocks = scale_shape[blocked_axis];
  if (num_blocks <= 0 || static_cast<int64_t>(weight_shape[blocked_axis]) % num_blocks != 0) {
    return false;
  }
  return true;
}

}  // namespace

class GemmOpBuilder : public BaseOpBuilder {
 public:
  GemmOpBuilder() : BaseOpBuilder("GemmOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GemmOpBuilder);

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
  Ort::Status ExplictOpCheck(const OrtNodeUnit& node_unit) const;
  // Block-quantized (BW_FLOAT_BLOCK) weight path for Gemm→QNN FullyConnected.
  Ort::Status ProcessInputsForBQGemm(QnnModelWrapper& qnn_model_wrapper,
                                     const OrtNodeUnit& node_unit,
                                     int64_t trans_b,
                                     float beta,
                                     const Ort::Logger& logger,
                                     std::vector<std::string>& input_names,
                                     bool do_op_validation) const ORT_MUST_USE_RESULT;
};

Ort::Status GemmOpBuilder::ExplictOpCheck(const OrtNodeUnit& node_unit) const {
  OrtNodeAttrHelper node_helper(node_unit);
  auto alpha = node_helper.Get("alpha", (float)1.0);
  RETURN_IF(alpha != 1.0, "QNN FullyConnected Op only support alpha=1.0.");
  auto beta = node_helper.Get("beta", (float)1.0);
  RETURN_IF(beta != 1.0 && beta != 0.0, "QNN FullyConnected Op only support beta=1.0 or beta=0.0.");

  // input C shape need to be [M] or [1, M] (skip validation when beta=0.0 — C is ignored)
  if (node_unit.Inputs().size() == 3 && beta != 0.0f) {
    auto& inputB = node_unit.Inputs()[1];
    std::vector<uint32_t> inputB_shape;
    QnnModelWrapper::GetOnnxShape(inputB.shape, inputB_shape);

    auto& inputC = node_unit.Inputs()[2];
    std::vector<uint32_t> inputC_shape;
    QnnModelWrapper::GetOnnxShape(inputC.shape, inputC_shape);

    auto transB = node_helper.Get("transB", static_cast<int64_t>(0));
    auto M = (transB == 0) ? inputB_shape.at(1) : inputB_shape.at(0);
    RETURN_IF(inputC_shape.size() == 0 || (inputC_shape.size() == 1 && inputC_shape.at(0) != M) ||
                  (inputC_shape.size() == 2 && inputC_shape.at(1) != M),
              "QNN FullyConnected Op only support C with shape [N, M].");

    RETURN_IF(inputC_shape.size() == 2 && node_unit.Inputs()[2].quant_param.has_value() && inputC_shape.at(0) != 1,
              "QNN FullyConnected Op only support quantized C with shape [1, M].");
  }

  return Ort::Status();
}

Ort::Status GemmOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_ERROR(ExplictOpCheck(node_unit));
  }

  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t trans_b = node_helper.Get("transB", static_cast<int64_t>(0));
  const float beta = node_helper.Get("beta", 1.0f);
  const auto& inputs = node_unit.Inputs();

  // Block-quantized weight: translate to QNN FullyConnected with BW_FLOAT_BLOCK weight.
  if (IsBQGemmWeight(qnn_model_wrapper, inputs[1], trans_b)) {
    return ProcessInputsForBQGemm(qnn_model_wrapper, node_unit, trans_b, beta, logger, input_names,
                                  do_op_validation);
  }

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  // for Input A, B, C: 1 -- need transpose, 0 -- not needed
  std::vector<int64_t> input_trans_flag(3, 0);
  input_trans_flag.at(0) = node_helper.Get("transA", (int64_t)0);
  auto transB = node_helper.Get("transB", (int64_t)0);
  // QNN input_1 [m, n] vs Onnx [n, m]
  input_trans_flag.at(1) = transB == 0 ? 1 : 0;
  for (size_t input_i = 0; input_i < inputs.size(); ++input_i) {
    // beta=0.0: C has no effect on the output — skip it so FC receives only (A, B)
    if (input_i == 2 && beta == 0.0f) {
      continue;
    }

    QnnQuantParamsWrapper quantize_param;
    RETURN_IF_ERROR(quantize_param.Init(qnn_model_wrapper, inputs[input_i]));

    bool is_quantized_tensor = inputs[input_i].quant_param.has_value();
    const auto& input_name = inputs[input_i].name;

    // Only skip if the input tensor has already been added (by producer op) *and* we don't need
    // to transpose it.
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name) && input_trans_flag[input_i] == 0) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_name).c_str());
      input_names.push_back(input_name);
      continue;
    }

    ONNXTensorElementDataType input_type = inputs[input_i].type;
    RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_tensor, input_type, qnn_data_type));

    std::vector<uint32_t> input_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].shape, input_shape), "Cannot get shape");

    std::vector<uint8_t> unpacked_tensor;
    bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
    if (is_constant_input) {
      const auto* input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
      if (1 == input_trans_flag.at(input_i)) {
        RETURN_IF_ERROR(quantize_param.HandleTranspose<size_t>(std::vector<size_t>({1, 0})));
        RETURN_IF_ERROR(
            utils::TwoDimensionTranspose(qnn_model_wrapper, input_shape, input_tensor, unpacked_tensor, logger));
      } else {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_tensor, unpacked_tensor));
      }
    }

    std::string input_tensor_name = input_name;
    if (1 == input_trans_flag.at(input_i) && !is_constant_input) {
      RETURN_IF(quantize_param.IsPerChannel(), "Non-constant Gemm inputs only support per-tensor quantization");

      // Add Transpose node
      std::vector<uint32_t> old_input_shape(input_shape);
      input_shape[0] = old_input_shape[1];
      input_shape[1] = old_input_shape[0];
      const std::string& node_input_name(input_name);
      input_tensor_name = utils::UniqueNameGenerator().New(input_tensor_name, "_transpose");
      std::vector<uint32_t> perm{1, 0};
      RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), node_input_name, input_tensor_name,
                                                         old_input_shape, perm, input_shape,
                                                         qnn_data_type, quantize_param, do_op_validation,
                                                         qnn_model_wrapper.IsGraphInput(node_input_name)));
    }

    // Reshape [1, M] shape Bias.
    if (2 == input_i && 2 == input_shape.size() && input_shape[0] == 1) {
      input_shape[0] = input_shape[1];
      input_shape.resize(1);
    }

    input_names.push_back(input_tensor_name);
    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input_tensor_name);
    QnnTensorWrapper input_tensorwrapper(input_tensor_name, tensor_type, qnn_data_type, std::move(quantize_param),
                                         std::move(input_shape), std::move(unpacked_tensor));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  return Ort::Status();
}

Ort::Status GemmOpBuilder::ProcessInputsForBQGemm(QnnModelWrapper& qnn_model_wrapper,
                                                  const OrtNodeUnit& node_unit,
                                                  int64_t trans_b,
                                                  float beta,
                                                  const Ort::Logger& logger,
                                                  std::vector<std::string>& input_names,
                                                  bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  // transA=1 means the ONNX activation is [K, M]; QNN FullyConnected needs [M, K], so we insert a
  // Transpose after the FP16 dequantize.
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t trans_a = node_helper.Get("transA", static_cast<int64_t>(0));

  // Determine weight shape and K, N dimensions.
  // transB=0: B=[K,N], blocked on axis 0; transB=1: B=[N,K], blocked on axis 1.
  TensorInfo weight_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], weight_info));
  RETURN_IF_NOT(weight_info.is_initializer, "QNN EP: BQ Gemm weight must be a constant initializer");
  RETURN_IF_NOT(weight_info.shape.size() == 2, "QNN EP: BQ Gemm weight must be rank-2");

  // QNN FC weight is [N, K]. From ONNX:
  //   transB=0: B=[K,N] → must transpose to [N,K]; K is the blocked axis.
  //   transB=1: B=[N,K] → already [N,K], no transpose needed.
  const int64_t N = (trans_b == 0) ? static_cast<int64_t>(weight_info.shape[1])
                                   : static_cast<int64_t>(weight_info.shape[0]);
  const int64_t K = (trans_b == 0) ? static_cast<int64_t>(weight_info.shape[0])
                                   : static_cast<int64_t>(weight_info.shape[1]);

  //
  // Input A (activation): dequantize INT16→FP16 if needed, then transpose to [M, K] if transA=1.
  // QNN HTP BQ FullyConnected accepts a 2-D [M, K] activation with a 2-D BW_FLOAT_BLOCK weight.
  //
  TensorInfo act_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], act_info));
  RETURN_IF_NOT(act_info.shape.size() == 2,
                "QNN EP: BQ Gemm activation must be rank-2 ([M, K], or [K, M] when transA=1)");

  // Add activation to QNN graph (handles graph-input / already-added cases).
  if (!qnn_model_wrapper.IsQnnTensorWrapperExist(inputs[0].name)) {
    QnnTensorWrapper act_wrapper;
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(act_info, inputs[0].name, act_wrapper));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(act_wrapper)), "Failed to add act tensor.");
  }
  input_names.push_back(inputs[0].name);

  {
    const std::string act_name = input_names[0];
    const auto& act_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(act_name);
    const Qnn_DataType_t act_dtype = act_wrapper.GetTensorDataType();
    // BW_FLOAT_BLOCK FC requires FP16 activation. The only activation dtype reaching this
    // path through the QDQ selector is INT16 (SFIXED or UFIXED), so anything else is unexpected.
    RETURN_IF_NOT(act_dtype == QNN_DATATYPE_SFIXED_POINT_16 || act_dtype == QNN_DATATYPE_UFIXED_POINT_16,
                  "QNN EP: BQ Gemm activation must be INT16-quantized for the BW_FLOAT_BLOCK kernel");
    // Reuse the original DequantizeLinear node's output name for the FP16 tensor.
    const std::string fp16_name = Ort::ConstNode(&node_unit.GetNode()).GetInputs()[0].GetName();
    const std::vector<uint32_t> act_shape_2d = act_wrapper.GetTensorDims();
    QnnTensorWrapper fp16_wrapper(fp16_name, QNN_TENSOR_TYPE_NATIVE,
                                  QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(act_shape_2d));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp16_wrapper)),
                  "Failed to add FP16 activation tensor for BQ Gemm.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(act_name, "_int16_dequantize"),
                      QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_DEQUANTIZE,
                      {act_name}, {fp16_name}, {}, do_op_validation),
                  "Failed to add INT16→FP16 Dequantize node for BQ Gemm activation.");
    input_names[0] = fp16_name;

    // transA=1: the FP16 activation is [K, M]; transpose to [M, K] for QNN FullyConnected.
    if (trans_a != 0) {
      RETURN_IF_NOT(act_shape_2d.size() == 2, "QNN EP: BQ Gemm transA=1 requires a rank-2 activation");
      const std::vector<uint32_t> transposed_shape = {act_shape_2d[1], act_shape_2d[0]};
      const std::string transposed_name = utils::UniqueNameGenerator().New(fp16_name, "_transpose");
      RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), fp16_name, transposed_name,
                                                         act_shape_2d, /*transpose_perm=*/{1u, 0u},
                                                         transposed_shape, QNN_DATATYPE_FLOAT_16,
                                                         QnnQuantParamsWrapper(), do_op_validation,
                                                         /*is_for_input=*/false, /*is_for_output=*/false));
      input_names[0] = transposed_name;
    }
  }

  //
  // Weight B: orient to [N, K] (QNN FC weight layout), build 2-D BW_FLOAT_BLOCK quant params.
  // QNN HTP BQ FullyConnected accepts a 2-D weight [N, K] with block_sizes={1, block_size}:
  // K is the contraction axis (axis 1), blocked into num_blocks chunks.
  //
  const std::string& weight_name = inputs[1].name;
  const auto scale_shape = utils::GetInitializerShape(inputs[1].quant_param->scale, qnn_model_wrapper.GetOrtApi());
  RETURN_IF_NOT(scale_shape.size() == 2, "QNN EP: BQ Gemm scale must be rank-2");

  // num_blocks is the number of blocks along K.
  const int64_t num_blocks = (trans_b == 0) ? scale_shape[0] : scale_shape[1];
  RETURN_IF(num_blocks <= 0 || K % num_blocks != 0, "QNN EP: BQ Gemm K must be divisible by num_blocks");
  const int64_t block_size = K / num_blocks;
  const uint32_t bitwidth = GetBQBitwidth(inputs[1].type);
  auto bq_it = kHtpGemmBQBitsAndBlockSizeMultipliers.find(bitwidth);
  RETURN_IF(bq_it == kHtpGemmBQBitsAndBlockSizeMultipliers.end(),
            ("QNN HTP Gemm BQ: unsupported weight bitwidth=" + std::to_string(bitwidth)).c_str());
  RETURN_IF(block_size % bq_it->second != 0,
            ("QNN HTP Gemm BQ: block_size=" + std::to_string(block_size) +
             " must be a multiple of " + std::to_string(bq_it->second) +
             " for " + std::to_string(bitwidth) + "-bit weight")
                .c_str());

  // Unpack weight to one byte per element (sub-byte INT2/INT4 expanded to INT8).
  std::vector<uint8_t> unpacked_weight;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(weight_info.initializer_tensor, unpacked_weight));

  // For unsigned types, shift to signed domain.
  const bool is_unsigned = (inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2 ||
                            inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 ||
                            inputs[1].type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
  if (is_unsigned) {
    RETURN_IF_ERROR(utils::TransformUnsignedToSignedFixedPoint(unpacked_weight, static_cast<int64_t>(bitwidth)));
  }

  // Transpose B to [N, K] if transB=0 (ONNX B is [K, N]).
  if (trans_b == 0) {
    std::vector<uint32_t> kn_shape = {static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
    std::vector<uint8_t> transposed;
    RETURN_IF_ERROR(utils::TwoDimensionTranspose<uint8_t>(unpacked_weight, kn_shape, transposed,
                                                          logger, do_op_validation));
    unpacked_weight = std::move(transposed);
  }
  // transB=1: B=[N,K] already in the right layout.

  // block_sizes for 2-D weight [N, K]: K is axis 1, so block_sizes = {1, block_size}.
  const std::vector<uint32_t> block_size_arr = {1u, static_cast<uint32_t>(block_size)};

  // Scales/offsets must be in [N, num_blocks] order (N-major = row-major for [N,K] weight):
  //   transB=0: ONNX scale=[num_blocks, N] → transpose to [N, num_blocks].
  //   transB=1: ONNX scale=[N, num_blocks] → already correct.
  std::vector<float> onnx_scales;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(inputs[1].quant_param->scale, onnx_scales));
  RETURN_IF_NOT(static_cast<int64_t>(onnx_scales.size()) == N * num_blocks,
                "QNN EP: BQ Gemm scale size mismatch");

  // Float offsets: unsigned_bias - onnx_zp (matching Conv BQ convention).
  const float unsigned_bias = is_unsigned ? static_cast<float>(1u << (bitwidth - 1)) : 0.0f;
  std::vector<float> onnx_offsets(static_cast<size_t>(N * num_blocks), unsigned_bias);
  if (inputs[1].quant_param->zero_point != nullptr) {
    std::vector<int32_t> zp_values;
    ONNXTensorElementDataType zp_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(inputs[1].quant_param->zero_point, zp_values, zp_type));
    RETURN_IF_NOT(static_cast<int64_t>(zp_values.size()) == N * num_blocks,
                  "QNN EP: BQ Gemm zero_point size must match N * num_blocks");
    for (size_t i = 0; i < zp_values.size(); ++i) {
      onnx_offsets[i] = unsigned_bias - static_cast<float>(zp_values[i]);
    }
  }

  std::vector<float> scales_qnn, offsets_qnn;
  if (trans_b == 0) {
    // Transpose from [num_blocks, N] to [N, num_blocks].
    scales_qnn.resize(static_cast<size_t>(N * num_blocks));
    offsets_qnn.resize(static_cast<size_t>(N * num_blocks));
    for (int64_t b = 0; b < num_blocks; ++b) {
      for (int64_t n = 0; n < N; ++n) {
        const size_t src = static_cast<size_t>(b * N + n);
        const size_t dst = static_cast<size_t>(n * num_blocks + b);
        scales_qnn[dst] = onnx_scales[src];
        offsets_qnn[dst] = onnx_offsets[src];
      }
    }
  } else {
    scales_qnn = std::move(onnx_scales);
    offsets_qnn = std::move(onnx_offsets);
  }

  QnnQuantParamsWrapper bq_quant_params(gsl::span<const float>(scales_qnn),
                                        gsl::span<const float>(offsets_qnn),
                                        bitwidth,
                                        gsl::span<const uint32_t>(block_size_arr));

  // 2-D weight [N, K] with BW_FLOAT_BLOCK encoding.
  std::vector<uint32_t> weight_shape_2d = {static_cast<uint32_t>(N), static_cast<uint32_t>(K)};
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(weight_name);
  QnnTensorWrapper bq_weight_wrapper(weight_name, tensor_type,
                                     QNN_DATATYPE_SFIXED_POINT_8,
                                     std::move(bq_quant_params),
                                     std::move(weight_shape_2d),
                                     std::move(unpacked_weight));
  RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bq_weight_wrapper)),
                "Failed to add BQ Gemm weight tensor.");
  input_names.push_back(weight_name);

  //
  // Input C (bias): must be an INT32-quantized initializer; dequantize to FP16 for the BQ kernel.
  // If beta=0.0, skip bias (existing convention). Float bias is not yet supported.
  //
  if (inputs.size() == 3 && beta != 0.0f) {
    TensorInfo bias_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[2], bias_info));
    RETURN_IF(!bias_info.is_initializer, "QNN EP: BQ Gemm bias must be a constant initializer");

    std::vector<uint32_t> bias_shape = bias_info.shape;
    // Collapse [1, N]→[N] (the existing Gemm convention for bias).
    if (bias_shape.size() == 2 && bias_shape[0] == 1) {
      bias_shape = {bias_shape[1]};
    }

    const std::string fp16_bias_name = utils::UniqueNameGenerator().New(inputs[2].name, "_fp16");
    std::vector<uint8_t> fp16_bias_bytes(static_cast<size_t>(N) * sizeof(uint16_t));

    RETURN_IF_NOT(bias_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_32,
                  "QNN EP: BQ Gemm bias must be INT32-quantized; float bias is not yet supported");
    // Dequantize INT32 bias to FP16 using per-tensor or per-channel scale.
    std::vector<uint8_t> raw_bias_bytes;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(bias_info.initializer_tensor, raw_bias_bytes));
    std::vector<float> bias_scales;
    if (inputs[2].quant_param.has_value() && inputs[2].quant_param->scale != nullptr) {
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(inputs[2].quant_param->scale, bias_scales));
    }
    // The dequantization below assumes a symmetric (zero-point == 0) bias, which is the convention
    // for INT32 QDQ bias (bias_scale = input_scale * weight_scale, zp = 0). A non-zero zero-point
    // would require subtracting it before scaling; reject it rather than silently mis-dequantizing.
    if (inputs[2].quant_param.has_value() && inputs[2].quant_param->zero_point != nullptr) {
      std::vector<int32_t> bias_zps;
      ONNXTensorElementDataType bias_zp_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(inputs[2].quant_param->zero_point, bias_zps, bias_zp_type));
      for (const int32_t zp : bias_zps) {
        RETURN_IF(zp != 0, "QNN EP: BQ Gemm bias must use zero-point 0 (symmetric); non-zero is not supported");
      }
    }
    RETURN_IF_NOT(raw_bias_bytes.size() == static_cast<size_t>(N) * sizeof(int32_t),
                  "QNN EP: BQ Gemm INT32 bias size mismatch");
    const bool is_per_channel_bias = bias_scales.size() == static_cast<size_t>(N);
    const auto* i32_ptr = reinterpret_cast<const int32_t*>(raw_bias_bytes.data());
    auto* u16_ptr = reinterpret_cast<uint16_t*>(fp16_bias_bytes.data());
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
      const float scale = bias_scales.empty() ? 1.0f : (is_per_channel_bias ? bias_scales[i] : bias_scales[0]);
      const Ort::Float16_t fp16(static_cast<float>(i32_ptr[i]) * scale);
      memcpy(&u16_ptr[i], &fp16.val, sizeof(uint16_t));
    }

    QnnTensorWrapper fp16_bias_wrapper(fp16_bias_name, QNN_TENSOR_TYPE_STATIC,
                                       QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                       std::move(bias_shape),
                                       std::move(fp16_bias_bytes));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp16_bias_wrapper)),
                  "Failed to add FP16 bias tensor for BQ Gemm.");
    input_names.push_back(fp16_bias_name);
  }

  return Ort::Status();
}

Ort::Status GemmOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  OrtNodeAttrHelper node_helper(node_unit);
  auto beta = node_helper.Get("beta", (float)1.0);
  const int64_t trans_b_out = node_helper.Get("transB", static_cast<int64_t>(0));

  // Detect BQ (BW_FLOAT_BLOCK) Gemm using IsBQGemmWeight, consistent with ProcessInputs detection.
  // BQ Gemm→FC: activation stays 2-D, weight is 2-D [N,K] with BW_FLOAT_BLOCK.
  // FC outputs FP16 → re-quantize to INT16.
  if (IsBQGemmWeight(qnn_model_wrapper, node_unit.Inputs()[1], trans_b_out)) {
    const std::string& org_output_name = node_unit.Outputs()[0].name;
    TensorInfo output_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
    const std::vector<uint32_t>& output_shape = output_info.shape;  // [M, N]
    RETURN_IF_NOT(output_shape.size() == 2, "QNN EP: BQ Gemm output must be rank-2 [M, N]");
    RETURN_IF_NOT(output_info.quant_param.IsQuantized(),
                  "QNN EP: BQ Gemm output must be INT16-quantized; float output is not yet supported");

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);
    const Qnn_TensorType_t out_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

    // FullyConnected → 2-D FP16 intermediate tensor. Reuse the original QL node's input name
    // to keep the QNN graph aligned with the ONNX graph naming.
    const std::string fc_fp16_out = Ort::ConstNode(&node_unit.GetNode()).GetOutputs()[0].GetName();
    QnnTensorWrapper fc_fp16_wrapper(fc_fp16_out, QNN_TENSOR_TYPE_NATIVE,
                                     QNN_DATATYPE_FLOAT_16, QnnQuantParamsWrapper(),
                                     std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fc_fp16_wrapper)),
                  "Failed to add FP16 BQ Gemm FC output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                                  std::move(input_names), {fc_fp16_out},
                                                  {}, do_op_validation),
                  "Failed to add BQ FullyConnected node.");

    // FP16 → INT16 quantized output.
    QnnTensorWrapper int16_out_wrapper(org_output_name, out_tensor_type, output_info.qnn_data_type,
                                       output_info.quant_param.Copy(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(int16_out_wrapper)),
                  "Failed to add INT16 BQ Gemm output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(org_output_name, "_fp16_quantize"),
                      QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_QUANTIZE,
                      {fc_fp16_out}, {org_output_name}, {}, do_op_validation),
                  "Failed to add FP16→INT16 Quantize node for BQ Gemm output.");
    return Ort::Status();
  }

  // Non-BQ path: decompose Gemm into FullyConnected + Add when needed.
  bool split_gemm = false;
  if (node_unit.Inputs().size() == 3 && beta != 0.0f) {
    auto& input_c = node_unit.Inputs()[2];
    std::vector<uint32_t> input_c_shape;
    QnnModelWrapper::GetOnnxShape(input_c.shape, input_c_shape);

    // Split when input_c has 2d shape and not [1, M]
    split_gemm = (input_c_shape.size() == 2 && input_c_shape.at(0) != 1);

    // Split when bias is an intermediate (NATIVE) tensor produced by another op.
    // ORT's MatMulAddFusion can fuse MatMul+Add->Gemm where the Add's other input
    // is an intermediate tensor (e.g., output of another MatMul). QNN FC requires
    // bias to be either STATIC (constant) or APP_WRITE (graph input), not NATIVE.
    split_gemm = split_gemm || qnn_model_wrapper.GetTensorType(input_c.name) == QNN_TENSOR_TYPE_NATIVE;
  }

  if (split_gemm) {
    // If split_gemm, input and output of Gemm must at least 2d.
    const std::string& org_output_name = node_unit.Outputs()[0].name;
    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
    TensorInfo output_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));
    std::vector<uint32_t> output_shape = output_info.shape;
    QnnQuantParamsWrapper op_output_quant_param = output_info.quant_param.Copy();

    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);

    // Create FullyConnected Node
    std::vector<std::string> gemm_input_0_1;
    gemm_input_0_1.push_back(input_names[0]);
    gemm_input_0_1.push_back(input_names[1]);
    const std::string fc_output_name = onnxruntime::qnn::utils::UniqueNameGenerator().New(org_output_name, "_fc");
    QnnTensorWrapper fully_connected_output(fc_output_name, QNN_TENSOR_TYPE_NATIVE, input_info.qnn_data_type,
                                            QnnQuantParamsWrapper(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fully_connected_output)),
                  "Failed to add FullyConnected output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_FULLY_CONNECTED),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_FULLY_CONNECTED,
                                                  std::move(gemm_input_0_1),
                                                  {fc_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add FullyConnected node.");

    // Create Add Node
    Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper op_output_tensor_wrapper(org_output_name, op_output_tensor_type, output_info.qnn_data_type,
                                              op_output_quant_param.Copy(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(op_output_tensor_wrapper)),
                  "Failed to add ElementWiseAdd output tensor.");
    std::string bias_name = input_names[2];

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_ADD),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_ADD,
                                                  {fc_output_name, bias_name},
                                                  {org_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add ElementWiseAdd node.");
  } else {
    RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), {},
                                   logger, do_op_validation, GetQnnOpType(node_unit.OpType())));
  }
  return Ort::Status();
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GemmOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
