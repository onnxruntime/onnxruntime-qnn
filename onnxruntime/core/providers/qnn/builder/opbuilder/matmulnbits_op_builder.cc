// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_bq_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {
/* Op resolution
 *
 * =============================================================================================
 * Incoming ONNX node
 * =============================================================================================
 * MatMulNBits
 *   Attributes (int64)
 *     bits        : 2 (HTP), 4 (GPU/HTP), 8 (HTP)
 *     block_size  : 32 (GPU); multiple of 16 / 8 / 4 (HTP, for bits = 2 / 4 / 8)
 *     K           : contraction dim of input A
 *     N           : output channels
 *
 *     Name           Kind        Data type                       Shape
 *   Inputs
 *     A              input       fp16/fp32 (GPU)                 [batch_size, sequence_len, K]
 *                                fp16/fp32/uint16/int16 (HTP)
 *     B              init        uint8 (packed)                  [N, K / block_size, (block_size * bits) / 8]
 *     scales         init        fp16/fp32                       [N, K / block_size]
 *     zero_points    init (opt)  uint8 (packed)                  [N, (K / block_size) * bits / 8]
 *   Outputs
 *     Y              output      same as A                       [batch_size, sequence_len, N]
 *
 * =============================================================================================
 * Outgoing QNN nodes (GPU)
 *   A -> FullyConnected -> Reshape -> Y
 * =============================================================================================
 * 1. FullyConnected
 *      in   Input     fp16/fp32                [batch_size, sequence_len, K]
 *      in   Weight    qint4 (BlockEncoding)    [N, K]
 *                       scales   fp32          [N * (K / block_size)]
 *                       offsets  int32_t       [N * (K / block_size)]
 *      out  Output    fp16/fp32                [batch_size * sequence_len, N]
 * 2. Reshape
 *      in   Input     fp16/fp32                [batch_size * sequence_len, N]
 *      out  Output    fp16/fp32                [batch_size, sequence_len, N]
 *
 * =============================================================================================
 * Outgoing QNN nodes (HTP)
 *   A -> Reshape -> [Cast | Dequantize] -> Conv2d -> [Cast | Quantize] -> Reshape -> Y
 * =============================================================================================
 * 1. Reshape
 *      in   Input     fp16/fp32/uint16/int16   [batch_size, sequence_len, K]
 *      out  Output    fp16/fp32/uint16/int16   [batch_size, 1, sequence_len, K]
 * 2a. Cast
 *      in   Input     fp32                     [batch_size, 1, sequence_len, K]
 *      out  Output    fp16                     [batch_size, 1, sequence_len, K]
 * 2b. Dequantize
 *      in   Input     uint16/int16             [batch_size, 1, sequence_len, K]
 *      out  Output    fp16                     [batch_size, 1, sequence_len, K]
 * 3. Conv2d
 *      in   Input     fp16                     [batch_size, 1, sequence_len, K]
 *      in   Weight    qint8 (BwFloatBlock)     [1, 1, K, N]
 *                       scales   fp32          [N * (K / block_size)]
 *                       offsets  fp32          [N * (K / block_size)]
 *      out  Output    fp16                     [batch_size, 1, sequence_len, N]
 * 4a. Cast
 *      in   Input     fp16                     [batch_size, 1, sequence_len, N]
 *      out  Output    fp32                     [batch_size, 1, sequence_len, N]
 * 4b. Quantize
 *      in   Input     fp16                     [batch_size, 1, sequence_len, N]
 *      out  Output    uint16/int16             [batch_size, 1, sequence_len, N]
 * 5. Reshape
 *      in   Input     fp16/fp32/uint16/int16   [batch_size, 1, sequence_len, N]
 *      out  Output    fp16/fp32/uint16/int16   [batch_size, sequence_len, N]
 */

class MatMulNBitsOpBuilder : public BaseOpBuilder {
 public:
  MatMulNBitsOpBuilder() : BaseOpBuilder("MatMulNBitsOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulNBitsOpBuilder);

  Ort::Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger) const override ORT_MUST_USE_RESULT;

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

namespace {
constexpr int64_t kByteBits = 8;

// GPU-related constraints.
const std::unordered_set<int64_t> kGpuSupportedBits{4};
const std::unordered_set<int64_t> kGpuSupportedBlockSize{32};

// HTP-related constraints.
// HTP expects block size to be multiple of a value according to bits.
const std::unordered_map<int64_t, int64_t> kHtpSupportedBitsAndBlockSizeMultipliers{{2, 16}, {4, 8}, {8, 4}};

template <typename T>
void UnpackDataToDatatype(const std::vector<uint8_t>& packed_data,
                          const int64_t bits,
                          const int64_t num_elements_per_uint8,
                          std::vector<T>& unpacked_data) {
  unpacked_data.clear();
  unpacked_data.reserve(packed_data.size() * num_elements_per_uint8);

  const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1);

  for (const uint8_t& value : packed_data) {
    for (int64_t idx = 0; idx < num_elements_per_uint8; ++idx) {
      int64_t shift = bits * idx;
      uint8_t masked_val = (value >> shift) & mask;
      unpacked_data.push_back(static_cast<T>(masked_val));
    }
  }
}

}  // namespace

Ort::Status MatMulNBitsOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger) const {
  bool is_gpu_backend = IsGpuBackend(qnn_model_wrapper.GetQnnBackendType());
  bool is_htp_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  RETURN_IF_NOT(is_gpu_backend || is_htp_backend, "MatMulNBits is supported only for QNN GPU/HTP backend.");

  // Extract Parameters
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t K = node_helper.Get("K", static_cast<int64_t>(0));
  const int64_t N = node_helper.Get("N", static_cast<int64_t>(0));
  const int64_t bits = node_helper.Get("bits", static_cast<int64_t>(0));
  const int64_t block_size = node_helper.Get("block_size", static_cast<int64_t>(0));

  if (is_gpu_backend) {
    RETURN_IF(kGpuSupportedBits.find(bits) == kGpuSupportedBits.end(),
              ("QNN GPU does not support MatMulNBits with bits=" + std::to_string(bits)).c_str());
    RETURN_IF(kGpuSupportedBlockSize.find(block_size) == kGpuSupportedBlockSize.end(),
              ("QNN GPU does not support MatMulNBits with block_size=" + std::to_string(block_size)).c_str());
  } else {
    auto it = kHtpSupportedBitsAndBlockSizeMultipliers.find(bits);
    RETURN_IF(it == kHtpSupportedBitsAndBlockSizeMultipliers.end(),
              ("QNN HTP does not support MatMulNBits with bits=" + std::to_string(bits)).c_str());
    RETURN_IF(block_size % it->second != 0,
              ("QNN HTP does not support MatMulNBits with block_size=" + std::to_string(block_size)).c_str());
  }
  RETURN_IF_NOT((K % block_size) == 0, "K must be divisible by block_size.");

  const int64_t total_blocks = (N * K) / block_size;
  RETURN_IF_NOT(total_blocks > 0, "(N * K) / block_size must be > 0");

  const int64_t num_elements_per_uint8 = kByteBits / bits;
  const int64_t num_zp_per_uint8 = K / block_size < num_elements_per_uint8 ? K / block_size : num_elements_per_uint8;

  const auto& inputs = node_unit.Inputs();
  // 1. A: Input datatype should be:
  //   - GPU: float32, float16
  //   - HTP: float32, float16, uint16, int16
  {
    const OrtNodeUnitIODef& input_tensor = inputs[0];
    TensorInfo input_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(input_tensor, input_info));

    Qnn_DataType_t input_datatype;
    RETURN_IF_ERROR(utils::GetQnnDataType(input_tensor.quant_param.has_value(),
                                          input_tensor.type,
                                          input_datatype));

    if (is_gpu_backend) {
      RETURN_IF(input_datatype != QNN_DATATYPE_FLOAT_32 && input_datatype != QNN_DATATYPE_FLOAT_16,
                "Unsupported input A datatype, expecting float32 or float16.");
    } else {
      RETURN_IF(input_datatype != QNN_DATATYPE_FLOAT_32 &&
                    input_datatype != QNN_DATATYPE_FLOAT_16 &&
                    !utils::IsQuant16bit(input_datatype),
                "Unsupported input A datatype, expecting float32, float16, uint16, or int16.");
      // Restrict to 3D input due to later inserted Reshape.
      RETURN_IF(input_info.shape.size() != 3, "Unsupported input A rank, expecting 3D shape.");
    }
  }

  // 2. B: Weight is supported in packed uint8 format.
  {
    const OrtNodeUnitIODef& weight_tensor = inputs[1];
    TensorInfo weight_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(weight_tensor, weight_info));
    SafeInt<int64_t> safe_total_elements = std::accumulate(weight_info.shape.begin(),
                                                           weight_info.shape.end(),
                                                           SafeInt<int64_t>{1},
                                                           std::multiplies<>());
    const int64_t total_elements = static_cast<int64_t>(safe_total_elements);
    RETURN_IF_NOT((total_elements * num_elements_per_uint8) == (N * K),
                  ("Unexpected input B size, expecting " + std::to_string(N * K / num_elements_per_uint8)).c_str());
  }

  // 3. scales: Scales should be float32 datatype and have N*K/block_size elements.
  {
    const OrtNodeUnitIODef& scales_tensor = inputs[2];
    Qnn_DataType_t scales_datatype;
    RETURN_IF_ERROR(utils::GetQnnDataType(scales_tensor.quant_param.has_value(),
                                          scales_tensor.type,
                                          scales_datatype));
    RETURN_IF(scales_datatype != QNN_DATATYPE_FLOAT_32 && scales_datatype != QNN_DATATYPE_FLOAT_16,
              "Unsupported input scales datatype, expecting float32 or float16.");

    TensorInfo scales_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(scales_tensor, scales_info));
    SafeInt<int64_t> safe_total_elements = std::accumulate(scales_info.shape.begin(),
                                                           scales_info.shape.end(),
                                                           SafeInt<int64_t>{1},
                                                           std::multiplies<>());
    const int64_t total_elements = static_cast<int64_t>(safe_total_elements);
    RETURN_IF_NOT(total_elements == total_blocks,
                  ("Unexpected input scales size, expecting " + std::to_string(total_blocks)).c_str());
  }

  // 4. zero_points: (optional) Zero points should be uint8 datatype and have N*K/block_size/(8/bits).
  if (inputs.size() > 3 && inputs[3].Exists()) {
    const OrtNodeUnitIODef& zp_tensor = inputs[3];
    Qnn_DataType_t zp_datatype;
    RETURN_IF_ERROR(utils::GetQnnDataType(zp_tensor.quant_param.has_value(),
                                          zp_tensor.type,
                                          zp_datatype));
    RETURN_IF((zp_datatype != QNN_DATATYPE_UINT_8), "Unsupported input zero_points datatype, expecting uint8.");

    TensorInfo zp_info{};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(zp_tensor, zp_info));
    SafeInt<int64_t> safe_total_elements = std::accumulate(zp_info.shape.begin(),
                                                           zp_info.shape.end(),
                                                           SafeInt<int64_t>{1},
                                                           std::multiplies<>());
    const int64_t total_elements = static_cast<int64_t>(safe_total_elements);
    RETURN_IF_NOT((total_elements * num_zp_per_uint8) == total_blocks,
                  ("Unexpected input zero_points size, expecting " +
                   std::to_string(total_blocks / num_zp_per_uint8))
                      .c_str());

    // QNN GPU expects symmetric quantization.
    if (is_gpu_backend) {
      RETURN_IF_NOT(utils::AreZeroPointsSymmetricConstant(qnn_model_wrapper, zp_tensor.name, bits),
                    ("Unsupported input zero_points value, expecting symmetric zero_points for bits=" + std::to_string(bits)).c_str());
    }
  }

  RETURN_IF((inputs.size() > 4 && inputs[4].Exists()) || (inputs.size() > 5 && inputs[5].Exists()),
            "Unsupported inputs g_idx or bias.");

  // Validate Process
  std::vector<std::string> input_names;
  RETURN_IF_ERROR(ProcessInputs(qnn_model_wrapper, node_unit, logger, input_names, true));
  RETURN_IF_ERROR(ProcessAttributesAndOutputs(qnn_model_wrapper, node_unit, std::move(input_names), logger, true));

  return Ort::Status();
}

Ort::Status MatMulNBitsOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                                const OrtNodeUnit& node_unit,
                                                const Ort::Logger& logger,
                                                std::vector<std::string>& input_names,
                                                bool do_op_validation) const {
  bool is_htp_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());

  // Extract Parameters
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t K = node_helper.Get("K", static_cast<int64_t>(0));
  const int64_t N = node_helper.Get("N", static_cast<int64_t>(0));
  const int64_t bits = node_helper.Get("bits", static_cast<int64_t>(0));
  const int64_t block_size = node_helper.Get("block_size", static_cast<int64_t>(0));

  // Should already be guaranteed in IsOpSupported.
  assert(K > 0 && N > 0 && bits > 0 && block_size > 0);

  const int64_t num_elements_per_uint8 = kByteBits / bits;
  const int64_t num_zp_per_uint8 = K / block_size < num_elements_per_uint8 ? K / block_size : num_elements_per_uint8;

  // Prepare essential parameters
  const int64_t total_blocks = (N * K) / block_size;
  const auto& inputs = node_unit.Inputs();
  bool is_act_16bitquant = false;

  // 1. Add input A.
  {
    // 1.1 Create QNN wrapper.
    RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

    if (is_htp_backend) {
      // 1.2 Add pre-Reshape to unsqueeze shape to 4D for HTP backend.
      TensorInfo input_info = {};
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input_info));
      is_act_16bitquant = utils::IsQuant16bit(input_info.qnn_data_type);

      // Input A having 3D shape is guaranteed in IsOpSupported.
      assert(input_info.shape.size() == 3);
      std::vector<uint32_t> reshape_output_shape = {input_info.shape[0], 1, input_info.shape[1], input_info.shape[2]};

      const std::string reshape_output_name = utils::UniqueNameGenerator().New(input_names[0], "_reshape_4d");
      QnnTensorWrapper reshape_output_tensor_wrapper(reshape_output_name,
                                                     QNN_TENSOR_TYPE_NATIVE,
                                                     input_info.qnn_data_type,
                                                     input_info.quant_param.Copy(),
                                                     std::vector<uint32_t>(reshape_output_shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output_tensor_wrapper)),
                    "Failed to add pre-Reshape output tensor.");

      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_reshape_4d"),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    QNN_OP_RESHAPE,
                                                    {input_names[0]},
                                                    {reshape_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add pre-Reshape node.");

      // Reroute to pre-Reshape output.
      input_names[0] = reshape_output_name;

      if (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_32) {
        // 1.3 (Workaround) Add pre-Cast to cast float32 to float16 to pass HTP validation.
        // TODO: Remove the additional Cast once HTP supports float32.
        const std::string cast_output_name = utils::UniqueNameGenerator().New(input_names[0], "_cast_fp16");
        RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::UniqueNameGenerator().New(node_unit, "_cast_fp16"),
                                                      reshape_output_name,
                                                      cast_output_name,
                                                      QNN_TENSOR_TYPE_NATIVE,
                                                      QNN_DATATYPE_FLOAT_16,
                                                      input_info.quant_param.Copy(),
                                                      std::vector<uint32_t>(reshape_output_shape),
                                                      do_op_validation));

        // Reroute to pre-Cast output.
        input_names[0] = cast_output_name;
      }
    }
  }

  // 2. Add input B and corresponding quantization parameters.
  {
    const auto& weight_tensor = inputs[1];
    const auto& scales_tensor = inputs[2];

    const auto& weight_tensor_name = weight_tensor.name;
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(weight_tensor_name)) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + weight_tensor_name).c_str());
    } else {
      // 2.1 Block-quantized data.
      std::vector<uint8_t> quant_data, weight_data;
      Qnn_TensorType_t weight_tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
      const OrtValueInfo* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
      std::vector<uint32_t> weight_shape = {};
      Qnn_DataType_t weight_datatype = QNN_DATATYPE_UNDEFINED;
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(weight_tensor_proto, quant_data, false));
      RETURN_IF_ERROR(utils::TransformUnsignedToSignedFixedPoint(quant_data, bits));

      // 2.2 Block-quantized scales.
      std::vector<uint8_t> per_block_uint8_scale;
      const OrtValueInfo* scale_tensor_proto = qnn_model_wrapper.GetConstantTensor(scales_tensor.name);
      RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(scale_tensor_proto, per_block_uint8_scale));

      const size_t elem_byte_size = utils::GetElementSizeByType(scales_tensor.type);
      RETURN_IF_NOT(per_block_uint8_scale.size() == (total_blocks * elem_byte_size), "Unexpected scales data size.");

      std::vector<float> per_block_float_scale;
      if (scales_tensor.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float* per_block_float_scale_ptr = reinterpret_cast<float*>(per_block_uint8_scale.data());
        per_block_float_scale = std::vector<float>(per_block_float_scale_ptr, per_block_float_scale_ptr + total_blocks);
      } else {
        Ort::Float16_t* per_block_fp16_scale_ptr = reinterpret_cast<Ort::Float16_t*>(per_block_uint8_scale.data());
        per_block_float_scale.reserve(total_blocks);
        for (int64_t idx = 0; idx < total_blocks; ++idx) {
          per_block_float_scale.emplace_back(static_cast<float>(per_block_fp16_scale_ptr[idx]));
        }
      }

      QnnQuantParamsWrapper quantize_param;

      if (IsGpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
        // 2.3 Block-quantized offsets
        // QNN GPU only supports symmetric quantization. Since block-quantized data is transformed to signed fixed
        // point 4 below, the value should be 0.
        std::vector<int32_t> per_block_int32_offset(total_blocks, 0);

        // 2.4 Create quant params.
        const std::vector<uint32_t> block_sizes = {1, gsl::narrow_cast<uint32_t>(block_size)};
        quantize_param = QnnQuantParamsWrapper::Block(per_block_float_scale,
                                                      per_block_int32_offset,
                                                      block_sizes);

        weight_shape = {gsl::narrow_cast<uint32_t>(N), gsl::narrow_cast<uint32_t>(K)};
        weight_datatype = QNN_DATATYPE_SFIXED_POINT_4;
        weight_data = std::move(quant_data);
      } else {
        // 2.3 Unpack data to one byte per element,
        // and transpose [N,K] -> [K,N] (= [1,1,K,N] in HWCN Conv2D layout).
        std::vector<uint8_t> unpacked_quant_data;
        UnpackDataToDatatype<uint8_t>(quant_data, bits, num_elements_per_uint8, unpacked_quant_data);

        std::vector<uint8_t> transposed_unpacked_quant_data;
        RETURN_IF_ERROR(utils::TwoDimensionTranspose<uint8_t>(
            unpacked_quant_data,
            {gsl::narrow_cast<uint32_t>(N), gsl::narrow_cast<uint32_t>(K)},
            transposed_unpacked_quant_data,
            logger,
            do_op_validation));

        weight_shape = {1u, 1u, gsl::narrow_cast<uint32_t>(K), gsl::narrow_cast<uint32_t>(N)};
        weight_datatype = QNN_DATATYPE_SFIXED_POINT_8;
        weight_data = std::move(transposed_unpacked_quant_data);

        // 2.4 Compute quant params: try LPBQ (BLOCKWISE_EXPANSION) first, otherwise fall back to BwFloatBlock.
        bool used_lpbq = false;
        const bool zp_is_symmetric = !(inputs.size() > 3 && inputs[3].Exists()) ||
                                     utils::AreZeroPointsSymmetricConstant(qnn_model_wrapper, inputs[3].name, bits);

        if (bits == 4 && is_act_16bitquant && zp_is_symmetric) {
          // Build a synthetic OrtNodeUnitIODef to drive QnnQuantParamsWrapper::Init's
          // block-quant branch, which handles the BQ→LPBQ conversion.
          //
          // The weight tensor's quant_param does not carry block-quantization info in
          // the standard ONNX format, so we construct a synthetic def with:
          //   - scale  = the real scale initializer (provides the BQ scales)
          //   - zero_point = nullptr  ← ZP symmetry has already been verified above via
          //                             AreZeroPointsSymmetricConstant, so re-checking is redundant
          //   - axis   = std::nullopt ← uses Init's DEFAULT_QDQ_AXIS=1, which matches
          //                             the MatMulNBits scale tensor layout [N, K/block_size]
          //   - block_size = block_size
          //   - type   = INT4           4-bit weights
          //
          // Only quant_param is consumed from the result; the rest of synthetic_weight_info
          // is discarded.
          OrtNodeUnitIODef::QuantParam synthetic_qp;
          synthetic_qp.scale = scale_tensor_proto;
          synthetic_qp.zero_point = nullptr;
          synthetic_qp.axis = std::nullopt;
          synthetic_qp.block_size = block_size;

          OrtNodeUnitIODef synthetic_weight_def = weight_tensor;
          synthetic_weight_def.quant_param = synthetic_qp;
          synthetic_weight_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4;

          TensorInfo synthetic_weight_info = {};
          Ort::Status lpbq_status = qnn_model_wrapper.GetTensorInfo(synthetic_weight_def, synthetic_weight_info);
          if (!lpbq_status.IsOK()) {
            ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("LPBQ conversion failed: " + lpbq_status.GetErrorMessage()).c_str());
          } else if (synthetic_weight_info.quant_param.IsLPBQ()) {
            //   Update quant params for the shape transformations:
            //   [N,K] -> transpose -> [K,N]: axis 0 (N) -> axis 1 (N)
            //   [K,N] -> unsqueeze -> [1,1,K,N]: axis 1 (N) -> axis 3 (N)
            quantize_param = synthetic_weight_info.quant_param.Copy();
            RETURN_IF_ERROR(quantize_param.HandleTranspose<uint32_t>(std::vector<uint32_t>({1u, 0u})));
            const std::vector<uint32_t> kn_shape = {gsl::narrow_cast<uint32_t>(K), gsl::narrow_cast<uint32_t>(N)};
            RETURN_IF_ERROR(quantize_param.HandleUnsqueeze<uint32_t>(kn_shape, weight_shape));
            used_lpbq = true;
            ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("MatMulNBits weight encoding: LPBQ (BLOCKWISE_EXPANSION) for " + weight_tensor_name).c_str());
          }
        }

        bool used_bw_block_mapped = false;
#if defined(QNN_SDK_VERSION_MINOR) && (QNN_SDK_VERSION_MAJOR > 2 || (QNN_SDK_VERSION_MAJOR == 2 && QNN_SDK_VERSION_MINOR >= 51))
        // 2-bit Standard Symmetric BW_BLOCK_MAPPED: keeps int16 activations natively (no DQ needed).
        // Gated to SDK >= 2.51: the native W2A16 HTP kernel is not available until 2.51.
        if (bits == 2 && is_act_16bitquant && zp_is_symmetric) {
          const std::vector<uint32_t> block_sizes = {1, 1, gsl::narrow_cast<uint32_t>(block_size), 1};
          const std::vector<int32_t> per_block_int32_offset(total_blocks, 0);
          quantize_param = QnnQuantParamsWrapper::BwBlockMapped(per_block_float_scale,
                                                                per_block_int32_offset,
                                                                gsl::narrow_cast<uint32_t>(bits),
                                                                block_sizes,
                                                                QNN_QUANTIZATION_ENCODING_MAPPING_STANDARD_SYMMETRIC);
          used_bw_block_mapped = true;
          ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("MatMulNBits weight encoding: BW_BLOCK_MAPPED (STANDARD_SYMMETRIC) for " + weight_tensor_name).c_str());
        }
#endif  // QNN_SDK_VERSION_MINOR >= 51

        if (!used_lpbq && !used_bw_block_mapped) {
          // BwFloatBlock (float block-quantized) path.
          const char* reason = !is_act_16bitquant ? "activation not 16-bit quantized"
                               : bits != 4        ? "bits != 4 (LPBQ only supports INT4)"
                               : !zp_is_symmetric ? "zero-points not symmetric"
                                                  : "LPBQ conversion failed (enable_block_quant_weight_optimization=0)";
          ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("MatMulNBits weight encoding: BW_FLOAT_BLOCK for " + weight_tensor_name + " [LPBQ skipped: " + reason + "]").c_str());
          // 2.5 Block-quantized offsets.
          std::vector<float> per_block_float_zp;
          if (inputs.size() > 3 && inputs[3].Exists()) {
            // Unpack block-quantized offsets to float each.
            std::vector<uint8_t> per_block_uint8_zp;
            const OrtValueInfo* zp_tensor_proto = qnn_model_wrapper.GetConstantTensor(inputs[3].name);
            RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(zp_tensor_proto, per_block_uint8_zp));
            UnpackDataToDatatype<float>(per_block_uint8_zp, bits, num_zp_per_uint8, per_block_float_zp);

            // Shift offsets for transformation from unsigned to signed fixed point and negate the values to align with
            // QNN definition for offsets.
            const float offset_shift = static_cast<float>(1 << (bits - 1));
            for (size_t idx = 0; idx < per_block_float_zp.size(); ++idx) {
              per_block_float_zp[idx] = -(per_block_float_zp[idx] - offset_shift);
            }
          } else {
            // Default to 0.
            per_block_float_zp.assign(total_blocks, 0);
          }

          // Note that unlike weights requiring transpose, scales/offsets are expected in original ONNX shape.
          const std::vector<uint32_t> block_sizes = {1, 1, gsl::narrow_cast<uint32_t>(block_size), 1};
          quantize_param = QnnQuantParamsWrapper::BwFloatBlock(per_block_float_scale,
                                                               per_block_float_zp,
                                                               gsl::narrow_cast<uint32_t>(bits),
                                                               block_sizes);
          if (is_act_16bitquant) {
            // 2.6 Add Dequantize to UINT16/INT16 → FP16.
            const std::string fp16_act_name = utils::UniqueNameGenerator().New(input_names[0], "_dq_fp16");
            RETURN_IF_ERROR(bq::AddInt16ToFp16DequantForActivation(qnn_model_wrapper,
                                                                   input_names[0],
                                                                   fp16_act_name,
                                                                   do_op_validation,
                                                                   "MatMulNBits"));
            input_names[0] = fp16_act_name;
          }
        }
      }
      // 2.7 Create and register the weight tensor wrapper.
      QnnTensorWrapper weight_tensor_wrapper(weight_tensor_name,
                                             weight_tensor_type,
                                             weight_datatype,
                                             std::move(quantize_param),
                                             std::move(weight_shape),
                                             std::move(weight_data));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor_wrapper)),
                    "Failed to add weight tensor.");
    }
    input_names.push_back(weight_tensor_name);
  }

  return Ort::Status();
}

Ort::Status MatMulNBitsOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                              const OrtNodeUnit& node_unit,
                                                              std::vector<std::string>&& input_names,
                                                              const Ort::Logger& logger,
                                                              bool do_op_validation) const {
  // Extract Parameters
  OrtNodeAttrHelper node_helper(node_unit);
  const int64_t N = node_helper.Get("N", static_cast<int64_t>(0));

  const OrtNodeUnitIODef& output_tensor = node_unit.Outputs()[0];
  TensorInfo output_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output_tensor, output_info));

  if (IsGpuBackend(qnn_model_wrapper.GetQnnBackendType())) {
    // 1. Add Output for Reshape
    const std::string& output_tensor_name = output_tensor.name;
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(output_tensor_name)) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + output_tensor_name).c_str());
    } else {
      QnnTensorWrapper output_tensor_wrapper;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_tensor, output_tensor_wrapper));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                    "Failed to add post-Reshape output tensor");
    }

    // 2. Add Output for Pre Reshape(FullyConnected)
    const std::string pre_reshape_name = utils::UniqueNameGenerator().New(output_tensor_name, "_pre_reshape");
    std::vector<uint32_t> pre_reshape_shape(2);
    pre_reshape_shape[0] = gsl::narrow_cast<uint32_t>(std::accumulate(output_info.shape.begin(),
                                                                      output_info.shape.end(),
                                                                      SafeInt<uint32_t>{1},
                                                                      std::multiplies<>()) /
                                                      N);
    pre_reshape_shape[1] = gsl::narrow_cast<uint32_t>(N);
    QnnTensorWrapper output_tensor_wrapper(pre_reshape_name,
                                           QNN_TENSOR_TYPE_NATIVE,
                                           output_info.qnn_data_type,
                                           output_info.quant_param.Copy(),
                                           std::vector<uint32_t>(pre_reshape_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)),
                  "Failed to add FC output tensor.");

    // 3. Add FullyConnected Op
    const std::string fully_connected_node_name = utils::UniqueNameGenerator().New(node_unit, QNN_OP_FULLY_CONNECTED);
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(fully_connected_node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_FULLY_CONNECTED,
                                                  std::move(input_names),
                                                  {pre_reshape_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add FC node.");

    // 4. Add Reshape Op
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_tensor_name);
    RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(pre_reshape_name,
                                                     output_tensor_name,
                                                     pre_reshape_shape,
                                                     output_info.shape,
                                                     output_info.qnn_data_type,
                                                     output_info.quant_param,
                                                     do_op_validation,
                                                     false,
                                                     is_graph_output));
  } else {
    // 1. Add MatMul as Conv2d with default stride/pad amount.
    std::vector<std::string> param_tensor_names;

    std::vector<uint32_t> stride = {1, 1};
    QnnParamWrapper stride_param_wrapper(node_unit.Index(),
                                         node_unit.Name(),
                                         QNN_OP_CONV_2D_PARAM_STRIDE,
                                         {2},
                                         std::move(stride));
    param_tensor_names.push_back(stride_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(stride_param_wrapper));

    std::vector<uint32_t> pad_amount = {0, 0, 0, 0};
    QnnParamWrapper pad_amount_param_wrapper(node_unit.Index(),
                                             node_unit.Name(),
                                             QNN_OP_CONV_2D_PARAM_PAD_AMOUNT,
                                             {2, 2},
                                             std::move(pad_amount));
    param_tensor_names.push_back(pad_amount_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_param_wrapper));

    std::vector<uint32_t> dilation = {1, 1};
    QnnParamWrapper dilation_param_wrapper(node_unit.Index(),
                                           node_unit.Name(),
                                           QNN_OP_CONV_2D_PARAM_DILATION,
                                           {2},
                                           std::move(dilation));
    param_tensor_names.push_back(dilation_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(dilation_param_wrapper));

    RETURN_IF_ERROR(AddQnnScalar<uint32_t>(qnn_model_wrapper,
                                           node_unit.Index(),
                                           node_unit.Name(),
                                           1,
                                           QNN_OP_CONV_2D_PARAM_GROUP,
                                           param_tensor_names));

    // Input originally having 3D shape is guaranteed in IsOpSupported.
    assert(output_info.shape.size() == 3);
    std::vector<uint32_t> conv2d_output_shape = {output_info.shape[0], 1, output_info.shape[1], output_info.shape[2]};

    // Detect LPBQ / BW_BLOCK_MAPPED from the registered weight tensor's quant encoding.
    // For LPBQ and BW_BLOCK_MAPPED, the Conv2D output uses the actual output data type (e.g., uint16/int16
    // for QDQ models). For BwFloatBlock, the Conv2D kernel always outputs FP16.
    const bool is_lpbq = qnn_model_wrapper.IsQnnTensorWrapperExist(input_names[1]) &&
                         qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]).GetQnnQuantParams().IsLPBQ();
    const bool is_bw_block_mapped = qnn_model_wrapper.IsQnnTensorWrapperExist(input_names[1]) &&
                                    qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]).GetQnnQuantParams().IsBwBlockMapped();
    const Qnn_DataType_t conv2d_output_dtype = (is_lpbq || is_bw_block_mapped)
                                                    ? output_info.qnn_data_type
                                                    : QNN_DATATYPE_FLOAT_16;

    const std::string conv2d_output_name = utils::UniqueNameGenerator().New(output_tensor.name, "_conv2d");
    QnnTensorWrapper conv2d_output_tensor_wrapper(conv2d_output_name,
                                                  QNN_TENSOR_TYPE_NATIVE,
                                                  conv2d_output_dtype,
                                                  output_info.quant_param.Copy(),
                                                  std::vector<uint32_t>(conv2d_output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(conv2d_output_tensor_wrapper)),
                  "Failed to add Conv2d output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_CONV_2D,
                                                  std::move(input_names),
                                                  {conv2d_output_name},
                                                  std::move(param_tensor_names),
                                                  do_op_validation),
                  "Failed to add Conv2d node.");

    std::string reshape_input_name = conv2d_output_name;
    if (output_info.qnn_data_type == QNN_DATATYPE_FLOAT_32) {
      // 2. Workaround: Add post-Cast to cast float16 back to float32 to pass HTP validation.
      // TODO: Remove the additional Cast once HTP supports float32.
      const std::string cast_output_name = utils::UniqueNameGenerator().New(output_tensor.name, "_cast_fp32");
      RETURN_IF_ERROR(qnn_model_wrapper.AddCastNode(utils::UniqueNameGenerator().New(node_unit, "_cast_fp32"),
                                                    conv2d_output_name,
                                                    cast_output_name,
                                                    QNN_TENSOR_TYPE_NATIVE,
                                                    QNN_DATATYPE_FLOAT_32,
                                                    output_info.quant_param.Copy(),
                                                    std::vector<uint32_t>(conv2d_output_shape),
                                                    do_op_validation));

      reshape_input_name = cast_output_name;
    } else if (utils::IsQuant16bit(output_info.qnn_data_type) && !is_lpbq && !is_bw_block_mapped) {
      // 2. Add Quantize to FP16 → UINT16/INT16.
      const std::string q_suffix = output_info.qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16 ? "_q_int16" : "_q_uint16";
      const std::string q_output_name = utils::UniqueNameGenerator().New(output_tensor.name, q_suffix);
      RETURN_IF_ERROR(bq::AddFp16ToInt16QuantizeOutput(qnn_model_wrapper,
                                                       conv2d_output_name,
                                                       q_output_name,
                                                       QNN_TENSOR_TYPE_NATIVE,
                                                       output_info.qnn_data_type,
                                                       output_info.quant_param.Copy(),
                                                       std::vector<uint32_t>(conv2d_output_shape),
                                                       do_op_validation));

      reshape_input_name = q_output_name;
    }

    // 3. Add post-Reshape to squeeze shape back.
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_tensor.name);
    QnnTensorWrapper reshape_output_tensor_wrapper(output_tensor.name,
                                                   is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE,
                                                   output_info.qnn_data_type,
                                                   output_info.quant_param.Copy(),
                                                   std::vector<uint32_t>(output_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(reshape_output_tensor_wrapper)),
                  "Failed to add post-Reshape output tensor.");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, "_reshape_3d"),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_RESHAPE,
                                                  {reshape_input_name},
                                                  {output_tensor.name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add post-Reshape node.");
  }

  return Ort::Status();
}

void CreateMatMulNBitsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulNBitsOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
