// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <type_traits>

#include "model_test_builder.h"

#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

using GetQDQTestCaseFn = std::function<void(ModelTestBuilder& builder)>;

template <typename T>
std::string
AddQDQNodePair(ModelTestBuilder& builder, std::string qdq_name, std::string inp_name, float scale, T zp = T(), bool use_ms_domain = false) {
  builder.AddQuantizeLinearNode<T>(qdq_name + "_q", inp_name.c_str(), scale, zp, (qdq_name + "_q_out").c_str(), use_ms_domain);
  builder.AddDequantizeLinearNode<T>(qdq_name + "_dq", (qdq_name + "_q_out").c_str(), scale, zp, (qdq_name + "_dq_out").c_str(), use_ms_domain);
  return qdq_name + "_dq_out";
}

template <typename T>
std::string
AddQDQNodePairWithOutputAsGraphOutput(ModelTestBuilder& builder, std::string qdq_name, std::string inp_name, float scale, T zp = T(),
                                      bool use_ms_domain = false) {
  builder.AddQuantizeLinearNode<T>(qdq_name + "_q", inp_name.c_str(), scale, zp, (qdq_name + "_q_out").c_str(), use_ms_domain);
  builder.AddDequantizeLinearNode<T>(qdq_name + "_dq", (qdq_name + "_q_out").c_str(), scale, zp, (qdq_name + "_dq_out").c_str(), use_ms_domain);
  builder.MakeOutput((qdq_name + "_dq_out").c_str());
  return qdq_name + "_dq_out";
}

// Overload for per-channel quantization with vector scales and zero points
template <typename T>
std::string
AddQDQNodePair(ModelTestBuilder& builder, std::string qdq_name, std::string inp_name,
               const std::vector<float>& scales, const std::vector<T>& zps,
               const std::vector<ONNX_NAMESPACE::AttributeProto>& q_attrs = {},
               const std::vector<ONNX_NAMESPACE::AttributeProto>& dq_attrs = {},
               bool use_ms_domain = false) {
  builder.AddQuantizeLinearNode<T>(qdq_name + "_q", inp_name.c_str(), scales, zps, (qdq_name + "_q_out").c_str(), q_attrs, use_ms_domain);
  builder.AddDequantizeLinearNode<T>(qdq_name + "_dq", (qdq_name + "_q_out").c_str(), scales, zps, (qdq_name + "_dq_out").c_str(), dq_attrs, use_ms_domain);
  return qdq_name + "_dq_out";
}

GetQDQTestCaseFn BuildQDQReshapeTestCase(const std::vector<int64_t>& input_shape,
                                         const std::vector<int64_t>& reshape_shape);

// Below utility functions are copied from ORT Core with few simpliciations.
// Refer to onnxruntime/core/mlas/lib/q4_dq.cpp for original implementations.

template <int qbits, bool signed_quant>
struct BitsTraits {
  static_assert(qbits <= 8, "Only BitsTraits are for small number of bits!");

  static constexpr int kBits = qbits;
  static constexpr int kMax = signed_quant ? (1 << (qbits - 1)) - 1 : (1 << qbits) - 1;
  static constexpr int kMid = signed_quant ? 0 : (1 << (qbits - 1));
  static constexpr int kMin = signed_quant ? -(1 << (qbits - 1)) : 0;
  static constexpr float kMaxFp = static_cast<float>(kMax);
  static constexpr float kMinFp = static_cast<float>(kMin);
  static constexpr float fullRange = kMaxFp - kMinFp;
  static constexpr float halfRange = static_cast<float>(kMid - kMin);

  // number of qbit elements to pack into whole bytes
  static constexpr int kPackSize = (qbits == 8) ? 1 : ((qbits == 4) ? 2 : ((qbits == 2) ? 4 : 0));
  static_assert(kPackSize != 0, "Packing to whole bytes not supported for this qbits!");
};

/**
 * @brief Rectify min/max from a set of weights, and convert to scale and zero point
 *        for quantization.
 * @tparam ScaleT        type of scale, usually floating point of various bits
 * @tparam qbits         number of int bits used for zero point value
 * @tparam signed_quant  output quantized type is signed
 * @param[in]   min
 * @param[in]   max
 * @param[out]  scale
 * @param[out]  zp
 */
template <typename ScaleT, int qbits, bool signed_quant>
void range2scalezp(float min, float max, ScaleT& scale, uint8_t& zp) {
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);

  float scale_f = (max - min) / BitsTraits<qbits, signed_quant>::fullRange;

  float zero_point_fp = min;
  if (scale_f != 0.0f) {
    zero_point_fp = BitsTraits<qbits, signed_quant>::kMinFp - min / scale_f;
  }

  if (zero_point_fp < BitsTraits<qbits, signed_quant>::kMinFp) {
    zp = static_cast<uint8_t>(BitsTraits<qbits, signed_quant>::kMin);
  } else if (zero_point_fp > BitsTraits<qbits, signed_quant>::kMaxFp) {
    zp = static_cast<uint8_t>(BitsTraits<qbits, signed_quant>::kMax);
  } else {
    zp = (uint8_t)std::roundf(zero_point_fp);
  }
  scale = ScaleT(scale_f);
}

/**
 * @brief Rectify min/max from a set of symmetric weights, and convert
 *        to scale for quantization.
 */
template <typename ScaleT, int qbits, bool signed_quant>
void range2scale(float min, float max, ScaleT& scale) {
  max = std::fabs(max) > std::fabs(min) ? max : min;

  // Original implementation allows negative scale to better fit larger half FP space. However, since HTP backend does
  // not support negative scale, simplify to common formulation.
  scale = ScaleT(std::fabs(max) * 2 / BitsTraits<qbits, signed_quant>::fullRange);
};

/**
 * @brief Blockwise quantization for test purposes. This is a simplified version of MlasQuantizeBlockwise
 *        that doesn't require internal MLAS APIs.
 *
 * @tparam T            Element type (float)
 * @tparam qbits        Number of quantization bits (4)
 * @param dst           Output quantized data (column major, packed)
 * @param scales        Output scales (column major)
 * @param zero_points   Output zero points (column major, packed), can be nullptr for symmetric quantization
 * @param src           Input float data (row major)
 * @param block_size    Block size for quantization
 * @param columnwise    True for column-wise quantization
 * @param rows          Number of rows
 * @param columns       Number of columns
 * @param leading_dimension Leading dimension of source matrix
 */
template <typename T, int qbits>
inline void QuantizeBlockwise(uint8_t* dst,
                              T* scales,
                              uint8_t* zero_points,
                              const T* src,
                              int block_size,
                              bool columnwise,
                              int rows,
                              int columns,
                              int leading_dimension) {
  static_assert(qbits == 2 || qbits == 4 || qbits == 8, "Only 2-bit, 4-bit, or 8-bit quantization is supported");
  static_assert(std::is_same<T, float>::value, "Only float type is supported");

  if (!columnwise) {
    throw std::runtime_error("Only column-wise quantization is supported in test utilities");
  }

  constexpr int kPackSize = BitsTraits<qbits, false>::kPackSize;
  const int k_blocks = (rows + block_size - 1) / block_size;
  const int blob_size = (block_size + kPackSize - 1) / kPackSize;
  const bool symmetric = (zero_points == nullptr);

  // Process each column
  for (int n = 0; n < columns; n++) {
    const T* src_col = src + n;

    // Process each block in the column
    for (int k_blk = 0; k_blk < k_blocks; k_blk++) {
      const int row_start = k_blk * block_size;
      const int row_end = std::min(row_start + block_size, rows);
      const int block_len = row_end - row_start;

      // Find min/max in the block
      float min_val = std::numeric_limits<float>::max();
      float max_val = std::numeric_limits<float>::lowest();

      for (int i = row_start; i < row_end; i++) {
        const float val = static_cast<float>(src_col[i * leading_dimension]);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
      }

      // Calculate scale and zero point
      const int scale_idx = n * k_blocks + k_blk;
      uint8_t zp = BitsTraits<qbits, false>::kMid;  // Default zero point for symmetric

      if (symmetric) {
        range2scale<T, qbits, false>(min_val, max_val, scales[scale_idx]);
      } else {
        range2scalezp<T, qbits, false>(min_val, max_val, scales[scale_idx], zp);
      }

      // Quantize and pack the block
      const float scale = static_cast<float>(scales[scale_idx]);
      const float reciprocal_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

      // Calculate destination offset (column major, packed)
      uint8_t* dst_block = dst + n * k_blocks * blob_size + k_blk * blob_size;

      // Quantize values in pairs and pack
      for (int i = 0; i < block_len; i += kPackSize) {
        uint8_t q_vals[kPackSize];
        std::fill_n(q_vals, kPackSize, 0);

        for (int pack_idx = 0; pack_idx < kPackSize && i + pack_idx < block_len; ++pack_idx) {
          const int src_idx = (row_start + i + pack_idx) * leading_dimension;
          const float val = static_cast<float>(src_col[src_idx]);
          q_vals[pack_idx] = (uint8_t)std::clamp(std::roundf(val * reciprocal_scale + zp),
                                                 0.0f,
                                                 BitsTraits<qbits, false>::kMaxFp);
        }

        // Pack {kPackSize} {qbits}-bit values into one byte
        if constexpr (qbits == 8) {
          dst_block[i] = q_vals[0];
        } else if constexpr (qbits == 4) {
          dst_block[i / 2] = (q_vals[0] & 0xf) | (q_vals[1] << 4);
        } else if constexpr (qbits == 2) {
          dst_block[i / 4] = (q_vals[0] & 0x3) | (q_vals[1] << 2) | (q_vals[2] << 4) | (q_vals[3] << 6);
        } else {
          throw std::runtime_error("Unsupported qbits");
        }
      }

      // Store zero point if asymmetric
      if (!symmetric) {
        const int zp_blob_size = (k_blocks + kPackSize - 1) / kPackSize;
        uint8_t* zp_block = zero_points + n * zp_blob_size;

        if constexpr (qbits == 8) {
          zp_block[k_blk] = zp;
        } else if constexpr (qbits == 4) {
          if (k_blk % 2 == 0) {
            zp_block[k_blk / 2] = (zp & 0xf);
          } else {
            zp_block[k_blk / 2] |= (zp << 4);
          }
        } else if constexpr (qbits == 2) {
          if (k_blk % 4 == 0) {
            zp_block[k_blk / 4] = (zp & 0x3);
          } else if (k_blk % 4 == 1) {
            zp_block[k_blk / 4] |= (zp << 2);
          } else if (k_blk % 4 == 2) {
            zp_block[k_blk / 4] |= (zp << 4);
          } else {
            zp_block[k_blk / 4] |= (zp << 6);
          }
        } else {
          throw std::runtime_error("Unsupported qbits");
        }
      }
    }
  }
}

/**
 * @brief Blockwise dequantization for test purposes. This is a simplified version of MlasDequantizeBlockwise
 *        that doesn't require internal MLAS APIs.
 *
 * @tparam T            Element type (float)
 * @tparam qbits        Number of quantization bits (4)
 * @param dst           Output dequantized data (column major)
 * @param src           Input quantized data (column major, packed)
 * @param scales        Input scales (column major)
 * @param zero_points   Input zero points (column major, packed), can be nullptr for symmetric quantization
 * @param block_size    Block size for quantization
 * @param columnwise    True for column-wise quantization
 * @param rows          Number of rows
 * @param columns       Number of columns
 */
template <typename T, int qbits>
inline void DequantizeBlockwise(T* dst,
                                const uint8_t* src,
                                const T* scales,
                                const uint8_t* zero_points,
                                int block_size,
                                bool columnwise,
                                int rows,
                                int columns) {
  static_assert(qbits == 2 || qbits == 4 || qbits == 8, "Only 2-bit, 4-bit, or 8-bit quantization is supported");
  static_assert(std::is_same<T, float>::value, "Only float type is supported");

  if (!columnwise) {
    throw std::runtime_error("Only column-wise dequantization is supported in test utilities");
  }

  constexpr int kPackSize = BitsTraits<qbits, false>::kPackSize;
  const int k_blocks = (rows + block_size - 1) / block_size;
  const int blob_size = (block_size + kPackSize - 1) / kPackSize;
  const bool symmetric = (zero_points == nullptr);

  // Process each column
  for (int n = 0; n < columns; n++) {
    T* dst_col = dst + n * rows;

    // Process each block in the column
    for (int k_blk = 0; k_blk < k_blocks; k_blk++) {
      const int row_start = k_blk * block_size;
      const int row_end = std::min(row_start + block_size, rows);
      const int block_len = row_end - row_start;

      // Get scale and zero point
      const int scale_idx = n * k_blocks + k_blk;
      const float scale = static_cast<float>(scales[scale_idx]);

      uint8_t zp = BitsTraits<qbits, false>::kMid;  // Default zero point for symmetric
      if (!symmetric) {
        const int zp_blob_size = (k_blocks + kPackSize - 1) / kPackSize;
        const uint8_t* zp_block = zero_points + n * zp_blob_size;

        if constexpr (qbits == 8) {
          zp = zp_block[k_blk];
        } else if constexpr (qbits == 4) {
          if (k_blk % 2 == 0) {
            zp = zp_block[k_blk / 2] & 0xf;
          } else {
            zp = (zp_block[k_blk / 2] >> 4) & 0xf;
          }
        } else if constexpr (qbits == 2) {
          if (k_blk % 4 == 0) {
            zp = zp_block[k_blk / 4] & 0x3;
          } else if (k_blk % 4 == 1) {
            zp = (zp_block[k_blk / 4] >> 2) & 0x3;
          } else if (k_blk % 4 == 2) {
            zp = (zp_block[k_blk / 4] >> 4) & 0x3;
          } else {
            zp = (zp_block[k_blk / 4] >> 6) & 0x3;
          }
        } else {
          throw std::runtime_error("Unsupported qbits");
        }
      }

      // Calculate source offset (column major, packed)
      const uint8_t* src_block = src + n * k_blocks * blob_size + k_blk * blob_size;

      // Dequantize values
      for (int i = 0; i < block_len; i++) {
        const uint8_t packed = src_block[i / kPackSize];
        uint8_t q_val;

        if constexpr (qbits == 8) {
          q_val = packed;
        } else if constexpr (qbits == 4) {
          if (i % 2 == 0) {
            q_val = packed & 0xf;
          } else {
            q_val = (packed >> 4) & 0xf;
          }
        } else if constexpr (qbits == 2) {
          if (i % 4 == 0) {
            q_val = packed & 0x3;
          } else if (i % 4 == 1) {
            q_val = (packed >> 2) & 0x3;
          } else if (i % 4 == 2) {
            q_val = (packed >> 4) & 0x3;
          } else {
            q_val = (packed >> 6) & 0x3;
          }
        } else {
          throw std::runtime_error("Unsupported qbits");
        }

        float dequant_val = (q_val - zp) * scale;
        dst_col[row_start + i] = static_cast<T>(dequant_val);
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
