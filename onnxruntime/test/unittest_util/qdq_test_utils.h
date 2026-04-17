// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <string>
#include <type_traits>

#include "graph_transform_test_builder.h"

#include "core/framework/int4.h"
#include "core/common/span_utils.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/session/inference_session.h"

#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

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

std::vector<std::string> GetNodeOpTypesInTopologicalOrder(const Graph& graph, bool include_domain = false);

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
inline void QuantizeBlockwise(
    uint8_t* dst,
    T* scales,
    uint8_t* zero_points,
    const T* src,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int leading_dimension) {
  static_assert(qbits == 4, "Only 4-bit quantization is supported");
  static_assert(std::is_same<T, float>::value, "Only float type is supported");

  if (!columnwise) {
    throw std::runtime_error("Only column-wise quantization is supported in test utilities");
  }

  const int k_blocks = (rows + block_size - 1) / block_size;
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
      uint8_t zp = 8;  // Default zero point for symmetric

      if (symmetric) {
        // Symmetric quantization: map to [-8, 7]
        // Negative scale follows MLAS convention: q in [-8,7] maps to [abs_max, -(7/8)*abs_max]
        const float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        scales[scale_idx] = static_cast<T>(-abs_max / 8.0f);
      } else {
        // Asymmetric quantization: map to [0, 15]
        min_val = std::min(min_val, 0.0f);
        max_val = std::max(max_val, 0.0f);

        const float scale = (max_val - min_val) / 15.0f;
        scales[scale_idx] = static_cast<T>(scale);

        if (scale != 0.0f) {
          const float zp_fp = -min_val / scale;
          zp = static_cast<uint8_t>(std::clamp(std::round(zp_fp), 0.0f, 15.0f));
        } else {
          zp = 0;
        }
      }

      // Quantize and pack the block
      const float scale = static_cast<float>(scales[scale_idx]);
      const float reciprocal_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

      // Calculate destination offset (column major, packed)
      const int blob_size = (block_size + 1) / 2;  // 2 values per byte for 4-bit
      uint8_t* dst_block = dst + n * k_blocks * blob_size + k_blk * blob_size;

      // Quantize values in pairs and pack
      for (int i = 0; i < block_len; i += 2) {
        const int src_idx0 = (row_start + i) * leading_dimension;
        const float val0 = static_cast<float>(src_col[src_idx0]);

        int8_t q0;
        if (symmetric) {
          const float q_fp = std::round(val0 * reciprocal_scale);
          q0 = static_cast<int8_t>(std::clamp(q_fp, -8.0f, 7.0f));
          q0 = q0 + 8;  // Convert to [0, 15] for packing
        } else {
          const float q_fp = std::round(val0 * reciprocal_scale + zp);
          q0 = static_cast<int8_t>(std::clamp(q_fp, 0.0f, 15.0f));
        }

        int8_t q1 = symmetric ? 8 : zp;  // Default for padding
        if (i + 1 < block_len) {
          const int src_idx1 = (row_start + i + 1) * leading_dimension;
          const float val1 = static_cast<float>(src_col[src_idx1]);

          if (symmetric) {
            const float q_fp = std::round(val1 * reciprocal_scale);
            q1 = static_cast<int8_t>(std::clamp(q_fp, -8.0f, 7.0f));
            q1 = q1 + 8;  // Convert to [0, 15] for packing
          } else {
            const float q_fp = std::round(val1 * reciprocal_scale + zp);
            q1 = static_cast<int8_t>(std::clamp(q_fp, 0.0f, 15.0f));
          }
        }

        // Pack two 4-bit values into one byte (low nibble, high nibble)
        dst_block[i / 2] = (static_cast<uint8_t>(q0) & 0x0F) | ((static_cast<uint8_t>(q1) & 0x0F) << 4);
      }

      // Store zero point if asymmetric
      if (!symmetric) {
        const int zp_blob_size = (k_blocks + 1) / 2;
        uint8_t* zp_block = zero_points + n * zp_blob_size;

        if (k_blk % 2 == 0) {
          zp_block[k_blk / 2] = (zp & 0x0F);
        } else {
          zp_block[k_blk / 2] |= ((zp & 0x0F) << 4);
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
inline void DequantizeBlockwise(
    T* dst,
    const uint8_t* src,
    const T* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns) {
  static_assert(qbits == 4, "Only 4-bit quantization is supported");
  static_assert(std::is_same<T, float>::value, "Only float type is supported");

  if (!columnwise) {
    throw std::runtime_error("Only column-wise dequantization is supported in test utilities");
  }

  const int k_blocks = (rows + block_size - 1) / block_size;
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

      uint8_t zp = 8;  // Default zero point for symmetric
      if (!symmetric) {
        const int zp_blob_size = (k_blocks + 1) / 2;
        const uint8_t* zp_block = zero_points + n * zp_blob_size;

        if (k_blk % 2 == 0) {
          zp = zp_block[k_blk / 2] & 0x0F;
        } else {
          zp = (zp_block[k_blk / 2] >> 4) & 0x0F;
        }
      }

      // Calculate source offset (column major, packed)
      const int blob_size = (block_size + 1) / 2;  // 2 values per byte for 4-bit
      const uint8_t* src_block = src + n * k_blocks * blob_size + k_blk * blob_size;

      // Dequantize values
      for (int i = 0; i < block_len; i++) {
        const uint8_t packed = src_block[i / 2];
        uint8_t q_val;

        if (i % 2 == 0) {
          q_val = packed & 0x0F;  // Low nibble
        } else {
          q_val = (packed >> 4) & 0x0F;  // High nibble
        }

        float dequant_val;
        if (symmetric) {
          // Convert from [0, 15] back to [-8, 7]
          const int8_t signed_val = static_cast<int8_t>(q_val) - 8;
          dequant_val = signed_val * scale;
        } else {
          dequant_val = (static_cast<float>(q_val) - zp) * scale;
        }

        dst_col[row_start + i] = static_cast<T>(dequant_val);
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
