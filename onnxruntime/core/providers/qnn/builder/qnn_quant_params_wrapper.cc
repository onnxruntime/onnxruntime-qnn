// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <vector>

#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#define ALIGN_PTR_UP(ptr, align, type) \
  reinterpret_cast<type>((reinterpret_cast<std::uintptr_t>(ptr) + (align) - 1) & ~((align) - 1))

namespace onnxruntime {
namespace qnn {

QnnQuantParamsWrapper::QnnQuantParamsWrapper(const QnnQuantParamsWrapper& other)
    : params_(QNN_QUANTIZE_PARAMS_INIT) {
  size_t num_scaleoffsets = 0;
  if (other.IsLPBQ()) {
    num_scaleoffsets = other.per_channel_scales_size_;
  } else if (other.IsBlockQuantized()) {
    block_encoding_tensor_rank_ = other.block_encoding_tensor_rank_;
    num_scaleoffsets = other.num_blocks_;
  }
  Ort::Status status = Init(other.params_, num_scaleoffsets, block_encoding_tensor_rank_);
  assert(status.IsOK());  // Expect other QnnQuantParamsWrapper to always have a supported quantization encoding.
}

QnnQuantParamsWrapper& QnnQuantParamsWrapper::operator=(const QnnQuantParamsWrapper& other) {
  if (this != &other) {
    size_t num_scaleoffsets = 0;
    if (other.IsLPBQ()) {
      num_scaleoffsets = other.per_channel_scales_size_;
    } else if (other.IsBlockQuantized()) {
      block_encoding_tensor_rank_ = other.block_encoding_tensor_rank_;
      num_scaleoffsets = other.num_blocks_;
    }
    Ort::Status status = Init(other.params_, num_scaleoffsets, block_encoding_tensor_rank_);
    assert(status.IsOK());  // Expect other QnnQuantParamsWrapper to always have a supported quantization encoding.
  }

  return *this;
}

// Per-tensor quantization (SCALE_OFFSET).
QnnQuantParamsWrapper QnnQuantParamsWrapper::PerTensor(float scale, int32_t offset) {
  QnnQuantParamsWrapper qp;
  qp.params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp.params_.scaleOffsetEncoding.scale = scale;
  qp.params_.scaleOffsetEncoding.offset = offset;
  return qp;
}

// Per-tensor quantization with explicit bitwidth (BW_SCALE_OFFSET).
QnnQuantParamsWrapper QnnQuantParamsWrapper::PerTensorBw(float scale, int32_t offset, uint32_t bitwidth) {
  QnnQuantParamsWrapper qp;
  qp.params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  qp.params_.bwScaleOffsetEncoding.bitwidth = bitwidth;
  qp.params_.bwScaleOffsetEncoding.scale = scale;
  qp.params_.bwScaleOffsetEncoding.offset = offset;
  return qp;
}

// Per-channel quantization (AXIS_SCALE_OFFSET).
QnnQuantParamsWrapper QnnQuantParamsWrapper::PerChannel(gsl::span<const float> scales,
                                                        gsl::span<const int32_t> offsets,
                                                        int32_t axis) {
  assert(scales.size() == offsets.size());
  const uint32_t num_elems = static_cast<uint32_t>(scales.size());

  QnnQuantParamsWrapper qp;
  qp.params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  qp.params_.axisScaleOffsetEncoding.numScaleOffsets = num_elems;
  qp.params_.axisScaleOffsetEncoding.axis = axis;

  if (num_elems > 0) {
    const size_t num_bytes = num_elems * sizeof(Qnn_ScaleOffset_t);
    constexpr std::uintptr_t align = alignof(Qnn_ScaleOffset_t);
    qp.per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
    Qnn_ScaleOffset_t* aligned_dst = ALIGN_PTR_UP(qp.per_channel_data_.get(), align, Qnn_ScaleOffset_t*);

    for (size_t i = 0; i < static_cast<uint32_t>(num_elems); i++) {
      aligned_dst[i].offset = offsets[i];
      aligned_dst[i].scale = scales[i];
    }

    qp.params_.axisScaleOffsetEncoding.scaleOffset = aligned_dst;
  } else {
    qp.params_.axisScaleOffsetEncoding.scaleOffset = nullptr;
  }
  return qp;
}

// Per-channel quantization with explicit bitwidth (BW_AXIS_SCALE_OFFSET).
QnnQuantParamsWrapper QnnQuantParamsWrapper::PerChannelBw(gsl::span<const float> scales,
                                                          gsl::span<const int32_t> offsets,
                                                          int32_t axis,
                                                          uint32_t bitwidth) {
  assert(scales.size() == offsets.size());
  const uint32_t num_elems = static_cast<uint32_t>(scales.size());

  QnnQuantParamsWrapper qp;
  qp.params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
  qp.params_.bwAxisScaleOffsetEncoding.numElements = num_elems;
  qp.params_.bwAxisScaleOffsetEncoding.axis = axis;
  qp.params_.bwAxisScaleOffsetEncoding.bitwidth = bitwidth;

  if (num_elems > 0) {
    qp.per_channel_scales_size_ = num_elems;
    const size_t num_scale_bytes = num_elems * sizeof(float);
    const size_t num_zp_bytes = num_elems * sizeof(int32_t);
    const size_t num_bytes = num_scale_bytes + num_zp_bytes;
    constexpr std::uintptr_t align = alignof(float);
    static_assert(alignof(float) == alignof(int32_t));

    qp.per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
    char* scales_begin = ALIGN_PTR_UP(qp.per_channel_data_.get(), align, char*);
    char* zps_begin = scales_begin + num_scale_bytes;

    std::memcpy(scales_begin, scales.data(), num_scale_bytes);
    std::memcpy(zps_begin, offsets.data(), num_zp_bytes);
    qp.params_.bwAxisScaleOffsetEncoding.scales = reinterpret_cast<float*>(scales_begin);
    qp.params_.bwAxisScaleOffsetEncoding.offsets = reinterpret_cast<int32_t*>(zps_begin);
  } else {
    qp.params_.bwAxisScaleOffsetEncoding.scales = nullptr;
    qp.params_.bwAxisScaleOffsetEncoding.offsets = nullptr;
  }
  return qp;
}

// Low-power blockwise (LPBQ) quantization (BLOCKWISE_EXPANSION).
QnnQuantParamsWrapper QnnQuantParamsWrapper::LowPowerBlockwise(gsl::span<const float> per_channel_float_scales,
                                                               gsl::span<const uint8_t> per_block_int_scales,
                                                               gsl::span<const int32_t> offsets,
                                                               int64_t axis,
                                                               uint32_t block_scale_bitwidth) {
  assert(per_channel_float_scales.size() == offsets.size());
  const uint32_t num_elems = static_cast<uint32_t>(per_channel_float_scales.size());

  QnnQuantParamsWrapper qp;
  qp.params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION;

  // Allocate the blockwiseExpansion object.
  const size_t bwe_num_bytes = sizeof(Qnn_BlockwiseExpansion_t);
  constexpr std::uintptr_t bwe_align = alignof(Qnn_BlockwiseExpansion_t);
  qp.blockwise_expansion_data_ = std::make_unique<char[]>(bwe_num_bytes + bwe_align);
  Qnn_BlockwiseExpansion_t* lpbqPtr = ALIGN_PTR_UP(qp.blockwise_expansion_data_.get(), bwe_align,
                                                   Qnn_BlockwiseExpansion_t*);
  Qnn_BlockwiseExpansion_t& lpbq = *lpbqPtr;

  lpbq.axis = static_cast<int32_t>(axis);

  if (num_elems > 0) {
    qp.per_channel_scales_size_ = num_elems;
    const size_t num_bytes = num_elems * sizeof(Qnn_ScaleOffset_t);
    constexpr std::uintptr_t align = alignof(Qnn_ScaleOffset_t);
    qp.per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
    Qnn_ScaleOffset_t* aligned_dst = ALIGN_PTR_UP(qp.per_channel_data_.get(), align, Qnn_ScaleOffset_t*);

    for (size_t i = 0; i < static_cast<uint32_t>(num_elems); i++) {
      aligned_dst[i].offset = offsets[i];
      aligned_dst[i].scale = per_channel_float_scales[i];
    }

    lpbq.scaleOffsets = aligned_dst;
  }

  lpbq.numBlocksPerAxis = static_cast<uint32_t>(per_block_int_scales.size()) / num_elems;
  lpbq.blockScaleBitwidth = block_scale_bitwidth;
  lpbq.blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8;

  // Deep copy the per-block int scales.
  const size_t num_bytes = per_block_int_scales.size() * sizeof(uint8_t);
  constexpr std::uintptr_t align = alignof(uint8_t);
  qp.block_scales_data_ = std::make_unique<uint8_t[]>(num_bytes + align);
  uint8_t* aligned_dst = ALIGN_PTR_UP(qp.block_scales_data_.get(), align, uint8_t*);
  for (size_t i = 0; i < static_cast<uint32_t>(per_block_int_scales.size()); i++) {
    aligned_dst[i] = per_block_int_scales[i];
  }
  lpbq.blocksScale8 = aligned_dst;

  qp.params_.blockwiseExpansion = lpbqPtr;
  return qp;
}

// Block-encoded quantization (BLOCK).
QnnQuantParamsWrapper QnnQuantParamsWrapper::Block(gsl::span<const float> scales,
                                                   gsl::span<const int32_t> offsets,
                                                   gsl::span<const uint32_t> block_sizes) {
  assert(block_sizes.size() > 0);
  assert(scales.size() > 0);
  assert(scales.size() == offsets.size());

  QnnQuantParamsWrapper qp;
  qp.num_blocks_ = static_cast<uint32_t>(scales.size());
  qp.params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCK;

  qp.block_encoding_tensor_rank_ = static_cast<uint32_t>(block_sizes.size());
  qp.block_encoding_axis_data_ = std::make_unique<uint32_t[]>(qp.block_encoding_tensor_rank_);
  std::memcpy(qp.block_encoding_axis_data_.get(),
              block_sizes.data(),
              static_cast<size_t>(qp.block_encoding_tensor_rank_) * sizeof(uint32_t));
  qp.params_.blockEncoding.blockSize = qp.block_encoding_axis_data_.get();

  if (qp.num_blocks_ > 0) {
    qp.block_encoding_scale_offsets_data_ = std::make_unique<Qnn_ScaleOffset_t[]>(qp.num_blocks_);
    for (size_t i = 0; i < qp.num_blocks_; ++i) {
      qp.block_encoding_scale_offsets_data_[i].offset = offsets[i];
      qp.block_encoding_scale_offsets_data_[i].scale = scales[i];
    }
    qp.params_.blockEncoding.scaleOffset = qp.block_encoding_scale_offsets_data_.get();
  }
  return qp;
}

// Block-encoded quantization with explicit bitwidth and float offsets (BW_FLOAT_BLOCK).
QnnQuantParamsWrapper QnnQuantParamsWrapper::BwFloatBlock(gsl::span<const float> scales,
                                                          gsl::span<const float> offsets,
                                                          uint32_t bitwidth,
                                                          gsl::span<const uint32_t> block_sizes) {
  assert(scales.size() > 0);
  assert(scales.size() == offsets.size());
  assert(bitwidth > 0);
  assert(block_sizes.size() > 0);

  QnnQuantParamsWrapper qp;
  qp.params_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK;
  qp.params_.bwFloatBlockEncoding.bitwidth = bitwidth;

  qp.block_encoding_tensor_rank_ = static_cast<uint32_t>(block_sizes.size());
  qp.block_encoding_axis_data_ = std::make_unique<uint32_t[]>(qp.block_encoding_tensor_rank_);
  std::memcpy(qp.block_encoding_axis_data_.get(),
              block_sizes.data(),
              static_cast<size_t>(qp.block_encoding_tensor_rank_) * sizeof(uint32_t));
  qp.params_.bwFloatBlockEncoding.blockSize = qp.block_encoding_axis_data_.get();

  qp.num_blocks_ = static_cast<uint32_t>(scales.size());
  qp.bw_float_block_encoding_scale_offsets_data_ = std::make_unique<Qnn_FloatScaleOffset_t[]>(qp.num_blocks_);
  for (size_t idx = 0; idx < qp.num_blocks_; ++idx) {
    qp.bw_float_block_encoding_scale_offsets_data_[idx].offset = offsets[idx];
    qp.bw_float_block_encoding_scale_offsets_data_[idx].scale = scales[idx];
  }
  qp.params_.bwFloatBlockEncoding.floatScaleOffset = qp.bw_float_block_encoding_scale_offsets_data_.get();
  return qp;
}

// Get a copy of scales. Works for both per-tensor and per-channel.
Ort::Status QnnQuantParamsWrapper::GetScales(/*out*/ std::vector<float>& scales) const {
  RETURN_IF_NOT(params_.encodingDefinition == QNN_DEFINITION_DEFINED, "Unquantized qparams does not have scales");

  switch (params_.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      scales.resize(1);
      scales[0] = params_.scaleOffsetEncoding.scale;
      break;
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
      scales.resize(1);
      scales[0] = params_.bwScaleOffsetEncoding.scale;
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      const uint32_t num_elems = params_.axisScaleOffsetEncoding.numScaleOffsets;
      scales.resize(num_elems);

      if (num_elems > 0) {
        gsl::span<const Qnn_ScaleOffset_t> scale_offsets(params_.axisScaleOffsetEncoding.scaleOffset, num_elems);

        for (size_t i = 0; i < num_elems; i++) {
          scales[i] = scale_offsets[i].scale;
        }
      }
      break;
    }
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
      const uint32_t num_elems = params_.bwAxisScaleOffsetEncoding.numElements;
      scales.resize(num_elems);

      // Deep copy the scales[] and offsets[] arrays
      if (num_elems > 0) {
        gsl::span<const float> src_scales(params_.bwAxisScaleOffsetEncoding.scales, num_elems);
        for (size_t i = 0; i < num_elems; i++) {
          scales[i] = src_scales[i];
        }
      }
      break;
    }
    case QNN_QUANTIZATION_ENCODING_BLOCK: {
      scales.resize(num_blocks_);

      if (num_blocks_ > 0) {
        gsl::span<const Qnn_ScaleOffset_t> scale_offsets(params_.blockEncoding.scaleOffset, num_blocks_);

        for (size_t i = 0; i < num_blocks_; i++) {
          scales[i] = scale_offsets[i].scale;
        }
      }
      break;
    }
    default:
      return MAKE_EP_FAIL(("Unsupported QNN quantization encoding: " +
                           std::to_string(params_.quantizationEncoding))
                              .c_str());
  }

  return Ort::Status();
}

QnnQuantParamsWrapper QnnQuantParamsWrapper::Copy() const {
  return QnnQuantParamsWrapper(*this);
}

// Initializes by copying from a Qnn_QuantizeParams_t.
Ort::Status QnnQuantParamsWrapper::Init(const Qnn_QuantizeParams_t& params, const size_t num_scaleoffsets, const size_t tensor_rank) {
  if (per_channel_data_) {
    per_channel_data_.reset(nullptr);
    params_ = QNN_QUANTIZE_PARAMS_INIT;
  }

  if (params.encodingDefinition != QNN_DEFINITION_DEFINED) {
    params_ = params;
    return Ort::Status();
  }

  switch (params.quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
    case QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET:
      params_ = params;
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
      params_.encodingDefinition = params.encodingDefinition;
      params_.quantizationEncoding = params.quantizationEncoding;
      params_.axisScaleOffsetEncoding.axis = params.axisScaleOffsetEncoding.axis;
      params_.axisScaleOffsetEncoding.numScaleOffsets = params.axisScaleOffsetEncoding.numScaleOffsets;

      // Deep copy the scaleOffset data.
      const uint32_t num_elems = params.axisScaleOffsetEncoding.numScaleOffsets;

      if (num_elems > 0) {
        const size_t num_bytes = num_elems * sizeof(Qnn_ScaleOffset_t);
        constexpr std::uintptr_t align = alignof(Qnn_ScaleOffset_t);
        per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
        Qnn_ScaleOffset_t* aligned_dst = ALIGN_PTR_UP(per_channel_data_.get(), align, Qnn_ScaleOffset_t*);

        std::memcpy(aligned_dst, params.axisScaleOffsetEncoding.scaleOffset, num_bytes);
        params_.axisScaleOffsetEncoding.scaleOffset = aligned_dst;
      } else {
        params_.axisScaleOffsetEncoding.scaleOffset = nullptr;
      }
      break;
    }
    case QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET: {
      const uint32_t num_elems = params.bwAxisScaleOffsetEncoding.numElements;

      params_.encodingDefinition = params.encodingDefinition;
      params_.quantizationEncoding = params.quantizationEncoding;
      params_.bwAxisScaleOffsetEncoding.axis = params.bwAxisScaleOffsetEncoding.axis;
      params_.bwAxisScaleOffsetEncoding.bitwidth = params.bwAxisScaleOffsetEncoding.bitwidth;
      params_.bwAxisScaleOffsetEncoding.numElements = num_elems;

      // Deep copy the scales[] and offsets[] arrays
      if (num_elems > 0) {
        const size_t num_scale_bytes = num_elems * sizeof(float);
        const size_t num_zp_bytes = num_elems * sizeof(int32_t);
        const size_t num_bytes = num_scale_bytes + num_zp_bytes;
        constexpr std::uintptr_t align = alignof(float);
        static_assert(alignof(float) == alignof(int32_t));

        per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
        char* scales_begin = ALIGN_PTR_UP(per_channel_data_.get(), align, char*);
        char* zps_begin = scales_begin + num_scale_bytes;

        std::memcpy(scales_begin, params.bwAxisScaleOffsetEncoding.scales, num_scale_bytes);
        std::memcpy(zps_begin, params.bwAxisScaleOffsetEncoding.offsets, num_zp_bytes);
        params_.bwAxisScaleOffsetEncoding.scales = reinterpret_cast<float*>(scales_begin);
        params_.bwAxisScaleOffsetEncoding.offsets = reinterpret_cast<int32_t*>(zps_begin);
      } else {
        params_.bwAxisScaleOffsetEncoding.scales = nullptr;
        params_.bwAxisScaleOffsetEncoding.offsets = nullptr;
      }
      break;
    }
    case QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION: {
      assert(num_scaleoffsets && "Can't create BlockwiseExpansion encoding object with zero ScaleOffsets");
      params_.encodingDefinition = params.encodingDefinition;
      params_.quantizationEncoding = params.quantizationEncoding;

      per_channel_scales_size_ = static_cast<uint32_t>(num_scaleoffsets);

      // Deep copy the blockwiseExpansion
      const size_t bwe_num_bytes = sizeof(Qnn_BlockwiseExpansion_t);
      constexpr std::uintptr_t bwe_align = alignof(Qnn_BlockwiseExpansion_t);
      blockwise_expansion_data_ = std::make_unique<char[]>(bwe_num_bytes + bwe_align);
      Qnn_BlockwiseExpansion_t* bwe_aligned_dst = ALIGN_PTR_UP(blockwise_expansion_data_.get(), bwe_align, Qnn_BlockwiseExpansion_t*);
      std::memcpy(bwe_aligned_dst, params.blockwiseExpansion, bwe_num_bytes);
      params_.blockwiseExpansion = bwe_aligned_dst;

      // Deep copy the scaleoffsets
      const size_t so_num_elems = num_scaleoffsets;
      const size_t so_num_bytes = so_num_elems * sizeof(Qnn_ScaleOffset_t);
      constexpr std::uintptr_t so_align = alignof(Qnn_ScaleOffset_t);
      per_channel_data_ = std::make_unique<char[]>(so_num_bytes + so_align);
      Qnn_ScaleOffset_t* so_aligned_dst = ALIGN_PTR_UP(per_channel_data_.get(), so_align, Qnn_ScaleOffset_t*);

      std::memcpy(so_aligned_dst, params.blockwiseExpansion->scaleOffsets, so_num_bytes);
      params_.blockwiseExpansion->scaleOffsets = so_aligned_dst;

      // Deep copy blockscales
      const size_t bs_num_elems = num_scaleoffsets * params.blockwiseExpansion->numBlocksPerAxis;
      const size_t bs_num_bytes = bs_num_elems * sizeof(uint8_t);
      constexpr std::uintptr_t bs_align = alignof(uint8_t);
      block_scales_data_ = std::make_unique<uint8_t[]>(bs_num_bytes + bs_align);
      uint8_t* bs_aligned_dst = ALIGN_PTR_UP(block_scales_data_.get(), bs_align, uint8_t*);
      std::memcpy(bs_aligned_dst, params.blockwiseExpansion->blocksScale8, bs_num_bytes);
      params_.blockwiseExpansion->blocksScale8 = bs_aligned_dst;
      break;
    }
    case QNN_QUANTIZATION_ENCODING_BLOCK: {
      assert(num_scaleoffsets && "Can't create Block encoding object with zero ScaleOffsets");
      params_.encodingDefinition = params.encodingDefinition;
      params_.quantizationEncoding = params.quantizationEncoding;

      num_blocks_ = static_cast<uint32_t>(num_scaleoffsets);
      block_encoding_tensor_rank_ = static_cast<uint32_t>(tensor_rank);
      block_encoding_axis_data_ = std::make_unique<uint32_t[]>(block_encoding_tensor_rank_);
      std::memcpy(block_encoding_axis_data_.get(),
                  params.blockEncoding.blockSize,
                  static_cast<size_t>(block_encoding_tensor_rank_) * sizeof(uint32_t));
      params_.blockEncoding.blockSize = block_encoding_axis_data_.get();

      // Deep copy the scale offsets
      block_encoding_scale_offsets_data_ = std::make_unique<Qnn_ScaleOffset_t[]>(num_scaleoffsets);
      for (size_t i = 0; i < num_scaleoffsets; ++i) {
        block_encoding_scale_offsets_data_[i].scale = params.blockEncoding.scaleOffset[i].scale;
        block_encoding_scale_offsets_data_[i].offset = params.blockEncoding.scaleOffset[i].offset;
      }
      params_.blockEncoding.scaleOffset = block_encoding_scale_offsets_data_.get();

      break;
    }
    case QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK: {
      assert(num_scaleoffsets && "Can't create Block encoding object with zero ScaleOffsets");
      params_.encodingDefinition = params.encodingDefinition;
      params_.quantizationEncoding = params.quantizationEncoding;
      params_.bwFloatBlockEncoding.bitwidth = params.bwFloatBlockEncoding.bitwidth;

      block_encoding_tensor_rank_ = static_cast<uint32_t>(tensor_rank);
      block_encoding_axis_data_ = std::make_unique<uint32_t[]>(block_encoding_tensor_rank_);
      std::memcpy(block_encoding_axis_data_.get(),
                  params.bwFloatBlockEncoding.blockSize,
                  static_cast<size_t>(block_encoding_tensor_rank_) * sizeof(uint32_t));
      params_.bwFloatBlockEncoding.blockSize = block_encoding_axis_data_.get();

      bw_float_block_encoding_scale_offsets_data_ = std::make_unique<Qnn_FloatScaleOffset_t[]>(num_scaleoffsets);
      for (size_t i = 0; i < num_scaleoffsets; ++i) {
        bw_float_block_encoding_scale_offsets_data_[i].scale = params.bwFloatBlockEncoding.floatScaleOffset[i].scale;
        bw_float_block_encoding_scale_offsets_data_[i].offset = params.bwFloatBlockEncoding.floatScaleOffset[i].offset;
      }
      params_.bwFloatBlockEncoding.floatScaleOffset = bw_float_block_encoding_scale_offsets_data_.get();

      break;
    }
    default:
      return MAKE_EP_FAIL(("Unsupported QNN quantization encoding: " +
                           std::to_string(params.quantizationEncoding))
                              .c_str());
  }

  return Ort::Status();
}

// Initialize this object from a (potentially) quantized ONNX tensor.
// QnnModelWrapper provides utilities for unpacking scale and zero-point ONNX initializers.
Ort::Status QnnQuantParamsWrapper::Init(const QnnModelWrapper& qnn_model_wrapper,
                                        const OrtNodeUnitIODef& io_def) {
  const std::optional<OrtNodeUnitIODef::QuantParam>& ort_quant_params = io_def.quant_param;

  if (per_channel_data_) {
    per_channel_data_.reset(nullptr);
    params_ = QNN_QUANTIZE_PARAMS_INIT;
  }

  if (!ort_quant_params.has_value()) {
    params_.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    return Ort::Status();
  }

  std::vector<float> scales;
  std::vector<int32_t> zero_points;

  RETURN_IF_ERROR(qnn_model_wrapper.UnpackScales(ort_quant_params->scale, scales));

  bool is_int4_type = false;

  if (ort_quant_params->zero_point != nullptr) {
    ONNXTensorElementDataType onnx_tp_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_IF_ERROR(qnn_model_wrapper.UnpackZeroPoints(ort_quant_params->zero_point, zero_points, onnx_tp_type));

    is_int4_type = (onnx_tp_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4) ||
                   (onnx_tp_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4);
  }

  const bool is_block_quant = ort_quant_params->block_size.has_value() && ort_quant_params->block_size.value() > 0;
  const bool is_per_channel = scales.size() > 1 && !is_block_quant;
  const bool is_per_tensor = scales.size() == 1 && !is_block_quant;

  // QNN uses different structs to represent quantization parameters depending on:
  // - per-tensor (scales.size()==1, no block_size): SCALE_OFFSET or BW_SCALE_OFFSET
  // - per-channel (scales.size()>1, no block_size): AXIS_SCALE_OFFSET or BW_AXIS_SCALE_OFFSET
  // - block quantization (block_size>0): BLOCKWISE_EXPANSION (LPBQ) or ENCODING_BLOCK (BQ)
  // - fallback: error
  if (is_per_tensor && !is_int4_type) {
    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    params_.scaleOffsetEncoding.scale = scales[0];

    if (ort_quant_params->zero_point != nullptr) {
      RETURN_IF_NOT(zero_points.size() == 1, "Expected one zero-point value");
      params_.scaleOffsetEncoding.offset = zero_points[0];
    } else {
      params_.scaleOffsetEncoding.offset = 0;
    }
  } else if (is_per_tensor && is_int4_type) {
    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
    params_.bwScaleOffsetEncoding.bitwidth = 4;
    params_.bwScaleOffsetEncoding.scale = scales[0];

    if (ort_quant_params->zero_point != nullptr) {
      RETURN_IF_NOT(zero_points.size() == 1, "Expected one zero-point value");
      params_.bwScaleOffsetEncoding.offset = zero_points[0];
    } else {
      params_.bwScaleOffsetEncoding.offset = 0;
    }
  } else if (is_per_channel && is_int4_type) {
    std::vector<uint32_t> io_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(io_def.shape, io_shape), "Cannot get shape");
    const int32_t io_rank = static_cast<int32_t>(io_shape.size());

    constexpr int64_t DEFAULT_QDQ_AXIS = 1;
    int64_t axis = ort_quant_params->axis.value_or(DEFAULT_QDQ_AXIS);
    if (axis < 0) {
      axis += io_rank;
    }
    RETURN_IF_NOT(axis >= 0 && axis < io_rank,
                  "Quantization axis must be within the range [0, rank - 1]");

    const size_t num_elems = scales.size();
    const bool no_zero_points = zero_points.empty();
    RETURN_IF_NOT(num_elems > 1, "Expected more than one scale value");
    RETURN_IF_NOT(no_zero_points || zero_points.size() == num_elems,
                  "Expected the same number of zero-points and scales for per-channel quantization");

    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
    params_.bwAxisScaleOffsetEncoding.axis = static_cast<int32_t>(axis);
    params_.bwAxisScaleOffsetEncoding.bitwidth = 4;
    params_.bwAxisScaleOffsetEncoding.numElements = static_cast<uint32_t>(num_elems);

    const size_t num_scale_bytes = num_elems * sizeof(float);
    const size_t num_zp_bytes = num_elems * sizeof(int32_t);
    const size_t num_bytes = num_scale_bytes + num_zp_bytes;
    constexpr std::uintptr_t align = alignof(float);
    per_channel_data_ = std::make_unique<char[]>(num_bytes + align);

    char* scales_begin = ALIGN_PTR_UP(per_channel_data_.get(), align, char*);
    char* zps_begin = scales_begin + num_scale_bytes;
    gsl::span<float> scales_span(reinterpret_cast<float*>(scales_begin), num_elems);
    gsl::span<int32_t> zps_span(reinterpret_cast<int32_t*>(zps_begin), num_elems);

    for (size_t i = 0; i < num_elems; i++) {
      scales_span[i] = scales[i];
      zps_span[i] = no_zero_points ? 0 : zero_points[i];
    }

    params_.bwAxisScaleOffsetEncoding.scales = scales_span.data();
    params_.bwAxisScaleOffsetEncoding.offsets = zps_span.data();
  } else if (is_per_channel && !is_int4_type) {
    std::vector<uint32_t> io_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(io_def.shape, io_shape), "Cannot get shape");
    const int32_t io_rank = static_cast<int32_t>(io_shape.size());

    constexpr int64_t DEFAULT_QDQ_AXIS = 1;
    int64_t axis = ort_quant_params->axis.value_or(DEFAULT_QDQ_AXIS);
    if (axis < 0) {
      axis += io_rank;
    }
    RETURN_IF_NOT(axis >= 0 && axis < io_rank,
                  "Quantization axis must be within the range [0, rank - 1]");

    params_.encodingDefinition = QNN_DEFINITION_DEFINED;
    params_.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;

    const size_t num_elems = scales.size();
    const bool no_zero_points = zero_points.empty();
    RETURN_IF_NOT(num_elems > 1, "Expected more than one scale value");
    RETURN_IF_NOT(no_zero_points || zero_points.size() == num_elems,
                  "Expected the same number of zero-points and scales for per-channel quantization");

    const size_t num_bytes = num_elems * sizeof(Qnn_ScaleOffset_t);
    constexpr std::uintptr_t align = alignof(Qnn_ScaleOffset_t);
    per_channel_data_ = std::make_unique<char[]>(num_bytes + align);
    Qnn_ScaleOffset_t* aligned_dst = ALIGN_PTR_UP(per_channel_data_.get(), align, Qnn_ScaleOffset_t*);
    gsl::span<Qnn_ScaleOffset_t> data_span(aligned_dst, num_elems);

    for (size_t i = 0; i < num_elems; i++) {
      data_span[i].scale = scales[i];
      data_span[i].offset = no_zero_points ? 0 : zero_points[i];
    }

    params_.axisScaleOffsetEncoding.axis = static_cast<int32_t>(axis);
    params_.axisScaleOffsetEncoding.numScaleOffsets = static_cast<uint32_t>(num_elems);
    params_.axisScaleOffsetEncoding.scaleOffset = data_span.data();
  } else if (is_block_quant) {
    if (!qnn_model_wrapper.GetModelSettings().enable_block_quant_weight_optimization) {
      ORT_CXX_LOG(qnn_model_wrapper.GetLogger(), ORT_LOGGING_LEVEL_VERBOSE,
                  ("Block quant weight optimization disabled, falling back to float BQ path"));
      return Ort::Status();
    }
    // ONNX block quantization -> QNN LPBQ (BLOCKWISE_EXPANSION) conversion only supported for 4-bit.
    if (io_def.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4) {
      ORT_CXX_LOG(qnn_model_wrapper.GetLogger(), ORT_LOGGING_LEVEL_VERBOSE,
                  ("BQ to LPBQ conversion only supported for int4 weights, falling back to float BQ path"));
      return Ort::Status();
    }
    // LPBQ requires symmetric quantization (all zero-points must be zero).
    for (const int32_t zp : zero_points) {
      if (zp != 0) {
        ORT_CXX_LOG(qnn_model_wrapper.GetLogger(), ORT_LOGGING_LEVEL_VERBOSE,
                    ("BQ to LPBQ conversion requires symmetric quantization, falling back to float BQ path"));
        return Ort::Status();
      }
    }

    std::vector<uint32_t> io_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(io_def.shape, io_shape), "Cannot get shape");
    const int32_t io_rank = static_cast<int32_t>(io_shape.size());

    // Get scale tensor shape to determine block/channel dimensions.
    // Scale tensor may be rank 2 (e.g., MatMul/Gemm) or higher rank (e.g., Conv with rank-4 weights).
    // Only the first two dimensions (indexed by onnx_axis and 1 - onnx_axis) are used for LPBQ conversion.
    const std::vector<int64_t> scale_shape =
        utils::GetInitializerShape(ort_quant_params->scale, qnn_model_wrapper.GetOrtApi());
    RETURN_IF_NOT(scale_shape.size() >= 2 && scale_shape.size() <= 4,
                  "Block quantization scale tensors must have rank between 2 and 4 for LPBQ conversion");
    RETURN_IF_NOT(scale_shape[0] > 0 && scale_shape[1] > 0,
                  "Block quantization scale tensor dimensions must be positive");
    RETURN_IF_NOT(scale_shape[0] * scale_shape[1] == static_cast<int64_t>(scales.size()),
                  "Block quantization scale tensor shape product must equal number of scales");

    // Determine block axis (= ONNX axis attribute).
    constexpr int64_t DEFAULT_QDQ_AXIS = 1;
    int64_t axis = ort_quant_params->axis.value_or(DEFAULT_QDQ_AXIS);
    if (axis < 0) axis += io_rank;
    RETURN_IF_NOT(axis == 0 || axis == 1,
                  "Only axis 0 or 1 is supported for block quantization LPBQ conversion");

    // Scale shape: [num_blocks_per_channel, num_channels] when axis=0
    //              [num_channels, num_blocks_per_channel] when axis=1
    const uint32_t num_blocks_per_channel = static_cast<uint32_t>(scale_shape[axis]);
    const uint32_t num_channels = static_cast<uint32_t>(scale_shape[1 - axis]);

    // The conversion algorithm expects scales in block-major order [num_blocks, num_channels].
    // If axis=1 the raw tensor is channel-major [num_channels, num_blocks]; transpose it.
    std::vector<float> bq_scales_bm;
    if (axis == 0) {
      bq_scales_bm = std::move(scales);
    } else {
      // Transpose [num_channels, num_blocks] -> [num_blocks, num_channels]
      bq_scales_bm.resize(scales.size());
      for (uint32_t c = 0; c < num_channels; ++c) {
        for (uint32_t b = 0; b < num_blocks_per_channel; ++b) {
          bq_scales_bm[static_cast<size_t>(b) * num_channels + c] =
              scales[static_cast<size_t>(c) * num_blocks_per_channel + b];
        }
      }
    }

    // Apply BQ -> LPBQ algorithm
    std::vector<float> per_channel_scales;
    std::vector<uint8_t> per_block_int_scales;
    std::vector<int32_t> lpbq_offsets;
    const uint32_t bitwidth = 4u;
    Ort::Status status = utils::ConvertBlockQuantScalesToLpbq(bq_scales_bm, zero_points, num_blocks_per_channel,
                                                              num_channels, bitwidth, per_channel_scales,
                                                              per_block_int_scales, lpbq_offsets);
    if (!status.IsOK()) {
      ORT_CXX_LOG(qnn_model_wrapper.GetLogger(), ORT_LOGGING_LEVEL_VERBOSE,
                  ("BQ to LPBQ conversion failed, falling back to float BQ path: " + std::string(status.GetErrorMessage())).c_str());
      return Ort::Status();
    }

    // QNN LPBQ axis = the non-block axis in the weight tensor.
    // For ONNX axis=0 (block axis=0): QNN axis=1; for axis=1: QNN axis=0.
    const int64_t qnn_axis = 1 - axis;

    *this = QnnQuantParamsWrapper::LowPowerBlockwise(per_channel_scales, per_block_int_scales, lpbq_offsets,
                                                     qnn_axis, /*block_scale_bitwidth=*/is_int4_type ? 4 : 8);
  } else {
    return MAKE_EP_FAIL("Unexpected tensor kind for QuantParamsWrapper::Init()");
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
