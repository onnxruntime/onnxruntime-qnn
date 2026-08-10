// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Function-level unit tests for qnn_quant_params_wrapper.cc — QNN quantization
// parameter wrapper covering all 5 supported encodings (per-tensor, per-channel,
// LPBQ blockwise expansion, block encoding, BW float block encoding) plus deep
// copy semantics, GetScales, axis-handling templated helpers, and the two
// Init() overloads.
//
// Init(QnnModelWrapper, OrtNodeUnitIODef) is exercised via mock_init_registry —
// stubs the OrtApi function pointers so UnpackScales / UnpackZeroPoints /
// UnpackInitializerData succeed without a real ORT graph.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <optional>
#include <vector>

#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_quant_params_wrapper.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/mock_init_registry.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

using qnn::QnnQuantParamsWrapper;

namespace {

// Return a fresh per-channel non-int4 wrapper (3 channels at axis 1).
// Useful as a starting state for tests that need to verify reset / re-init.
QnnQuantParamsWrapper MakePerChannelNonInt4(int32_t axis = 1) {
  const std::vector<float> scales{0.1f, 0.2f, 0.3f};
  const std::vector<int32_t> offsets{1, 2, 3};
  return QnnQuantParamsWrapper::PerChannel(gsl::make_span(scales), gsl::make_span(offsets), axis);
}

QnnQuantParamsWrapper MakePerChannelInt4(int32_t axis = 1) {
  const std::vector<float> scales{0.1f, 0.2f, 0.3f};
  const std::vector<int32_t> offsets{1, 2, 3};
  return QnnQuantParamsWrapper::PerChannelBw(gsl::make_span(scales), gsl::make_span(offsets), axis, /*bitwidth=*/4);
}

QnnQuantParamsWrapper MakeLPBQ() {
  // 3 per-channel scales, 2 blocks per axis ⇒ 6 per-block int scales.
  const std::vector<float> per_channel_scales{0.1f, 0.2f, 0.3f};
  const std::vector<uint8_t> per_block_scales{1, 2, 3, 4, 5, 6};
  const std::vector<int32_t> offsets{0, 0, 0};
  return QnnQuantParamsWrapper::LowPowerBlockwise(gsl::make_span(per_channel_scales),
                                                  gsl::make_span(per_block_scales),
                                                  gsl::make_span(offsets),
                                                  /*axis=*/1,
                                                  /*block_scale_bitwidth=*/4);
}

QnnQuantParamsWrapper MakeBlockEncoding() {
  const std::vector<float> scales{0.5f, 0.25f};
  const std::vector<int32_t> offsets{1, 2};
  const std::vector<uint32_t> block_sizes{2, 1};
  return QnnQuantParamsWrapper::Block(gsl::make_span(scales),
                                      gsl::make_span(offsets),
                                      gsl::make_span(block_sizes));
}

QnnQuantParamsWrapper MakeBwFloatBlock() {
  const std::vector<float> scales{0.5f, 0.25f};
  const std::vector<float> offsets{0.0f, 0.1f};
  const std::vector<uint32_t> block_sizes{2, 1};
  return QnnQuantParamsWrapper::BwFloatBlock(gsl::make_span(scales),
                                             gsl::make_span(offsets),
                                             /*bitwidth=*/8u,
                                             gsl::make_span(block_sizes));
}

}  // namespace

// =============================================================================
// Constructors
// =============================================================================

TEST(QnnUnit_QuantParamsWrapperTest, Default_NotQuantized_AndAllPredicatesFalse) {
  QnnQuantParamsWrapper q;
  EXPECT_FALSE(q.IsQuantized());
  EXPECT_FALSE(q.IsPerTensor());
  EXPECT_FALSE(q.IsPerTensor(/*include_bw=*/true));
  EXPECT_FALSE(q.IsPerChannel());
  EXPECT_FALSE(q.IsLPBQ());
  EXPECT_FALSE(q.IsBlockQuantized());
  EXPECT_EQ(q.Get().encodingDefinition, QNN_DEFINITION_UNDEFINED);
}

TEST(QnnUnit_QuantParamsWrapperTest, PerTensor_SetsScaleOffsetEncodingAndScale) {
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerTensor(/*scale=*/0.5f, /*offset=*/-3);
  EXPECT_TRUE(q.IsQuantized());
  EXPECT_TRUE(q.IsPerTensor());
  EXPECT_FALSE(q.IsPerChannel());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(q.Get().scaleOffsetEncoding.scale, 0.5f);
  EXPECT_EQ(q.Get().scaleOffsetEncoding.offset, -3);
}

TEST(QnnUnit_QuantParamsWrapperTest, PerTensorBw_SetsBwScaleOffsetEncodingAndBitwidth) {
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerTensorBw(/*scale=*/0.25f, /*offset=*/7, /*bitwidth=*/4);
  EXPECT_TRUE(q.IsQuantized());
  EXPECT_TRUE(q.IsPerTensor(/*include_bw=*/true));
  EXPECT_FALSE(q.IsPerTensor(/*include_bw=*/false));
  EXPECT_FALSE(q.IsPerChannel());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET);
  EXPECT_EQ(q.Get().bwScaleOffsetEncoding.bitwidth, 4u);
  EXPECT_FLOAT_EQ(q.Get().bwScaleOffsetEncoding.scale, 0.25f);
  EXPECT_EQ(q.Get().bwScaleOffsetEncoding.offset, 7);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     PerChannel_NotInt4_NonEmpty_SetsAxisScaleOffsetAndDeepCopiesData) {
  const std::vector<float> scales{0.1f, 0.2f, 0.3f};
  const std::vector<int32_t> offsets{1, 2, 3};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannel(gsl::make_span(scales), gsl::make_span(offsets), /*axis=*/2);
  EXPECT_TRUE(q.IsPerChannel());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.axis, 2);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.numScaleOffsets, 3u);
  ASSERT_NE(q.Get().axisScaleOffsetEncoding.scaleOffset, nullptr);
  for (size_t i = 0; i < scales.size(); ++i) {
    EXPECT_FLOAT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset[i].scale, scales[i]);
    EXPECT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset[i].offset, offsets[i]);
  }
}

TEST(QnnUnit_QuantParamsWrapperTest, PerChannel_NotInt4_Empty_SetsNullScaleOffsetPointer) {
  const std::vector<float> scales{};
  const std::vector<int32_t> offsets{};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannel(gsl::make_span(scales), gsl::make_span(offsets), /*axis=*/0);
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.numScaleOffsets, 0u);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset, nullptr);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     PerChannel_Int4_NonEmpty_SetsBwAxisScaleOffsetAndDeepCopiesScalesAndOffsets) {
  const std::vector<float> scales{0.1f, 0.2f};
  const std::vector<int32_t> offsets{4, 5};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannelBw(gsl::make_span(scales), gsl::make_span(offsets), /*axis=*/1, /*bitwidth=*/4);
  EXPECT_TRUE(q.IsPerChannel());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.axis, 1);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.bitwidth, 4u);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.numElements, 2u);
  ASSERT_NE(q.Get().bwAxisScaleOffsetEncoding.scales, nullptr);
  ASSERT_NE(q.Get().bwAxisScaleOffsetEncoding.offsets, nullptr);
  for (size_t i = 0; i < scales.size(); ++i) {
    EXPECT_FLOAT_EQ(q.Get().bwAxisScaleOffsetEncoding.scales[i], scales[i]);
    EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.offsets[i], offsets[i]);
  }
}

TEST(QnnUnit_QuantParamsWrapperTest, PerChannel_Int4_Empty_SetsNullScalesAndOffsetsPointers) {
  const std::vector<float> scales{};
  const std::vector<int32_t> offsets{};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannelBw(gsl::make_span(scales), gsl::make_span(offsets), /*axis=*/0, /*bitwidth=*/4);
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.numElements, 0u);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.scales, nullptr);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.offsets, nullptr);
}

TEST(QnnUnit_QuantParamsWrapperTest, LPBQ_SetsBlockwiseExpansionAndDeepCopiesAllArrays) {
  QnnQuantParamsWrapper q = MakeLPBQ();
  EXPECT_TRUE(q.IsLPBQ());
  EXPECT_FALSE(q.IsPerChannel());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION);

  const Qnn_BlockwiseExpansion_t* lpbq = q.Get().blockwiseExpansion;
  ASSERT_NE(lpbq, nullptr);
  EXPECT_EQ(lpbq->axis, 1);
  EXPECT_EQ(lpbq->numBlocksPerAxis, 2u);  // 6 block scales / 3 channels
  EXPECT_EQ(lpbq->blockScaleBitwidth, 4u);
  EXPECT_EQ(lpbq->blockScaleStorageType, QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8);

  ASSERT_NE(lpbq->scaleOffsets, nullptr);
  EXPECT_FLOAT_EQ(lpbq->scaleOffsets[0].scale, 0.1f);
  EXPECT_FLOAT_EQ(lpbq->scaleOffsets[1].scale, 0.2f);
  EXPECT_FLOAT_EQ(lpbq->scaleOffsets[2].scale, 0.3f);
  EXPECT_EQ(lpbq->scaleOffsets[0].offset, 0);

  ASSERT_NE(lpbq->blocksScale8, nullptr);
  for (uint32_t i = 0; i < 6; ++i) {
    EXPECT_EQ(lpbq->blocksScale8[i], static_cast<uint8_t>(i + 1));
  }
}

TEST(QnnUnit_QuantParamsWrapperTest, BlockQuant_BlockEncoding_SetsBlockSizesAndScaleOffsets) {
  QnnQuantParamsWrapper q = MakeBlockEncoding();
  EXPECT_TRUE(q.IsBlockQuantized());
  EXPECT_FALSE(q.IsPerChannel());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BLOCK);

  ASSERT_NE(q.Get().blockEncoding.blockSize, nullptr);
  EXPECT_EQ(q.Get().blockEncoding.blockSize[0], 2u);
  EXPECT_EQ(q.Get().blockEncoding.blockSize[1], 1u);

  ASSERT_NE(q.Get().blockEncoding.scaleOffset, nullptr);
  EXPECT_FLOAT_EQ(q.Get().blockEncoding.scaleOffset[0].scale, 0.5f);
  EXPECT_EQ(q.Get().blockEncoding.scaleOffset[0].offset, 1);
  EXPECT_FLOAT_EQ(q.Get().blockEncoding.scaleOffset[1].scale, 0.25f);
  EXPECT_EQ(q.Get().blockEncoding.scaleOffset[1].offset, 2);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     BlockQuant_BwFloatBlockEncoding_SetsBitwidthAndFloatScaleOffsets) {
  QnnQuantParamsWrapper q = MakeBwFloatBlock();
  EXPECT_TRUE(q.IsBlockQuantized());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK);
  EXPECT_EQ(q.Get().bwFloatBlockEncoding.bitwidth, 8u);

  ASSERT_NE(q.Get().bwFloatBlockEncoding.blockSize, nullptr);
  EXPECT_EQ(q.Get().bwFloatBlockEncoding.blockSize[0], 2u);
  EXPECT_EQ(q.Get().bwFloatBlockEncoding.blockSize[1], 1u);

  ASSERT_NE(q.Get().bwFloatBlockEncoding.floatScaleOffset, nullptr);
  EXPECT_FLOAT_EQ(q.Get().bwFloatBlockEncoding.floatScaleOffset[0].scale, 0.5f);
  EXPECT_FLOAT_EQ(q.Get().bwFloatBlockEncoding.floatScaleOffset[0].offset, 0.0f);
  EXPECT_FLOAT_EQ(q.Get().bwFloatBlockEncoding.floatScaleOffset[1].scale, 0.25f);
  EXPECT_FLOAT_EQ(q.Get().bwFloatBlockEncoding.floatScaleOffset[1].offset, 0.1f);
}

// =============================================================================
// Copy / move semantics — drives both code paths in the copy ctor / operator=
// (the IsLPBQ() and IsBlockQuantized() branches).
// =============================================================================

TEST(QnnUnit_QuantParamsWrapperTest, CopyCtor_PerTensor_DeepCopies) {
  QnnQuantParamsWrapper src = QnnQuantParamsWrapper::PerTensor(0.7f, 4);
  QnnQuantParamsWrapper dst(src);
  EXPECT_TRUE(dst.IsPerTensor());
  EXPECT_FLOAT_EQ(dst.Get().scaleOffsetEncoding.scale, 0.7f);
  EXPECT_EQ(dst.Get().scaleOffsetEncoding.offset, 4);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyCtor_PerTensorBw_CopiesAllFields) {
  QnnQuantParamsWrapper src = QnnQuantParamsWrapper::PerTensorBw(0.125f, -5, /*bitwidth=*/4);
  QnnQuantParamsWrapper dst(src);
  EXPECT_TRUE(dst.IsPerTensor(/*include_bw=*/true));
  EXPECT_EQ(dst.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET);
  EXPECT_EQ(dst.Get().bwScaleOffsetEncoding.bitwidth, 4u);
  EXPECT_FLOAT_EQ(dst.Get().bwScaleOffsetEncoding.scale, 0.125f);
  EXPECT_EQ(dst.Get().bwScaleOffsetEncoding.offset, -5);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyCtor_PerChannelInt4_DeepCopies) {
  QnnQuantParamsWrapper src = MakePerChannelInt4(/*axis=*/2);
  const auto* src_scales_ptr = src.Get().bwAxisScaleOffsetEncoding.scales;
  QnnQuantParamsWrapper dst(src);
  // Pointers must differ — deep copy, not aliasing.
  EXPECT_NE(dst.Get().bwAxisScaleOffsetEncoding.scales, src_scales_ptr);
  EXPECT_EQ(dst.Get().bwAxisScaleOffsetEncoding.numElements, 3u);
  EXPECT_FLOAT_EQ(dst.Get().bwAxisScaleOffsetEncoding.scales[0], 0.1f);
  EXPECT_EQ(dst.Get().bwAxisScaleOffsetEncoding.axis, 2);
  EXPECT_EQ(dst.Get().bwAxisScaleOffsetEncoding.bitwidth, 4u);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyCtor_LPBQ_DeepCopies) {
  QnnQuantParamsWrapper src = MakeLPBQ();
  const auto* src_lpbq = src.Get().blockwiseExpansion;
  QnnQuantParamsWrapper dst(src);
  ASSERT_NE(dst.Get().blockwiseExpansion, nullptr);
  EXPECT_NE(dst.Get().blockwiseExpansion, src_lpbq);
  EXPECT_EQ(dst.Get().blockwiseExpansion->axis, 1);
  EXPECT_EQ(dst.Get().blockwiseExpansion->numBlocksPerAxis, 2u);
  EXPECT_FLOAT_EQ(dst.Get().blockwiseExpansion->scaleOffsets[1].scale, 0.2f);
  EXPECT_EQ(dst.Get().blockwiseExpansion->blocksScale8[5], 6u);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyCtor_BlockEncoding_DeepCopies) {
  QnnQuantParamsWrapper src = MakeBlockEncoding();
  const uint32_t* src_block_size = src.Get().blockEncoding.blockSize;
  QnnQuantParamsWrapper dst(src);
  EXPECT_NE(dst.Get().blockEncoding.blockSize, src_block_size);
  EXPECT_EQ(dst.Get().blockEncoding.blockSize[0], 2u);
  EXPECT_FLOAT_EQ(dst.Get().blockEncoding.scaleOffset[0].scale, 0.5f);
  EXPECT_FLOAT_EQ(dst.Get().blockEncoding.scaleOffset[1].scale, 0.25f);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyAssign_PerTensor_DeepCopies) {
  QnnQuantParamsWrapper src = QnnQuantParamsWrapper::PerTensor(1.5f, -2);
  QnnQuantParamsWrapper dst;
  dst = src;
  EXPECT_TRUE(dst.IsPerTensor());
  EXPECT_FLOAT_EQ(dst.Get().scaleOffsetEncoding.scale, 1.5f);
  EXPECT_EQ(dst.Get().scaleOffsetEncoding.offset, -2);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyAssign_PerTensorBw_CopiesAllFields) {
  QnnQuantParamsWrapper src = QnnQuantParamsWrapper::PerTensorBw(0.0625f, 3, /*bitwidth=*/4);
  QnnQuantParamsWrapper dst;
  dst = src;
  EXPECT_TRUE(dst.IsPerTensor(/*include_bw=*/true));
  EXPECT_EQ(dst.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET);
  EXPECT_EQ(dst.Get().bwScaleOffsetEncoding.bitwidth, 4u);
  EXPECT_FLOAT_EQ(dst.Get().bwScaleOffsetEncoding.scale, 0.0625f);
  EXPECT_EQ(dst.Get().bwScaleOffsetEncoding.offset, 3);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyAssign_LPBQ_DeepCopies) {
  // Drives the IsLPBQ() branch inside operator=.
  QnnQuantParamsWrapper src = MakeLPBQ();
  QnnQuantParamsWrapper dst;
  dst = src;
  EXPECT_TRUE(dst.IsLPBQ());
  ASSERT_NE(dst.Get().blockwiseExpansion, nullptr);
  EXPECT_NE(dst.Get().blockwiseExpansion, src.Get().blockwiseExpansion);
  EXPECT_EQ(dst.Get().blockwiseExpansion->numBlocksPerAxis, 2u);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyAssign_BlockEncoding_DeepCopies) {
  // Drives the IsBlockQuantized() branch inside operator=.
  QnnQuantParamsWrapper src = MakeBlockEncoding();
  QnnQuantParamsWrapper dst;
  dst = src;
  EXPECT_TRUE(dst.IsBlockQuantized());
  EXPECT_NE(dst.Get().blockEncoding.blockSize, src.Get().blockEncoding.blockSize);
  EXPECT_FLOAT_EQ(dst.Get().blockEncoding.scaleOffset[0].scale, 0.5f);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyCtor_BwFloatBlock_DeepCopies) {
  QnnQuantParamsWrapper src = MakeBwFloatBlock();
  const auto* src_ptr = src.Get().bwFloatBlockEncoding.floatScaleOffset;
  QnnQuantParamsWrapper dst(src);
  EXPECT_TRUE(dst.IsBlockQuantized());
  EXPECT_EQ(dst.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK);
  EXPECT_NE(dst.Get().bwFloatBlockEncoding.floatScaleOffset, src_ptr);
  EXPECT_EQ(dst.Get().bwFloatBlockEncoding.bitwidth, 8u);
  EXPECT_FLOAT_EQ(dst.Get().bwFloatBlockEncoding.floatScaleOffset[0].scale, 0.5f);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyAssign_BwFloatBlock_DeepCopies) {
  QnnQuantParamsWrapper src = MakeBwFloatBlock();
  QnnQuantParamsWrapper dst;
  dst = src;
  EXPECT_TRUE(dst.IsBlockQuantized());
  EXPECT_EQ(dst.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK);
  EXPECT_NE(dst.Get().bwFloatBlockEncoding.floatScaleOffset, src.Get().bwFloatBlockEncoding.floatScaleOffset);
  EXPECT_EQ(dst.Get().bwFloatBlockEncoding.bitwidth, 8u);
  EXPECT_FLOAT_EQ(dst.Get().bwFloatBlockEncoding.floatScaleOffset[1].offset, 0.1f);
}

TEST(QnnUnit_QuantParamsWrapperTest, CopyAssign_SelfAssignment_LeavesUnchanged) {
  QnnQuantParamsWrapper q = MakePerChannelNonInt4();
  const auto* original_scaleoffset_ptr = q.Get().axisScaleOffsetEncoding.scaleOffset;
  q = q;  // NOLINT(clang-diagnostic-self-assign-overloaded)
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.numScaleOffsets, 3u);
  // Pointer should be unchanged (self-assign early-out).
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset, original_scaleoffset_ptr);
}

TEST(QnnUnit_QuantParamsWrapperTest, Copy_Method_ReturnsEquivalentDeepCopy) {
  QnnQuantParamsWrapper src = MakePerChannelNonInt4(/*axis=*/3);
  QnnQuantParamsWrapper dst = src.Copy();
  EXPECT_NE(dst.Get().axisScaleOffsetEncoding.scaleOffset,
            src.Get().axisScaleOffsetEncoding.scaleOffset);
  EXPECT_EQ(dst.Get().axisScaleOffsetEncoding.axis, 3);
  EXPECT_EQ(dst.Get().axisScaleOffsetEncoding.numScaleOffsets, 3u);
}

TEST(QnnUnit_QuantParamsWrapperTest, MoveCtor_TransfersOwnership) {
  QnnQuantParamsWrapper src = MakePerChannelNonInt4();
  const auto* src_ptr = src.Get().axisScaleOffsetEncoding.scaleOffset;
  QnnQuantParamsWrapper dst(std::move(src));
  // Moved-to object owns the buffer.
  EXPECT_EQ(dst.Get().axisScaleOffsetEncoding.scaleOffset, src_ptr);
  EXPECT_EQ(dst.Get().axisScaleOffsetEncoding.numScaleOffsets, 3u);
}

// =============================================================================
// Predicates
// =============================================================================

TEST(QnnUnit_QuantParamsWrapperTest, IsPerTensor_IncludeBwTrue_AcceptsBwScaleOffset) {
  // Construct a BW_SCALE_OFFSET encoding via Init from raw.
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  raw.bwScaleOffsetEncoding.bitwidth = 4;
  raw.bwScaleOffsetEncoding.scale = 0.5f;
  raw.bwScaleOffsetEncoding.offset = 1;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_TRUE(q.IsPerTensor(/*include_bw=*/true));
}

TEST(QnnUnit_QuantParamsWrapperTest, IsPerTensor_IncludeBwFalse_RejectsBwScaleOffset) {
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_FALSE(q.IsPerTensor(/*include_bw=*/false));
}

TEST(QnnUnit_QuantParamsWrapperTest, IsPerChannel_AcceptsBothAxisAndBwAxisEncodings) {
  QnnQuantParamsWrapper non_int4 = MakePerChannelNonInt4();
  QnnQuantParamsWrapper int4 = MakePerChannelInt4();
  EXPECT_TRUE(non_int4.IsPerChannel());
  EXPECT_TRUE(int4.IsPerChannel());
}

TEST(QnnUnit_QuantParamsWrapperTest, IsLPBQ_OnlyForBlockwiseExpansion) {
  QnnQuantParamsWrapper lpbq = MakeLPBQ();
  QnnQuantParamsWrapper per_channel = MakePerChannelNonInt4();
  EXPECT_TRUE(lpbq.IsLPBQ());
  EXPECT_FALSE(per_channel.IsLPBQ());
}

TEST(QnnUnit_QuantParamsWrapperTest, IsBlockQuantized_AcceptsBlockAndBwFloatBlock) {
  QnnQuantParamsWrapper block = MakeBlockEncoding();
  QnnQuantParamsWrapper bw_float_block = MakeBwFloatBlock();
  QnnQuantParamsWrapper per_tensor = QnnQuantParamsWrapper::PerTensor(0.5f, 0);
  EXPECT_TRUE(block.IsBlockQuantized());
  EXPECT_TRUE(bw_float_block.IsBlockQuantized());
  EXPECT_FALSE(per_tensor.IsBlockQuantized());
}

TEST(QnnUnit_QuantParamsWrapperTest, Predicates_FalseWhenEncodingDefinitionUndefined) {
  QnnQuantParamsWrapper q;  // default = UNDEFINED.
  EXPECT_FALSE(q.IsQuantized());
  EXPECT_FALSE(q.IsPerTensor());
  EXPECT_FALSE(q.IsPerChannel());
  EXPECT_FALSE(q.IsLPBQ());
  EXPECT_FALSE(q.IsBlockQuantized());
}

// =============================================================================
// GetScales — covers all 5 encoding cases plus the default-error path.
// =============================================================================

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_Undefined_ReturnsError) {
  QnnQuantParamsWrapper q;
  std::vector<float> out;
  Ort::Status status = q.GetScales(out);
  EXPECT_FALSE(status.IsOK());
}

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_ScaleOffset_ReturnsSingleScalar) {
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerTensor(0.25f, 0);
  std::vector<float> out;
  ASSERT_TRUE(q.GetScales(out).IsOK());
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0], 0.25f);
}

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_BwScaleOffset_ReturnsSingleScalar) {
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  raw.bwScaleOffsetEncoding.bitwidth = 4;
  raw.bwScaleOffsetEncoding.scale = 0.125f;
  raw.bwScaleOffsetEncoding.offset = 0;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  std::vector<float> out;
  ASSERT_TRUE(q.GetScales(out).IsOK());
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0], 0.125f);
}

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_AxisScaleOffset_ReturnsAllScales) {
  QnnQuantParamsWrapper q = MakePerChannelNonInt4();
  std::vector<float> out;
  ASSERT_TRUE(q.GetScales(out).IsOK());
  ASSERT_EQ(out.size(), 3u);
  EXPECT_FLOAT_EQ(out[0], 0.1f);
  EXPECT_FLOAT_EQ(out[1], 0.2f);
  EXPECT_FLOAT_EQ(out[2], 0.3f);
}

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_AxisScaleOffset_NumElemsZero_ReturnsEmpty) {
  const std::vector<float> scales{};
  const std::vector<int32_t> offsets{};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannel(gsl::make_span(scales), gsl::make_span(offsets), 0);
  std::vector<float> out{1.0f, 2.0f};  // Pre-existing content should be cleared.
  ASSERT_TRUE(q.GetScales(out).IsOK());
  EXPECT_TRUE(out.empty());
}

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_BwAxisScaleOffset_ReturnsAllScales) {
  QnnQuantParamsWrapper q = MakePerChannelInt4();
  std::vector<float> out;
  ASSERT_TRUE(q.GetScales(out).IsOK());
  ASSERT_EQ(out.size(), 3u);
  EXPECT_FLOAT_EQ(out[0], 0.1f);
  EXPECT_FLOAT_EQ(out[2], 0.3f);
}

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_BwAxisScaleOffset_NumElemsZero_ReturnsEmpty) {
  const std::vector<float> scales{};
  const std::vector<int32_t> offsets{};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannelBw(gsl::make_span(scales), gsl::make_span(offsets), 0, /*bitwidth=*/4);
  std::vector<float> out{1.0f};
  ASSERT_TRUE(q.GetScales(out).IsOK());
  EXPECT_TRUE(out.empty());
}

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_BlockEncoding_ReturnsAllBlockScales) {
  QnnQuantParamsWrapper q = MakeBlockEncoding();
  std::vector<float> out;
  ASSERT_TRUE(q.GetScales(out).IsOK());
  ASSERT_EQ(out.size(), 2u);
  EXPECT_FLOAT_EQ(out[0], 0.5f);
  EXPECT_FLOAT_EQ(out[1], 0.25f);
}

TEST(QnnUnit_QuantParamsWrapperTest, GetScales_UnsupportedEncoding_ReturnsError) {
  // BLOCKWISE_EXPANSION is intentionally NOT a case in GetScales' switch.
  QnnQuantParamsWrapper q = MakeLPBQ();
  std::vector<float> out;
  EXPECT_FALSE(q.GetScales(out).IsOK());
}

// =============================================================================
// Init(Qnn_QuantizeParams_t, num_scaleoffsets, tensor_rank) — 7 encoding cases
// + UNDEFINED early-return + unsupported-encoding error + reset-when-prior-init.
// =============================================================================

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_Undefined_CopiesParamsOnly) {
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_FALSE(q.IsQuantized());
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_ScaleOffset_CopiesShallow) {
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  raw.scaleOffsetEncoding.scale = 0.875f;
  raw.scaleOffsetEncoding.offset = 7;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_TRUE(q.IsPerTensor());
  EXPECT_FLOAT_EQ(q.Get().scaleOffsetEncoding.scale, 0.875f);
  EXPECT_EQ(q.Get().scaleOffsetEncoding.offset, 7);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_BwScaleOffset_CopiesShallow) {
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  raw.bwScaleOffsetEncoding.bitwidth = 8;
  raw.bwScaleOffsetEncoding.scale = 0.0625f;
  raw.bwScaleOffsetEncoding.offset = -1;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET);
  EXPECT_EQ(q.Get().bwScaleOffsetEncoding.bitwidth, 8u);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_AxisScaleOffset_NonEmpty_DeepCopiesScaleOffset) {
  std::vector<Qnn_ScaleOffset_t> src_data(2);
  src_data[0].scale = 0.1f;
  src_data[0].offset = 1;
  src_data[1].scale = 0.2f;
  src_data[1].offset = 2;

  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  raw.axisScaleOffsetEncoding.axis = 1;
  raw.axisScaleOffsetEncoding.numScaleOffsets = 2;
  raw.axisScaleOffsetEncoding.scaleOffset = src_data.data();

  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  // Pointer should differ — deep copy.
  EXPECT_NE(q.Get().axisScaleOffsetEncoding.scaleOffset, src_data.data());
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.numScaleOffsets, 2u);
  EXPECT_FLOAT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset[1].scale, 0.2f);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_AxisScaleOffset_Empty_SetsNullScaleOffset) {
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  raw.axisScaleOffsetEncoding.axis = 0;
  raw.axisScaleOffsetEncoding.numScaleOffsets = 0;
  raw.axisScaleOffsetEncoding.scaleOffset = nullptr;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset, nullptr);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_BwAxisScaleOffset_NonEmpty_DeepCopiesArrays) {
  std::vector<float> src_scales{0.1f, 0.2f};
  std::vector<int32_t> src_offsets{3, 4};

  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
  raw.bwAxisScaleOffsetEncoding.axis = 1;
  raw.bwAxisScaleOffsetEncoding.bitwidth = 4;
  raw.bwAxisScaleOffsetEncoding.numElements = 2;
  raw.bwAxisScaleOffsetEncoding.scales = src_scales.data();
  raw.bwAxisScaleOffsetEncoding.offsets = src_offsets.data();

  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_NE(q.Get().bwAxisScaleOffsetEncoding.scales, src_scales.data());
  EXPECT_NE(q.Get().bwAxisScaleOffsetEncoding.offsets, src_offsets.data());
  EXPECT_FLOAT_EQ(q.Get().bwAxisScaleOffsetEncoding.scales[0], 0.1f);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.offsets[1], 4);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.bitwidth, 4u);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_BwAxisScaleOffset_Empty_SetsNullArrays) {
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
  raw.bwAxisScaleOffsetEncoding.axis = 0;
  raw.bwAxisScaleOffsetEncoding.numElements = 0;
  raw.bwAxisScaleOffsetEncoding.scales = nullptr;
  raw.bwAxisScaleOffsetEncoding.offsets = nullptr;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.scales, nullptr);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.offsets, nullptr);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     InitFromRaw_BlockwiseExpansion_DeepCopiesScaleOffsetsAndBlockScales) {
  // Re-init from an already-built LPBQ wrapper's params — exercises the
  // BLOCKWISE_EXPANSION case in Init().
  QnnQuantParamsWrapper src = MakeLPBQ();
  QnnQuantParamsWrapper dst;
  ASSERT_TRUE(dst.Init(src.Get(), /*num_scaleoffsets=*/3, /*tensor_rank=*/0).IsOK());
  EXPECT_TRUE(dst.IsLPBQ());
  EXPECT_NE(dst.Get().blockwiseExpansion, src.Get().blockwiseExpansion);
  EXPECT_EQ(dst.Get().blockwiseExpansion->numBlocksPerAxis, 2u);
  EXPECT_EQ(dst.Get().blockwiseExpansion->blocksScale8[0], 1u);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_BlockEncoding_DeepCopiesAxisAndScaleOffsets) {
  QnnQuantParamsWrapper src = MakeBlockEncoding();
  QnnQuantParamsWrapper dst;
  ASSERT_TRUE(dst.Init(src.Get(), /*num_scaleoffsets=*/2, /*tensor_rank=*/2).IsOK());
  EXPECT_TRUE(dst.IsBlockQuantized());
  EXPECT_EQ(dst.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BLOCK);
  EXPECT_NE(dst.Get().blockEncoding.blockSize, src.Get().blockEncoding.blockSize);
  EXPECT_EQ(dst.Get().blockEncoding.blockSize[0], 2u);
  EXPECT_FLOAT_EQ(dst.Get().blockEncoding.scaleOffset[1].scale, 0.25f);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     InitFromRaw_BwFloatBlockEncoding_DeepCopiesAxisAndFloatScaleOffsets) {
  QnnQuantParamsWrapper src = MakeBwFloatBlock();
  QnnQuantParamsWrapper dst;
  ASSERT_TRUE(dst.Init(src.Get(), /*num_scaleoffsets=*/2, /*tensor_rank=*/2).IsOK());
  EXPECT_EQ(dst.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK);
  EXPECT_EQ(dst.Get().bwFloatBlockEncoding.bitwidth, 8u);
  EXPECT_FLOAT_EQ(dst.Get().bwFloatBlockEncoding.floatScaleOffset[1].offset, 0.1f);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_UnsupportedEncoding_ReturnsError) {
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_VECTOR;  // not in the switch.
  QnnQuantParamsWrapper q;
  EXPECT_FALSE(q.Init(raw).IsOK());
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromRaw_AfterPriorPerChannelInit_ResetsThenRecopies) {
  // Start in per-channel state so per_channel_data_ is non-null.
  QnnQuantParamsWrapper q = MakePerChannelNonInt4();
  ASSERT_NE(q.Get().axisScaleOffsetEncoding.scaleOffset, nullptr);

  // Re-init with a per-tensor encoding; per_channel_data_ must be reset.
  Qnn_QuantizeParams_t raw = QNN_QUANTIZE_PARAMS_INIT;
  raw.encodingDefinition = QNN_DEFINITION_DEFINED;
  raw.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  raw.scaleOffsetEncoding.scale = 0.99f;
  raw.scaleOffsetEncoding.offset = 0;
  ASSERT_TRUE(q.Init(raw).IsOK());
  EXPECT_TRUE(q.IsPerTensor());
  EXPECT_FLOAT_EQ(q.Get().scaleOffsetEncoding.scale, 0.99f);
}

// =============================================================================
// Init(QnnModelWrapper, OrtNodeUnitIODef) — uses mock_init_registry.
// =============================================================================

namespace {

// Build a wrapper + register a per-tensor scale/zp pair.
struct PerTensorIODefFixture {
  OrtApiStubContext ctx;
  QNN_INTERFACE_VER_TYPE qnn_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle = nullptr;
  QNN_INTERFACE_VER_TYPE qnn_validator_interface = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t validator_backend_handle = nullptr;
  Ort::Logger null_logger_{MakeNullLogger()};
  int fake_graph_sentinel_{};
  qnn::GraphInputOutputInfo input_info;
  qnn::GraphInputOutputInfo output_info;
  std::unique_ptr<qnn::QnnModelWrapper> wrapper;

  PerTensorIODefFixture() {
    g_mock_init_reg.clear();
    SetupMockInitRegistryStubs(ctx);
    ApiPtrs api_ptrs = ctx.MakeApiPtrs();
    const OrtGraph& fake_graph = *reinterpret_cast<const OrtGraph*>(&fake_graph_sentinel_);
    wrapper = std::make_unique<qnn::QnnModelWrapper>(
        fake_graph, api_ptrs, null_logger_,
        qnn_interface, backend_handle,
        qnn_validator_interface, validator_backend_handle,
        input_info, output_info,
        qnn::QnnBackendType::HTP, qnn::ModelSettings{});
  }
};

OrtNodeUnitIODef MakeIODef(const std::string& name,
                           ONNXTensorElementDataType type,
                           std::vector<int64_t> shape,
                           const OrtValueInfo* scale_vi,
                           const OrtValueInfo* zp_vi,
                           std::optional<int64_t> axis = std::nullopt) {
  OrtNodeUnitIODef io_def;
  io_def.name = name;
  io_def.type = type;
  io_def.shape = std::optional<std::vector<int64_t>>(std::move(shape));
  io_def.quant_param = OrtNodeUnitIODef::QuantParam{scale_vi, zp_vi, axis};
  return io_def;
}

}  // namespace

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_NoQuantParam_SetsUndefined) {
  PerTensorIODefFixture fx;
  OrtNodeUnitIODef io_def;
  io_def.name = "in";
  io_def.type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  io_def.shape = std::optional<std::vector<int64_t>>{{1, 4}};
  io_def.quant_param = std::nullopt;
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_FALSE(q.IsQuantized());
  EXPECT_EQ(q.Get().encodingDefinition, QNN_DEFINITION_UNDEFINED);
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_UNDEFINED);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerTensor_NotInt4_SetsScaleOffsetEncoding) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {}, {0.1f});
  auto zp_vi = g_mock_init_reg.AddTensorUint8("zp", {}, {static_cast<uint8_t>(5)});
  auto io_def = MakeIODef("in", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 3, 4, 4}, scale_vi, zp_vi);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_FLOAT_EQ(q.Get().scaleOffsetEncoding.scale, 0.1f);
  EXPECT_EQ(q.Get().scaleOffsetEncoding.offset, -5);  // QNN flips the sign.
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerTensor_NotInt4_NoZeroPoint_SetsZeroOffset) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {}, {0.1f});
  auto io_def = MakeIODef("in", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4},
                          scale_vi, /*zp_vi=*/nullptr);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_EQ(q.Get().scaleOffsetEncoding.offset, 0);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerTensor_Int4_SetsBwScaleOffsetEncoding) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {}, {0.0625f});
  auto zp_vi = g_mock_init_reg.AddTensorInt4As8bit("zp", {}, {static_cast<int8_t>(0)});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {1, 4}, scale_vi, zp_vi);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET);
  EXPECT_EQ(q.Get().bwScaleOffsetEncoding.bitwidth, 4u);
  EXPECT_FLOAT_EQ(q.Get().bwScaleOffsetEncoding.scale, 0.0625f);
  EXPECT_EQ(q.Get().bwScaleOffsetEncoding.offset, 0);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerTensor_Int4_NoZeroPoint_SetsZeroOffset) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {}, {0.0625f});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {1, 4},
                          scale_vi, /*zp_vi=*/nullptr);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  // Without a zp VI, Init() cannot determine the bitwidth from the IODef alone
  // and treats the input as non-int4 (falls into the SCALE_OFFSET path). This is
  // the intended behavior: is_int4_type is derived from the presence of an int4
  // zp tensor, not from the ONNX element type alone.
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_SCALE_OFFSET);
  EXPECT_EQ(q.Get().scaleOffsetEncoding.offset, 0);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     InitFromIODef_PerChannel_NotInt4_SetsAxisScaleOffsetEncoding) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {3}, {0.1f, 0.2f, 0.3f});
  auto zp_vi = g_mock_init_reg.AddTensorUint8("zp", {3}, {1, 2, 3});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 3, 4, 4},
                          scale_vi, zp_vi, /*axis=*/1);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.axis, 1);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.numScaleOffsets, 3u);
  EXPECT_FLOAT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset[0].scale, 0.1f);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset[2].offset, -3);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     InitFromIODef_PerChannel_NotInt4_NoZeroPoint_FillsOffsetsZero) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {3}, {0.1f, 0.2f, 0.3f});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 3, 4, 4},
                          scale_vi, /*zp_vi=*/nullptr, /*axis=*/1);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset[0].offset, 0);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset[1].offset, 0);
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.scaleOffset[2].offset, 0);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerChannel_Int4_SetsBwAxisScaleOffsetEncoding) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {3}, {0.0625f, 0.125f, 0.25f});
  auto zp_vi = g_mock_init_reg.AddTensorInt4As8bit("zp", {3}, {0, 0, 0});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {1, 3, 4, 4},
                          scale_vi, zp_vi, /*axis=*/1);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.bitwidth, 4u);
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.numElements, 3u);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     InitFromIODef_PerChannel_NoZeroPoint_FillsOffsetsZero_Int4) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {3}, {0.0625f, 0.125f, 0.25f});
  // No zp_vi → is_int4_type = false → falls into the non-int4 per-channel branch.
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {1, 3, 4, 4},
                          scale_vi, /*zp_vi=*/nullptr, /*axis=*/1);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().quantizationEncoding, QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerChannel_NegativeAxis_NormalisesToPositive) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {3}, {0.1f, 0.2f, 0.3f});
  auto zp_vi = g_mock_init_reg.AddTensorUint8("zp", {3}, {0, 0, 0});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 3, 4, 4},
                          scale_vi, zp_vi, /*axis=*/-3);  // -3 + rank 4 = 1
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.axis, 1);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerChannelInt4_NegativeAxis_NormalisesToPositive) {
  // Per-channel INT4 path has its own negative-axis normalisation block
  // separate from the non-int4 path. Both must be covered.
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {3}, {0.0625f, 0.125f, 0.25f});
  auto zp_vi = g_mock_init_reg.AddTensorInt4As8bit("zp", {3}, {0, 0, 0});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {1, 3, 4, 4},
                          scale_vi, zp_vi, /*axis=*/-3);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def).IsOK());
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.axis, 1);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerChannel_AxisOutOfRange_ReturnsError) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {3}, {0.1f, 0.2f, 0.3f});
  auto zp_vi = g_mock_init_reg.AddTensorUint8("zp", {3}, {0, 0, 0});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 3},
                          scale_vi, zp_vi, /*axis=*/5);  // rank = 2
  QnnQuantParamsWrapper q;
  EXPECT_FALSE(q.Init(*fx.wrapper, io_def).IsOK());
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerChannel_ScaleZpSizeMismatch_ReturnsError) {
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {3}, {0.1f, 0.2f, 0.3f});
  auto zp_vi = g_mock_init_reg.AddTensorUint8("zp", {2}, {0, 0});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 3, 4, 4},
                          scale_vi, zp_vi, /*axis=*/1);
  QnnQuantParamsWrapper q;
  EXPECT_FALSE(q.Init(*fx.wrapper, io_def).IsOK());
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_AfterPriorPerChannelInit_ResetsState) {
  // Drives the `if (per_channel_data_) { reset(); params_ = INIT; }` head of
  // Init(io_def): first call leaves per_channel_data_ non-null, second call
  // must reset it before re-initialising.
  PerTensorIODefFixture fx;

  auto pc_scale = g_mock_init_reg.AddTensorFloat("pc_scale", {3}, {0.1f, 0.2f, 0.3f});
  auto pc_zp = g_mock_init_reg.AddTensorUint8("pc_zp", {3}, {0, 0, 0});
  auto pt_scale = g_mock_init_reg.AddTensorFloat("pt_scale", {}, {0.5f});
  auto pt_zp = g_mock_init_reg.AddTensorUint8("pt_zp", {}, {static_cast<uint8_t>(7)});

  auto io_def_pc = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 3, 4, 4},
                             pc_scale, pc_zp, /*axis=*/1);
  auto io_def_pt = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, {1, 4},
                             pt_scale, pt_zp, std::nullopt);
  QnnQuantParamsWrapper q;
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def_pc).IsOK());
  ASSERT_TRUE(q.IsPerChannel());
  ASSERT_TRUE(q.Init(*fx.wrapper, io_def_pt).IsOK());
  EXPECT_TRUE(q.IsPerTensor());
  EXPECT_FLOAT_EQ(q.Get().scaleOffsetEncoding.scale, 0.5f);
  EXPECT_EQ(q.Get().scaleOffsetEncoding.offset, -7);
}

TEST(QnnUnit_QuantParamsWrapperTest, InitFromIODef_PerTensor_Int4_MultiZp_ReturnsError) {
  // Per-tensor INT4 path expects exactly one zero-point. A multi-element INT4
  // zp tensor with a scalar scale exercises the `zero_points.size() == 1`
  // guard inside the BW_SCALE_OFFSET branch.
  PerTensorIODefFixture fx;
  auto scale_vi = g_mock_init_reg.AddTensorFloat("scale", {}, {0.0625f});
  auto zp_vi = g_mock_init_reg.AddTensorInt4As8bit("zp", {2}, {0, 0});
  auto io_def = MakeIODef("w", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, {1, 4}, scale_vi, zp_vi);
  QnnQuantParamsWrapper q;
  EXPECT_FALSE(q.Init(*fx.wrapper, io_def).IsOK());
}

// =============================================================================
// Templated handlers — header-defined, NOT counted toward .cc coverage but
// tested for behavioural completeness.
//
// Use the SAME type instantiations as production: `uint32_t` for both helpers
// (matmul_op_builder.cc, conv_op_builder.cc, instancenormalization, etc.) plus
// `size_t` for HandleTranspose (conv_op_builder.cc:233 — perm_inv is size_t).
// gcov tracks each template instantiation as a separate function, so testing
// with `int32_t` / `int64_t` would not register hits for production-instantiated
// types.
//
// HandleUnsqueeze's "unhandled encoding" inner branch is unreachable from the
// public API (the IsPerChannel() early-return prevents any non-AXIS encoding
// from getting that far); we do not test that dead-on-public-surface branch.
// =============================================================================

TEST(QnnUnit_QuantParamsWrapperTest, HandleTranspose_NotPerChannel_NoOp) {
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerTensor(0.5f, 0);
  std::vector<uint32_t> perm{0, 1, 2, 3};
  EXPECT_TRUE(q.HandleTranspose<uint32_t>(gsl::make_span(perm)).IsOK());
}

TEST(QnnUnit_QuantParamsWrapperTest, HandleTranspose_AxisScaleOffset_RemapsAxis) {
  QnnQuantParamsWrapper q = MakePerChannelNonInt4(/*axis=*/1);
  std::vector<uint32_t> perm{0, 2, 3, 1};  // moves dim-1 to position 3
  ASSERT_TRUE(q.HandleTranspose<uint32_t>(gsl::make_span(perm)).IsOK());
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.axis, 2);
}

TEST(QnnUnit_QuantParamsWrapperTest, HandleTranspose_BwAxisScaleOffset_RemapsAxis) {
  QnnQuantParamsWrapper q = MakePerChannelInt4(/*axis=*/2);
  std::vector<uint32_t> perm{0, 1, 3, 2};
  ASSERT_TRUE(q.HandleTranspose<uint32_t>(gsl::make_span(perm)).IsOK());
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.axis, 3);
}

TEST(QnnUnit_QuantParamsWrapperTest, HandleTranspose_AxisOutOfRange_ReturnsError) {
  QnnQuantParamsWrapper q = MakePerChannelNonInt4(/*axis=*/5);
  std::vector<uint32_t> perm{0, 1, 2};  // perm.size() == 3, axis = 5 → OOR
  EXPECT_FALSE(q.HandleTranspose<uint32_t>(gsl::make_span(perm)).IsOK());
}

// Mirror conv_op_builder.cc:233 — production instantiates HandleTranspose with
// size_t (perm_inv is a vector<size_t> there). Each instantiation is a separate
// function in gcov, so we test it explicitly.
TEST(QnnUnit_QuantParamsWrapperTest, HandleTranspose_SizeT_RemapsAxis) {
  QnnQuantParamsWrapper q = MakePerChannelNonInt4(/*axis=*/1);
  std::vector<size_t> perm{0, 2, 3, 1};
  ASSERT_TRUE(q.HandleTranspose<size_t>(gsl::make_span(perm)).IsOK());
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.axis, 2);
}

TEST(QnnUnit_QuantParamsWrapperTest, HandleUnsqueeze_NotPerChannel_NoOp) {
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerTensor(0.5f, 0);
  std::vector<uint32_t> orig{4, 5};
  std::vector<uint32_t> nu{1, 4, 5};
  EXPECT_TRUE(q.HandleUnsqueeze<uint32_t>(gsl::make_span(orig), gsl::make_span(nu)).IsOK());
}

TEST(QnnUnit_QuantParamsWrapperTest, HandleUnsqueeze_RankNotIncreased_ReturnsError) {
  QnnQuantParamsWrapper q = MakePerChannelNonInt4(/*axis=*/0);
  std::vector<uint32_t> orig{3, 4};
  std::vector<uint32_t> nu{3, 4};  // same size — error
  EXPECT_FALSE(q.HandleUnsqueeze<uint32_t>(gsl::make_span(orig), gsl::make_span(nu)).IsOK());
}

TEST(QnnUnit_QuantParamsWrapperTest,
     HandleUnsqueeze_AxisScaleOffset_ShiftsWhenOnesInsertedBefore) {
  // Per-channel along axis 1 of a {3, 4} tensor; unsqueeze to {1, 3, 4} should
  // move the per-channel axis from 1 to 2.
  const std::vector<float> scales{0.1f, 0.2f, 0.3f};
  const std::vector<int32_t> offsets{0, 0, 0};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannel(gsl::make_span(scales), gsl::make_span(offsets), /*axis=*/1);
  std::vector<uint32_t> orig{3, 4};
  std::vector<uint32_t> nu{1, 3, 4};
  ASSERT_TRUE(q.HandleUnsqueeze<uint32_t>(gsl::make_span(orig), gsl::make_span(nu)).IsOK());
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.axis, 2);
}

TEST(QnnUnit_QuantParamsWrapperTest,
     HandleUnsqueeze_BwAxisScaleOffset_ShiftsWhenOnesInsertedBefore) {
  const std::vector<float> scales{0.1f, 0.2f, 0.3f};
  const std::vector<int32_t> offsets{0, 0, 0};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannelBw(gsl::make_span(scales), gsl::make_span(offsets), /*axis=*/0, /*bitwidth=*/4);
  std::vector<uint32_t> orig{3};
  std::vector<uint32_t> nu{1, 3};
  ASSERT_TRUE(q.HandleUnsqueeze<uint32_t>(gsl::make_span(orig), gsl::make_span(nu)).IsOK());
  EXPECT_EQ(q.Get().bwAxisScaleOffsetEncoding.axis, 1);
}

TEST(QnnUnit_QuantParamsWrapperTest, HandleUnsqueeze_NoShiftNeeded_LeavesAxisUnchanged) {
  // axis=0; new shape {3, 1, 4} appends 1 *after* the per-channel axis ⇒ axis stays at 0.
  const std::vector<float> scales{0.1f, 0.2f, 0.3f};
  const std::vector<int32_t> offsets{0, 0, 0};
  QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerChannel(gsl::make_span(scales), gsl::make_span(offsets), /*axis=*/0);
  std::vector<uint32_t> orig{3, 4};
  std::vector<uint32_t> nu{3, 1, 4};
  ASSERT_TRUE(q.HandleUnsqueeze<uint32_t>(gsl::make_span(orig), gsl::make_span(nu)).IsOK());
  EXPECT_EQ(q.Get().axisScaleOffsetEncoding.axis, 0);
}

// `Get() const` overload — only reachable from a const-context call site.
// All other tests use a non-const wrapper and therefore call the non-const
// overload (line 49 in the header). Without this test the const-Get line
// would be header coverage's only genuine gap.
TEST(QnnUnit_QuantParamsWrapperTest, ConstGet_ReturnsSameParamsAsNonConst) {
  const QnnQuantParamsWrapper q = QnnQuantParamsWrapper::PerTensor(0.5f, -2);
  EXPECT_FLOAT_EQ(q.Get().scaleOffsetEncoding.scale, 0.5f);
  EXPECT_EQ(q.Get().scaleOffsetEncoding.offset, -2);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
