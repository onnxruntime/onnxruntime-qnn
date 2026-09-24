// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64)

constexpr int QBits = 4;

/**
 * Quantize float data into block-wise INT4 (packed into uint8).
 * NOTE:
 *  - QuantizeBlockwise expects a flat packed buffer.
 *  - DO NOT dequantize here.
 */
void QuantizeBlockwiseOnly(
    const std::vector<float>& raw_vals,
    std::vector<uint8_t>& quant_vals,
    std::vector<float>& scales,
    int32_t rows,
    int32_t cols,
    int32_t block_size,
    int32_t quantize_axis) {
  QuantizeBlockwise<float, QBits>(
      quant_vals.data(),
      scales.data(),
      nullptr,  // zero_point (symmetric int4)
      raw_vals.data(),
      block_size,
      quantize_axis == 1,  // column-wise quantization
      rows,
      cols,
      cols);
}

/**
 * Quantize float data into block-wise INT4, producing already-unpacked int4 elements
 * (rows x cols logical shape, Int4x2 packed storage). This mimics a weight that arrives
 * pre-quantized as int4 rather than packed as UInt4x2 in a uint8 buffer -- i.e. the
 * "weight_shape[1] already is the element count" case (no *2 for packed nibbles).
 * Only quantize_axis == 1 (column-wise) is supported on QNN GPU, matching QuantizeBlockwiseOnly.
 */
void QuantizeBlockwiseInt4Unpacked(
    const std::vector<float>& raw_vals,
    std::vector<Int4x2>& quant_vals,
    std::vector<float>& scales,
    int32_t rows,
    int32_t cols,
    int32_t block_size) {
  TestInputDef<float> weight_def({rows, cols}, true, raw_vals);
  std::vector<Int4x2> zero_points;  // symmetric quant: zero-points resolve to 0
  GetTestInputQuantParamsBlockQuant<Int4x2>(weight_def, scales, zero_points, block_size,
                                            /*axis=*/1, /*symmetric=*/true);
  QuantizeValuesBlockQuant<float, Int4x2>(raw_vals, quant_vals, {rows, cols}, scales, zero_points,
                                          block_size, /*axis=*/1);
}

struct GatherBQTestParams {
  int64_t vocab_size;
  int64_t hidden_size;
  int64_t batch_size{1};
  int64_t seq_len{1};
  int64_t block_size{32};
  int64_t gather_axis{0};
  int64_t quantize_axis{1};
};

// When weight_is_unpacked_int4 is true, the weight input arrives already-unpacked as INT4
// (Int4x2-typed initializer with logical shape [vocab, hidden]) rather than a UInt4x2-packed
// uint8 buffer with shape [vocab, hidden * bits / 8]. The unpacked form exercises the
// ProcessInputs path where weight_type != QNN_DATATYPE_UINT_8, so weight_shape[1] is already
// the unpacked element count and must NOT be doubled when checked against scale_shape[1] *
// block_size.
static void RunGatherBlockQuantizedTest(
    const GatherBQTestParams& params,
    ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
    const std::string& backend_name = "gpu",
    float fp32_abs_err = 0.05f,
    bool weight_is_unpacked_int4 = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  auto model_builder = [&params, weight_is_unpacked_int4](ModelTestBuilder& builder) {
    RandomValueGenerator random{1234};

    // ------------------------------------------------------------
    // Block-quant storage
    // ------------------------------------------------------------
    const int64_t blocks_per_row =
        params.hidden_size / params.block_size;

    const size_t scale_count =
        static_cast<size_t>(params.vocab_size * blocks_per_row);

    // ------------------------------------------------------------
    // Float reference weights [vocab, hidden]
    // ------------------------------------------------------------
    std::vector<float> weight_f_vals(
        random.Gaussian<float>(
            AsSpan({params.vocab_size, params.hidden_size}),
            0.0f,
            0.25f));
    std::vector<float> scales(scale_count);

    // ------------------------------------------------------------
    // Weight input: either UInt4x2-packed uint8 or already-unpacked Int4x2.
    //
    // NOTE: Int4x2 has no Ort::TypeToTensorType specialization, so it can only be used as a
    // graph initializer (via MakeInitializer directly), not as a dynamic input -- MakeTestInput's
    // MakeInput<T> branch is instantiated unconditionally and would fail to compile.
    // ------------------------------------------------------------
    if (weight_is_unpacked_int4) {
      std::vector<Int4x2> weight_q_vals;
      QuantizeBlockwiseInt4Unpacked(
          weight_f_vals,
          weight_q_vals,
          scales,
          static_cast<int32_t>(params.vocab_size),
          static_cast<int32_t>(params.hidden_size),
          static_cast<int32_t>(params.block_size));
      // Shape is [vocab, hidden] (the unpacked element count), not [vocab, hidden * bits / 8].
      builder.MakeInitializer<Int4x2>("data", {params.vocab_size, params.hidden_size}, weight_q_vals);
    } else {
      const size_t quant_bytes =
          static_cast<size_t>(params.vocab_size *
                              params.hidden_size * QBits / 8);
      std::vector<uint8_t> weight_q_vals(quant_bytes);
      QuantizeBlockwiseOnly(
          weight_f_vals,
          weight_q_vals,
          scales,
          static_cast<int32_t>(params.vocab_size),
          static_cast<int32_t>(params.hidden_size),
          static_cast<int32_t>(params.block_size),
          static_cast<int32_t>(params.quantize_axis));
      auto weight_def = TestInputDef<uint8_t>(
          {params.vocab_size,
           params.hidden_size * QBits / 8},
          true,
          weight_q_vals);
      MakeTestInput<uint8_t>(builder, "data", weight_def);
    }

    // ------------------------------------------------------------
    // Indices
    // ------------------------------------------------------------
    int64_t num_indices = params.batch_size * params.seq_len;
    std::vector<int64_t> indices(num_indices);
    for (int64_t i = 0; i < num_indices; ++i) {
      indices[i] = i % params.vocab_size;
    }

    // ------------------------------------------------------------
    // Graph inputs
    // ------------------------------------------------------------
    auto indices_def =
        TestInputDef<int64_t>({params.batch_size, params.seq_len}, false, indices);
    MakeTestInput<int64_t>(builder, "indices", indices_def);

    auto scales_def =
        TestInputDef<float>({params.vocab_size, blocks_per_row},
                            true,
                            scales);
    MakeTestInput<float>(builder, "scales", scales_def);

    builder.MakeOutput("output");

    // ------------------------------------------------------------
    // Attributes
    // ------------------------------------------------------------
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(
        builder.MakeScalarAttribute("bits", static_cast<int64_t>(QBits)));
    attributes.push_back(
        builder.MakeScalarAttribute("block_size", params.block_size));
    attributes.push_back(
        builder.MakeScalarAttribute("gather_axis", params.gather_axis));
    attributes.push_back(
        builder.MakeScalarAttribute("quantize_axis", params.quantize_axis));

    builder.AddNode(
        "gather_block_quantize",
        "GatherBlockQuantized",
        {"data", "indices", "scales"},
        {"output"},
        kMSDomain,
        attributes);
  };

  RunQnnModelTest(
      model_builder,
      provider_options,
      13,  // opset
      EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}

// =============================================================
// Test cases
// =============================================================

TEST_F(QnnGPUBackendTests, GatherBlockQuantized_Basic) {
  GatherBQTestParams params;
  params.vocab_size = 128;
  params.hidden_size = 64;
  params.batch_size = 1;
  params.seq_len = 1;
  params.block_size = 32;
  params.gather_axis = 0;
  params.quantize_axis = 1;
  RunGatherBlockQuantizedTest(params);
}

TEST_F(QnnGPUBackendTests, GatherBlockQuantized_LargerIndices) {
  GatherBQTestParams params;
  params.vocab_size = 4096;
  params.hidden_size = 512;
  params.batch_size = 1;
  params.seq_len = 64;
  params.block_size = 32;
  params.gather_axis = 0;
  params.quantize_axis = 1;
  RunGatherBlockQuantizedTest(params);
}

// Weight arrives already-unpacked as INT4 (Int4x2 element type, logical shape == element
// count), rather than packed UInt4x2-in-uint8. Regression test for the weight_shape[1] * 2
// mismatch: weight_shape[1] must not be doubled when the weight isn't the packed uint8 form.
TEST_F(QnnGPUBackendTests, GatherBlockQuantized_UnpackedInt4Weight) {
  GatherBQTestParams params;
  params.vocab_size = 128;
  params.hidden_size = 64;
  params.batch_size = 1;
  params.seq_len = 1;
  params.block_size = 32;
  params.gather_axis = 0;
  params.quantize_axis = 1;
  RunGatherBlockQuantizedTest(params, ExpectedEPNodeAssignment::All, "gpu", 0.05f,
                              /*weight_is_unpacked_int4=*/true);
}

// Same shapes as the reported failure: hidden_size=3072, block_size=32 (96 blocks/row),
// weight already-unpacked as INT4.
TEST_F(QnnGPUBackendTests, GatherBlockQuantized_UnpackedInt4Weight_LargeHidden) {
  GatherBQTestParams params;
  params.vocab_size = 256;
  params.hidden_size = 3072;
  params.batch_size = 1;
  params.seq_len = 8;
  params.block_size = 32;
  params.gather_axis = 0;
  params.quantize_axis = 1;
  RunGatherBlockQuantizedTest(params, ExpectedEPNodeAssignment::All, "gpu", 0.05f,
                              /*weight_is_unpacked_int4=*/true);
}

// Regression for vision embedding tables whose packed data and scales are rank 3.
// data [2, 10240, 384] encodes logical INT4 weights [2, 10240, 768]; scales are
// [2, 10240, 24]. Quantization is along the last axis and gathering is along axis 0.
TEST_F(QnnGPUBackendTests, GatherBlockQuantized_Rank3_LastAxis) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";
  provider_options["offload_graph_io_quantization"] = "0";

  auto model_builder = [](ModelTestBuilder& builder) {
    constexpr int64_t kOuter = 2;
    constexpr int64_t kVocab = 10240;
    constexpr int64_t kLogicalHidden = 768;
    constexpr int64_t kPackedHidden = kLogicalHidden * QBits / 8;
    constexpr int64_t kBlockSize = 32;
    constexpr int64_t kBlocks = kLogicalHidden / kBlockSize;
    RandomValueGenerator random{1234};
    std::vector<float> weights(random.Gaussian<float>(
        AsSpan({kOuter, kVocab, kLogicalHidden}), 0.0f, 0.25f));
    std::vector<uint8_t> packed_weights(static_cast<size_t>(kOuter * kVocab * kPackedHidden));
    std::vector<float> scales(static_cast<size_t>(kOuter * kVocab * kBlocks));
    QuantizeBlockwiseOnly(weights, packed_weights, scales,
                          static_cast<int32_t>(kOuter * kVocab),
                          static_cast<int32_t>(kLogicalHidden),
                          static_cast<int32_t>(kBlockSize), 1);

    MakeTestInput<uint8_t>(builder, "data",
                           TestInputDef<uint8_t>({kOuter, kVocab, kPackedHidden}, true, packed_weights));
    MakeTestInput<int64_t>(builder, "indices", TestInputDef<int64_t>({1}, false, {0}));
    MakeTestInput<float>(builder, "scales", TestInputDef<float>({kOuter, kVocab, kBlocks}, true, scales));
    builder.MakeOutput("output");

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("bits", static_cast<int64_t>(QBits)));
    attributes.push_back(builder.MakeScalarAttribute("block_size", kBlockSize));
    attributes.push_back(builder.MakeScalarAttribute("gather_axis", static_cast<int64_t>(0)));
    attributes.push_back(builder.MakeScalarAttribute("quantize_axis", static_cast<int64_t>(2)));
    builder.AddNode("gather_block_quantize", "GatherBlockQuantized",
                    {"data", "indices", "scales"}, {"output"}, kMSDomain, attributes);
  };

  RunQnnModelTest(model_builder, provider_options, 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.05f)});
}
// A scalar ONNX index gathers the first rank-3 table slice and produces a rank-2
// output. QNN Gather needs a rank-1-or-greater index, so the provider must lower
// this to a one-element index and reshape the temporary result back to rank 2.
TEST_F(QnnGPUBackendTests, GatherBlockQuantized_ScalarIndex_Rank2Output) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";
  provider_options["offload_graph_io_quantization"] = "0";

  auto model_builder = [](ModelTestBuilder& builder) {
    constexpr int64_t kTables = 2;
    constexpr int64_t kVocab = 128;
    constexpr int64_t kLogicalHidden = 64;
    constexpr int64_t kPackedHidden = kLogicalHidden * QBits / 8;
    constexpr int64_t kBlockSize = 32;
    constexpr int64_t kBlocks = kLogicalHidden / kBlockSize;
    RandomValueGenerator random{1234};
    std::vector<float> weights(random.Gaussian<float>(
        AsSpan({kTables, kVocab, kLogicalHidden}), 0.0f, 0.25f));
    std::vector<uint8_t> packed_weights(static_cast<size_t>(kTables * kVocab * kPackedHidden));
    std::vector<float> scales(static_cast<size_t>(kTables * kVocab * kBlocks));
    QuantizeBlockwiseOnly(weights, packed_weights, scales,
                          static_cast<int32_t>(kTables * kVocab),
                          static_cast<int32_t>(kLogicalHidden),
                          static_cast<int32_t>(kBlockSize), 1);

    MakeTestInput<uint8_t>(builder, "data",
                           TestInputDef<uint8_t>({kTables, kVocab, kPackedHidden}, true, packed_weights));
    MakeTestInput<int64_t>(builder, "indices", TestInputDef<int64_t>({}, true, {0}));
    MakeTestInput<float>(builder, "scales", TestInputDef<float>({kTables, kVocab, kBlocks}, true, scales));
    builder.MakeOutput("output");

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("bits", static_cast<int64_t>(QBits)));
    attributes.push_back(builder.MakeScalarAttribute("block_size", kBlockSize));
    attributes.push_back(builder.MakeScalarAttribute("gather_axis", static_cast<int64_t>(0)));
    attributes.push_back(builder.MakeScalarAttribute("quantize_axis", static_cast<int64_t>(2)));
    builder.AddNode("gather_block_quantize", "GatherBlockQuantized",
                    {"data", "indices", "scales"}, {"output"}, kMSDomain, attributes);
  };

  RunQnnModelTest(model_builder, provider_options, 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.05f)});
}
// Negative test: GatherBlockQuantized is GPU-only. On the CPU backend the op
// must be rejected by IsOpSupported and fall back (no node assigned to QNN EP).
TEST_F(QnnCPUBackendTests, GatherBlockQuantized_CpuBackendNotSupported) {
  GatherBQTestParams params;
  params.vocab_size = 128;
  params.hidden_size = 64;
  params.batch_size = 1;
  params.seq_len = 1;
  params.block_size = 32;
  params.gather_axis = 0;
  params.quantize_axis = 1;
  RunGatherBlockQuantizedTest(params, ExpectedEPNodeAssignment::None, "cpu");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !ORT_MINIMAL_BUILD
