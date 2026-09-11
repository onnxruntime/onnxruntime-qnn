// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

// Declared in test_main.cc.
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

enum class IndexElementType { kInt64,
                              kInt32 };

// [1,3,8,8] tiled strided-slice + concat stem (any NCHW input with even H/W
// shares the pattern); concat_order selects the Concat input sequence.
// multiaxis_slices emits the parallel form (4 dual-axis Slices off the input,
// as in production exports); otherwise the cascaded form (2 H-slices feeding
// 4 single-axis W-slices).
template <typename QuantType = uint16_t>
GetTestModelFn BuildTiledSliceConcatTestCase(bool use_qdq, bool use_contrib_qdq,
                                             std::vector<std::string> concat_order = {"h0w0", "h1w0", "h0w1", "h1w1"},
                                             IndexElementType index_type = IndexElementType::kInt64,
                                             bool mismatched_scales = false,
                                             bool multiaxis_slices = false) {
  return [=](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("tiled_slice_concat_graph");
    const std::vector<int64_t> input_shape{1, 3, 8, 8};
    const auto input_def = TestInputDef<float>(input_shape, false, -1.0f, 1.0f);
    MakeTestInput<float>(builder, "image", input_def);

    std::string stem_in = "image";
    QuantParams<QuantType> stem_quant{0.05f, 0};
    if (use_qdq) {
      stem_quant = GetTestInputQuantParams<QuantType>(input_def);
      stem_in = AddQDQNodePair<QuantType>(builder, "qdq_in", "image", stem_quant.scale,
                                          stem_quant.zero_point, use_contrib_qdq);
    }

    auto add_slice = [&](const std::string& name, const std::string& data, int64_t axis,
                         int64_t start, int64_t end, const std::string& output) {
      if (index_type == IndexElementType::kInt64) {
        builder.Make1DInitializer<int64_t>(name + "_starts", {start});
        builder.Make1DInitializer<int64_t>(name + "_ends", {end});
        builder.Make1DInitializer<int64_t>(name + "_axes", {axis});
        builder.Make1DInitializer<int64_t>(name + "_steps", {2});
      } else {
        builder.Make1DInitializer<int32_t>(name + "_starts", {static_cast<int32_t>(start)});
        builder.Make1DInitializer<int32_t>(name + "_ends", {static_cast<int32_t>(end)});
        builder.Make1DInitializer<int32_t>(name + "_axes", {static_cast<int32_t>(axis)});
        builder.Make1DInitializer<int32_t>(name + "_steps", {2});
      }
      builder.AddNode(name, "Slice",
                      {data, name + "_starts", name + "_ends", name + "_axes", name + "_steps"},
                      {output}, kOnnxDomain);
    };

    if (multiaxis_slices) {
      // Parallel form: each tile cut from the input in one dual-axis Slice.
      auto add_tile = [&](const std::string& name, int64_t h_start, int64_t w_start,
                          const std::string& output) {
        if (index_type == IndexElementType::kInt64) {
          builder.Make1DInitializer<int64_t>(name + "_starts", {0, 0, h_start, w_start});
          builder.Make1DInitializer<int64_t>(name + "_ends", {1, 3, 8, 8});
          builder.Make1DInitializer<int64_t>(name + "_axes", {0, 1, 2, 3});
          builder.Make1DInitializer<int64_t>(name + "_steps", {1, 1, 2, 2});
        } else {
          builder.Make1DInitializer<int32_t>(name + "_starts", {0, 0, static_cast<int32_t>(h_start),
                                                                static_cast<int32_t>(w_start)});
          builder.Make1DInitializer<int32_t>(name + "_ends", {1, 3, 8, 8});
          builder.Make1DInitializer<int32_t>(name + "_axes", {0, 1, 2, 3});
          builder.Make1DInitializer<int32_t>(name + "_steps", {1, 1, 2, 2});
        }
        builder.AddNode(name, "Slice",
                        {stem_in, name + "_starts", name + "_ends", name + "_axes", name + "_steps"},
                        {output}, kOnnxDomain);
      };
      add_tile("SliceH0W0", 0, 0, "h0w0");
      add_tile("SliceH1W0", 1, 0, "h1w0");
      add_tile("SliceH0W1", 0, 1, "h0w1");
      add_tile("SliceH1W1", 1, 1, "h1w1");
    } else {
      add_slice("SliceH0", stem_in, 2, 0, 8, "h0");
      add_slice("SliceH1", stem_in, 2, 1, 8, "h1");
      add_slice("SliceH0W0", "h0", 3, 0, 8, "h0w0");
      add_slice("SliceH1W0", "h1", 3, 0, 8, "h1w0");
      add_slice("SliceH0W1", "h0", 3, 1, 8, "h0w1");
      add_slice("SliceH1W1", "h1", 3, 1, 8, "h1w1");
    }

    // Optional intermediate requant with a different scale on one branch.
    // Fusion must fail closed here; S2D would skip the requant error.
    std::vector<std::string> concat_inputs = concat_order;
    if (mismatched_scales) {
      concat_inputs.back() = AddQDQNodePair<QuantType>(builder, "qdq_mismatch", concat_inputs.back(),
                                                       stem_quant.scale * 2.0f, stem_quant.zero_point,
                                                       use_contrib_qdq);
    }
    builder.AddNode("Concat", "Concat", concat_inputs, {"cat_out"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(1))});

    std::string tail_in = "cat_out";
    if (use_qdq) {
      tail_in = AddQDQNodePair<QuantType>(builder, "qdq_out", "cat_out", stem_quant.scale,
                                          stem_quant.zero_point, use_contrib_qdq);
    }
    builder.MakeInitializer<float>("stem_w", {12, 12, 1, 1}, -1.f, 1.f);
    builder.AddNode("StemConv", "Conv", {tail_in, "stem_w"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  };
}

ProviderOptions HtpOptions() {
  ProviderOptions options;
  options["backend_type"] = "htp";
  options["offload_graph_io_quantization"] = "0";
  return options;
}

void RunTiledSliceConcatFusionTest(const std::filesystem::path& dir, GetTestModelFn model_fn, int expect_s2d,
                                   int expect_gather, float tolerance = 1e-2f) {
  std::filesystem::remove_all(dir);
  ASSERT_TRUE(std::filesystem::create_directory(dir));
  auto cleanup = gsl::finally([&dir]() { std::filesystem::remove_all(dir); });

  ProviderOptions options = HtpOptions();
  options["dump_json_qnn_graph"] = "1";
  options["json_qnn_graph_dir"] = dir.string();

  RunQnnModelTest(model_fn, options, /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(tolerance)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);
  AssertOpInQnnGraph(dir, "SpaceToDepth", expect_s2d);
  AssertOpInQnnGraph(dir, "Gather", expect_gather);
}

}  // namespace

// Non-canonical phase order: S2D(DCR) + channel Gather restoring exact order.
TEST_F(QnnHTPBackendTests, TiledSliceConcat_QDQ_U16_Fused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatQDQU16_HTP", BuildTiledSliceConcatTestCase(true, true), 1, 1, 3e-2f);
}

TEST_F(QnnHTPBackendTests, TiledSliceConcat_QDQ_U16_Int32Indices_Fused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatQDQU16Int32_HTP",
                                BuildTiledSliceConcatTestCase(true, true, {"h0w0", "h1w0", "h0w1", "h1w1"},
                                                              IndexElementType::kInt32),
                                1, 1, 3e-2f);
}

// Canonical DCR order needs no reorder: bare S2D, no Gather.
TEST_F(QnnHTPBackendTests, TiledSliceConcat_QDQ_CanonicalOrder_FusedWithoutGather) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatQDQCanonical_HTP",
                                BuildTiledSliceConcatTestCase(true, true, {"h0w0", "h0w1", "h1w0", "h1w1"}), 1, 0,
                                3e-2f);
}

// Parallel tiling (production export form): same lowering and numerics.
TEST_F(QnnHTPBackendTests, TiledSliceConcat_Parallel_QDQ_U16_Fused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatParallelQDQU16_HTP",
                                BuildTiledSliceConcatTestCase(true, true, {"h0w0", "h1w0", "h0w1", "h1w1"},
                                                              IndexElementType::kInt64, false, true),
                                1, 1, 3e-2f);
}

TEST_F(QnnHTPBackendTests, TiledSliceConcat_Parallel_QDQ_CanonicalOrder_FusedWithoutGather) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatParallelQDQCanonical_HTP",
                                BuildTiledSliceConcatTestCase(true, true, {"h0w0", "h0w1", "h1w0", "h1w1"},
                                                              IndexElementType::kInt64, false, true),
                                1, 0, 3e-2f);
}

// W8A8 production quantization (uint8, standard QDQ): same fusion, same numerics.
TEST_F(QnnHTPBackendTests, TiledSliceConcat_QDQ_U8_Fused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatQDQU8_HTP",
                                BuildTiledSliceConcatTestCase<uint8_t>(
                                    true, false, {"h0w0", "h1w0", "h0w1", "h1w1"}),
                                1, 1, 3.9e-2f);
}

// Second non-canonical order exercises a different Gather permutation.
TEST_F(QnnHTPBackendTests, TiledSliceConcat_QDQ_ReversedOrder_Fused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatQDQReversed_HTP",
                                BuildTiledSliceConcatTestCase(true, true, {"h1w1", "h0w1", "h1w0", "h0w0"}), 1, 1,
                                3e-2f);
}

// Float S2D-DCR is inaccurate on HTP (AISW-175353): fail closed until fixed.
TEST_F(QnnHTPBackendTests, TiledSliceConcat_Float_NotFused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatFloat_HTP", BuildTiledSliceConcatTestCase(false, false), 0, 0);
}

// Duplicated phase must fail closed (S2D would drop a block).
TEST_F(QnnHTPBackendTests, TiledSliceConcat_DuplicatePhase_NotFused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatDuplicate_HTP",
                                BuildTiledSliceConcatTestCase(true, true, {"h0w0", "h0w0", "h1w0", "h1w1"}), 0, 0,
                                3e-2f);
}

TEST_F(QnnHTPBackendTests, TiledSliceConcat_MismatchedScales_NotFused) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  RunTiledSliceConcatFusionTest("TiledSliceConcatMismatch_HTP",
                                BuildTiledSliceConcatTestCase(true, true, {"h0w0", "h1w0", "h0w1", "h1w1"},
                                                              IndexElementType::kInt64, /*mismatched_scales=*/true),
                                0, 0, 3e-2f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
