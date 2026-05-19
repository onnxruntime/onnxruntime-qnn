// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "onnxruntime_c_api.h"
#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace onnxruntime {
namespace test {

namespace {

ProviderOptions MakeHtpProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

}  // namespace

// Trailing Cast keeps scatter_out non-graph-output.
TEST_F(QnnHTPBackendTests, ScatterNDInternalOutputNegativeIndex) {
  constexpr int64_t kRows = 1;
  constexpr int64_t kCols = 50;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<int32_t> data(kRows * kCols, 0);
    builder.MakeInitializer<int32_t>("data", {kRows, kCols}, data);

    std::vector<int64_t> indices = {0, -1};  // -> [0, kCols-1] after normalize
    builder.MakeInitializer<int64_t>("indices", {1, 1, 2}, indices);

    std::vector<int32_t> updates = {42};
    builder.MakeInitializer<int32_t>("updates", {1, 1}, updates);

    builder.AddNode("scatter", "ScatterND", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain);
    builder.AddNode("cast", "Cast", {"scatter_out"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("to",
                                         static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("Y");
  };

  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All);
}

// Exercises the column-indexed axis_dim lookup.
TEST_F(QnnHTPBackendTests, ScatterNDMultipleNegativeIndicesAcrossColumns) {
  constexpr int64_t kDim0 = 4;
  constexpr int64_t kDim1 = 6;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<int32_t> data(kDim0 * kDim1, 0);
    builder.MakeInitializer<int32_t>("data", {kDim0, kDim1}, data);

    // Two index tuples, both containing negative values across columns.
    // Tuple 0: (-1, -1) -> (kDim0-1, kDim1-1)
    // Tuple 1: (-kDim0, -kDim1) -> (0, 0)
    std::vector<int64_t> indices = {-1, -1, -kDim0, -kDim1};
    builder.MakeInitializer<int64_t>("indices", {2, 2}, indices);

    std::vector<int32_t> updates = {99, 77};
    builder.MakeInitializer<int32_t>("updates", {2}, updates);

    builder.AddNode("scatter", "ScatterND", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain);
    builder.AddNode("cast", "Cast", {"scatter_out"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("to",
                                         static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("Y");
  };

  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ScatterNDReductionAddWithNegativeIndices) {
  constexpr int64_t kRows = 1;
  constexpr int64_t kCols = 8;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<int32_t> data(kRows * kCols, 0);
    builder.MakeInitializer<int32_t>("data", {kRows, kCols}, data);

    std::vector<int64_t> indices = {0, -2};  // -> [0, kCols-2]
    builder.MakeInitializer<int64_t>("indices", {1, 1, 2}, indices);

    std::vector<int32_t> updates = {5};
    builder.MakeInitializer<int32_t>("updates", {1, 1}, updates);

    builder.AddNode("scatter", "ScatterND", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain,
                    {test::MakeAttribute("reduction", std::string("add"))});
    builder.AddNode("cast", "Cast", {"scatter_out"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("to",
                                         static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("Y");
  };

  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All);
}

// ScatterND(-1) between producer/consumer ops must compile through QNN finalization.
TEST_F(QnnHTPBackendTests, ScatterNDEndToEndNegativeIndexInGraph) {
  constexpr int64_t kRows = 2;
  constexpr int64_t kCols = 64;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<float> data(kRows * kCols);
    for (int64_t i = 0; i < kRows * kCols; ++i) {
      data[i] = static_cast<float>(i) * 0.01f;
    }
    builder.MakeInitializer<float>("data_src", {kRows, kCols}, data);

    std::vector<float> bias(kRows * kCols, 0.5f);
    builder.MakeInitializer<float>("bias", {kRows, kCols}, bias);
    builder.AddNode("pre_add", "Add", {"data_src", "bias"}, {"data"}, kOnnxDomain);

    // Negative indices across both tuple columns.
    std::vector<int64_t> indices = {0, -1, 1, -2, -kRows, 0};
    builder.MakeInitializer<int64_t>("indices", {3, 2}, indices);

    std::vector<float> updates = {10.0f, 20.0f, 30.0f};
    builder.MakeInitializer<float>("updates", {3}, updates);

    builder.AddNode("scatter", "ScatterND", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain);

    std::vector<float> scale(kRows * kCols, 2.0f);
    builder.MakeInitializer<float>("scale", {kRows, kCols}, scale);
    builder.AddNode("post_mul", "Mul", {"scatter_out", "scale"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  };

  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All);
}

// TSM-style rectangular slice-assignment: PyTorch
// `out[..., t_lo:t_hi, c_lo:c_hi] = src` lowers (via aten::copy_ ->
// index_put_) to a ScatterND whose `indices` is the Cartesian product of
// contiguous ranges — i.e. a slice-assignment, not a real scatter.
//
// On rank-5 FP16, the HTP host compiler (QAIRT 2.43.x and 2.45.40) selects
// the disabled `q::ScatterNd.tcm` kernel and aborts graph_prepare with
// exit code 15.  ScatterNDOpBuilder detects the rectangular-indices
// pattern and lowers to Slice + Concat instead, which uses the always-
// supported elementwise data-movement kernels.
//
// The test below WAS the bug repro: without the decomposition it fails
// during HTP graph_prepare; with it, it compiles and runs to numerical
// equivalence with CPU EP.
TEST_F(QnnHTPBackendTests, ScatterNDRectangleSliceAssignmentRank5Fp16) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  // (B, T, C, H, W) — TSM video tensor. Touch t in [1..7], c in [0..7].
  constexpr int64_t kB = 1;
  constexpr int64_t kT = 8;
  constexpr int64_t kC = 64;
  constexpr int64_t kH = 4;
  constexpr int64_t kW = 4;
  constexpr int64_t kTSel = 7;   // t in [1, 8)
  constexpr int64_t kCSel = 8;   // c in [0, 8)

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<float> data(kB * kT * kC * kH * kW);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(i % 17) * 0.01f;
    }
    builder.MakeInitializer<float>("data", {kB, kT, kC, kH, kW}, data);

    // Cartesian product: B={0}, T={1..7}, C={0..7}; tuple width K = 3.
    std::vector<int64_t> indices;
    indices.reserve(kB * kTSel * kCSel * 3);
    for (int64_t b = 0; b < kB; ++b) {
      for (int64_t t = 1; t < 1 + kTSel; ++t) {
        for (int64_t c = 0; c < kCSel; ++c) {
          indices.push_back(b);
          indices.push_back(t);
          indices.push_back(c);
        }
      }
    }
    builder.MakeInitializer<int64_t>("indices", {kB, kTSel, kCSel, 3}, indices);

    std::vector<float> updates(kB * kTSel * kCSel * kH * kW);
    for (size_t i = 0; i < updates.size(); ++i) {
      updates[i] = 1.0f + static_cast<float>(i % 13) * 0.01f;
    }
    builder.MakeInitializer<float>("updates", {kB, kTSel, kCSel, kH, kW}, updates);

    builder.AddNode("scatter", "ScatterND", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain);
    // Trailing op so scatter_out is non-graph-output — exercises the
    // NATIVE path of the final concat too.
    std::vector<float> bias(kB * kT * kC * kH * kW, 0.0f);
    builder.MakeInitializer<float>("bias", {kB, kT, kC, kH, kW}, bias);
    builder.AddNode("post_add", "Add", {"scatter_out", "bias"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  };

  ProviderOptions provider_options = MakeHtpProviderOptions();
  // FP16 internal precision is what triggers the disabled-kernel path;
  // FP32 ScatterND uses a different HTP kernel that has always worked.
  provider_options["enable_htp_fp16_precision"] = "1";

  RunQnnModelTest(build_model, provider_options, 17,
                  ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-2f);  // FP16 rounding noise.
}

// Verifies the rename avoids collisions when a shared initializer is rewritten.
TEST_F(QnnHTPBackendTests, ScatterNDSharedNegativeIndicesInitializer) {
  constexpr int64_t kRows = 1;
  constexpr int64_t kCols = 16;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<int32_t> data_a(kRows * kCols, 0);
    std::vector<int32_t> data_b(kRows * kCols, 0);
    builder.MakeInitializer<int32_t>("dataA", {kRows, kCols}, data_a);
    builder.MakeInitializer<int32_t>("dataB", {kRows, kCols}, data_b);

    std::vector<int64_t> indices = {0, -1};
    builder.MakeInitializer<int64_t>("indices", {1, 1, 2}, indices);

    std::vector<int32_t> updates_a = {11};
    std::vector<int32_t> updates_b = {22};
    builder.MakeInitializer<int32_t>("updatesA", {1, 1}, updates_a);
    builder.MakeInitializer<int32_t>("updatesB", {1, 1}, updates_b);

    builder.AddNode("scatterA", "ScatterND", {"dataA", "indices", "updatesA"},
                    {"outA_i32"}, kOnnxDomain);
    builder.AddNode("scatterB", "ScatterND", {"dataB", "indices", "updatesB"},
                    {"outB_i32"}, kOnnxDomain);
    builder.AddNode("castA", "Cast", {"outA_i32"}, {"YA"}, kOnnxDomain,
                    {test::MakeAttribute("to",
                                         static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.AddNode("castB", "Cast", {"outB_i32"}, {"YB"}, kOnnxDomain,
                    {test::MakeAttribute("to",
                                         static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("YA");
    builder.MakeOutput("YB");
  };

  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#endif  // !defined(ORT_MINIMAL_BUILD)
