// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "onnxruntime_c_api.h"
#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

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
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
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
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
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
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
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
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
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
                  EPVerificationParams{ExpectedEPNodeAssignment::All});
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#endif  // !defined(ORT_MINIMAL_BUILD)
