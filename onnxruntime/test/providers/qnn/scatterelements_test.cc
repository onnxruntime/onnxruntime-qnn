// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "onnxruntime_c_api.h"
#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

// HTP backend is only enumerable on NPU or Linux x86_64 simulator; excludes
// Windows x86_64, matching the guard in simple_op_test.cc / scatternd_test.cc.
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

TEST_F(QnnHTPBackendTests, ScatterElementsNegativeIndexDefaultAxis) {
  constexpr int64_t kLen = 8;

  auto build_model = [=](ModelTestBuilder& builder) {
    // `data` must be a runtime input -- otherwise ORT constant-folds
    // ScatterElements away before the QNN EP sees it.
    std::vector<int32_t> data(kLen, 0);
    builder.MakeInput<int32_t>("data", {kLen}, data);

    std::vector<int64_t> indices = {-1};  // -> kLen - 1 after normalize
    builder.MakeInitializer<int64_t>("indices", {1}, indices);

    std::vector<int32_t> updates = {42};
    builder.MakeInitializer<int32_t>("updates", {1}, updates);

    builder.AddNode("scatter", "ScatterElements", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain);
    builder.AddNode("cast", "Cast", {"scatter_out"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("to",
                                         static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("Y");
  };

  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ScatterElementsNegativeIndexNonDefaultAxis) {
  constexpr int64_t kRows = 3;
  constexpr int64_t kCols = 5;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<int32_t> data(kRows * kCols, 0);
    builder.MakeInput<int32_t>("data", {kRows, kCols}, data);

    std::vector<int64_t> indices = {-1, -kCols, 2, 0, 1, 3};  // shape [3, 2]
    builder.MakeInitializer<int64_t>("indices", {kRows, 2}, indices);

    std::vector<int32_t> updates = {11, 12, 13, 14, 15, 16};
    builder.MakeInitializer<int32_t>("updates", {kRows, 2}, updates);

    builder.AddNode("scatter", "ScatterElements", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(1))});
    builder.AddNode("cast", "Cast", {"scatter_out"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("to",
                                         static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("Y");
  };

  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All);
}

// Negative axis attribute (count-from-end) must resolve before bounds check.
TEST_F(QnnHTPBackendTests, ScatterElementsNegativeIndexNegativeAxis) {
  constexpr int64_t kRows = 2;
  constexpr int64_t kCols = 6;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<int32_t> data(kRows * kCols, 0);
    builder.MakeInput<int32_t>("data", {kRows, kCols}, data);

    // axis=-1 == axis=1; index -2 -> kCols-2.
    std::vector<int64_t> indices = {-2, 0};
    builder.MakeInitializer<int64_t>("indices", {kRows, 1}, indices);

    std::vector<int32_t> updates = {7, 9};
    builder.MakeInitializer<int32_t>("updates", {kRows, 1}, updates);

    builder.AddNode("scatter", "ScatterElements", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(-1))});
    builder.AddNode("cast", "Cast", {"scatter_out"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("to",
                                         static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))});
    builder.MakeOutput("Y");
  };

  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All);
}

// ScatterElements(-1) embedded between producer/consumer ops must compile
// through QNN finalization without CPU fallback.
TEST_F(QnnHTPBackendTests, ScatterElementsEndToEndNegativeIndexInGraph) {
  constexpr int64_t kRows = 2;
  constexpr int64_t kCols = 32;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<float> data(kRows * kCols);
    for (int64_t i = 0; i < kRows * kCols; ++i) {
      data[i] = static_cast<float>(i) * 0.01f;
    }
    builder.MakeInput<float>("data_src", {kRows, kCols}, data);

    std::vector<float> bias(kRows * kCols, 0.5f);
    builder.MakeInitializer<float>("bias", {kRows, kCols}, bias);
    builder.AddNode("pre_add", "Add", {"data_src", "bias"}, {"data"}, kOnnxDomain);

    std::vector<int64_t> indices = {-1, -2, 0, 1};
    builder.MakeInitializer<int64_t>("indices", {kRows, 2}, indices);

    std::vector<float> updates = {10.0f, 20.0f, 30.0f, 40.0f};
    builder.MakeInitializer<float>("updates", {kRows, 2}, updates);

    builder.AddNode("scatter", "ScatterElements", {"data", "indices", "updates"},
                    {"scatter_out"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(-1))});

    std::vector<float> scale(kRows * kCols, 2.0f);
    builder.MakeInitializer<float>("scale", {kRows, kCols}, scale);
    builder.AddNode("post_mul", "Mul", {"scatter_out", "scale"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  };

  // HTP fp16 path -- loosen tolerance vs. CPU fp32 reference.
  RunQnnModelTest(build_model, MakeHtpProviderOptions(), 17,
                  ExpectedEPNodeAssignment::All, 1e-2f);
}

// Same axis bound -- rewritten bytes are identical; both nodes land on QNN.
TEST_F(QnnHTPBackendTests, ScatterElementsSharedNegativeIndicesInitializer) {
  constexpr int64_t kLen = 12;

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<int32_t> data_a(kLen, 0);
    std::vector<int32_t> data_b(kLen, 0);
    builder.MakeInput<int32_t>("dataA", {kLen}, data_a);
    builder.MakeInput<int32_t>("dataB", {kLen}, data_b);

    std::vector<int64_t> indices = {-1};
    builder.MakeInitializer<int64_t>("indices", {1}, indices);

    std::vector<int32_t> updates_a = {11};
    std::vector<int32_t> updates_b = {22};
    builder.MakeInitializer<int32_t>("updatesA", {1}, updates_a);
    builder.MakeInitializer<int32_t>("updatesB", {1}, updates_b);

    builder.AddNode("scatterA", "ScatterElements", {"dataA", "indices", "updatesA"},
                    {"outA_i32"}, kOnnxDomain);
    builder.AddNode("scatterB", "ScatterElements", {"dataB", "indices", "updatesB"},
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

// Different axis bounds -- indices `[-1]` resolves to `kRows-1` for scatterA
// and `kCols-1` for scatterB. The `_qnn_idx` rename prevents the two
// per-axis rewrites from aliasing.
TEST_F(QnnHTPBackendTests, ScatterElementsSharedNegativeIndicesDifferentAxes) {
  constexpr int64_t kRows = 3;
  constexpr int64_t kCols = 7;  // != kRows so rewritten bytes differ.

  auto build_model = [=](ModelTestBuilder& builder) {
    std::vector<int32_t> data_a(kRows * kCols, 0);
    std::vector<int32_t> data_b(kRows * kCols, 0);
    builder.MakeInput<int32_t>("dataA", {kRows, kCols}, data_a);
    builder.MakeInput<int32_t>("dataB", {kRows, kCols}, data_b);

    std::vector<int64_t> indices = {-1, -1, -1};
    builder.MakeInitializer<int64_t>("indices", {kRows, 1}, indices);

    std::vector<int32_t> updates_a = {10, 20, 30};
    std::vector<int32_t> updates_b = {40, 50, 60};
    builder.MakeInitializer<int32_t>("updatesA", {kRows, 1}, updates_a);
    builder.MakeInitializer<int32_t>("updatesB", {kRows, 1}, updates_b);

    builder.AddNode("scatterA", "ScatterElements", {"dataA", "indices", "updatesA"},
                    {"outA_i32"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(0))});
    builder.AddNode("scatterB", "ScatterElements", {"dataB", "indices", "updatesB"},
                    {"outB_i32"}, kOnnxDomain,
                    {test::MakeAttribute("axis", static_cast<int64_t>(1))});
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
