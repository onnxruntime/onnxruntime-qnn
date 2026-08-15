// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ai.onnx.ml domain string and opset version used in opset_import.
static constexpr const char* kAiOnnxMlDomain = "ai.onnx.ml";
static constexpr int kAiOnnxMlOpsetVersion = 3;

// Wraps a GetTestModelFn to inject the ai.onnx.ml opset import into the model proto.
// RunQnnModelTest only adds "" and kMSDomain; this wrapper adds "ai.onnx.ml" so that
// the ORT model loader can resolve ArrayFeatureExtractor.
static GetTestModelFn WrapWithMLDomain(GetTestModelFn inner_fn) {
  return [inner_fn](ModelTestBuilder& builder) {
    inner_fn(builder);
    auto* opset = builder.model_.add_opset_import();
    opset->set_domain(kAiOnnxMlDomain);
    opset->set_version(kAiOnnxMlOpsetVersion);
  };
}

// Runs an ArrayFeatureExtractor model on QNN HTP and compares outputs to CPU EP.
// opset_version applies to the default "" (ai.onnx) domain, not ai.onnx.ml.
// It must be >= 7 for ORT's layout transformer to allow QNN EP participation;
// the ai.onnx.ml opset is handled separately by WrapWithMLDomain.
template <typename DataType, typename IndicesType = int64_t>
static void RunArrayFeatureExtractorTest(
    const TestInputDef<DataType>& x_def,
    const TestInputDef<IndicesType>& y_def,
    ExpectedEPNodeAssignment expected_ep_assignment,
    float abs_err = 1e-3f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  auto model_fn = BuildOpTestCase<DataType, IndicesType>(
      "afe_node",
      "ArrayFeatureExtractor",
      {x_def},
      {y_def},
      {},
      kAiOnnxMlDomain);

  RunQnnModelTest(WrapWithMLDomain(model_fn),
                  provider_options,
                  /*opset_version=*/9,
                  EPVerificationParams{expected_ep_assignment,
                                       ElementwiseAbsoluteVerifier(abs_err)});
}

// ---------------------------------------------------------------------------
// Float32 tests gated on real HTP hardware.
// ---------------------------------------------------------------------------
#if !defined(__linux__) || defined(__aarch64__)

// 2D float32: X[3, 4], Y = [0, 2] selects columns 0 and 2.
TEST_F(QnnHTPBackendTests, ArrayFeatureExtractor_Float32_2D_Basic) {
  RunArrayFeatureExtractorTest<float, int64_t>(
      TestInputDef<float>({3, 4}, false,
                          {1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f}),
      TestInputDef<int64_t>({2}, true, {0, 2}),
      ExpectedEPNodeAssignment::All);
}

// 3D float32: X[2, 3, 4], Y = [1, 3] selects elements at last-axis positions 1 and 3.
TEST_F(QnnHTPBackendTests, ArrayFeatureExtractor_Float32_3D) {
  std::vector<float> x_data;
  x_data.reserve(2 * 3 * 4);
  for (int i = 0; i < 2 * 3 * 4; ++i) {
    x_data.push_back(static_cast<float>(i) * 0.5f);
  }
  RunArrayFeatureExtractorTest<float, int64_t>(
      TestInputDef<float>({2, 3, 4}, false, x_data),
      TestInputDef<int64_t>({2}, true, {1, 3}),
      ExpectedEPNodeAssignment::All);
}

// Dynamic int64 indices: values are graph inputs, requiring a Cast int64→int32 node.
TEST_F(QnnHTPBackendTests, ArrayFeatureExtractor_Float32_DynamicInt64Indices) {
  RunArrayFeatureExtractorTest<float, int64_t>(
      TestInputDef<float>({4, 6}, false,
                          GetSequentialFloatData({4, 6}, 1.0f, 1.0f)),
      TestInputDef<int64_t>({3}, false, {0, 2, 5}),
      ExpectedEPNodeAssignment::All);
}

#endif  // !defined(__linux__) || defined(__aarch64__)

// ---------------------------------------------------------------------------
// Integer data tests
// ---------------------------------------------------------------------------

// 2D int32 X, int64 static indices.
TEST_F(QnnHTPBackendTests, ArrayFeatureExtractor_Int32_2D) {
  RunArrayFeatureExtractorTest<int32_t, int64_t>(
      TestInputDef<int32_t>({2, 5}, false, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      TestInputDef<int64_t>({3}, true, {0, 2, 4}),
      ExpectedEPNodeAssignment::All);
}

// 2D int64 X: requires Cast X→int32 before Gather and Cast output→int64 after.
TEST_F(QnnHTPBackendTests, ArrayFeatureExtractor_Int64_2D) {
  RunArrayFeatureExtractorTest<int64_t, int64_t>(
      TestInputDef<int64_t>({2, 4}, false, {10, 20, 30, 40, 50, 60, 70, 80}),
      TestInputDef<int64_t>({2}, true, {0, 3}),
      ExpectedEPNodeAssignment::All);
}

// Negative int64 indices (static initializer): -2 → axis_dim-2, -1 → axis_dim-1.
// Exercises the idx += axis_dim normalisation in ProcessInputs.
TEST_F(QnnHTPBackendTests, ArrayFeatureExtractor_Int32_2D_NegativeIndices) {
  RunArrayFeatureExtractorTest<int32_t, int64_t>(
      TestInputDef<int32_t>({3, 4}, false, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
      TestInputDef<int64_t>({2}, true, {-2, -1}),  // normalised to [2, 3]
      ExpectedEPNodeAssignment::All);
}

// ---------------------------------------------------------------------------
// Scalar index test
// ---------------------------------------------------------------------------

// Scalar Y (shape []) is not supported: ORT does not infer the output shape for
// ai.onnx.ml.ArrayFeatureExtractor with rank-0 indices, so
// QnnModel::SetGraphInputOutputInfo throws at compile time. Falls back to CPU.
TEST_F(QnnHTPBackendTests, ArrayFeatureExtractor_Float32_ScalarIndex) {
  RunArrayFeatureExtractorTest<float, int64_t>(
      TestInputDef<float>({3, 5}, false,
                          {1.f, 2.f, 3.f, 4.f, 5.f,
                           6.f, 7.f, 8.f, 9.f, 10.f,
                           11.f, 12.f, 13.f, 14.f, 15.f}),
      TestInputDef<int64_t>({}, true, {2}),
      ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
