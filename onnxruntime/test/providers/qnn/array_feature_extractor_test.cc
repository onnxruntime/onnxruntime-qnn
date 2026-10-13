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

// Runs a model on QNN EP only — no CPU reference session.
// Use when the CPU EP rejects the inputs (e.g. negative indices, which are
// outside the ONNX spec for ArrayFeatureExtractor but handled defensively
// by the QNN EP builder). Verifies EP assignment and compares output against
// hardcoded expected values.
template <typename OutputType>
static void RunQnnOnlyModelTest(
    const GetTestModelFn& build_test_case,
    const std::vector<OutputType>& expected_output,
    const std::vector<int64_t>& expected_shape) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 9}, {kMSDomain, 1}};
  ModelTestBuilder helper;
  build_test_case(helper);
  for (const auto& [domain, version] : domain_to_version) {
    auto* opset = helper.model_.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif
  TryEnableQNNSaver(provider_options);

  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");

  RegisteredEpDeviceUniquePtr ep_device;
  const std::string registration_name = "QNNExecutionProvider";
  RegisterQnnEpLibrary(ep_device, session_options, registration_name, provider_options);

  ScopedOrtSession scoped(std::move(ep_device),
                          Ort::Session(*GetOrtEnv(), model_data.data(),
                                       model_data.size(), session_options));

  ASSERT_NO_FATAL_FAILURE(
      VerifyEPNodeAssignment(scoped.session(), registration_name,
                             ExpectedEPNodeAssignment::All));

  Ort::RunOptions run_options;
  run_options.SetRunTag("QNN_EP_TestLogID");
  std::vector<Ort::Value> fetches;
  RunWithEP(scoped.session(), run_options, helper.feeds_, fetches);

  ASSERT_EQ(fetches.size(), 1u);
  const auto shape = fetches[0].GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(shape, expected_shape);
  const OutputType* data = fetches[0].GetTensorData<OutputType>();
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_EQ(data[i], expected_output[i]) << "mismatch at index " << i;
  }
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
// Uses RunQnnOnlyModelTest because the CPU EP rejects negative indices.
TEST_F(QnnHTPBackendTests, ArrayFeatureExtractor_Int32_2D_NegativeIndices) {
  // X[3,4], Y=[-2,-1] → normalised to [2,3] → output {2,3,6,7,10,11} shape (3,2).
  RunQnnOnlyModelTest<int32_t>(
      WrapWithMLDomain(
          BuildOpTestCase<int32_t, int64_t>(
              "afe_node", "ArrayFeatureExtractor",
              {TestInputDef<int32_t>({3, 4}, false, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})},
              {TestInputDef<int64_t>({2}, true, {-2, -1})},
              {}, kAiOnnxMlDomain)),
      /*expected_output=*/{2, 3, 6, 7, 10, 11},
      /*expected_shape=*/{3, 2});
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
