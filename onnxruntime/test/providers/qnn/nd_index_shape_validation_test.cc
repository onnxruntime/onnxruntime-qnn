// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace onnxruntime {
namespace test {
namespace {

// Creates a session without running the malformed model. The CPU kernels may
// reject these inputs at execution time; this test is specifically checking
// that QNN EP shape validation declines the node during graph partitioning.
void VerifyMalformedNdIndexShapeIsRejected(const GetTestModelFn& build_model) {
  ModelTestBuilder builder;
  build_model(builder);

  auto* opset = builder.model_.add_opset_import();
  opset->set_domain(kOnnxDomain);
  opset->set_version(18);
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  ASSERT_TRUE(builder.model_.SerializeToString(&model_data));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  TryEnableQNNSaver(provider_options);

  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");

  RegisteredEpDeviceUniquePtr ep_device;
  const std::string registration_name = "QNNExecutionProvider";
  RegisterQnnEpLibrary(ep_device, session_options, registration_name, provider_options);

  ScopedOrtSession scoped(std::move(ep_device),
                          Ort::Session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options));
  ASSERT_NO_FATAL_FAILURE(
      VerifyEPNodeAssignment(scoped.session(), registration_name, ExpectedEPNodeAssignment::None));
}

TEST_F(QnnHTPBackendTests, GatherNDRejectsNegativeBatchDims) {
  VerifyMalformedNdIndexShapeIsRejected([](ModelTestBuilder& builder) {
    builder.MakeInput<float>("data", {2}, {1.0f, 2.0f});
    builder.MakeInitializer<int64_t>("indices", {1, 1}, {0});
    builder.AddNode("gathernd", "GatherND", {"data", "indices"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("batch_dims", static_cast<int64_t>(-1))});
    builder.MakeOutput("Y");
  });
}

TEST_F(QnnHTPBackendTests, GatherNDRejectsBatchDimsAtIndicesRank) {
  VerifyMalformedNdIndexShapeIsRejected([](ModelTestBuilder& builder) {
    builder.MakeInput<float>("data", {2, 2, 2},
                             {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    builder.MakeInitializer<int64_t>("indices", {1, 1}, {0});
    builder.AddNode("gathernd", "GatherND", {"data", "indices"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("batch_dims", static_cast<int64_t>(2))});
    builder.MakeOutput("Y");
  });
}

// HTP supports dynamic INT64 GatherND indices. Keep this input dynamic so the
// rejection also covers the non-initializer path through ProcessGatherNDIndices.
TEST_F(QnnHTPBackendTests, GatherNDRejectsDynamicIndicesWithBatchDimsAtIndicesRank) {
  VerifyMalformedNdIndexShapeIsRejected([](ModelTestBuilder& builder) {
    builder.MakeInput<float>("data", {2, 2, 2},
                             {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    builder.MakeInput<int64_t>("indices", {1, 1}, {0});
    builder.AddNode("gathernd", "GatherND", {"data", "indices"}, {"Y"}, kOnnxDomain,
                    {test::MakeAttribute("batch_dims", static_cast<int64_t>(2))});
    builder.MakeOutput("Y");
  });
}

TEST_F(QnnHTPBackendTests, ScatterNDRejectsRankZeroIndices) {
  VerifyMalformedNdIndexShapeIsRejected([](ModelTestBuilder& builder) {
    builder.MakeInput<float>("data", {2}, {1.0f, 2.0f});
    builder.MakeInitializer<int64_t>("indices", {}, {0});
    builder.MakeInitializer<float>("updates", {2}, {3.0f, 4.0f});
    builder.AddNode("scatternd", "ScatterND", {"data", "indices", "updates"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  });
}

TEST_F(QnnHTPBackendTests, ScatterNDRejectsZeroIndexDepth) {
  VerifyMalformedNdIndexShapeIsRejected([](ModelTestBuilder& builder) {
    builder.MakeInput<float>("data", {2}, {1.0f, 2.0f});
    builder.MakeInitializer<int64_t>("indices", {1, 0}, {});
    builder.MakeInitializer<float>("updates", {1, 2}, {3.0f, 4.0f});
    builder.AddNode("scatternd", "ScatterND", {"data", "indices", "updates"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  });
}

TEST_F(QnnHTPBackendTests, ScatterNDRejectsIndexDepthPastDataRank) {
  VerifyMalformedNdIndexShapeIsRejected([](ModelTestBuilder& builder) {
    builder.MakeInput<float>("data", {2}, {1.0f, 2.0f});
    builder.MakeInitializer<int64_t>("indices", {1, 2}, {0, 0});
    builder.MakeInitializer<float>("updates", {1}, {3.0f});
    builder.AddNode("scatternd", "ScatterND", {"data", "indices", "updates"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  });
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#endif  // !defined(ORT_MINIMAL_BUILD)
