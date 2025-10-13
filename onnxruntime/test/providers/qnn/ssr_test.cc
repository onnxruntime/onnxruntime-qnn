// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/inference_session.h"
#include "core/framework/session_options.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))

TEST_F(QnnMockSSRBackendTests, SSRBackendGetBuildId) {
  controller->SetTiming(QnnMockSSRController::Timing::BackendGetBuildId);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRBackendCreate) {
  controller->SetTiming(QnnMockSSRController::Timing::BackendCreate);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRContextCreate) {
  controller->SetTiming(QnnMockSSRController::Timing::ContextCreate);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRBackendValidateOpConfig) {
  controller->SetTiming(QnnMockSSRController::Timing::BackendValidateOpConfig);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRLogCreate) {
  controller->SetTiming(QnnMockSSRController::Timing::LogCreate);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphCreate) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphCreate);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphRetrieve) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphRetrieve);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRContextGetBinarySize) {
  controller->SetTiming(QnnMockSSRController::Timing::ContextGetBinarySize);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRContextGetBinary) {
  controller->SetTiming(QnnMockSSRController::Timing::ContextGetBinary);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRTensorCreateGraphTensor) {
  controller->SetTiming(QnnMockSSRController::Timing::TensorCreateGraphTensor);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphAddNode) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphAddNode);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphFinalize) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphFinalize);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnMockSSRBackendTests, SSRGraphExecute) {
  controller->SetTiming(QnnMockSSRController::Timing::GraphExecute);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                                         {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64)
}  // namespace test
}  // namespace onnxruntime

#endif
