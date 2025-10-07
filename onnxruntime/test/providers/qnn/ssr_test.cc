// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/inference_session.h"
#include "core/framework/session_options.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "ssr/qnn_mock_ssr_controller.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace qnn_ssr {
#if defined(_WIN32)
  #include <windows.h>
  HMODULE lib_handle = LoadLibraryW(L"QnnMockSSR.dll");
  FARPROC addr = GetProcAddress(lib_handle, "GetQnnMockSSRController");
  typedef QnnMockSSRController* (*GetQnnMockSSRControllerFn_t)();
  GetQnnMockSSRControllerFn_t GetQnnMockSSRController = reinterpret_cast<GetQnnMockSSRControllerFn_t>(addr);
  QnnMockSSRController* controller = GetQnnMockSSRController();
#endif  // defined(_WIN32)
  auto input_def = TestInputDef<float>({1, 2, 3, 3}, false, {-10.0f, 10.0f});
  auto scale_def = TestInputDef<float>({2}, true, {1.0f, 2.0f});
  auto bias_def = TestInputDef<float>({2}, true, {1.0f, 3.0f});
  ProviderOptions provider_options = {
    {"backend_path", "QnnMockSSR.dll"},
    {"offload_graph_io_quantization", "0"},
    {"enable_ssr_handling", "1"},
  };
}  // namespace qnn_ssr

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, SSRTensorCreateGraphTensor) {
  qnn_ssr::controller->SetTiming(QnnMockSSRController::Timing::TensorCreateGraphTensor);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                  {qnn_ssr::input_def, qnn_ssr::scale_def, qnn_ssr::bias_def}, {}, {}),
                  qnn_ssr::provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnHTPBackendTests, SSRGraphAddNode) {
  qnn_ssr::controller->SetTiming(QnnMockSSRController::Timing::GraphAddNode);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                  {qnn_ssr::input_def, qnn_ssr::scale_def, qnn_ssr::bias_def}, {}, {}),
                  qnn_ssr::provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnHTPBackendTests, SSRGraphFinalize) {
  qnn_ssr::controller->SetTiming(QnnMockSSRController::Timing::GraphFinalize);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                  {qnn_ssr::input_def, qnn_ssr::scale_def, qnn_ssr::bias_def}, {}, {}),
                  qnn_ssr::provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnHTPBackendTests, SSRGraphExecute) {
  qnn_ssr::controller->SetTiming(QnnMockSSRController::Timing::GraphExecute);
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization",
                  {qnn_ssr::input_def, qnn_ssr::scale_def, qnn_ssr::bias_def}, {}, {}),
                  qnn_ssr::provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
