// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/inference_session.h"
#include "core/framework/session_options.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "ssr/ssr_controller.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace qnn_ssr {
#if defined(_WIN32)
  #include <windows.h>
  HMODULE lib_handle = LoadLibraryW(L"QnnMockSSR.dll");
  FARPROC addr = GetProcAddress(lib_handle, "GetQnnSSRController");
  typedef QnnSSRController* (*GetQnnSSRControllerFn_t)();
  GetQnnSSRControllerFn_t GetQnnSSRController = reinterpret_cast<GetQnnSSRControllerFn_t>(addr);
  QnnSSRController* controller = GetQnnSSRController();
#endif  // defined(_WIN32)
}  // namespace qnn_ssr
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, SSRGraphExecute) {
  ProviderOptions provider_options;
  provider_options["backend_path"] = "QnnMockSSR.dll";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_ssr_handling"] = "1";
  qnn_ssr::controller->SetTiming(QnnSSRController::Timing::GraphExecute);
  auto input_def = TestInputDef<float>({1, 2, 3, 3}, false, {-10.0f, 10.0f});
  auto scale_def = TestInputDef<float>({2}, true, {1.0f, 2.0f});
  auto bias_def = TestInputDef<float>({2}, true, {1.0f, 3.0f});
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization", {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
                  18,
                  ExpectedEPNodeAssignment::All,
                  5e-3f,
                  logging::Severity::kVERBOSE,
                  true,
                  nullptr);
}

TEST_F(QnnHTPBackendTests, SSRGraphFinalize) {
  ProviderOptions provider_options;
  provider_options["backend_path"] = "QnnMockSSR.dll";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_ssr_handling"] = "1";
  qnn_ssr::controller->SetTiming(QnnSSRController::Timing::GraphFinalize);
  auto input_def = TestInputDef<float>({1, 2, 3, 3}, false, {-10.0f, 10.0f});
  auto scale_def = TestInputDef<float>({2}, true, {1.0f, 2.0f});
  auto bias_def = TestInputDef<float>({2}, true, {1.0f, 3.0f});
  RunQnnModelTest(BuildOpTestCase<float>("InstanceNormalization", {input_def, scale_def, bias_def}, {}, {}),
                  provider_options,
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
