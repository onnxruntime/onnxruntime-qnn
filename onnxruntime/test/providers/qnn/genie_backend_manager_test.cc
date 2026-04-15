// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <memory>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"

using namespace onnxruntime;
using namespace onnxruntime::test;

class GenieBackendManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GenieBackendManagerTest");
  }

  void TearDown() override {
    env_.reset();
  }
  std::unique_ptr<Ort::Env> env_;
};

static GetTestModelFn CreateDlcContextGraph() {
    static constexpr const char* kMSDomain = "com.microsoft";
    return [](onnxruntime::test::ModelTestBuilder& builder) {
      std::vector<int32_t> input_data = {0};
      const std::vector<int64_t> input_shape = {1, 1};
      MakeTestInput(builder, "genie_input",
                    TestInputDef<int32_t>(input_shape, false, input_data));
      builder.MakeOutput<float>("genie_output", std::vector<int64_t>{1, 1, 1});

      std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
      attrs.push_back(builder.MakeStringAttribute("ep_context_type", "dlc"));
      attrs.push_back(builder.MakeStringAttribute("source", "QAIRTExport"));
      attrs.push_back(builder.MakeStringAttribute("ep_dlc_context", "model.dlc"));
      builder.AddNode("GenieDlcContextNode", "EPContext",
                      {"genie_input"}, {"genie_output"}, kMSDomain, attrs);
    };
  }

// ---------------------------------------------------------------------------
// Shared helper: builds the serialised model bytes and creates a session
// backed by MockGenie.dll.  Returns the session; the registered_ep_device
// lifetime must outlive the session.
// ---------------------------------------------------------------------------
static Ort::Session MakeGenieSession(
    Ort::Env& env,
    RegisteredEpDeviceUniquePtr& registered_ep_device) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};
  ModelTestBuilder helper;
  CreateDlcContextGraph()(helper);

  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{
        helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_path"] = "MockGenie.dll";
  RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", provider_options);

  return Ort::Session(env, model_data_span.data(), model_data_span.size(), so);
}

// ---------------------------------------------------------------------------
// Test: session creation with MockGenie.dll succeeds (no exception thrown).
// ---------------------------------------------------------------------------
TEST_F(GenieBackendManagerTest, CreateBackendManager) {
  RegisteredEpDeviceUniquePtr registered_ep_device;
  EXPECT_NO_THROW({
    Ort::Session session = MakeGenieSession(*env_, registered_ep_device);
  });
}

// ---------------------------------------------------------------------------
// Test: session Run() succeeds end-to-end through the mock Genie call flow.
// Verifies: Node_setData, Node_execute, Node_getData are each called once,
// and the output tensor is populated without error.
// ---------------------------------------------------------------------------
TEST_F(GenieBackendManagerTest, SessionRunSucceeds) {
  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::Session session = MakeGenieSession(*env_, registered_ep_device);

  // Input: int32 tensor of shape {1, 1} — matches the graph definition.
  std::vector<int32_t> input_data = {0};
  const std::vector<int64_t> input_shape = {1, 1};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
      mem_info, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  const char* input_names[]  = {"genie_input"};
  const char* output_names[] = {"genie_output"};
  std::vector<Ort::Value> outputs;

  EXPECT_NO_THROW({
    outputs = session.Run(Ort::RunOptions{nullptr},
                          input_names,  &input_tensor, 1,
                          output_names, 1);
  });

  ASSERT_EQ(outputs.size(), 1u);
  auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  // MockGenie callback returns outputConfig "[1,1]";
  // ComputeImpl inserts a 1 at index 1 → final shape {1, 1, 1}.
  ASSERT_EQ(shape.size(), 3u);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 1);
  EXPECT_EQ(shape[2], 1);
}

TEST_F(GenieBackendManagerTest, MockBackendAccessSuccess) {

  HMODULE h = LoadLibraryA("MockGenie.dll");
  auto reset = (void (*)())GetProcAddress(h, "ResetMockGenieCalls");
  auto get_count = (int (*)(const char*))GetProcAddress(h, "GetMockGenieCallCount");
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(get_count, nullptr);
  reset();

  EXPECT_EQ(get_count("DlcConfig_create"), 0);
  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::Session session = MakeGenieSession(*env_, registered_ep_device);
  EXPECT_EQ(get_count("DlcConfig_create"), 1);
  reset();
  EXPECT_EQ(get_count("DlcConfig_create"), 0);
}

// static void* GetMockGenieHandle(Ort::Session& session) {
//   // Access the underlying InferenceSession through the Ort::Session handle.
//   auto* inference_session = reinterpret_cast<InferenceSession*>(session.operator OrtSession*());
//   if (!inference_session) return nullptr;

//   for (const auto& ep : inference_session->GetRegisteredExecutionProviders()) {
//     if (ep->Type() == "QNNExecutionProvider") {
//       auto* qnn_ep = static_cast<QnnEp*>(ep.get());
//       return qnn_ep->GetGenieBackendHandleForTest();
//     }
//   }
//   return nullptr;
// }

TEST_F(GenieBackendManagerTest, VerifyGenieApiCallsDuringSetup) {

  HMODULE dll_handle = LoadLibraryA("MockGenie.dll");
  ASSERT_NE(dll_handle, nullptr) << "MockGenie.dll handle must be non-null after session creation";
  auto reset = (void (*)())GetProcAddress(dll_handle, "ResetMockGenieCalls");
  auto get_count = (int (*)(const char*))GetProcAddress(dll_handle, "GetMockGenieCallCount");
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(get_count, nullptr);
  reset();
  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::Session session = MakeGenieSession(*env_, registered_ep_device);

  // CreateStateImpl sequence (one EPContext node → one call each):
  EXPECT_EQ(get_count("DlcConfig_create"),         1) << "GenieDlcConfig_create";
  EXPECT_EQ(get_count("Dlc_create"),               1) << "GenieDlc_create";
  EXPECT_EQ(get_count("NodeConfig_createFromDlc"), 1) << "GenieNodeConfig_createFromDlc";
  EXPECT_EQ(get_count("Log_create"),               1) << "GenieLog_create";
  EXPECT_EQ(get_count("NodeConfig_bindLogger"),    1) << "GenieNodeConfig_bindLogger";
  EXPECT_EQ(get_count("Node_create"),              1) << "GenieNode_create";
}

// ---------------------------------------------------------------------------
// Test: the expected Genie API sequence is called during Run().
// ---------------------------------------------------------------------------
TEST_F(GenieBackendManagerTest, VerifyGenieApiCallsDuringRun) {
  HMODULE dll_handle = LoadLibraryA("MockGenie.dll");
  ASSERT_NE(dll_handle, nullptr) << "MockGenie.dll handle must be non-null after session creation";
  auto reset = (void (*)())GetProcAddress(dll_handle, "ResetMockGenieCalls");
  auto get_count = (int (*)(const char*))GetProcAddress(dll_handle, "GetMockGenieCallCount");
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(get_count, nullptr);
  reset();

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::Session session = MakeGenieSession(*env_, registered_ep_device);

  std::vector<int32_t> input_data = {0};
  const std::vector<int64_t> input_shape = {1, 1};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
      mem_info, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  const char* input_names[]  = {"genie_input"};
  const char* output_names[] = {"genie_output"};
  session.Run(Ort::RunOptions{nullptr},
              input_names, &input_tensor, 1,
              output_names, 1);

  // ComputeImpl sequence (one input, one output):
  EXPECT_EQ(get_count("Node_setData"), 1) << "GenieNode_setData";
  EXPECT_EQ(get_count("Node_execute"), 1) << "GenieNode_execute";
  EXPECT_EQ(get_count("Node_getData"), 1) << "GenieNode_getData";
  // KV-cache rewind is not triggered on a fresh session (rewind_ starts at 1):
  EXPECT_EQ(get_count("Node_reset"),   0) << "GenieNode_reset should not be called on first Run";
}
