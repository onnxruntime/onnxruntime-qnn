// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <memory>
#include <string>

#if defined(_WIN32)
#include <windows.h>
static void* LoadMockLib() { return LoadLibraryA("MockGenie.dll"); }
static void* GetSym(void* h, const char* s) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(h), s)); }
static constexpr const char* kMockGeniePath = "MockGenie.dll";
#else
#include <dlfcn.h>
static void* LoadMockLib() { return dlopen("libMockGenie.so", RTLD_NOW); }
static void* GetSym(void* h, const char* s) { return dlsym(h, s); }
static constexpr const char* kMockGeniePath = "libMockGenie.so";
#endif

#include "test/providers/qnn/qnn_test_utils.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace test {

// ==============================================================================
// GenieSessionTest
//
// Integration tests for the full Genie execution path. All tests create a real
// Ort::Session backed by the MockGenie shared library and verify end-to-end
// behavior — session creation, inference, and Genie API call sequences.
//
// The fixture inherits GenieBackendTests so tests skip gracefully when the
// Genie backend is unavailable, matching repo convention for all QNN tests.
// ==============================================================================

class GenieSessionTest : public GenieBackendTests {
 protected:
  void SetUp() override {
    GenieBackendTests::SetUp();  // runs availability skip logic
    // The Genie execution pathway in the QNN EP is guarded to ARM64 Windows only
    // (see qnn_execution_provider.cc). Skip on all other platforms.
    QNN_SKIP_TEST_ON_NON_ARM64_WINDOWS("Genie integration tests require ARM64 Windows");
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GenieSessionTest");
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
// backed by MockGenie. Returns a ScopedOrtSession that owns both the
// registered EP device and the Ort::Session, ensuring the session is
// destroyed before the EP device is unregistered.
// ---------------------------------------------------------------------------
static ScopedOrtSession MakeGenieSession(Ort::Env& env) {
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
  provider_options["backend_path"] = kMockGeniePath;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, "QNNExecutionProvider", provider_options);

  return ScopedOrtSession(std::move(registered_ep_device),
                          Ort::Session(env, model_data_span.data(), model_data_span.size(), so));
}

// ---------------------------------------------------------------------------
// Test: session creation with MockGenie succeeds (no exception thrown).
// ---------------------------------------------------------------------------
TEST_F(GenieSessionTest, CreateSession_Succeeds) {
  EXPECT_NO_THROW({
    ScopedOrtSession scoped = MakeGenieSession(*env_);
  });
}

// ---------------------------------------------------------------------------
// Test: session Run() succeeds end-to-end through the mock Genie call flow.
// MockGenie callback returns outputConfig "[1,1]"; ComputeImpl inserts a 1
// at index 1, so the final output shape is {1, 1, 1}.
// ---------------------------------------------------------------------------
TEST_F(GenieSessionTest, Run_ProducesExpectedOutputShape) {
  ScopedOrtSession scoped = MakeGenieSession(*env_);

  // Input: int32 tensor of shape {1, 1} — matches the graph definition.
  std::vector<int32_t> input_data = {0};
  const std::vector<int64_t> input_shape = {1, 1};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
      mem_info, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  const char* input_names[] = {"genie_input"};
  const char* output_names[] = {"genie_output"};
  std::vector<Ort::Value> outputs;

  EXPECT_NO_THROW({
    outputs = scoped.session().Run(Ort::RunOptions{nullptr},
                                   input_names, &input_tensor, 1,
                                   output_names, 1);
  });

  ASSERT_EQ(outputs.size(), 1u);
  auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(shape.size(), 3u);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 1);
  EXPECT_EQ(shape[2], 1);
}

// ---------------------------------------------------------------------------
// Test: MockGenie's exported call-tracking functions (ResetMockGenieCallCounts,
// GetMockGenieCallCount) resolve correctly and behave as expected.
// ---------------------------------------------------------------------------
TEST_F(GenieSessionTest, MockGenie_CallTrackingFunctions_Resolvable) {
  void* h = LoadMockLib();
  auto reset = reinterpret_cast<void (*)()>(GetSym(h, "ResetMockGenieCallCounts"));
  auto get_count = reinterpret_cast<int (*)(const char*)>(GetSym(h, "GetMockGenieCallCount"));
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(get_count, nullptr);
  reset();

  EXPECT_EQ(get_count("DlcConfig_create"), 0);
  ScopedOrtSession scoped = MakeGenieSession(*env_);
  EXPECT_EQ(get_count("DlcConfig_create"), 1);
  reset();
  EXPECT_EQ(get_count("DlcConfig_create"), 0);
}

// ---------------------------------------------------------------------------
// Test: the expected Genie API sequence is called during CreateStateImpl.
// One EPContext node → one call to each setup function.
// ---------------------------------------------------------------------------
TEST_F(GenieSessionTest, CreateState_InvokesExpectedApiSequence) {
  void* dll_handle = LoadMockLib();
  ASSERT_NE(dll_handle, nullptr) << "MockGenie handle must be non-null after session creation";
  auto reset = reinterpret_cast<void (*)()>(GetSym(dll_handle, "ResetMockGenieCallCounts"));
  auto get_count = reinterpret_cast<int (*)(const char*)>(GetSym(dll_handle, "GetMockGenieCallCount"));
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(get_count, nullptr);
  reset();

  ScopedOrtSession scoped = MakeGenieSession(*env_);

  EXPECT_EQ(get_count("DlcConfig_create"), 1) << "GenieDlcConfig_create";
  EXPECT_EQ(get_count("Dlc_create"), 1) << "GenieDlc_create";
  EXPECT_EQ(get_count("NodeConfig_createFromDlc"), 1) << "GenieNodeConfig_createFromDlc";
  EXPECT_EQ(get_count("Log_create"), 1) << "GenieLog_create";
  EXPECT_EQ(get_count("NodeConfig_bindLogger"), 1) << "GenieNodeConfig_bindLogger";
  EXPECT_EQ(get_count("Node_create"), 1) << "GenieNode_create";
}

// ---------------------------------------------------------------------------
// Test: the expected Genie API sequence is called during ComputeImpl.
// One input, one output → one call each to setData, execute, getData.
// Node_reset is not triggered on a fresh session (rewind_ starts at 1).
// ---------------------------------------------------------------------------
TEST_F(GenieSessionTest, Compute_InvokesExpectedApiSequence) {
  void* dll_handle = LoadMockLib();
  ASSERT_NE(dll_handle, nullptr) << "MockGenie handle must be non-null after session creation";
  auto reset = reinterpret_cast<void (*)()>(GetSym(dll_handle, "ResetMockGenieCallCounts"));
  auto get_count = reinterpret_cast<int (*)(const char*)>(GetSym(dll_handle, "GetMockGenieCallCount"));
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(get_count, nullptr);
  reset();

  ScopedOrtSession scoped = MakeGenieSession(*env_);

  std::vector<int32_t> input_data = {0};
  const std::vector<int64_t> input_shape = {1, 1};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
      mem_info, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  const char* input_names[] = {"genie_input"};
  const char* output_names[] = {"genie_output"};
  scoped.session().Run(Ort::RunOptions{nullptr},
                       input_names, &input_tensor, 1,
                       output_names, 1);

  EXPECT_EQ(get_count("Node_setData"), 1) << "GenieNode_setData";
  EXPECT_EQ(get_count("Node_execute"), 1) << "GenieNode_execute";
  EXPECT_EQ(get_count("Node_getData"), 1) << "GenieNode_getData";
  // KV-cache rewind is not triggered on a fresh session (rewind_ starts at 1):
  EXPECT_EQ(get_count("Node_reset"), 0) << "GenieNode_reset should not be called on first Run";
}

// ---------------------------------------------------------------------------
// Test: two sequential sessions backed by the same EP registration both
// succeed end-to-end. The second session reuses scoped1's already-registered
// OrtEpDevice directly (via AppendExecutionProvider_V2) rather than calling
// RegisterQnnEpLibrary again, which would fail because the ORT environment
// rejects duplicate registrations for the same name.
// ---------------------------------------------------------------------------
TEST_F(GenieSessionTest, TwoSequentialSessions_BothSucceed) {
  void* h = LoadMockLib();
  auto reset = reinterpret_cast<void (*)()>(GetSym(h, "ResetMockGenieCallCounts"));
  auto get_count = reinterpret_cast<int (*)(const char*)>(GetSym(h, "GetMockGenieCallCount"));
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(get_count, nullptr);

  // First session — registers the EP library and creates the backend.
  ScopedOrtSession scoped1 = MakeGenieSession(*env_);
  reset();

  // Second session — reuse scoped1's registration; do NOT call RegisterQnnEpLibrary.
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

  Ort::SessionOptions so2;
  ProviderOptions provider_options;
  provider_options["backend_path"] = kMockGeniePath;
  so2.AppendExecutionProvider_V2(*GetOrtEnv(), {Ort::ConstEpDevice(scoped1.ep_device())}, provider_options);
  {
    Ort::Session session2(*env_, model_data_span.data(), model_data_span.size(), so2);
    // session2's CreateStateImpl ran exactly once.
    EXPECT_EQ(get_count("DlcConfig_create"), 1);
  }  // session2 destroyed before scoped1
  // scoped1 destroyed here: session1 first, then EP unregistered
}

}  // namespace test
}  // namespace onnxruntime
