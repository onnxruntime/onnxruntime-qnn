// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#include "core/providers/qnn/builder/qnn_def.h"

#if QNN_EP_INTERNAL_SYMBOL_ACCESS && defined(QNN_FILE_MAPPED_WEIGHTS_AVAILABLE) && ORT_API_HAS_EPCONTEXT_ENCRYPTION

#include <windows.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <stdexcept>

#include "core/providers/qnn/builder/qnn_context_decrypt_source.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "test/providers/qnn/infra/qnn_fake_ort_graph.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

namespace onnxruntime::test {
namespace {
struct SourceState {
  size_t legacy_calls = 0;
  size_t source_calls = 0;
  const void* source = nullptr;
  size_t source_bytes = 0;
  void* output = nullptr;
  bool empty_output = false;
  bool null_output = false;
  bool fail = false;
  bool throw_exception = false;
  std::string name;

  static OrtStatus* ORT_API_CALL Legacy(void* state, const char* name, OrtAllocator* allocator,
                                        void** data, size_t* bytes) {
    auto& self = *static_cast<SourceState*>(state);
    ++self.legacy_calls;
    self.name = name;
    *data = allocator->Alloc(allocator, 4);
    *bytes = 4;
    if (!*data) return Ort::GetApi().CreateStatus(ORT_FAIL, "Test allocation failed.");
    std::memcpy(*data, "data", 4);
    self.output = *data;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL Decrypt(void* state, const char* name, const void* source,
                                         size_t source_bytes, OrtAllocator* allocator,
                                         void** data, size_t* bytes) {
    auto& self = *static_cast<SourceState*>(state);
    ++self.source_calls;
    self.source = source;
    self.source_bytes = source_bytes;
    self.name = name;
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(source, &info, sizeof(info)) || info.Type != MEM_MAPPED || info.Protect != PAGE_READONLY ||
        source_bytes != 6 || std::memcmp(source, "XXdata", 6) != 0) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, "Expected exact mapped source bytes.");
    }
    if (self.throw_exception) throw std::runtime_error("Test decoder exception.");
    *bytes = source_bytes - 2;
    *data = self.null_output ? nullptr : allocator->Alloc(allocator, *bytes);
    if (self.null_output) return nullptr;
    if (!*data) return Ort::GetApi().CreateStatus(ORT_FAIL, "Test allocation failed.");
    std::memcpy(*data, static_cast<const char*>(source) + 2, *bytes);
    self.output = *data;
    if (self.empty_output) *bytes = 0;
    if (self.fail) {
      allocator->Free(allocator, *data);
      return Ort::GetApi().CreateStatus(ORT_FAIL, "Test decrypt failure after output cleanup.");
    }
    return nullptr;
  }
};

class QnnDecryptSourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    wchar_t folder[MAX_PATH], file[MAX_PATH];
    ASSERT_NE(GetTempPathW(MAX_PATH, folder), 0u);
    ASSERT_NE(GetTempFileNameW(folder, L"qnn", 0, file), 0u);
    path_ = file;
    std::ofstream source(path_, std::ios::binary);
    source << "XXdata";
    ASSERT_TRUE(source.good());
    Ort::ThrowOnError(Ort::GetApi().CreateSessionOptions(&options_));
    Ort::ThrowOnError(QnnEpCreateDecryptSourceBinding(SourceState::Legacy, SourceState::Decrypt,
                                                      &state_, &binding_, &bridge_));
    auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
    ASSERT_NE(setter, nullptr);
    Ort::ThrowOnError(setter(options_, bridge_, binding_));
  }
  void TearDown() override {
    Ort::GetApi().ReleaseSessionOptions(options_);
    QnnEpReleaseDecryptSourceBinding(binding_);
    if (!path_.empty()) {
      EXPECT_TRUE(DeleteFileW(path_.c_str()));
    }
  }
  void ExpectSourceReleased() {
    MEMORY_BASIC_INFORMATION info{};
    ASSERT_NE(VirtualQuery(state_.source, &info, sizeof(info)), 0u);
    EXPECT_EQ(info.State, static_cast<DWORD>(MEM_FREE));
  }
  std::filesystem::path path_;
  SourceState state_;
  OrtSessionOptions* options_ = nullptr;
  QnnEpDecryptSourceBinding* binding_ = nullptr;
  OrtReadNamedBufferFunc bridge_ = nullptr;
  Ort::Logger logger_ = MakeNullLogger();
};

TEST_F(QnnDecryptSourceTest, OffDelegatesLegacyWithOriginalStateWithoutOpeningSource) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION});
  std::vector<char> out;
  std::future<void> cleanup;
  ASSERT_TRUE(dispatch.Read("logical-name", out, "nonexistent-source", &cleanup).IsOK());
  EXPECT_FALSE(cleanup.valid());
  EXPECT_EQ(state_.legacy_calls, 1u);
  EXPECT_EQ(state_.source_calls, 0u);
  EXPECT_EQ(state_.name, "logical-name");
  EXPECT_EQ(std::string(out.data(), out.size()), "data");
  EXPECT_NE(out.data(), state_.output);
}

TEST_F(QnnDecryptSourceTest, OnCopiesIndependentDifferentSizedOutput) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  std::vector<char> out;
  ASSERT_TRUE(dispatch.Read("logical-name", out, path_.u8string()).IsOK());
  EXPECT_EQ(state_.legacy_calls, 0u);
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(state_.source_bytes, 6u);
  EXPECT_EQ(out.size(), 4u);
  EXPECT_EQ(std::string(out.data(), out.size()), "data");
  EXPECT_NE(out.data(), state_.output);
  EXPECT_NE(out.data(), state_.source);
  ExpectSourceReleased();
  EXPECT_EQ(std::memcmp(out.data(), "data", 4), 0);
}

TEST_F(QnnDecryptSourceTest, Utf8SourcePathPreservesCallbackNameAndUnmapsAfterReturn) {
  const auto unicode_path = std::filesystem::path(path_.wstring() + L"_\u6E2C\u8A66_\u00E9.bin");
  ASSERT_TRUE(MoveFileW(path_.c_str(), unicode_path.c_str()));
  path_ = unicode_path;
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  std::vector<char> out;
  const auto name = path_.filename().u8string();
  ASSERT_TRUE(dispatch.Read(name, out, path_.u8string()).IsOK());
  EXPECT_EQ(state_.name, name);
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  ASSERT_EQ(out.size(), 4u);
  EXPECT_EQ(std::memcmp(out.data(), "data", 4), 0);
  ExpectSourceReleased();
}

TEST_F(QnnDecryptSourceTest, FailedDecryptReleasesViewWithoutRetryOrReplacingPreviousOutput) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  std::vector<char> out;
  out.assign(4, 'p');
  auto* previous = out.data();
  state_.fail = true;
  std::future<void> cleanup;
  EXPECT_FALSE(dispatch.Read("logical-name", out, path_.u8string(), &cleanup).IsOK());
  EXPECT_FALSE(cleanup.valid());
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  EXPECT_EQ(out.data(), previous);
  ExpectSourceReleased();
}

TEST_F(QnnDecryptSourceTest, MappingFailureDoesNotInvokeEitherCallback) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  std::vector<char> out;
  EXPECT_FALSE(dispatch.Read("logical-name", out, path_.u8string() + ".absent").IsOK());
  EXPECT_EQ(state_.source_calls, 0u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  EXPECT_EQ(out.size(), 0u);
}

TEST_F(QnnDecryptSourceTest, UnrelatedCallbackIsRejectedWithoutInterpretingItsApplicationState) {
  auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
  Ort::ThrowOnError(setter(options_, SourceState::Legacy, &state_));
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  std::vector<char> out;
  EXPECT_FALSE(dispatch.HasSourceReadCallback());
  EXPECT_FALSE(dispatch.Read("logical-name", out, path_.u8string()).IsOK());
  EXPECT_EQ(state_.legacy_calls, 0u);
  EXPECT_EQ(state_.source_calls, 0u);
}

TEST_F(QnnDecryptSourceTest, MappingFailurePreservesPreviousOutput) {
  std::vector<char> out;
  qnn::EpContextIoDispatch legacy(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION});
  ASSERT_TRUE(legacy.Read("previous", out).IsOK());
  auto* previous = out.data();
  qnn::EpContextIoDispatch mapped(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  EXPECT_FALSE(mapped.Read("logical-name", out, path_.u8string() + ".absent").IsOK());
  EXPECT_EQ(out.data(), previous);
  EXPECT_EQ(out.size(), 4u);
  EXPECT_EQ(std::memcmp(out.data(), "data", 4), 0);
  EXPECT_EQ(state_.legacy_calls, 1u);
  EXPECT_EQ(state_.source_calls, 0u);
}

TEST_F(QnnDecryptSourceTest, InvalidSuccessfulSourceOutputUnmapsAndPreservesPreviousOutput) {
  std::vector<char> out;
  qnn::EpContextIoDispatch legacy(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION});
  ASSERT_TRUE(legacy.Read("previous", out).IsOK());
  auto* previous = out.data();
  qnn::EpContextIoDispatch mapped(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  std::future<void> cleanup;
  state_.empty_output = true;
  EXPECT_FALSE(mapped.Read("empty", out, path_.u8string(), &cleanup).IsOK());
  EXPECT_FALSE(cleanup.valid());
  ExpectSourceReleased();
  state_.empty_output = false;
  state_.null_output = true;
  EXPECT_FALSE(mapped.Read("null", out, path_.u8string(), &cleanup).IsOK());
  EXPECT_FALSE(cleanup.valid());
  ExpectSourceReleased();
  EXPECT_EQ(out.data(), previous);
  EXPECT_EQ(out.size(), 4u);
  EXPECT_EQ(std::memcmp(out.data(), "data", 4), 0);
  EXPECT_EQ(state_.legacy_calls, 1u);
  EXPECT_EQ(state_.source_calls, 2u);
}

TEST_F(QnnDecryptSourceTest, SsrSourceOnPreservesResolvedUtf8PathAndOriginalOutput) {
  const auto unicode_path = std::filesystem::path(path_.wstring() + L"_\u6E2C\u8A66_\u00E9.bin");
  ASSERT_TRUE(MoveFileW(path_.c_str(), unicode_path.c_str()));
  path_ = unicode_path;
  StubApiEnv env;
  StubBackendManager backend(env.api_ptrs, env.logger);
  static thread_local SourceState* current_state = nullptr;
  current_state = &state_;
  backend.QnnInterface().contextFree = [](Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
    return QNN_SUCCESS;
  };
  backend.QnnInterface().contextCreateFromBinary = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                                      const QnnContext_Config_t**, const void* data,
                                                      Qnn_ContextBinarySize_t size, Qnn_ContextHandle_t* context,
                                                      Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
    EXPECT_NE(data, current_state->output);
    EXPECT_EQ(size, 4u);
    EXPECT_EQ(std::memcmp(data, "data", 4), 0);
    MEMORY_BASIC_INFORMATION info{};
    EXPECT_NE(VirtualQuery(current_state->source, &info, sizeof(info)), 0u);
    EXPECT_EQ(info.State, static_cast<DWORD>(MEM_FREE));
    *context = current_state;
    return QNN_SUCCESS;
  };
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(backend.GetMutable()->ReloadContextForSSR(path_.u8string(), 0, context, dispatch).IsOK());
  EXPECT_EQ(state_.name, path_.filename().u8string());
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  ExpectSourceReleased();
  current_state = nullptr;
}

TEST_F(QnnDecryptSourceTest, SsrDecodeFailureUnmapsWithoutRetryingLegacyOrCallingQnn) {
  StubApiEnv env;
  StubBackendManager backend(env.api_ptrs, env.logger);
  backend.QnnInterface().contextCreateFromBinary = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                                      const QnnContext_Config_t**, const void*,
                                                      Qnn_ContextBinarySize_t, Qnn_ContextHandle_t*,
                                                      Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
    ADD_FAILURE() << "QNN must not receive a failed decode";
    return QNN_SUCCESS;
  };
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, &logger_, true);
  state_.fail = true;
  Qnn_ContextHandle_t context = nullptr;
  auto status = backend.GetMutable()->ReloadContextForSSR(path_.u8string(), 0, context, dispatch);
  EXPECT_EQ(status.GetErrorMessage(), "Test decrypt failure after output cleanup.");
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  ExpectSourceReleased();
}

TEST_F(QnnDecryptSourceTest, SourceMappingDoesNotRequireALogger) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, nullptr, true);
  std::vector<char> out;
  ASSERT_TRUE(dispatch.Read("logical-name", out, path_.u8string()).IsOK());
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(std::string(out.data(), out.size()), "data");
  ExpectSourceReleased();
}

TEST_F(QnnDecryptSourceTest, EmptySourceFailsBeforeCallbacksAndClosesItsHandle) {
  {
    std::ofstream empty(path_, std::ios::binary | std::ios::trunc);
  }
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, nullptr, true);
  std::vector<char> out;
  EXPECT_FALSE(dispatch.Read("logical-name", out, path_.u8string()).IsOK());
  EXPECT_EQ(state_.source_calls, 0u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  HANDLE exclusive = CreateFileW(path_.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(exclusive, INVALID_HANDLE_VALUE);
  EXPECT_TRUE(CloseHandle(exclusive));
}

TEST_F(QnnDecryptSourceTest, InvalidPathsFailBeforeCallbacksAndPreserveOutput) {
  std::vector<char> out;
  qnn::EpContextIoDispatch legacy(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION});
  ASSERT_TRUE(legacy.Read("previous", out).IsOK());
  const auto* previous = out.data();
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, nullptr, true);
  for (const auto& path : {std::string{}, std::string("invalid\0path", 12),
                           std::string("\xc3\x28", 2), path_.u8string() + ".missing"}) {
    EXPECT_FALSE(dispatch.Read("logical-name", out, path).IsOK());
    EXPECT_EQ(out.data(), previous);
    EXPECT_EQ(std::string(out.data(), out.size()), "data");
  }
  EXPECT_EQ(state_.source_calls, 0u);
  EXPECT_EQ(state_.legacy_calls, 1u);
}

TEST_F(QnnDecryptSourceTest, DecoderExceptionUnmapsWithoutReplacingOutput) {
  std::vector<char> out;
  qnn::EpContextIoDispatch legacy(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION});
  ASSERT_TRUE(legacy.Read("previous", out).IsOK());
  const auto* previous = out.data();
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, nullptr, true);
  std::future<void> cleanup;
  state_.throw_exception = true;
  EXPECT_EQ(dispatch.Read("logical-name", out, path_.u8string(), &cleanup).GetErrorMessage(),
            "Test decoder exception.");
  EXPECT_FALSE(cleanup.valid());
  EXPECT_EQ(out.data(), previous);
  EXPECT_EQ(state_.source_calls, 1u);
  EXPECT_EQ(state_.legacy_calls, 1u);
  ExpectSourceReleased();
}

TEST_F(QnnDecryptSourceTest, NullBindingIsRejectedBeforeOpeningSource) {
  auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
  Ort::ThrowOnError(setter(options_, bridge_, nullptr));
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, nullptr, true);
  std::vector<char> out;
  EXPECT_FALSE(dispatch.HasSourceReadCallback());
  EXPECT_FALSE(dispatch.Read("logical-name", out, path_.u8string()).IsOK());
  EXPECT_EQ(state_.source_calls, 0u);
  EXPECT_EQ(state_.legacy_calls, 0u);
}

TEST_F(QnnDecryptSourceTest, AsyncReadKeepsIndependentOutputAndReleasesSourceAtScopeExit) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, nullptr, true);
  std::vector<char> out;
  {
    std::future<void> cleanup;
    ASSERT_TRUE(dispatch.Read("logical-name", out, path_.u8string(), &cleanup).IsOK());
    EXPECT_TRUE(cleanup.valid());
    EXPECT_EQ(state_.source_calls, 1u);
    EXPECT_EQ(state_.legacy_calls, 0u);
    EXPECT_EQ(state_.source_bytes, 6u);
    EXPECT_NE(out.data(), state_.source);
    EXPECT_NE(out.data(), state_.output);
    EXPECT_EQ(std::string(out.begin(), out.end()), "data");
  }
  ExpectSourceReleased();
  EXPECT_EQ(std::string(out.begin(), out.end()), "data");
}

TEST_F(QnnDecryptSourceTest, PendingCleanupIsNotReplacedAndSourceReleasesSynchronously) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, nullptr, true);
  std::promise<void> pending;
  auto cleanup = pending.get_future();
  std::vector<char> out;
  ASSERT_TRUE(dispatch.Read("logical-name", out, path_.u8string(), &cleanup).IsOK());
  EXPECT_EQ(cleanup.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  ExpectSourceReleased();
  EXPECT_EQ(std::string(out.begin(), out.end()), "data");
  pending.set_value();
  cleanup.get();
}

TEST_F(QnnDecryptSourceTest, InitialLoadWaitsForCleanupOnSuccessSdkErrorMetadataErrorAndException) {
  qnn::EpContextIoDispatch dispatch(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION}, nullptr, true);
  StubApiEnv env;
  env.stub_ort_api = Ort::GetApi();
  InstallFakeGraphApiStubs(env.stub_ort_api);
  OrtGlobalApiOverride api_guard{&env.stub_ort_api};
  FakeOpAttr embed = FakeOpAttr::MakeInt64(qnn::EMBED_MODE, 0);
  FakeOpAttr path = FakeOpAttr::MakeString(qnn::EP_CACHE_CONTEXT, path_.filename().u8string());
  FakeNode node{"ep_ctx", "EPContext", "", 1, {}, {}};
  node.attrs[qnn::EMBED_MODE] = &embed;
  node.attrs[qnn::EP_CACHE_CONTEXT] = &path;
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  QnnSystemContext_GraphInfo_t graph_info{};
  graph_info.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
  graph_info.graphInfoV1.graphName = "ep_ctx";
  graph_info.graphInfoV1.graphInputs = &tensor;
  graph_info.graphInfoV1.graphOutputs = &tensor;
  QnnSystemContext_BinaryInfo_t binary_info{};
  binary_info.version = QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1;
  binary_info.contextBinaryInfoV1.numGraphs = 1;
  binary_info.contextBinaryInfoV1.graphs = &graph_info;
  static thread_local SourceState* active_state = nullptr;
  static thread_local QnnSystemContext_BinaryInfo_t* active_info = nullptr;
  static thread_local int outcome = 0;
  static thread_local size_t sdk_calls = 0;
  active_state = &state_;
  active_info = &binary_info;
  for (int next_outcome : {0, 1, 2, 3}) {
    SCOPED_TRACE(next_outcome);
    outcome = next_outcome;
    sdk_calls = 0;
    StubBackendManager backend(env.api_ptrs, env.logger);
    backend.SystemInterface().systemContextCreate = [](QnnSystemContext_Handle_t* handle) -> Qnn_ErrorHandle_t {
      *handle = active_info;
      return QNN_SUCCESS;
    };
    backend.SystemInterface().systemContextFree = [](QnnSystemContext_Handle_t) -> Qnn_ErrorHandle_t {
      return QNN_SUCCESS;
    };
    backend.SystemInterface().systemContextGetBinaryInfo = [](
                                                               QnnSystemContext_Handle_t handle, void* data, Qnn_ContextBinarySize_t size,
                                                               const QnnSystemContext_BinaryInfo_t** info, Qnn_ContextBinarySize_t* info_size) -> Qnn_ErrorHandle_t {
      EXPECT_EQ(size, 4u);
      EXPECT_EQ(std::memcmp(data, "data", 4), 0);
      EXPECT_NE(data, active_state->source);
      *info = static_cast<QnnSystemContext_BinaryInfo_t*>(handle);
      *info_size = sizeof(**info);
      return outcome == 3 ? QNN_COMMON_ERROR_GENERAL : QNN_SUCCESS;
    };
    backend.QnnInterface().contextCreateFromBinary = [](
                                                         Qnn_BackendHandle_t, Qnn_DeviceHandle_t, const QnnContext_Config_t**, const void* data,
                                                         Qnn_ContextBinarySize_t size, Qnn_ContextHandle_t* context, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      ++sdk_calls;
      EXPECT_EQ(size, 4u);
      EXPECT_EQ(std::memcmp(data, "data", 4), 0);
      EXPECT_NE(data, active_state->source);
      EXPECT_NE(data, active_state->output);
      if (outcome == 2) throw std::runtime_error("Test SDK exception.");
      if (outcome == 1) return QNN_COMMON_ERROR_GENERAL;
      *context = active_state;
      return QNN_SUCCESS;
    };
    backend.QnnInterface().contextFree = [](Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      return QNN_SUCCESS;
    };
    backend.QnnInterface().graphRetrieve = [](Qnn_ContextHandle_t context, const char*,
                                              Qnn_GraphHandle_t* graph) -> Qnn_ErrorHandle_t {
      *graph = context;
      return QNN_SUCCESS;
    };
    backend.QnnInterface().errorGetMessage = [](Qnn_ErrorHandle_t, const char** message) -> Qnn_ErrorHandle_t {
      *message = "Test SDK error.";
      return QNN_SUCCESS;
    };
    qnn::QnnModelLookupTable models;
    auto load = [&] {
      return qnn::GetEpContextFromMainNode(node.AsNode(), env.stub_ort_api,
                                           (path_.parent_path() / L"model.onnx").wstring(),
                                           backend.GetMutable(), models, 0, dispatch, &logger_);
    };
    if (outcome == 2) {
      EXPECT_THROW(load(), std::runtime_error);
    } else {
      auto status = load();
      EXPECT_EQ(status.IsOK(), outcome == 0);
      EXPECT_EQ(models.size(), outcome == 0 ? 1u : 0u);
    }
    EXPECT_EQ(sdk_calls, outcome == 3 ? 0u : 1u);
    ExpectSourceReleased();
  }
  EXPECT_EQ(state_.source_calls, 4u);
  EXPECT_EQ(state_.legacy_calls, 0u);
  active_state = nullptr;
  active_info = nullptr;
}

}  // namespace
}  // namespace onnxruntime::test
#endif
