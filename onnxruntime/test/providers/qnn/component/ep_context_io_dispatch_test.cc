// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for EpContextIoDispatch (ep_context_io_dispatch.cc).
//
// Covers the runtime-capability guard added to fix an access violation that
// occurred when a QNN EP built against ORT 1.29+ headers was loaded by an older
// (<1.28) ORT runtime: the experimental EPContext encryption APIs don't exist in
// that runtime's OrtApi struct, so touching Ort::Experimental::EpContextConfig
// reads past the end of the struct instead of cleanly throwing.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <unordered_set>

#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/qnn_model.h"
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#endif
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/infra/qnn_fake_ort_graph.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

namespace onnxruntime {
namespace test {

namespace {
// Minimal OrtApi stub: GetExperimentalFunction always reports "unavailable",
// standing in for a runtime that doesn't have the v28 experimental vtable.
// If EpContextIoDispatch's effective_api_version guard is skipped and it
// calls into Ort::Experimental::EpContextConfig anyway, this stub throws
// ORT_NOT_IMPLEMENTED (via GetExperimentalFunction returning nullptr) rather
// than crashing, so the guard itself is exercised through the call counter
// below, not through a real ABI mismatch.
struct ExperimentalFnStubEnv {
  OrtApi stub_ort_api{};
  int get_experimental_function_calls = 0;

  ExperimentalFnStubEnv() {
    static thread_local ExperimentalFnStubEnv* current = nullptr;
    current = this;
    stub_ort_api.GetExperimentalFunction = [](const char*) noexcept -> OrtExperimentalFnPtr {
      if (current != nullptr) ++current->get_experimental_function_calls;
      return nullptr;
    };
  }
};
}  // namespace

// A runtime without the named EPContext-encryption capability must short-circuit
// before ever touching the experimental API, even with a non-null session_options.
TEST(QnnUnit_EpContextIoDispatchTest, EpContextEncryptionUnsupported_NeverTouchesExperimentalApi) {
  OrtSessionOptions* session_options = nullptr;
  ASSERT_EQ(Ort::GetApi().CreateSessionOptions(&session_options), nullptr);
  ASSERT_NE(session_options, nullptr);

  {
    ExperimentalFnStubEnv env;
    OrtGlobalApiOverride api_override(&env.stub_ort_api);
    const OrtApiRuntimeCapabilities runtime_capabilities{
        OrtApiRuntimeCapabilities::kEpContextEncryptionMinApiVersion - 1};

    qnn::EpContextIoDispatch dispatch(*session_options, runtime_capabilities);

    EXPECT_EQ(env.get_experimental_function_calls, 0);
    EXPECT_FALSE(dispatch.HasReadCallback());
    EXPECT_FALSE(dispatch.HasWriteCallback());
  }

  Ort::GetApi().ReleaseSessionOptions(session_options);
}

// The default constructor represents a disabled/no-session dispatch. It must not require a
// fabricated runtime version, or callers could accidentally depend on the session constructor's
// short-circuit order.
TEST(QnnUnit_EpContextIoDispatchTest, DefaultConstructor_IsDisabled) {
  qnn::EpContextIoDispatch dispatch;

  EXPECT_FALSE(dispatch.HasReadCallback());
  EXPECT_FALSE(dispatch.HasWriteCallback());
}

TEST(QnnUnit_EpContextIoDispatchTest, DisabledDispatch_ReadAndWrite_ReturnFailures) {
  qnn::EpContextIoDispatch dispatch;
  qnn::EpContextBuffer bytes;
  Ort::Logger logger;

  EXPECT_FALSE(dispatch.Read("context.bin", bytes).IsOK());
  EXPECT_FALSE(dispatch.Write("context.bin", nullptr, 0, logger).IsOK());
}

// The named EPContext-encryption capability proceeds to resolve callbacks
// (and, with no experimental API registered by this stub, fails via the
// existing try/catch rather than the version guard).
TEST(QnnUnit_EpContextIoDispatchTest, EpContextEncryptionSupported_AttemptsExperimentalApi) {
  OrtSessionOptions* session_options = nullptr;
  ASSERT_EQ(Ort::GetApi().CreateSessionOptions(&session_options), nullptr);
  ASSERT_NE(session_options, nullptr);

  {
    ExperimentalFnStubEnv env;
    OrtGlobalApiOverride api_override(&env.stub_ort_api);
    const OrtApiRuntimeCapabilities runtime_capabilities{
        OrtApiRuntimeCapabilities::kEpContextEncryptionMinApiVersion};

    qnn::EpContextIoDispatch dispatch(*session_options, runtime_capabilities);

    EXPECT_GT(env.get_experimental_function_calls, 0);
    EXPECT_FALSE(dispatch.HasReadCallback());
    EXPECT_FALSE(dispatch.HasWriteCallback());
  }

  Ort::GetApi().ReleaseSessionOptions(session_options);
}

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION

// Keep private access local while exercising the production VTCM caller.
struct CallbackListMethodTag {
  using type = Ort::Status (qnn::QnnBackendManager::*)(
      std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>>&,
      const qnn::EpContextIoDispatch&);
  friend type GetCallbackBackendMember(CallbackListMethodTag);
};
struct CallbackBackendSetupTag {
  using type = bool qnn::QnnBackendManager::*;
  friend type GetCallbackBackendMember(CallbackBackendSetupTag);
};
struct CallbackMappingFlagTag {
  using type = bool qnn::QnnBackendManager::*;
  friend type GetCallbackBackendMember(CallbackMappingFlagTag);
};
template <typename Tag, typename Tag::type Member>
struct CallbackBackendMemberAccess {
  friend typename Tag::type GetCallbackBackendMember(Tag) { return Member; }
};
template struct CallbackBackendMemberAccess<CallbackListMethodTag,
                                            &qnn::QnnBackendManager::CreateContextVtcmBackupBufferSharingEnabled>;
template struct CallbackBackendMemberAccess<CallbackBackendSetupTag,
                                            &qnn::QnnBackendManager::backend_setup_completed_>;
template struct CallbackBackendMemberAccess<CallbackMappingFlagTag,
                                            &qnn::QnnBackendManager::file_mapped_weights_enabled_>;

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
struct CallbackFileMapperTag {
  using type = std::unique_ptr<qnn::FileMappingInterface> qnn::QnnBackendManager::*;
  friend type GetCallbackBackendMember(CallbackFileMapperTag);
};
struct CallbackSharingModeTag {
  using type = uint8_t qnn::QnnBackendManager::*;
  friend type GetCallbackBackendMember(CallbackSharingModeTag);
};
template struct CallbackBackendMemberAccess<CallbackFileMapperTag,
                                            &qnn::QnnBackendManager::file_mapper_>;
template struct CallbackBackendMemberAccess<CallbackSharingModeTag,
                                            &qnn::QnnBackendManager::htp_share_resource_optimization_>;
#endif

// Track the supplied allocator so a copy, premature free, or double free fails.
class QnnUnit_CallbackBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_ = this;
    api_ = Ort::GetApi();
    ASSERT_EQ(api_.GetAllocatorWithDefaultOptions(&backing_), nullptr);
    allocator_ = *backing_;
    allocator_.Alloc = [](OrtAllocator*, size_t size) -> void* {
      auto& test = *current_;
      void* data = test.backing_->Alloc(test.backing_, size);
      if (data) test.live_.insert(data);
      return data;
    };
    allocator_.Free = [](OrtAllocator* allocator, void* data) {
      auto& test = *current_;
      EXPECT_EQ(allocator, &test.allocator_);
      if (test.live_.erase(data) != 1) {
        ADD_FAILURE() << "Allocation was freed twice or with the wrong allocator";
        return;
      }
      ++test.frees_;
      test.events_.push_back("free");
      test.backing_->Free(test.backing_, data);
    };
    api_.GetAllocatorWithDefaultOptions = [](OrtAllocator** allocator) noexcept -> OrtStatus* {
      auto& test = *current_;
      if (test.allocator_failure_) {
        return test.api_.CreateStatus(ORT_FAIL, "allocator lookup failure");
      }
      *allocator = &test.allocator_;
      return nullptr;
    };
    InstallFakeGraphApiStubs(api_);
    auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&api_);
    ASSERT_NE(setter, nullptr) << "Callback tests require the v28 runtime API";
    Ort::ThrowOnError(setter(options_, Read, this));
    api_override_ = std::make_unique<OrtGlobalApiOverride>(&api_);
    dispatch_ = std::make_unique<qnn::EpContextIoDispatch>(*options_, OrtApiRuntimeCapabilities{ORT_API_VERSION});
    ASSERT_TRUE(dispatch_->HasReadCallback());
    backend_ = std::make_unique<StubBackendManager>(api_ptrs_, logger_);
    auto& sdk = backend_->QnnInterface();
    sdk.errorGetMessage = [](Qnn_ErrorHandle_t, const char** message) -> Qnn_ErrorHandle_t {
      *message = "stub QNN failure";
      return QNN_SUCCESS;
    };
    sdk.contextFree = [](Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      return QNN_SUCCESS;
    };
    sdk.contextCreateFromBinary = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                     const QnnContext_Config_t**, const void* data,
                                     Qnn_ContextBinarySize_t size, Qnn_ContextHandle_t* context,
                                     Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      test.CheckBinary(data, size);
      ++test.ordinary_calls_;
      test.events_.push_back("ordinary");
      if (test.sdk_result_ == QNN_SUCCESS) {
        *context = reinterpret_cast<Qnn_ContextHandle_t>(
            static_cast<uintptr_t>(100 + test.ordinary_calls_ + test.mapped_calls_));
      }
      return test.sdk_result_;
    };
    sdk.graphRetrieve = [](Qnn_ContextHandle_t context, const char* name,
                           Qnn_GraphHandle_t* graph) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      test.CheckBinary(test.metadata_buffer_, test.metadata_size_);
      EXPECT_STREQ(name, test.native_reads_ ? "native_graph" : "callback_graph");
      test.events_.push_back("graph");
      *graph = context;
      return test.graph_result_;
    };
    sdk.contextCreateFromBinaryListAsync = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                              const QnnContext_Params_t** params,
                                              const QnnContext_Config_t**, Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      ++test.list_calls_;
      EXPECT_EQ(test.live_.size(), test.outputs_.size());
      size_t count = 0;
      for (; params[count]; ++count) {
        const auto& item = params[count]->v1;
        EXPECT_EQ(item.binaryBuffer, test.outputs_.at(count));
        EXPECT_EQ(item.binaryBufferSize, 32u);
        auto context = reinterpret_cast<Qnn_ContextHandle_t>(static_cast<uintptr_t>(count + 1));
        item.notifyFunc(context, nullptr, "callback_graph", QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT,
                        item.notifyParam, QNN_CONTEXT_NO_ERROR);
        item.notifyFunc(context, nullptr, nullptr, QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT,
                        item.notifyParam, QNN_CONTEXT_NO_ERROR);
        // All allocations survive every notification, including the final one.
        EXPECT_EQ(test.live_.size(), test.outputs_.size());
        for (auto* output : test.outputs_) EXPECT_STREQ(static_cast<const char*>(output), "callback_graph");
      }
      EXPECT_EQ(count, test.outputs_.size());
      return test.sdk_result_;
    };
    auto& system = backend_->SystemInterface();
    system.systemContextCreate = [](QnnSystemContext_Handle_t* handle) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      EXPECT_EQ(test.live_.size(), test.native_reads_ ? 0u : 1u);
      test.events_.push_back("system_create");
      *handle = &test;
      return QNN_SUCCESS;
    };
    system.systemContextGetBinaryInfo = [](QnnSystemContext_Handle_t, void* data, uint64_t size,
                                           const QnnSystemContext_BinaryInfo_t** info,
                                           Qnn_ContextBinarySize_t* info_size) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      test.CheckBinary(data, size);
      test.metadata_buffer_ = data;
      test.metadata_size_ = size;
      test.events_.push_back("metadata");
      if (test.metadata_result_ != QNN_SUCCESS) return test.metadata_result_;
      test.input_.version = QNN_TENSOR_VERSION_1;
      test.input_.v1.name = "input";
      test.input_.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
      test.input_.v1.dataType = QNN_DATATYPE_FLOAT_32;
      test.input_.v1.rank = 1;
      test.input_.v1.dimensions = &test.dimension_;
      test.output_ = test.input_;
      test.output_.v1.name = "output";
      test.output_.v1.type = QNN_TENSOR_TYPE_APP_READ;
      test.graph_info_.version = QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1;
      auto& graph = test.graph_info_.graphInfoV1;
      graph.graphName = static_cast<const char*>(data);
      graph.numGraphInputs = 1;
      graph.graphInputs = &test.input_;
      graph.numGraphOutputs = 1;
      graph.graphOutputs = &test.output_;
      test.binary_info_.version = QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1;
      auto& binary = test.binary_info_.contextBinaryInfoV1;
      binary.contextBlobVersion = test.binary_blob_version_;
      binary.numGraphs = 1;
      binary.graphs = &test.graph_info_;
      *info = &test.binary_info_;
      *info_size = sizeof(test.binary_info_);
      return QNN_SUCCESS;
    };
    system.systemContextFree = [](QnnSystemContext_Handle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      test.CheckBinary(test.metadata_buffer_, test.metadata_size_);
      test.events_.push_back("system_free");
      return QNN_SUCCESS;
    };
  }

  void TearDown() override {
    models_.clear();
    backend_.reset();
    for (const auto& path : native_files_) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
      EXPECT_FALSE(ec);
    }
    dispatch_.reset();
    EXPECT_TRUE(live_.empty());
    for (auto* data : live_) backing_->Free(backing_, data);
    live_.clear();
    api_override_.reset();
    current_ = nullptr;
  }

  static OrtStatus* ORT_API_CALL Read(void* opaque, const char* name, OrtAllocator* allocator,
                                      void** data, size_t* size) {
    auto& test = *static_cast<QnnUnit_CallbackBufferTest*>(opaque);
    EXPECT_EQ(allocator, &test.allocator_);
    ++test.read_calls_;
    test.read_names_.push_back(name);
    if (test.read_calls_ == test.fail_read_) return test.api_.CreateStatus(ORT_FAIL, "read failure");
    *size = 32;
    *data = test.null_output_ ? nullptr : allocator->Alloc(allocator, *size);
    if (*data) {
      std::memset(*data, 0, *size);
      std::memcpy(*data, "callback_graph", sizeof("callback_graph"));
    }
    if (test.callback_failure_) {
      allocator->Free(allocator, *data);
      return test.api_.CreateStatus(ORT_INVALID_ARGUMENT, "decoder failure");
    }
    if (test.empty_output_) *size = 0;
    if (*data && *size) {
      test.outputs_.push_back(*data);
      test.events_.push_back("read");
    }
    return nullptr;
  }

  void CheckBinary(const void* data, uint64_t size) {
    if (native_reads_) {
      CheckNativeBinary(data, size);
      return;
    }
    EXPECT_EQ(data, outputs_.back());
    EXPECT_EQ(size, 32u);
    ASSERT_EQ(live_.count(const_cast<void*>(data)), 1u);
    EXPECT_STREQ(static_cast<const char*>(data), "callback_graph");
  }

  std::string WriteNativeBinary(const char* name) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("qnn_callback_owner_" + std::to_string(reinterpret_cast<uintptr_t>(this)) +
                       "_" + name + ".bin");
    native_files_.push_back(path);
    std::ofstream file(path, std::ios::binary);
    file.write("native_graph", sizeof("native_graph"));
    EXPECT_TRUE(file.good());
    return path.string();
  }

  static void CheckNativeBinary(const void* data, uint64_t size) {
    EXPECT_EQ(size, sizeof("native_graph"));
    EXPECT_STREQ(static_cast<const char*>(data), "native_graph");
  }

  void UseNativeReads() {
    native_reads_ = true;
    auto& sdk = backend_->QnnInterface();
    sdk.contextCreateFromBinary = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                     const QnnContext_Config_t**, const void* data,
                                     Qnn_ContextBinarySize_t size, Qnn_ContextHandle_t* context,
                                     Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      CheckNativeBinary(data, size);
      test.ordinary_buffer_ = data;
      ++test.ordinary_calls_;
      if (test.sdk_result_ == QNN_SUCCESS) {
        *context = reinterpret_cast<Qnn_ContextHandle_t>(
            static_cast<uintptr_t>(100 + test.ordinary_calls_ + test.mapped_calls_));
      }
      return test.sdk_result_;
    };
    sdk.contextCreateFromBinaryListAsync = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                              const QnnContext_Params_t** params,
                                              const QnnContext_Config_t**, Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      ++test.list_calls_;
      const bool mapped = params[0] && params[0]->version == QNN_CONTEXT_PARAMS_VERSION_2;
      if (mapped) ++test.mapped_list_calls_;
      for (size_t count = 0; params[count]; ++count) {
        const auto* data = mapped ? params[count]->v2->binaryBuffer : params[count]->v1.binaryBuffer;
        const auto size = mapped ? params[count]->v2->binaryBufferSize : params[count]->v1.binaryBufferSize;
        auto notify = mapped ? params[count]->v2->notifyFunc : params[count]->v1.notifyFunc;
        auto notify_param = mapped ? params[count]->v2->notifyParam : params[count]->v1.notifyParam;
        CheckNativeBinary(data, size);
        if (mapped) {
          EXPECT_NE(params[count]->v2->callback, nullptr);
        }
        auto context = reinterpret_cast<Qnn_ContextHandle_t>(
            static_cast<uintptr_t>(1000 + test.list_calls_ * 10 + count));
        notify(context, nullptr, "native_graph", QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT,
               notify_param, QNN_CONTEXT_NO_ERROR);
        notify(context, nullptr, nullptr, QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT,
               notify_param, QNN_CONTEXT_NO_ERROR);
        CheckNativeBinary(data, size);
      }
      return test.sdk_result_;
    };
  }

  Ort::Status NativeInitialLoad(const std::string& filename,
                                const qnn::EpContextIoDispatch& io_dispatch) {
    const auto path = std::filesystem::path(filename);
    FakeOpAttr embed = FakeOpAttr::MakeInt64(qnn::EMBED_MODE, 0);
    FakeOpAttr context_path = FakeOpAttr::MakeString(qnn::EP_CACHE_CONTEXT, path.filename().string());
    FakeNode node{"native_node", "EPContext", "", 1, {}, {}};
    node.attrs[qnn::EMBED_MODE] = &embed;
    node.attrs[qnn::EP_CACHE_CONTEXT] = &context_path;
    const auto model_path = (path.parent_path() / "model.onnx").native();
    return qnn::GetEpContextFromMainNode(node.AsNode(), api_, model_path,
                                         backend_->GetMutable(), models_, 0, io_dispatch, &logger_);
  }

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  void UseNativeMappingSdk() {
    UseNativeReads();
    backend_->QnnInterface().contextCreateFromBinaryWithCallback =
        [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t, const QnnContext_Config_t**,
           const Qnn_ContextBinaryCallback_t*, const void* data, Qnn_ContextBinarySize_t size,
           Qnn_ContextHandle_t* context, Qnn_ProfileHandle_t, Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      CheckNativeBinary(data, size);
      EXPECT_EQ(data, test.metadata_buffer_);
      ++test.mapped_calls_;
      if (test.mapped_result_ == QNN_SUCCESS) {
        *context = reinterpret_cast<Qnn_ContextHandle_t>(
            static_cast<uintptr_t>(100 + test.ordinary_calls_ + test.mapped_calls_));
      }
      return test.mapped_result_;
    };
  }
#endif

  Ort::Status InitialLoad() {
    FakeOpAttr embed = FakeOpAttr::MakeInt64(qnn::EMBED_MODE, 0);
    FakeOpAttr path = FakeOpAttr::MakeString(qnn::EP_CACHE_CONTEXT, "callback.bin");
    FakeNode node{"callback_node", "EPContext", "", 1, {}, {}};
    node.attrs[qnn::EMBED_MODE] = &embed;
    node.attrs[qnn::EP_CACHE_CONTEXT] = &path;
    return qnn::GetEpContextFromMainNode(node.AsNode(), api_, ORT_TSTR("not-existing/model.onnx"),
                                         backend_->GetMutable(), models_, 0, *dispatch_, &logger_);
  }

  Ort::Status ListLoad() {
    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
    for (const auto* name : {"one", "two", "three"}) {
      binaries.emplace(std::string("not-existing/") + name + ".bin",
                       std::make_unique<std::vector<std::string>>(1, name));
    }
    return (backend_->GetMutable()->*GetCallbackBackendMember(CallbackListMethodTag{}))(binaries, *dispatch_);
  }

  static thread_local QnnUnit_CallbackBufferTest* current_;
  OrtApi api_{};
  OrtEpApi ep_api_{};
  OrtModelEditorApi editor_api_{};
  ApiPtrs api_ptrs_{api_, ep_api_, editor_api_};
  OrtAllocator* backing_ = nullptr;
  OrtAllocator allocator_{};
  Ort::Logger logger_{MakeNullLogger()};
  Ort::SessionOptions options_;
  std::unique_ptr<OrtGlobalApiOverride> api_override_;
  std::unique_ptr<qnn::EpContextIoDispatch> dispatch_;
  std::unique_ptr<StubBackendManager> backend_;
  qnn::QnnModelLookupTable models_;
  std::vector<std::filesystem::path> native_files_;
  std::unordered_set<void*> live_;
  std::vector<void*> outputs_;
  std::vector<std::string> read_names_, events_;
  size_t read_calls_ = 0, frees_ = 0, ordinary_calls_ = 0, list_calls_ = 0, fail_read_ = 0;
  size_t mapped_calls_ = 0, mapped_list_calls_ = 0;
  const void* metadata_buffer_ = nullptr;
  const void* ordinary_buffer_ = nullptr;
  uint64_t metadata_size_ = 0;
  bool native_reads_ = false;
  Qnn_ErrorHandle_t mapped_result_ = QNN_SUCCESS;
  Qnn_Version_t binary_blob_version_ = {3, 3, 3};
  bool null_output_ = false, empty_output_ = false, callback_failure_ = false, allocator_failure_ = false;
  Qnn_ErrorHandle_t sdk_result_ = QNN_SUCCESS, metadata_result_ = QNN_SUCCESS, graph_result_ = QNN_SUCCESS;
  uint32_t dimension_ = 1;
  Qnn_Tensor_t input_ = QNN_TENSOR_INIT, output_ = QNN_TENSOR_INIT;
  QnnSystemContext_GraphInfo_t graph_info_ = QNN_SYSTEM_CONTEXT_GRAPH_INFO_INIT;
  QnnSystemContext_BinaryInfo_t binary_info_{};
};
thread_local QnnUnit_CallbackBufferTest* QnnUnit_CallbackBufferTest::current_ = nullptr;

TEST_F(QnnUnit_CallbackBufferTest, AdoptReplacementAndMoveFreeEachAllocationOnce) {
  static_assert(!std::is_copy_constructible_v<qnn::EpContextBuffer>);
  static_assert(std::is_nothrow_move_constructible_v<qnn::EpContextBuffer>);
  qnn::EpContextBuffer first, second;
  ASSERT_TRUE(dispatch_->Read("first.bin", first).IsOK());
  EXPECT_EQ(first.data(), outputs_.back());
  ASSERT_TRUE(dispatch_->Read("replacement.bin", first).IsOK());
  EXPECT_EQ(first.data(), outputs_.back());
  EXPECT_EQ(frees_, 1u);
  ASSERT_TRUE(dispatch_->Read("second.bin", second).IsOK());
  auto* retained = first.data();
  qnn::EpContextBuffer moved(std::move(first));
  EXPECT_EQ(first.data(), nullptr);
  EXPECT_EQ(first.size(), 0u);
  second = std::move(moved);
  EXPECT_EQ(second.data(), retained);
  EXPECT_EQ(second.size(), 32u);
  EXPECT_EQ(moved.size(), 0u);
  EXPECT_EQ(frees_, 2u);
  EXPECT_EQ(live_.size(), 1u);
  second = {};
  EXPECT_EQ(frees_, 3u);
}

TEST_F(QnnUnit_CallbackBufferTest, InvalidSuccessfulOutputPreservesPreviousOutput) {
  qnn::EpContextBuffer output;
  ASSERT_TRUE(dispatch_->Read("valid.bin", output).IsOK());
  auto* retained = output.data();
  null_output_ = true;
  EXPECT_FALSE(dispatch_->Read("null.bin", output).IsOK());
  EXPECT_EQ(frees_, 0u);
  null_output_ = false;
  empty_output_ = true;
  EXPECT_FALSE(dispatch_->Read("empty.bin", output).IsOK());
  EXPECT_EQ(frees_, 1u);
  EXPECT_EQ(output.data(), retained);
  EXPECT_EQ(output.size(), 32u);
  EXPECT_STREQ(output.data(), "callback_graph");
}

TEST_F(QnnUnit_CallbackBufferTest, CallbackErrorPreservesStatusAndApplicationCleanup) {
  qnn::EpContextBuffer output;
  ASSERT_TRUE(dispatch_->Read("valid.bin", output).IsOK());
  auto* retained = output.data();
  callback_failure_ = true;
  auto status = dispatch_->Read("failed.bin", output);
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  EXPECT_EQ(status.GetErrorMessage(), "decoder failure");
  EXPECT_EQ(output.data(), retained);
  EXPECT_EQ(frees_, 1u);
  EXPECT_EQ(live_.size(), 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, AllocatorFailureDoesNotInvokeCallbackOrReplaceOutput) {
  qnn::EpContextBuffer output;
  ASSERT_TRUE(dispatch_->Read("valid.bin", output).IsOK());
  auto* retained = output.data();
  allocator_failure_ = true;
  auto status = dispatch_->Read("failed.bin", output);
  EXPECT_EQ(status.GetErrorMessage(), "allocator lookup failure");
  EXPECT_EQ(read_calls_, 1u);
  EXPECT_EQ(output.data(), retained);
  EXPECT_EQ(frees_, 0u);
}

TEST_F(QnnUnit_CallbackBufferTest, InitialLoadUsesOriginalPointerThroughGraphMetadata) {
  ASSERT_TRUE(InitialLoad().IsOK());
  EXPECT_EQ(read_names_, (std::vector<std::string>{"callback.bin"}));
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(models_.at("callback_node")->Name(), "callback_graph");
  EXPECT_EQ(events_, (std::vector<std::string>{"read", "system_create", "metadata", "ordinary", "graph", "system_free", "free"}));
  EXPECT_EQ(frees_, 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, InitialMetadataFailureFreesOutputOnce) {
  metadata_result_ = QNN_COMMON_ERROR_GENERAL;
  EXPECT_FALSE(InitialLoad().IsOK());
  EXPECT_EQ(ordinary_calls_, 0u);
  EXPECT_TRUE(models_.empty());
  EXPECT_EQ(events_.back(), "free");
  EXPECT_EQ(frees_, 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, InitialContextFailureFreesOutputOnce) {
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  EXPECT_FALSE(InitialLoad().IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_TRUE(models_.empty());
  EXPECT_EQ(events_.back(), "free");
  EXPECT_EQ(frees_, 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, InitialGraphFailureFreesOutputOnce) {
  graph_result_ = QNN_COMMON_ERROR_GENERAL;
  EXPECT_FALSE(InitialLoad().IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_TRUE(models_.empty());
  EXPECT_EQ(events_.back(), "free");
  EXPECT_EQ(frees_, 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, SsrUsesOriginalCallbackPointerAndFreesAfterCreation) {
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(backend_->GetMutable()->ReloadContextForSSR("not-existing/nested/callback.bin", 0, context, *dispatch_).IsOK());
  EXPECT_EQ(read_names_, (std::vector<std::string>{"callback.bin"}));
  EXPECT_EQ(events_, (std::vector<std::string>{"read", "ordinary", "free"}));
  EXPECT_EQ(frees_, 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, SsrContextFailureFreesOutputOnce) {
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  Qnn_ContextHandle_t context = nullptr;
  EXPECT_FALSE(backend_->GetMutable()->ReloadContextForSSR("callback.bin", 0, context, *dispatch_).IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(frees_, 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, ListKeepsAllOriginalPointersThroughEveryNotification) {
  ASSERT_TRUE(ListLoad().IsOK());
  EXPECT_EQ(list_calls_, 1u);
  EXPECT_EQ(outputs_.size(), 3u);
  EXPECT_EQ(frees_, 3u);
  EXPECT_EQ(backend_->GetMutable()->GetQnnContextSize(), 3u);
}

TEST_F(QnnUnit_CallbackBufferTest, ListSdkFailureFreesEveryOutputOnce) {
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  EXPECT_FALSE(ListLoad().IsOK());
  EXPECT_EQ(list_calls_, 1u);
  EXPECT_EQ(frees_, 3u);
}

TEST_F(QnnUnit_CallbackBufferTest, ListReadFailureReleasesPreviouslyAdoptedOutput) {
  fail_read_ = 2;
  EXPECT_FALSE(ListLoad().IsOK());
  EXPECT_EQ(read_calls_, 2u);
  EXPECT_EQ(list_calls_, 0u);
  EXPECT_EQ(frees_, 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, NativeSsrBufferSurvivesSdkAndPreservesFailures) {
  UseNativeReads();
  const auto path = WriteNativeBinary("ssr");
  const qnn::EpContextIoDispatch native_dispatch;
  const Qnn_ErrorHandle_t results[] = {QNN_SUCCESS, QNN_CONTEXT_ERROR_INVALID_ARGUMENT};
  for (auto result : results) {
    sdk_result_ = result;
    Qnn_ContextHandle_t context = nullptr;
    auto status = backend_->GetMutable()->ReloadContextForSSR(path, 0, context, native_dispatch);
    EXPECT_EQ(status.IsOK(), result == QNN_SUCCESS);
    if (result != QNN_SUCCESS) {
      EXPECT_EQ(status.GetErrorCode(), ORT_EP_FAIL);
    }
  }
  EXPECT_EQ(ordinary_calls_, 2u);
  EXPECT_TRUE(std::filesystem::remove(path));
  Qnn_ContextHandle_t context = nullptr;
  auto status = backend_->GetMutable()->ReloadContextForSSR(path, 0, context, native_dispatch);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(ordinary_calls_, 2u);
  EXPECT_EQ(read_calls_, 0u);
  EXPECT_EQ(frees_, 0u);
}

TEST_F(QnnUnit_CallbackBufferTest, NativeListBuffersSurviveNotificationsAndReadFailureSkipsSdk) {
  UseNativeReads();
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
  for (const auto* name : {"one", "two"}) {
    binaries.emplace(WriteNativeBinary(name), std::make_unique<std::vector<std::string>>(1, name));
  }
  const qnn::EpContextIoDispatch native_dispatch;
  const Qnn_ErrorHandle_t results[] = {QNN_SUCCESS, QNN_CONTEXT_ERROR_INVALID_ARGUMENT};
  for (auto result : results) {
    sdk_result_ = result;
    auto status = (backend_->GetMutable()->*GetCallbackBackendMember(CallbackListMethodTag{}))(binaries, native_dispatch);
    EXPECT_EQ(status.IsOK(), result == QNN_SUCCESS);
    if (result != QNN_SUCCESS) {
      EXPECT_EQ(status.GetErrorCode(), ORT_EP_FAIL);
    }
  }
  EXPECT_EQ(list_calls_, 2u);
  auto missing = binaries.begin();
  ++missing;  // Fail the second read after the first native buffer has been loaded.
  EXPECT_TRUE(std::filesystem::remove(missing->first));
  auto status = (backend_->GetMutable()->*GetCallbackBackendMember(CallbackListMethodTag{}))(binaries, native_dispatch);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(list_calls_, 2u);
  EXPECT_EQ(read_calls_, 0u);
  EXPECT_EQ(frees_, 0u);
}

TEST_F(QnnUnit_CallbackBufferTest, SharedManagerCallbackDoesNotClearMappingFlag) {
  auto* manager = backend_->GetMutable();
  manager->*GetCallbackBackendMember(CallbackBackendSetupTag{}) = true;
  manager->*GetCallbackBackendMember(CallbackMappingFlagTag{}) = true;
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  manager->*GetCallbackBackendMember(CallbackFileMapperTag{}) = std::make_unique<qnn::WindowsFileMapper>(logger_);
#endif
  const qnn::EpContextIoDispatch native_dispatch;
  const qnn::EpContextIoDispatch* sequence[] = {&native_dispatch, dispatch_.get(), &native_dispatch};
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
  for (const auto* io_dispatch : sequence) {
    ASSERT_TRUE(manager->SetupBackend(true, false, true, 0, true, nullptr, binaries, *io_dispatch).IsOK());
    EXPECT_TRUE(manager->*GetCallbackBackendMember(CallbackMappingFlagTag{}));
  }
  EXPECT_EQ(read_calls_, 0u);
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

TEST_F(QnnUnit_CallbackBufferTest, SharedManagerKeepsDisabledMappingForNativeInitialAndList) {
  auto* manager = backend_->GetMutable();
  backend_->BackendType() = qnn::QnnBackendType::HTP;
  manager->*GetCallbackBackendMember(CallbackBackendSetupTag{}) = true;
  const auto backend_handle = backend_->BackendHandle();
  Ort::SessionOptions native_options;
  const OrtApiRuntimeCapabilities capabilities{ORT_API_VERSION};
  const qnn::EpContextIoDispatch disabled(*native_options, capabilities);
  const qnn::EpContextIoDispatch enabled(*native_options, capabilities);
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;

  ASSERT_TRUE(manager->SetupBackend(true, false, true, 0, false, nullptr, binaries, disabled).IsOK());
  EXPECT_FALSE(manager->FileMappingIsEnabled());
  EXPECT_FALSE(manager->SetupBackend(true, false, true, 0, true, nullptr, binaries, enabled).IsOK());
  EXPECT_FALSE(manager->FileMappingIsEnabled());
  EXPECT_EQ(backend_->BackendHandle(), backend_handle);
  EXPECT_EQ((manager->*GetCallbackBackendMember(CallbackFileMapperTag{})).get(), nullptr);
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 0, false, nullptr, binaries, disabled).IsOK());
  EXPECT_FALSE(manager->FileMappingIsEnabled());

  UseNativeMappingSdk();
  const auto path = WriteNativeBinary("disabled_native");
  ASSERT_TRUE(NativeInitialLoad(path, disabled).IsOK());
  binaries.emplace(path, std::make_unique<std::vector<std::string>>(1, "native_node"));
  ASSERT_TRUE((manager->*GetCallbackBackendMember(CallbackListMethodTag{}))(binaries, disabled).IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(mapped_calls_, 0u);
  EXPECT_EQ(mapped_list_calls_, 0u);
}

TEST_F(QnnUnit_CallbackBufferTest, SharedMappingKeepsCallbacksOrdinaryAndAllowsNativeMapping) {
  auto* manager = backend_->GetMutable();
  backend_->BackendType() = qnn::QnnBackendType::HTP;
  manager->*GetCallbackBackendMember(CallbackMappingFlagTag{}) = true;
  manager->*GetCallbackBackendMember(CallbackFileMapperTag{}) = std::make_unique<qnn::WindowsFileMapper>(logger_);
  manager->*GetCallbackBackendMember(CallbackBackendSetupTag{}) = true;
  const qnn::EpContextIoDispatch native_dispatch;
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 0, true, nullptr, binaries, *dispatch_).IsOK());
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 0, true, nullptr, binaries, native_dispatch).IsOK());
  ASSERT_TRUE(manager->FileMappingIsEnabled());

  ASSERT_TRUE(InitialLoad().IsOK());
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(manager->ReloadContextForSSR("not-existing/callback.bin", 0, context, *dispatch_).IsOK());
  ASSERT_TRUE(live_.empty());
  outputs_.clear();
  ASSERT_TRUE(ListLoad().IsOK());
  EXPECT_EQ(ordinary_calls_, 2u);
  EXPECT_EQ(list_calls_, 1u);
  EXPECT_EQ(read_calls_, 5u);
  EXPECT_EQ(frees_, 5u);
  UseNativeMappingSdk();
  ASSERT_TRUE(NativeInitialLoad(WriteNativeBinary("shared_native"), native_dispatch).IsOK());
  EXPECT_EQ(mapped_calls_, 1u);
  EXPECT_TRUE(manager->FileMappingIsEnabled());
}

TEST_F(QnnUnit_CallbackBufferTest, MappingMismatchPreservesExistingBackend) {
  auto* manager = backend_->GetMutable();
  backend_->BackendType() = qnn::QnnBackendType::HTP;
  manager->*GetCallbackBackendMember(CallbackBackendSetupTag{}) = true;
  ASSERT_TRUE(manager->AddQnnContextHandle(this).IsOK());
  const qnn::EpContextIoDispatch native_dispatch;
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
  auto status = manager->SetupBackend(true, false, true, 0, true, nullptr, binaries, native_dispatch);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorMessage(), "Cannot share QNN backend manager with a different file-mapping mode.");
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 0, false, nullptr, binaries, native_dispatch).IsOK());
  EXPECT_TRUE(manager->IsBackendSetup());
  EXPECT_TRUE(manager->HasContextHandle(this));
  EXPECT_FALSE(manager->FileMappingIsEnabled());
  EXPECT_EQ((manager->*GetCallbackBackendMember(CallbackFileMapperTag{})).get(), nullptr);
}

TEST_F(QnnUnit_CallbackBufferTest, MappingMismatchPreservesVtcmContexts) {
  UseNativeReads();
  auto* manager = backend_->GetMutable();
  backend_->BackendType() = qnn::QnnBackendType::HTP;
  manager->*GetCallbackBackendMember(CallbackMappingFlagTag{}) = true;
  manager->*GetCallbackBackendMember(CallbackFileMapperTag{}) = std::make_unique<qnn::WindowsFileMapper>(logger_);
  manager->*GetCallbackBackendMember(CallbackBackendSetupTag{}) = true;
  manager->*GetCallbackBackendMember(CallbackSharingModeTag{}) = 1;
  Ort::SessionOptions native_options;
  const OrtApiRuntimeCapabilities capabilities{ORT_API_VERSION};
  const qnn::EpContextIoDispatch disabled(*native_options, capabilities);
  const qnn::EpContextIoDispatch enabled(*native_options, capabilities);
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> first, second;
  first.emplace(WriteNativeBinary("first"), std::make_unique<std::vector<std::string>>(1, "first"));
  first.begin()->second->push_back(first.begin()->first);
  second.emplace(WriteNativeBinary("second"), std::make_unique<std::vector<std::string>>(1, "second"));
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 1, true, nullptr, first, enabled).IsOK());
  EXPECT_EQ(mapped_list_calls_, 1u);
  EXPECT_FALSE(manager->SetupBackend(true, false, true, 1, false, nullptr, second, disabled).IsOK());
  EXPECT_EQ(manager->GetQnnContextSize(), 1u);
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 1, true, nullptr, first, enabled).IsOK());
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 1, true, nullptr, second, enabled).IsOK());
  EXPECT_TRUE(manager->FileMappingIsEnabled());
  EXPECT_EQ(list_calls_, 2u);
  EXPECT_EQ(mapped_list_calls_, 2u);
  EXPECT_EQ(manager->GetQnnContextSize(), 2u);
}

TEST_F(QnnUnit_CallbackBufferTest, MappedSsrFailureRereadsNativeFile) {
  UseNativeMappingSdk();
  mapped_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  auto* manager = backend_->GetMutable();
  backend_->BackendType() = qnn::QnnBackendType::HTP;
  manager->*GetCallbackBackendMember(CallbackMappingFlagTag{}) = true;
  manager->*GetCallbackBackendMember(CallbackFileMapperTag{}) = std::make_unique<qnn::WindowsFileMapper>(logger_);
  manager->*GetCallbackBackendMember(CallbackBackendSetupTag{}) = true;
  const qnn::EpContextIoDispatch native_dispatch;
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 0, true, nullptr, binaries, native_dispatch).IsOK());
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(manager->ReloadContextForSSR(WriteNativeBinary("mapped_failure"), 0, context, native_dispatch).IsOK());
  EXPECT_NE(ordinary_buffer_, metadata_buffer_);
  EXPECT_EQ(mapped_calls_, 1u);
  EXPECT_EQ(ordinary_calls_, 1u);
}

TEST_F(QnnUnit_CallbackBufferTest, BinaryVersionFallbackReusesMappedBytes) {
  UseNativeMappingSdk();
  binary_blob_version_ = {3, 3, 2};
  auto* manager = backend_->GetMutable();
  backend_->BackendType() = qnn::QnnBackendType::HTP;
  manager->*GetCallbackBackendMember(CallbackMappingFlagTag{}) = true;
  manager->*GetCallbackBackendMember(CallbackFileMapperTag{}) = std::make_unique<qnn::WindowsFileMapper>(logger_);
  manager->*GetCallbackBackendMember(CallbackBackendSetupTag{}) = true;
  const qnn::EpContextIoDispatch native_dispatch;
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
  ASSERT_TRUE(manager->SetupBackend(true, false, true, 0, true, nullptr, binaries, native_dispatch).IsOK());
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(manager->ReloadContextForSSR(WriteNativeBinary("old_version"), 0, context, native_dispatch).IsOK());
  EXPECT_EQ(ordinary_buffer_, metadata_buffer_);
  EXPECT_EQ(mapped_calls_, 0u);
  EXPECT_EQ(ordinary_calls_, 1u);
}

#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

#endif  // ORT_API_HAS_EPCONTEXT_ENCRYPTION

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
