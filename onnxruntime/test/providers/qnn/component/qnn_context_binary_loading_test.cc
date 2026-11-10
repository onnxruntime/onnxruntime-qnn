// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "test/providers/qnn/infra/qnn_module_state_test_utils.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
#include <windows.h>

#include <filesystem>
#include <fstream>

#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#endif

namespace onnxruntime::test {

class QnnContextBinaryLoadingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    logging_api_ = Ort::GetApi();
    logging_api_.Logger_GetLoggingSeverityLevel = [](const OrtLogger* logger, OrtLoggingLevel* severity) noexcept -> OrtStatus* {
      // Standalone component tests have no provider logger for global DMA cleanup logging.
      if (logger == nullptr) {
        *severity = ORT_LOGGING_LEVEL_FATAL;
        return nullptr;
      }
      return OrtGetApiBase()->GetApi(ORT_API_VERSION)->Logger_GetLoggingSeverityLevel(logger, severity);
    };
    logging_override_ = std::make_unique<OrtGlobalApiOverride>(&logging_api_);
    qnn::QnnBackendManagerConfig config{};
    config.profiling_level = qnn::ProfilingLevel::OFF;
    config.profiling_level_etw = qnn::ProfilingLevel::OFF;
    config.context_priority = qnn::ContextPriority::NORMAL;
    manager_ = qnn::QnnBackendManager::Create(config, api_ptrs_, logger_);
    current_ = this;
    // Override only this manager's API table; no SDK or process-wide API is modified.
    auto& interface = const_cast<QNN_INTERFACE_VER_TYPE&>(manager_->GetQnnInterface());
    interface.errorGetMessage = [](Qnn_ErrorHandle_t error, const char** message) -> Qnn_ErrorHandle_t {
      if (error == QNN_COMMON_ERROR_NOT_SUPPORTED) {
        *message = "stub DMA rejection";
      } else {
        EXPECT_EQ(error, current_->sdk_result_);
        *message = "stub context loading failure";
      }
      return QNN_SUCCESS;
    };
    interface.contextCreateFromBinary = [](Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                           const QnnContext_Config_t** configs, const void* binary,
                                           Qnn_ContextBinarySize_t bytes, Qnn_ContextHandle_t* context,
                                           Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      ++test.ordinary_calls_;
      EXPECT_EQ(binary, test.expected_binary_);
      EXPECT_EQ(bytes, test.expected_size_);
      if (test.check_configs_) EXPECT_EQ(configs, test.configs_);
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
      if (test.check_configs_ && test.dma_calls_) {
        EXPECT_EQ(configs, test.dma_configs_);
        EXPECT_EQ(test.registered_, nullptr);
      }
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        EXPECT_FALSE(test.system_freed_);
        test.events_.push_back("ordinary");
      }
#endif
      if (test.sdk_result_ == QNN_SUCCESS) {
        *context = const_cast<void*>(binary);
      }
      return test.sdk_result_;
    };
    interface.contextFree = [](Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      return QNN_SUCCESS;
    };
  }
  void TearDown() override {
    manager_.reset();
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
    EXPECT_EQ(registered_, nullptr);
    if (!path_.empty()) EXPECT_TRUE(DeleteFileW(path_.c_str()));
#endif
    current_ = nullptr;
    logging_override_.reset();
  }
  Ort::Status Load(bool request_dma, Qnn_ContextHandle_t& context) {
    qnn::EpContextIoDispatch io_dispatch;
    // A nonexistent filename makes an accidental fallback reread fail.
    return manager_->CreateContextHandleFromBinary(binary_, sizeof(binary_), request_dma,
                                                   configs_, "not-a-context-file.bin", io_dispatch,
                                                   nullptr, context);
  }

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  bool EnableDma() {
    try {
      rpcmem_ = std::make_shared<qnn::RpcMemLibrary>();
    } catch (const std::exception& error) {
      rpcmem_error_ = error.what();
      return false;
    }
    auto& api = const_cast<qnn::RpcMemApi&>(rpcmem_->Api());
    api.register_buf = [](void* pointer, size_t size, int fd, int attributes) {
      auto& test = *current_;
      EXPECT_EQ(attributes, qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | qnn::rpcmem::RPCMEM_ATTR_READ_ONLY);
      if (fd == -1) {
        EXPECT_EQ(pointer, test.registered_);
        EXPECT_EQ(size, test.registered_size_);
        ++test.deregister_calls_;
        test.registered_ = nullptr;
      } else {
        EXPECT_EQ(fd, 0);
        EXPECT_EQ(test.registered_, nullptr);
        ++test.register_calls_;
        test.registered_ = pointer;
        test.registered_size_ = size;
      }
    };
    api.to_fd = [](void* pointer) {
      return pointer == current_->registered_ ? 42 : -1;
    };
    manager_->file_mapped_weights_enabled_ = true;
    if (!manager_->file_mapper_) {
      manager_->file_mapper_ = std::make_unique<qnn::WindowsFileMapper>(logger_);
    }
    auto& interface = const_cast<QNN_INTERFACE_VER_TYPE&>(manager_->GetQnnInterface());
    interface.contextCreateFromBinaryWithCallback = [](
                                                        Qnn_BackendHandle_t, Qnn_DeviceHandle_t, const QnnContext_Config_t** configs,
                                                        const Qnn_ContextBinaryCallback_t* callbacks, const void* binary,
                                                        Qnn_ContextBinarySize_t bytes, Qnn_ContextHandle_t* context,
                                                        Qnn_ProfileHandle_t, Qnn_SignalHandle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      ++test.dma_calls_;
      test.dma_configs_ = configs;
      EXPECT_EQ(binary, test.expected_binary_);
      EXPECT_EQ(bytes, test.expected_size_);
      if (test.check_configs_) EXPECT_EQ(configs, test.configs_);
      EXPECT_EQ(callbacks->type, QNN_CONTEXT_CALLBACK_DMA_BUFFER);
      const auto& dma = callbacks->dmaBufferCallback.v1;
      Qnn_ContextBinaryDataRequest_t request{};
      request.offset = 1;
      request.size = bytes - 1;
      request.isBackendMappingNeeded = 1;
      Qnn_ContextBinaryDmaDataResponse_t response{};
      auto status = dma.dataProvide(request, &response, dma.notifyParam);
      EXPECT_EQ(status, QNN_SUCCESS);
      EXPECT_EQ(response.dmaBuffer.data, static_cast<const char*>(binary) + request.offset);
      EXPECT_EQ(response.dmaBuffer.fd, 42);
      EXPECT_EQ(response.alignedSize, request.size);
      EXPECT_EQ(response.dataStartOffset, 0u);
      if (test.dma_result_ == QNN_SUCCESS) *context = const_cast<void*>(binary);
      return test.dma_result_;
    };
    return true;
  }
  void PublishDma() {
    std::atomic_store(&manager_->rpcmem_library_, rpcmem_);
  }

  class TrackedMapper : public qnn::FileMappingInterface {
   public:
    explicit TrackedMapper(QnnContextBinaryLoadingTest& test) : test_(test), mapper_(test.logger_) {}
    Ort::Status GetContextBinMappedMemoryPtr(const std::string& path, void** data) override {
      auto status = mapper_.GetContextBinMappedMemoryPtr(path, data);
      if (status.IsOK()) test_.expected_binary_ = *data;
      return status;
    }
    Ort::Status MapContextBinary(const std::string& path, std::shared_ptr<char>& data) override {
      std::shared_ptr<char> owner;
      auto status = mapper_.MapContextBinary(path, owner);
      if (!status.IsOK()) return status;
      test_.expected_binary_ = owner.get();
      test_.mapping_alive_ = true;
      if (test_.track_lifetime_) test_.events_.push_back("map");
      auto* pointer = owner.get();
      data = std::shared_ptr<char>(pointer, [this, owner = std::move(owner)](char*) mutable {
        EXPECT_TRUE(test_.system_freed_);
        owner.reset();
        test_.mapping_alive_ = false;
        if (test_.track_lifetime_) test_.events_.push_back("unmap");
      });
      return status;
    }

   private:
    QnnContextBinaryLoadingTest& test_;
    qnn::WindowsFileMapper mapper_;
  };

  void PrepareInitialLoad() {
    wchar_t folder[MAX_PATH], filename[MAX_PATH];
    ASSERT_NE(GetTempPathW(MAX_PATH, folder), 0u);
    ASSERT_NE(GetTempFileNameW(folder, L"qnn", 0, filename), 0u);
    path_ = filename;
    std::ofstream file(path_, std::ios::binary);
    file.write("native_graph", sizeof("native_graph"));
    ASSERT_TRUE(file.good());
    file.close();
    expected_size_ = sizeof("native_graph");
    check_configs_ = false;
    manager_->file_mapped_weights_enabled_ = true;
    manager_->file_mapper_ = std::make_unique<TrackedMapper>(*this);

    auto& system = const_cast<QNN_SYSTEM_INTERFACE_VER_TYPE&>(manager_->GetQnnSystemInterface());
    system.systemContextCreate = [](QnnSystemContext_Handle_t* handle) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      test.system_freed_ = false;
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        test.events_.push_back("system_create");
      }
      *handle = &test;
      return QNN_SUCCESS;
    };
    system.systemContextGetBinaryInfo = [](
                                            QnnSystemContext_Handle_t, void* binary, uint64_t bytes,
                                            const QnnSystemContext_BinaryInfo_t** info, Qnn_ContextBinarySize_t* info_size) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      EXPECT_EQ(binary, test.expected_binary_);
      EXPECT_EQ(bytes, test.expected_size_);
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        test.events_.push_back("metadata");
      }
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
      graph.graphName = static_cast<const char*>(binary);
      graph.numGraphInputs = 1;
      graph.graphInputs = &test.input_;
      graph.numGraphOutputs = 1;
      graph.graphOutputs = &test.output_;
      test.binary_info_.version = QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1;
      auto& context = test.binary_info_.contextBinaryInfoV1;
      context.contextBlobVersion = {3, 3, 3};
      context.numGraphs = 1;
      context.graphs = &test.graph_info_;
      *info = &test.binary_info_;
      *info_size = sizeof(test.binary_info_);
      return QNN_SUCCESS;
    };
    system.systemContextFree = [](QnnSystemContext_Handle_t) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        test.events_.push_back("system_free");
      }
      test.system_freed_ = true;
      return QNN_SUCCESS;
    };
    auto& interface = const_cast<QNN_INTERFACE_VER_TYPE&>(manager_->GetQnnInterface());
    interface.graphRetrieve = [](Qnn_ContextHandle_t context, const char* name,
                                 Qnn_GraphHandle_t* graph) -> Qnn_ErrorHandle_t {
      auto& test = *current_;
      EXPECT_STREQ(name, static_cast<const char*>(test.expected_binary_));
      EXPECT_FALSE(test.system_freed_);
      if (test.track_lifetime_) {
        EXPECT_TRUE(test.mapping_alive_);
        test.events_.push_back("graph");
      }
      *graph = context;
      return QNN_SUCCESS;
    };
  }

  Ort::Status LoadNative(std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& models) {
    return manager_->LoadCachedQnnContextFromBuffer(nullptr, 0, path_.string(), "native", models,
                                                    0, qnn::EpContextIoDispatch{});
  }

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION
  static OrtStatus* ORT_API_CALL ReadCallback(void* opaque, const char*, OrtAllocator* allocator,
                                              void** data, size_t* size) {
    auto& test = *static_cast<QnnContextBinaryLoadingTest*>(opaque);
    ++test.read_calls_;
    *size = sizeof("callback_graph");
    auto status = Ort::GetApi().AllocatorAlloc(allocator, *size, data);
    if (status) return status;
    std::memcpy(*data, "callback_graph", *size);
    return nullptr;
  }
  void CheckSharingOrder(bool callback_first) {
    PrepareInitialLoad();
    ASSERT_FALSE(path_.empty());
    if (!EnableDma()) GTEST_SKIP() << rpcmem_error_;
    dma_result_ = QNN_SUCCESS;
    // Exercise SetupBackend's reused-manager branch after callback-only initialization.
    manager_->backend_setup_completed_ = true;
    auto setter = Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_Fn(&Ort::GetApi());
    ASSERT_NE(setter, nullptr);
    Ort::SessionOptions options;
    Ort::ThrowOnError(setter(options, ReadCallback, this));
    qnn::EpContextIoDispatch dispatch(*options, OrtApiRuntimeCapabilities{ORT_API_VERSION});
    ASSERT_TRUE(dispatch.HasReadCallback());
    std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> models;
    auto callback_load = [&] {
      std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
      ASSERT_TRUE(manager_->SetupBackend(true, true, true, 0, true, nullptr, binaries, dispatch,
                                         false, false, false, 0)
                      .IsOK());
      qnn::EpContextBuffer output;
      ASSERT_TRUE(manager_->ReadContextBinIfValid("not-a-context-file.bin", output, dispatch).IsOK());
      expected_binary_ = output.data();
      expected_size_ = output.size();
      ASSERT_TRUE(manager_->LoadCachedQnnContextFromBuffer(output.data(), output.size(), "not-a-context-file.bin",
                                                           "callback", models, 0, dispatch)
                      .IsOK());
    };
    auto native_load = [&] {
      std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> binaries;
      ASSERT_TRUE(manager_->SetupBackend(true, true, true, 0, true, rpcmem_, binaries, qnn::EpContextIoDispatch{},
                                         false, false, false, 0)
                      .IsOK());
      EXPECT_EQ(std::atomic_load(&manager_->rpcmem_library_), rpcmem_);
      expected_size_ = sizeof("native_graph");
      ASSERT_TRUE(LoadNative(models).IsOK());
    };
    if (callback_first) {
      callback_load();
      EXPECT_EQ(std::atomic_load(&manager_->rpcmem_library_), nullptr);
      native_load();
    } else {
      native_load();
      callback_load();
      EXPECT_EQ(std::atomic_load(&manager_->rpcmem_library_), rpcmem_);
    }
    ASSERT_EQ(models.size(), 2u);
    EXPECT_EQ(models.at("callback")->Name(), "callback_graph");
    EXPECT_EQ(models.at("native")->Name(), "native_graph");
    EXPECT_EQ(read_calls_, 1u);
    EXPECT_EQ(ordinary_calls_, 1u);
    EXPECT_EQ(dma_calls_, 1u);
    EXPECT_TRUE(manager_->file_mapped_weights_enabled_);
  }
#endif

  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_;
  std::string rpcmem_error_;
  std::filesystem::path path_;
  Qnn_ErrorHandle_t dma_result_ = QNN_COMMON_ERROR_NOT_SUPPORTED;
  Qnn_ErrorHandle_t metadata_result_ = QNN_SUCCESS;
  size_t dma_calls_ = 0, register_calls_ = 0, deregister_calls_ = 0, read_calls_ = 0;
  void* registered_ = nullptr;
  size_t registered_size_ = 0;
  const QnnContext_Config_t** dma_configs_ = nullptr;
  bool mapping_alive_ = false, system_freed_ = false, track_lifetime_ = false;
  std::vector<std::string> events_;
  uint32_t dimension_ = 1;
  Qnn_Tensor_t input_ = QNN_TENSOR_INIT;
  Qnn_Tensor_t output_ = QNN_TENSOR_INIT;
  QnnSystemContext_GraphInfo_t graph_info_ = QNN_SYSTEM_CONTEXT_GRAPH_INFO_INIT;
  QnnSystemContext_BinaryInfo_t binary_info_{};
#endif

  static thread_local QnnContextBinaryLoadingTest* current_;
  OrtApi logging_api_{};
  std::unique_ptr<OrtGlobalApiOverride> logging_override_;
  OrtApi ort_api_{};
  OrtEpApi ep_api_{};
  OrtModelEditorApi editor_api_{};
  ApiPtrs api_ptrs_{ort_api_, ep_api_, editor_api_};
  Ort::Logger logger_{MakeNullLogger()};
  std::shared_ptr<qnn::QnnBackendManager> manager_;
  char binary_[4] = {1, 2, 3, 4};
  const void* expected_binary_ = binary_;
  uint64_t expected_size_ = sizeof(binary_);
  bool check_configs_ = true;
  const QnnContext_Config_t* configs_[1] = {nullptr};
  size_t ordinary_calls_ = 0;
  Qnn_ErrorHandle_t sdk_result_ = QNN_SUCCESS;
};

thread_local QnnContextBinaryLoadingTest* QnnContextBinaryLoadingTest::current_ = nullptr;

TEST_F(QnnContextBinaryLoadingTest, OrdinaryLoadingUsesSuppliedBufferWithoutReadingFile) {
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(false, context).IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
}

TEST_F(QnnContextBinaryLoadingTest, MissingDmaResourcesFallBackToSameBuffer) {
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
}

TEST_F(QnnContextBinaryLoadingTest, OrdinarySdkFailureAfterDmaFallbackIsPropagated) {
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  Qnn_ContextHandle_t context = nullptr;
  auto status = Load(true, context);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, nullptr);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("contextCreateFromBinary failed"), std::string::npos);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("stub context loading failure"), std::string::npos);
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
TEST_F(QnnContextBinaryLoadingTest, SdkDmaRejectionRetriesSameBufferAfterDeregistering) {
  if (!EnableDma()) GTEST_SKIP() << rpcmem_error_;
  PublishDma();
  Qnn_ContextHandle_t context = nullptr;
  ASSERT_TRUE(Load(true, context).IsOK());
  EXPECT_EQ(dma_calls_, 1u);
  EXPECT_EQ(register_calls_, 1u);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, binary_);
}

TEST_F(QnnContextBinaryLoadingTest, OrdinaryFailureAfterSdkDmaRejectionIsPropagated) {
  if (!EnableDma()) GTEST_SKIP() << rpcmem_error_;
  PublishDma();
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  Qnn_ContextHandle_t context = nullptr;
  auto status = Load(true, context);
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(dma_calls_, 1u);
  EXPECT_EQ(register_calls_, 1u);
  EXPECT_EQ(deregister_calls_, 1u);
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(context, nullptr);
  EXPECT_NE(std::string(status.GetErrorMessage()).find("stub context loading failure"), std::string::npos);
}

#if ORT_API_HAS_EPCONTEXT_ENCRYPTION
TEST_F(QnnContextBinaryLoadingTest, CallbackThenNativePublishesDmaOnSharedManager) {
  CheckSharingOrder(true);
}
TEST_F(QnnContextBinaryLoadingTest, NativeThenCallbackKeepsDmaOnSharedManager) {
  CheckSharingOrder(false);
}
#endif

TEST_F(QnnContextBinaryLoadingTest, InitialCpuMappingOutlivesSdkMetadataAndSystemContext) {
  PrepareInitialLoad();
  track_lifetime_ = true;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> models;
  ASSERT_TRUE(LoadNative(models).IsOK());
  EXPECT_FALSE(mapping_alive_);
  EXPECT_EQ(models.at("native")->Name(), "native_graph");
  EXPECT_EQ(events_, (std::vector<std::string>{"map", "system_create", "metadata", "ordinary", "graph", "system_free", "unmap"}));
}

TEST_F(QnnContextBinaryLoadingTest, InitialCpuMappingIsReleasedAfterSdkLoadingFailure) {
  PrepareInitialLoad();
  track_lifetime_ = true;
  sdk_result_ = QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> models;
  EXPECT_FALSE(LoadNative(models).IsOK());
  EXPECT_TRUE(models.empty());
  EXPECT_FALSE(mapping_alive_);
  EXPECT_EQ(ordinary_calls_, 1u);
  EXPECT_EQ(events_, (std::vector<std::string>{"map", "system_create", "metadata", "ordinary", "system_free", "unmap"}));
}

TEST_F(QnnContextBinaryLoadingTest, InitialCpuMappingIsReleasedAfterMetadataFailure) {
  PrepareInitialLoad();
  track_lifetime_ = true;
  metadata_result_ = QNN_COMMON_ERROR_GENERAL;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> models;
  EXPECT_FALSE(LoadNative(models).IsOK());
  EXPECT_TRUE(models.empty());
  EXPECT_FALSE(mapping_alive_);
  EXPECT_EQ(ordinary_calls_, 0u);
  EXPECT_EQ(events_, (std::vector<std::string>{"map", "system_create", "metadata", "system_free", "unmap"}));
}
#endif

}  // namespace onnxruntime::test

#endif
