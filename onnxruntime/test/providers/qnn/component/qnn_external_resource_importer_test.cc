// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && defined(_WIN32)

#include <malloc.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "HTP/QnnHtpMem.h"
#include "core/providers/qnn/builder/qnn_context_mem_handle_manager.h"
#include "core/providers/qnn/qnn_external_resource_importer.h"
#include "test/providers/qnn/infra/qnn_module_state_test_utils.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

namespace onnxruntime::test {
namespace {

constexpr size_t kPage = kHostAllocationImportAlignment;

// Fake FastRPC registration API: register_buf(fd != -1) maps a buffer, fd == -1 unmaps it.
struct FakeFastRpc {
  struct Call {
    void* buf;
    size_t size;
    int fd;
    int attr;
  };

  std::mutex mutex;
  std::unordered_map<void*, int> registered;  // address -> fd
  std::vector<Call> calls;
  int next_fd = 100;
  bool fail_registration = false;
  static FakeFastRpc* current;

  static void RegisterBuf(void* buf, size_t size, int fd, int attr) {
    std::scoped_lock lock{current->mutex};
    current->calls.push_back({buf, size, fd, attr});
    if (fd == -1) {
      current->registered.erase(buf);
    } else if (!current->fail_registration) {
      current->registered[buf] = current->next_fd++;
    }
  }

  static int ToFd(void* buf) {
    std::scoped_lock lock{current->mutex};
    const auto it = current->registered.find(buf);
    return it == current->registered.end() ? -1 : it->second;
  }

  size_t Count(bool deregistration) {
    std::scoped_lock lock{mutex};
    size_t n = 0;
    for (const auto& call : calls) {
      n += (call.fd == -1) == deregistration ? 1 : 0;
    }
    return n;
  }
};

FakeFastRpc* FakeFastRpc::current = nullptr;

struct AlignedBufferDeleter {
  void operator()(void* p) const { _aligned_free(p); }
};
using AlignedBuffer = std::unique_ptr<std::byte, AlignedBufferDeleter>;

AlignedBuffer MakeAlignedBuffer(size_t size) {
  return AlignedBuffer{static_cast<std::byte*>(_aligned_malloc(size, kPage))};
}

class QnnUnit_HostAllocationImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FakeFastRpc::current = &fastrpc_;
    library_ = std::make_shared<qnn::RpcMemLibrary>(
        qnn::RpcMemApi{nullptr, nullptr, FakeFastRpc::ToFd, FakeFastRpc::RegisterBuf});
    importer_ = std::make_unique<QnnExternalResourceImporterImpl>(0, api_, /*supports_host_allocation*/ true,
                                                                  library_);
  }

  void TearDown() override {
    importer_.reset();
    EXPECT_TRUE(fastrpc_.registered.empty());
    FakeFastRpc::current = nullptr;
  }

  static OrtApi MakeApi() {
    OrtApi api = *OrtGetApiBase()->GetApi(ORT_API_VERSION);
    api.Logger_GetLoggingSeverityLevel = [](const OrtLogger*, OrtLoggingLevel* level) noexcept -> OrtStatus* {
      *level = ORT_LOGGING_LEVEL_FATAL;
      return nullptr;
    };
    return api;
  }

  static OrtExternalMemoryDescriptor HostDesc(void* p, size_t size) {
    OrtExternalMemoryDescriptor desc{};
    desc.version = ORT_API_VERSION;
    desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION;
    desc.native_handle = p;
    desc.size_bytes = size;
    desc.offset_bytes = 0;
    return desc;
  }

  // Returns the imported handle, or nullptr and the error message.
  OrtExternalMemoryHandle* Import(const OrtExternalMemoryDescriptor& desc, std::string* error = nullptr) {
    OrtExternalMemoryHandle* handle = nullptr;
    OrtStatus* status = importer_->ImportMemory(importer_.get(), &desc, &handle);
    if (status != nullptr) {
      if (error != nullptr) {
        *error = api_.GetErrorMessage(status);
      }
      api_.ReleaseStatus(status);
      EXPECT_EQ(handle, nullptr);
      return nullptr;
    }
    return handle;
  }

  void Release(OrtExternalMemoryHandle* handle) { importer_->ReleaseMemory(importer_.get(), handle); }

  OrtApi api_ = MakeApi();
  OrtGlobalApiOverride api_scope_{&api_};
  FakeFastRpc fastrpc_;
  std::shared_ptr<qnn::RpcMemLibrary> library_;
  std::unique_ptr<QnnExternalResourceImporterImpl> importer_;
};

TEST_F(QnnUnit_HostAllocationImportTest, CanImportHostAllocationOnlyOnNpu) {
  EXPECT_TRUE(importer_->CanImportMemory(importer_.get(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION));
  EXPECT_TRUE(importer_->CanImportMemory(importer_.get(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE));
  EXPECT_FALSE(importer_->CanImportMemory(importer_.get(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP));

  QnnExternalResourceImporterImpl gpu_importer(0, api_, /*supports_host_allocation*/ false, library_);
  EXPECT_FALSE(gpu_importer.CanImportMemory(&gpu_importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION));

  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryDescriptor desc = HostDesc(buffer.get(), kPage);
  OrtExternalMemoryHandle* handle = nullptr;
  OrtStatus* status = gpu_importer.ImportMemory(&gpu_importer, &desc, &handle);
  ASSERT_NE(status, nullptr);
  api_.ReleaseStatus(status);
  EXPECT_EQ(handle, nullptr);
  EXPECT_TRUE(fastrpc_.calls.empty());
}

TEST_F(QnnUnit_HostAllocationImportTest, ImportRegistersOnceAndReleaseDeregistersAfterCleanUp) {
  auto buffer = MakeAlignedBuffer(2 * kPage);
  void* base = buffer.get();
  OrtExternalMemoryHandle* handle = Import(HostDesc(base, 2 * kPage));
  ASSERT_NE(handle, nullptr);

  ASSERT_EQ(fastrpc_.calls.size(), 1u);
  EXPECT_EQ(fastrpc_.calls[0].buf, base);
  EXPECT_EQ(fastrpc_.calls[0].size, 2 * kPage);
  EXPECT_EQ(fastrpc_.calls[0].attr, qnn::rpcmem::RPCMEM_ATTR_IMPORT_BUFFER);  // not read-only

  // Interior addresses resolve to the import; the end of the range does not.
  const auto info = QnnExternalResourceImporterImpl::FindImportedHostMemory(buffer.get() + kPage + 4);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->base_address, base);
  EXPECT_EQ(info->size, 2 * kPage);
  EXPECT_EQ(info->fd, fastrpc_.registered.at(base));
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportedHostMemory(buffer.get() + 2 * kPage).has_value());
  // Host imports are not D3D12 handles.
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportMemory(base));
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportMemory(handle));

  size_t cleanups = 0;
  ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(buffer.get() + 64, [&](void*) {
                // QNN mem handles must be released while FastRPC still maps the buffer.
                EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 0u);
                ++cleanups;
              }).IsOK());

  Release(handle);
  EXPECT_EQ(cleanups, 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ false), 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 1u);
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportedHostMemory(base).has_value());
  EXPECT_FALSE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(base, [](void*) {}).IsOK());
}

TEST_F(QnnUnit_HostAllocationImportTest, RejectsInvalidDescriptors) {
  auto buffer = MakeAlignedBuffer(2 * kPage);
  std::string error;

  EXPECT_EQ(Import(HostDesc(nullptr, kPage), &error), nullptr);
  EXPECT_EQ(Import(HostDesc(buffer.get(), 0), &error), nullptr);
  EXPECT_EQ(Import(HostDesc(buffer.get() + 64, kPage), &error), nullptr);
  EXPECT_NE(error.find("aligned"), std::string::npos);
  EXPECT_EQ(Import(HostDesc(buffer.get(), kPage + 64), &error), nullptr);
  EXPECT_NE(error.find("aligned"), std::string::npos);
  OrtExternalMemoryDescriptor with_offset = HostDesc(buffer.get(), kPage);
  with_offset.offset_bytes = kPage;
  EXPECT_EQ(Import(with_offset, &error), nullptr);

  EXPECT_TRUE(fastrpc_.calls.empty());
}

TEST_F(QnnUnit_HostAllocationImportTest, FailedFastRpcRegistrationFailsImport) {
  auto buffer = MakeAlignedBuffer(kPage);
  fastrpc_.fail_registration = true;
  std::string error;
  EXPECT_EQ(Import(HostDesc(buffer.get(), kPage), &error), nullptr);
  EXPECT_NE(error.find("FastRPC"), std::string::npos);
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportedHostMemory(buffer.get()).has_value());

  // A later import of the same range can still succeed.
  fastrpc_.fail_registration = false;
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);
  Release(handle);
}

TEST_F(QnnUnit_HostAllocationImportTest, RejectsOverlappingImports) {
  auto buffer = MakeAlignedBuffer(4 * kPage);
  OrtExternalMemoryHandle* middle = Import(HostDesc(buffer.get() + kPage, 2 * kPage));
  ASSERT_NE(middle, nullptr);

  std::string error;
  EXPECT_EQ(Import(HostDesc(buffer.get(), 2 * kPage), &error), nullptr);  // overlaps start
  EXPECT_NE(error.find("overlaps"), std::string::npos);
  EXPECT_EQ(Import(HostDesc(buffer.get() + 2 * kPage, 2 * kPage), &error), nullptr);  // overlaps end
  EXPECT_EQ(Import(HostDesc(buffer.get() + kPage, 2 * kPage), &error), nullptr);      // identical

  // Adjacent ranges do not overlap.
  OrtExternalMemoryHandle* before = Import(HostDesc(buffer.get(), kPage));
  OrtExternalMemoryHandle* after = Import(HostDesc(buffer.get() + 3 * kPage, kPage));
  ASSERT_NE(before, nullptr);
  ASSERT_NE(after, nullptr);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ false), 3u);

  Release(before);
  Release(middle);
  Release(after);
}

TEST_F(QnnUnit_HostAllocationImportTest, CreateTensorFromMemoryUsesHostAddressAndHostAccessibleInfo) {
  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);

  int64_t shape[] = {2, 8};
  OrtExternalTensorDescriptor tensor_desc{};
  tensor_desc.version = ORT_API_VERSION;
  tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  tensor_desc.shape = shape;
  tensor_desc.rank = 2;
  tensor_desc.offset_bytes = 256;

  OrtValue* value = nullptr;
  ASSERT_EQ(importer_->CreateTensorFromMemory(importer_.get(), handle, &tensor_desc, &value), nullptr);
  ASSERT_NE(value, nullptr);

  void* data = nullptr;
  ASSERT_EQ(api_.GetTensorMutableData(value, &data), nullptr);
  EXPECT_EQ(data, buffer.get() + 256);

  const OrtMemoryInfo* info = nullptr;
  ASSERT_EQ(api_.GetTensorMemoryInfo(value, &info), nullptr);
  const char* name = nullptr;
  ASSERT_EQ(api_.MemoryInfoGetName(info, &name), nullptr);
  EXPECT_STREQ(name, "QnnHtpShared");
  OrtMemoryInfoDeviceType device_type{};
  api_.MemoryInfoGetDeviceType(info, &device_type);
  EXPECT_EQ(device_type, OrtMemoryInfoDeviceType_CPU);
  EXPECT_EQ(api_.MemoryInfoGetDeviceMemType(info), OrtDeviceMemoryType_HOST_ACCESSIBLE);
  api_.ReleaseValue(value);

  // A tensor that does not fit in the import is rejected.
  tensor_desc.offset_bytes = kPage - 16;
  value = nullptr;
  OrtStatus* status = importer_->CreateTensorFromMemory(importer_.get(), handle, &tensor_desc, &value);
  ASSERT_NE(status, nullptr);
  api_.ReleaseStatus(status);
  EXPECT_EQ(value, nullptr);

  Release(handle);
}

TEST_F(QnnUnit_HostAllocationImportTest, ExpiredOwnerHooksAreDropped) {
  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);

  size_t expired_calls = 0;
  size_t live_calls = 0;
  auto expired_owner = std::make_shared<int>(0);
  auto live_owner = std::make_shared<int>(0);
  ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(
                  buffer.get(), [&](void*) { ++expired_calls; }, std::weak_ptr<void>{expired_owner})
                  .IsOK());
  expired_owner.reset();  // e.g. the session's QNN context was destroyed
  ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(
                  buffer.get(), [&](void*) { ++live_calls; }, std::weak_ptr<void>{live_owner})
                  .IsOK());

  Release(handle);
  EXPECT_EQ(expired_calls, 0u);
  EXPECT_EQ(live_calls, 1u);
}

TEST_F(QnnUnit_HostAllocationImportTest, ContextRegistrationUsesImportFdAndOffsetAndIsRetained) {
  auto buffer = MakeAlignedBuffer(2 * kPage);
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), 2 * kPage));
  ASSERT_NE(handle, nullptr);

  struct Expected {
    int fd;
    uint64_t offset;
    size_t registrations = 0;
    size_t deregistrations = 0;
  };
  static Expected* expected = nullptr;
  Expected state{fastrpc_.registered.at(buffer.get()), kPage};
  expected = &state;

  QNN_INTERFACE_VER_TYPE interface = QNN_INTERFACE_VER_TYPE_INIT;
  interface.memRegister = [](Qnn_ContextHandle_t, const Qnn_MemDescriptor_t* desc,
                             uint32_t count, Qnn_MemHandle_t* mem_handle) -> Qnn_ErrorHandle_t {
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(desc->memType, QNN_MEM_TYPE_CUSTOM);
    auto* htp = static_cast<const QnnMemHtp_Descriptor_t*>(desc->customInfo);
    EXPECT_EQ(htp->type, QNN_HTP_MEM_SHARED_BUFFER);
    EXPECT_EQ(htp->size, 2 * kPage);
    EXPECT_EQ(htp->sharedBufferConfig.fd, expected->fd);
    EXPECT_EQ(htp->sharedBufferConfig.offset, expected->offset);
    *mem_handle = reinterpret_cast<Qnn_MemHandle_t>(++expected->registrations);
    return QNN_SUCCESS;
  };
  interface.memDeRegister = [](const Qnn_MemHandle_t*, uint32_t) -> Qnn_ErrorHandle_t {
    ++expected->deregistrations;
    return QNN_SUCCESS;
  };

  // The shared-memory allocator option is not required for imported host memory.
  qnn::QnnContextMemHandleManager manager(interface, nullptr, qnn::QnnBackendType::HTP,
                                          qnn::QnnAllocatorType::NONE);
  auto logger = MakeNullLogger();
  uint32_t dim = 16;
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
  tensor.v1.rank = 1;
  tensor.v1.dimensions = &dim;

  void* address = buffer.get() + kPage;
  Qnn_MemHandle_t first{};
  bool did_register = false;
  ASSERT_TRUE(manager.GetOrRegister(address, tensor, first, did_register, logger).IsOK());
  EXPECT_TRUE(did_register);
  ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(address, [&](void* /*handle*/) {
                EXPECT_TRUE(manager.Unregister(address).IsOK());
              }).IsOK());

  // Later runs reuse the registration.
  for (int run = 0; run < 3; ++run) {
    Qnn_MemHandle_t again{};
    ASSERT_TRUE(manager.GetOrRegister(address, tensor, again, did_register, logger).IsOK());
    EXPECT_EQ(again, first);
    EXPECT_FALSE(did_register);
  }
  EXPECT_EQ(state.registrations, 1u);

  // A tensor that would run past the end of the import is rejected.
  dim = static_cast<uint32_t>(kPage);  // 4 * kPage bytes
  Qnn_MemHandle_t too_big{};
  EXPECT_FALSE(manager.GetOrRegister(buffer.get() + 2 * kPage - 64, tensor, too_big, did_register, logger).IsOK());

  Release(handle);
  EXPECT_EQ(state.deregistrations, 1u);
  expected = nullptr;
}

}  // namespace
}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && defined(_WIN32)
