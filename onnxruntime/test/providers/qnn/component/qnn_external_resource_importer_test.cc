// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && defined(_WIN32)

#include <malloc.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "HTP/QnnHtpMem.h"
#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_context_mem_handle_manager.h"
#include "core/providers/qnn/qnn_external_resource_importer.h"
#include "test/providers/qnn/infra/qnn_module_state_test_utils.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"
#include "test/providers/qnn/infra/stub_backend_manager.h"

namespace onnxruntime::test {
namespace {

const size_t kPage = GetHostAllocationImportAlignment();

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
  bool throw_on_registration = false;           // register_buf throws before mapping
  bool throw_on_fd_lookup_once_mapped = false;  // the next to_fd of a mapped buffer throws
  static FakeFastRpc* current;

  static void RegisterBuf(void* buf, size_t size, int fd, int attr) {
    std::scoped_lock lock{current->mutex};
    if (fd != -1 && current->throw_on_registration) {
      throw std::runtime_error("fake register_buf failure");
    }
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
    if (it != current->registered.end() && current->throw_on_fd_lookup_once_mapped) {
      current->throw_on_fd_lookup_once_mapped = false;
      throw std::runtime_error("fake to_fd failure");
    }
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
    importer_ = std::make_unique<QnnExternalResourceImporterImpl>(0, api_, OrtHardwareDeviceType_NPU, library_);
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
  std::shared_ptr<int> hook_owner_ = std::make_shared<int>(0);  // owner of hooks added directly by tests
};

TEST_F(QnnUnit_HostAllocationImportTest, ImportableHandleTypesFollowDeviceType) {
  // NPU (HTP): host allocations only. The HTP backend rejects QNN_MEM_TYPE_DX12.
  EXPECT_TRUE(importer_->CanImportMemory(importer_.get(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION));
  EXPECT_FALSE(importer_->CanImportMemory(importer_.get(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE));
  EXPECT_FALSE(importer_->CanImportMemory(importer_.get(), ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP));

  // GPU: D3D12 resources only.
  QnnExternalResourceImporterImpl gpu_importer(0, api_, OrtHardwareDeviceType_GPU, library_);
  EXPECT_FALSE(gpu_importer.CanImportMemory(&gpu_importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION));
  EXPECT_TRUE(gpu_importer.CanImportMemory(&gpu_importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE));
  EXPECT_FALSE(gpu_importer.CanImportMemory(&gpu_importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP));

  // CPU: nothing.
  QnnExternalResourceImporterImpl cpu_importer(0, api_, OrtHardwareDeviceType_CPU, library_);
  EXPECT_FALSE(cpu_importer.CanImportMemory(&cpu_importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION));
  EXPECT_FALSE(cpu_importer.CanImportMemory(&cpu_importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE));

  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryDescriptor desc = HostDesc(buffer.get(), kPage);
  OrtExternalMemoryHandle* handle = nullptr;
  OrtStatus* status = gpu_importer.ImportMemory(&gpu_importer, &desc, &handle);
  ASSERT_NE(status, nullptr);
  api_.ReleaseStatus(status);
  EXPECT_EQ(handle, nullptr);
  EXPECT_TRUE(fastrpc_.calls.empty());
}

TEST_F(QnnUnit_HostAllocationImportTest, NpuRejectsD3D12ResourceImport) {
  // Rejected up front rather than failing later in QnnMem_register; the handle is never opened.
  OrtExternalMemoryDescriptor desc{};
  desc.version = ORT_API_VERSION;
  desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  desc.native_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1234));
  desc.size_bytes = kPage;
  OrtExternalMemoryHandle* handle = nullptr;
  OrtStatus* status = importer_->ImportMemory(importer_.get(), &desc, &handle);
  ASSERT_NE(status, nullptr);
  EXPECT_EQ(api_.GetErrorCode(status), ORT_INVALID_ARGUMENT);
  EXPECT_NE(std::string(api_.GetErrorMessage(status)).find("GPU"), std::string::npos);
  api_.ReleaseStatus(status);
  EXPECT_EQ(handle, nullptr);
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
  ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(
                  buffer.get() + 64, [&](void*) {
                    // QNN mem handles must be released while FastRPC still maps the buffer.
                    EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 0u);
                    ++cleanups;
                  },
                  hook_owner_)
                  .IsOK());

  Release(handle);
  EXPECT_EQ(cleanups, 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ false), 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 1u);
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportedHostMemory(base).has_value());
  EXPECT_FALSE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(base, [](void*) {}, hook_owner_).IsOK());
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
  with_offset.offset_bytes = kPage;  // offset must leave at least one byte
  EXPECT_EQ(Import(with_offset, &error), nullptr);
  EXPECT_NE(error.find("offset"), std::string::npos);

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

TEST_F(QnnUnit_HostAllocationImportTest, ImportOffsetShiftsTensorDataButNotFastRpcMapping) {
  ASSERT_EQ(kPage % 4096, 0u);  // the OS page size (4 KiB on Windows) is the import alignment
  auto buffer = MakeAlignedBuffer(2 * kPage);
  OrtExternalMemoryDescriptor desc = HostDesc(buffer.get(), 2 * kPage);
  desc.offset_bytes = 100;  // the offset itself need not be aligned
  OrtExternalMemoryHandle* handle = Import(desc);
  ASSERT_NE(handle, nullptr);

  // FastRPC maps the whole page-aligned range.
  ASSERT_EQ(fastrpc_.calls.size(), 1u);
  EXPECT_EQ(fastrpc_.calls[0].buf, buffer.get());
  EXPECT_EQ(fastrpc_.calls[0].size, 2 * kPage);

  int64_t shape[] = {4};
  OrtExternalTensorDescriptor tensor_desc{};
  tensor_desc.version = ORT_API_VERSION;
  tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  tensor_desc.shape = shape;
  tensor_desc.rank = 1;
  tensor_desc.offset_bytes = 16;

  OrtValue* value = nullptr;
  ASSERT_EQ(importer_->CreateTensorFromMemory(importer_.get(), handle, &tensor_desc, &value), nullptr);
  void* data = nullptr;
  ASSERT_EQ(api_.GetTensorMutableData(value, &data), nullptr);
  EXPECT_EQ(data, buffer.get() + 100 + 16);
  api_.ReleaseValue(value);

  // The data region ends at the end of the import, so the usable size shrinks by the import offset.
  tensor_desc.offset_bytes = 2 * kPage - 100 - 8;
  value = nullptr;
  OrtStatus* status = importer_->CreateTensorFromMemory(importer_.get(), handle, &tensor_desc, &value);
  ASSERT_NE(status, nullptr);
  api_.ReleaseStatus(status);
  EXPECT_EQ(value, nullptr);

  // Addresses before the data offset are still inside the FastRPC mapping.
  EXPECT_TRUE(QnnExternalResourceImporterImpl::FindImportedHostMemory(buffer.get() + 4).has_value());

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

  // A hook whose owner expires after it was added is skipped at release.
  size_t late_expired_calls = 0;
  auto late_owner = std::make_shared<int>(0);
  ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(
                  buffer.get(), [&](void*) { ++late_expired_calls; }, std::weak_ptr<void>{late_owner})
                  .IsOK());
  late_owner.reset();

  Release(handle);
  EXPECT_EQ(expired_calls, 0u);
  EXPECT_EQ(late_expired_calls, 0u);
  EXPECT_EQ(live_calls, 1u);
}

TEST_F(QnnUnit_HostAllocationImportTest, RejectsMemoryAlreadyMappedByFastRpc) {
  // e.g. the app's own rpcmem_alloc buffer. Releasing the import would tear down a mapping the EP does not own.
  auto buffer = MakeAlignedBuffer(kPage);
  fastrpc_.registered[buffer.get()] = 42;
  std::string error;
  EXPECT_EQ(Import(HostDesc(buffer.get(), kPage), &error), nullptr);
  EXPECT_NE(error.find("already mapped"), std::string::npos);
  EXPECT_TRUE(fastrpc_.calls.empty());
  EXPECT_EQ(fastrpc_.registered.at(buffer.get()), 42);
  fastrpc_.registered.erase(buffer.get());
}

TEST_F(QnnUnit_HostAllocationImportTest, ReleaseOutstandingHostImportsUndoesUnreleasedImports) {
  auto buffer = MakeAlignedBuffer(2 * kPage);
  ASSERT_NE(Import(HostDesc(buffer.get(), kPage)), nullptr);
  ASSERT_NE(Import(HostDesc(buffer.get() + kPage, kPage)), nullptr);
  size_t cleanups = 0;
  ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(
                  buffer.get(), [&](void*) {
                    EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 0u);
                    ++cleanups;
                  },
                  hook_owner_)
                  .IsOK());

  // The handles are intentionally not released, as when the EP library is unregistered first.
  EXPECT_EQ(QnnExternalResourceImporterImpl::ReleaseOutstandingHostImports(), 2u);
  EXPECT_EQ(cleanups, 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 2u);
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportedHostMemory(buffer.get()).has_value());
  EXPECT_EQ(QnnExternalResourceImporterImpl::ReleaseOutstandingHostImports(), 0u);
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
  ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(
                  address, [&](void* /*handle*/) { EXPECT_TRUE(manager.Unregister(address).IsOK()); }, hook_owner_)
                  .IsOK());

  // Later runs reuse the registration.
  for (int run = 0; run < 3; ++run) {
    Qnn_MemHandle_t again{};
    ASSERT_TRUE(manager.GetOrRegister(address, tensor, again, did_register, logger).IsOK());
    EXPECT_EQ(again, first);
    EXPECT_FALSE(did_register);
  }
  EXPECT_EQ(state.registrations, 1u);

  // Reuse is by byte capacity: a tensor with a different data type, shape or a smaller size at the same address
  // reuses the first registration (HTP binds the shared buffer by fd and offset).
  {
    uint32_t reshaped_dims[] = {4, 4};  // same 64 bytes as 16 floats
    Qnn_Tensor_t reshaped = tensor;
    reshaped.v1.rank = 2;
    reshaped.v1.dimensions = reshaped_dims;
    Qnn_Tensor_t retyped = tensor;
    retyped.v1.dataType = QNN_DATATYPE_INT_32;  // same 64 bytes
    uint32_t smaller_dim = 8;
    Qnn_Tensor_t smaller = tensor;
    smaller.v1.dimensions = &smaller_dim;
    Qnn_Tensor_t narrower = tensor;
    narrower.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;  // 16 bytes
    for (const Qnn_Tensor_t* variant : {&reshaped, &retyped, &smaller, &narrower}) {
      Qnn_MemHandle_t reused{};
      ASSERT_TRUE(manager.GetOrRegister(address, *variant, reused, did_register, logger).IsOK());
      EXPECT_EQ(reused, first);
      EXPECT_FALSE(did_register);
    }
    EXPECT_EQ(state.registrations, 1u);
  }

  // Re-bind policy: the first bind at an address fixes its registration. A larger tensor at the same address is
  // rejected with guidance rather than replacing a handle other tensors may still use.
  uint32_t larger_dim = dim + 1;
  Qnn_Tensor_t larger = tensor;
  larger.v1.dimensions = &larger_dim;
  Qnn_MemHandle_t rejected{};
  const Ort::Status larger_status = manager.GetOrRegister(address, larger, rejected, did_register, logger);
  ASSERT_FALSE(larger_status.IsOK());
  EXPECT_NE(std::string(larger_status.GetErrorMessage()).find("re-import"), std::string::npos);
  EXPECT_EQ(state.registrations, 1u);

  // A tensor that would run past the end of the import is rejected.
  dim = static_cast<uint32_t>(kPage);  // 4 * kPage bytes
  Qnn_MemHandle_t too_big{};
  EXPECT_FALSE(manager.GetOrRegister(buffer.get() + 2 * kPage - 64, tensor, too_big, did_register, logger).IsOK());

  Release(handle);
  EXPECT_EQ(state.deregistrations, 1u);
  expected = nullptr;
}

TEST_F(QnnUnit_HostAllocationImportTest, ImportRegistrationExceptionIsReportedAndRollsBack) {
  auto buffer = MakeAlignedBuffer(kPage);
  std::string error;

  // ImportMemory is noexcept, so exceptions become error statuses instead of terminating.
  fastrpc_.throw_on_registration = true;
  EXPECT_EQ(Import(HostDesc(buffer.get(), kPage), &error), nullptr);
  EXPECT_NE(error.find("Failed to import"), std::string::npos);
  fastrpc_.throw_on_registration = false;
  EXPECT_TRUE(fastrpc_.registered.empty());
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 1u);
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportedHostMemory(buffer.get()).has_value());

  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);
  Release(handle);
}

TEST_F(QnnUnit_HostAllocationImportTest, ImportExceptionsAreReportedAndUndoTheMapping) {
  auto buffer = MakeAlignedBuffer(kPage);
  std::string error;

  // A failure after FastRPC mapped the buffer, but before the import was published, undoes the mapping.
  fastrpc_.throw_on_fd_lookup_once_mapped = true;
  EXPECT_EQ(Import(HostDesc(buffer.get(), kPage), &error), nullptr);
  EXPECT_NE(error.find("Failed to import"), std::string::npos);
  EXPECT_TRUE(fastrpc_.registered.empty());
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 1u);
  EXPECT_FALSE(QnnExternalResourceImporterImpl::FindImportedHostMemory(buffer.get()).has_value());

  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);
  Release(handle);
}

TEST_F(QnnUnit_HostAllocationImportTest, DeregisteredRecordsAreRetiredWithoutAddressReuse) {
  struct Counters {
    size_t registrations = 0;
    size_t deregistrations = 0;
  };
  static Counters* counters = nullptr;
  Counters state;
  counters = &state;
  QNN_INTERFACE_VER_TYPE interface = QNN_INTERFACE_VER_TYPE_INIT;
  interface.memRegister = [](Qnn_ContextHandle_t, const Qnn_MemDescriptor_t*, uint32_t,
                             Qnn_MemHandle_t* mem_handle) -> Qnn_ErrorHandle_t {
    *mem_handle = reinterpret_cast<Qnn_MemHandle_t>(++counters->registrations);
    return QNN_SUCCESS;
  };
  interface.memDeRegister = [](const Qnn_MemHandle_t*, uint32_t) -> Qnn_ErrorHandle_t {
    ++counters->deregistrations;
    return QNN_SUCCESS;
  };
  qnn::QnnContextMemHandleManager manager(interface, nullptr, qnn::QnnBackendType::HTP,
                                          qnn::QnnAllocatorType::NONE);
  auto logger = MakeNullLogger();
  uint32_t dim = 16;
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
  tensor.v1.rank = 1;
  tensor.v1.dimensions = &dim;

  // One long-lived context binds many short-lived buffers, each at a new address.
  constexpr size_t kCycles = 8;
  std::vector<AlignedBuffer> buffers;
  for (size_t i = 0; i < kCycles; ++i) {
    buffers.push_back(MakeAlignedBuffer(kPage));
    void* address = buffers.back().get();
    OrtExternalMemoryHandle* handle = Import(HostDesc(address, kPage));
    ASSERT_NE(handle, nullptr);
    Qnn_MemHandle_t mem_handle{};
    bool did_register = false;
    ASSERT_TRUE(manager.GetOrRegister(address, tensor, mem_handle, did_register, logger).IsOK());
    ASSERT_TRUE(did_register);
    // Same hook as QnnBackendManager attaches.
    const std::weak_ptr<qnn::QnnMemHandleRegistration> weak_registration = manager.GetRegistration(address);
    ASSERT_TRUE(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(
                    address, [weak_registration](void*) {
                      if (auto registration = weak_registration.lock()) {
                        registration->Deregister();
                      }
                    },
                    std::weak_ptr<void>{weak_registration})
                    .IsOK());

    // The previous buffer's record was retired by this registration, not by a rebind of its address.
    if (i > 0) {
      EXPECT_EQ(manager.GetRegistration(buffers[i - 1].get()), nullptr) << "cycle " << i;
    }
    Release(handle);
    EXPECT_EQ(state.deregistrations, i + 1);
  }
  EXPECT_EQ(state.registrations, kCycles);

  // Only the last, already-deregistered record remains; it is retired by the next registration.
  const auto last = manager.GetRegistration(buffers.back().get());
  ASSERT_NE(last, nullptr);
  EXPECT_FALSE(last->IsRegistered());
  for (size_t i = 0; i + 1 < kCycles; ++i) {
    EXPECT_EQ(manager.GetRegistration(buffers[i].get()), nullptr);
  }
  manager.Clear();
  EXPECT_EQ(state.deregistrations, kCycles);  // nothing deregistered twice
  counters = nullptr;
}

// Goes through QnnBackendManager::GetOrRegisterContextMemHandle, which pairs each QNN registration with a clean up
// hook on the import, using a stub QNN interface and the fake FastRPC.
class QnnUnit_HostAllocationImportBackendTest : public QnnUnit_HostAllocationImportTest {
 protected:
  struct QnnStubState {
    FakeFastRpc* fastrpc = nullptr;
    size_t registrations = 0;
    size_t deregistrations = 0;
    size_t deregistrations_after_fastrpc_unmap = 0;
    int last_registered_fd = -1;
    std::unordered_map<Qnn_MemHandle_t, int> handle_fds;  // QNN mem handle -> FastRPC fd it was registered with
    std::function<void()> during_register;                // runs inside memRegister
    std::atomic<size_t> context_frees{0};
    // While block_deregister is set, memDeRegister waits inside the native call (after counting it).
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool block_deregister = false;
    bool in_deregister = false;
  };
  static inline QnnStubState* stub = nullptr;

  void SetUp() override {
    QnnUnit_HostAllocationImportTest::SetUp();
    stub_state_.fastrpc = &fastrpc_;
    stub = &stub_state_;

    backend_ = std::make_unique<StubBackendManager>(api_ptrs_, logger_);
    ASSERT_NE(backend_->GetMutable(), nullptr);
    backend_->BackendType() = qnn::QnnBackendType::HTP;
    QNN_INTERFACE_VER_TYPE& interface = backend_->QnnInterface();
    interface.memRegister = [](Qnn_ContextHandle_t, const Qnn_MemDescriptor_t* desc, uint32_t,
                               Qnn_MemHandle_t* mem_handle) -> Qnn_ErrorHandle_t {
      stub->last_registered_fd = static_cast<const QnnMemHtp_Descriptor_t*>(desc->customInfo)->sharedBufferConfig.fd;
      if (stub->during_register) {
        stub->during_register();
      }
      *mem_handle = reinterpret_cast<Qnn_MemHandle_t>(++stub->registrations);
      stub->handle_fds[*mem_handle] = stub->last_registered_fd;
      return QNN_SUCCESS;
    };
    interface.memDeRegister = [](const Qnn_MemHandle_t* mem_handles, uint32_t) -> Qnn_ErrorHandle_t {
      ++stub->deregistrations;
      {
        std::unique_lock gate{stub->gate_mutex};
        stub->in_deregister = true;
        stub->gate_cv.notify_all();
        stub->gate_cv.wait(gate, [] { return !stub->block_deregister; });
      }
      // A QNN mem handle must never outlive the FastRPC mapping it refers to.
      const int fd = stub->handle_fds.at(mem_handles[0]);
      bool still_mapped = false;
      {
        std::scoped_lock lock{stub->fastrpc->mutex};
        for (const auto& [buf, mapped_fd] : stub->fastrpc->registered) {
          still_mapped = still_mapped || mapped_fd == fd;
        }
      }
      if (!still_mapped) {
        ++stub->deregistrations_after_fastrpc_unmap;
      }
      return QNN_SUCCESS;
    };
    interface.contextFree = [](Qnn_ContextHandle_t, Qnn_ProfileHandle_t) -> Qnn_ErrorHandle_t {
      ++stub->context_frees;
      return QNN_SUCCESS;
    };
    ASSERT_TRUE(backend_->GetMutable()->AddQnnContextHandle(kContext).IsOK());
  }

  void TearDown() override {
    backend_.reset();
    EXPECT_EQ(stub_state_.deregistrations_after_fastrpc_unmap, 0u);
    stub = nullptr;
    QnnUnit_HostAllocationImportTest::TearDown();
  }

  Ort::Status Bind(void* address, Qnn_MemHandle_t& mem_handle) {
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    tensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
    tensor.v1.rank = 1;
    tensor.v1.dimensions = &dim_;
    return backend_->GetMutable()->GetOrRegisterContextMemHandle(kContext, address, tensor, mem_handle);
  }

  static inline const Qnn_ContextHandle_t kContext = reinterpret_cast<Qnn_ContextHandle_t>(uintptr_t{0x10});
  OrtEpApi ep_api_{};
  OrtModelEditorApi editor_api_{};
  ApiPtrs api_ptrs_{api_, ep_api_, editor_api_};
  Ort::Logger logger_{MakeNullLogger()};
  QnnStubState stub_state_;
  std::unique_ptr<StubBackendManager> backend_;
  uint32_t dim_ = 16;
};

TEST_F(QnnUnit_HostAllocationImportBackendTest, ReleasingImportDeregistersQnnHandleBeforeFastRpc) {
  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);

  Qnn_MemHandle_t first{};
  ASSERT_TRUE(Bind(buffer.get(), first).IsOK());
  for (int run = 0; run < 3; ++run) {
    Qnn_MemHandle_t again{};
    ASSERT_TRUE(Bind(buffer.get(), again).IsOK());
    EXPECT_EQ(again, first);
  }
  EXPECT_EQ(stub_state_.registrations, 1u);
  EXPECT_EQ(stub_state_.last_registered_fd, fastrpc_.registered.at(buffer.get()));

  Release(handle);
  EXPECT_EQ(stub_state_.deregistrations, 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 1u);

  // The context no longer holds the handle, so destroying it does not deregister again.
  backend_->GetMutable()->ReleaseSpecificContextHandle(kContext);
  EXPECT_EQ(stub_state_.deregistrations, 1u);
}

TEST_F(QnnUnit_HostAllocationImportBackendTest, ContextReleasedBeforeImportDeregistersOnce) {
  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);
  Qnn_MemHandle_t mem_handle{};
  ASSERT_TRUE(Bind(buffer.get(), mem_handle).IsOK());

  // e.g. the session is destroyed, or the context is replaced after SSR.
  backend_->GetMutable()->ReleaseSpecificContextHandle(kContext);
  EXPECT_EQ(stub_state_.deregistrations, 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 0u);

  // The import's hook belonged to the destroyed context and is skipped.
  Release(handle);
  EXPECT_EQ(stub_state_.deregistrations, 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 1u);
}

// Regression: the last session's backend manager is being destroyed (its shared_ptr has expired, but it has not
// released its QNN contexts yet) when another thread releases the import. The import's hook must still deregister the
// QNN mem handle before FastRPC unmaps the buffer, rather than give up because the backend can no longer be locked.
TEST_F(QnnUnit_HostAllocationImportBackendTest, ImportReleasedDuringBackendTeardownDeregistersBeforeUnmap) {
  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);
  Qnn_MemHandle_t mem_handle{};
  ASSERT_TRUE(Bind(buffer.get(), mem_handle).IsOK());

  const std::weak_ptr<qnn::QnnBackendManager> weak_backend = backend_->Weak();
  std::unique_lock pause_teardown{backend_->EpContextHandleMapMutex()};
  std::thread teardown([this]() { backend_.reset(); });  // blocks in ReleaseContext() until the lock is released
  while (!weak_backend.expired()) {
    std::this_thread::yield();
  }

  Release(handle);  // concurrent with the paused backend destruction
  EXPECT_EQ(stub_state_.deregistrations, 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 1u);

  pause_teardown.unlock();
  teardown.join();
  // The context found the registration already done and did not deregister it again.
  EXPECT_EQ(stub_state_.deregistrations, 1u);
  EXPECT_EQ(stub_state_.deregistrations_after_fastrpc_unmap, 0u);
}

// The import's hook and the context both try to deregister the same handle at once. Neither the FastRPC unmap nor
// contextFree may happen until the native deregistration in progress has finished, and it happens exactly once.
TEST_F(QnnUnit_HostAllocationImportBackendTest, CompetingDeregistrationsFinishOnceBeforeUnmapAndContextFree) {
  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);
  Qnn_MemHandle_t mem_handle{};
  ASSERT_TRUE(Bind(buffer.get(), mem_handle).IsOK());

  stub_state_.block_deregister = true;
  std::thread import_release;
  std::thread context_release;
  const auto unblock_and_join = [&]() {
    {
      std::scoped_lock gate{stub_state_.gate_mutex};
      stub_state_.block_deregister = false;
    }
    stub_state_.gate_cv.notify_all();
    if (import_release.joinable()) import_release.join();
    if (context_release.joinable()) context_release.join();
  };
  struct JoinOnExit {
    const std::function<void()> join;
    ~JoinOnExit() { join(); }
  } join_on_exit{unblock_and_join};  // also on an early ASSERT return

  // The import's hook starts deregistering and is held inside memDeRegister.
  import_release = std::thread([&]() { Release(handle); });
  {
    std::unique_lock gate{stub_state_.gate_mutex};
    ASSERT_TRUE(stub_state_.gate_cv.wait_for(gate, std::chrono::seconds(10), [&] { return stub_state_.in_deregister; }));
  }
  // The context is destroyed concurrently and must wait for that deregistration.
  context_release = std::thread([&]() { backend_->GetMutable()->ReleaseSpecificContextHandle(kContext); });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_EQ(stub_state_.deregistrations, 1u);
  EXPECT_EQ(stub_state_.context_frees.load(), 0u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 0u);

  unblock_and_join();
  EXPECT_EQ(stub_state_.deregistrations, 1u);
  EXPECT_EQ(stub_state_.context_frees.load(), 1u);
  EXPECT_EQ(fastrpc_.Count(/*deregistration*/ true), 1u);
  EXPECT_EQ(stub_state_.deregistrations_after_fastrpc_unmap, 0u);
}

TEST_F(QnnUnit_HostAllocationImportBackendTest, ImportReleasedDuringRegistrationRollsBackRegistration) {
  auto buffer = MakeAlignedBuffer(kPage);
  OrtExternalMemoryHandle* handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);
  const int first_fd = fastrpc_.registered.at(buffer.get());

  // The import is released after QNN registered the address but before the clean up hook was attached.
  stub_state_.during_register = [&]() { Release(handle); };
  Qnn_MemHandle_t mem_handle{};
  EXPECT_FALSE(Bind(buffer.get(), mem_handle).IsOK());
  stub_state_.during_register = nullptr;
  EXPECT_EQ(stub_state_.registrations, 1u);
  EXPECT_EQ(stub_state_.deregistrations, 1u);  // rolled back instead of left cached without a hook

  // The rollback necessarily deregisters after FastRPC unmapped the old import; that ordering only arises in this race.
  EXPECT_EQ(stub_state_.deregistrations_after_fastrpc_unmap, 1u);
  stub_state_.deregistrations_after_fastrpc_unmap = 0;

  // A later import of the same address registers again with the new mapping's fd instead of reusing a stale entry.
  handle = Import(HostDesc(buffer.get(), kPage));
  ASSERT_NE(handle, nullptr);
  const int second_fd = fastrpc_.registered.at(buffer.get());
  ASSERT_NE(second_fd, first_fd);
  ASSERT_TRUE(Bind(buffer.get(), mem_handle).IsOK());
  EXPECT_EQ(stub_state_.registrations, 2u);
  EXPECT_EQ(stub_state_.last_registered_fd, second_fd);
  Release(handle);
  EXPECT_EQ(stub_state_.deregistrations, 2u);
}

}  // namespace
}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && defined(_WIN32)
