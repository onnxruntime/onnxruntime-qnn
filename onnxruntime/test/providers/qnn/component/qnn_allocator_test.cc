// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "HTP/QnnHtpMem.h"
#include "core/providers/qnn/builder/qnn_context_mem_handle_manager.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "test/providers/qnn/infra/qnn_module_state_test_utils.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

namespace onnxruntime::test {
namespace {

struct RpcMemCounters {
  std::mutex mutex;
  std::unordered_map<void*, int> live;
  size_t allocations = 0;
  size_t frees = 0;
  size_t registrations = 0;
  size_t deregistrations = 0;
  bool fail_registration = false;
  bool fail_to_fd = false;
  static RpcMemCounters* current;

  static void* Alloc(int, uint32_t, int size) {
    void* p = ::operator new(static_cast<size_t>(size), std::align_val_t{64});
    std::scoped_lock lock{current->mutex};
    current->live.emplace(p, static_cast<int>(++current->allocations));
    return p;
  }

  static void Free(void* p) {
    {
      std::scoped_lock lock{current->mutex};
      EXPECT_EQ(current->live.erase(p), 1u);
      ++current->frees;
    }
    ::operator delete(p, std::align_val_t{64});
  }

  static int ToFd(void* p) {
    std::scoped_lock lock{current->mutex};
    return current->fail_to_fd ? -1 : current->live.at(p);
  }
};

RpcMemCounters* RpcMemCounters::current = nullptr;

class QnnUnit_HtpSharedAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RpcMemCounters::current = &counters_;
    library_ = std::make_shared<qnn::RpcMemLibrary>(qnn::RpcMemApi{
        RpcMemCounters::Alloc, RpcMemCounters::Free, RpcMemCounters::ToFd, nullptr});
  }

  void TearDown() override {
    EXPECT_TRUE(counters_.live.empty());
    RpcMemCounters::current = nullptr;
  }

  std::unique_ptr<qnn::HtpSharedMemoryAllocator> MakeAllocator() {
    return std::make_unique<qnn::HtpSharedMemoryAllocator>(nullptr, library_);
  }

  static OrtApi MakeApi() {
    OrtApi api = *OrtGetApiBase()->GetApi(ORT_API_VERSION);
    api.Logger_GetLoggingSeverityLevel = [](const OrtLogger*, OrtLoggingLevel* level) noexcept -> OrtStatus* {
      *level = ORT_LOGGING_LEVEL_FATAL;
      return nullptr;
    };
    return api;
  }

  OrtApi api_ = MakeApi();
  OrtGlobalApiOverride api_scope_{&api_};
  RpcMemCounters counters_;
  std::shared_ptr<qnn::RpcMemLibrary> library_;
};

TEST_F(QnnUnit_HtpSharedAllocatorTest, ReuseRetainsBackingMemoryAndCleanup) {
  auto allocator = MakeAllocator();
  EXPECT_EQ(allocator->version, ORT_API_VERSION);
  EXPECT_EQ(allocator->GetStats, nullptr);
  EXPECT_EQ(allocator->AllocOnStream, nullptr);
  size_t cleanups = 0;
  void* first = allocator->Alloc(allocator.get(), 1024);
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(qnn::HtpSharedMemoryAllocator::AddAllocationCleanUp(first, [&](void*) {
                EXPECT_EQ(counters_.frees, 0u);
                ++cleanups;
              }).IsOK());
  allocator->Free(allocator.get(), first);
  EXPECT_FALSE(qnn::HtpSharedMemoryAllocator::IsAllocationTracked(first));
  EXPECT_EQ(cleanups, 0u);
  void* second = allocator->Alloc(allocator.get(), 1024);
  EXPECT_EQ(first, second);
  EXPECT_TRUE(qnn::HtpSharedMemoryAllocator::IsAllocationTracked(second));
  EXPECT_EQ(counters_.allocations, 1u);
  allocator->Free(allocator.get(), second);
  ASSERT_EQ(allocator->Shrink(allocator.get()), nullptr);
  EXPECT_EQ(cleanups, 1u);
  EXPECT_EQ(counters_.frees, 1u);
  EXPECT_FALSE(qnn::HtpSharedMemoryAllocator::IsAllocationTracked(first));
}

TEST_F(QnnUnit_HtpSharedAllocatorTest, ReuseDropsExpiredContextHooks) {
  auto allocator = MakeAllocator();
  void* p = allocator->Alloc(allocator.get(), 64);
  auto owner = std::make_shared<int>(0);
  size_t expired_calls = 0;
  ASSERT_TRUE(qnn::HtpSharedMemoryAllocator::AddAllocationCleanUp(
                  p, [&](void*) { ++expired_calls; }, std::weak_ptr<void>{owner})
                  .IsOK());
  allocator->Free(allocator.get(), p);
  owner.reset();
  void* reused = allocator->Alloc(allocator.get(), 64);
  ASSERT_EQ(p, reused);
  allocator->Free(allocator.get(), reused);
  allocator.reset();
  EXPECT_EQ(expired_calls, 0u);
}

TEST_F(QnnUnit_HtpSharedAllocatorTest, ShrinkPreservesLiveBuffersAndOffsetInfo) {
  auto allocator = MakeAllocator();
  auto* live = static_cast<std::byte*>(allocator->Alloc(allocator.get(), 1024));
  void* idle = allocator->Alloc(allocator.get(), 2048);
  allocator->Free(allocator.get(), idle);
  ASSERT_EQ(allocator->Shrink(allocator.get()), nullptr);
  qnn::HtpSharedMemoryAllocator::SharedMemoryInfo info{};
  ASSERT_TRUE(qnn::HtpSharedMemoryAllocator::GetAllocationSharedMemoryInfo(live + 64, info).IsOK());
  EXPECT_EQ(info.offset, 64u);
  EXPECT_EQ(info.total_size, 1024u);
  EXPECT_TRUE(qnn::HtpSharedMemoryAllocator::IsAllocationTracked(live));
  EXPECT_EQ(counters_.frees, 1u);
  allocator->Free(allocator.get(), live);
  allocator.reset();
  EXPECT_EQ(counters_.frees, 2u);
}

TEST_F(QnnUnit_HtpSharedAllocatorTest, ExactSizesAndLiveBuffersAreNotReused) {
  auto allocator = MakeAllocator();
  void* first = allocator->Alloc(allocator.get(), 64);
  void* second = allocator->Alloc(allocator.get(), 64);
  EXPECT_NE(first, second);
  allocator->Free(allocator.get(), first);
  void* larger = allocator->Alloc(allocator.get(), 128);
  EXPECT_NE(first, larger);
  EXPECT_EQ(allocator->Alloc(allocator.get(), 0), nullptr);
  EXPECT_EQ(counters_.allocations, 3u);
  allocator->Free(allocator.get(), second);
  allocator->Free(allocator.get(), larger);
}

TEST_F(QnnUnit_HtpSharedAllocatorTest, CountLimitEvictsOldestIdleBuffer) {
  auto allocator = MakeAllocator();
  std::vector<void*> buffers;
  for (size_t i = 0; i <= qnn::HtpSharedMemoryAllocator::kMaxCachedAllocations; ++i) {
    buffers.push_back(allocator->Alloc(allocator.get(), 64));
  }
  size_t first_cleanup = 0;
  ASSERT_TRUE(qnn::HtpSharedMemoryAllocator::AddAllocationCleanUp(buffers[0], [&](void*) {
                ++first_cleanup;
              }).IsOK());
  for (void* p : buffers) {
    allocator->Free(allocator.get(), p);
  }
  EXPECT_EQ(first_cleanup, 1u);
  EXPECT_EQ(counters_.frees, 1u);
  allocator.reset();
  EXPECT_EQ(counters_.frees, buffers.size());
}

TEST_F(QnnUnit_HtpSharedAllocatorTest, ByteLimitAndOversizedAllocations) {
  auto allocator = MakeAllocator();
  constexpr size_t half_limit = qnn::HtpSharedMemoryAllocator::kMaxCachedBytes / 2;
  void* first = allocator->Alloc(allocator.get(), half_limit);
  void* second = allocator->Alloc(allocator.get(), half_limit);
  void* third = allocator->Alloc(allocator.get(), half_limit);
  allocator->Free(allocator.get(), first);
  allocator->Free(allocator.get(), second);
  allocator->Free(allocator.get(), third);
  EXPECT_EQ(counters_.frees, 1u);
  void* oversized = allocator->Alloc(allocator.get(), qnn::HtpSharedMemoryAllocator::kMaxCachedBytes + 64);
  allocator->Free(allocator.get(), oversized);
  EXPECT_EQ(counters_.frees, 2u);
  void* reused = allocator->Alloc(allocator.get(), half_limit);
  EXPECT_EQ(counters_.allocations, 4u);
  allocator->Free(allocator.get(), reused);
}

TEST_F(QnnUnit_HtpSharedAllocatorTest, FailedFdLookupReleasesUnpublishedAllocation) {
  auto allocator = MakeAllocator();
  counters_.fail_to_fd = true;
  EXPECT_THROW(allocator->Alloc(allocator.get(), 64), Ort::Exception);
  EXPECT_EQ(counters_.allocations, 1u);
  EXPECT_EQ(counters_.frees, 1u);
  counters_.fail_to_fd = false;
  void* p = allocator->Alloc(allocator.get(), 64);
  allocator->Free(allocator.get(), p);
}

TEST_F(QnnUnit_HtpSharedAllocatorTest, ConcurrentAllocateFreeAndShrink) {
  auto allocator = MakeAllocator();
  std::atomic<size_t> completed{0};
  std::vector<std::thread> threads;
  for (size_t t = 0; t < 4; ++t) {
    threads.emplace_back([&, t]() {
      for (size_t i = 0; i < 100; ++i) {
        const size_t size = (t + 1) * 64;
        void* p = allocator->Alloc(allocator.get(), size);
        std::memset(p, static_cast<int>(t), size);
        allocator->Free(allocator.get(), p);
        if (i % 10 == 0) {
          EXPECT_EQ(allocator->Shrink(allocator.get()), nullptr);
        }
        ++completed;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(completed, 400u);
  allocator.reset();
  EXPECT_EQ(counters_.allocations, counters_.frees);
}

TEST_F(QnnUnit_HtpSharedAllocatorTest, CachedRegistrationSurvivesReuseAndCanGrow) {
  auto allocator = MakeAllocator();
  QNN_INTERFACE_VER_TYPE interface = QNN_INTERFACE_VER_TYPE_INIT;
  interface.memRegister = [](Qnn_ContextHandle_t, const Qnn_MemDescriptor_t* desc,
                             uint32_t count, Qnn_MemHandle_t* handle) -> Qnn_ErrorHandle_t {
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(desc->memType, QNN_MEM_TYPE_CUSTOM);
    auto* htp = static_cast<const QnnMemHtp_Descriptor_t*>(desc->customInfo);
    EXPECT_EQ(htp->type, QNN_HTP_MEM_SHARED_BUFFER);
    EXPECT_EQ(htp->size, 64u);
    EXPECT_GE(htp->sharedBufferConfig.fd, 0);
    if (RpcMemCounters::current->fail_registration) {
      return QNN_MEM_ERROR_MAPPING;
    }
    *handle = reinterpret_cast<Qnn_MemHandle_t>(++RpcMemCounters::current->registrations);
    return QNN_SUCCESS;
  };
  interface.errorGetVerboseMessage = [](Qnn_ErrorHandle_t, const char**) -> Qnn_ErrorHandle_t {
    return QNN_MEM_ERROR_MAPPING;
  };
  interface.memDeRegister = [](const Qnn_MemHandle_t*, uint32_t) -> Qnn_ErrorHandle_t {
    ++RpcMemCounters::current->deregistrations;
    return QNN_SUCCESS;
  };
  qnn::QnnContextMemHandleManager manager(interface, nullptr, qnn::QnnBackendType::HTP,
                                          qnn::QnnAllocatorType::HTP_SHARED);
  auto logger = MakeNullLogger();
  uint32_t dim = 4;
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
  tensor.v1.rank = 1;
  tensor.v1.dimensions = &dim;
  void* p = allocator->Alloc(allocator.get(), 64);
  Qnn_MemHandle_t first{};
  bool needs_cleanup = false;
  ASSERT_TRUE(manager.GetOrRegister(p, tensor, first, needs_cleanup, logger).IsOK());
  ASSERT_TRUE(needs_cleanup);
  ASSERT_TRUE(qnn::HtpSharedMemoryAllocator::AddAllocationCleanUp(p, [&](void* address) {
                EXPECT_TRUE(manager.Unregister(address).IsOK());
              }).IsOK());
  allocator->Free(allocator.get(), p);
  void* reused = allocator->Alloc(allocator.get(), 64);
  ASSERT_EQ(reused, p);
  Qnn_MemHandle_t handle{};
  ASSERT_TRUE(manager.GetOrRegister(reused, tensor, handle, needs_cleanup, logger).IsOK());
  EXPECT_EQ(handle, first);
  EXPECT_FALSE(needs_cleanup);
  EXPECT_EQ(counters_.registrations, 1u);
  dim = 16;
  counters_.fail_registration = true;
  EXPECT_FALSE(manager.GetOrRegister(reused, tensor, handle, needs_cleanup, logger).IsOK());
  EXPECT_EQ(counters_.deregistrations, 0u);
  counters_.fail_registration = false;
  ASSERT_TRUE(manager.GetOrRegister(reused, tensor, handle, needs_cleanup, logger).IsOK());
  EXPECT_NE(handle, first);
  EXPECT_FALSE(needs_cleanup);
  EXPECT_EQ(counters_.registrations, 2u);
  EXPECT_EQ(counters_.deregistrations, 1u);
  allocator->Free(allocator.get(), reused);
  ASSERT_EQ(allocator->Shrink(allocator.get()), nullptr);
  EXPECT_EQ(counters_.deregistrations, 2u);
}

}  // namespace
}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
