// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include <array>
#include <limits>
#include <memory>
#include <unordered_map>

#include "QnnInterface.h"
#include "gtest/gtest.h"

#if QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 32)
#include "core/providers/qnn/builder/qnn_context_binary_dma_buffer.h"

namespace onnxruntime::test {
namespace {

using qnn::QnnContextBinaryDmaBuffer;

class QnnContextBinaryDmaBufferTest : public ::testing::Test {
 protected:
  static inline std::unordered_map<void*, size_t> registered;
  static inline bool fail_register = false;
  static inline bool fail_deregister = false;
  static inline size_t registrations = 0;

  static void Register(void* address, size_t size, int fd, int) {
    if (fd == -1) {
      if (!fail_deregister) registered.erase(address);
    } else if (!fail_register) {
      registered[address] = size;
      ++registrations;
    }
  }
  static int ToFd(void* address) { return registered.count(address) ? 42 : -1; }

  void SetUp() override {
    registered.clear();
    fail_register = fail_deregister = false;
    registrations = 0;
  }

  static std::shared_ptr<QnnContextBinaryDmaBuffer> Make(std::shared_ptr<char> bytes) {
    return QnnContextBinaryDmaBuffer::Create(std::move(bytes), 128, {nullptr, nullptr, ToFd, Register}, {});
  }
  static std::shared_ptr<char> Bytes() { return {new char[128], std::default_delete<char[]>()}; }
  static Qnn_ContextBinaryDataRequest_t Request(uint64_t offset = 0, uint64_t size = 64) {
    Qnn_ContextBinaryDataRequest_t r{};
    r.offset = offset;
    r.size = size;
    r.isBackendMappingNeeded = true;
    return r;
  }
  static Qnn_ContextBinaryDmaDataMem_t Memory(const Qnn_ContextBinaryDmaDataResponse_t& response) {
    Qnn_ContextBinaryDmaDataMem_t m{};
    m.dmaBuffer = response.dmaBuffer;
    m.memSize = response.alignedSize;
    return m;
  }
};

TEST_F(QnnContextBinaryDmaBufferTest, OwnsOnlyItsContextAndReleasesOutstandingImports) {
  auto bytes1 = Bytes(), bytes2 = Bytes();
  std::weak_ptr<char> weak1 = bytes1, weak2 = bytes2;
  auto first = Make(std::move(bytes1)), second = Make(std::move(bytes2));
  Qnn_ContextBinaryDmaDataResponse_t r1{}, r2{};
  ASSERT_EQ(QNN_SUCCESS, QnnContextBinaryDmaBuffer::Provide(Request(), &r1, first.get()));
  ASSERT_EQ(QNN_SUCCESS, QnnContextBinaryDmaBuffer::Provide(Request(), &r2, second.get()));
  first->ContextCreated();
  second->ContextCreated();
  first->ContextReleased();
  first.reset();
  EXPECT_TRUE(weak1.expired());
  EXPECT_FALSE(weak2.expired());
  EXPECT_EQ(registered.size(), 1u);
  EXPECT_NE(ToFd(r2.dmaBuffer.data), -1);
  second->ContextReleased();
  second.reset();
  EXPECT_TRUE(weak2.expired());
  EXPECT_TRUE(registered.empty());
}

TEST_F(QnnContextBinaryDmaBufferTest, RejectsInvalidRequestsBeforeRegistering) {
  auto state = Make(Bytes());
  Qnn_ContextBinaryDmaDataResponse_t response{};
  for (auto request : {Request(0, 0), Request(129, 1), Request(64, 65),
                       Request(std::numeric_limits<uint64_t>::max(), 2)}) {
    EXPECT_EQ(QNN_CONTEXT_ERROR_INVALID_ARGUMENT, QnnContextBinaryDmaBuffer::Provide(request, &response, state.get()));
  }
  EXPECT_EQ(QNN_CONTEXT_ERROR_INVALID_ARGUMENT, QnnContextBinaryDmaBuffer::Provide(Request(), nullptr, state.get()));
  EXPECT_EQ(QNN_CONTEXT_ERROR_INVALID_ARGUMENT, QnnContextBinaryDmaBuffer::Provide(Request(), &response, nullptr));
  EXPECT_EQ(registrations, 0u);
}

TEST_F(QnnContextBinaryDmaBufferTest, RepeatedImportsAreReleasedAfterTheirLastReference) {
  auto state = Make(Bytes());
  Qnn_ContextBinaryDmaDataResponse_t response{};
  ASSERT_EQ(QNN_SUCCESS, QnnContextBinaryDmaBuffer::Provide(Request(), &response, state.get()));
  ASSERT_EQ(QNN_SUCCESS, QnnContextBinaryDmaBuffer::Provide(Request(), &response, state.get()));
  EXPECT_EQ(registrations, 1u);
  EXPECT_EQ(QNN_SUCCESS, QnnContextBinaryDmaBuffer::Release(Memory(response), state.get()));
  EXPECT_EQ(registered.size(), 1u);
  EXPECT_EQ(QNN_SUCCESS, QnnContextBinaryDmaBuffer::Release(Memory(response), state.get()));
  EXPECT_TRUE(registered.empty());
}

TEST_F(QnnContextBinaryDmaBufferTest, FailedRegistrationDoesNotRetainTheAllocation) {
  auto bytes = Bytes();
  std::weak_ptr<char> weak = bytes;
  auto state = Make(std::move(bytes));
  fail_register = true;
  Qnn_ContextBinaryDmaDataResponse_t response{};
  EXPECT_EQ(QNN_COMMON_ERROR_SYSTEM, QnnContextBinaryDmaBuffer::Provide(Request(), &response, state.get()));
  state.reset();
  EXPECT_TRUE(registered.empty());
  EXPECT_TRUE(weak.expired());
}

TEST_F(QnnContextBinaryDmaBufferTest, FailedDeregistrationKeepsBackingMemoryUntilRetrySucceeds) {
  auto bytes = Bytes();
  std::weak_ptr<char> weak = bytes;
  auto state = Make(std::move(bytes));
  Qnn_ContextBinaryDmaDataResponse_t response{};
  ASSERT_EQ(QNN_SUCCESS, QnnContextBinaryDmaBuffer::Provide(Request(), &response, state.get()));
  fail_deregister = true;
  EXPECT_EQ(QNN_CONTEXT_ERROR_MEM_ALLOC, QnnContextBinaryDmaBuffer::Release(Memory(response), state.get()));
  EXPECT_FALSE(state->TryRelease());
  EXPECT_FALSE(weak.expired());
  fail_deregister = false;
  EXPECT_TRUE(state->TryRelease());
  state.reset();
  EXPECT_TRUE(weak.expired());
}

TEST_F(QnnContextBinaryDmaBufferTest, TerminalContextFreeFailureRetainsNotifyStateAndAllocation) {
  auto bytes = Bytes();
  std::weak_ptr<char> weak = bytes;
  auto state = Make(std::move(bytes));
  auto* retained = state.get();
  state->ContextCreated();
  EXPECT_FALSE(state->TryRelease());
  state.reset();
  EXPECT_FALSE(weak.expired());
  // Simulate the external user finally confirming quiescence. Production
  // intentionally never guesses that a failed contextFree has become safe.
  retained->ContextReleased();
  ASSERT_TRUE(retained->TryRelease());
  delete retained;
  EXPECT_TRUE(weak.expired());
}

}  // namespace
}  // namespace onnxruntime::test
#endif
