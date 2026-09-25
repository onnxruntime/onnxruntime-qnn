// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>
#include <string>

#include "core/providers/qnn/qnn_allocator.h"

namespace onnxruntime {
namespace test {

TEST(QnnUnit_AllocatorTest, Alloc_DeferredRpcMemProvider_LoadsOnFirstAllocation) {
  size_t provider_call_count = 0;
  qnn::HtpSharedMemoryAllocator allocator(
      reinterpret_cast<const OrtMemoryInfo*>(0x1),
      [&provider_call_count](std::string& error_message) -> std::shared_ptr<qnn::RpcMemLibrary> {
        ++provider_call_count;
        error_message = "RPCMEM intentionally unavailable";
        return nullptr;
      });

  EXPECT_EQ(provider_call_count, 0u)
      << "Constructing the allocator during EP registration must not load RPCMEM.";

  EXPECT_THROW(qnn::HtpSharedMemoryAllocator::AllocImpl(&allocator, 64), Ort::Exception);
  EXPECT_EQ(provider_call_count, 1u)
      << "The RPCMEM provider should first be invoked by an actual allocation.";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
