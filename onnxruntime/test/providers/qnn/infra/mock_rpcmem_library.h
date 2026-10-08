// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// MockRpcMemLibrary — an IRpcMemLibrary whose RpcMemApi callbacks are implemented
// in-process, so tests can observe FastRPC buffer registration / deregistration
// without a FastRPC driver (libcdsprpc) present.
//
// The mock models the one piece of rpcmem behaviour the EP depends on: production
// code does not get a return value from register_buf(), so it infers success by
// calling to_fd() afterwards (see QnnBackendManager::MapDmaData, ReleaseDmaData and
// DeallocateMappedDmaBuffers). The mock therefore maintains a pointer -> fd table:
//
//   register_buf(p, size, fd != -1, attr)  registers p, after which to_fd(p) != -1
//   register_buf(p, size, -1, attr)        deregisters p, after which to_fd(p) == -1
//
// A request made to fail via fault injection leaves the table untouched, which is
// exactly what the production error paths look for (a registration that still
// reports -1, or a deregistration that still reports a valid fd).
//
// This header is self-contained and pulls in no EP symbols at link time, so it can
// be included from any test translation unit.
//
// Threading: RpcMemApi is a set of plain C function pointers with no context
// parameter, so the static thunks below route to a single live instance. Creating a
// second MockRpcMemLibrary while one is alive throws. Instance state is guarded by a
// mutex, so the callbacks may be invoked from multiple threads.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime {
namespace test {

// One recorded register_buf() call, in the order the EP made it.
struct RpcMemRegisterBufCall {
  void* buff;
  size_t size;
  int fd;  // as passed by the caller; -1 requests deregistration
  int attr;
  bool succeeded;  // false if fault injection made the mock reject the request

  // The EP signals deregistration by passing fd == -1. Registration passes NULL.
  bool IsDeregister() const { return fd == -1; }
};

// One recorded alloc() call.
struct RpcMemAllocCall {
  int heapid;
  uint32_t flags;
  int size;
  void* result;  // nullptr if the mock failed the allocation
};

class MockRpcMemLibrary : public qnn::IRpcMemLibrary {
 public:
  // Invoked for every register_buf() request before the mock acts on it. Return
  // false to fail the request. Combines with FailRegisterFor() / FailDeregisterFor():
  // a request fails if either the pointer is in the matching fail set or the hook
  // returns false. The hook runs with the mock's lock held, so it must not call back
  // into the mock.
  using RegisterBufHook = std::function<bool(const RpcMemRegisterBufCall& call)>;

  MockRpcMemLibrary() {
    MockRpcMemLibrary*& active = ActiveInstance();
    if (active != nullptr) {
      throw std::runtime_error(
          "A MockRpcMemLibrary instance is already live. RpcMemApi callbacks carry no context "
          "pointer, so only one mock may exist at a time.");
    }
    active = this;

    api_.alloc = AllocThunk;
    api_.free = FreeThunk;
    api_.to_fd = ToFdThunk;
    api_.register_buf = RegisterBufThunk;
  }

  ~MockRpcMemLibrary() override {
    // Release anything alloc() handed out that the test did not free. Registered
    // buffers are owned by the caller (e.g. a memory-mapped weight file), so they
    // are only dropped from the table, never freed.
    for (const auto& [address, state] : buffers_) {
      if (state.origin == BufferOrigin::kAllocated) {
        ::operator delete(const_cast<void*>(address), std::align_val_t{kAllocationAlignment});
      }
    }
    ActiveInstance() = nullptr;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MockRpcMemLibrary);

  const qnn::RpcMemApi& Api() const override { return api_; }

  // --- register_buf() call history -----------------------------------------

  // Every register_buf() call, registrations and deregistrations alike, in call order.
  std::vector<RpcMemRegisterBufCall> RegisterBufCalls() const {
    std::scoped_lock g{mutex_};
    return register_buf_calls_;
  }

  size_t RegisterCallCount() const { return CountRegisterBufCalls(/* deregister */ false); }

  size_t DeregisterCallCount() const { return CountRegisterBufCalls(/* deregister */ true); }

  // Deregistration calls the mock accepted, i.e. those the EP observes as successful.
  size_t SuccessfulDeregisterCallCount() const {
    std::scoped_lock g{mutex_};
    return static_cast<size_t>(std::count_if(
        register_buf_calls_.begin(), register_buf_calls_.end(),
        [](const RpcMemRegisterBufCall& call) { return call.IsDeregister() && call.succeeded; }));
  }

  void ClearCallHistory() {
    std::scoped_lock g{mutex_};
    register_buf_calls_.clear();
    alloc_calls_.clear();
    free_call_count_ = 0;
  }

  // --- current registration state ------------------------------------------

  // True while to_fd(buff) would return a valid fd.
  bool IsRegistered(void* buff) const {
    std::scoped_lock g{mutex_};
    const auto it = buffers_.find(buff);
    return it != buffers_.end() && it->second.origin == BufferOrigin::kRegistered;
  }

  // Buffers registered and not yet deregistered, sorted by address for determinism.
  std::vector<std::pair<void*, size_t>> RegisteredBuffers() const {
    std::scoped_lock g{mutex_};
    std::vector<std::pair<void*, size_t>> registered;
    for (const auto& [address, state] : buffers_) {
      if (state.origin == BufferOrigin::kRegistered) {
        registered.emplace_back(const_cast<void*>(address), state.size);
      }
    }
    std::sort(registered.begin(), registered.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    return registered;
  }

  size_t RegisteredBufferCount() const { return RegisteredBuffers().size(); }

  // --- alloc() / free() ----------------------------------------------------

  std::vector<RpcMemAllocCall> AllocCalls() const {
    std::scoped_lock g{mutex_};
    return alloc_calls_;
  }

  size_t AllocCallCount() const {
    std::scoped_lock g{mutex_};
    return alloc_calls_.size();
  }

  size_t FreeCallCount() const {
    std::scoped_lock g{mutex_};
    return free_call_count_;
  }

  // Allocations handed out by alloc() and not yet passed to free().
  size_t LiveAllocationCount() const {
    std::scoped_lock g{mutex_};
    return static_cast<size_t>(std::count_if(
        buffers_.begin(), buffers_.end(),
        [](const auto& entry) { return entry.second.origin == BufferOrigin::kAllocated; }));
  }

  // --- fault injection -----------------------------------------------------

  void SetRegisterBufHook(RegisterBufHook hook) {
    std::scoped_lock g{mutex_};
    register_buf_hook_ = std::move(hook);
  }

  // Fail registration requests for `buff` until ClearFaults(); to_fd(buff) keeps
  // reporting -1, which the EP reports as "Failed to register ... to RPCMEM".
  void FailRegisterFor(void* buff) {
    std::scoped_lock g{mutex_};
    register_failures_.insert(buff);
  }

  // Fail deregistration requests for `buff` until ClearFaults(); the buffer stays
  // registered, so to_fd(buff) keeps returning a valid fd and the EP retains the
  // entry in mapped_fastrpc_buffers_.
  void FailDeregisterFor(void* buff) {
    std::scoped_lock g{mutex_};
    deregister_failures_.insert(buff);
  }

  // Make alloc() return nullptr until re-enabled.
  void SetAllocShouldFail(bool should_fail) {
    std::scoped_lock g{mutex_};
    alloc_should_fail_ = should_fail;
  }

  void ClearFaults() {
    std::scoped_lock g{mutex_};
    register_failures_.clear();
    deregister_failures_.clear();
    register_buf_hook_ = nullptr;
    alloc_should_fail_ = false;
  }

 private:
  // Matches HtpSharedMemoryAllocator's AllocationAlignment(), which throws if an
  // allocation comes back less aligned than this.
  static constexpr size_t kAllocationAlignment = 64;
  // First synthetic fd. Chosen to be clearly neither -1 nor a real low-numbered fd.
  static constexpr int kFirstFd = 1000;

  enum class BufferOrigin {
    kAllocated,   // came from alloc(); the mock owns the memory
    kRegistered,  // came from register_buf(); the caller owns the memory
  };

  struct BufferState {
    int fd;
    size_t size;
    BufferOrigin origin;
  };

  size_t CountRegisterBufCalls(bool deregister) const {
    std::scoped_lock g{mutex_};
    return static_cast<size_t>(std::count_if(
        register_buf_calls_.begin(), register_buf_calls_.end(),
        [deregister](const RpcMemRegisterBufCall& call) { return call.IsDeregister() == deregister; }));
  }

  // --- callback implementations --------------------------------------------

  void* OnAlloc(int heapid, uint32_t flags, int size) {
    std::scoped_lock g{mutex_};

    void* address = nullptr;
    if (!alloc_should_fail_ && size > 0) {
      const size_t size_bytes = static_cast<size_t>(size);
      // Round up so the size is a multiple of the alignment, as required by
      // aligned operator new.
      const size_t padded_size = (size_bytes + kAllocationAlignment - 1) & ~(kAllocationAlignment - 1);
      address = ::operator new(padded_size, std::align_val_t{kAllocationAlignment});
      buffers_[address] = BufferState{next_fd_++, size_bytes, BufferOrigin::kAllocated};
    }

    alloc_calls_.push_back(RpcMemAllocCall{heapid, flags, size, address});
    return address;
  }

  void OnFree(void* po) {
    std::scoped_lock g{mutex_};
    ++free_call_count_;

    if (po == nullptr) {
      return;
    }

    // rpcmem_free() ignores buffers it did not allocate.
    const auto it = buffers_.find(po);
    if (it == buffers_.end() || it->second.origin != BufferOrigin::kAllocated) {
      return;
    }

    buffers_.erase(it);
    ::operator delete(po, std::align_val_t{kAllocationAlignment});
  }

  int OnToFd(void* po) {
    std::scoped_lock g{mutex_};
    const auto it = buffers_.find(po);
    return it != buffers_.end() ? it->second.fd : -1;
  }

  void OnRegisterBuf(void* buff, size_t size, int fd, int attr) {
    std::scoped_lock g{mutex_};

    RpcMemRegisterBufCall call{buff, size, fd, attr, /* succeeded */ false};
    const bool deregister = call.IsDeregister();
    const auto& failures = deregister ? deregister_failures_ : register_failures_;

    call.succeeded = failures.find(buff) == failures.end() &&
                     (register_buf_hook_ == nullptr || register_buf_hook_(call));

    if (call.succeeded) {
      if (deregister) {
        buffers_.erase(buff);
      } else {
        buffers_[buff] = BufferState{next_fd_++, size, BufferOrigin::kRegistered};
      }
    }

    register_buf_calls_.push_back(call);
  }

  // --- static thunks -------------------------------------------------------

  static MockRpcMemLibrary*& ActiveInstance() {
    static MockRpcMemLibrary* instance = nullptr;
    return instance;
  }

  static void* AllocThunk(int heapid, uint32_t flags, int size) {
    MockRpcMemLibrary* self = ActiveInstance();
    return self != nullptr ? self->OnAlloc(heapid, flags, size) : nullptr;
  }

  static void FreeThunk(void* po) {
    if (MockRpcMemLibrary* self = ActiveInstance(); self != nullptr) {
      self->OnFree(po);
    }
  }

  static int ToFdThunk(void* po) {
    MockRpcMemLibrary* self = ActiveInstance();
    return self != nullptr ? self->OnToFd(po) : -1;
  }

  static void RegisterBufThunk(void* buff, size_t size, int fd, int attr) {
    if (MockRpcMemLibrary* self = ActiveInstance(); self != nullptr) {
      self->OnRegisterBuf(buff, size, fd, attr);
    }
  }

  mutable std::mutex mutex_;

  qnn::RpcMemApi api_{};
  int next_fd_ = kFirstFd;

  // Buffers currently visible to to_fd(), keyed by the exact pointer the EP passes
  // (MapDmaData registers mapped_base + offset, not the base address).
  std::unordered_map<const void*, BufferState> buffers_;

  std::vector<RpcMemRegisterBufCall> register_buf_calls_;
  std::vector<RpcMemAllocCall> alloc_calls_;
  size_t free_call_count_ = 0;

  RegisterBufHook register_buf_hook_;
  std::unordered_set<const void*> register_failures_;
  std::unordered_set<const void*> deregister_failures_;
  bool alloc_should_fail_ = false;
};

}  // namespace test
}  // namespace onnxruntime
