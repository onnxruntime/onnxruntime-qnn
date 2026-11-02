// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "QnnContext.h"
#include "core/providers/qnn/rpcmem_library.h"

namespace onnxruntime::qnn {

// Owns one plaintext allocation and its FastRPC imports. The QNN notify pointer
// refers to this object, never to a backend manager. Context teardown must call
// ContextReleased only after contextFree succeeds.
class QnnContextBinaryDmaBuffer {
 public:
  static std::shared_ptr<QnnContextBinaryDmaBuffer> Create(
      std::shared_ptr<char> buffer, size_t size, RpcMemApi api,
      std::shared_ptr<void> library) {
    return {new QnnContextBinaryDmaBuffer(std::move(buffer), size, api, std::move(library)),
            [](QnnContextBinaryDmaBuffer* state) {
              if (state->TryRelease()) {
                delete state;
              } else {
                // The driver or QNN may still reference the allocation/notify
                // pointer. Intentionally retain this isolated state on terminal
                // teardown failure instead of unmapping live DMA memory.
                if (OrtLoggingManager::HasDefaultLogger()) {
                  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR,
                              "Retaining EPContext DMA buffer after failed context or FastRPC teardown.");
                }
              }
            }};
  }

  Qnn_ContextBinaryCallback_t Callbacks() {
    Qnn_ContextBinaryCallback_t callbacks{};
    callbacks.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
    callbacks.dmaBufferCallback.version = QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
    callbacks.dmaBufferCallback.v1.dataProvide = Provide;
    callbacks.dmaBufferCallback.v1.dataRelease = Release;
    callbacks.dmaBufferCallback.v1.notifyParam = this;
    return callbacks;
  }

  void ContextCreated() {
    std::lock_guard<std::mutex> lock(mutex_);
    context_live_ = true;
  }

  void ContextReleased() {
    std::lock_guard<std::mutex> lock(mutex_);
    context_live_ = false;
  }

  // Also used when context creation fails before returning a live handle.
  // Only this allocation's imports are touched; other contexts are unaffected.
  bool TryRelease() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (context_live_) return false;
    for (auto it = imports_.begin(); it != imports_.end();) {
      if (Deregister(it->first, it->second.size)) {
        it = imports_.erase(it);
      } else {
        ++it;
      }
    }
    return imports_.empty();
  }

  static Qnn_ErrorHandle_t Provide(Qnn_ContextBinaryDataRequest_t request,
                                   Qnn_ContextBinaryDmaDataResponse_t* response,
                                   void* opaque) noexcept {
    if (!opaque || !response) return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
    auto& state = *static_cast<QnnContextBinaryDmaBuffer*>(opaque);
    if (!state.buffer_ || !state.api_.register_buf || !state.api_.to_fd ||
        !request.isBackendMappingNeeded || request.size == 0 ||
        request.offset > state.size_ || request.size > state.size_ - request.offset) {
      return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
    }
    try {
      std::lock_guard<std::mutex> lock(state.mutex_);
      void* address = state.buffer_.get() + static_cast<size_t>(request.offset);
      // Allocate tracking storage before registration, so an allocation failure
      // cannot leave an untracked DMA import.
      auto [it, inserted] = state.imports_.try_emplace(address, Import{static_cast<size_t>(request.size), 0});
      if (!inserted && it->second.size != request.size) return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
      if (inserted) {
        state.api_.register_buf(address, static_cast<size_t>(request.size), 0, kAttributes);
      }
      const int fd = state.api_.to_fd(address);
      if (fd == -1) {
        if (inserted) state.imports_.erase(it);
        return QNN_COMMON_ERROR_SYSTEM;
      }
      ++it->second.references;
      response->dmaBuffer.fd = fd;
      response->dmaBuffer.data = address;
      response->dataStartOffset = 0;
      response->alignedSize = request.size;
      return QNN_SUCCESS;
    } catch (...) {
      return QNN_CONTEXT_ERROR_MEM_ALLOC;
    }
  }

  static Qnn_ErrorHandle_t Release(Qnn_ContextBinaryDmaDataMem_t memory, void* opaque) noexcept {
    if (!opaque || !memory.dmaBuffer.data || !memory.memSize) return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
    auto& state = *static_cast<QnnContextBinaryDmaBuffer*>(opaque);
    try {
      std::lock_guard<std::mutex> lock(state.mutex_);
      auto it = state.imports_.find(memory.dmaBuffer.data);
      if (it == state.imports_.end() || it->second.size != memory.memSize) return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
      if (it->second.references > 1) {
        --it->second.references;
        return QNN_SUCCESS;
      }
      if (!state.Deregister(it->first, it->second.size)) return QNN_CONTEXT_ERROR_MEM_ALLOC;
      state.imports_.erase(it);
      return QNN_SUCCESS;
    } catch (...) {
      return QNN_CONTEXT_ERROR_MEM_ALLOC;
    }
  }

 private:
  struct Import {
    size_t size;
    size_t references;
  };
  static constexpr int kAttributes = rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY;

  QnnContextBinaryDmaBuffer(std::shared_ptr<char> buffer, size_t size, RpcMemApi api,
                            std::shared_ptr<void> library)
      : buffer_(std::move(buffer)), size_(size), api_(api), library_(std::move(library)) {}

  bool Deregister(void* address, size_t size) {
    api_.register_buf(address, size, -1, kAttributes);
    return api_.to_fd(address) == -1;
  }

  std::shared_ptr<char> buffer_;
  const size_t size_;
  const RpcMemApi api_;
  // Keep the FastRPC function pointers valid even if teardown must retain state.
  std::shared_ptr<void> library_;
  std::mutex mutex_;
  std::unordered_map<void*, Import> imports_;
  bool context_live_ = false;
};

}  // namespace onnxruntime::qnn
