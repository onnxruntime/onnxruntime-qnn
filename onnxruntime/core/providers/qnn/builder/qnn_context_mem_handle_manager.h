// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

#include "QnnInterface.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/common/inlined_containers.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime::qnn {

// One QNN mem handle registration (QnnMem_register result) in one QNN context.
//
// It is shared by the context's QnnContextMemHandleManager and by the clean up hook attached to the backing memory
// (an import or a shared-memory allocation), and is deregistered exactly once by whichever reaches Deregister()
// first. Deregister() is serialized, and the context deregisters its registrations before the QNN context is freed,
// so a hook that runs while the context is being destroyed either deregisters first or finds the work done. This
// keeps "QNN deregistration before the memory is unmapped or freed" independent of whether the backend manager or the
// context record can still be locked.
class QnnMemHandleRegistration {
 public:
  QnnMemHandleRegistration(const QNN_INTERFACE_VER_TYPE& qnn_interface, Qnn_MemHandle_t mem_handle,
                           size_t registered_tensor_data_size);

  ~QnnMemHandleRegistration();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnMemHandleRegistration);

  // Calls QnnMem_deRegister if the handle is still registered. Safe to call more than once and from any thread.
  void Deregister() noexcept;

  bool IsRegistered() const;

  // The registered handle, or nullptr once deregistered.
  Qnn_MemHandle_t Handle() const;

  size_t RegisteredTensorDataSize() const { return registered_tensor_data_size_; }

 private:
  // Copied so a hook never reads the backend manager's interface table. The function pointers are only called while
  // the handle is registered, which implies the QNN context, and so the backend library, is still alive.
  const QNN_INTERFACE_VER_TYPE qnn_interface_;
  const size_t registered_tensor_data_size_;
  mutable std::mutex mutex_;
  Qnn_MemHandle_t mem_handle_;  // nullptr once deregistered
};

// This class manages QNN mem handles (Qnn_MemHandle_t) associated with a QNN context (Qnn_ContextHandle_t).
// In particular, it handles the registration and deregistration of mem handles.
// The associated QNN context is expected to be in scope for the lifetime of the QnnContextMemHandleManager.
class QnnContextMemHandleManager {
 public:
  QnnContextMemHandleManager(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                             Qnn_ContextHandle_t qnn_context,
                             QnnBackendType qnn_backend_type,
                             QnnAllocatorType qnn_allocator_type);

  ~QnnContextMemHandleManager();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnContextMemHandleManager);

  // Gets an existing QNN mem handle or registers a new one.
  // `qnn_mem_handle` is set to the QNN mem handle and `did_register` is true if `qnn_mem_handle` was newly registered.
  Ort::Status GetOrRegister(void* shared_memory_address,
                            const Qnn_Tensor_t& qnn_tensor,
                            Qnn_MemHandle_t& qnn_mem_handle,
                            bool& did_register,
                            const Ort::Logger& logger);

  // Returns the registration for `shared_memory_address`, or nullptr if there is none. Records whose registration was
  // deregistered by the memory's clean up hook are retired on the next registration in this context.
  std::shared_ptr<QnnMemHandleRegistration> GetRegistration(const void* shared_memory_address);

  Ort::Status Unregister(void* shared_memory_address);

  // Drops the record for `shared_memory_address`, if any, deregistering its handle. Allocates nothing, so it is safe
  // on rollback paths, including during exception unwinding.
  void Discard(const void* shared_memory_address) noexcept;

  void Clear();

 private:
  const QNN_INTERFACE_VER_TYPE& qnn_interface_;
  Qnn_ContextHandle_t context_;
  QnnBackendType qnn_backend_type_;
  QnnAllocatorType qnn_allocator_type_;

  // Deregisters its registration when the record is dropped (Unregister, Clear, or context destruction).
  struct MemHandleRecord {
    explicit MemHandleRecord(std::shared_ptr<QnnMemHandleRegistration> registration_in)
        : registration{std::move(registration_in)} {}
    MemHandleRecord(MemHandleRecord&&) noexcept = default;
    MemHandleRecord& operator=(MemHandleRecord&& other) noexcept {
      if (this != &other) {
        if (registration != nullptr) {
          registration->Deregister();
        }
        registration = std::move(other.registration);
      }
      return *this;
    }
    MemHandleRecord(const MemHandleRecord&) = delete;
    MemHandleRecord& operator=(const MemHandleRecord&) = delete;
    ~MemHandleRecord() {
      if (registration != nullptr) {
        registration->Deregister();
      }
    }

    std::shared_ptr<QnnMemHandleRegistration> registration;
  };

  // Drops records whose registration has already been deregistered. Requires mem_handles_mutex_.
  void RetireDeregisteredLocked();

  // shared memory address -> associated mem handle record
  InlinedHashMap<const void*, MemHandleRecord> mem_handles_;
  std::mutex mem_handles_mutex_;  // synchronize access to mem_handles_
};

}  // namespace onnxruntime::qnn
