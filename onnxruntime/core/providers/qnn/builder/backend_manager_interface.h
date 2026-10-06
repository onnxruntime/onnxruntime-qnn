// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "System/QnnSystemInterface.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations
class QnnModel;
class EpContextIoDispatch;

// Pure virtual interface for backend lifecycle: library load, backend/context creation,
// power configuration, context binary serialize/deserialize, and profiling handles.
// Concrete implementations:
//   - QnnBackendManager  : QNN C function-pointer path (default)
//   - QairtBackendManager: QAIRT C++ API (Phase 2, USE_QAIRT_API=ON)
class IBackendManager {
 public:
  virtual ~IBackendManager() = default;

  // Accessors for the underlying QNN C interface and handles.
  virtual const QNN_INTERFACE_VER_TYPE& GetQnnInterface() const = 0;
  virtual const QNN_INTERFACE_VER_TYPE& GetQnnValidatorInterface() const = 0;
  virtual const Qnn_BackendHandle_t& GetQnnBackendHandle() const = 0;
  virtual const Qnn_BackendHandle_t& GetQnnValidatorBackendHandle() const = 0;
  virtual const Qnn_ContextHandle_t& GetQnnContext(int index = 0) = 0;
  virtual QnnBackendType GetQnnBackendType() const = 0;

  // Context binary: serialize current context to a buffer.
  virtual Ort::Status GetContextBinaryBuffer(bool is_multi_soc_buffer,
                                             /*out*/ unsigned char** context_buffer,
                                             /*out*/ uint64_t& buffer_size) = 0;

  // Context binary: deserialize from a buffer.
  virtual Ort::Status LoadCachedQnnContextFromBuffer(
      char* buffer,
      uint64_t buffer_length,
      const std::string& context_bin_filepath,
      std::string node_name,
      std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models,
      int64_t max_spill_fill_size,
      const qnn::EpContextIoDispatch& io_dispatch,
      bool is_multi_soc_buffer = false) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
