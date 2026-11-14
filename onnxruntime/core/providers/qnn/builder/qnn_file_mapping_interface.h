// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <string>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

class FileMappingInterface {
 public:
  using MappedMemoryPtr = std::unique_ptr<char[], std::function<void(void*)>>;

  virtual ~FileMappingInterface() = default;

  virtual Ort::Status GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                                   void** mapped_data_ptr) = 0;

  // Returns a caller-owned read-only view.
  virtual Ort::Status MapContextBinary(const std::string& bin_filepath,
                                       MappedMemoryPtr& mapped_memory) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
