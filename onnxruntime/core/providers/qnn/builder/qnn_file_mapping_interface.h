// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

class FileMappingInterface {
 public:
  virtual ~FileMappingInterface() = default;

  virtual Ort::Status GetContextBinMappedMemoryPtr(const std::string& bin_filepath,
                                                   void** mapped_data_ptr) = 0;

  // Returns an independently owned view. A context can keep this mapping alive
  // without retaining the mapper or another context's DMA callback state.
  virtual Ort::Status MapContextBinary(const std::string& bin_filepath,
                                       std::shared_ptr<char>& mapped_memory) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
