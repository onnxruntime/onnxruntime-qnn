// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <QnnContext.h>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {

class FileMappingInterface {
 public:
  virtual ~FileMappingInterface() = default;

  virtual Ort::Status GetContextBinMappedMemoryPtr(const std::string& bin_filepath, void** mapped_data_ptr) = 0;
};

}  // namespace qnn
}  // namespace onnxruntime
