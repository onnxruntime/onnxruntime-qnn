// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "core/providers/qnn/genie/genie_node.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

class QnnEp;

struct GenieNodeComputeInfo : OrtNodeComputeInfo {
  GenieNodeComputeInfo(QnnEp& ep, std::shared_ptr<GenieNodeBuilder> builder);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr,
                                             void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr,
                                            void* compute_state);

  QnnEp& ep;
  std::shared_ptr<GenieNodeBuilder> builder;
};

}  // namespace onnxruntime
