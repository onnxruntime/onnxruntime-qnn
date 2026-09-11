// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Forward declaration.
class QnnBackendManager;

class QnnBackendSystemDlcPlugin {
 public:
  QnnBackendSystemDlcPlugin(QnnBackendManager* qnn_backend_manager);

  ~QnnBackendSystemDlcPlugin();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnBackendSystemDlcPlugin);

  // Add current context handle into DLC handle.
  Ort::Status AddContextToDlc(const Qnn_ContextHandle_t& context_handle);

  // Create DLC handle.
  Ort::Status CreateDlc();

  // Release DLC handle.
  Ort::Status ReleaseDlc();

  // Get binary buffer from DLC handle.
  Ort::Status GetDlcBinaryBuffer(/*out*/ unsigned char** dlc_buffer, /*out*/ uint64_t& buffer_size);

  // Get binary info from given DLC binary.
  // TODO:
  //   This function is designed based on current usage where DLC handle is not created beforehand and thus DLC buffer
  //   is passed to create DLC handle. However, the buffer could be optional if DLC handle is already created. Revise
  //   this function in the future to accommodate new usage if necessary.
  Ort::Status GetDlcBinaryInfo(QnnSystemContext_Handle_t sys_ctx_handle,
                               const uint8_t* buffer,
                               uint64_t buffer_length,
                               /*out*/ Qnn_Version_t& blob_version,
                               /*out*/ uint32_t& graph_count,
                               /*out*/ QnnSystemContext_GraphInfo_t** graphs_info);

  // Get max spill-fill buffer size from given DLC binary.
  // TODO:
  //   This function is designed based on current usage where DLC handle is created beforehand and thus no need to
  //   provide DLC buffer. However, the DLC buffer could be mandatory if DLC handle is not already created. Revise this
  //   function in the future to accommodate new usage if necessary.
  Ort::Status GetDlcMaxSpillFillBufferSize(/*out*/ uint64_t& max_spill_fill_buffer_size);

 private:
  // Get HTP cache record buffers (i.e., context binaries) from DLC handle.
  Ort::Status GetDlcRecordBuffers(bool most_optimal_only,
                                  /*out*/ std::vector<const uint8_t*>& record_buffers,
                                  /*out*/ std::vector<uint64_t>& record_buffer_sizes);

 private:
  // Unowned backend manager pointer.
  QnnBackendManager* qnn_backend_manager_;

  // As all uses are guarded by QNN_SYSTEM_DLC_API_ENABLED, guard them here to avoid compiler errors.
#ifdef QNN_SYSTEM_DLC_API_ENABLED
  bool dlc_created_ = false;
  QnnSystemDlc_Handle_t dlc_handle_ = nullptr;
#endif  // QNN_SYSTEM_DLC_API_ENABLED
};

}  // namespace qnn
}  // namespace onnxruntime
