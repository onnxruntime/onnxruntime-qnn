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

  // Get binary buffer from DLC handle.
  Ort::Status GetDlcBinaryBuffer(/*out*/ unsigned char** dlc_buffer, /*out*/ uint64_t& buffer_size);

  // Get binary info from DLC handle.
  // TODO:
  //   This function is designed based on current usage where DLC handle is created from binary beforehand. It may fail
  //   to get info for a normal DLC handle. Revise this function in the future to accommodate new usage if necessary.
  Ort::Status GetDlcBinaryInfo(QnnSystemContext_Handle_t sys_ctx_handle,
                               /*out*/ Qnn_Version_t& blob_version,
                               /*out*/ uint32_t& graph_count,
                               /*out*/ QnnSystemContext_GraphInfo_t** graphs_info);

  // Get max spill-fill buffer size from DLC handle.
  // TODO:
  //   This function is designed based on current usage where DLC handle is created from binary beforehand. It may fail
  //   to get info for a normal DLC handle. Revise this function in the future to accommodate new usage if necessary.
  Ort::Status GetDlcMaxSpillFillBufferSize(/*out*/ uint64_t& max_spill_fill_buffer_size);

  // Release handles.
  Ort::Status Release();

  // Setup DLC from scratch.
  Ort::Status SetupDlc();

  // Setup DLC from binary.
  Ort::Status SetupDlcFromBinary(const uint8_t* buffer, uint64_t buffer_length);

 private:
  // Create DLC handle from binary.
  Ort::Status CreateDlcFromBinary(const uint8_t* buffer, uint64_t buffer_length);

  // Create empty DLC handle.
  Ort::Status CreateEmptyDlc();

  // Create SystemLog handle.
  Ort::Status CreateSystemLog();

  // Get HTP cache record buffers (i.e., context binaries) from DLC handle.
  Ort::Status GetDlcRecordBuffers(bool most_optimal_only,
                                  /*out*/ std::vector<const uint8_t*>& record_buffers,
                                  /*out*/ std::vector<uint64_t>& record_buffer_sizes);

  // Release DLC handle.
  Ort::Status ReleaseDlc();

  // Release SystemLog handle.
  Ort::Status ReleaseSystemLog();

 private:
  // Unowned backend manager pointer.
  QnnBackendManager* qnn_backend_manager_;

  // As all uses are guarded by QNN_SYSTEM_DLC_API_ENABLED, guard them here to avoid compiler errors.
#ifdef QNN_SYSTEM_DLC_API_ENABLED
  bool dlc_created_ = false;
  QnnSystemDlc_Handle_t dlc_handle_ = nullptr;
#endif  // QNN_SYSTEM_DLC_API_ENABLED
  Qnn_LogHandle_t system_log_handle_ = nullptr;
};

}  // namespace qnn
}  // namespace onnxruntime
