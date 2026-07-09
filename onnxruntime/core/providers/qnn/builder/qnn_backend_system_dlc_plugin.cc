// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_backend_system_dlc_plugin.h"

#include <memory>
#include <vector>

#include "QnnCommon.h"
#include "HTP/QnnHtpSystemContext.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemDlc.h"
#include "System/QnnSystemInterface.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

QnnBackendSystemDlcPlugin::QnnBackendSystemDlcPlugin(QnnBackendManager* qnn_backend_manager)
    : qnn_backend_manager_(qnn_backend_manager) {}

QnnBackendSystemDlcPlugin::~QnnBackendSystemDlcPlugin() {
#ifdef QNN_SYSTEM_DLC_API_ENABLED
  Ort::Status status = ReleaseDlc();
  if (!status.IsOK()) {
    ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_,
                    ORT_LOGGING_LEVEL_ERROR,
                    ("Failed to ReleaseDlc: " + status.GetErrorMessage()).c_str());
  }
#endif  // QNN_SYSTEM_DLC_API_ENABLED
}

Ort::Status QnnBackendSystemDlcPlugin::AddContextToDlc(const Qnn_ContextHandle_t& context_handle) {
  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Adding context to DLC.");

#ifdef QNN_SYSTEM_DLC_API_ENABLED
  RETURN_IF(qnn_backend_manager_->qnn_interface_.contextAddToDlc == nullptr,
            "Failed to add context to DLC without QnnContext_addToDlc API.");

  Qnn_ErrorHandle_t result = qnn_backend_manager_->qnn_interface_.contextAddToDlc(context_handle, dlc_handle_);
  RETURN_IF(result != QNN_SUCCESS,
            ("Failed to add context to DLC. Error: " + qnn_backend_manager_->QnnErrorHandleToString(result)).c_str());

  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Context added to DLC.");
  return Ort::Status();
#else
  ORT_UNUSED_PARAMETER(context_handle);
  return MAKE_EP_FAIL("Context adding to DLC is only supported in QAIRT 2.48+ SDK.");
#endif  // QNN_SYSTEM_DLC_API_ENABLED
}

Ort::Status QnnBackendSystemDlcPlugin::CreateDlc() {
  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Creating DLC.");

#ifdef QNN_SYSTEM_DLC_API_ENABLED
  if (dlc_created_) {
    ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "DLC created already.");
    return Ort::Status();
  }

  RETURN_IF(qnn_backend_manager_->qnn_sys_interface_.systemDlcCreateWithDestinationDir == nullptr,
            "Failed to create DLC without QnnSystemDlc_createWithDestinationDir API.");

  Qnn_ErrorHandle_t result = qnn_backend_manager_->qnn_sys_interface_.systemDlcCreateWithDestinationDir(
      qnn_backend_manager_->log_handle_,
      nullptr,
      &dlc_handle_);
  RETURN_IF(result != QNN_SUCCESS,
            ("Failed to create DLC. Error: " + qnn_backend_manager_->QnnErrorHandleToString(result)).c_str());

  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "DLC created.");
  dlc_created_ = true;

  return Ort::Status();
#else
  return MAKE_EP_FAIL("DLC creation is only supported in QAIRT 2.48+ SDK.");
#endif  // QNN_SYSTEM_DLC_API_ENABLED
}

Ort::Status QnnBackendSystemDlcPlugin::ReleaseDlc() {
  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Freeing DLC.");

#ifdef QNN_SYSTEM_DLC_API_ENABLED
  if (!dlc_created_) {
    ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "No DLC to be freed.");
    return Ort::Status();
  }

  RETURN_IF(qnn_backend_manager_->qnn_sys_interface_.systemDlcFree == nullptr,
            "Failed to free DLC without QnnSystemDlc_free API.");

  Qnn_ErrorHandle_t result = qnn_backend_manager_->qnn_sys_interface_.systemDlcFree(dlc_handle_);
  RETURN_IF(result != QNN_SUCCESS,
            ("Failed to free DLC. Error: " + qnn_backend_manager_->QnnErrorHandleToString(result)).c_str());

  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "DLC freed");
  dlc_handle_ = nullptr;
  dlc_created_ = false;

  return Ort::Status();
#else
  return MAKE_EP_FAIL("DLC free is only supported in QAIRT 2.48+ SDK.");
#endif  // QNN_SYSTEM_DLC_API_ENABLED
}

Ort::Status QnnBackendSystemDlcPlugin::GetDlcBinaryBuffer(unsigned char** dlc_buffer, uint64_t& buffer_size) {
  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Getting DLC binary.");

#ifdef QNN_SYSTEM_DLC_API_ENABLED
  RETURN_IF(dlc_buffer == nullptr, "Null dlc_buffer pointer provided.");
  // In current workflow, DLC binary is acquired when creating EP context nodes, and thus DLC should already be created.
  RETURN_IF_NOT(dlc_created_, "No QNN DLC to get DLC binary from.");
  RETURN_IF(qnn_backend_manager_->qnn_sys_interface_.systemDlcGetBinarySize == nullptr,
            "Failed to get DLC binary buffer without QnnSystemDlc_getBinarySize API.");
  RETURN_IF(qnn_backend_manager_->qnn_sys_interface_.systemDlcGetBinary == nullptr,
            "Failed to get DLC binary buffer without QnnSystemDlc_getBinary API.");

  uint64_t required_buffer_size = 0;
  Qnn_ErrorHandle_t rt = qnn_backend_manager_->qnn_sys_interface_.systemDlcGetBinarySize(dlc_handle_,
                                                                                         &required_buffer_size);
  RETURN_IF(rt != QNN_SUCCESS,
            ("Failed to get QNN DLC binary size. Error: " + qnn_backend_manager_->QnnErrorHandleToString(rt)).c_str());

  auto buffer = std::make_unique<unsigned char[]>(required_buffer_size);
  RETURN_IF(buffer == nullptr, "Failed to allocate buffer for DLC binary.");

  uint64_t written_buffer_size = 0;
  rt = qnn_backend_manager_->qnn_sys_interface_.systemDlcGetBinary(dlc_handle_,
                                                                   reinterpret_cast<uint8_t*>(buffer.get()),
                                                                   required_buffer_size,
                                                                   &written_buffer_size);
  RETURN_IF(rt != QNN_SUCCESS,
            ("Failed to get QNN DLC binary. Error: " + qnn_backend_manager_->QnnErrorHandleToString(rt)).c_str());
  RETURN_IF(required_buffer_size < written_buffer_size,
            ("Context written buffer size: " + std::to_string(written_buffer_size) +
             " exceeds allocated buffer size: " + std::to_string(required_buffer_size))
                .c_str());

  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "DLC binary buffer got.");
  *dlc_buffer = buffer.release();
  buffer_size = written_buffer_size;

  return Ort::Status();
#else
  ORT_UNUSED_PARAMETER(dlc_buffer);
  ORT_UNUSED_PARAMETER(buffer_size);
  return MAKE_EP_FAIL("DLC binary acquisition is only supported in QAIRT 2.48+ SDK.");
#endif  // QNN_SYSTEM_DLC_API_ENABLED
}

Ort::Status QnnBackendSystemDlcPlugin::GetDlcBinaryInfo(QnnSystemContext_Handle_t sys_ctx_handle,
                                                        const uint8_t* buffer,
                                                        uint64_t buffer_length,
                                                        Qnn_Version_t& blob_version,
                                                        uint32_t& graph_count,
                                                        QnnSystemContext_GraphInfo_t** graphs_info) {
  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Getting DLC binary info.");

#ifdef QNN_SYSTEM_DLC_API_ENABLED
  // In current workflow, DLC binary info is required when backend attempts to load from EP context, and thus DLC
  // should not be created already.
  RETURN_IF(dlc_created_, "DLC is unexpectedly created already.");
  RETURN_IF(qnn_backend_manager_->qnn_sys_interface_.systemDlcCreateFromBinary == nullptr,
            "Failed to get DLC binary info without QnnSystemDlc_createFromBinary API.");

  Qnn_ErrorHandle_t rt = qnn_backend_manager_->qnn_sys_interface_.systemDlcCreateFromBinary(
      qnn_backend_manager_->log_handle_,
      buffer,
      buffer_length,
      &dlc_handle_);
  RETURN_IF(rt != QNN_SUCCESS,
            ("Failed to create QNN DLC from binary. Error: " + qnn_backend_manager_->QnnErrorHandleToString(rt))
                .c_str());
  dlc_created_ = true;

  std::vector<const uint8_t*> record_buffers;
  std::vector<uint64_t> record_buffer_sizes;
  RETURN_IF_ERROR(GetDlcRecordBuffers(true, record_buffers, record_buffer_sizes));

  // Only one buffer is acquired as setting `most_optimal_only=true` above, and thus only the first one in vector is
  // passed to get binary info.
  RETURN_IF_ERROR(qnn_backend_manager_->GetGraphInfoAndBinVersion(
      sys_ctx_handle,
      const_cast<void*>(static_cast<const void*>(record_buffers[0])),
      record_buffer_sizes[0],
      blob_version,
      graph_count,
      graphs_info));

  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "DLC binary info got.");
  RETURN_IF_ERROR(ReleaseDlc());

  return Ort::Status();
#else
  ORT_UNUSED_PARAMETER(sys_ctx_handle);
  ORT_UNUSED_PARAMETER(buffer);
  ORT_UNUSED_PARAMETER(buffer_length);
  ORT_UNUSED_PARAMETER(blob_version);
  ORT_UNUSED_PARAMETER(graph_count);
  ORT_UNUSED_PARAMETER(graphs_info);
  return MAKE_EP_FAIL("DLC binary info acquisition is only supported in QAIRT 2.48+ SDK.");
#endif  // QNN_SYSTEM_DLC_API_ENABLED
}

Ort::Status QnnBackendSystemDlcPlugin::GetDlcMaxSpillFillBufferSize(uint64_t& max_spill_fill_buffer_size) {
  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Getting DLC max spill-fill buffer size.");

#ifdef QNN_SYSTEM_DLC_API_ENABLED
  // In current workflow, spill-fill buffer size is queried after DLC buffer is acquired, and thus DLC should already
  // be created.
  RETURN_IF_NOT(dlc_created_, "No DLC to get max spill-fill buffer size from.");

  std::vector<const uint8_t*> record_buffers;
  std::vector<uint64_t> record_buffer_sizes;
  RETURN_IF_ERROR(GetDlcRecordBuffers(false, record_buffers, record_buffer_sizes));

  auto sys_ctx_handle = qnn_backend_manager_->GetSystemContextHandle();
  RETURN_IF(sys_ctx_handle == nullptr, "Failed to get DLC max spill-fill buffer size info without system context.");

  // Note that spill-fill buffer requires QNN API >= 2.21 and DLC-related usage requires QNN API >= 2.37. Since DLC
  // usage is already guarded upfront, skip adding macro guard here.
  max_spill_fill_buffer_size = 0;
  for (size_t record_idx = 0; record_idx < record_buffers.size(); ++record_idx) {
    Qnn_Version_t blob_version = {0, 0, 0};
    uint32_t graph_count = 0;
    QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
    RETURN_IF_ERROR(qnn_backend_manager_->GetGraphInfoAndBinVersion(
        sys_ctx_handle.get(),
        const_cast<void*>(static_cast<const void*>(record_buffers[record_idx])),
        record_buffer_sizes[record_idx],
        blob_version,
        graph_count,
        &graphs_info));

    for (uint32_t graph_idx = 0; graph_idx < graph_count; ++graph_idx) {
      if (graphs_info[graph_idx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        auto htp_graph_info = reinterpret_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(
            graphs_info[graph_idx].graphInfoV3.graphBlobInfo);
        if (htp_graph_info->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
          auto spill_fill_buffer_size = htp_graph_info->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
          max_spill_fill_buffer_size = spill_fill_buffer_size > max_spill_fill_buffer_size ? spill_fill_buffer_size
                                                                                           : max_spill_fill_buffer_size;
        } else {
          ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_,
                          ORT_LOGGING_LEVEL_VERBOSE,
                          "Unknown system context HTP graph info blob version.");
        }
      } else if (graphs_info[graph_idx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1 ||
                 graphs_info[graph_idx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_,
                        ORT_LOGGING_LEVEL_VERBOSE,
                        "Skip as not supported in system context graph info v1 & v2.");
      } else {
        ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_,
                        ORT_LOGGING_LEVEL_VERBOSE,
                        "Unknown system context graph info version.");
      }
    }
  }

  ORT_CXX_LOG_PTR(qnn_backend_manager_->logger_ptr_, ORT_LOGGING_LEVEL_INFO, "DLC max spill-fill buffer size got.");
  return Ort::Status();
#else
  ORT_UNUSED_PARAMETER(max_spill_fill_buffer_size);
  return MAKE_EP_FAIL("DLC max spill-fill buffer size acquisition is only supported in QAIRT 2.48+ SDK.");
#endif  // QNN_SYSTEM_DLC_API_ENABLED
}

Ort::Status QnnBackendSystemDlcPlugin::GetDlcRecordBuffers(bool most_optimal_only,
                                                           std::vector<const uint8_t*>& record_buffers,
                                                           std::vector<uint64_t>& record_buffer_sizes) {
#ifdef QNN_SYSTEM_DLC_API_ENABLED
  RETURN_IF_NOT(dlc_created_, "No DLC to get record buffers from.");
  RETURN_IF(qnn_backend_manager_->qnn_sys_interface_.systemDlcGetRecordsByType == nullptr,
            "Failed to get DLC record buffers without QnnSystemDlc_getRecordsByType API.");
  RETURN_IF(qnn_backend_manager_->qnn_sys_interface_.systemDlcReadRecordDataMemoryMapped == nullptr,
            "Failed to get DLC record buffers without QnnSystemDlc_readRecordDataMemoryMapped API.");

  QnnSystemDlc_RecordHandle_t* record_handles = nullptr;
  uint32_t num_record_handles = 0;
  Qnn_ErrorHandle_t rt = qnn_backend_manager_->qnn_sys_interface_.systemDlcGetRecordsByType(
      dlc_handle_,
      QNN_SYSTEM_DLC_RECORD_PREFIX_HTP_CACHE_RECORD,
      static_cast<uint8_t>(most_optimal_only),
      &record_handles,
      &num_record_handles);
  RETURN_IF(rt != QNN_SUCCESS,
            ("Failed to get record from QNN DLC by type. Error: " + qnn_backend_manager_->QnnErrorHandleToString(rt))
                .c_str());
  RETURN_IF_NOT(num_record_handles > 0, "Expecting at least one record handle but got none.");

  record_buffers.assign(num_record_handles, nullptr);
  record_buffer_sizes.assign(num_record_handles, 0);

  for (size_t record_idx = 0; record_idx < num_record_handles; ++record_idx) {
    rt = qnn_backend_manager_->qnn_sys_interface_.systemDlcReadRecordDataMemoryMapped(record_handles[record_idx],
                                                                                      &record_buffers[record_idx],
                                                                                      &record_buffer_sizes[record_idx]);
    RETURN_IF(rt != QNN_SUCCESS,
              ("Failed to read record data. Error: " + qnn_backend_manager_->QnnErrorHandleToString(rt)).c_str());
  }

  return Ort::Status();
#else
  ORT_UNUSED_PARAMETER(most_optimal_only);
  ORT_UNUSED_PARAMETER(record_buffers);
  ORT_UNUSED_PARAMETER(record_buffer_sizes);
  return MAKE_EP_FAIL("DLC record buffer acquisition is only supported in QAIRT 2.48+ SDK.");
#endif  // QNN_SYSTEM_DLC_API_ENABLED
}

}  // namespace qnn
}  // namespace onnxruntime
