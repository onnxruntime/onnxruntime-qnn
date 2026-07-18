// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <memory>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/custom_op/qnn_custom_op.h"
#include "core/providers/qnn/qnn_execution_provider.h"

namespace onnxruntime::qnn {
class RpcMemLibrary;
}  // namespace onnxruntime::qnn

namespace onnxruntime {

class QnnEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  QnnEpFactory(const char* ep_name, ApiPtrs ort_api_in);

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept;
  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ OrtEp** ep) noexcept;
  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;
  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEpFactory* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _In_opt_ const OrtKeyValuePairs* allocator_options,
                                                     _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept;
  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;
  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;
  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
      _In_ OrtEpFactory* this_ptr,
      _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
      _In_ size_t num_devices,
      _In_ const char* compatibility_info,
      _Out_ OrtCompiledModelCompatibility* model_compatibility) noexcept;
  static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
      _In_ OrtEpFactory* this_ptr,
      _In_ const OrtHardwareDevice* hw,
      _Inout_ OrtDeviceEpIncompatibilityDetails* details) noexcept;
  static OrtStatus* ORT_API_CALL CreateExternalResourceImporterForDeviceImpl(
      _In_ OrtEpFactory* this_ptr,
      _In_ const OrtEpDevice* ep_device,
      _Out_ OrtExternalResourceImporterImpl** out_importer) noexcept;
  static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(
      _In_ OrtEpFactory* this_ptr,
      _Out_ size_t* num_domains) noexcept;
  static OrtStatus* ORT_API_CALL GetCustomOpDomainsImpl(
      _In_ OrtEpFactory* this_ptr,
      _Out_writes_all_(num_domains) OrtCustomOpDomain** domains,
      _In_ size_t num_domains) noexcept;

  const std::string ep_name_;              // EP name
  const std::string vendor_{"Qualcomm"};   // EP vendor name
  const std::string ep_version_{"0.1.0"};  // EP version

  // Qualcomm vendor ID. Refer to the ACPI ID registry (search Qualcomm): https://uefi.org/ACPI_ID_List
  const uint32_t vendor_id_{'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)};

  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
  MemoryInfoUniquePtr host_accessible_memory_info_;

  // Non-null when libcdsprpc is loadable; probed once in the factory ctor.
  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library_;

  QnnEp* qnn_ep_ = nullptr;
  std::vector<OrtEpDevice*> ep_devices_;

  using HardwareDeviceUniquePtr = std::unique_ptr<OrtHardwareDevice, FuncDeleter<OrtHardwareDevice>>;
  // Actual NPU hardware that ORT Core did not enumerate (e.g. Makena without DXCore).
  HardwareDeviceUniquePtr undetected_npu_hw_device_;

  // Tracks the allocator type for ReleaseAllocatorImpl dispatch.
  qnn::QnnAllocatorType qnn_allocator_type_ = qnn::QnnAllocatorType::NONE;

  // Custom op domains registered via ORT_QNN_CUSTOM_OP_DOMAINS.
  // Both vectors must outlive any session that uses this factory (factory is a per-library singleton).
  // domain.Add(op*) does NOT transfer ownership; op objects must be kept alive here.
  std::vector<Ort::CustomOpDomain> custom_op_domains_;
  std::vector<std::unique_ptr<qnn::QnnUdoPlaceholderOp>> custom_op_objects_;
};

}  // namespace onnxruntime
