// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/qnn_provider_factory.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>

#include "onnxruntime_c_api.h"
#include "onnxruntime_ep_device_ep_metadata_keys.h"
#include "QnnCommon.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/ort_api_version_parser.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/soc_utils.h"

// We allow `backend_type` (e.g., `htp`) or `backend_path` in relative path (e.g., `QnnHtp.dll`) for configurations,
// and QnnBackendManager will later find the appropriate library and load it relative to the OnnxRuntime library.
// But if QNN-EP is distributed separately from the OnnxRuntime library (e.g. EP ABI or WinML), the backend library may
// well not be relative to the OnnxRuntime but to the EP library itself instead.
// If the EP library is co-located with the OnnxRuntime library, then this is consistent with the existing behavior,
// but an EP library that is shipped 'out-of-band' will use a backend relative to itself.
static const std::unordered_map<OrtHardwareDeviceType, std::string> kDefaultBackends = {
#if defined(_WIN32)
    {OrtHardwareDeviceType_NPU, "QnnHtp.dll"},
    {OrtHardwareDeviceType_GPU, "QnnGpu.dll"},
#else
    {OrtHardwareDeviceType_NPU, "libQnnHtp.so"},
    {OrtHardwareDeviceType_GPU, "libQnnGpu.so"},
#endif
};

static const std::unordered_map<OrtHardwareDeviceType, std::string> kSupportedBackendTypes = {
    {OrtHardwareDeviceType_CPU, "cpu"},
    {OrtHardwareDeviceType_NPU, "htp"},
    {OrtHardwareDeviceType_GPU, "gpu"},
};

// x86 advertises CPU always (HTP emulator); arm64 is opt-in via ORT_QNN_ENABLE_CPU_BACKEND (tests).
static bool QnnCpuBackendEnabled() {
#if !defined(__aarch64__) && !defined(_M_ARM64)
  return true;  // x86 host: CPU backend is the HTP emulator.
#else
  static const bool enabled = []() {
#if defined(_WIN32)
    // std::getenv is a fatal C4996 under -WX on MSVC.
    char* value = nullptr;
    size_t value_size = 0;
    const bool found = _dupenv_s(&value, &value_size, "ORT_QNN_ENABLE_CPU_BACKEND") == 0 && value != nullptr;
    const bool is_enabled = found && value[0] != '\0' && value[0] != '0';
    free(value);
    return is_enabled;
#else
    const char* value = std::getenv("ORT_QNN_ENABLE_CPU_BACKEND");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
#endif
  }();
  return enabled;
#endif
}

namespace onnxruntime {

// OrtEpApi infrastructure to be able to use the QNN EP as an OrtEpFactory for auto EP selection.
QnnEpFactory::QnnEpFactory(const char* ep_name,
                           ApiPtrs ort_api_in)
    : OrtEpFactory{}, ApiPtrs(ort_api_in), ep_name_{ep_name} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;
  GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;

  // HOST_ACCESSIBLE memory.
  OrtMemoryInfo* mem_info = nullptr;
  auto* status = ort_api.CreateMemoryInfo_V2("QnnHtpShared",
                                             OrtMemoryInfoDeviceType_CPU,
                                             /*vendor*/ 0x5143,
                                             /*device_id*/ 0,
                                             OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                             /*alignment*/ 0,
                                             OrtAllocatorType::OrtDeviceAllocator,
                                             &mem_info);
  if (status != nullptr) {
    ort_api.ReleaseMemoryInfo(mem_info);
  }
  host_accessible_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);
}

// Returns the name for the EP. Each unique factory configuration must have a unique name.
// Ex: a factory that supports NPU should have a different than a factory that supports GPU.
const char* ORT_API_CALL QnnEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

const char* ORT_API_CALL QnnEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

uint32_t ORT_API_CALL QnnEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

const char* ORT_API_CALL QnnEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

// Creates and returns OrtEpDevice instances for all OrtHardwareDevices that this factory supports.
OrtStatus* ORT_API_CALL QnnEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                              const OrtHardwareDevice* const* devices,
                                                              size_t num_devices,
                                                              OrtEpDevice** ep_devices,
                                                              size_t max_ep_devices,
                                                              size_t* p_num_ep_devices) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);

  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  auto create_ep_device = [&factory, &ep_devices, &num_ep_devices](const OrtHardwareDevice* device) {
    OrtEpDevice* ep_device = nullptr;
    OrtStatus* status = factory->ep_api.CreateEpDevice(factory, device, nullptr, nullptr, &ep_device);
    ep_devices[num_ep_devices++] = ep_device;
    factory->ep_devices_.push_back(ep_device);

    return status;
  };

  auto create_hw_device = [&factory](const OrtHardwareDeviceType device_type,
                                     OrtHardwareDevice*& device,
                                     const bool is_virtual = true) {
    OrtKeyValuePairs* hw_metadata = nullptr;
    if (is_virtual) {
      factory->ort_api.CreateKeyValuePairs(&hw_metadata);
      factory->ort_api.AddKeyValuePair(hw_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual, "1");
    }

    OrtStatus* status = factory->ep_api.CreateHardwareDevice(device_type,
                                                             factory->vendor_id_,
                                                             0,
                                                             factory->vendor_.c_str(),
                                                             hw_metadata,
                                                             &device);

    if (hw_metadata) {
      factory->ort_api.ReleaseKeyValuePairs(hw_metadata);
    }

    return status;
  };

  bool has_npu_hw_device = false;

  for (size_t idx = 0; idx < num_devices && num_ep_devices < max_ep_devices; ++idx) {
    const OrtHardwareDevice* device = devices[idx];
    auto device_type = factory->ort_api.HardwareDevice_Type(device);
    auto vendor_id = factory->ort_api.HardwareDevice_VendorId(device);

    if ((kDefaultBackends.find(device_type) != kDefaultBackends.end() && vendor_id == factory->vendor_id_) ||
        (device_type == OrtHardwareDeviceType_CPU && QnnCpuBackendEnabled())) {
      RETURN_IF_NOT_NULL(create_ep_device(device));

      if (device_type == OrtHardwareDeviceType_NPU) {
        has_npu_hw_device = true;
      }
    }
  }

  if (!has_npu_hw_device && num_ep_devices < max_ep_devices) {
    if (qnn::soc::GetSocId() != 0) {
      // If ORT Core does not detect NPU hardware but we recognize the device as WoS (through qnn::soc::GetSocId),
      // exploit virtual hardware device to create an NPU hardware device for user to select from.
      // Such case happens for older WoS devices (e.g., Makena) that ORT Core's device discovery logic could not detect
      // NPU through DXCore.
      OrtHardwareDevice* undetected_npu_hw_device = nullptr;
      RETURN_IF_NOT_NULL(create_hw_device(OrtHardwareDeviceType_NPU, undetected_npu_hw_device, false));
      factory->undetected_npu_hw_device_ = HardwareDeviceUniquePtr(
          undetected_npu_hw_device,
          FuncDeleter<OrtHardwareDevice>{factory->ep_api.ReleaseHardwareDevice});

      RETURN_IF_NOT_NULL(create_ep_device(factory->undetected_npu_hw_device_.get()));
    } else {
      // Enable originally expected usage of virtual hardware device for cross-platform compilation if necessary.
    }
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                   _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                                   _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                   _In_ size_t num_devices,
                                                   _In_ const OrtSessionOptions* session_options,
                                                   _In_ const OrtLogger* logger,
                                                   _Out_ OrtEp** ep) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);
  *ep = nullptr;

  // Check if logger is nullptr and get default logger if available
  if (logger == nullptr) {
    if (!OrtLoggingManager::HasDefaultLogger()) {
      return factory->ort_api.CreateStatus(ORT_FAIL, "Logger is nullptr and OrtLoggingManager does not have a default logger.");
    }
    logger = OrtLoggingManager::GetDefaultLoggerPtr();
  }

  // Create the execution provider
  RETURN_IF_NOT_NULL(factory->ort_api.Logger_LogMessage(logger,
                                                        OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                        "Creating QNN EP", ORT_FILE, __LINE__, __FUNCTION__));

  const auto provider_prefix = GetProviderOptionPrefix(factory->ep_name_);

  // Setting allocator info is delayed from GetSupportedDevices to here as QNN-EP relies on provider options to
  // determine whether to use HTP shared memory but they are not available until now. This workaround works since
  // PluginExecutionProvider collects the allocator infos after creating the EP (refer to
  // ep_plugin_provider_interfaces.cc for the detail flow).
  std::string enable_htp_shared_memory_allocator_str;
  GetSessionConfigEntryOrDefault(factory->ort_api,
                                 *session_options,
                                 provider_prefix + "enable_htp_shared_memory_allocator",
                                 "0",
                                 enable_htp_shared_memory_allocator_str);
  if (enable_htp_shared_memory_allocator_str == "1") {
    for (OrtEpDevice* ep_device : factory->ep_devices_) {
      RETURN_IF_NOT_NULL(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->host_accessible_memory_info_.get()));
    }
  }

  const auto backend_type_key = provider_prefix + "backend_type";
  const auto backend_path_key = provider_prefix + "backend_path";
  int has_backend_type = 0;
  int has_backend_path = 0;
  RETURN_IF_NOT_NULL(factory->ort_api.HasSessionConfigEntry(session_options, backend_type_key.c_str(), &has_backend_type));
  RETURN_IF_NOT_NULL(factory->ort_api.HasSessionConfigEntry(session_options, backend_path_key.c_str(), &has_backend_path));

  using SessionOptionsUniquePtr = std::unique_ptr<OrtSessionOptions, std::function<void(OrtSessionOptions*)>>;
  SessionOptionsUniquePtr autoep_session_options;

  if (!has_backend_type && !has_backend_path) {
    // If neither "backend_path" nor "backend_type" has been given in the provider options, then determine the backend based
    // on the provided devices. As QNN EP does not support partitioning across backends, if multiple devices are provided,
    // default to HTP (if present) or else to the GPU.
    const OrtHardwareDevice* device_to_use = nullptr;
    if (num_devices == 0) {
      return factory->ort_api.CreateStatus(ORT_FAIL, "No devices were provided to QNN EP.");
    } else if (num_devices == 1) {
      device_to_use = devices[0];
    } else {
      const auto is_npu = [&factory](const OrtHardwareDevice* device) {
        return factory->ort_api.HardwareDevice_Type(device) == OrtHardwareDeviceType_NPU;
      };
      const auto is_gpu = [&factory](const OrtHardwareDevice* device) {
        return factory->ort_api.HardwareDevice_Type(device) == OrtHardwareDeviceType_GPU;
      };

      auto device_it = std::find_if(devices, devices + num_devices, is_npu);
      if (device_it != devices + num_devices) {
        RETURN_IF_NOT_NULL(factory->ort_api.Logger_LogMessage(
            logger,
            OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            "QNN EP only supports one device. Only the NPU device will be used.",
            ORT_FILE, __LINE__, __FUNCTION__));
        device_to_use = *device_it;
      } else {
        device_it = std::find_if(devices, devices + num_devices, is_gpu);
        if (device_it != devices + num_devices) {
          RETURN_IF_NOT_NULL(factory->ort_api.Logger_LogMessage(
              logger,
              OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
              "QNN EP only supports one device. An NPU device was not provided, so only the GPU device will be used.",
              ORT_FILE, __LINE__, __FUNCTION__));
          device_to_use = *device_it;
        } else {
          return factory->ort_api.CreateStatus(ORT_FAIL,
                                               "Multiple devices were provided to QNN EP, but neither an NPU nor a GPU was included.");
        }
      }
    }
    assert(device_to_use != nullptr);

    auto default_backends_it = kDefaultBackends.find(factory->ort_api.HardwareDevice_Type(device_to_use));
    if (default_backends_it == kDefaultBackends.end()) {
      return factory->ort_api.CreateStatus(ORT_FAIL, "Could not determine default backend path for device");
    }

    // Identify the path of the current dynamic library, and expect that the backend library is in the same directory.
    auto current_path = onnxruntime::GetDynamicLibraryLocationByAddress(
        reinterpret_cast<const void*>(&kDefaultBackends));

    std::filesystem::path parent_path;
    if (!current_path.empty()) {
      parent_path = std::filesystem::path{std::move(current_path)}.parent_path();
    }

    // Add the "backend_path" key based on the autoep device selection above. As the
    // `session_options` param is const, we must first clone it before adding the option.
    OrtSessionOptions* cloned_session_options = nullptr;
    RETURN_IF_NOT_NULL(factory->ort_api.CloneSessionOptions(session_options, &cloned_session_options));
    autoep_session_options = SessionOptionsUniquePtr(cloned_session_options, factory->ort_api.ReleaseSessionOptions);
    RETURN_IF_NOT_NULL(factory->ort_api.AddSessionConfigEntry(autoep_session_options.get(),
                                                              (provider_prefix + "backend_path").c_str(),
                                                              (parent_path / default_backends_it->second).string().c_str()));

    // Use the amended session options with the autoep backend path during creation of QnnEp
    session_options = autoep_session_options.get();
  }

  std::unique_ptr<QnnEp> qnn_ep;
  try {
    qnn_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, *session_options, logger);
  } catch (const std::runtime_error& e) {
    return factory->ort_api.CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return factory->ort_api.CreateStatus(ORT_FAIL, "Unknown exception occurred while creating QNN EP.");
  }

  factory->qnn_ep_ = qnn_ep.get();
  *ep = qnn_ep.release();

  return nullptr;
}

void ORT_API_CALL QnnEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  if (ep == nullptr) {
    return;
  }

  QnnEp* dummy_ep = static_cast<QnnEp*>(ep);
  delete dummy_ep;
}

void ORT_API_CALL QnnEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/, OrtAllocator* allocator) noexcept {
  delete static_cast<qnn::HtpSharedMemoryAllocator*>(allocator);
}

OrtStatus* ORT_API_CALL QnnEpFactory::CreateDataTransferImpl(OrtEpFactory* /* this_ptr */,
                                                             OrtDataTransferImpl** data_transfer) noexcept {
  *data_transfer = nullptr;

  return nullptr;
}

bool ORT_API_CALL QnnEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

OrtStatus* ORT_API_CALL QnnEpFactory::ValidateCompiledModelCompatibilityInfoImpl(
    _In_ OrtEpFactory* this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
    _In_ size_t num_devices,
    _In_ const char* compatibility_info,
    _Out_ OrtCompiledModelCompatibility* model_compatibility) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);

  if (factory->qnn_ep_ != nullptr) {
    return factory->qnn_ep_->ValidateCompiledModelCompatibilityInfo(devices,
                                                                    num_devices,
                                                                    compatibility_info,
                                                                    model_compatibility);
  }

  // EP has not been created yet. Create a temporary QNN EP for validation.
  const auto provider_prefix = GetProviderOptionPrefix(factory->ep_name_);

  // Determine backend type from the provided devices (Only supports NPU currently).
  std::string backend_type;
  for (size_t i = 0; i < num_devices; ++i) {
    auto device_type = factory->ort_api.HardwareDevice_Type(devices[i]);
    if (device_type == OrtHardwareDeviceType_NPU) {
      backend_type = "htp";
      break;
    }
  }
  if (backend_type.empty()) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return factory->ort_api.CreateStatus(ORT_EP_FAIL,
                                         "Currently QnnEpFactory::ValidateCompiledModelCompatibilityInfoImpl only supports OrtHardwareDeviceType_NPU, but "
                                         "no OrtHardwareDeviceType_NPU is found in the `devices` argument.");
  }

  using SessionOptionsUniquePtr = std::unique_ptr<OrtSessionOptions, std::function<void(OrtSessionOptions*)>>;
  OrtSessionOptions* temp_session_options = nullptr;
  if (OrtStatus* _status = factory->ort_api.CreateSessionOptions(&temp_session_options)) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return _status;
  }
  SessionOptionsUniquePtr session_options(temp_session_options, factory->ort_api.ReleaseSessionOptions);

  if (OrtStatus* _status = factory->ort_api.AddSessionConfigEntry(session_options.get(),
                                                                  (provider_prefix + "backend_type").c_str(),
                                                                  backend_type.c_str())) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return _status;
  }

  if (!OrtLoggingManager::HasDefaultLogger()) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return factory->ort_api.CreateStatus(ORT_EP_FAIL, "Default logger is not available for model compatibility check.");
  }
  const OrtLogger* logger = OrtLoggingManager::GetDefaultLoggerPtr();

  std::unique_ptr<QnnEp> temp_qnn_ep;
  try {
    temp_qnn_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, *session_options.get(), logger);
  } catch (const std::exception& e) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return factory->ort_api.CreateStatus(ORT_EP_FAIL, e.what());
  } catch (...) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return factory->ort_api.CreateStatus(ORT_EP_FAIL, "Unknown exception occurred while creating temporary QNN EP for compatibility check.");
  }

  return temp_qnn_ep->ValidateCompiledModelCompatibilityInfo(devices,
                                                             num_devices,
                                                             compatibility_info,
                                                             model_compatibility);
}

OrtStatus* ORT_API_CALL QnnEpFactory::GetHardwareDeviceIncompatibilityDetailsImpl(
    _In_ OrtEpFactory* this_ptr,
    _In_ const OrtHardwareDevice* hw,
    _Inout_ OrtDeviceEpIncompatibilityDetails* details) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);
  const auto provider_prefix = GetProviderOptionPrefix(factory->ep_name_);

  // Check if the device type is supported by QNN EP
  auto device_type = factory->ort_api.HardwareDevice_Type(hw);
  auto vendor_id = factory->ort_api.HardwareDevice_VendorId(hw);

  // QNN EP supports NPU/GPU devices with Qualcomm vendor ID, plus CPU only when the QNN CPU backend is enabled.
  const bool is_cpu = device_type == OrtHardwareDeviceType_CPU;
  auto supported_backend_types_it = kSupportedBackendTypes.find(device_type);
  const bool type_unsupported = supported_backend_types_it == kSupportedBackendTypes.end() ||
                                (is_cpu && !QnnCpuBackendEnabled());
  if (type_unsupported || (vendor_id != factory->vendor_id_ && !is_cpu)) {
    OrtDeviceEpIncompatibilityReason reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;
    return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
        details,
        reasons,
        QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED,
        "QNN EP only supports Qualcomm NPU and GPU devices");
  }

  // Create a temporary QNN EP and to check device compatibility
  // Create minimal session options for backend setup
  using SessionOptionsUniquePtr = std::unique_ptr<OrtSessionOptions, std::function<void(OrtSessionOptions*)>>;
  OrtSessionOptions* temp_session_options = nullptr;
  RETURN_IF_NOT_NULL(factory->ort_api.CreateSessionOptions(&temp_session_options));
  SessionOptionsUniquePtr session_options(temp_session_options, factory->ort_api.ReleaseSessionOptions);

  // Determine backend type based on device type
  std::string backend_type = supported_backend_types_it->second;
  RETURN_IF_NOT_NULL(factory->ort_api.AddSessionConfigEntry(session_options.get(), (provider_prefix + "backend_type").c_str(), backend_type.c_str()));

  // Use default logger for the compatibility check
  if (!OrtLoggingManager::HasDefaultLogger()) {
    return factory->ort_api.CreateStatus(ORT_FAIL, "Default logger is not available for device compatibility check.");
  }
  const OrtLogger* logger = OrtLoggingManager::GetDefaultLoggerPtr();

  // Try to create a temporary QNN EP to test backend setup
  std::unique_ptr<QnnEp> temp_qnn_ep;
  try {
    temp_qnn_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, *session_options.get(), logger);
    RETURN_IF_NOT_NULL(temp_qnn_ep->GetHardwareDeviceIncompatibilityDetails(hw, details));
  } catch (...) {
    OrtDeviceEpIncompatibilityReason reasons = OrtDeviceEpIncompatibility_UNKNOWN;
    return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
        details,
        reasons,
        QNN_COMMON_ERROR_UNDEFINED,
        "Unknown exception occurred while creating QNN EP for compatibility check");
  }
  return nullptr;
}

}  // namespace onnxruntime

extern "C" {
//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* registration_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories) {
  if (ort_api_base == nullptr) {
    return nullptr;
  }

  // kMinOrtApiVersion must be at least the ORT API version that introduced
  // the newest ORT API method this EP calls. Below this floor, GetApi()
  // returns a function table missing members the EP would dereference.
  constexpr uint32_t kMinOrtApiVersion = 24;
  static_assert(kMinOrtApiVersion <= ORT_API_VERSION,
                "kMinOrtApiVersion must not exceed ORT_API_VERSION");

  const char* version_str = ort_api_base->GetVersionString();
  const uint32_t runtime_api_version = onnxruntime::qnn::detail::ParseRuntimeOrtApiVersion(version_str);

  if (runtime_api_version == 0) {
    const OrtApi* fallback_api = ort_api_base->GetApi(1);
    if (fallback_api == nullptr) {
      return nullptr;
    }
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "QNN EP could not parse host ORT version string \"%s\" "
                  "(expected \"1.X.Y\" with major == 1).",
                  version_str != nullptr ? version_str : "(null)");
    return fallback_api->CreateStatus(ORT_FAIL, msg);
  }

  if (runtime_api_version < kMinOrtApiVersion) {
    const OrtApi* fallback_api = ort_api_base->GetApi(runtime_api_version);
    if (fallback_api == nullptr) {
      return nullptr;
    }
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "QNN EP requires ORT >= 1.%u (API %u). "
                  "Host ORT is %s (API %u).",
                  kMinOrtApiVersion, kMinOrtApiVersion,
                  version_str, runtime_api_version);
    return fallback_api->CreateStatus(ORT_FAIL, msg);
  }

  const uint32_t requested_api_version =
      std::min(runtime_api_version, static_cast<uint32_t>(ORT_API_VERSION));
  const OrtApi* ort_api = ort_api_base->GetApi(requested_api_version);
  if (ort_api == nullptr) {
    return nullptr;
  }

  // Manual init for the C++ API
  Ort::InitApi(ort_api);

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  if (factories == nullptr || num_factories == nullptr) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Invalid arguments: factories and num_factories cannot be null.");
  }

  const OrtEpApi* ep_api = ort_api->GetEpApi();
  if (ep_api == nullptr) {
    return ort_api->CreateStatus(ORT_FAIL, "Failed to get EP API.");
  }

  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
  if (model_editor_api == nullptr) {
    return ort_api->CreateStatus(ORT_FAIL, "Failed to get Model Editor API.");
  }

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<onnxruntime::QnnEpFactory> factory;
  try {
    factory = std::make_unique<onnxruntime::QnnEpFactory>(registration_name,
                                                          onnxruntime::ApiPtrs{*ort_api,
                                                                               *ep_api,
                                                                               *model_editor_api});
  } catch (const std::exception& e) {
    return ort_api->CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return ort_api->CreateStatus(ORT_FAIL, "Unknown exception occurred while creating QNN EP factory.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  // Set default logger for later use.
  onnxruntime::OrtLoggingManager::SetDefaultLogger(default_logger);

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  if (factory == nullptr) {
    return nullptr;
  }

  delete static_cast<onnxruntime::QnnEpFactory*>(factory);
  return nullptr;
}
}
