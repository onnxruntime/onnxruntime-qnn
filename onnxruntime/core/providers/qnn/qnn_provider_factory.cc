// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/qnn_provider_factory.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>

#include "QnnCommon.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_ep_device_ep_metadata_keys.h"
#include "onnxruntime_env_config_keys.h"
#include "gsl/gsl"

#include "core/providers/qnn/custom_op/qnn_custom_op_domain_registry.h"
#include "core/providers/qnn/genie/genie_backend_manager.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/ort_api_version_parser.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/qnn_data_transfer.h"
#include "core/providers/qnn/soc_utility/soc_utils.h"
#include "qnn_ep_min_ort_api_version.h"

#ifdef _WIN32
#include "core/providers/qnn/qnn_external_resource_importer.h"
#endif

namespace {

// Chromium transports the DXCore device ID as an unprefixed hexadecimal uint32.
// Also accept the conventional 0x prefix, while rejecting partial and overflow parses.
bool ParseHardwareDeviceId(std::string_view text, uint32_t& device_id) {
  if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
    text.remove_prefix(2);
  }
  const auto result = std::from_chars(text.data(), text.data() + text.size(), device_id, 16);
  return !text.empty() && result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

}  // namespace

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
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;
  GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;
  GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
  GetCustomOpDomains = GetCustomOpDomainsImpl;

#ifdef _WIN32
  CreateExternalResourceImporterForDevice = CreateExternalResourceImporterForDeviceImpl;
#else
  CreateExternalResourceImporterForDevice = nullptr;
#endif

  // Register QNN-only placeholder ops for the qti_aisw block ops so models that use them pass ORT
  // model validation (Graph::Resolve) when this EP is appended. Their fixed OPTIONAL schema
  // preserves empty ONNX RNN input slots. QNN fuses/compiles supported nodes; unsupported block
  // ops have no CPU fallback and fail during session initialization.
  {
    Ort::CustomOpDomain qti_aisw_domain{kQtiAiswDomain};
    for (const char* op_type : kQtiAiswBlockOpTypes) {
      qti_aisw_op_objects_.push_back(
          std::make_unique<qnn::QtiAiswPlaceholderOp>(op_type, ep_name_));
      qti_aisw_domain.Add(qti_aisw_op_objects_.back().get());
    }
    custom_op_domains_.push_back(std::move(qti_aisw_domain));
  }

  // Build additional custom-op domains from ORT_QNN_CUSTOM_OP_DOMAINS env var (env-var UDO ops).
  // GetCustomOpDomains is called at SessionOptionsAppendExecutionProvider_V2 time (before CreateEp),
  // so we parse once here at factory construction and cache the result for all sessions.
  // SetDefaultLogger is called before factory construction in CreateEpFactories, so
  // OrtLoggingManager::GetDefaultLoggerPtr() is already set and valid here.
  BuildCustomOpDomainsFromEnv(OrtLoggingManager::GetDefaultLogger(), ep_name_, custom_op_domains_, custom_op_objects_);

  // HOST_ACCESSIBLE memory for HTP and GPU backends.
  OrtMemoryInfo* mem_info = nullptr;
  auto* status = ort_api.CreateMemoryInfo_V2("QnnHtpShared",
                                             OrtMemoryInfoDeviceType_CPU,
                                             qnn::kQualcommPciVendorId,
                                             /*device_id*/ 0,
                                             OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                             /*alignment*/ 0,
                                             OrtAllocatorType::OrtDeviceAllocator,
                                             &mem_info);
  if (status != nullptr) {
    ort_api.ReleaseMemoryInfo(mem_info);
    ort_api.ReleaseStatus(status);
  }
  host_accessible_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);
}

void QnnEpFactory::RegisterQnnBackendManager(const std::shared_ptr<qnn::QnnBackendManager>& manager) {
  if (manager) {
    RegisterBackendLibraryLifetime(manager->GetBackendLibraryLifetime());
  }
}

void QnnEpFactory::RegisterGenieBackendManager(const std::shared_ptr<qnn::GenieBackendManager>& manager) {
  if (manager) {
    RegisterBackendLibraryLifetime(manager->GetBackendLibraryLifetime());
  }
}

void QnnEpFactory::RegisterBackendLibraryLifetime(std::shared_ptr<void> lifetime) {
  std::lock_guard<std::mutex> lock(backend_libraries_mutex_);
  if (std::find(backend_library_lifetimes_.begin(), backend_library_lifetimes_.end(), lifetime) ==
      backend_library_lifetimes_.end()) {
    backend_library_lifetimes_.push_back(std::move(lifetime));
  }
}

void QnnEpFactory::UnloadBackendLibraries() {
  std::vector<std::shared_ptr<void>> lifetimes;
  {
    std::lock_guard<std::mutex> lock(backend_libraries_mutex_);
    lifetimes.swap(backend_library_lifetimes_);
  }
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
  const size_t initial_ep_count = factory->ep_devices_.size();
  const size_t initial_hw_count = factory->synthesized_hw_devices_.size();
  bool discovery_complete = false;
  auto rollback = gsl::finally([&]() {
    if (!discovery_complete) {
      for (size_t i = 0; i < num_ep_devices; ++i) {
        factory->ep_api.ReleaseEpDevice(ep_devices[i]);
        ep_devices[i] = nullptr;
      }
      factory->ep_devices_.resize(initial_ep_count);
      factory->synthesized_hw_devices_.resize(initial_hw_count);
      num_ep_devices = 0;
    }
  });

  // Check for keys related to Chromium's WebNN implementation. Graph compilation runs in a sandboxed process where hardware
  // devices are not visible to the EP. Instead, we must create virtual devices based on the signaled architecture.
  OrtKeyValuePairs* env_config = nullptr;
  RETURN_IF_NOT_NULL(factory->ep_api.GetEnvConfigEntries(&env_config));
  std::unique_ptr<OrtKeyValuePairs, decltype(factory->ort_api.ReleaseKeyValuePairs)> entries(
      env_config, factory->ort_api.ReleaseKeyValuePairs);
  const char* allow = entries ? factory->ort_api.GetKeyValue(entries.get(), kOrtEnvAllowVirtualDevices) : nullptr;
  const bool allow_virtual_devices = allow != nullptr && std::string_view(allow) == "1";

  const std::string hardware_device_id_key = "ep_factory." + factory->ep_name_ + ".qnn_hardware_device_id";
  std::optional<uint32_t> configured_hardware_device_id;
  if (entries) {
    if (const char* value = factory->ort_api.GetKeyValue(entries.get(), hardware_device_id_key.c_str())) {
      uint32_t parsed = 0;
      if (!ParseHardwareDeviceId(value, parsed)) {
        return factory->ort_api.CreateStatus(
            ORT_INVALID_ARGUMENT,
            (hardware_device_id_key + " must be a hexadecimal uint32 hardware device ID.").c_str());
      }
      configured_hardware_device_id = parsed;
    }
  }

  const std::string target_key = "ep_factory." + factory->ep_name_ + ".qnn_target_architecture";
  const char* target_value = entries ? factory->ort_api.GetKeyValue(entries.get(), target_key.c_str()) : nullptr;
  const std::optional<uint32_t> configured_soc_model =
      target_value ? qnn::soc::ParseSocModelTarget(target_value) : std::nullopt;
  if (target_value) {
    if (!configured_soc_model) {
      return factory->ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT,
          (target_key + " must be a nonzero decimal QNN SoC model, e.g. 60.").c_str());
    }
    if (!configured_hardware_device_id.has_value()) {
      return factory->ort_api.CreateStatus(
          ORT_INVALID_ARGUMENT,
          "QNN target architecture requires qnn_hardware_device_id in the same environment.");
    }
  }

  auto create_ep_device = [&factory, &ep_devices, &num_ep_devices](
                              const OrtHardwareDevice* device,
                              std::optional<uint32_t> target_soc_model = std::nullopt) -> OrtStatus* {
    const auto device_type = factory->ort_api.HardwareDevice_Type(device);
    const OrtKeyValuePairs* hw_device_metadata = factory->ort_api.HardwareDevice_Metadata(device);
    const char* is_virtual_value = factory->ort_api.GetKeyValue(hw_device_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual);
    const bool is_virtual = is_virtual_value != nullptr && std::strcmp(is_virtual_value, "1") == 0;

    OrtKeyValuePairs* metadata = nullptr;
    factory->ort_api.CreateKeyValuePairs(&metadata);
    std::unique_ptr<OrtKeyValuePairs, decltype(factory->ort_api.ReleaseKeyValuePairs)> owned_metadata(
        metadata, factory->ort_api.ReleaseKeyValuePairs);
    if ((device_type == OrtHardwareDeviceType_NPU || device_type == OrtHardwareDeviceType_GPU) &&
        target_soc_model.has_value()) {
      factory->ort_api.AddKeyValuePair(
          metadata, "target_architecture", std::to_string(*target_soc_model).c_str());
    }

    OrtKeyValuePairs* ep_options = nullptr;
    factory->ort_api.CreateKeyValuePairs(&ep_options);
    std::unique_ptr<OrtKeyValuePairs, decltype(factory->ort_api.ReleaseKeyValuePairs)> owned_options(
        ep_options, factory->ort_api.ReleaseKeyValuePairs);
    if (device_type == OrtHardwareDeviceType_GPU && is_virtual &&
        target_soc_model.has_value() && *target_soc_model != QNN_SOC_MODEL_UNKNOWN) {
      factory->ort_api.AddKeyValuePair(ep_options, "soc_model", std::to_string(*target_soc_model).c_str());
    }

    OrtEpDevice* ep_device = nullptr;
    OrtStatus* status = factory->ep_api.CreateEpDevice(factory, device, metadata, ep_options, &ep_device);
    if (status) return status;

    // Advertise HOST_ACCESSIBLE memory for NPU devices so OrtEnv can create a
    // QnnHtpShared allocator before a QNN session exists. Loading RPCMEM is deferred
    // until that allocator (or an opted-in session) actually needs it.
    if (device_type == OrtHardwareDeviceType_NPU &&
        factory->host_accessible_memory_info_ != nullptr) {
      status = factory->ep_api.EpDevice_AddAllocatorInfo(
          ep_device, factory->host_accessible_memory_info_.get());
      if (status != nullptr) {
        factory->ep_api.ReleaseEpDevice(ep_device);
        return status;
      }
    }

    ep_devices[num_ep_devices++] = ep_device;
    factory->ep_devices_.push_back(ep_device);

    return nullptr;
  };

  auto create_hw_device = [&factory](const OrtHardwareDeviceType device_type,
                                     OrtHardwareDevice*& device,
                                     const bool is_virtual = true,
                                     const uint32_t device_id = 0) {
    OrtKeyValuePairs* hw_metadata = nullptr;
    factory->ort_api.CreateKeyValuePairs(&hw_metadata);
    if (device_type == OrtHardwareDeviceType_NPU) {
      factory->ort_api.AddKeyValuePair(hw_metadata, "Description",
                                       is_virtual ? "Qualcomm(R) NPU - Virtual" : "Qualcomm NPU");
    } else if (device_type == OrtHardwareDeviceType_GPU) {
      factory->ort_api.AddKeyValuePair(hw_metadata, "Description",
                                       is_virtual ? "Qualcomm(R) GPU - Virtual" : "Qualcomm GPU");
    }
    if (is_virtual) {
      factory->ort_api.AddKeyValuePair(hw_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual, "1");
    }

    OrtStatus* status = factory->ep_api.CreateHardwareDevice(device_type,
                                                             factory->vendor_id_,
                                                             device_id,
                                                             factory->vendor_.c_str(),
                                                             hw_metadata,
                                                             &device);

    if (hw_metadata) {
      factory->ort_api.ReleaseKeyValuePairs(hw_metadata);
    }

    return status;
  };

  std::optional<uint32_t> discovered_soc_model;
  bool has_npu_hw_device = false;
  bool has_gpu_hw_device = false;
  if (!configured_soc_model.has_value()) {
    const uint32_t soc_model = qnn::soc::MapSocModelFromSocId(qnn::soc::GetSocId());
    if (soc_model != 0) discovered_soc_model = soc_model;
  }

  for (size_t idx = 0; idx < num_devices && num_ep_devices < max_ep_devices; ++idx) {
    const OrtHardwareDevice* device = devices[idx];
    auto device_type = factory->ort_api.HardwareDevice_Type(device);
    auto vendor_id = factory->ort_api.HardwareDevice_VendorId(device);

    const auto& default_backends = QnnEp::GetDefaultBackends();
    if ((default_backends.find(device_type) != default_backends.end() && vendor_id == factory->vendor_id_) ||
        (device_type == OrtHardwareDeviceType_CPU && QnnCpuBackendEnabled())) {
      RETURN_IF_NOT_NULL(create_ep_device(device, discovered_soc_model));

      if (device_type == OrtHardwareDeviceType_NPU) {
        has_npu_hw_device = true;
      } else if (device_type == OrtHardwareDeviceType_GPU) {
        has_gpu_hw_device = true;
      }
    }
  }

  if (allow_virtual_devices && num_ep_devices < max_ep_devices && !has_npu_hw_device) {
    // On older platforms, DXCore does not discover the physical NPU device. Check for that case and
    // and synthesize a non-virtual NPU device if applicable.
    const bool has_real_undetected_npu = qnn::soc::GetSocId() != 0 || qnn::soc::HasFastRpcCdspDevice();
    const bool is_virtual = !has_real_undetected_npu;

    OrtHardwareDevice* synthesized_npu_hw_device = nullptr;
    RETURN_IF_NOT_NULL(create_hw_device(OrtHardwareDeviceType_NPU,
                                        synthesized_npu_hw_device,
                                        is_virtual,
                                        configured_hardware_device_id.value_or(0)));
    factory->synthesized_hw_devices_.push_back(HardwareDeviceUniquePtr(
        synthesized_npu_hw_device,
        FuncDeleter<OrtHardwareDevice>{factory->ep_api.ReleaseHardwareDevice}));

    RETURN_IF_NOT_NULL(create_ep_device(factory->synthesized_hw_devices_.back().get(),
                                        configured_soc_model ? configured_soc_model : discovered_soc_model));
  }

  if (allow_virtual_devices && num_ep_devices < max_ep_devices && !has_gpu_hw_device) {
    OrtHardwareDevice* virtual_gpu_hw_device = nullptr;
    RETURN_IF_NOT_NULL(create_hw_device(OrtHardwareDeviceType_GPU,
                                        virtual_gpu_hw_device,
                                        true,
                                        configured_hardware_device_id.value_or(0)));
    factory->synthesized_hw_devices_.push_back(HardwareDeviceUniquePtr(
        virtual_gpu_hw_device,
        FuncDeleter<OrtHardwareDevice>{factory->ep_api.ReleaseHardwareDevice}));

    const auto device_soc_model = configured_soc_model.value_or(
        qnn::soc::MapSocModelFromSocId(qnn::soc::GetSocId()));
    RETURN_IF_NOT_NULL(create_ep_device(factory->synthesized_hw_devices_.back().get(), device_soc_model));
  }

  discovery_complete = true;
  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                   _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                                   _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata,
                                                   _In_ size_t num_devices,
                                                   _In_ const OrtSessionOptions* session_options,
                                                   _In_ const OrtLogger* logger,
                                                   _Out_ OrtEp** ep) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);
  *ep = nullptr;
  (void)ep_metadata;  // Used only by the Windows ARM64 AOT path below.

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

#if defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
  const auto provider_prefix = GetProviderOptionPrefix(factory->ep_name_);

  using SessionOptionsUniquePtr = std::unique_ptr<OrtSessionOptions, std::function<void(OrtSessionOptions*)>>;
  SessionOptionsUniquePtr autoep_session_options;

  // TODO: move this out of CreateEpImpl.
  // Auto-configure lockdown AOT when a virtual device is selected.
  // Injects cross_device_prepare, prepare_only, and context_enable as defaults (skipped if user already set them).
  {
    bool has_virtual_device = false;
    std::optional<uint32_t> target_soc;
    for (size_t i = 0; i < num_devices; ++i) {
      if (factory->ort_api.HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_NPU) continue;
      const auto* metadata = factory->ort_api.HardwareDevice_Metadata(devices[i]);
      if (metadata) {
        const char* const* keys = nullptr;
        const char* const* values = nullptr;
        size_t num_entries = 0;
        factory->ort_api.GetKeyValuePairs(metadata, &keys, &values, &num_entries);
        for (size_t j = 0; j < num_entries; ++j) {
          if (strcmp(keys[j], kOrtHardwareDevice_MetadataKey_IsVirtual) == 0 &&
              strcmp(values[j], "1") == 0) {
            has_virtual_device = true;
            if (ep_metadata && ep_metadata[i]) {
              const char* target = factory->ort_api.GetKeyValue(ep_metadata[i], "target_architecture");
              if (target) {
                const auto parsed = qnn::soc::ParseSocModelTarget(target);
                if (!parsed || (target_soc && *target_soc != *parsed)) {
                  return factory->ort_api.CreateStatus(
                      ORT_INVALID_ARGUMENT, "Invalid or conflicting QNN virtual-device targets.");
                }
                target_soc = parsed;
              }
            }
            break;
          }
        }
      }
    }

    if (has_virtual_device) {
      if (!autoep_session_options) {
        OrtSessionOptions* cloned = nullptr;
        RETURN_IF_NOT_NULL(factory->ort_api.CloneSessionOptions(session_options, &cloned));
        autoep_session_options = SessionOptionsUniquePtr(cloned, factory->ort_api.ReleaseSessionOptions);
      }

      auto inject_default = [&](const char* key, const char* value) -> OrtStatus* {
        int has_key = 0;
        RETURN_IF_NOT_NULL(factory->ort_api.HasSessionConfigEntry(autoep_session_options.get(), key, &has_key));
        if (!has_key) {
          RETURN_IF_NOT_NULL(factory->ort_api.AddSessionConfigEntry(autoep_session_options.get(), key, value));
        }
        return nullptr;
      };

      const auto arch_key = provider_prefix + "htp_arch";
      const auto soc_key = provider_prefix + "soc_model";
      auto read_option = [&](const std::string& key, std::string& value) -> OrtStatus* {
        int present = 0;
        RETURN_IF_NOT_NULL(factory->ort_api.HasSessionConfigEntry(session_options, key.c_str(), &present));
        if (present) {
          size_t size = 0;
          RETURN_IF_NOT_NULL(factory->ort_api.GetSessionConfigEntry(session_options, key.c_str(), nullptr, &size));
          std::vector<char> buffer(size);
          RETURN_IF_NOT_NULL(factory->ort_api.GetSessionConfigEntry(
              session_options, key.c_str(), buffer.data(), &size));
          value = buffer.data();
        }
        return nullptr;
      };
      std::string explicit_arch;
      std::string explicit_soc;
      RETURN_IF_NOT_NULL(read_option(arch_key, explicit_arch));
      RETURN_IF_NOT_NULL(read_option(soc_key, explicit_soc));
      if (target_soc) {
        const uint32_t target_arch = qnn::soc::MapHtpArchFromSocModel(*target_soc);
        if (target_arch == 0) {
          return factory->ort_api.CreateStatus(
              ORT_INVALID_ARGUMENT, "No HTP architecture mapping for the selected QNN SoC model.");
        }
        const bool automatic_arch = explicit_arch.empty() || explicit_arch == "0";
        if (!automatic_arch && explicit_arch != std::to_string(target_arch)) {
          return factory->ort_api.CreateStatus(
              ORT_INVALID_ARGUMENT, "QNN htp_arch conflicts with the selected virtual-device target.");
        }
        if (!explicit_soc.empty() && explicit_soc != "0") {
          uint32_t model = qnn::soc::MapSocModelFromSocName(explicit_soc);
          if (model == 0) {
            const auto parsed = std::from_chars(
                explicit_soc.data(), explicit_soc.data() + explicit_soc.size(), model, 10);
            if (parsed.ec != std::errc{} || parsed.ptr != explicit_soc.data() + explicit_soc.size()) model = 0;
          }
          if (model != *target_soc) {
            return factory->ort_api.CreateStatus(
                ORT_INVALID_ARGUMENT,
                "QNN soc_model is unknown or conflicts with the selected virtual-device target.");
          }
        }
        // An explicitly present empty/zero option also requests automatic
        // selection; replace it with the transported target instead of leaving
        // QNN to fall back to its HOST-mode default architecture.
        RETURN_IF_NOT_NULL(factory->ort_api.AddSessionConfigEntry(
            autoep_session_options.get(), arch_key.c_str(), std::to_string(target_arch).c_str()));
        RETURN_IF_NOT_NULL(factory->ort_api.AddSessionConfigEntry(
            autoep_session_options.get(), soc_key.c_str(), std::to_string(*target_soc).c_str()));
      }
#if defined(_WIN32) && QNN_ARCH_ARM64
      else if ((explicit_arch.empty() || explicit_arch == "0") &&
               (explicit_soc.empty() || explicit_soc == "0")) {
        return factory->ort_api.CreateStatus(
            ORT_INVALID_ARGUMENT,
            "QNN virtual compilation requires target_architecture metadata or an explicit htp_arch/soc_model.");
      }
#endif

      RETURN_IF_NOT_NULL(inject_default("ep.context_enable", "1"));
      RETURN_IF_NOT_NULL(inject_default((provider_prefix + "enable_htp_prepare_only").c_str(), "1"));
      RETURN_IF_NOT_NULL(inject_default((provider_prefix + "enable_htp_cross_device_prepare").c_str(), "1"));

      session_options = autoep_session_options.get();
    }
  }
#endif  // defined(QNN_HTP_CROSS_DEVICE_PREPARE_AVAILABLE)
  std::unique_ptr<QnnEp> qnn_ep;
  try {
    const std::vector<const OrtHardwareDevice*> v_devices(devices, devices + num_devices);
    qnn_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, *session_options, v_devices, logger);
  } catch (const std::runtime_error& e) {
    return factory->ort_api.CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return factory->ort_api.CreateStatus(ORT_FAIL, "Unknown exception occurred while creating QNN EP.");
  }

  // HOST_ACCESSIBLE memory info is optional for regular QNN sessions, but is
  // required when a shared-memory allocator has been explicitly enabled.
  if (qnn_ep->qnn_allocator_type_ != qnn::QnnAllocatorType::NONE &&
      factory->host_accessible_memory_info_ == nullptr) {
    return factory->ort_api.CreateStatus(
        ORT_FAIL,
        "QNN shared-memory allocator was enabled, but its HOST_ACCESSIBLE memory info could not be created.");
  }

  // Preserve the existing session-level DX12 allocator path. Unlike HTP, GPU
  // does not advertise this memory info before session creation because the
  // factory cannot distinguish the two allocators while they share it.
  if (qnn::IsDx12SharedMemoryAllocator(qnn_ep->qnn_allocator_type_)) {
    for (OrtEpDevice* ep_device : factory->ep_devices_) {
      const OrtHardwareDevice* device = factory->ort_api.EpDevice_Device(ep_device);
      if (factory->ort_api.HardwareDevice_Type(device) == OrtHardwareDeviceType_GPU) {
        RETURN_IF_NOT_NULL(factory->ep_api.EpDevice_AddAllocatorInfo(
            ep_device, factory->host_accessible_memory_info_.get()));
      }
    }
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

OrtStatus* ORT_API_CALL QnnEpFactory::CreateAllocatorImpl(_In_ OrtEpFactory* this_ptr,
                                                          _In_ const OrtMemoryInfo* memory_info,
                                                          _In_opt_ const OrtKeyValuePairs* /*allocator_options*/,
                                                          _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);
  *allocator = nullptr;

  if (memory_info == nullptr) {
    return nullptr;
  }

  OrtDeviceMemoryType mem_type = factory->ort_api.MemoryInfoGetDeviceMemType(memory_info);
  if (mem_type != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
    return nullptr;
  }

  return qnn::HtpSharedMemoryAllocator::Create(factory->ort_api, memory_info, allocator);
}

void ORT_API_CALL QnnEpFactory::ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept {
  if (allocator == nullptr) {
    return;
  }

  // ORT Core releases allocators through the factory even when its preferred
  // allocator was created by QnnEp. Dispatch by the concrete allocator callback,
  // not factory/session state, because HTP and DX12 sessions may coexist.
  if (allocator->Alloc == qnn::HtpSharedMemoryAllocator::AllocImpl) {
    delete static_cast<qnn::HtpSharedMemoryAllocator*>(allocator);
#ifdef _WIN32
  } else if (allocator->Alloc == qnn::Dx12SharedMemoryAllocator::AllocImpl) {
    delete static_cast<qnn::Dx12SharedMemoryAllocator*>(allocator);
#endif
  } else {
    auto* factory = static_cast<QnnEpFactory*>(this_ptr);
    std::ignore = factory->ort_api.Logger_LogMessage(
        OrtLoggingManager::GetDefaultLoggerPtr(), OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        "Cannot release allocator of unknown type!", ORT_FILE, __LINE__, __FUNCTION__);
  }
}

OrtStatus* ORT_API_CALL QnnEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                             OrtDataTransferImpl** data_transfer) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);
  *data_transfer = nullptr;

  try {
    *data_transfer = new qnn::QnnHostAccessibleDataTransfer(factory->ort_api, factory->ep_api);
  } catch (const std::exception& e) {
    return factory->ort_api.CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return factory->ort_api.CreateStatus(ORT_FAIL, "Failed to create QNN data transfer.");
  }

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
  const auto is_npu = [&factory](const OrtHardwareDevice* device) {
    return factory->ort_api.HardwareDevice_Type(device) == OrtHardwareDeviceType_NPU;
  };
  const bool has_npu = std::find_if(devices, devices + num_devices, is_npu) != devices + num_devices;
  if (!has_npu) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return factory->ort_api.CreateStatus(ORT_EP_FAIL,
                                         "Currently QnnEpFactory::ValidateCompiledModelCompatibilityInfoImpl only supports OrtHardwareDeviceType_NPU, but "
                                         "no OrtHardwareDeviceType_NPU is found in the `devices` argument.");
  }

  using SessionOptionsUniquePtr = std::unique_ptr<OrtSessionOptions, std::function<void(OrtSessionOptions*)>>;
  OrtSessionOptions* temp_session_options = nullptr;
  if (OrtStatus* status = factory->ort_api.CreateSessionOptions(&temp_session_options)) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return status;
  }
  SessionOptionsUniquePtr session_options(temp_session_options, factory->ort_api.ReleaseSessionOptions);

  if (!OrtLoggingManager::HasDefaultLogger()) {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return factory->ort_api.CreateStatus(ORT_EP_FAIL, "Default logger is not available for model compatibility check.");
  }
  const OrtLogger* logger = OrtLoggingManager::GetDefaultLoggerPtr();

  std::unique_ptr<QnnEp> temp_qnn_ep;
  try {
    const std::vector<const OrtHardwareDevice*> v_devices(devices, devices + num_devices);
    temp_qnn_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, *session_options.get(), v_devices, logger);
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

  // Check if the device type is supported by QNN EP
  auto device_type = factory->ort_api.HardwareDevice_Type(hw);
  auto vendor_id = factory->ort_api.HardwareDevice_VendorId(hw);

  // QNN EP supports NPU/GPU devices with Qualcomm vendor ID, plus CPU only when the QNN CPU backend is enabled.
  const bool is_cpu = device_type == OrtHardwareDeviceType_CPU;
  const auto& default_backends = QnnEp::GetDefaultBackends();
  const bool type_unsupported = is_cpu ? !QnnCpuBackendEnabled()
                                       : default_backends.find(device_type) == default_backends.end();
  if (type_unsupported || (vendor_id != factory->vendor_id_ && !is_cpu)) {
    OrtDeviceEpIncompatibilityReason reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;
    return factory->ep_api.DeviceEpIncompatibilityDetails_SetDetails(
        details,
        reasons,
        QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED,
        "QNN EP only supports Qualcomm NPU and GPU devices");
  }

  using SessionOptionsUniquePtr = std::unique_ptr<OrtSessionOptions, std::function<void(OrtSessionOptions*)>>;
  OrtSessionOptions* temp_session_options = nullptr;
  RETURN_IF_NOT_NULL(factory->ort_api.CreateSessionOptions(&temp_session_options));
  SessionOptionsUniquePtr session_options(temp_session_options, factory->ort_api.ReleaseSessionOptions);

  // CPU has no default backend for session creation, so select it explicitly for this probe.
  if (is_cpu) {
    const auto provider_prefix = GetProviderOptionPrefix(factory->ep_name_);
    RETURN_IF_NOT_NULL(factory->ort_api.AddSessionConfigEntry(session_options.get(),
                                                              (provider_prefix + "backend_type").c_str(), "cpu"));
  }

  // Use default logger for the compatibility check
  if (!OrtLoggingManager::HasDefaultLogger()) {
    return factory->ort_api.CreateStatus(ORT_FAIL, "Default logger is not available for device compatibility check.");
  }
  const OrtLogger* logger = OrtLoggingManager::GetDefaultLoggerPtr();

  // Try to create a temporary QNN EP to test backend setup
  std::unique_ptr<QnnEp> temp_qnn_ep;
  try {
    const std::vector<const OrtHardwareDevice*> v_devices{hw};
    temp_qnn_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, *session_options.get(), v_devices, logger);
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

// External resource importer from D3D12
OrtStatus* ORT_API_CALL QnnEpFactory::CreateExternalResourceImporterForDeviceImpl(
    OrtEpFactory* this_ptr,
    const OrtEpDevice* /*ep_device*/,
    OrtExternalResourceImporterImpl** out_importer) noexcept {
  auto* factory = static_cast<QnnEpFactory*>(this_ptr);

  if (out_importer == nullptr) {
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "out_importer cannot be nullptr");
  }

  *out_importer = nullptr;

#ifdef _WIN32
  // Since CreateExternalResourceImporterForDeviceImpl doesn't take memory_device
  // as a parameter (unlike CreateSyncStreamForDeviceImpl), and we don't have direct API
  // to extract device ID from OrtEpDevice, we currently use 0.
  // In the future, we could extract device ID from ep_device using OrtApi::EpDevice_MemoryInfo
  // and then query the resulting OrtMemoryInfo for device ID.
  int device_id = 0;

  // Create the external resource importer
  try {
    *out_importer = std::make_unique<QnnExternalResourceImporterImpl>(device_id, factory->ort_api).release();
  } catch (...) {
    return factory->ort_api.CreateStatus(ORT_FAIL, "Failed to create external resource importer");
  }

  return nullptr;
#else
  return factory->ort_api.CreateStatus(
      ORT_NOT_IMPLEMENTED, "External resource import is not supported on non-Windows platforms");
#endif
}

OrtStatus* ORT_API_CALL QnnEpFactory::GetNumCustomOpDomainsImpl(
    _In_ OrtEpFactory* this_ptr,
    _Out_ size_t* num_domains) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  *num_domains = factory->custom_op_domains_.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEpFactory::GetCustomOpDomainsImpl(
    _In_ OrtEpFactory* this_ptr,
    _Out_writes_all_(num_domains) OrtCustomOpDomain** domains,
    _In_ size_t num_domains) noexcept {
  const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
  if (num_domains > factory->custom_op_domains_.size()) {
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "GetCustomOpDomains: num_domains exceeds the value returned by GetNumCustomOpDomains");
  }
  for (size_t i = 0; i < num_domains; ++i) {
    domains[i] = factory->custom_op_domains_[i];
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

  // kMinOrtApiVersion is computed at build time by
  // qcom/scripts/all/compute_min_ort_api_version.py from \since annotations
  // on every ORT API method this EP calls. Below this floor, GetApi() returns
  // a function table that lacks members the EP would dereference.
  constexpr uint32_t kMinOrtApiVersion = QNN_EP_MIN_ORT_API_VERSION;
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

  // Set default logger before factory construction so that the factory ctor can read it
  // via OrtLoggingManager::GetDefaultLoggerPtr() without needing a separate parameter.
  onnxruntime::OrtLoggingManager::SetDefaultLogger(default_logger);

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<onnxruntime::QnnEpFactory> factory;
  try {
    factory = std::make_unique<onnxruntime::QnnEpFactory>(
        registration_name,
        onnxruntime::ApiPtrs{*ort_api, *ep_api, *model_editor_api,
                             onnxruntime::OrtApiRuntimeCapabilities(requested_api_version)});
  } catch (const std::exception& e) {
    return ort_api->CreateStatus(ORT_FAIL, e.what());
  } catch (...) {
    return ort_api->CreateStatus(ORT_FAIL, "Unknown exception occurred while creating QNN EP factory.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  if (factory == nullptr) {
    return nullptr;
  }

  auto* qnn_factory = static_cast<onnxruntime::QnnEpFactory*>(factory);
  qnn_factory->UnloadBackendLibraries();
  delete qnn_factory;
  return nullptr;
}
}
