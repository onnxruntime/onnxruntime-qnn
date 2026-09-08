//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_backend_manager.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <gsl/gsl>
#include <memory>
#include <string>

#include "CPU/QnnCpuCommon.h"
#include "DSP/QnnDspCommon.h"
#include "GPU/QnnGpuCommon.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpSystemContext.h"
#include "IR/QnnIrCommon.h"
#include "IR/QnnIrGraph.h"
#include "QnnOpDef.h"
#include "Saver/QnnSaver.h"
#include "Saver/QnnSaverCommon.h"

#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#include "core/providers/qnn/builder/qnn_backend_system_dlc_plugin.h"
#include "core/providers/qnn/builder/qnn_configs_helper.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/qnn_telemetry.h"
#include "core/providers/qnn/shared_context.h"
#include "core/providers/qnn/qnn_external_resource_importer.h"

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
#include "core/providers/qnn/builder/qnn_windows_file_mapper.h"
#endif

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

// Ensure that we have a recent enough version of QNN
static_assert(QNN_API_VERSION_MAJOR > 2 ||
                  (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 29),
              "Minimum required QAIRT SDK version is 2.39.0");

namespace onnxruntime {
namespace qnn {

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                          uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList,
                                                                uint32_t* numProviders);

static Qnn_Version_t GetQnnInterfaceApiVersion(const QnnInterface_t* qnn_interface) {
  return qnn_interface->apiVersion.coreApiVersion;
}

static Qnn_Version_t GetQnnInterfaceApiVersion(const QnnSystemInterface_t* qnn_interface) {
  return qnn_interface->systemApiVersion;
}

static const char* DlError() {
#ifdef _WIN32
  return "";
#else
  return ::dlerror();
#endif
}

class QnnIrConfig : public QnnSerializerConfig {
 public:
  QnnIrConfig(std::string backend_path, std::string dlc_dir)
      : QnnSerializerConfig(std::move(backend_path)), dlc_dir_(std::move(dlc_dir)), configs_builder_(MakeConfigsBuilder()) {
  }

  const QnnGraph_Config_t** Configure() override {
    auto configs_builder = MakeConfigsBuilder();

    std::filesystem::path dlc_path = (dlc_dir_ / (GetGraphName() + ".dlc"));
    std::string dlc_path_str = dlc_path.string();
    gsl::not_null<QnnIrGraph_CustomConfig_t*> dlc_path_config = configs_builder.PushCustomConfig();
    dlc_path_config->option = QNN_IR_GRAPH_CONFIG_OPTION_SERIALIZATION;
    dlc_path_config->serializationOption.serializationType = QNN_IR_GRAPH_SERIALIZATION_TYPE_FLAT_BUFFER;
    dlc_path_config->serializationOption.outputPath = dlc_path_str.c_str();

    gsl::not_null<QnnGraph_Config_t*> dlc_path_custom_config = configs_builder.PushConfig();
    dlc_path_custom_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    dlc_path_custom_config->customConfig = dlc_path_config;

    std::filesystem::create_directories(dlc_path.parent_path());

    // Keep the pointer to dlc_path_str's null-terminated string alive.
    std::swap(dlc_path_str, dlc_path_str_);

    std::swap(configs_builder, configs_builder_);
    return configs_builder_.GetQnnConfigs();
  }

  bool SupportsArbitraryGraphConfigs() const override {
    return false;
  }

 private:
  static QnnConfigsBuilder<QnnGraph_Config_t, QnnIrGraph_CustomConfig_t> MakeConfigsBuilder() {
    return QnnConfigsBuilder<QnnGraph_Config_t, QnnIrGraph_CustomConfig_t>(QNN_GRAPH_CONFIG_INIT, QNN_IR_GRAPH_CUSTOM_CONFIG_INIT);
  }

  std::filesystem::path dlc_dir_;
  std::string dlc_path_str_;
  QnnConfigsBuilder<QnnGraph_Config_t, QnnIrGraph_CustomConfig_t> configs_builder_;
};

class QnnSaverConfig : public QnnSerializerConfig {
 public:
  QnnSaverConfig(std::string backend_path) : QnnSerializerConfig(std::move(backend_path)) {}

  const QnnGraph_Config_t** Configure() override {
    return nullptr;
  }

  bool SupportsArbitraryGraphConfigs() const override {
    return true;
  }
};

QnnSerializerConfig::~QnnSerializerConfig() = default;

QnnSerializerConfig::QnnSerializerConfig(std::string backend_path)
    : backend_path_(std::move(backend_path)) {}

std::unique_ptr<QnnSerializerConfig> QnnSerializerConfig::CreateIr(std::string backend_path, std::string dlc_dir) {
  return std::make_unique<QnnIrConfig>(std::move(backend_path), std::move(dlc_dir));
}

std::unique_ptr<QnnSerializerConfig> QnnSerializerConfig::CreateSaver(std::string backend_path) {
  return std::make_unique<QnnSaverConfig>(std::move(backend_path));
}

const std::string& QnnSerializerConfig::GetBackendPath() const {
  return backend_path_;
}

const std::string& QnnSerializerConfig::GetGraphName() const {
  return graph_name_;
}

void QnnSerializerConfig::SetGraphName(std::string graph_name) {
  graph_name_ = std::move(graph_name);
}

Ort::Status ReadBinaryFromFile(const std::string& file_path, uint8_t* buffer, size_t buffer_size) {
  RETURN_IF(nullptr == buffer, "Binary buffer is nullptr");
  std::ifstream in(file_path, std::ifstream::binary);
  RETURN_IF(!in, ("Failed to open input file: " + file_path).c_str());
  RETURN_IF(!in.read(reinterpret_cast<char*>(buffer), buffer_size),
            ("Failed to read the contents of: " + file_path).c_str());
  return Ort::Status();
}

Ort::Status QnnBackendManager::ParseLoraConfig(std::string lora_config_path) {
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, ("Acquiring the QnnInterface " + lora_config_path).c_str());

  // QNN Lora Config file format should be a single line, with the graph name first,
  // followed by the qnn lora context binary path, separated by a semicolon (;)
  // Example: <graph_name>;<binary_path>
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, ("Loading Lora Config " + lora_config_path).c_str());
  std::ifstream file(lora_config_path);
  std::string line;

  if (file.is_open()) {
    if (std::getline(file, line)) {
      std::istringstream ss(line);
      std::string graph_name;
      std::string lora_adapter_bin_path;

      if (std::getline(ss, graph_name, ';') && std::getline(ss, lora_adapter_bin_path)) {
        size_t buffer_size = std::filesystem::file_size(lora_adapter_bin_path.c_str());

        RETURN_IF(0 == buffer_size, "Received path to an empty file. Nothing to deserialize.");
        std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(buffer_size);
        void* voidBufferPtr = static_cast<void*>(buffer.get());
        QnnContext_Buffer_t contextBuffer{QNN_CONTEXT_BUFFER_VERSION_1,
                                          {QNN_CONTEXTMEMTYPE_RAW, {{voidBufferPtr, buffer_size}}}};

        auto status = ReadBinaryFromFile(lora_adapter_bin_path,
                                         reinterpret_cast<uint8_t*>(buffer.get()),
                                         buffer_size);

        RETURN_IF(!status.IsOK(), "Failed to read binary data.");
        Qnn_GraphHandle_t graph;
        bool graph_retrieve_success = false;
        for (size_t cIdx = 0; cIdx < contexts_.size(); cIdx++) {
          auto graph_retrieve_rt = qnn_interface_.graphRetrieve(contexts_[cIdx], graph_name.c_str(), &graph);
          if (QNN_SUCCESS != graph_retrieve_rt) {
            continue;
          }

          graph_retrieve_success = true;

          auto context_apply_binary_section_rt = qnn_interface_.contextApplyBinarySection(
              contexts_[cIdx], graph, QNN_CONTEXT_SECTION_UPDATABLE, &contextBuffer, profile_backend_handle_, nullptr);
          RETURN_IF(QNN_SUCCESS != context_apply_binary_section_rt,
                    ("Failed to apply binary section. " +
                     utils::FormatQnnError(qnn_interface_, context_apply_binary_section_rt))
                        .c_str());
          break;
        }
        RETURN_IF_NOT(graph_retrieve_success,
                      ("Failed to retrieve graph: " + graph_name + " and apply binary section.").c_str());
      }
    }
    file.close();
  } else {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, ("Unable to load Lora Config " + lora_config_path).c_str());
  }

  return Ort::Status();
}

template <typename F, class T>
Ort::Status QnnBackendManager::GetQnnInterfaceProvider(const char* lib_path,
                                                       const char* interface_provider_name,
                                                       void** backend_lib_handle,
                                                       Qnn_Version_t req_version,
                                                       T** interface_provider) {
  std::string error_msg;
  *backend_lib_handle = LoadLib(lib_path,
                                static_cast<int>(DlOpenFlag::DL_NOW) | static_cast<int>(DlOpenFlag::DL_GLOBAL),
                                error_msg);
  RETURN_IF(nullptr == *backend_lib_handle, ("Unable to load backend, error: " + error_msg + " " + DlError()).c_str());

  // Get QNN Interface providers
  F GetInterfaceProviders{nullptr};
  GetInterfaceProviders = ResolveSymbol<F>(*backend_lib_handle, interface_provider_name, *logger_ptr_);
  RETURN_IF(nullptr == GetInterfaceProviders, "Failed to get QNN providers!");

  T** interface_providers{nullptr};
  uint32_t num_providers{0};

  auto result = GetInterfaceProviders((const T***)&interface_providers, &num_providers);
  RETURN_IF((QNN_SUCCESS != result || nullptr == *interface_providers || 0 == num_providers),
            "Failed to get QNN providers.");

  if (skip_qnn_version_check_) {
    // When skipping version check, use the first available provider.
    *interface_provider = interface_providers[0];
    return Ort::Status();
  }

  bool found_valid_interface{false};
  for (size_t pIdx = 0; pIdx < num_providers; pIdx++) {
    Qnn_Version_t interface_version = GetQnnInterfaceApiVersion(interface_providers[pIdx]);

    std::ostringstream oss;
    oss << lib_path << " interface version: " << interface_version.major << "."
        << interface_version.minor << "." << interface_version.patch;
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, oss.str().c_str());

    // Check the interface's API version against the required version.
    // Major versions must match. The interface's minor version must be greater OR equal with a suitable patch version.
    if (interface_version.major == req_version.major) {
      bool minor_and_patch_version_ok = (interface_version.minor > req_version.minor) ||
                                        (interface_version.minor == req_version.minor &&
                                         interface_version.patch >= req_version.patch);
      if (minor_and_patch_version_ok) {
        found_valid_interface = true;
        *interface_provider = interface_providers[pIdx];
        break;
      }
    }
  }

  RETURN_IF_NOT(found_valid_interface, ("Unable to find a valid interface for " + std::string(lib_path)).c_str());

  return Ort::Status();
}

void QnnBackendManager::SetQnnBackendType(uint32_t backend_id) {
  switch (backend_id) {
    case QNN_BACKEND_ID_CPU:
      qnn_backend_type_ = QnnBackendType::CPU;
      break;
    case QNN_BACKEND_ID_GPU:
      qnn_backend_type_ = QnnBackendType::GPU;
      break;
    case QNN_BACKEND_ID_DSP:
      qnn_backend_type_ = QnnBackendType::DSP;
      break;
    case QNN_BACKEND_ID_HTP:
      qnn_backend_type_ = QnnBackendType::HTP;
      break;
    case QNN_BACKEND_ID_IR:
    case QNN_BACKEND_ID_SAVER:
      qnn_backend_type_ = QnnBackendType::SERIALIZER;
      break;
    default:
      qnn_backend_type_ = QnnBackendType::CPU;
      break;
  }
}

Ort::Status QnnBackendManager::LoadBackend() {
  if (backend_lib_loaded_) {
    // Backend already loaded
    return Ort::Status();
  }

#if defined(__aarch64__) && defined(__linux__)
  // QNN requires ADSP_LIBRARY_PATH to be set in order to find skel libs on Linux
  static std::once_flag set_adsp_path_once;

  std::call_once(set_adsp_path_once, [&backend_path = backend_path_]() {
    constexpr std::string_view kAdspLibraryPathEnvVar{"ADSP_LIBRARY_PATH"};
    const char* existingPath = getenv(kAdspLibraryPathEnvVar.data());
    if (existingPath != nullptr) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                  ORT_LOGGING_LEVEL_WARNING,
                  ("Using existing ADSP_LIBRARY_PATH setting of " +
                   std::string(existingPath) + ", which may cause the HTP backend to fail.")
                      .c_str());
      return;
    }

    std::filesystem::path _backend_path(backend_path);
    std::filesystem::path qnnLibPath = _backend_path.is_absolute()
                                           ? _backend_path.parent_path()
                                           : std::filesystem::path(OrtGetRuntimePath());
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                ORT_LOGGING_LEVEL_WARNING,
                ("Setting " + std::string(kAdspLibraryPathEnvVar) + " = " + qnnLibPath.string()).c_str());
    setenv(kAdspLibraryPathEnvVar.data(), qnnLibPath.c_str(), 1);
  });
#endif

  QnnInterface_t* backend_interface_provider{nullptr};
  RETURN_IF_ERROR((GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t, QnnInterface_t>(backend_path_.c_str(),
                                                                                         "QnnInterface_getProviders",
                                                                                         &backend_lib_handle_,
                                                                                         {QNN_API_VERSION_MAJOR,
                                                                                          QNN_API_VERSION_MINOR,
                                                                                          QNN_API_VERSION_PATCH},
                                                                                         &backend_interface_provider)));
  qnn_interface_ = backend_interface_provider->QNN_INTERFACE_VER_NAME;
  htp_power_config_manager_.Init(qnn_interface_);
  backend_id_ = backend_interface_provider->backendId;
  core_api_version_ = backend_interface_provider->apiVersion.coreApiVersion;
  backend_api_version_ = backend_interface_provider->apiVersion.backendApiVersion;
  SetQnnBackendType(backend_id_);

  Qnn_Version_t backend_interface_version = GetQnnInterfaceApiVersion(backend_interface_provider);
  std::ostringstream oss;
  oss << "Found valid interface, version: " << backend_interface_version.major
      << "." << backend_interface_version.minor << "." << backend_interface_version.patch
      << " backend provider name: " << backend_interface_provider->providerName
      << " backend id: " << backend_id_;
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, oss.str().c_str());

  backend_lib_loaded_ = true;
  return Ort::Status();
}

QnnSerializerConfig* QnnBackendManager::GetQnnSerializerConfig() {
  return qnn_serializer_config_.get();
}

// Loads the intended backend (e.g., HTP, CPU, etc) to get its type, and then
// sets QnnSaver or QnnIr as the active backend. QNN op builders will still see the intended backend
// (e.g., HTP) as the backend type to ensure they emit the expected QNN API calls.
// The intended backend's interface is also retained so that calls to QnnBackend_validateOpConfig
// are routed to it, rather than the serializer backend.
//
// QnnSaver and QnnIr are "debugging" backends that serialize all QNN API calls (and weights) into
// local files: Saver dumps to C++ sources and Ir to .dlc archives.
// This information can be used to debug issues by replaying QNN API calls with another backend.
Ort::Status QnnBackendManager::LoadQnnSerializerBackend() {
  // Load the intended backend (e.g., HTP, CPU) to get its type and validator interface.
  // The library handle is stored in validator_backend_lib_handle_ and kept alive for the
  // lifetime of this object so that backendValidateOpConfig remains callable.
  QnnInterface_t* backend_interface_provider{nullptr};
  RETURN_IF_ERROR((GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t, QnnInterface_t>(backend_path_.c_str(),
                                                                                         "QnnInterface_getProviders",
                                                                                         &validator_backend_lib_handle_,
                                                                                         {QNN_API_VERSION_MAJOR,
                                                                                          QNN_API_VERSION_MINOR,
                                                                                          QNN_API_VERSION_PATCH},
                                                                                         &backend_interface_provider)));

  // Set the "intended" backend type so that QNN builders still make the expected QNN API calls.
  backend_id_ = backend_interface_provider->backendId;
  core_api_version_ = backend_interface_provider->apiVersion.coreApiVersion;
  backend_api_version_ = backend_interface_provider->apiVersion.backendApiVersion;
  SetQnnBackendType(backend_id_);

  // Store the validator interface (e.g., HTP) for use in backendValidateOpConfig calls. Leaving it
  // null when disabled keeps validator interface/handle both null, so ValidateQnnNode() falls back
  // to the serializer's generic checks -- correct on a device-less host where the target backend
  // validates against an arch-agnostic op table and over-rejects arch-specific ops.
  if (!skip_backend_op_validation_) {
    qnn_validator_interface_ = backend_interface_provider->QNN_INTERFACE_VER_NAME;
  }

  // Load the serializer backend and set it as the activate backend.
  QnnInterface_t* serializer_interface_provider{nullptr};
  RETURN_IF_ERROR((GetQnnInterfaceProvider<QnnInterfaceGetProvidersFn_t, QnnInterface_t>(
      qnn_serializer_config_->GetBackendPath().c_str(),
      "QnnInterface_getProviders",
      &backend_lib_handle_,  // NOTE: QnnSaver/Ir library handle is set
      {QNN_API_VERSION_MAJOR,
       QNN_API_VERSION_MINOR,
       QNN_API_VERSION_PATCH},
      &serializer_interface_provider)));
  qnn_interface_ = serializer_interface_provider->QNN_INTERFACE_VER_NAME;  // NOTE: QnnSaver/Ir will provide the interfaces
  htp_power_config_manager_.Init(qnn_interface_);

  Qnn_Version_t backend_interface_version = GetQnnInterfaceApiVersion(backend_interface_provider);
  Qnn_Version_t serializer_interface_version = GetQnnInterfaceApiVersion(serializer_interface_provider);

  std::ostringstream oss1;
  oss1 << "Using QnnSaver/Ir version: " << serializer_interface_version.major << "."
       << serializer_interface_version.minor << "." << serializer_interface_version.patch
       << " provider name : " << serializer_interface_provider->providerName;
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, oss1.str().c_str());

  std::ostringstream oss2;
  oss2 << "Intended backend provider name: " << backend_interface_provider->providerName
       << " backend id: " << backend_id_
       << " interface version: " << backend_interface_version.major
       << "." << backend_interface_version.minor << "." << backend_interface_version.patch;
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, oss2.str().c_str());

  return Ort::Status();
}

Ort::Status QnnBackendManager::LoadQnnSystemLib() {
  if (system_lib_loaded_) {
    return Ort::Status();
  }

#ifdef _WIN32
  std::string system_lib_file = "QnnSystem.dll";
#else
  std::string system_lib_file = "libQnnSystem.so";
#endif  // #ifdef _WIN32
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Loading QnnSystem lib");
  std::filesystem::path lib_file_path(backend_path_.c_str());
  std::string sys_file_path(lib_file_path.remove_filename().string() + system_lib_file);
  QnnSystemInterface_t* system_interface_provider{nullptr};
  RETURN_IF_ERROR((GetQnnInterfaceProvider<QnnSystemInterfaceGetProvidersFn_t, QnnSystemInterface_t>(
      sys_file_path.c_str(),
      "QnnSystemInterface_getProviders",
      &system_lib_handle_,
      {QNN_SYSTEM_API_VERSION_MAJOR,
       QNN_SYSTEM_API_VERSION_MINOR,
       QNN_SYSTEM_API_VERSION_PATCH},
      &system_interface_provider)));
  Qnn_Version_t system_interface_version = GetQnnInterfaceApiVersion(system_interface_provider);
  qnn_sys_interface_ = system_interface_provider->QNN_SYSTEM_INTERFACE_VER_NAME;

  std::ostringstream oss;
  oss << "Found valid system interface, version: " << system_interface_version.major
      << "." << system_interface_version.minor
      << " backend provider name: " << system_interface_provider->providerName;
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, oss.str().c_str());

  system_lib_loaded_ = true;
  return Ort::Status();
}

void QnnLogging(const char* format,
                QnnLog_Level_t level,
                uint64_t timestamp,
                va_list argument_parameter) {
  ORT_UNUSED_PARAMETER(level);
  ORT_UNUSED_PARAMETER(timestamp);

  if (!OrtLoggingManager::HasDefaultLogger()) {
    return;
  }

  // QNN-EP COPY START
  // There is an unknown bug in Ort::Logger::LogFormattedMessage which causes crashes.
  // Below implementations are directly copied from core/common/logging/capture.cc to create formatted string
  // and leverage Ort::Logger::LogMessage instead.

  static constexpr auto kTruncatedWarningText = "[...truncated...]";
  static constexpr int kMaxMessageSize = 2048;
  char message_buffer[kMaxMessageSize];
  const auto message = gsl::make_span(message_buffer);

  bool error = false;
  bool truncated = false;

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) && !defined(__GNUC__))
  errno = 0;
  const int nbrcharacters = vsnprintf_s(message.data(), message.size(), _TRUNCATE, format, argument_parameter);
  if (nbrcharacters < 0) {
    error = errno != 0;
    truncated = !error;
  }
#else
  const int nbrcharacters = vsnprintf(message.data(), message.size(), format, argument_parameter);
  error = nbrcharacters < 0;
  truncated = (nbrcharacters >= 0 && static_cast<size_t>(nbrcharacters) > message.size());
#endif

  std::ostringstream stream;
  if (error) {
    stream << "\n\tERROR LOG MSG NOTIFICATION: Failure to successfully parse the message";
    stream << '"' << format << '"' << std::endl;
  } else if (truncated) {
    stream << message.data() << kTruncatedWarningText;
  } else {
    stream << message.data();
  }
  // QNN-EP COPY END

  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_VERBOSE, stream.str().c_str());
}

Ort::Status QnnBackendManager::InitializeQnnLogCommon(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                      Qnn_LogHandle_t& log_handle,
                                                      const std::string& backend_label) {
  if (log_handle) {
    // Log already initialized
    return Ort::Status();
  }

  auto ort_log_level = logger_ptr_->GetLoggingSeverityLevel();
  QnnLog_Level_t qnn_log_level = MapOrtSeverityToQNNLogLevel(ort_log_level);
  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE,
                  ("Set Qnn log level for " + backend_label + ": " + std::to_string(qnn_log_level)).c_str());

  // NOTE: Even if logCreate() fails and QNN does not return a valid log_handle, QNN may still
  // call the QnnLogging() callback. So, we have to make sure that QnnLogging() can handle calls
  // in which ORT logging is not available.
  Qnn_ErrorHandle_t result = qnn_interface.logCreate(QnnLogging, qnn_log_level, &log_handle);

  if (result != QNN_SUCCESS) {
    switch (result) {
      case QNN_COMMON_ERROR_NOT_SUPPORTED:
        ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Logging not supported in the QNN backend.");
        break;
      case QNN_LOG_ERROR_INVALID_ARGUMENT:
        ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Invalid argument provided to QnnLog_create.");
        break;
      case QNN_LOG_ERROR_MEM_ALLOC:
        ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Memory allocation error during QNN logging initialization.");
        break;
      case QNN_LOG_ERROR_INITIALIZATION:
        ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Initialization of logging failed in the QNN backend.");
        break;
      default:
        ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_WARNING,
                        "Unknown error occurred while initializing logging in the QNN backend.");
        break;
    }
  }

  RETURN_IF(QNN_BACKEND_NO_ERROR != result,
            ("Failed to initialize logging in the " + backend_label + ". Error: " +
             QnnErrorHandleToString(result))
                .c_str());
  return Ort::Status();
}

Ort::Status QnnBackendManager::InitializeQnnLog() {
  return InitializeQnnLogCommon(qnn_interface_, log_handle_, "backend");
}

Ort::Status QnnBackendManager::InitializeQnnValidatorLog() {
  return InitializeQnnLogCommon(qnn_validator_interface_, validator_log_handle_, "validator");
}

QnnLog_Level_t QnnBackendManager::MapOrtSeverityToQNNLogLevel(OrtLoggingLevel ort_log_level) {
  // Map ORT log severity to Qnn log level
  switch (ort_log_level) {
    case ORT_LOGGING_LEVEL_VERBOSE: {
      switch ((GetQnnBackendType())) {
        case QnnBackendType::GPU:
          // Currently GPU needs this log level to work.
          // This switch will be removed once this is resolved.
          return QNN_LOG_LEVEL_DEBUG;
        default:
          return QNN_LOG_LEVEL_VERBOSE;
      }
    }
    case ORT_LOGGING_LEVEL_INFO:
      return QNN_LOG_LEVEL_INFO;
    case ORT_LOGGING_LEVEL_WARNING:
      return QNN_LOG_LEVEL_WARN;
    case ORT_LOGGING_LEVEL_ERROR:
    case ORT_LOGGING_LEVEL_FATAL:
    default:
      return QNN_LOG_LEVEL_ERROR;
  }
}

Ort::Status QnnBackendManager::SetQnnLogLevelCommon(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                    Qnn_LogHandle_t log_handle,
                                                    QnnLog_Level_t qnn_log_level,
                                                    const std::string& label) {
  Qnn_ErrorHandle_t result = qnn_interface.logSetLogLevel(log_handle, qnn_log_level);
  if (QNN_SUCCESS != result) {
    if (result == QNN_LOG_ERROR_INVALID_ARGUMENT) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR,
                      ("Invalid log level argument for QnnLog_setLogLevel (" + label + ").").c_str());
    } else if (result == QNN_LOG_ERROR_INVALID_HANDLE) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR,
                      ("Invalid log handle for QnnLog_setLogLevel (" + label + ").").c_str());
    }
  }
  RETURN_IF(QNN_BACKEND_NO_ERROR != result,
            ("Failed to set log level in Qnn backend (" + label + "). Error: " + QnnErrorHandleToString(result)).c_str());
  return Ort::Status();
}

Ort::Status QnnBackendManager::ResetQnnLogLevel(std::optional<OrtLoggingLevel> ort_log_level) {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (!backend_setup_completed_) {
    return Ort::Status();
  }
  RETURN_IF(nullptr == log_handle_, "Unable to update QNN Log Level. Invalid QNN log handle.");

  OrtLoggingLevel actual_log_level = ort_log_level.has_value() ? *ort_log_level
                                                               : logger_ptr_->GetLoggingSeverityLevel();
  QnnLog_Level_t qnn_log_level = MapOrtSeverityToQNNLogLevel(actual_log_level);

  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, ("Updating Qnn log level to: " + std::to_string(qnn_log_level)).c_str());

  RETURN_IF_ERROR(SetQnnLogLevelCommon(qnn_interface_, log_handle_, qnn_log_level, "main"));

  // Also update the validator backend log level if it is active (serializer flow: QnnIr).
  if (validator_log_handle_ != nullptr) {
    RETURN_IF_ERROR(SetQnnLogLevelCommon(qnn_validator_interface_, validator_log_handle_, qnn_log_level, "validator"));
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::InitializeBackendCommon(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                       Qnn_LogHandle_t log_handle,
                                                       Qnn_BackendHandle_t& backend_handle,
                                                       bool& initialized_flag,
                                                       const std::string& backend_label) {
  Qnn_ErrorHandle_t result = qnn_interface.backendCreate(log_handle,
                                                         (const QnnBackend_Config_t**)backend_config_,
                                                         &backend_handle);
  RETURN_IF(QNN_BACKEND_NO_ERROR != result,
            ("Failed to initialize backend (" + backend_label + "). Error: " +
             QnnErrorHandleToString(result))
                .c_str());

  initialized_flag = true;
  return Ort::Status();
}

Ort::Status QnnBackendManager::InitializeBackend(bool enable_gpu_weight_sharing) {
  if (backend_initialized_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Backend initialized already.");
    return Ort::Status();
  }

  if (IsGpuBackend(GetQnnBackendType()) && enable_gpu_weight_sharing) {
    gpu_backend_custom_config_.option = QNN_GPU_BACKEND_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    gpu_backend_custom_config_.weightSharingEnabled = 1;

    backend_config_wrapper_.option = QNN_BACKEND_CONFIG_OPTION_CUSTOM;
    backend_config_wrapper_.customConfig = &gpu_backend_custom_config_;

    backend_configs_ptr_[0] = &backend_config_wrapper_;
    backend_configs_ptr_[1] = nullptr;

    backend_config_ = backend_configs_ptr_;
  }

  return InitializeBackendCommon(qnn_interface_, log_handle_, backend_handle_, backend_initialized_, "backend");
}

Ort::Status QnnBackendManager::InitializeValidatorBackend() {
  if (validator_backend_initialized_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Validator Backend initialized already.");
    return Ort::Status();
  }

  return InitializeBackendCommon(qnn_validator_interface_, validator_log_handle_,
                                 validator_backend_handle_, validator_backend_initialized_, "validator");
}

Ort::Status QnnBackendManager::ShutdownBackendCommon(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                     Qnn_BackendHandle_t& backend_handle,
                                                     bool& initialized_flag,
                                                     const std::string& backend_label) {
  if (qnn_interface.backendFree != nullptr) {
    RETURN_IF(QNN_BACKEND_NO_ERROR != qnn_interface.backendFree(backend_handle),
              ("Failed to shutdown " + backend_label + "!").c_str());
  }

  initialized_flag = false;

  return Ort::Status();
}

Ort::Status QnnBackendManager::ShutdownBackend() {
  if (!backend_initialized_) {
    return Ort::Status();
  }

  return ShutdownBackendCommon(qnn_interface_, backend_handle_, backend_initialized_, "backend");
}

Ort::Status QnnBackendManager::ShutdownValidatorBackend() {
  if (!validator_backend_initialized_) {
    return Ort::Status();
  }

  return ShutdownBackendCommon(qnn_validator_interface_, validator_backend_handle_,
                               validator_backend_initialized_, "validator");
}

bool QnnBackendManager::IsDevicePropertySupported(const QNN_INTERFACE_VER_TYPE& qnn_interface) {
  if (nullptr != qnn_interface.propertyHasCapability) {
    auto rt = qnn_interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == rt || QNN_PROPERTY_ERROR_UNKNOWN_KEY == rt) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Device property not supported or unknown to backend.");
      return false;
    }
  }

  return true;
}

namespace {

struct PlatformInfoDeleter {
  PlatformInfoDeleter(const QNN_INTERFACE_VER_TYPE& qnn_interface, const Qnn_LogHandle_t log_handle)
      : qnn_interface_(&qnn_interface), log_handle_(log_handle) {}

  void operator()(const QnnDevice_PlatformInfo_t* p) {
    if (p) {
      qnn_interface_->deviceFreePlatformInfo(log_handle_, p);
    }
  }

 private:
  const QNN_INTERFACE_VER_TYPE* qnn_interface_;
  Qnn_LogHandle_t log_handle_;
};

using DevicePlatformInfoPtr = std::unique_ptr<const QnnDevice_PlatformInfo_t, PlatformInfoDeleter>;

std::pair<DevicePlatformInfoPtr, Qnn_ErrorHandle_t> GetDevicePlatformInfo(
    const QNN_INTERFACE_VER_TYPE& qnn_interface,
    const Qnn_LogHandle_t log_handle) {
  const QnnDevice_PlatformInfo_t* platformInfoRawPtr = nullptr;
  PlatformInfoDeleter deleter(qnn_interface, log_handle);
  Qnn_ErrorHandle_t status = qnn_interface.deviceGetPlatformInfo(log_handle, &platformInfoRawPtr);

  // If deviceGetPlatformInfo failed, return nullptr with appropriate status
  return std::make_pair(DevicePlatformInfoPtr(platformInfoRawPtr, deleter), status);
}

}  // namespace

Ort::Status QnnBackendManager::CreateDeviceCommon(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                  Qnn_LogHandle_t log_handle,
                                                  Qnn_DeviceHandle_t& device_handle,
                                                  bool& device_created_flag,
                                                  bool allow_hw_device_enumeration) {
  // Create device if its property supported
  if (!IsDevicePropertySupported(qnn_interface)) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Skip to create device.");
    return Ort::Status();
  }

  qnn::QnnConfigsBuilder<QnnDevice_Config_t, QnnHtpDevice_CustomConfig_t> device_configs_builder(QNN_DEVICE_CONFIG_INIT,
                                                                                                 {});

  // These will hold device selection data when device_id_ != 0
  DevicePlatformInfoPtr device_platform_info(nullptr, PlatformInfoDeleter(qnn_interface, log_handle));
  std::unique_ptr<QnnDevice_PlatformInfo_t> device_platform_info_config;
  std::unique_ptr<QnnDevice_Config_t> device_platform_info_std_config;

  if (qnn_backend_type_ == QnnBackendType::HTP) {
    // Set SoC Model. The *enum* Qnn_SocModel_t is deprecated and will not be updated in the future. Therefore,
    // must use the latest SDK documentation to get the SoC model of the latest HW.
    if (soc_model_ != QNN_SOC_MODEL_UNKNOWN) {
      gsl::not_null<QnnHtpDevice_CustomConfig_t*> custom_config = device_configs_builder.PushCustomConfig();
      custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
      custom_config->socModel = soc_model_;

      gsl::not_null<QnnDevice_Config_t*> device_config = device_configs_builder.PushConfig();
      device_config->option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      device_config->customConfig = custom_config;
    }

    // Set the minimum HTP architecture. The driver will use ops that are compatible with this minimum architecture.
    if (htp_arch_ != QNN_HTP_DEVICE_ARCH_NONE) {
      gsl::not_null<QnnHtpDevice_CustomConfig_t*> custom_config = device_configs_builder.PushCustomConfig();
      custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
      custom_config->arch.arch = htp_arch_;
      custom_config->arch.deviceId = device_id_;

      gsl::not_null<QnnDevice_Config_t*> device_config = device_configs_builder.PushConfig();
      device_config->option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      device_config->customConfig = custom_config;
    }

    QnnDevice_HardwareDeviceInfo_t* selected_device = nullptr;

    if (allow_hw_device_enumeration && device_id_ != 0) {
      Qnn_ErrorHandle_t result;
      std::tie(device_platform_info, result) = GetDevicePlatformInfo(qnn_interface, log_handle);
      if (QNN_SUCCESS != result) {
        return MAKE_EP_FAIL(("Failed to get platform info. Error: " + QnnErrorHandleToString(result)).c_str());
      }

      if (device_platform_info->version != QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
        return MAKE_EP_FAIL("Unsupported device platform info.");
      }

      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, ("QNN Found " + std::to_string(device_platform_info->v1.numHwDevices) + " HTP devices.").c_str());

      for (uint32_t deviceIdx = 0; deviceIdx < device_platform_info->v1.numHwDevices; ++deviceIdx) {
        auto& hwDevice = device_platform_info->v1.hwDevices[deviceIdx];
        if (device_id_ == hwDevice.v1.deviceId) {
          selected_device = &hwDevice;
          break;
        }
      }

      if (!selected_device) {
        return MAKE_EP_FAIL(("Could not find device with ID = " + std::to_string(device_id_)).c_str());
      }

      device_platform_info_config = std::make_unique<QnnDevice_PlatformInfo_t>();
      device_platform_info_config->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
      device_platform_info_config->v1.numHwDevices = 1;
      device_platform_info_config->v1.hwDevices = selected_device;

      device_platform_info_std_config = std::make_unique<QnnDevice_Config_t>();
      device_platform_info_std_config->option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
      device_platform_info_std_config->hardwareInfo = device_platform_info_config.get();
    }
  }

  // Consolidate device configs of all types
  std::vector<const QnnDevice_Config_t*> all_device_configs;

  // Reserve enough for HTP configs, platform config, and null terminator
  all_device_configs.reserve(device_configs_builder.GetSize() + (device_platform_info_std_config ? 2 : 1));

  // Add HTP configs
  const QnnDevice_Config_t** htp_configs = device_configs_builder.GetQnnConfigs();
  if (htp_configs) {
    for (const QnnDevice_Config_t** config = htp_configs; *config; ++config) {
      all_device_configs.push_back(*config);
    }
  }

  if (device_platform_info_std_config) {
    // Add platform info config
    all_device_configs.push_back(device_platform_info_std_config.get());
  }

  // Null terminator
  all_device_configs.push_back(nullptr);

  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Create device.");
  if (nullptr != qnn_interface.deviceCreate) {
    Qnn_ErrorHandle_t result = qnn_interface.deviceCreate(log_handle, all_device_configs.data(), &device_handle);
    if (QNN_SUCCESS != result) {
      return MAKE_EP_FAIL(("Failed to create device. Error: " + QnnErrorHandleToString(result)).c_str());
    }
  }
  device_created_flag = true;

  return Ort::Status();
}

Ort::Status QnnBackendManager::CreateDevice() {
  if (device_created_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Device initialized already.");
    return Ort::Status();
  }
  return CreateDeviceCommon(qnn_interface_, log_handle_,
                            device_handle_, device_created_,
                            /*allow_hw_device_enumeration=*/true);
}

Ort::Status QnnBackendManager::CreateValidatorDevice() {
  if (validator_device_created_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Validator device initialized already.");
    return Ort::Status();
  }
  // The validator backend is a host-side shim with no real hardware to enumerate,
  // so device_id_ / platform-info handling is skipped. soc_model_ / htp_arch_
  // configs still flow through so that backendValidateOpConfig validates against
  // the user's intended HTP arch instead of the shim's default.
  return CreateDeviceCommon(qnn_validator_interface_, validator_log_handle_,
                            validator_device_handle_, validator_device_created_,
                            /*allow_hw_device_enumeration=*/false);
}

Ort::Status QnnBackendManager::ReleaseDeviceCommon(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                   Qnn_DeviceHandle_t& device_handle,
                                                   bool& device_created_flag) {
  if (nullptr != qnn_interface.deviceFree) {
    Qnn_ErrorHandle_t result = qnn_interface.deviceFree(device_handle);
    if (QNN_SUCCESS != result) {
      return MAKE_EP_FAIL(("Failed to release device. Error: " + QnnErrorHandleToString(result)).c_str());
    }
  }

  device_created_flag = false;

  return Ort::Status();
}

Ort::Status QnnBackendManager::ReleaseDevice() {
  if (!device_created_) {
    return Ort::Status();
  }
  return ReleaseDeviceCommon(qnn_interface_, device_handle_, device_created_);
}

Ort::Status QnnBackendManager::ReleaseValidatorDevice() {
  if (!validator_device_created_) {
    return Ort::Status();
  }
  return ReleaseDeviceCommon(qnn_validator_interface_,
                             validator_device_handle_, validator_device_created_);
}

Ort::Status QnnBackendManager::InitializeProfiling() {
  profiling_level_merge_ = profiling_level_;
  // Only honor ETW-driven profile escalation when the app has explicitly opted into profiling.
  // If the app did not set profiling_level (default OFF), do not let ETW silently turn it on.
  if (profiling_level_ != ProfilingLevel::OFF &&
      profiling_level_etw_ != ProfilingLevel::INVALID &&
      profiling_level_etw_ > profiling_level_) {
    profiling_level_merge_ = profiling_level_etw_;
  }

  if (ProfilingLevel::OFF == profiling_level_merge_ || ProfilingLevel::INVALID == profiling_level_merge_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Profiling turned off.");
    return Ort::Status();
  }

  QnnProfile_Level_t qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
  bool enable_optrace = false;
  if (ProfilingLevel::BASIC == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_BASIC;
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to basic.");
  } else if (ProfilingLevel::DETAILED == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to detailed.");
  } else if (ProfilingLevel::OPTRACE == profiling_level_merge_) {
    qnn_profile_level = QNN_PROFILE_LEVEL_DETAILED;
    enable_optrace = true;
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Profiling level set to optrace.");
  }

  Qnn_ErrorHandle_t result = qnn_interface_.profileCreate(backend_handle_, qnn_profile_level, &profile_backend_handle_);
  RETURN_IF(QNN_PROFILE_NO_ERROR != result,
            ("Failed to create QNN profile! Error: " + QnnErrorHandleToString(result)).c_str());

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
  profiling_enabled_ = true;
  RETURN_IF_ERROR(LoadQnnSystemLib());

  if (enable_optrace) {
    QnnProfile_Config_t optrace_config = QNN_PROFILE_CONFIG_INIT;
    optrace_config.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
    optrace_config.enableOptrace = enable_optrace;

    const QnnProfile_Config_t* profile_configs[] = {&optrace_config, nullptr};
    result = qnn_interface_.profileSetConfig(profile_backend_handle_, profile_configs);

    RETURN_IF(QNN_PROFILE_NO_ERROR != result,
              ("Failed to enable op trace! Error: " + QnnErrorHandleToString(result)).c_str());
  }
#else
  if (enable_optrace) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_WARNING,
                    "Profiling level set to optrace, but QNN SDK Version is older than 2.29.0. "
                    "Profiling level will be set to detailed instead.");
  }
#endif

  return Ort::Status();
}

Ort::Status QnnBackendManager::ReleaseProfilehandle() {
  // Free Profiling object if it was created
  if (nullptr != profile_backend_handle_) {
    RETURN_IF(QNN_PROFILE_NO_ERROR != qnn_interface_.profileFree(profile_backend_handle_),
              "Could not free backend profile handle!");
  }
  profile_backend_handle_ = nullptr;

  return Ort::Status();
}

Ort::Status QnnBackendManager::SetProfilingLevelETW(ProfilingLevel profiling_level_etw_param) {
  if (profiling_level_etw_ != profiling_level_etw_param) {
    profiling_level_etw_ = profiling_level_etw_param;

    auto result = ReleaseProfilehandle();
    if (!result.IsOK()) {
      ORT_CXX_API_THROW("Failed to ReleaseProfilehandle for previous QNN profiling", ORT_EP_FAIL);
    }

    result = InitializeProfiling();
    if (!result.IsOK()) {
      ORT_CXX_API_THROW("Failed to Re-InitializeProfiling for QNN ETW profiling", ORT_EP_FAIL);
    }
  }
  return Ort::Status();
}

Ort::Status SetQnnContextConfig(ContextPriority context_priority, QnnContext_Config_t& qnn_context_config) {
  qnn_context_config.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
  switch (context_priority) {
    case ContextPriority::LOW: {
      qnn_context_config.priority = QNN_PRIORITY_LOW;
      break;
    }
    case ContextPriority::NORMAL_LOW: {
      qnn_context_config.priority = QNN_PRIORITY_NORMAL_LOW;
      break;
    }
    case ContextPriority::NORMAL: {
      qnn_context_config.priority = QNN_PRIORITY_NORMAL;
      break;
    }
    case ContextPriority::NORMAL_HIGH: {
      qnn_context_config.priority = QNN_PRIORITY_NORMAL_HIGH;
      break;
    }
    case ContextPriority::HIGH: {
      qnn_context_config.priority = QNN_PRIORITY_HIGH;
      break;
    }
    case ContextPriority::HIGH_PLUS: {
      qnn_context_config.priority = QNN_PRIORITY_HIGH_PLUS;
      break;
    }
    case ContextPriority::CRITICAL: {
      qnn_context_config.priority = QNN_PRIORITY_CRITICAL;
      break;
    }
    case ContextPriority::CRITICAL_PLUS: {
      qnn_context_config.priority = QNN_PRIORITY_CRITICAL_PLUS;
      break;
    }
    case ContextPriority::UNDEFINED: {
      return MAKE_EP_FAIL("Invalid Qnn context priority.");
    }
    default:
      qnn_context_config.priority = QNN_PRIORITY_NORMAL;
  }  // switch

  return Ort::Status();
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
// Callback required for allocating file mapping resources
static Qnn_ErrorHandle_t MapDmaDataCallback(Qnn_ContextBinaryDataRequest_t request,
                                            Qnn_ContextBinaryDmaDataResponse_t* response, void* notify_param) {
  if (notify_param == nullptr) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, "MapDmaDataCallback: notify_param is null");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  auto callback_info = reinterpret_cast<QnnBackendManager::FileMappingCallbackInfo_t*>(notify_param);

  if (callback_info->backend_manager == nullptr) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, "MapDmaDataCallback: QnnBackendManager is null");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  return callback_info->backend_manager->MapDmaData(request, response,
                                                    callback_info->mapped_file_ptr,
                                                    callback_info->file_size);
}

Qnn_ErrorHandle_t QnnBackendManager::MapDmaData(Qnn_ContextBinaryDataRequest_t request,
                                                Qnn_ContextBinaryDmaDataResponse_t* response,
                                                void* const mapped_base_ptr,
                                                const size_t file_size) {
  if (!file_mapped_weights_enabled_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_WARNING,
                    "Attempting to map DMA data but file mapping has been disabled, possibly due to an error in a previous request.");
    return QNN_CONTEXT_ERROR_ABORTED;
  }

  if (mapped_base_ptr == nullptr) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Attempting to map DMA data for null memory mapped base pointer");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }
  ORT_CXX_LOG_PTR(logger_ptr_,
                  ORT_LOGGING_LEVEL_INFO,
                  ("Mapping DMA data for request: memory mapped base pointer(" + utils::PtrToString(mapped_base_ptr) + "), offset(" + std::to_string(request.offset) + "), size(" + std::to_string(request.size) + "), total file size(" + std::to_string(file_size) + ") isBackendMappingNeeded(" + std::to_string(request.isBackendMappingNeeded) + ")").c_str());

  auto size = request.size;
  if (size == 0 || !request.isBackendMappingNeeded) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Mapping request size must be > 0 with backend mapping required");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  // offset & size are type uint64_t
  // Should never be an issue, but if this occurs then there is something inherently wrong with QNN
  if ((UINT64_MAX - request.offset) < size) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Critical error in QNN: mapping request offset + size will overflow 64 bits");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  // file_size will be promoted to 64 bits on 32-bit systems
  if ((request.offset + size) > file_size) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Requested offset and size includes memory outside of mapped file");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  void* unaligned_data_ptr = static_cast<char*>(mapped_base_ptr) + request.offset;
  rpcmem_library_->Api().register_buf(unaligned_data_ptr, size, NULL,
                                      rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY);

  auto fd = rpcmem_library_->Api().to_fd(unaligned_data_ptr);
  if (fd == -1) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, "Failed to register DMA data mapping to RPCMEM");
    return QNN_COMMON_ERROR_SYSTEM;
  }

  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, ("Created DMA data mapping with address: " + utils::PtrToString(unaligned_data_ptr)).c_str());

  response->dmaBuffer.fd = fd;
  response->dmaBuffer.data = unaligned_data_ptr;
  response->dataStartOffset = 0;
  response->alignedSize = size;

  return QNN_SUCCESS;
}

// Callback required for releasing file mapping resources
static Qnn_ErrorHandle_t ReleaseDmaDataCallback(Qnn_ContextBinaryDmaDataMem_t data_mem, void* notify_param) {
  if (notify_param == nullptr) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, "ReleaseDmaDataCallback: notify_param is null");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  auto callback_info = reinterpret_cast<QnnBackendManager::FileMappingCallbackInfo_t*>(notify_param);

  if (callback_info->backend_manager == nullptr) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, "ReleaseDmaDataCallback: QnnBackendManager is null");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  return callback_info->backend_manager->ReleaseDmaData(data_mem, callback_info->mapped_file_ptr);
}

// Use LOGS_DEFAULT here as this function will be called during destruction of QnnBackendManager
// At time of destruction, usage of logger_ will not be available and will result in a seg fault
Qnn_ErrorHandle_t QnnBackendManager::ReleaseDmaData(Qnn_ContextBinaryDmaDataMem_t data_mem,
                                                    void* mapped_base_ptr) {
  if (mapped_base_ptr == nullptr) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, ("Attempting to release DMA data for null memory mapped pointer"));
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
              ORT_LOGGING_LEVEL_INFO,
              ("Releasing DMA data mapping for memory mapped pointer(" + utils::PtrToString(mapped_base_ptr) + "), address(" + utils::PtrToString(data_mem.dmaBuffer.data) + "), size: (" + std::to_string(data_mem.memSize) + ")").c_str());

  if (data_mem.dmaBuffer.data == nullptr || data_mem.memSize == 0) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, "Mapping release request address must not be null and size must be > 0");
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  }

  // Deregister file mapped data from NPU regardless of file_mapped_weights_enabled_
  // as there may be file mapped data registered to the NPU prior to any mapping error
  void* unaligned_data_ptr = data_mem.dmaBuffer.data;
  rpcmem_library_->Api().register_buf(unaligned_data_ptr, data_mem.memSize, -1,
                                      rpcmem::RPCMEM_ATTR_IMPORT_BUFFER | rpcmem::RPCMEM_ATTR_READ_ONLY);

  auto fd = rpcmem_library_->Api().to_fd(unaligned_data_ptr);
  if (fd != -1) {
    ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, ("Failed to deregister buffer from RPCMEM: " + utils::PtrToString(unaligned_data_ptr)).c_str());
    return QNN_CONTEXT_ERROR_MEM_ALLOC;
  }
  return QNN_SUCCESS;
}
#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

// callback required to add context handles to class list
// when using contextCreateFromBinaryListAsync()
static void ContextCreateAsyncCallback(Qnn_ContextHandle_t context,
                                       Qnn_GraphHandle_t /* graph */,
                                       const char* /* graph_name */,
                                       QnnContext_createFromBinaryAsyncNotifyType_t /* notify_type */,
                                       void* notify_param,
                                       Qnn_ErrorHandle_t /* status */) {
  auto qnn_backend_manager = SharedContext::GetInstance().GetSharedQnnBackendManager();

  if (context) {
    qnn_backend_manager->ProcessContextFromBinListAsync(context, notify_param);
  }
}

void QnnBackendManager::ProcessContextFromBinListAsync(Qnn_ContextHandle_t context, void* notifyParam) {
  std::ostringstream context_ss;
  context_ss << context;

  std::lock_guard<std::mutex> guard(ep_context_handle_map_mutex_);
  if (!notifyParam) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_WARNING,
                    ("No known node names associated with context handle: " + context_ss.str()).c_str());
    return;
  }

  std::vector<std::string>* ep_node_names = reinterpret_cast<std::vector<std::string>*>(notifyParam);
  for (const auto& node_name : *ep_node_names) {
    if (!(ep_context_handle_map_.emplace(node_name, context).second)) {
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_VERBOSE,
                      ("Unable to map " + context_ss.str() + " to " + node_name).c_str());
    }
  }

  auto s = AddQnnContextHandle(context);
  if (!s.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_WARNING, ("Unable to add context " + context_ss.str()).c_str());
  } else {
    context_created_ = true;
  }
}

Ort::Status QnnBackendManager::GetFileSizeIfValid(const std::string& filepath,
                                                  size_t& file_size) {
  std::error_code ec;
  RETURN_IF(!std::filesystem::exists(filepath, ec), ("Context binary does not exist: " + filepath).c_str());
  RETURN_IF(ec, ("Failed to read file: " + filepath + ", error: " + ec.message()).c_str());

  auto size = std::filesystem::file_size(filepath, ec);
  RETURN_IF(ec, ("Failed to retrieve size of file: " + filepath + ", error: " + ec.message()).c_str());

  RETURN_IF(size == 0, ("File is empty: " + filepath).c_str());
  RETURN_IF(size > SIZE_MAX, ("File (" + filepath + ") file size (" + std::to_string(size) + " bytes) exceeds maximum value of size_t for this platform (" + std::to_string(SIZE_MAX) + " bytes).").c_str());

  file_size = static_cast<size_t>(size);
  return Ort::Status();
}

Ort::Status QnnBackendManager::ReadContextBinIfValid(const std::string& context_bin_filepath,
                                                     std::vector<char>& buffer,
                                                     const qnn::EpContextIoDispatch& io_dispatch) {
  if (io_dispatch.HasReadCallback()) {
    const std::string filename = std::filesystem::path(context_bin_filepath).filename().string();
    RETURN_IF_ERROR(io_dispatch.Read(filename, buffer));
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO,
                    ("EPContext binary decrypted via App read callback: " + filename +
                     " (" + std::to_string(buffer.size()) + " bytes)")
                        .c_str());
    return Ort::Status();
  }

  size_t buffer_size;
  RETURN_IF_ERROR(GetFileSizeIfValid(context_bin_filepath, buffer_size));

  buffer.resize(buffer_size);

  std::ifstream cache_file(context_bin_filepath.c_str(), std::ifstream::binary);
  RETURN_IF(!cache_file || !cache_file.good(), ("Failed to read context binary from: " + context_bin_filepath).c_str());

  const auto& read_result = cache_file.read(buffer.data(), buffer_size);
  RETURN_IF(!read_result, "Failed to read contents from cached context file.");

  return Ort::Status();
}

Ort::Status QnnBackendManager::BuildContextBinaryConfigs(
    int64_t max_spill_fill_size,
    Qnn_ContextHandle_t first_group_handle,
    QnnConfigsBuilder<QnnContext_Config_t, QnnHtpContext_CustomConfig_t>& configs_builder) {
  gsl::not_null<QnnContext_Config_t*> priority_cfg = configs_builder.PushConfig();
  RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, *priority_cfg));

  if (max_spill_fill_size > 0) {
    gsl::not_null<QnnHtpContext_CustomConfig_t*> spill_custom = configs_builder.PushCustomConfig();
    spill_custom->option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    spill_custom->groupRegistration.firstGroupHandle = first_group_handle;
    spill_custom->groupRegistration.maxSpillFillBuffer = max_spill_fill_size;

    gsl::not_null<QnnContext_Config_t*> spill_cfg = configs_builder.PushConfig();
    spill_cfg->option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    spill_cfg->customConfig = spill_custom;
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::CreateContextHandleFromBinary(
    void* bin_buffer,
    uint64_t buffer_length,
    bool use_file_mapping,
    const QnnContext_Config_t** context_configs,
    const std::string& context_bin_filepath,
    const qnn::EpContextIoDispatch& io_dispatch,
    Qnn_ContextHandle_t& context) {
  RETURN_IF(nullptr == qnn_interface_.contextCreateFromBinary,
            "Invalid function pointer for contextCreateFromBinary.");

  Qnn_ErrorHandle_t rt = QNN_SUCCESS;

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  if (use_file_mapping && file_mapper_) {
    RETURN_IF(nullptr == qnn_interface_.contextCreateFromBinaryWithCallback,
              "Invalid function pointer for contextCreateFromBinaryWithCallback.");

    auto notify_param_ptr = std::make_unique<FileMappingCallbackInfo_t>(bin_buffer, buffer_length, this);

    Qnn_ContextBinaryCallback_t callbacks;
    callbacks.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
    callbacks.dmaBufferCallback.version = QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
    callbacks.dmaBufferCallback.v1.dataProvide = MapDmaDataCallback;
    callbacks.dmaBufferCallback.v1.dataRelease = ReleaseDmaDataCallback;
    callbacks.dmaBufferCallback.v1.notifyParam = reinterpret_cast<void*>(notify_param_ptr.get());

    file_mapping_notify_params_.push_back(std::move(notify_param_ptr));

    rt = qnn_interface_.contextCreateFromBinaryWithCallback(backend_handle_,
                                                            device_handle_,
                                                            context_configs,
                                                            &callbacks,
                                                            bin_buffer,
                                                            static_cast<Qnn_ContextBinarySize_t>(buffer_length),
                                                            &context,
                                                            profile_backend_handle_,
                                                            NULL);
    if (rt != QNN_SUCCESS) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_WARNING,
                      ("contextCreateFromBinaryWithCallback failed (" + QnnErrorHandleToString(rt) +
                       "). Retrying with direct read.")
                          .c_str());
    }
  }
#endif

  if (!use_file_mapping || rt != QNN_SUCCESS) {
    // Read from file if no pre-read buffer was supplied or if file mapping failed.
    std::vector<char> file_buffer;
    if (bin_buffer == nullptr || rt != QNN_SUCCESS) {
      RETURN_IF_ERROR(ReadContextBinIfValid(context_bin_filepath, file_buffer, io_dispatch));
      bin_buffer = static_cast<void*>(file_buffer.data());
      buffer_length = static_cast<uint64_t>(file_buffer.size());
    }

    rt = qnn_interface_.contextCreateFromBinary(backend_handle_,
                                                device_handle_,
                                                context_configs,
                                                bin_buffer,
                                                static_cast<Qnn_ContextBinarySize_t>(buffer_length),
                                                &context,
                                                profile_backend_handle_);
  }

  RETURN_IF(QNN_SUCCESS != rt,
            ("contextCreateFromBinary failed. Error: " + QnnErrorHandleToString(rt)).c_str());
  return Ort::Status();
}

Ort::Status QnnBackendManager::ReloadContextForSSR(const std::string& context_bin_filepath,
                                                   int64_t max_spill_fill_size,
                                                   Qnn_ContextHandle_t& new_context,
                                                   const qnn::EpContextIoDispatch& io_dispatch) {
  QnnConfigsBuilder<QnnContext_Config_t, QnnHtpContext_CustomConfig_t> configs_builder(
      QNN_CONTEXT_CONFIG_INIT, QnnHtpContext_CustomConfig_t{});
  // SSR always starts a new spill-fill group (firstGroupHandle = 0x0): all prior handles are invalid.
  RETURN_IF_ERROR(BuildContextBinaryConfigs(max_spill_fill_size, 0x0, configs_builder));

  void* bin_buffer = nullptr;
  uint64_t buffer_length = 0;
  bool use_file_mapping = false;

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  // Attempt file mapping first if enabled (matches initial load behavior).
  if (file_mapped_weights_enabled_ && file_mapper_) {
    RETURN_IF_ERROR(GetFileSizeIfValid(context_bin_filepath, buffer_length));
    RETURN_IF_ERROR(file_mapper_->GetContextBinMappedMemoryPtr(context_bin_filepath, &bin_buffer));

    // Cannot use contextCreateFromBinaryWithCallback() unless context bin version is >= 3.3.3
    auto sys_ctx_handle = GetSystemContextHandle();
    RETURN_IF(sys_ctx_handle == nullptr, "System context handle is null.");
    Qnn_Version_t blob_version = {0, 0, 0};
    uint32_t graph_count = 0;
    QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
    RETURN_IF_ERROR(GetGraphInfoAndBinVersion(sys_ctx_handle.get(),
                                              bin_buffer,
                                              static_cast<Qnn_ContextBinarySize_t>(buffer_length),
                                              blob_version,
                                              graph_count,
                                              &graphs_info));
    if (!MinVersionMet(blob_version, {3, 3, 3})) {
      std::string version_str = std::to_string(blob_version.major) + "." +
                                std::to_string(blob_version.minor) + "." +
                                std::to_string(blob_version.patch);
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_WARNING,
                      ("SSR recovery: context binary version " + version_str +
                       " < 3.3.3, file mapping not supported. Falling back to direct read.")
                          .c_str());
    } else {
      use_file_mapping = true;
    }
  }
#endif

  RETURN_IF_ERROR(CreateContextHandleFromBinary(bin_buffer, buffer_length, use_file_mapping,
                                                configs_builder.GetQnnConfigs(),
                                                context_bin_filepath, io_dispatch, new_context));
  RETURN_IF_ERROR(AddQnnContextHandle(new_context));
  return Ort::Status();
}

Ort::Status QnnBackendManager::CreateContextVtcmBackupBufferSharingEnabled(
    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>>& context_bin_map,
    const qnn::EpContextIoDispatch& io_dispatch) {
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  QnnContext_Config_t context_config_resource_sharing = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t resource_sharing_custom_config;
  resource_sharing_custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES;
  resource_sharing_custom_config.shareResources = true;
  context_config_resource_sharing.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_resource_sharing.customConfig = &resource_sharing_custom_config;

  // Set share resource optimization type to SEQUENTIAL_WITHOUT_VA_OPTIMIZATION
  // This is controlled by the htp_share_resource_optimization option
  QnnHtpContext_CustomConfig_t context_config_resource_sharing_opt_type;
  context_config_resource_sharing_opt_type.option = QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE;
  context_config_resource_sharing_opt_type.shareResOptType = SEQUENTIAL_WITHOUT_VA_OPTIMIZATION;
  QnnContext_Config_t resource_sharing_opt_type_config;
  resource_sharing_opt_type_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  resource_sharing_opt_type_config.customConfig = &context_config_resource_sharing_opt_type;

  QnnContext_Config_t context_config_weight_sharing = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t custom_config;
  custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
  custom_config.weightSharingEnabled = true;
  context_config_weight_sharing.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_weight_sharing.customConfig = &custom_config;
#else
  ORT_CXX_LOG_PTR(logger_ptr_,
                  ORT_LOGGING_LEVEL_WARNING,
                  "Called CreateContextVtcmBackupBufferSharingEnabled() but QNN API version is older than 2.26!");
#endif
  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, context_priority_config));

  std::vector<const QnnContext_Config_t*> configs_vec;
  configs_vec.push_back(&context_priority_config);
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  configs_vec.push_back(&context_config_resource_sharing);
  configs_vec.push_back(&resource_sharing_opt_type_config);
  configs_vec.push_back(&context_config_weight_sharing);
#endif
  configs_vec.push_back(nullptr);

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  if (file_mapped_weights_enabled_ && file_mapper_) {
    // Retry logic -- if context creation failed with file mapped weights, then retry with feature disabled
    auto res = CreateContextFromListAsyncWithCallback(configs_vec.data(), context_bin_map);
    if (!res.IsOK()) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_WARNING, (res.GetErrorMessage() + ". Retrying with feature disabled.").c_str());
    } else {
      return Ort::Status();
    }
  }
#endif
  return CreateContextFromListAsync(configs_vec.data(), context_bin_map, io_dispatch);
}

Ort::Status QnnBackendManager::CreateContextFromListAsync(const QnnContext_Config_t** configs,
                                                          std::unordered_map<std::string,
                                                                             std::unique_ptr<std::vector<std::string>>>& context_bin_map,
                                                          const qnn::EpContextIoDispatch& io_dispatch) {
  std::vector<QnnContext_Params_t> context_params_list;
  std::vector<QnnContext_ParamsV1_t> context_paramsv1_list;
  std::vector<const QnnContext_Params_t*> context_params_ptr_list;
  std::vector<std::vector<char>> buffer_list;

  context_params_list.reserve(context_bin_map.size());
  context_params_ptr_list.reserve(context_bin_map.size() + 1);

  for (auto& it : context_bin_map) {
    auto context_bin_filepath = it.first;

    std::vector<char> buffer;
    RETURN_IF_ERROR(ReadContextBinIfValid(context_bin_filepath, buffer, io_dispatch));

    size_t buffer_size = buffer.size();
    buffer_list.push_back(std::move(buffer));

    QnnContext_ParamsV1_t context_params_v1 = {nullptr,
                                               buffer_list.back().data(),
                                               buffer_size,
                                               nullptr,
                                               ContextCreateAsyncCallback,
                                               it.second.get()};

    QnnContext_Params_t context_params = {QnnContext_ParamsVersion_t::QNN_CONTEXT_PARAMS_VERSION_1,
                                          {context_params_v1}};

    context_params_list.push_back(std::move(context_params));
    context_paramsv1_list.push_back(std::move(context_params_v1));
    context_params_ptr_list.push_back(&context_params_list.back());
  }
  context_params_ptr_list.push_back(nullptr);
  auto result = qnn_interface_.contextCreateFromBinaryListAsync(backend_handle_,
                                                                device_handle_,
                                                                context_params_ptr_list.data(),
                                                                configs,
                                                                nullptr);

  RETURN_IF(QNN_CONTEXT_NO_ERROR != result, ("Failed to create context. Error: " + QnnErrorHandleToString(result) + ", Code:" + std::to_string(result)).c_str());
  return Ort::Status();
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
Ort::Status QnnBackendManager::CreateContextFromListAsyncWithCallback(const QnnContext_Config_t** configs,
                                                                      std::unordered_map<std::string,
                                                                                         std::unique_ptr<std::vector<std::string>>>& context_bin_map) {
  std::vector<QnnContext_Params_t> context_params_list;
  std::vector<QnnContext_ParamsV2_t> context_paramsv2_list;
  std::vector<Qnn_ContextBinaryCallback_t> context_callbacks_list;
  std::vector<const QnnContext_Params_t*> context_params_ptr_list;

  context_params_list.reserve(context_bin_map.size());
  context_paramsv2_list.reserve(context_bin_map.size());
  context_callbacks_list.reserve(context_bin_map.size());
  context_params_ptr_list.reserve(context_bin_map.size() + 1);

  for (auto& it : context_bin_map) {
    auto context_bin_filepath = it.first;

    size_t buffer_size;
    RETURN_IF_ERROR(GetFileSizeIfValid(context_bin_filepath, buffer_size));

    void* buffer;
    RETURN_IF_ERROR(file_mapper_->GetContextBinMappedMemoryPtr(context_bin_filepath, &buffer));

    auto sys_ctx_handle = GetSystemContextHandle();
    RETURN_IF(sys_ctx_handle == nullptr, "System context handle is null.");

    uint32_t graph_count = 0;
    QnnSystemContext_GraphInfo_t* graphs_info = nullptr;

    Qnn_Version_t blob_version = {0, 0, 0};
    RETURN_IF_ERROR(GetGraphInfoAndBinVersion(sys_ctx_handle.get(),
                                              buffer,
                                              static_cast<Qnn_ContextBinarySize_t>(buffer_size),
                                              blob_version,
                                              graph_count,
                                              &graphs_info));

    // Cannot use contextCreateFromBinaryWithCallback() unless context bin version is >= 3.3.3
    if (!MinVersionMet(blob_version, {3, 3, 3})) {
      std::string version_str(std::to_string(blob_version.major) + "." + std::to_string(blob_version.minor) + "." + std::to_string(blob_version.patch));
      return MAKE_EP_FAIL(("Context binary of " + context_bin_filepath + " is v" + version_str + ". File mapping is only supported for versions >= 3.3.3. Disabling file mapping for this node.").c_str());
    }

    auto notify_param_ptr = std::make_unique<FileMappingCallbackInfo_t>(buffer, buffer_size, this);

    Qnn_ContextBinaryCallback_t context_file_map_callbacks;
    context_file_map_callbacks.type = QNN_CONTEXT_CALLBACK_DMA_BUFFER;
    context_file_map_callbacks.dmaBufferCallback.version = QNN_CONTEXT_CALLBACK_DMA_BUFFER_VERSION_1;
    context_file_map_callbacks.dmaBufferCallback.v1.dataProvide = MapDmaDataCallback;
    context_file_map_callbacks.dmaBufferCallback.v1.dataRelease = ReleaseDmaDataCallback;
    context_file_map_callbacks.dmaBufferCallback.v1.notifyParam = reinterpret_cast<void*>(notify_param_ptr.get());

    file_mapping_notify_params_.push_back(std::move(notify_param_ptr));
    context_callbacks_list.push_back(std::move(context_file_map_callbacks));

    // Callbacks require QnnContext_ParamsV2_t which is new to QNN API 2.32
    QnnContext_ParamsV2_t context_params_v2 = {nullptr,
                                               buffer,
                                               buffer_size,
                                               nullptr,
                                               ContextCreateAsyncCallback,
                                               it.second.get(),
                                               &context_callbacks_list.back()};

    QnnContext_Params_t context_params = {QnnContext_ParamsVersion_t::QNN_CONTEXT_PARAMS_VERSION_2,
                                          {}};

    context_paramsv2_list.push_back(std::move(context_params_v2));

    context_params.v2 = &context_paramsv2_list.back();
    context_params_list.push_back(std::move(context_params));
    context_params_ptr_list.push_back(&(context_params_list.back()));
  }
  context_params_ptr_list.push_back(nullptr);
  auto result = qnn_interface_.contextCreateFromBinaryListAsync(backend_handle_,
                                                                device_handle_,
                                                                context_params_ptr_list.data(),
                                                                configs,
                                                                nullptr);

  RETURN_IF(QNN_CONTEXT_NO_ERROR != result, ("Failed to create context with file mapping enabled. Error: " +
                                             QnnErrorHandleToString(result) + ", Code:" + std::to_string(result))
                                                .c_str());
  return Ort::Status();
}
#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

Ort::Status QnnBackendManager::SetContextPriority(ContextPriority context_priority) {
  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  RETURN_IF_ERROR(SetQnnContextConfig(context_priority, context_priority_config));

  QnnContext_Config_t* configs[] = {&context_priority_config, nullptr};
  for (const auto& context_handle : contexts_) {
    auto result = qnn_interface_.contextSetConfig(context_handle, (const QnnContext_Config_t**)configs);
    RETURN_IF(QNN_CONTEXT_NO_ERROR != result, "Failed to set context priority for context handle.");
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::ResetContextPriority() {
  return SetContextPriority(context_priority_);
}

Ort::Status QnnBackendManager::CreateContext(bool enable_htp_weight_sharing,
                                             bool enable_htp_extended_udma_mode,
                                             bool enable_htp_prepare_only,
                                             bool enable_htp_ref_weight_sharing,
                                             bool enable_htp_graph_splitting) {
  if (true == context_created_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO, "Context created already.");
    return Ort::Status();
  }

  QnnContext_Config_t context_config_weight_sharing = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t custom_config;
  custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
  custom_config.weightSharingEnabled = enable_htp_weight_sharing;
  context_config_weight_sharing.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_weight_sharing.customConfig = &custom_config;

  QnnContext_Config_t context_priority_config = QNN_CONTEXT_CONFIG_INIT;
  RETURN_IF_ERROR(SetQnnContextConfig(context_priority_, context_priority_config));

  QnnContext_Config_t context_config_extended_udma = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t udma_custom_config;
  udma_custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA;
  udma_custom_config.useExtendedUdma = enable_htp_extended_udma_mode;
  context_config_extended_udma.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_extended_udma.customConfig = &udma_custom_config;

  QnnContext_Config_t context_config_prepare_only = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t prepare_only_custom_config;
  prepare_only_custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY;
  prepare_only_custom_config.isPrepareOnly = enable_htp_prepare_only;
  context_config_prepare_only.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  context_config_prepare_only.customConfig = &prepare_only_custom_config;

  QnnContext_Config_t context_config_ref_weight_sharing = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t ref_weight_sharing_custom_config;
#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 33
  if (core_api_version_.major == 2 && core_api_version_.minor >= 33) {
    ref_weight_sharing_custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REFERENCE_WEIGHT_SHARING_ENABLED;
    ref_weight_sharing_custom_config.referenceWeightSharingEnabled = enable_htp_ref_weight_sharing;
    context_config_ref_weight_sharing.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    context_config_ref_weight_sharing.customConfig = &ref_weight_sharing_custom_config;
  } else if (enable_htp_ref_weight_sharing) {
#else
  if (enable_htp_ref_weight_sharing) {
#endif
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_WARNING,
                    "HTP reference weight sharing is only supported in QAIRT 2.44+ SDK.");
    enable_htp_ref_weight_sharing = false;
  }

#ifdef QNN_HTP_GRAPH_SPLITTING_AVAILABLE
  QnnContext_Config_t context_config_graph_splitting = QNN_CONTEXT_CONFIG_INIT;
  QnnHtpContext_CustomConfig_t graph_splitting_custom_config;
  if (enable_htp_graph_splitting) {
    graph_splitting_custom_config.option = QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_CONFIGS;
    graph_splitting_custom_config.graphSplittingConfigs.graphSplittingEnabled = true;
    context_config_graph_splitting.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    context_config_graph_splitting.customConfig = &graph_splitting_custom_config;
  }
#else
  ORT_UNUSED_PARAMETER(enable_htp_graph_splitting);
#endif

  std::vector<const QnnContext_Config_t*> npu_context_configs_vec;
  npu_context_configs_vec.push_back(&context_priority_config);
  npu_context_configs_vec.push_back(&context_config_weight_sharing);
  npu_context_configs_vec.push_back(&context_config_extended_udma);
  npu_context_configs_vec.push_back(&context_config_prepare_only);
  if (enable_htp_ref_weight_sharing) {
    npu_context_configs_vec.push_back(&context_config_ref_weight_sharing);
  }
#ifdef QNN_HTP_GRAPH_SPLITTING_AVAILABLE
  if (enable_htp_graph_splitting) {
    npu_context_configs_vec.push_back(&context_config_graph_splitting);
  }
#endif
  npu_context_configs_vec.push_back(nullptr);

  const QnnContext_Config_t* empty_context_configs[] = {nullptr};

  const QnnContext_Config_t** configs = nullptr;
  switch (GetQnnBackendType()) {
    case QnnBackendType::HTP:
    case QnnBackendType::DSP:
      configs = npu_context_configs_vec.data();
      break;
    case QnnBackendType::GPU:
    case QnnBackendType::SERIALIZER:
      configs = nullptr;
      break;
    default:
      configs = empty_context_configs;
      break;
  }

  // Not all serialization backends allow for hardware configs to be applied.
  if (qnn_serializer_config_ && !qnn_serializer_config_->SupportsArbitraryGraphConfigs()) {
    configs = nullptr;
  }

  Qnn_ContextHandle_t context = nullptr;
  Qnn_ErrorHandle_t result = 0;

  result = qnn_interface_.contextCreate(backend_handle_,
                                        device_handle_,
                                        configs,
                                        &context);

  RETURN_IF(QNN_CONTEXT_NO_ERROR != result,
            ("Failed to create context. Error: " + QnnErrorHandleToString(result)).c_str());

  RETURN_IF_ERROR(AddQnnContextHandle(context));

  context_created_ = true;
  return Ort::Status();
}

Ort::Status QnnBackendManager::ReleaseContext() {
  if (false == context_created_) {
    return Ort::Status();
  }

  // release QNN context handles
  contexts_.clear();
  context_map_.clear();
  ep_context_handle_map_.clear();

  context_created_ = false;
  return Ort::Status();
}

Ort::Status QnnBackendManager::GetContextBinaryBuffer(bool is_multi_soc_buffer,
                                                      unsigned char** context_buffer,
                                                      uint64_t& buffer_size) {
  RETURN_IF(context_buffer == nullptr, "Null context_buffer pointer provided.");

  if (is_multi_soc_buffer) {
    RETURN_IF(system_dlc_plugin_ == nullptr, "Unable to get multi-SoC binary without system DLC.");
    return system_dlc_plugin_->GetDlcBinaryBuffer(context_buffer, buffer_size);
  }

  RETURN_IF(qnn_interface_.contextGetBinarySize == nullptr || qnn_interface_.contextGetBinary == nullptr,
            "Failed to get valid function pointers.");
  RETURN_IF(contexts_.size() <= 0, "No QNN context to get context binary from.");

  uint64_t required_buffer_size = 0;
  // Generate all graphs in one single context
  Qnn_ErrorHandle_t rt = qnn_interface_.contextGetBinarySize(contexts_[0], &required_buffer_size);
  RETURN_IF(rt != QNN_CONTEXT_NO_ERROR,
            ("Failed to get QNN context binary size. Error: " + QnnErrorHandleToString(rt)).c_str());

  auto buffer = std::make_unique<unsigned char[]>(required_buffer_size);
  RETURN_IF(buffer == nullptr, "Failed to allocate buffer for context binary.");

  uint64_t written_buffer_size = 0;
  rt = qnn_interface_.contextGetBinary(contexts_[0],
                                       reinterpret_cast<void*>(buffer.get()),
                                       required_buffer_size,
                                       &written_buffer_size);
  RETURN_IF(rt != QNN_CONTEXT_NO_ERROR,
            ("Failed to get QNN context binary. Error: " + QnnErrorHandleToString(rt)).c_str());
  RETURN_IF(required_buffer_size < written_buffer_size,
            ("Context written buffer size: " + std::to_string(written_buffer_size) +
             " exceeds allocated buffer size: " + std::to_string(required_buffer_size))
                .c_str());

  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Get context binary buffer succeed.");
  *context_buffer = buffer.release();
  buffer_size = written_buffer_size;

  return Ort::Status();
}

Ort::Status QnnBackendManager::GetMaxSpillFillBufferSize(unsigned char* buffer,
                                                         uint64_t buffer_length,
                                                         bool is_multi_soc_buffer,
                                                         uint64_t& max_spill_fill_buffer_size) {
  if (is_multi_soc_buffer) {
    RETURN_IF(system_dlc_plugin_ == nullptr,
              "Unable to get max spill-fill buffer size for multi-SoC binary without system DLC.");
    return system_dlc_plugin_->GetDlcMaxSpillFillBufferSize(max_spill_fill_buffer_size);
  }

  max_spill_fill_buffer_size = 0;
  // spill fill starts from 2.28
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 21)
  auto sys_ctx_handle = GetSystemContextHandle();
  RETURN_IF(sys_ctx_handle == nullptr, "System context handle is null.");

  Qnn_Version_t blob_version = {0, 0, 0};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
  RETURN_IF_ERROR(GetGraphInfoAndBinVersion(sys_ctx_handle.get(),
                                            static_cast<void*>(buffer),
                                            static_cast<Qnn_ContextBinarySize_t>(buffer_length),
                                            blob_version,
                                            graph_count,
                                            &graphs_info));

  for (uint32_t i = 0; i < graph_count; ++i) {
    if (graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
      auto htp_graph_info = reinterpret_cast<QnnHtpSystemContext_GraphBlobInfo_t*>(graphs_info[i].graphInfoV3.graphBlobInfo);
      if (htp_graph_info->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1) {
        auto spill_fill_buffer_size = htp_graph_info->contextBinaryGraphBlobInfoV1.spillFillBufferSize;
        max_spill_fill_buffer_size = spill_fill_buffer_size > max_spill_fill_buffer_size ? spill_fill_buffer_size : max_spill_fill_buffer_size;
      } else {
        ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Unknown context binary graph info blob version.");
      }
    } else if (graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 ||
               graphs_info[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_VERBOSE,
                      "Skip retrieve spill file buffer size, it is not supported with graph info v1 & v2.");
    } else {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Unknown context binary graph info version.");
    }
  }
#else
  ORT_UNUSED_PARAMETER(buffer);
  ORT_UNUSED_PARAMETER(buffer_length);
#endif

  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Get max spill fill buffer size completed.");
  return Ort::Status();
}

Ort::Status QnnBackendManager::LoadCachedQnnContextFromBuffer(
    char* buffer,
    uint64_t buffer_length,
    const std::string& context_bin_filepath,
    std::string node_name,
    std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>& qnn_models,
    int64_t max_spill_fill_size,
    const qnn::EpContextIoDispatch& io_dispatch,
    bool is_multi_soc_buffer) {
  bool result = nullptr == qnn_sys_interface_.systemContextCreate ||
                nullptr == qnn_sys_interface_.systemContextGetBinaryInfo ||
                nullptr == qnn_sys_interface_.systemContextFree;
  RETURN_IF(result, "Failed to get valid function pointer.");

  void* bin_buffer = nullptr;
  bool use_file_mapping = file_mapped_weights_enabled_;
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  // A nonzero buffer length implies an embedded context
  if (use_file_mapping && buffer_length == 0) {
    RETURN_IF(!file_mapper_, "Attempting to use File Mapping feature but file_mapper_ is uninitialized");

    RETURN_IF_ERROR(GetFileSizeIfValid(context_bin_filepath, buffer_length));

    RETURN_IF(buffer_length == 0, ("Context bin has a size of 0 bytes: " + context_bin_filepath).c_str());
    RETURN_IF_ERROR(file_mapper_->GetContextBinMappedMemoryPtr(context_bin_filepath, &bin_buffer));

  } else {
    if (use_file_mapping) {
      use_file_mapping = false;
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_WARNING,
                      ("Node " + node_name + " is using an embedded cache. Disabling file mapping for this node.").c_str());
    }
    RETURN_IF(buffer == nullptr, "Attempting to load QNN context from buffer but buffer is null");
    bin_buffer = static_cast<void*>(buffer);
  }
#else
  bin_buffer = static_cast<void*>(buffer);
#endif

  auto sys_ctx_handle = GetSystemContextHandle();
  RETURN_IF(sys_ctx_handle == nullptr, "System context handle is null.");

  Qnn_Version_t blob_version = {0, 0, 0};
  uint32_t graph_count = 0;
  QnnSystemContext_GraphInfo_t* graphs_info = nullptr;
  if (!is_multi_soc_buffer) {
    RETURN_IF_ERROR(GetGraphInfoAndBinVersion(sys_ctx_handle.get(),
                                              bin_buffer,
                                              static_cast<Qnn_ContextBinarySize_t>(buffer_length),
                                              blob_version,
                                              graph_count,
                                              &graphs_info));
  } else {
    // `CreateSystemDlcPlugin` is not suitable here as it creates an empty DLC.
    // Instead, `QnnBackendSystemDlcPlugin.GetDlcBinaryInfo` creates DLC from binary.
    auto system_dlc_plugin = std::make_unique<QnnBackendSystemDlcPlugin>(this);
    RETURN_IF_ERROR(system_dlc_plugin->GetDlcBinaryInfo(sys_ctx_handle.get(),
                                                        static_cast<const uint8_t*>(bin_buffer),
                                                        buffer_length,
                                                        blob_version,
                                                        graph_count,
                                                        &graphs_info));
  }

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  // Cannot use contextCreateFromBinaryWithCallback() unless context bin version is >= 3.3.3
  if (use_file_mapping && !MinVersionMet(blob_version, {3, 3, 3})) {
    std::string version_str(std::to_string(blob_version.major) + "." + std::to_string(blob_version.minor) + "." + std::to_string(blob_version.patch));
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_WARNING,
                    ("Context binary of " + node_name + " is " + version_str + ". File mapping is only supported for versions >= 3.3.3. Disabling file mapping for this node.").c_str());
    use_file_mapping = false;
  }
#endif

  RETURN_IF(graph_count < 1 || graphs_info == nullptr, "Failed to get graph info from Qnn cached context.");

  ORT_CXX_LOG_PTR(logger_ptr_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Graph count from QNN context: " + std::to_string(graph_count)).c_str());

  Qnn_ContextHandle_t context = nullptr;
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  if (htp_share_resource_optimization_ == 1) {
    if (ep_context_handle_map_.find(node_name) != ep_context_handle_map_.end()) {
      context = ep_context_handle_map_.at(node_name);
    }
    RETURN_IF(nullptr == context, ("Failed to retrieve context for " + node_name).c_str());

  } else {
#endif
    // Join the existing spill-fill group if one exists; otherwise start a new one.
    size_t current_contexts_size = GetQnnContextSize();
    Qnn_ContextHandle_t first_group_handle =
        (max_spill_fill_size > 0 && current_contexts_size > 0) ? GetQnnContext(0) : 0x0;
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE,
                    ("Max spill fill buffer size: " + std::to_string(max_spill_fill_size)).c_str());

    QnnConfigsBuilder<QnnContext_Config_t, QnnHtpContext_CustomConfig_t> configs_builder(
        QNN_CONTEXT_CONFIG_INIT, QnnHtpContext_CustomConfig_t{});
    RETURN_IF_ERROR(BuildContextBinaryConfigs(max_spill_fill_size, first_group_handle, configs_builder));

    qnn::profile::ProfilingInfo profiling_info;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (ProfilingEnabled()) {
      profiling_info.start_time = qnn::utils::GetTimeStampInUs();
    }
#endif

    RETURN_IF_ERROR(CreateContextHandleFromBinary(bin_buffer, buffer_length, use_file_mapping,
                                                  configs_builder.GetQnnConfigs(), context_bin_filepath,
                                                  io_dispatch, context));

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    if (ProfilingEnabled()) {
      profiling_info.stop_time = qnn::utils::GetTimeStampInUs();
      profiling_info.method_type = ProfilingMethodType::CREATE_FROM_BINARY;
      profiling_info.graph_name = node_name;
    }
#endif

    RETURN_IF_ERROR(AddQnnContextHandle(context));
    RETURN_IF_ERROR(ExtractBackendProfilingInfo(profiling_info));

#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 26)
  }
#endif

  if (1 == graph_count) {
    // in case the EPContext node is generated from script
    // the graph name from the context binary may not match the EPContext node name
    auto qnn_model = std::make_unique<qnn::QnnModel>(this, api_ptrs_);
    RETURN_IF_ERROR(qnn_model->DeserializeGraphInfoFromBinaryInfo(graphs_info[0], context));
    qnn_models.emplace(node_name, std::move(qnn_model));
  } else {
    for (uint32_t i = 0; i < graph_count; ++i) {
      auto qnn_model = std::make_unique<qnn::QnnModel>(this, api_ptrs_);
      RETURN_IF_ERROR(qnn_model->DeserializeGraphInfoFromBinaryInfo(graphs_info[i], context));
      qnn_models.emplace(graphs_info[i].graphInfoV1.graphName, std::move(qnn_model));
    }
  }

  // Seed recovery info for embed_mode=0 so ExecuteGraph can reload after SSR.
  if (!context_bin_filepath.empty()) {
    for (auto& [name, model] : qnn_models) {
      model->SetContextRecoveryInfo(context_bin_filepath, max_spill_fill_size, context_priority_);
    }
  }

  context_created_ = true;

  ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Load from cached QNN Context completed.");
  return Ort::Status();
}

// need to load system lib if load from Qnn context binary
// or generate Qnn context binary is enabled -- to get the max spill fill buffer size
Ort::Status QnnBackendManager::SetupBackend(
    bool load_from_cached_context,
    bool need_load_system_lib,
    bool share_ep_contexts,
    int htp_share_resource_optimization,
    bool enable_file_mapped_weights,
    std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library,
    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>>& context_bin_map,
    const qnn::EpContextIoDispatch& io_dispatch,
    bool enable_htp_extended_udma_mode,
    bool enable_htp_prepare_only,
    bool enable_htp_graph_splitting) {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);
  if (backend_setup_completed_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Backend setup already!");

    // Shared manager: file_mapped_weights_enabled_ was latched by an earlier session and
    // wouldn't otherwise see this session's read callback.
    if (file_mapped_weights_enabled_ && io_dispatch.HasReadCallback()) {
      file_mapped_weights_enabled_ = false;
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_WARNING,
                      "EPContext read callback registered on a session reusing an already-set-up "
                      "backend manager; disabling file mapping for this session.");
    }

    if (htp_share_resource_optimization_ == 1) {
      // If a context bin filepath has not been processed yet,
      // then a new context must be created for the set of context bins
      auto first_mapping_it = ep_context_handle_map_.find(context_bin_map.begin()->first);
      if (first_mapping_it == ep_context_handle_map_.end()) {
        ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Creating context for new set of context binaries");
        return CreateContextVtcmBackupBufferSharingEnabled(context_bin_map,
                                                           io_dispatch);
      }

      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Mapping contexts to new EP main context nodes");

      for (auto& it : context_bin_map) {
        auto context_bin_filepath = it.first;
        auto ep_node_names = *(it.second);

        auto context = ep_context_handle_map_.at(context_bin_filepath);
        for (auto node_name : ep_node_names) {
          ep_context_handle_map_.emplace(node_name, context);
        }
      }
    }
    return Ort::Status();
  }

  htp_share_resource_optimization_ = htp_share_resource_optimization;

  auto status = Ort::Status();
  if (!qnn_serializer_config_) {
    status = LoadBackend();
  } else {
    status = LoadQnnSerializerBackend();
  }

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  // Backend is determined after LoadBackend() or LoadQnnSerializerBackend()
  if (enable_file_mapped_weights && !file_mapper_ && GetQnnBackendType() == QnnBackendType::HTP) {
    RETURN_IF(!rpcmem_library, "RPCMem Library is required for file mapping but is uninitialized.");
    rpcmem_library_ = rpcmem_library;
    file_mapped_weights_enabled_ = true;
    file_mapper_ = std::make_unique<WindowsFileMapper>(*logger_ptr_);
  }
#else
  ORT_UNUSED_PARAMETER(enable_file_mapped_weights);
  ORT_UNUSED_PARAMETER(rpcmem_library);
#endif

  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "LoadBackend succeed.");
  }

  if (status.IsOK() && (load_from_cached_context || need_load_system_lib)) {
    status = LoadQnnSystemLib();
  }

  if (status.IsOK()) {
    sdk_build_version_ = GetBackendBuildId();
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, ("Backend build version: " + sdk_build_version_).c_str());
  }

  if (status.IsOK()) {
    status = InitializeQnnLog();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "InitializeQnnLog succeed.");
  }

  bool enable_gpu_weight_sharing = false;
  if (share_ep_contexts && !load_from_cached_context) {
#if defined(__aarch64__) || defined(_M_ARM64)
    enable_gpu_weight_sharing = true;
#endif
  }

  if (IsGpuBackend(GetQnnBackendType())) {
    const std::string msg = std::string("GPU weight sharing: ") +
                            (enable_gpu_weight_sharing ? "enabled" : "disabled");
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, msg.c_str());
  }

  if (status.IsOK()) {
    status = InitializeBackend(enable_gpu_weight_sharing);
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "InitializeBackend succeed.");
  }

  if (status.IsOK()) {
    status = CreateDevice();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "CreateDevice succeed.");

    if (Ort::Status _status = GetPlatformInfo(); !_status.IsOK()) {
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_WARNING,
                      ("Unable to get platform info: " + _status.GetErrorMessage()).c_str());
    }
  }

  if (status.IsOK() && qnn_serializer_config_ && !skip_backend_op_validation_) {
    status = InitializeQnnValidatorLog();
    if (status.IsOK()) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "InitializeQnnValidatorLog succeed.");
    }
    if (status.IsOK()) {
      status = InitializeValidatorBackend();
    }
    if (status.IsOK()) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "InitializeValidatorBackend succeed.");
    }
    if (status.IsOK()) {
      status = CreateValidatorDevice();
    }
    if (status.IsOK()) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "CreateValidatorDevice succeed.");
    }
  } else if (qnn_serializer_config_ && skip_backend_op_validation_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_INFO,
                    "skip_backend_op_validation=1: skipping target-backend op validation; "
                    "DLC dump will use the serializer's generic op validation.");
  }

  if (status.IsOK()) {
    status = InitializeProfiling();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "InitializeProfiling succeed.");
  }

  if (status.IsOK()) {
    status = LoadOpPackage();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "LoadOpPackage succeed.");
  }

  bool enable_htp_weight_sharing = false;
  if (share_ep_contexts && !load_from_cached_context) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    (QNN_API_VERSION_MAJOR < 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR < 34))
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_WARNING,
                    "Weight sharing on Windows arm64 device requires QNN API version >=2.34. Since Current version is too old, disabling the Weight sharing.");
#elif defined(__ANDROID__)
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_WARNING,
                    "Weight sharing on Android devices is disabled");
#else
    enable_htp_weight_sharing = true;
#endif
  }

  if (status.IsOK() && (htp_share_resource_optimization_ == 1 || !load_from_cached_context)) {
    status = htp_share_resource_optimization_ == 1
                 ? CreateContextVtcmBackupBufferSharingEnabled(context_bin_map,
                                                               io_dispatch)
                 : CreateContext(enable_htp_weight_sharing,
                                 enable_htp_extended_udma_mode,
                                 enable_htp_prepare_only,
                                 false /*enable_htp_ref_weight_sharing*/,
                                 enable_htp_graph_splitting);

    if (status.IsOK()) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "CreateContext succeed.");
    }
  }

  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN SetupBackend succeed");
    backend_setup_completed_ = true;
  } else {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Failed to setup so cleaning up");
    ReleaseResources();
  }

  return status;
}

Ort::Status QnnBackendManager::SetupBackendExceptDeviceAndContext() {
  if (backend_partial_setup_completed_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN backend manager partially setup already.");
    return Ort::Status();
  }

  Ort::Status status = LoadBackend();
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Backend library loaded.");
  }

  if (status.IsOK()) {
    status = LoadQnnSystemLib();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "System library loaded.");
  }

  if (status.IsOK()) {
    sdk_build_version_ = GetBackendBuildId();
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, ("Backend build version: " + sdk_build_version_).c_str());
  }

  if (status.IsOK()) {
    status = InitializeQnnLog();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN log created.");
  }

  if (status.IsOK()) {
    status = InitializeBackend();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN backend created.");
  }

  if (status.IsOK()) {
    status = CreateSystemDlcPlugin();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN system DLC plugin created.");
  }

  if (status.IsOK()) {
    status = InitializeProfiling();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN profile created.");
  }

  if (status.IsOK()) {
    status = LoadOpPackage();
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Op package registered to backend.");
  }

  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN backend manager is partially setup.");
    backend_partial_setup_completed_ = true;
  } else {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Failed to partially setup so cleaning up.");
    ReleaseResources();
  }

  return status;
}

Ort::Status QnnBackendManager::SetupDeviceAndContext(QnnHtpDevice_Arch_t htp_arch,
                                                     uint32_t soc_model,
                                                     bool enable_htp_extended_udma_mode,
                                                     bool enable_htp_prepare_only,
                                                     bool enable_htp_ref_weight_sharing,
                                                     bool enable_htp_graph_splitting) {
  RETURN_IF_NOT(backend_partial_setup_completed_, "QNN backend manager must be partially setup first.");
  if (backend_setup_completed_) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN backend manager completely setup already.");
    return Ort::Status();
  }

  // Override cached values.
  // Update htp_arch_internal_ as well to make sure GetHtpArch returns correct value.
  htp_arch_ = htp_arch_internal_ = htp_arch;
  soc_model_ = soc_model;

  Ort::Status status = CreateDevice();
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN device created.");
  }

  if (status.IsOK()) {
    status = CreateContext(false /*enable_htp_weight_sharing*/,
                           enable_htp_extended_udma_mode,
                           enable_htp_prepare_only,
                           enable_htp_ref_weight_sharing,
                           enable_htp_graph_splitting);
  }
  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN context created.");
  }

  if (status.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "QNN backend manager is completely setup.");
    backend_setup_completed_ = true;
  } else {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "Failed to completely setup so cleaning up.");
    ReleaseDeviceAndContext();
  }

  return status;
}

Ort::Status QnnBackendManager::InitializePowerCfgId(uint32_t device_id, uint32_t core_id, uint32_t& htp_power_config_id) {
  RETURN_IF_ERROR(CreateHtpPowerCfgId(device_id, core_id, htp_power_config_id));
  htp_power_config_manager_.CreateTimerThread(htp_power_config_id);
  return Ort::Status();
}

void QnnBackendManager::DeInitializePerfTimer() {
  htp_power_config_manager_.ReleaseTimerThread();
}

void QnnBackendManager::DropBoostedPowerConfigId(uint32_t htp_power_config_id) {
  htp_power_config_manager_.DropBoostedPowerConfigId(htp_power_config_id);
}

Ort::Status QnnBackendManager::CreateHtpPowerCfgId(uint32_t device_id, uint32_t core_id, uint32_t& htp_power_config_id) {
  // This function is called in QNN EP's OnRunStart() even if QNN backend setup failed and the model is assigned
  // to a different EP. Therefore, we have to check that backend setup actually completed before trying to
  // create an HTP power config ID. Otherwise, this causes a segfault because the QNN backend lib is unloaded.
  RETURN_IF_NOT(backend_setup_completed_, "Cannot create HTP power config ID if backend setup is not complete.");
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
            ("HTP infra type = " + std::to_string(htp_infra->infraType) + ", which is not perf infra type.").c_str());
  QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;
  // Get power client id
  status = htp_perf_infra.createPowerConfigId(device_id, core_id, &htp_power_config_id);
  RETURN_IF(QNN_SUCCESS != status, "createPowerConfigId failed.");

  return Ort::Status();
}

Ort::Status QnnBackendManager::SetPerThreadHtpPowerConfigs(const std::thread::id& thread_id, bool pre_run) {
  PerThreadHtpPowerConfigs_t htp_power_configs;
  if (!GetPerThreadHtpPowerConfigMapping(thread_id, htp_power_configs)) {
    return Ort::Status();
  }

  auto htp_power_config_id = htp_power_configs.power_config_id;
  if (pre_run) {
    // add in htp_power_configs the default power config id also so to run when we execute
    if (htp_power_configs.pre_run_perf_mode.has_value()) {
      power::HtpPerfConfig_t config{htp_power_config_id, *htp_power_configs.pre_run_perf_mode, *htp_power_configs.rpc_polling_time, *htp_power_configs.rpc_control_latency};
      RETURN_IF_ERROR(htp_power_config_manager_.SetState(power::GraphState::RUN_START, config, *logger_ptr_));
    } else if (htp_power_configs.default_perf_mode.has_value()) {
      power::HtpPerfConfig_t config{htp_power_config_id, *htp_power_configs.default_perf_mode, *htp_power_configs.rpc_polling_time, *htp_power_configs.rpc_control_latency};
      RETURN_IF_ERROR(htp_power_config_manager_.SetState(power::GraphState::RUN_START, config, *logger_ptr_));
    }
  } else {
    if (htp_power_configs.post_run_perf_mode.has_value()) {
      power::HtpPerfConfig_t config{htp_power_config_id, *htp_power_configs.post_run_perf_mode, *htp_power_configs.rpc_polling_time, *htp_power_configs.rpc_control_latency};
      RETURN_IF_ERROR(htp_power_config_manager_.SetState(power::GraphState::RUN_DONE, config, *logger_ptr_));
    } else if (htp_power_configs.default_perf_mode.has_value()) {
      power::HtpPerfConfig_t config{htp_power_config_id, *htp_power_configs.default_perf_mode, *htp_power_configs.rpc_polling_time, *htp_power_configs.rpc_control_latency};
      RETURN_IF_ERROR(htp_power_config_manager_.SetState(power::GraphState::RUN_DONE, config, *logger_ptr_));
    }
  }
  return Ort::Status();
}

Ort::Status QnnBackendManager::AddPerThreadHtpPowerConfigMapping(const std::thread::id& thread_id,
                                                                 const PerThreadHtpPowerConfigs_t& htp_power_configs) {
  std::lock_guard<std::mutex> lock(per_thread_power_configs_mutex_);

  auto res = per_thread_power_configs_.find(thread_id);
  RETURN_IF(res != per_thread_power_configs_.end(),
            "Trying to set HtpPowerConfigs for the given thread id but it already exists!");

  per_thread_power_configs_.emplace(thread_id, std::move(htp_power_configs));

  return Ort::Status();
}

bool QnnBackendManager::GetPerThreadHtpPowerConfigMapping(const std::thread::id& thread_id,
                                                          PerThreadHtpPowerConfigs_t& htp_power_configs) {
  std::lock_guard<std::mutex> lock(per_thread_power_configs_mutex_);

  auto it = per_thread_power_configs_.find(thread_id);
  if (it == per_thread_power_configs_.end()) {
    return false;
  }

  htp_power_configs = it->second;
  return true;
}

void QnnBackendManager::RemovePerThreadHtpPowerConfigMapping(const std::thread::id& thread_id) {
  std::lock_guard<std::mutex> lock(per_thread_power_configs_mutex_);
  per_thread_power_configs_.erase(thread_id);
}

Ort::Status QnnBackendManager::DestroyHtpPowerConfigId(uint32_t htp_power_config_id) {
  QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
  auto status = qnn_interface_.deviceGetInfrastructure(&qnn_device_infra);
  RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

  auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
  RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
            ("HTP infra type = " + std::to_string(htp_infra->infraType) + ", which is not perf infra type.").c_str());
  QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

  Qnn_ErrorHandle_t destroy_ret = htp_perf_infra.destroyPowerConfigId(htp_power_config_id);
  RETURN_IF(QNN_SUCCESS != destroy_ret, "destroyPowerConfigId failed.");
  return Ort::Status();
}

Ort::Status QnnBackendManager::TerminateQnnLogCommon(const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                     Qnn_LogHandle_t& log_handle,
                                                     const std::string& backend_label) {
  if (qnn_interface.logFree == nullptr || log_handle == nullptr) {
    return Ort::Status();
  }

  auto ret_val = qnn_interface.logFree(log_handle);

  // Reset to nullptr BEFORE checking the result so that other threads waiting on
  // logger_recursive_mutex_ can observe the handle is gone, even if logFree failed.
  log_handle = nullptr;

  RETURN_IF(QNN_SUCCESS != ret_val,
            ("Unable to terminate logging in the " + backend_label + ".").c_str());
  return Ort::Status();
}

Ort::Status QnnBackendManager::TerminateQnnLog() {
  std::lock_guard<std::recursive_mutex> lock(logger_recursive_mutex_);

  // Attempt to free both log handles even if the first one fails (best-effort).
  Ort::Status s1 = TerminateQnnLogCommon(qnn_interface_, log_handle_, "backend");
  Ort::Status s2 = TerminateQnnLogCommon(qnn_validator_interface_, validator_log_handle_, "validator");
  if (!s1.IsOK()) return s1;
  if (!s2.IsOK()) return s2;
  return Ort::Status();
}

void QnnBackendManager::ReleaseResources() {
  // Each sub-function guards against releasing resources that were never created,
  // so all calls are safe regardless of how far setup progressed.

  // Tear down the HTP release timer FIRST. Its callback drives the QNN device
  // (SetPowerConfig), so it must be joined before the device/context is released
  // to avoid the callback touching freed QNN handles. Tying timer teardown to the
  // manager's lifetime (rather than to an individual QnnEp destructor) is also
  // what makes a shared manager safe: the timer now dies with the manager (the
  // last session out), not when the first sharing session is destroyed.
  DeInitializePerfTimer();

  auto result = ReleaseContext();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, ("Failed to ReleaseContext: " + result.GetErrorMessage()).c_str());
  }

  result = ReleaseProfilehandle();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_ERROR,
                    ("Failed to ReleaseProfilehandle: " + result.GetErrorMessage()).c_str());
  }

  result = ReleaseDevice();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, ("Failed to ReleaseDevice: " + result.GetErrorMessage()).c_str());
  }

  result = ReleaseValidatorDevice();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, ("Failed to ReleaseValidatorDevice: " + result.GetErrorMessage()).c_str());
  }

  result = ReleaseSystemDlcPlugin();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_ERROR,
                    ("Failed to ReleaseSystemDlcPlugin: " + result.GetErrorMessage()).c_str());
  }

  result = ShutdownBackend();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, ("Failed to ShutdownBackend: " + result.GetErrorMessage()).c_str());
  }

  result = ShutdownValidatorBackend();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, ("Failed to ShutdownValidatorBackend: " + result.GetErrorMessage()).c_str());
  }

  result = TerminateQnnLog();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_ERROR, ("Failed to TerminateQnnLog: " + result.GetErrorMessage()).c_str());
  }

  if (backend_lib_handle_) {
    result = UnloadLib(backend_lib_handle_);
    if (!result.IsOK()) {
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_ERROR,
                      ("Failed to unload backend library: " + result.GetErrorMessage()).c_str());
    }
    backend_lib_handle_ = nullptr;
  }

  if (validator_backend_lib_handle_) {
    result = UnloadLib(validator_backend_lib_handle_);
    if (!result.IsOK()) {
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_ERROR,
                      ("Failed to unload validator backend library: " + result.GetErrorMessage()).c_str());
    }
    validator_backend_lib_handle_ = nullptr;
  }

  backend_lib_loaded_ = false;
  backend_partial_setup_completed_ = false;
  backend_setup_completed_ = false;
}

void QnnBackendManager::ReleaseDeviceAndContext() {
  Ort::Status result = ReleaseContext();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_ERROR,
                    ("Failed to free QNN context: " + result.GetErrorMessage()).c_str());
  }

  result = ReleaseDevice();
  if (!result.IsOK()) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_ERROR,
                    ("Failed to free QNN device: " + result.GetErrorMessage()).c_str());
  }

  // Reset to default values as opposed to cached values set in `SetupDeviceAndContext`.
  htp_arch_ = htp_arch_internal_ = QNN_HTP_DEVICE_ARCH_NONE;
  soc_model_ = QNN_SOC_MODEL_UNKNOWN;

  backend_setup_completed_ = false;
}

Ort::Status QnnBackendManager::ExtractBackendProfilingInfo(qnn::profile::ProfilingInfo& profiling_info) {
  if (ProfilingLevel::OFF == profiling_level_merge_ || ProfilingLevel::INVALID == profiling_level_merge_) {
    return Ort::Status();
  }

  bool tracelogging_provider_ep_enabled = false;
#ifdef _WIN32
  auto& provider = QnnTelemetry::Instance();
  if (provider.IsEnabled()) {
    auto level = provider.Level();
    auto keyword = provider.Keyword();
    if ((keyword & static_cast<uint64_t>(qnn::ORTTraceLoggingKeyword::Profiling)) != 0 && level >= 5) {
      tracelogging_provider_ep_enabled = true;
    }
  }
#endif  // defined(_WIN32)

  // ETW disabled previously, but enabled now
  if (ProfilingLevel::INVALID == profiling_level_etw_ && tracelogging_provider_ep_enabled) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_ERROR,
                    "ETW disabled previously, but enabled now. Can't do the switch! Won't output any profiling.");
    return Ort::Status();
  }

  // ETW enabled previously, but disabled now
  if (ProfilingLevel::INVALID != profiling_level_etw_ && !tracelogging_provider_ep_enabled) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_ERROR,
                    "ETW enabled previously, but disabled now. Can't do the switch! Won't output any profiling.");
    return Ort::Status();
  }

  RETURN_IF(!tracelogging_provider_ep_enabled && profiling_file_path_.empty(),
            "Need to specify a CSV file via provider option profiling_file_path if ETW not enabled.");

  RETURN_IF(nullptr == profile_backend_handle_, "Backend profile handle not valid.");

  ORT_CXX_LOG_PTR(logger_ptr_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Extracting profiling events for graph " + profiling_info.graph_name).c_str());

  const QnnProfile_EventId_t* profile_events{nullptr};
  uint32_t num_events{0};
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetEvents(profile_backend_handle_, &profile_events, &num_events);
  if (qnn_serializer_config_) {  // Using QNN Saver or IR backend
    // QNN SDK 2.28.2 returns QNN_SAVER_ERROR_DUMMY_RETVALUE, but previous QNN versions return QNN_PROFILE_NO_ERROR.
    // We accept both values.
    RETURN_IF(QNN_PROFILE_NO_ERROR != result && QNN_SAVER_ERROR_DUMMY_RETVALUE != result,
              ("Failed to get profile events. Error: " + QnnErrorHandleToString(result)).c_str());
  } else {
    RETURN_IF(QNN_PROFILE_NO_ERROR != result,
              ("Failed to get profile events. Error: " + QnnErrorHandleToString(result)).c_str());
  }

  if (num_events > 0) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_VERBOSE,
                    ("profile_events: " + std::to_string(*profile_events) +
                     " num_events: " + std::to_string(num_events))
                        .c_str());

    bool backendSupportsExtendedEventData = false;
    Qnn_ErrorHandle_t resultPropertyHasCapability =
        qnn_interface_.propertyHasCapability(QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT);
    uint16_t errorCodePropertyHasCapability = static_cast<uint16_t>(resultPropertyHasCapability & 0xFFFF);
    if (errorCodePropertyHasCapability == QNN_PROPERTY_SUPPORTED) {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "The QNN backend supports extended event data.");
      backendSupportsExtendedEventData = true;
    } else {
      ORT_CXX_LOG_PTR(logger_ptr_, ORT_LOGGING_LEVEL_VERBOSE, "The QNN backend does not support extended event data.");
    }

    profiling_info.csv_output_filepath = profiling_file_path_;
#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    profiling_info.num_events = num_events;
#endif

    // When framework op tracing is enabled, attach the lookup so InitCsvFile()
    // emits the `ONNX Source Ops` column header and ProcessEvent() annotates each
    // NODE row with the originating ONNX op names. The lookup is read by pointer
    // and continues to fill as later graphs compose; only DETAILED/OPTRACE
    // profiling produces the per-NODE events this column annotates.
    const bool has_node_level_profiling = profiling_level_merge_ == ProfilingLevel::DETAILED ||
                                          profiling_level_merge_ == ProfilingLevel::OPTRACE;
    if (enable_framework_op_trace_ && has_node_level_profiling) {
      profiling_info.op_trace_lookup = &op_trace_lookup_;
    }

    profile::Serializer profile_writer(profiling_info,
                                       qnn_sys_interface_,
                                       tracelogging_provider_ep_enabled);
    if (!profiling_file_path_.empty()) {
      RETURN_IF_ERROR(profile_writer.InitCsvFile());
    }

    for (size_t event_idx = 0; event_idx < num_events; event_idx++) {
      RETURN_IF_ERROR(ExtractProfilingEvent(*(profile_events + event_idx),
                                            "ROOT",
                                            profile_writer,
                                            backendSupportsExtendedEventData));
      RETURN_IF_ERROR(ExtractProfilingSubEvents(*(profile_events + event_idx),
                                                profile_writer,
                                                backendSupportsExtendedEventData));
    }

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    RETURN_IF_ERROR(profile_writer.SerializeEventsToQnnLog());
#endif

    if (!profiling_file_path_.empty()) {
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_VERBOSE,
                      ("Wrote QNN profiling events (" + std::to_string(num_events) +
                       ") to file (" + profiling_file_path_ + ")")
                          .c_str());
    }

    if (tracelogging_provider_ep_enabled) {
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_VERBOSE,
                      ("Wrote QNN profiling events (" + std::to_string(num_events) + ") to ETW").c_str());
    }
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::ExtractProfilingSubEvents(QnnProfile_EventId_t profile_event_id,
                                                         profile::Serializer& profile_writer,
                                                         bool useExtendedEventData) {
  const QnnProfile_EventId_t* profile_sub_events{nullptr};
  uint32_t num_sub_events{0};
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetSubEvents(profile_event_id, &profile_sub_events, &num_sub_events);
  RETURN_IF(QNN_PROFILE_NO_ERROR != result,
            ("Failed to get profile sub events. Error: " + QnnErrorHandleToString(result)).c_str());

  if (num_sub_events > 0) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_VERBOSE,
                    ("profile_sub_events: " + std::to_string(*profile_sub_events) +
                     " num_sub_events: " + std::to_string(num_sub_events))
                        .c_str());

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
    QnnSystemProfile_ProfileEventV1_t* parent_system_event = nullptr;
    parent_system_event = profile_writer.GetParentSystemEvent(profile_event_id);
    if (parent_system_event == nullptr) {
      parent_system_event = profile_writer.GetSystemEventPointer(profile_event_id);
      profile_writer.AddSubEventList(num_sub_events, parent_system_event);
    }
#endif

    for (size_t sub_event_idx = 0; sub_event_idx < num_sub_events; sub_event_idx++) {
      QnnProfile_EventId_t subevent_id = *(profile_sub_events + sub_event_idx);

#ifdef QNN_SYSTEM_PROFILE_API_ENABLED
      RETURN_IF_ERROR(profile_writer.SetParentSystemEvent(subevent_id, parent_system_event));
#endif
      RETURN_IF_ERROR(ExtractProfilingEvent(subevent_id, "SUB-EVENT", profile_writer, useExtendedEventData));
      RETURN_IF_ERROR(ExtractProfilingSubEvents(subevent_id, profile_writer, useExtendedEventData));
    }

    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_VERBOSE,
                    ("Wrote QNN profiling sub events (" + std::to_string(num_sub_events) + ")").c_str());
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::ExtractProfilingEvent(QnnProfile_EventId_t profile_event_id,
                                                     const std::string& event_level,
                                                     profile::Serializer& profile_writer,
                                                     bool useExtendedEventData) {
  if (useExtendedEventData) {
    return ExtractProfilingEventExtended(profile_event_id, event_level, profile_writer);
  } else {
    return ExtractProfilingEventBasic(profile_event_id, event_level, profile_writer);
  }
}

Ort::Status QnnBackendManager::ExtractProfilingEventBasic(QnnProfile_EventId_t profile_event_id,
                                                          const std::string& event_level,
                                                          profile::Serializer& profile_writer) {
  QnnProfile_EventData_t event_data;
  Qnn_ErrorHandle_t result = qnn_interface_.profileGetEventData(profile_event_id, &event_data);
  QnnProfile_Error_t errorCode = static_cast<QnnProfile_Error_t>(result & 0xFFFF);
  RETURN_IF(QNN_PROFILE_NO_ERROR != result,
            ("Failed to get profile event data: " + std::string(QnnProfileErrorToString(errorCode))).c_str());

  RETURN_IF_ERROR(profile_writer.ProcessEvent(profile_event_id, event_level, event_data));

  return Ort::Status();
}

Ort::Status QnnBackendManager::ExtractProfilingEventExtended(QnnProfile_EventId_t profile_event_id,
                                                             const std::string& event_level,
                                                             profile::Serializer& profile_writer) {
  QnnProfile_ExtendedEventData_t event_data_extended;
  auto resultGetExtendedEventData = qnn_interface_.profileGetExtendedEventData(profile_event_id, &event_data_extended);
  QnnProfile_Error_t errorCode = static_cast<QnnProfile_Error_t>(resultGetExtendedEventData & 0xFFFF);
  RETURN_IF(QNN_PROFILE_NO_ERROR != errorCode,
            ("Failed to get profile event data: " + std::string(QnnProfileErrorToString(errorCode))).c_str());

  RETURN_IF_ERROR(profile_writer.ProcessExtendedEvent(profile_event_id, event_level, event_data_extended));

  return Ort::Status();
}

const char* QnnBackendManager::QnnProfileErrorToString(QnnProfile_Error_t error) {
  switch (error) {
    case QNN_PROFILE_NO_ERROR:
      return "QNN_PROFILE_NO_ERROR";
    case QNN_PROFILE_ERROR_UNSUPPORTED:
      return "QNN_PROFILE_ERROR_UNSUPPORTED";
    case QNN_PROFILE_ERROR_INVALID_ARGUMENT:
      return "QNN_PROFILE_ERROR_INVALID_ARGUMENT";
    case QNN_PROFILE_ERROR_MEM_ALLOC:
      return "QNN_PROFILE_ERROR_MEM_ALLOC";
    case QNN_PROFILE_ERROR_INVALID_HANDLE:
      return "QNN_PROFILE_ERROR_INVALID_HANDLE";
    case QNN_PROFILE_ERROR_HANDLE_IN_USE:
      return "QNN_PROFILE_ERROR_HANDLE_IN_USE";
    case QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT:
      return "QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT";
    default:
      return "UNKNOWN_ERROR";
  }
}

std::string QnnBackendManager::QnnErrorHandleToString(Qnn_ErrorHandle_t error) {
  return utils::GetQnnErrorMessage(qnn_interface_, error);
}

QnnBackendManager::~QnnBackendManager() {
  ReleaseResources();
}

void* QnnBackendManager::LoadLib(const char* file_name, int flags, std::string& error_msg) {
#ifdef _WIN32
  DWORD as_is, to_be;
  bool loaded_before = false;

  if (!file_name || ::strlen(file_name) == 0) {
    error_msg = "filename is null or empty";
    return nullptr;
  }

  // POSIX asks one of symbol resolving approaches:
  // NOW or LAZY must be specified
  if (!(flags & static_cast<int>(DlOpenFlag::DL_NOW))) {
    error_msg = "flags must include DL_NOW";
    return nullptr;
  }

  HANDLE cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, nullptr, 0, &as_is) == 0) {
    error_msg = "enumerate modules failed before loading module";
    return nullptr;
  }

  HMODULE mod;
  auto file_path = std::filesystem::path(file_name);
  if (!file_path.is_absolute()) {
    // construct an absolute path from ORT runtime path + file_name and check whether it exists.
    auto pathstring = std::filesystem::path(OrtGetRuntimePath()) / file_path;
    auto absolute_path = pathstring.c_str();
    if (std::filesystem::exists(pathstring)) {
      // load library from absolute path and search for dependencies there.
      mod = LoadLibraryExW(absolute_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    } else {
      // use default dll search order for file_name.
      mod = LoadLibraryExA(file_name, nullptr, 0);
    }
  } else {
    // file_name represents an absolute path.
    // load library from absolute path and search for dependencies there.
    mod = LoadLibraryExA(file_name, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  }
  if (!mod) {
    error_msg = "load library failed";
    return nullptr;
  }

  if (EnumProcessModules(cur_proc, nullptr, 0, &to_be) == 0) {
    error_msg = "enumerate modules failed after loading module";
    FreeLibrary(mod);
    return nullptr;
  }

  if (as_is == to_be) {
    loaded_before = true;
  }

  // (not loaded_before) and DL_LOCAL means this lib was not loaded yet
  // add it into the local set
  //
  // If loaded_before and DL_LOCAL, means this lib was already loaded
  // 2 cases here for how it was loaded before:
  // a. with DL_LOCAL, just ignore since it was already in local set
  // b. with DL_GLOBAL, POSIX asks it in global, ignore it, too
  if ((!loaded_before) && (flags & static_cast<int>(DlOpenFlag::DL_LOCAL))) {
    mod_handles_.insert(mod);
  }

  // once callers ask for global, needs to be in global thereafter
  // so the lib should be removed from local set
  if (flags & static_cast<int>(DlOpenFlag::DL_GLOBAL)) {
    mod_handles_.erase(mod);
  }

  return static_cast<void*>(mod);
#else
  ORT_UNUSED_PARAMETER(error_msg);
  int real_flags = 0;

  if (flags & static_cast<int>(DlOpenFlag::DL_NOW)) {
    real_flags |= RTLD_NOW;
  }

  if (flags & static_cast<int>(DlOpenFlag::DL_LOCAL)) {
    real_flags |= RTLD_LOCAL;
  }

  if (flags & static_cast<int>(DlOpenFlag::DL_GLOBAL)) {
    real_flags |= RTLD_GLOBAL;
  }

  return ::dlopen(file_name, real_flags);
#endif
}

Ort::Status QnnBackendManager::UnloadLib(void* handle) {
  if (!handle) {
    return Ort::Status();
  }

#ifdef _WIN32
  HMODULE mod = static_cast<HMODULE>(handle);

  if (FreeLibrary(mod) == 0) {
    return MAKE_EP_FAIL("Failed to free library.");
  }
  mod_handles_.erase(mod);
#else
  auto rt = ::dlclose(handle);
  if (rt != 0) {
    return MAKE_EP_FAIL("Failed to free library.");
  }
#endif  // defined(_WIN32)

  return Ort::Status();
}

void* QnnBackendManager::LibFunction(void* handle, const char* symbol, std::string& error_msg) {
#ifdef _WIN32
  FARPROC sym_addr = nullptr;
  DWORD size, size_needed;
  HMODULE mod = 0;

  if ((!handle) || (!symbol)) {
    return nullptr;
  }

  HANDLE cur_proc = GetCurrentProcess();

  if (EnumProcessModules(cur_proc, nullptr, 0, &size) == 0) {
    error_msg = "enumerate modules failed before memory allocation";
    return nullptr;
  }

  HMODULE* mod_list = static_cast<HMODULE*>(malloc(size));
  if (!mod_list) {
    error_msg = "malloc failed";
    return nullptr;
  }

  if (EnumProcessModules(cur_proc, mod_list, size, &size_needed) == 0) {
    error_msg = "enumerate modules failed after memory allocation";
    free(mod_list);
    return nullptr;
  }

  // DL_DEFAULT needs to bypass those modules with DL_LOCAL flag
  if (handle == DL_DEFAULT) {
    for (size_t i = 0; i < (size / sizeof(HMODULE)); i++) {
      auto iter = mod_handles_.find(mod_list[i]);
      if (iter != mod_handles_.end()) {
        continue;
      }
      // once find the first non-local module with symbol
      // return its address here to avoid unnecessary looping
      sym_addr = GetProcAddress(mod_list[i], symbol);
      if (sym_addr) {
        free(mod_list);
        return *(void**)(&sym_addr);
      }
    }
  } else {
    mod = static_cast<HMODULE>(handle);
  }

  free(mod_list);
  sym_addr = GetProcAddress(mod, symbol);
  if (!sym_addr) {
    error_msg = "can't resolve symbol";
    return NULL;
  }

  return *(void**)(&sym_addr);
#else
  ORT_UNUSED_PARAMETER(error_msg);
  if (handle == DL_DEFAULT) {
    return ::dlsym(RTLD_DEFAULT, symbol);
  }

  return ::dlsym(handle, symbol);
#endif
}

Ort::Status QnnBackendManager::AddQnnContextHandle(Qnn_ContextHandle_t raw_context_handle) {
  auto free_context_handle = [this](Qnn_ContextHandle_t raw_context_handle) {
    const auto free_result = qnn_interface_.contextFree(raw_context_handle, nullptr);
    if (free_result != QNN_CONTEXT_NO_ERROR) {
      ORT_CXX_LOG_PTR(logger_ptr_,
                      ORT_LOGGING_LEVEL_ERROR,
                      ("qnn_interface.contextFree() failed: " + utils::GetVerboseQnnErrorMessage(qnn_interface_, free_result)).c_str());
    }
  };

  // take ownership of `raw_context_handle`
  auto context_handle = UniqueQnnContextHandle(raw_context_handle, free_context_handle);
  auto mem_handle_manager = std::make_unique<QnnContextMemHandleManager>(GetQnnInterface(),
                                                                         raw_context_handle,
                                                                         qnn_backend_type_,
                                                                         qnn_allocator_type_);

  auto context_handle_record = std::make_shared<QnnContextHandleRecord>();
  context_handle_record->context_handle = std::move(context_handle);
  context_handle_record->mem_handles = std::move(mem_handle_manager);

  const bool inserted = context_map_.try_emplace(raw_context_handle, std::move(context_handle_record)).second;
  RETURN_IF_NOT(inserted, "QNN context was already added.");

  contexts_.push_back(raw_context_handle);

  return Ort::Status();
}

Ort::Status QnnBackendManager::GetOrRegisterContextMemHandle(Qnn_ContextHandle_t context_handle,
                                                             void* memory_address,
                                                             const Qnn_Tensor_t& qnn_tensor,
                                                             Qnn_MemHandle_t& mem_handle) {
  // Multi-threading situations to consider:
  // 1) Shared memory allocation is being freed in another thread while we are processing `memory_address`.
  //    This implies incorrect usage as the memory is being freed while it is still in use. Let's assume this won't
  //    happen.
  // 2) The shared memory allocation clean up function is being run from another thread while the
  //    QnnContextHandleRecord or QnnBackendManager objects are being destroyed.
  //    Usage of weak_ptrs from the clean up function should ensure that those objects are only accessed while they are
  //    in scope.

  const auto context_handle_record_it = context_map_.find(context_handle);
  RETURN_IF_NOT(context_handle_record_it != context_map_.end(), "QNN context not found.");

  auto& context_handle_record = context_handle_record_it->second;
  auto& context_mem_handle_manager = context_handle_record->mem_handles;

  bool did_register{};
  RETURN_IF_ERROR(context_mem_handle_manager->GetOrRegister(memory_address,
                                                            qnn_tensor,
                                                            mem_handle,
                                                            did_register,
                                                            *logger_ptr_));

  if (did_register) {
    // The cleanup lambda is the same for both HTP and DX12: unregister the QNN mem handle when the allocation is freed.
    auto unregister_mem_handle =
        [memory_address,
         weak_backend_manager = weak_from_this(),
         weak_context_handle_record = std::weak_ptr{context_handle_record}](
            void* /* allocation_base_address  */) {
          // Lock QnnBackendManager shared_ptr to ensure that QNN interface is still valid.
          auto backend_manager = weak_backend_manager.lock();
          if (!backend_manager) {
            return;
          }

          // Lock QnnContextHandleRecord shared_ptr to ensure that QNN context handle is still valid.
          auto context_handle_record = weak_context_handle_record.lock();
          if (!context_handle_record) {
            return;
          }

          auto& context_mem_handle_manager = context_handle_record->mem_handles;

          auto unregister_status = context_mem_handle_manager->Unregister(memory_address);
          if (!unregister_status.IsOK()) {
            std::ostringstream oss;
            oss << "Failed to unregister mem handle for address: "
                << memory_address
                << ", error: "
                << unregister_status.GetErrorMessage();
            ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(), ORT_LOGGING_LEVEL_ERROR, oss.str().c_str());
          }
        };

#ifdef _WIN32
    if (QnnExternalResourceImporterImpl::FindImportMemory(memory_address)) {
      RETURN_IF_ERROR(QnnExternalResourceImporterImpl::AddImportMemoryCleanUp(
          memory_address, QnnExternalResourceImporterImpl::ImportResourceCleanUpFn{std::move(unregister_mem_handle)}));
    } else if (IsDx12SharedMemoryAllocator(qnn_allocator_type_)) {
      RETURN_IF_ERROR(Dx12SharedMemoryAllocator::AddAllocationCleanUp(
          memory_address, Dx12SharedMemoryAllocator::AllocationCleanUpFn{std::move(unregister_mem_handle)}));
    } else
#endif  // _WIN32
      if (IsHtpSharedMemoryAllocator(qnn_allocator_type_)) {
        RETURN_IF_ERROR(HtpSharedMemoryAllocator::AddAllocationCleanUp(
            memory_address, HtpSharedMemoryAllocator::AllocationCleanUpFn{std::move(unregister_mem_handle)}));
      } else {
        return MAKE_EP_FAIL("Cannot add allocation clean up function for unsupported backend.");
      }
  }

  return Ort::Status();
}

Ort::Status QnnBackendManager::GetPlatformInfo() {
  if (!IsNpuBackend(GetQnnBackendType())) {
    // Add support other backends.
    return MAKE_EP_FAIL("Only support getting platform info for HTP backend.");
  }

  // Directly return if already acquired before.
  if (htp_arch_internal_ != QNN_HTP_DEVICE_ARCH_NONE) {
    return Ort::Status();
  }

#if defined(__aarch64__) || defined(_M_ARM64) || (defined(_M_ARM64EC))
  RETURN_IF(qnn_interface_.deviceGetPlatformInfo == nullptr || qnn_interface_.deviceFreePlatformInfo == nullptr,
            "Failed to get valid QnnDevice function pointers.");

  Qnn_ErrorHandle_t result;

  const QnnDevice_PlatformInfo_t* platform_info_raw_ptr = nullptr;
  result = qnn_interface_.deviceGetPlatformInfo(log_handle_, &platform_info_raw_ptr);
  RETURN_IF(result != QNN_SUCCESS || platform_info_raw_ptr == nullptr, "Failed to get platform info.");

  std::unique_ptr<const QnnDevice_PlatformInfo_t, std::function<void(const QnnDevice_PlatformInfo_t*)>>
  platform_info_ptr(
      platform_info_raw_ptr,
      [&qnn_interface = qnn_interface_, &log_handle = log_handle_](const QnnDevice_PlatformInfo_t* platform_info) {
        qnn_interface.deviceFreePlatformInfo(log_handle, platform_info);
      });

  if (platform_info_ptr->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
    const QnnDevice_PlatformInfoV1_t& plt_info = platform_info_ptr->v1;

    for (uint32_t idx = 0; idx < plt_info.numHwDevices; ++idx) {
      const QnnDevice_HardwareDeviceInfo_t& hw_info = plt_info.hwDevices[idx];
      if (hw_info.version != QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) {
        continue;
      }

      const auto* htp_ext = reinterpret_cast<const QnnHtpDevice_DeviceInfoExtension_t*>(hw_info.v1.deviceInfoExtension);
      if (htp_ext && htp_ext->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
        htp_arch_internal_ = htp_ext->onChipDevice.arch;
        vtcm_size_internal_ = static_cast<uint32_t>(htp_ext->onChipDevice.vtcmSize);
        break;
      }
    }
  } else {
    return MAKE_FAIL("Unrecognized platform info version.");
  }

  RETURN_IF(htp_arch_internal_ == QNN_HTP_DEVICE_ARCH_NONE, "Failed to get HTP arch.");
  RETURN_IF(vtcm_size_internal_ == 0, "Failed to get VTCM size.");
#else
  // QnnDevice_getPlatformInfo will always return HTP arch 68 and VTCM size 4 on x86 platform even if GetPlatformInfo
  // is called after device is created. Thus, adopting user-specified value is the only option.
  if (htp_arch_ != QNN_HTP_DEVICE_ARCH_NONE) {
    htp_arch_internal_ = htp_arch_;
  }
#endif  // defined(__aarch64__) || defined(_M_ARM64) || (defined(_M_ARM64EC))

  return Ort::Status();
}

std::unique_ptr<void, std::function<void(void*)>> QnnBackendManager::GetSystemContextHandle() {
  QnnSystemContext_Handle_t sys_ctx_handle = nullptr;
  auto rt = qnn_sys_interface_.systemContextCreate(&sys_ctx_handle);
  if (QNN_SUCCESS != rt) {
    ORT_CXX_LOG_PTR(logger_ptr_,
                    ORT_LOGGING_LEVEL_ERROR,
                    "Failed to create system handle.");
    return nullptr;
  }

  auto sys_ctx_handle_deleter = [&qnn_sys_interface = qnn_sys_interface_](void* handle) {
    qnn_sys_interface.systemContextFree(reinterpret_cast<QnnSystemContext_Handle_t>(handle));
  };

  std::unique_ptr<void, std::function<void(void*)>> sys_ctx_handle_uptr(sys_ctx_handle, sys_ctx_handle_deleter);
  return sys_ctx_handle_uptr;
}

Ort::Status QnnBackendManager::GetGraphInfoAndBinVersion(QnnSystemContext_Handle_t sys_ctx_handle,
                                                         void* buffer,
                                                         Qnn_ContextBinarySize_t buffer_length,
                                                         Qnn_Version_t& blob_version,
                                                         uint32_t& graph_count,
                                                         QnnSystemContext_GraphInfo_t** graphs_info) {
  RETURN_IF(sys_ctx_handle == nullptr, "System context handle is null.");

  // The lifetime of binary_info's contents is tied to the lifetime of
  // the obj pointed to by sys_ctx_handle (owned by caller)
  Qnn_ContextBinarySize_t binary_info_size{0};
  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  auto rt = qnn_sys_interface_.systemContextGetBinaryInfo(sys_ctx_handle,
                                                          buffer,
                                                          buffer_length,
                                                          &binary_info,
                                                          &binary_info_size);

  RETURN_IF(QNN_SUCCESS != rt, "Failed to get context binary info.");
  RETURN_IF(nullptr == binary_info, "Qnn cached binary info is nullptr.");

  // Extract graph info and context bin version from binary_info
  if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graph_count = binary_info->contextBinaryInfoV1.numGraphs;
    *graphs_info = binary_info->contextBinaryInfoV1.graphs;
    blob_version = binary_info->contextBinaryInfoV1.contextBlobVersion;
  }
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 15)  // starts from 2.22
  else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graph_count = binary_info->contextBinaryInfoV2.numGraphs;
    *graphs_info = binary_info->contextBinaryInfoV2.graphs;
    blob_version = binary_info->contextBinaryInfoV2.contextBlobVersion;
  }
#endif
#if QNN_API_VERSION_MAJOR == 2 && (QNN_API_VERSION_MINOR >= 21)  // starts from 2.28
  else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graph_count = binary_info->contextBinaryInfoV3.numGraphs;
    *graphs_info = binary_info->contextBinaryInfoV3.graphs;
    blob_version = binary_info->contextBinaryInfoV3.contextBlobVersion;
  }
#endif
  else {
    return MAKE_EP_FAIL("Unsupported context binary info version.");
  }

  return Ort::Status();
}

void QnnBackendManager::ReleaseSpecificContextHandle(Qnn_ContextHandle_t old_context) {
  // Remove name→handle mappings that reference this context.
  {
    std::lock_guard<std::mutex> lock(ep_context_handle_map_mutex_);
    for (auto it = ep_context_handle_map_.begin(); it != ep_context_handle_map_.end();) {
      if (it->second == old_context) {
        it = ep_context_handle_map_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Remove from the non-owning vector.
  contexts_.erase(std::remove(contexts_.begin(), contexts_.end(), old_context), contexts_.end());

  // Remove from the owning map — triggers contextFree via UniqueQnnContextHandle deleter.
  context_map_.erase(old_context);
}

bool QnnBackendManager::IsDx12SharedMemoryAllocatorSupported() {
#if !defined(_WIN32)
  return false;
#else
  if (dx12_shared_memory_allocator_supported_.has_value()) {
    return dx12_shared_memory_allocator_supported_.value();
  }

  bool supported = true;

  HRESULT hr = S_OK;
  Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device;
  Microsoft::WRL::ComPtr<ID3D12Resource> d3d12_resource;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_MemDescriptor_t mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
  Qnn_MemHandle_t raw_mem_handle = nullptr;

  // note: This function may be called before SetupBackend
  if (!LoadBackend().IsOK()) {
    supported = false;
  }

  if (supported && !InitializeQnnLog().IsOK()) {
    supported = false;
  }

  if (supported && !InitializeBackend().IsOK()) {
    supported = false;
  }

  if (supported && !CreateDevice().IsOK()) {
    supported = false;
  }

  if (supported) {
    hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3d12_device));
    if (FAILED(hr) || d3d12_device == nullptr) {
      supported = false;
    }
  }

  if (supported) {
    D3D12_RESOURCE_DESC buffer_desc = {};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = sizeof(float);
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buffer_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES heap_props = {};
    heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    hr = d3d12_device->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &buffer_desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&d3d12_resource));
    if (FAILED(hr) || d3d12_resource == nullptr) {
      supported = false;
    }
  }

  if (supported) {
    auto dims = std::array{1u};

    mem_descriptor.memShape.dimSize = dims.data();
    mem_descriptor.memShape.numDim = static_cast<uint32_t>(dims.size());
    mem_descriptor.memShape.shapeConfig = nullptr;
    mem_descriptor.dataType = QNN_DATATYPE_FLOAT_32;

    mem_descriptor.memType = QNN_MEM_TYPE_DX12;
    mem_descriptor.dx12BufInfo.resourceHandle =
        static_cast<Qnn_Dx12ResourceHandle_t>(d3d12_resource.Get());

    Qnn_ErrorHandle_t result = qnn_interface_.contextCreate(backend_handle_,
                                                            device_handle_,
                                                            nullptr,
                                                            &context);
    if (result != QNN_SUCCESS) {
      supported = false;
    }
  }

  if (supported) {
    const auto register_result = qnn_interface_.memRegister(context, &mem_descriptor, 1, &raw_mem_handle);

    if (IsGpuBackend(qnn_backend_type_) && register_result == QNN_MEM_ERROR_MAPPING) {
      ORT_CXX_LOG(OrtLoggingManager::GetDefaultLogger(),
                  ORT_LOGGING_LEVEL_ERROR,
                  "QnnMem_register failed with QNN_MEM_ERROR_MAPPING when using the DX12 shared memory allocator with the GPU"
                  " backend on Windows. This is likely due to outdated graphics drivers on the device. Please try installing"
                  " new drivers from https://softwarecenter.qualcomm.com/catalog/item/Windows_Graphics_Driver.");
    }

    if (register_result != QNN_SUCCESS) {
      supported = false;
    }
  }

  // clean up
  if (raw_mem_handle != nullptr) {
    qnn_interface_.memDeRegister(&raw_mem_handle, 1);
  }

  if (context != nullptr) {
    qnn_interface_.contextFree(context, nullptr);
  }

  dx12_shared_memory_allocator_supported_ = supported;
  return supported;
#endif
}

Ort::Status QnnBackendManager::CreateSystemDlcPlugin() {
  if (system_dlc_created_) {
    return Ort::Status();
  }

  system_dlc_plugin_ = std::make_shared<QnnBackendSystemDlcPlugin>(this);
  RETURN_IF_ERROR(system_dlc_plugin_->CreateDlc());

  system_dlc_created_ = true;
  return Ort::Status();
}

Ort::Status QnnBackendManager::ReleaseSystemDlcPlugin() {
  if (!system_dlc_created_) {
    return Ort::Status();
  }

  RETURN_IF_ERROR(system_dlc_plugin_->ReleaseDlc());
  system_dlc_plugin_.reset();

  system_dlc_created_ = false;
  return Ort::Status();
}

Ort::Status QnnBackendManager::AddContextToDlc() {
  RETURN_IF(system_dlc_plugin_ == nullptr, "Unexpected call of this function without DLC initialized.");
  RETURN_IF_NOT(GetQnnContextSize() == 1, "Expecting only one context to be added into DLC.");
  return system_dlc_plugin_->AddContextToDlc(GetQnnContext());
}

}  // namespace qnn
}  // namespace onnxruntime
