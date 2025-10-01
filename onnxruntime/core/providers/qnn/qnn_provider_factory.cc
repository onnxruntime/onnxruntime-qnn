// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <string>
#include <unordered_map>
#include <utility>
// #include "core/session/onnxruntime_c_api.h"
// #include "core/providers/shared_library/provider_api.h"
// #define ORT_API_MANUAL_INIT
// #include "core/session/onnxruntime_cxx_api.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_provider_factory_creator.h"
#include "core/providers/qnn/qnn_execution_provider.h"
// #include "core/framework/error_code_helper.h"


namespace onnxruntime {
struct QNNProviderFactory : IExecutionProviderFactory {
  QNNProviderFactory(const ProviderOptions& provider_options_map, const ConfigOptions* config_options)
      : provider_options_map_(provider_options_map), config_options_(config_options) {
  }

  ~QNNProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<QNNExecutionProvider>(provider_options_map_, config_options_, nullptr, nullptr);
  }

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override {
    const ConfigOptions& config_options = session_options.GetConfigOptions();
    const std::unordered_map<std::string, std::string>& config_options_map = config_options.GetConfigOptionsMap();

    // The implementation of the SessionOptionsAppendExecutionProvider C API function automatically adds EP options to
    // the session option configurations with the key prefix "ep.<lowercase_ep_name>.".
    // We extract those EP options and pass them to QNN EP as separate "provider options".
    std::unordered_map<std::string, std::string> provider_options = provider_options_map_;
    std::string key_prefix = "ep.";
    key_prefix += qnn::utils::GetLowercaseString(kQnnExecutionProvider);
    key_prefix += ".";

    for (const auto& [key, value] : config_options_map) {
      if (key.rfind(key_prefix, 0) == 0) {
        provider_options[key.substr(key_prefix.size())] = value;
      }
    }

    auto qnn_ep = std::make_unique<QNNExecutionProvider>(provider_options, &config_options, &session_options, &session_logger);
    return qnn_ep;
  }

 private:
  ProviderOptions provider_options_map_;
  const ConfigOptions* config_options_;
};

#if BUILD_QNN_EP_STATIC_LIB
std::shared_ptr<IExecutionProviderFactory> QNNProviderFactoryCreator::Create(const ProviderOptions& provider_options_map,
                                                                             const SessionOptions* session_options) {
  const ConfigOptions* config_options = nullptr;
  if (session_options != nullptr) {
    config_options = &session_options->config_options;
  }

  return std::make_shared<onnxruntime::QNNProviderFactory>(provider_options_map, config_options);
}
#else
struct QNN_Provider : Provider {
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* param) override {
    if (param == nullptr) {
      LOGS_DEFAULT(ERROR) << "[QNN EP] Passed NULL options to CreateExecutionProviderFactory()";
      return nullptr;
    }

    std::array<const void*, 2> pointers_array = *reinterpret_cast<const std::array<const void*, 2>*>(param);
    const ProviderOptions* provider_options = reinterpret_cast<const ProviderOptions*>(pointers_array[0]);
    const ConfigOptions* config_options = reinterpret_cast<const ConfigOptions*>(pointers_array[1]);

    if (provider_options == nullptr) {
      LOGS_DEFAULT(ERROR) << "[QNN EP] Passed NULL ProviderOptions to CreateExecutionProviderFactory()";
      return nullptr;
    }

    return std::make_shared<onnxruntime::QNNProviderFactory>(*provider_options, config_options);
  }

  Status CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                  const OrtKeyValuePairs* const* /*ep_metadata*/,
                                  size_t num_devices,
                                  ProviderOptions& provider_options,
                                  const OrtSessionOptions& session_options,
                                  const OrtLogger& logger,
                                  std::unique_ptr<IExecutionProvider>& ep) override {
    if (num_devices != 1) {
      return Status(common::ONNXRUNTIME, ORT_EP_FAIL, "QNN EP only supports one device.");
    }

    const ConfigOptions* config_options = &session_options.GetConfigOptions();

    std::array<const void*, 2> configs_array = {&provider_options, config_options};
    const void* arg = reinterpret_cast<const void*>(&configs_array);
    auto ep_factory = CreateExecutionProviderFactory(arg);
    ep = ep_factory->CreateProvider(session_options, logger);
    return Status::OK();
  }

  // OrtStatus* CreateEpFactories_ABI(const char* /*registration_name*/, const OrtApiBase* ort_api_base,
  //                             const OrtLogger* default_logger,
  //                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  //                               default_logger;
  //   const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);

  //   if (max_factories < 1) {
  //     return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
  //                                 "Not enough space to return EP factory. Need at least one.");
  //   }

  //   // Create the QNN EP factory using the ABI implementation
  //   onnxruntime::ApiPtrs api_ptrs{*ort_api, *ort_api->GetEpApi(), *ort_api->GetModelEditorApi()};
  //   auto factory_npu = std::make_unique<onnxruntime::QnnEpFactory>(onnxruntime::kQnnExecutionProvider, api_ptrs);

  //   factories[0] = factory_npu.release();
  //   *num_factories = 1;

  //   return nullptr;
  // }


  void Initialize() override {}
  void Shutdown() override {}
} g_provider;
#endif  // BUILD_QNN_EP_STATIC_LIB

}  // namespace onnxruntime

#if !BUILD_QNN_EP_STATIC_LIB
extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}

// #include "core/framework/error_code_helper.h"
// #include "onnxruntime_config.h"  // for ORT_VERSION

/// @brief Gets the path of directory containing the dynamic library that contains the address.
/// @param address An address of a function or variable in the dynamic library.
/// @return The path of the directory containing the dynamic library, or an empty string if the path cannot be determined.
// static onnxruntime::PathString GetDynamicLibraryLocationByAddress(const void* address) {
// #ifdef _WIN32
//   HMODULE moduleHandle;
//   if (!::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
//                             reinterpret_cast<LPCWSTR>(address), &moduleHandle)) {
//     return {};
//   }
//   std::wstring buffer;
//   for (std::uint32_t size{70}; size < 4096; size *= 2) {
//     buffer.resize(size, L'\0');
//     const std::uint32_t requiredSize = ::GetModuleFileNameW(moduleHandle, buffer.data(), size);
//     if (requiredSize == 0) {
//       break;
//     }
//     if (requiredSize == size) {
//       continue;
//     }
//     buffer.resize(requiredSize);
//     return {std::move(buffer)};
//   }
// #else
//   std::ignore = address;
// #endif
//   return {};
// }
#endif  // !BUILD_QNN_EP_STATIC_LIB
