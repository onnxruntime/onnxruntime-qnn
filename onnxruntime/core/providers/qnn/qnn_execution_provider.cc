// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/qnn_execution_provider.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <evntrace.h>
#endif

#include "HTP/QnnHtpGraph.h"

#include "core/providers/qnn/common/qnn_graph_utils.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/qnn_provider_factory.h"
#include "core/providers/qnn/shared_context.h"
#include "core/providers/qnn/qnn_allocator.h"
#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing.h"
#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_ep_input_graph_dumper.h"
#include "core/providers/qnn/builder/qnn_ep_sanitize_utils.h"
#include "core/providers/qnn/genie/genie_backend_manager.h"
#include "core/providers/qnn/builder/qnn_configs_helper.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/builder/qnn_thread_pool.h"
#include "core/providers/qnn/builder/op_package/op_package_parser.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/cache_compatibility/qnn_cache_compatibility_info.h"
#include "core/providers/qnn/cache_compatibility/qnn_cache_compatibility_manager.h"
#include "core/providers/qnn/htp_usr_drv_utils.h"
#include "core/providers/qnn/op_affinity/qnn_op_affinity_map.h"
#include "core/providers/qnn/qnn_ep_utils.h"
#include "core/providers/qnn/soc_utils.h"

// Forward declarations for NodeUnit-related classes
namespace onnxruntime {

static std::string MakeSharedLibraryPath(std::string_view name) {
#if defined(_WIN32)
  return std::string(name) + ".dll";
#else
  return "lib" + std::string(name) + ".so";
#endif
}

const std::string kDefaultCpuBackendPath = MakeSharedLibraryPath("QnnCpu");
const std::string kDefaultGenieBackendPath = MakeSharedLibraryPath("Genie");
const std::string kDefaultGpuBackendPath = MakeSharedLibraryPath("QnnGpu");
const std::string kDefaultHtpBackendPath = MakeSharedLibraryPath("QnnHtp");
const std::string kDefaultSaverBackendPath = MakeSharedLibraryPath("QnnSaver");
const std::string kDefaultIrBackendPath = MakeSharedLibraryPath("QnnIr");

// File-scope (unlike the other backend type name constants defined inside ParseBackendTypeName)
// because kGenieBackendTypeName is also referenced from other call sites in this file.
constexpr std::string_view kGenieBackendTypeName{"genie"};

constexpr const char* kQnnOpAffinityKey = "op_affinity";

// Best-effort mapping from a resolved backend library path to the QnnBackendType it will load.
// Both the 'backend_type' and 'backend_path' provider options funnel into a single backend_path
// string (ParseBackendTypeName turns the former into the latter).
static std::optional<qnn::QnnBackendType> InferBackendTypeFromPath(const std::string& backend_path) {
  const std::string filename = std::filesystem::path(backend_path).filename().string();

  if (filename == kDefaultCpuBackendPath) {
    return qnn::QnnBackendType::CPU;
  }
  if (filename == kDefaultGpuBackendPath) {
    return qnn::QnnBackendType::GPU;
  }
  if (filename == kDefaultHtpBackendPath) {
    return qnn::QnnBackendType::HTP;
  }
  return std::nullopt;
}

static bool ParseBackendTypeName(std::string_view backend_type_name,
                                 std::string& backend_path,
                                 const Ort::Logger& logger) {
  constexpr std::string_view kCpuBackendTypeName{"cpu"};
  constexpr std::string_view kGpuBackendTypeName{"gpu"};
  constexpr std::string_view kHtpBackendTypeName{"htp"};
  constexpr std::string_view kSaverBackendTypeName{"saver"};
  constexpr std::string_view kIrBackendTypeName{"ir"};

  constexpr std::array kAllowedBackendTypeNames{
      kCpuBackendTypeName,
      kGenieBackendTypeName,
      kGpuBackendTypeName,
      kHtpBackendTypeName,
      kSaverBackendTypeName,
      kIrBackendTypeName,
  };

  std::optional<std::string> associated_backend_path{};
  if (backend_type_name == kCpuBackendTypeName) {
    associated_backend_path = kDefaultCpuBackendPath;
  } else if (backend_type_name == kGenieBackendTypeName) {
    associated_backend_path = kDefaultGenieBackendPath;
  } else if (backend_type_name == kGpuBackendTypeName) {
    associated_backend_path = kDefaultGpuBackendPath;
  } else if (backend_type_name == kHtpBackendTypeName) {
    associated_backend_path = kDefaultHtpBackendPath;
  } else if (backend_type_name == kSaverBackendTypeName) {
    associated_backend_path = kDefaultSaverBackendPath;
  } else if (backend_type_name == kIrBackendTypeName) {
    associated_backend_path = kDefaultIrBackendPath;
  }

  if (associated_backend_path.has_value()) {
    backend_path = std::move(*associated_backend_path);
    return true;
  }

  std::ostringstream warning{};
  warning << "Invalid backend type name: " << backend_type_name << ". Allowed backend type names: ";
  for (size_t i = 0; i < kAllowedBackendTypeNames.size(); ++i) {
    warning << kAllowedBackendTypeNames[i];
    if (i + 1 < kAllowedBackendTypeNames.size()) {
      warning << ", ";
    }
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, warning.str().c_str());
  return false;
}

static GenieLog_Level_t ResolveGenieLogLevel(std::string_view lvl) {
  if (lvl == "warn") return GENIE_LOG_LEVEL_WARN;
  if (lvl == "verbose") return GENIE_LOG_LEVEL_VERBOSE;
  if (lvl == "info") return GENIE_LOG_LEVEL_INFO;
  return GENIE_LOG_LEVEL_ERROR;
}

static void ParseProfilingLevel(std::string profiling_level_string,
                                qnn::ProfilingLevel& profiling_level,
                                const Ort::Logger& logger) {
  std::transform(profiling_level_string.begin(), profiling_level_string.end(),
                 profiling_level_string.begin(), [](unsigned char c) { return std::tolower(c); });
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO, ("profiling_level: " + profiling_level_string).c_str());
  if (profiling_level_string == "off") {
    profiling_level = qnn::ProfilingLevel::OFF;
  } else if (profiling_level_string == "basic") {
    profiling_level = qnn::ProfilingLevel::BASIC;
  } else if (profiling_level_string == "detailed") {
    profiling_level = qnn::ProfilingLevel::DETAILED;
  } else if (profiling_level_string == "optrace") {
    profiling_level = qnn::ProfilingLevel::OPTRACE;
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "Profiling level not valid.");
  }
}

static void ParseHtpPerformanceMode(std::string htp_performance_mode_string,
                                    qnn::HtpPerformanceMode& htp_performance_mode,
                                    const Ort::Logger& logger) {
  std::transform(htp_performance_mode_string.begin(), htp_performance_mode_string.end(),
                 htp_performance_mode_string.begin(), [](unsigned char c) { return std::tolower(c); });
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Htp performance mode: " + htp_performance_mode_string).c_str());
  if (htp_performance_mode_string == "burst") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpBurst;
  } else if (htp_performance_mode_string == "balanced") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpBalanced;
  } else if (htp_performance_mode_string == "default") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  } else if (htp_performance_mode_string == "high_performance") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpHighPerformance;
  } else if (htp_performance_mode_string == "high_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpHighPowerSaver;
  } else if (htp_performance_mode_string == "low_balanced") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpLowBalanced;
  } else if (htp_performance_mode_string == "low_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpLowPowerSaver;
  } else if (htp_performance_mode_string == "power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpPowerSaver;
  } else if (htp_performance_mode_string == "extreme_power_saver") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpExtremePowerSaver;
  } else if (htp_performance_mode_string == "sustained_high_performance") {
    htp_performance_mode = qnn::HtpPerformanceMode::kHtpSustainedHighPerformance;
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "Htp performance mode not valid.");
  }
}

static void ParseQnnContextPriority(std::string context_priority_string,
                                    qnn::ContextPriority& context_priority,
                                    const Ort::Logger& logger) {
  std::transform(context_priority_string.begin(), context_priority_string.end(),
                 context_priority_string.begin(), [](unsigned char c) { return std::tolower(c); });
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("QNN context priority: " + context_priority_string).c_str());
  if (context_priority_string == "low") {
    context_priority = qnn::ContextPriority::LOW;
  } else if (context_priority_string == "normal_low") {
    context_priority = qnn::ContextPriority::NORMAL_LOW;
  } else if (context_priority_string == "normal") {
    context_priority = qnn::ContextPriority::NORMAL;
  } else if (context_priority_string == "normal_high") {
    context_priority = qnn::ContextPriority::NORMAL_HIGH;
  } else if (context_priority_string == "high") {
    context_priority = qnn::ContextPriority::HIGH;
  } else if (context_priority_string == "high_plus") {
    context_priority = qnn::ContextPriority::HIGH_PLUS;
  } else if (context_priority_string == "critical") {
    context_priority = qnn::ContextPriority::CRITICAL;
  } else if (context_priority_string == "critical_plus") {
    context_priority = qnn::ContextPriority::CRITICAL_PLUS;
  } else {
    context_priority = qnn::ContextPriority::UNDEFINED;
    std::string msg = "QNN context priority: " + context_priority_string + " not valid, set to undefined.";
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, msg.c_str());
  }
}

static void ParseHtpGraphFinalizationOptimizationMode(
    const std::string& htp_graph_finalization_opt_mode_string,
    qnn::HtpGraphFinalizationOptimizationMode& htp_graph_finalization_opt_mode,
    const Ort::Logger& logger) {
  ORT_CXX_LOG(logger,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("HTP graph finalization optimization mode: " + htp_graph_finalization_opt_mode_string).c_str());

  if (htp_graph_finalization_opt_mode_string.empty() || htp_graph_finalization_opt_mode_string == "0") {
    htp_graph_finalization_opt_mode = qnn::HtpGraphFinalizationOptimizationMode::kDefault;
  } else if (htp_graph_finalization_opt_mode_string == "1") {
    htp_graph_finalization_opt_mode = qnn::HtpGraphFinalizationOptimizationMode::kMode1;
  } else if (htp_graph_finalization_opt_mode_string == "2") {
    htp_graph_finalization_opt_mode = qnn::HtpGraphFinalizationOptimizationMode::kMode2;
  } else if (htp_graph_finalization_opt_mode_string == "3") {
    htp_graph_finalization_opt_mode = qnn::HtpGraphFinalizationOptimizationMode::kMode3;
  } else {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_WARNING,
                ("Invalid HTP graph finalization optimization mode: " + htp_graph_finalization_opt_mode_string).c_str());
  }
}

static void ParseVtcmSize(const std::string& vtcm_size_in_mb_string,
                          int32_t& vtcm_size_in_mb,
                          const Ort::Logger& logger) {
  try {
    vtcm_size_in_mb = std::stoi(vtcm_size_in_mb_string);
  } catch (const std::invalid_argument& /*ex*/) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "Ignoring malformed VTCM size, expecting a >0 integer.");
  } catch (const std::out_of_range& /*ex*/) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "Ignoring malformed VTCM size, expecting a >0 integer.");
  }

  if (vtcm_size_in_mb <= 0) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_WARNING,
                ("Invalid vtcm_mb: " + vtcm_size_in_mb_string + " will be skipped").c_str());
  }
}

static void ParseHtpArchitecture(const std::string& htp_arch_string,
                                 QnnHtpDevice_Arch_t& qnn_htp_arch,
                                 const Ort::Logger& logger) {
  if (htp_arch_string.empty() || htp_arch_string == "0") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  } else if (htp_arch_string == "68") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V68;
  } else if (htp_arch_string == "69") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V69;
  } else if (htp_arch_string == "73") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V73;
  } else if (htp_arch_string == "75") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V75;
  } else if (htp_arch_string == "79") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V79;
  } else if (htp_arch_string == "81") {
    qnn_htp_arch = QNN_HTP_DEVICE_ARCH_V81;
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, ("Invalid HTP architecture: " + htp_arch_string).c_str());
  }
}

static void ParseSocModel(const std::string& soc_model_string, uint32_t& soc_model, const Ort::Logger& logger) {
  // First try a chip-family name lookup (e.g. "SM8750", case-insensitive).
  uint32_t name_value = qnn::soc::SocModelFromName(soc_model_string);
  if (name_value != 0) {
    soc_model = name_value;
    return;
  }

  // Fall back to integer parsing for numeric IDs (e.g. "69", "43").
  int value = 0;
  try {
    value = std::stoi(soc_model_string);
  } catch (const std::invalid_argument& /*ex*/) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("Unrecognized soc_model '" + soc_model_string +
                 "'. Expected a numeric ID (e.g. 43, 69) or a chip name (e.g. SM8550, SM8750).")
                    .c_str());
    return;
  } catch (const std::out_of_range& /*ex*/) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("Unrecognized soc_model '" + soc_model_string +
                 "'. Expected a numeric ID (e.g. 43, 69) or a chip name (e.g. SM8550, SM8750).")
                    .c_str());
    return;
  }

  if (value < 0) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, ("Invalid soc_model: " + soc_model_string).c_str());
  } else {
    soc_model = static_cast<uint32_t>(value);
  }
}

static bool ParseBoolOption(const OrtApi& ort_api,
                            const OrtSessionOptions& session_options,
                            const std::string& key,
                            bool default_value,
                            const Ort::Logger& logger) {
  bool result = default_value;
  std::string value_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options, key, default_value ? "1" : "0", value_str);

  if ("1" == value_str) {
    result = true;
  } else if ("0" == value_str) {
    result = false;
  } else {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Invalid value for " + key + " (" + value_str + "). Only 0 or 1 allowed.").c_str());
  }
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("User specified " + key + ": " + (result ? "1" : "0")).c_str());

  return result;
}

// Creates `dir` (and any missing parents) and verifies it is writable by
// round-tripping a small probe file. Returns true on success. On failure,
// logs a WARNING tagged with `feature_name` so callers can disable the
// associated feature flag with a clear log trail. Used by every
// QNN-EP-side dump option whose output is written incrementally during
// session run (so a non-writable directory should disable the feature at
// session-start rather than mid-inference).
static bool ProbeDumpDirectoryWritable(const std::string& dir,
                                       const std::string& feature_name,
                                       const Ort::Logger& logger) {
  std::filesystem::path probe_dir(dir);
  std::error_code ec;
  std::filesystem::create_directories(probe_dir, ec);
  if (ec) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                (feature_name + " directory could not be created: " + probe_dir.string() +
                 " (" + ec.message() + "); the feature will be disabled.")
                    .c_str());
    return false;
  }
  std::filesystem::path probe_file = probe_dir / ".qnn_ep_dump_probe";
  bool ok = false;
  {
    std::ofstream ofs(probe_file);
    ok = ofs.is_open() && (ofs << "1").good();
  }
  std::filesystem::remove(probe_file, ec);
  if (!ok) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                (feature_name + " directory not writable: " + probe_dir.string() +
                 "; the feature will be disabled.")
                    .c_str());
    return false;
  }
  return true;
}

void QnnEp::ParsePerSocHtpConfigs() {
  std::string soc_model_per_soc_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("soc_model"), "", soc_model_per_soc_str);

  std::string htp_arch_per_soc_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("htp_arch"), "", htp_arch_per_soc_str);

  if (soc_model_per_soc_str.find(',') == std::string::npos && htp_arch_per_soc_str.find(',') == std::string::npos) {
    // Not multi-SoC configs.
    enable_multi_soc_ep_context_ = false;
    return;
  }

  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Parsing multi-SoC HTP backend configurations.");

  if (!soc_model_per_soc_str.empty()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("User specified soc_model: " + soc_model_per_soc_str).c_str());
    for (std::string_view soc_model_str : qnn::utils::SplitString(soc_model_per_soc_str, ",")) {
      uint32_t soc_model = QNN_SOC_MODEL_UNKNOWN;
      ParseSocModel(std::string(soc_model_str), soc_model, logger_);
      soc_model_per_soc_.push_back(soc_model);
    }
  }

  if (!htp_arch_per_soc_str.empty()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("User specified htp_arch: " + htp_arch_per_soc_str).c_str());
    for (std::string_view htp_arch_str : qnn::utils::SplitString(htp_arch_per_soc_str, ",")) {
      QnnHtpDevice_Arch_t htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
      ParseHtpArchitecture(std::string(htp_arch_str), htp_arch, logger_);
      htp_arch_per_soc_.push_back(htp_arch);
    }
  }

  if (!soc_model_per_soc_.empty() && !htp_arch_per_soc_.empty()) {
    if (soc_model_per_soc_.size() == htp_arch_per_soc_.size()) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_WARNING,
                  "Both soc_model and htp_arch are given but soc_model has higher priority if they do not match.");
    } else {
      LOG_AND_THROW_ERROR(logger_,
                          "Expecting soc_model and htp_arch having equal number of values in multi-SoC EP context.");
    }
  } else if (htp_arch_per_soc_.empty()) {
    htp_arch_per_soc_.assign(soc_model_per_soc_.size(), QNN_HTP_DEVICE_ARCH_NONE);
  } else {
    soc_model_per_soc_.assign(htp_arch_per_soc_.size(), QNN_SOC_MODEL_UNKNOWN);
  }

  const size_t num_socs = soc_model_per_soc_.size();

  // HTP config-related options below are allowed to be given in 0, 1 or N values, where N is equal to the number
  // of htp_arch/soc_model values parsed above. If not given, default value is applied and duplicate to match the
  // expected size, If only 1 value is given, it is duplicate to match the expected size as well.

  auto parse_per_soc_option = [&](const std::string& option_name, auto default_value, auto parse_element) {
    std::string option_str;
    GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey(option_name), "", option_str);
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified " + option_name + ": " + option_str).c_str());

    std::vector<decltype(default_value)> values;
    values.reserve(num_socs);

    if (option_str.empty()) {
      values.assign(num_socs, default_value);
      return values;
    }

    for (std::string_view token : qnn::utils::SplitString(option_str, ",")) {
      values.push_back(parse_element(token));
    }

    if (values.size() == 1) {
      values.assign(num_socs, values[0]);
    } else if (values.size() != num_socs) {
      LOG_AND_THROW_ERROR(logger_,
                          ("Expecting " + option_name + " having equal number of values with htp_arch/soc_model.")
                              .c_str());
    }

    return values;
  };

  // Per-SoC vtcm size.
  std::vector<int32_t> vtcm_size_in_mb_per_soc = parse_per_soc_option(
      "vtcm_mb",
      htp_graph_configs_.vtcm_size_in_mb,
      [this](std::string_view token) {
        int32_t vtcm_size_in_mb = htp_graph_configs_.vtcm_size_in_mb;
        ParseVtcmSize(std::string(token), vtcm_size_in_mb, logger_);
        return vtcm_size_in_mb;
      });

  // Construct Per-SoC HTP configs.
  htp_graph_configs_per_soc_.reserve(num_socs);
  for (size_t idx = 0; idx < num_socs; ++idx) {
    qnn::HtpGraphConfigs_t config{vtcm_size_in_mb_per_soc[idx],
                                  htp_graph_configs_.htp_graph_finalization_opt_mode,
                                  htp_graph_configs_.enable_htp_fp16_precision,
                                  htp_graph_configs_.enable_htp_monolithic_lstm,
                                  htp_graph_configs_.enable_htp_fp16_clamp_overflow};
    htp_graph_configs_per_soc_.push_back(std::move(config));
  }

  enable_multi_soc_ep_context_ = true;
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "Successfully parsed multi-SoC HTP backend configurations.");
}

#ifdef _WIN32
static qnn::ProfilingLevel GetProfilingLevelFromETWLevel(unsigned char level, const Ort::Logger& logger) {
  std::string level_string = std::to_string(static_cast<int>(level));
  if (level == 5) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_INFO,
                ("Overriding profiling to basic based on ETW level: " + level_string).c_str());
    return qnn::ProfilingLevel::BASIC;
  } else if (level < 5) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_INFO,
                ("QNN Profiler ETW level not supported below level 5. Level: " + level_string).c_str());
    return qnn::ProfilingLevel::OFF;
  } else {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_INFO,
                ("Overriding profiling to detailed based on ETW level: " + level_string).c_str());
    return qnn::ProfilingLevel::DETAILED;
  }
}
#endif  // defined(_WIN32)

std::unique_ptr<qnn::QnnSerializerConfig> QnnEp::InitQnnSerializerConfig() const {
  // Enable use of QNN Saver if the user provides a path the QNN Saver backend library.
  static const std::string QNN_SAVER_PATH_KEY = "qnn_saver_path";
  std::string qnn_saver_path;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey(QNN_SAVER_PATH_KEY),
                                 "",
                                 qnn_saver_path);
  if (!qnn_saver_path.empty()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("User specified QNN Saver path: " + qnn_saver_path).c_str());
    return qnn::QnnSerializerConfig::CreateSaver(qnn_saver_path);
  }

  static const std::string DUMP_QNN_IR_DLC = "dump_qnn_ir_dlc";
  auto dump_qnn_ir_dlc = ParseBoolOption(ort_api,
                                         session_options_,
                                         FormatEPConfigKey(DUMP_QNN_IR_DLC),
                                         false,
                                         logger_);

  static const std::string DUMP_QNN_IR_DLC_DIR = "dump_qnn_ir_dlc_dir";
  std::string qnn_ir_dlc_dir;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey(DUMP_QNN_IR_DLC_DIR),
                                 "",
                                 qnn_ir_dlc_dir);
  if (!qnn_ir_dlc_dir.empty()) {
    if (dump_qnn_ir_dlc) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, ("IR DLC directory: " + qnn_ir_dlc_dir).c_str());
    } else {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_WARNING,
                  "Provided a directory for dumping QNN graphs to DLC, but did not set dump_qnn_ir_dlc to 1.");
    }
  }

  static const std::string QNN_IR_BACKEND_PATH = "qnn_ir_backend_path";
  std::string qnn_ir_backend_path = kDefaultIrBackendPath;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey(QNN_IR_BACKEND_PATH),
                                 kDefaultIrBackendPath,
                                 qnn_ir_backend_path);
  if (qnn_ir_backend_path != kDefaultIrBackendPath) {
    if (dump_qnn_ir_dlc) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, ("IR  backend path: " + qnn_ir_backend_path).c_str());
    } else {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_WARNING,
                  "Provided a path to the Ir backend for dumping QNN graphs to DLC, "
                  "but did not set dump_qnn_ir_dlc to 1.");
    }
  }

  if (dump_qnn_ir_dlc) {
    return qnn::QnnSerializerConfig::CreateIr(std::move(qnn_ir_backend_path), std::move(qnn_ir_dlc_dir));
  }

  return nullptr;
}

QnnEp::QnnEp(QnnEpFactory& factory,
             const std::string& name,
             const OrtSessionOptions& session_options,
             const OrtLogger* logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      // factory_{factory},
      name_{name},
      logger_{Ort::Logger(logger)},
      session_options_{session_options} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;
  CreateAllocator = CreateAllocatorImpl;
  SetDynamicOptions = SetDynamicOptionsImpl;
  GetCompiledModelCompatibilityInfo = GetCompiledModelCompatibilityInfoImpl;

  // Initialize from session options
  {
    // Get disable_cpu_ep_fallback setting from session options
    std::string disable_cpu_ep_fallback_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionsDisableCPUEPFallback,
                                   "0",
                                   disable_cpu_ep_fallback_str);
    disable_cpu_ep_fallback_ = disable_cpu_ep_fallback_str == "1";

    // Get context cache settings
    std::string context_cache_enabled_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionEpContextEnable,
                                   "0",
                                   context_cache_enabled_str);
    context_cache_enabled_ = context_cache_enabled_str == "1";
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Context cache enable: " + context_cache_enabled_str).c_str());

    std::string embed_mode;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionEpContextEmbedMode,
                                   "0",
                                   embed_mode);
    if ("1" == embed_mode) {
      qnn_context_embed_mode_ = true;
    } else if ("0" == embed_mode) {
      qnn_context_embed_mode_ = false;
    } else {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("Invalid ep.context_embed_mode: " + embed_mode + " only 0 or 1 allowed. Set to 1.").c_str());
    }
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified context cache embed mode " + embed_mode).c_str());

    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionEpContextFilePath,
                                   "",
                                   context_cache_path_cfg_);
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified context cache path: " + context_cache_path_cfg_).c_str());

    std::string genie_log_level;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   FormatEPConfigKey("genie_log_level"),
                                   "",
                                   genie_log_level);
    genie_log_level_ = ResolveGenieLogLevel(genie_log_level);
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified Genie Log level: " + genie_log_level).c_str());

    // For the case that workaround QNN context PD memory limit, user need split the model into pieces and
    // generate the QNN context model separately.
    // It could happen that the generated EPContext node in separate graph has same node name.
    // User can set this context_node_name_prefix for each split pieces to avoid that happens.
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionEpContextNodeNamePrefix,
                                   "",
                                   context_node_name_prefix_);
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified QNN context node name prefix: " + context_node_name_prefix_).c_str());

    std::string share_ep_contexts_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionShareEpContexts,
                                   "0",
                                   share_ep_contexts_str);
    share_ep_contexts_ = share_ep_contexts_str == "1";
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified option - share EP contexts across sessions: " + share_ep_contexts_str).c_str());

    std::string stop_share_ep_contexts_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   kOrtSessionOptionStopShareEpContexts,
                                   "0",
                                   stop_share_ep_contexts_str);
    stop_share_ep_contexts_ = stop_share_ep_contexts_str == "1";
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified option - stop share EP contexts across sessions: " + stop_share_ep_contexts_str).c_str());

    std::string prepare_only_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   FormatEPConfigKey("enable_htp_prepare_only"),
                                   "0",
                                   prepare_only_str);
    prepare_only_ = prepare_only_str == "1";
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified option - enable_htp_prepare_only: " + prepare_only_str).c_str());

    if (prepare_only_ && !context_cache_enabled_) {
      throw std::runtime_error(
          "enable_htp_prepare_only=1 requires ep.context_enable=1. "
          "prepare_only mode only generates the context model for ahead-of-time compilation.");
    }

    std::string prepare_and_load_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   FormatEPConfigKey("enable_htp_prepare_and_load"),
                                   "0",
                                   prepare_and_load_str);
    prepare_and_load_ = prepare_and_load_str == "1";
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified option - enable_htp_prepare_and_load: " + prepare_and_load_str).c_str());

    if (prepare_and_load_ && prepare_only_) {
      throw std::runtime_error(
          "enable_htp_prepare_and_load and enable_htp_prepare_only are mutually exclusive. "
          "Set only one of them to 1.");
    }

    if (prepare_and_load_ && !context_cache_enabled_ && !context_cache_path_cfg_.empty()) {
      throw std::runtime_error(
          "Contradictory options: prepare_and_load=1 with context_enable=0 means no artifact is persisted, "
          "but ep.context_file_path is explicitly set. Either set context_enable=1 to persist the artifact, "
          "or remove context_file_path.");
    }
  }

  std::string backend_path = kDefaultHtpBackendPath;
  {
    std::optional<std::string> backend_path_from_options{};

    // Get backend type and path from session options
    std::string backend_type;
    std::string backend_path_option;

    GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("backend_type"), "", backend_type);
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   FormatEPConfigKey("backend_path"),
                                   "",
                                   backend_path_option);

    // Check if both options are provided
    if (!backend_type.empty() && !backend_path_option.empty()) {
      throw std::runtime_error("Only one of 'backend_type' and 'backend_path' should be set.");
    }
    if (!backend_type.empty()) {
      if (std::string parsed_backend_path; ParseBackendTypeName(backend_type, parsed_backend_path, logger_)) {
        backend_path_from_options = parsed_backend_path;
      } else {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Failed to parse 'backend_type' value.");
      }
    } else if (!backend_path_option.empty()) {
      backend_path_from_options = backend_path_option;
    }

    // Use the determined backend path or default
    if (backend_path_from_options.has_value()) {
      backend_path = std::move(*backend_path_from_options);
    } else {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Using default backend path: " + backend_path).c_str());
    }

    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Using backend path: " + backend_path).c_str());
  }

  std::unique_ptr<qnn::QnnSerializerConfig> qnn_serializer_config = InitQnnSerializerConfig();

  std::string profiling_file_path;
  static const std::string PROFILING_LEVEL = "profiling_level";
  qnn::ProfilingLevel profiling_level = qnn::ProfilingLevel::OFF;
  // separate out the profiling level for ETW in case it gets disabled later when we extract the events
  // set to invalid to indicate that ETW is no enabled when we setup QNN
  qnn::ProfilingLevel profiling_level_etw = qnn::ProfilingLevel::INVALID;

#ifdef _WIN32
  auto& provider = qnn::QnnTelemetry::Instance();
  if (provider.IsEnabled()) {
    auto level = provider.Level();
    auto keyword = provider.Keyword();
    if ((keyword & static_cast<uint64_t>(qnn::ORTTraceLoggingKeyword::Profiling)) != 0) {
      if (level != 0) {
        profiling_level_etw = GetProfilingLevelFromETWLevel(level, logger_);
      }
    }
  }
#endif  // defined(_WIN32)

  // Get profiling settings from session options
  std::string profiling_level_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey(PROFILING_LEVEL),
                                 "off",
                                 profiling_level_str);
  ParseProfilingLevel(profiling_level_str, profiling_level, logger_);

  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("profiling_file_path"),
                                 "",
                                 profiling_file_path);
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Profiling file path: " + profiling_file_path).c_str());

  // Get RPC control latency from session options
  std::string rpc_control_latency_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("rpc_control_latency"),
                                 "0",
                                 rpc_control_latency_str);
  if (!rpc_control_latency_str.empty() && rpc_control_latency_str != "0") {
    default_rpc_control_latency_ = static_cast<uint32_t>(std::stoul(rpc_control_latency_str));
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("rpc_control_latency: " + rpc_control_latency_str).c_str());
  }

  // default_htp_performance_mode from QNN EP option.
  // set it once only for each thread as default so user don't need to set it for every session run
  std::string htp_performance_mode_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("htp_performance_mode"),
                                 "",
                                 htp_performance_mode_str);
  if (!htp_performance_mode_str.empty()) {
    ParseHtpPerformanceMode(htp_performance_mode_str, default_htp_performance_mode_, logger_);

    if (qnn::HtpPerformanceMode::kHtpBurst == default_htp_performance_mode_) {
      default_rpc_polling_time_ = 9999;
    }
  }

  // QNN context priority
  qnn::ContextPriority context_priority = qnn::ContextPriority::NORMAL;
  std::string context_priority_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("qnn_context_priority"),
                                 "",
                                 context_priority_str);
  if (!context_priority_str.empty()) {
    ParseQnnContextPriority(context_priority_str, context_priority, logger_);
  }

  // HTP share resource optimization
  std::string htp_share_resource_optimization_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("htp_share_resource_optimization"),
                                 "",
                                 htp_share_resource_optimization_str);

  std::string enable_vtcm_backup_buffer_sharing_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("enable_vtcm_backup_buffer_sharing"),
                                 "0",
                                 enable_vtcm_backup_buffer_sharing_str);

  if (htp_share_resource_optimization_str == "1") {
    // htp_share_resource_optimization=1 overrides enable_vtcm_backup_buffer_sharing regardless of its value
    htp_share_resource_optimization_ = 1;
  } else if (!htp_share_resource_optimization_str.empty()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                ("Invalid value entered for htp_share_resource_optimization: " + htp_share_resource_optimization_str + ", only 1 is allowed.").c_str());
  } else if (enable_vtcm_backup_buffer_sharing_str == "1") {
    // htp_share_resource_optimization not set, fall back to enable_vtcm_backup_buffer_sharing
    htp_share_resource_optimization_ = 1;
  }

  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("htp_share_resource_optimization: " + std::to_string(htp_share_resource_optimization_)).c_str());

#if QNN_API_VERSION_MAJOR < 2 || ((QNN_API_VERSION_MAJOR) == 2 && (QNN_API_VERSION_MINOR < 26))
  if (htp_share_resource_optimization_ == 1) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING,
                "User specified htp_share_resource_optimization but QNN API version is older than 2.26.");
  }
#endif

  std::string disable_file_mapped_weights_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("disable_file_mapped_weights"), "0", disable_file_mapped_weights_str);
  if (disable_file_mapped_weights_str == "1") {
    enable_file_mapped_weights_ = false;
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING, ("User specified disable_file_mapped_weights: " + std::to_string(!enable_file_mapped_weights_)).c_str());
  }

#ifndef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
  enable_file_mapped_weights_ = false;
  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_WARNING, "File mapped weights feature is only available on Windows arm64 devices for QNN API versions >= 2.32. Feature will be disabled by default");
#else
  if (qnn_context_embed_mode_ && enable_file_mapped_weights_) {
    enable_file_mapped_weights_ = false;
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING, "File mapped weights feature is incompatible with embedded EP contexts. Feature will be disabled by default.");
  }
#endif

  std::string device_id_str;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("device_id"), "0", device_id_str);
  if (!device_id_str.empty()) {
    int value = std::stoi(device_id_str);
    if (value < 0) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_WARNING,
                  ("Invalid device ID '" +
                   device_id_str +
                   "', only >= 0 allowed. Set to " +
                   std::to_string(device_id_))
                      .c_str());
    } else {
      device_id_ = static_cast<uint32_t>(value);
    }
  }

  // Op packages
  std::string op_packages_str;
  std::vector<onnxruntime::qnn::OpPackage> op_packages;
  GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("op_packages"), "", op_packages_str);
  if (!op_packages_str.empty()) {
    ParseOpPackages(op_packages_str, op_packages, logger_);
  }

  // HTP graph finalization optimization mode
  std::string htp_graph_finalization_opt_mode_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("htp_graph_finalization_optimization_mode"),
                                 "",
                                 htp_graph_finalization_opt_mode_str);
  if (!htp_graph_finalization_opt_mode_str.empty()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified htp_graph_finalization_optimization_mode: " + htp_graph_finalization_opt_mode_str)
                    .c_str());
    ParseHtpGraphFinalizationOptimizationMode(htp_graph_finalization_opt_mode_str,
                                              htp_graph_configs_.htp_graph_finalization_opt_mode,
                                              logger_);
  }

  // HTP FP16 precision mode
  htp_graph_configs_.enable_htp_fp16_precision = ParseBoolOption(ort_api,
                                                                 session_options_,
                                                                 FormatEPConfigKey("enable_htp_fp16_precision"),
                                                                 false,
                                                                 logger_);

  // HTP monolithic lstm. Default false: lower LSTM as ORT-side per-timestep unrolled cells
  // (expand at ORT). Set to true to run the monolithic LSTM kernel on HTP.
  static constexpr const char* kEnableHtpMonolithicLstm = "enable_htp_monolithic_lstm";
  htp_graph_configs_.enable_htp_monolithic_lstm = ParseBoolOption(ort_api,
                                                                  session_options_,
                                                                  FormatEPConfigKey(kEnableHtpMonolithicLstm),
                                                                  false,
                                                                  logger_);
  model_settings_.enable_htp_monolithic_lstm = htp_graph_configs_.enable_htp_monolithic_lstm;

  // HTP fp16 clamp overflow. Default false. On HTP Arch v79+ (e.g. Glymur/v81),
  // fp16 Conv accumulator overflow becomes NaN and propagates to the output
  // (pre-v79 kept it finite). When true, saturate such overflow to the fp16 max
  // value instead of NaN. Requires QAIRT SDK >= 2.49.
  static constexpr const char* kEnableHtpFp16ClampOverflow = "enable_htp_fp16_clamp_overflow";
  htp_graph_configs_.enable_htp_fp16_clamp_overflow = ParseBoolOption(ort_api,
                                                                      session_options_,
                                                                      FormatEPConfigKey(kEnableHtpFp16ClampOverflow),
                                                                      false,
                                                                      logger_);
#ifndef QNN_HTP_FP16_CLAMP_OVERFLOW_AVAILABLE
  // Warn once at parse time (not per graph) if the option was requested but the SDK lacks support.
  if (htp_graph_configs_.enable_htp_fp16_clamp_overflow) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING,
                "enable_htp_fp16_clamp_overflow was requested but the QNN HTP SDK in use does "
                "not support it (requires QAIRT >= 2.49). Ignoring.");
  }
#endif

  // Try to parse multi-SoC HTP options first. If not multi-SoC htp_arch/soc_model is given, fallback to normal parsing.
  ParsePerSocHtpConfigs();
  // Declare outside the if scope since there are users later. They may be overwritten in the else branch.
  QnnHtpDevice_Arch_t htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
  uint32_t soc_model = QNN_SOC_MODEL_UNKNOWN;
  if (enable_multi_soc_ep_context_) {
#if defined(__aarch64__) || defined(_M_ARM64) || (defined(_M_ARM64EC))
    // Only enable on x86 platforms.
    LOG_AND_THROW_ERROR(logger_, "Multi-SoC EP context is only supported on x86 platforms and offline preparation.");
#endif  // defined(__aarch64__) || defined(_M_ARM64) || (defined(_M_ARM64EC))
    if (!context_cache_enabled_) {
      LOG_AND_THROW_ERROR(logger_, "Per-SoC configurations are only supported for EP context enabled.");
    }
    if ((share_ep_contexts_ || stop_share_ep_contexts_)) {
      LOG_AND_THROW_ERROR(logger_, "Multi-SoC EP context is currently unsupported with shared EP context usage.");
    }

    // Exploit prepare-only flag to avoid unexpected usage in overall workflow (e.g., no execution).
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_INFO,
                "Enable 'enable_htp_prepare_only' by default for multi-SoC EP context.");
    prepare_only_ = true;
  } else {
    // HTP architecture
    std::string htp_arch_str;
    GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("htp_arch"), "", htp_arch_str);
    if (!htp_arch_str.empty()) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("User specified htp_arch: " + htp_arch_str).c_str());
      ParseHtpArchitecture(htp_arch_str, htp_arch, logger_);
    }

    // SoC model
    std::string soc_model_str;
    GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("soc_model"), "0", soc_model_str);
    if (!soc_model_str.empty()) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("User specified soc_model: " + soc_model_str).c_str());
      ParseSocModel(soc_model_str, soc_model, logger_);
    }

    // VTCM MB
    std::string vtcm_mb_str;
    GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey("vtcm_mb"), "0", vtcm_mb_str);
    if (!vtcm_mb_str.empty() && vtcm_mb_str != "0") {
      ParseVtcmSize(vtcm_mb_str, htp_graph_configs_.vtcm_size_in_mb, logger_);
    }
  }

  // Parallel graph prepare.
  std::string num_graph_prepare_threads_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("num_graph_prepare_threads"),
                                 "",
                                 num_graph_prepare_threads_str);
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  uint8_t max_num_supported_threads = static_cast<uint8_t>(std::thread::hardware_concurrency());
  if (max_num_supported_threads) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Number of supported concurrent threads: " + std::to_string(max_num_supported_threads)).c_str());
  } else {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                "Unable to retrieve number of supported concurrent threads from hardware. Setting max to default value of 4.");
    max_num_supported_threads = 4;
  }
  // 8 threads provided the best initialization performance from testing
  // Default to max number of supported threads if less than 8. Otherwise default to 8 threads
  uint8_t def_num_graph_prepare_threads = max_num_supported_threads > 8 ? 8 : max_num_supported_threads;

  auto is_valid_number = [this](const std::string& s) {
    if (s[0] == '0') {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_ERROR,
                  "num_graph_prepare_threads cannot be 0 or start with 0");
      return false;
    }

    auto it = std::find_if(s.begin(), s.end(), [](const char c) {
      return !std::isdigit(c);
    });

    if (it == s.end()) {
      return true;
    }

    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                "num_graph_prepare_threads must be a positive number");
    return true;
  };
#endif

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  if (!num_graph_prepare_threads_str.empty() && is_valid_number(num_graph_prepare_threads_str)) {
    uint8_t value = static_cast<uint8_t>(std::stoi(num_graph_prepare_threads_str));
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("User specified num_graph_prepare_threads: " + std::to_string(value)).c_str());

    if (value > max_num_supported_threads) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_WARNING,
                  ("Specified number of graph prepare threads (" + std::to_string(value) + ") is outside of the allowable range [1," + std::to_string(max_num_supported_threads) + "]. Defaulting to " + std::to_string(def_num_graph_prepare_threads) + " threads.").c_str());
      num_graph_prepare_threads_ = def_num_graph_prepare_threads;
    } else {
      num_graph_prepare_threads_ = value;
    }
  } else {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Using default number threads for graph prepare: " + std::to_string(def_num_graph_prepare_threads)).c_str());
    num_graph_prepare_threads_ = def_num_graph_prepare_threads;
  }
#else
  if (!num_graph_prepare_threads_str.empty()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                "Multi-threaded graph compilation is currently only supported on Windows devices. Feature will not be enabled.");
  }
#endif  // _WIN32 && (defined(__aarch64__) || defined(_M_ARM64))

  // Check for conflicts
  if (qnn_context_embed_mode_ && share_ep_contexts_) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                "[EP context generation:] Weight sharing enabled conflict with EP context embed mode. "
                "Inference will not work as expected!");
  }

  if (qnn_context_embed_mode_ && htp_share_resource_optimization_ == 1) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_ERROR,
                "[EP context generation:] HTP share resource optimization enabled conflict with EP context embed mode. "
                "Inference will not work as expected!");
  }

  // HTP spill fill buffer
  std::string enable_htp_spill_fill_buffer_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey("enable_htp_spill_fill_buffer"),
                                 "0",
                                 enable_htp_spill_fill_buffer_str);
  enable_spill_fill_buffer_ = enable_htp_spill_fill_buffer_str == "1";

  model_settings_.offload_graph_io_quantization = ParseBoolOption(ort_api,
                                                                  session_options_,
                                                                  FormatEPConfigKey("offload_graph_io_quantization"),
                                                                  true,
                                                                  logger_);

  model_settings_.htp_bf16_enable = ParseBoolOption(ort_api,
                                                    session_options_,
                                                    FormatEPConfigKey("htp_bf16_enable"),
                                                    false,
                                                    logger_);

  // op_affinity: path to a JSON config gating which backend claims specific op types
  std::string op_affinity_path;
  GetSessionConfigEntryOrDefault(ort_api, session_options_,
                                 FormatEPConfigKey(kQnnOpAffinityKey), "", op_affinity_path);
  if (!op_affinity_path.empty()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE,
                (std::string(kQnnOpAffinityKey) + ": " + op_affinity_path).c_str());
    try {
      model_settings_.op_affinity = qnn::OpAffinityMap::FromConfigFile(std::filesystem::path(op_affinity_path));
    } catch (const std::exception& e) {
      std::string message = "Failed to load op_affinity config file '" + op_affinity_path +
                            "': " + e.what();
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, message.c_str());
      throw std::runtime_error(message);
    }

    // Reject a config that pins an op to a different non-CPU backend
    if (const std::optional<qnn::QnnBackendType> session_backend = InferBackendTypeFromPath(backend_path);
        session_backend.has_value()) {
      const Ort::Status status = model_settings_.op_affinity.ValidateForSessionBackend(*session_backend);
      if (!status.IsOK()) {
        const std::string message = "Op Affinity validation failed: " + status.GetErrorMessage();
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, message.c_str());
        throw std::runtime_error(message);
      }
    }
  }
  // Check BF16 compatibility early
  if (model_settings_.htp_bf16_enable) {
    // Check SoC model
    if (soc_model == QNN_SOC_MODEL_UNKNOWN) {
      std::string message =
          "BF16 mode is enabled but soc_model is not specified. Both parameters must be set together for BF16 support.";
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, message.c_str());
      throw std::runtime_error(message);
    } else if (soc_model < 88) {
      std::string message = "BF16 mode is enabled but SoC model is " + std::to_string(soc_model) +
                            " (expected 88 and above).";
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, message.c_str());
      throw std::runtime_error(message);
    }

    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_INFO,
                ("BF16 mode enabled with compatible hardware: SoC " + std::to_string(soc_model)).c_str());
  }

  // Enforce SoC model to be set on x86_64 Linux (simulator) when enable FP16.
#if defined(__linux__) && !defined(__aarch64__)
  if (htp_graph_configs_.enable_htp_fp16_precision && soc_model == QNN_SOC_MODEL_UNKNOWN && soc_model_per_soc_.empty()) {
    const std::string message =
        "FP16 precision mode is enabled but soc_model is not specified. "
        "Both parameters must be set together for FP16 precision support.";
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, message.c_str());
    throw std::runtime_error(message);
  }
#endif

  model_settings_.enable_block_quant_weight_optimization = ParseBoolOption(ort_api,
                                                                           session_options_,
                                                                           FormatEPConfigKey("enable_block_quant_weight_optimization"),
                                                                           false,
                                                                           logger_);

  if (disable_cpu_ep_fallback_ && model_settings_.offload_graph_io_quantization) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_INFO,
                "Fallback to CPU EP is disabled, but user tried to configure QNN EP to offload graph I/O "
                "quantization/dequantization to another EP. These are conflicting options. Fallback to CPU "
                "EP will remain disabled and graph I/O quantization/dequantization will not be offloaded "
                "to another EP.");
    model_settings_.offload_graph_io_quantization = false;
  }

  if (enable_file_mapped_weights_ && !rpcmem_library_) {
    // Attempt to init rpcmem_library_ if needed. If this fails, then
    // disable file mapped weights and proceed with normal operation
    try {
      rpcmem_library_ = std::make_shared<qnn::RpcMemLibrary>();
    } catch (const std::exception& e) {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_WARNING, ("Unable to load RPCMem library: " + std::string(e.what()) + " - Disabling file mapped weights.").c_str());
      enable_file_mapped_weights_ = false;
    }
  }

  dump_json_qnn_graph_ = ParseBoolOption(ort_api,
                                         session_options_,
                                         FormatEPConfigKey("dump_json_qnn_graph"),
                                         false,
                                         logger_);

  static const std::string QNN_GRAPH_DUMP_DIR = "json_qnn_graph_dir";
  std::string json_graph_dir_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey(QNN_GRAPH_DUMP_DIR),
                                 "",
                                 json_graph_dir_str);

  if (!json_graph_dir_str.empty()) {
    json_qnn_graph_dir_ = json_graph_dir_str;
    if (dump_json_qnn_graph_) {
      if (ProbeDumpDirectoryWritable(json_qnn_graph_dir_, "QNN JSON graph dump", logger_)) {
        ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, ("JSON graphs directory: " + json_qnn_graph_dir_).c_str());
      } else {
        dump_json_qnn_graph_ = false;
      }
    } else {
      ORT_CXX_LOG(logger_,
                  ORT_LOGGING_LEVEL_WARNING,
                  "Provided a directory for dumping QNN JSON graphs, but did not enable dumping of QNN JSON graphs.");
    }
  }

  // Framework op trace options
  static constexpr const char* kEnableFrameworkOpTrace = "enable_framework_op_trace";
  static constexpr const char* kFrameworkOpTraceDir = "framework_op_trace_dir";

  enable_framework_op_trace_ = ParseBoolOption(ort_api,
                                               session_options_,
                                               FormatEPConfigKey(kEnableFrameworkOpTrace),
                                               false,
                                               logger_);

  std::string trace_dir_str;
  GetSessionConfigEntryOrDefault(ort_api,
                                 session_options_,
                                 FormatEPConfigKey(kFrameworkOpTraceDir),
                                 "",
                                 trace_dir_str);
  if (!trace_dir_str.empty()) {
    framework_op_trace_dir_ = trace_dir_str;
  }

  if (enable_framework_op_trace_) {
    if (framework_op_trace_dir_.empty()) {
      framework_op_trace_dir_ = std::filesystem::current_path().string();
    }
    // Probe writability up-front. A non-writable path (read-only Android mount,
    // read-only network share, missing parent permissions) would otherwise only
    // surface as a WARNING after the entire compile + in-memory trace build
    // finishes. Disabling tracing here lets us skip all the per-graph collection
    // work when the trace can never be written.
    if (ProbeDumpDirectoryWritable(framework_op_trace_dir_,
                                   "Framework op trace",
                                   logger_)) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO,
                  ("Framework op tracing enabled. Output dir: " + framework_op_trace_dir_).c_str());
    } else {
      enable_framework_op_trace_ = false;
    }
  } else if (!framework_op_trace_dir_.empty()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING,
                "Provided a directory for framework op trace, but did not enable framework op tracing.");
  }

  // QNN EP input graph dump options. Emits the ONNX graph the EP receives in
  // GetCapabilityImpl (compile-time, pre-partition) as a QNN-Netron-schema JSON.
  static constexpr const char* kDumpQnnEpInputGraph = "dump_qnn_ep_input_graph";
  static constexpr const char* kDumpQnnEpInputGraphDir = "dump_qnn_ep_input_graph_dir";

  dump_qnn_ep_input_graph_ = ParseBoolOption(ort_api,
                                             session_options_,
                                             FormatEPConfigKey(kDumpQnnEpInputGraph),
                                             false,
                                             logger_);

  if (dump_qnn_ep_input_graph_) {
    // Resolve the dump directory only when the dump itself is enabled. The
    // session-config entry overrides the default; an unset/empty value falls
    // back to the current working directory so the option is usable without
    // a separate path config.
    std::string ep_input_graph_dir_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   FormatEPConfigKey(kDumpQnnEpInputGraphDir),
                                   "",
                                   ep_input_graph_dir_str);
    if (ep_input_graph_dir_str.empty()) {
      ep_input_graph_dir_str = std::filesystem::current_path().string();
    }
    dump_qnn_ep_input_graph_dir_ = std::move(ep_input_graph_dir_str);

    // Probe writability up-front so a non-writable path disables the feature
    // before the per-graph walk runs.
    if (ProbeDumpDirectoryWritable(dump_qnn_ep_input_graph_dir_,
                                   "QNN EP input graph dump",
                                   logger_)) {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO,
                  ("QNN EP input graph dump enabled. Output dir: " + dump_qnn_ep_input_graph_dir_).c_str());
    } else {
      dump_qnn_ep_input_graph_ = false;
    }
  }

  static const std::string QNN_HTP_EXTENDED_UDMA_MODE = "extended_udma";
  enable_htp_extended_udma_mode_ = ParseBoolOption(ort_api,
                                                   session_options_,
                                                   FormatEPConfigKey(QNN_HTP_EXTENDED_UDMA_MODE),
                                                   false,
                                                   logger_);

  // HTP Graph Splitting (Graph Program Executor). Requires QAIRT SDK 2.49+ at runtime.
  // Supported in both JIT and AOT workflows.
  enable_htp_graph_splitting_ = ParseBoolOption(ort_api,
                                                session_options_,
                                                FormatEPConfigKey("enable_htp_graph_splitting"),
                                                false,
                                                logger_);
#ifndef QNN_HTP_GRAPH_SPLITTING_AVAILABLE
  if (enable_htp_graph_splitting_) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING,
                "enable_htp_graph_splitting=1 was set but this build was compiled against QAIRT SDK < 2.49. "
                "Graph splitting is not available and the option will be ignored.");
    enable_htp_graph_splitting_ = false;
  }
#else
  if (enable_htp_graph_splitting_) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, "enable_htp_graph_splitting: 1");
  }
#endif

  // Number of threads used to prepare split subgraphs in parallel. Requires QAIRT SDK 2.51+.
  // UINT32_MAX (default) enables auto-selection: min(max(1, hardware_concurrency), num_splits).
  // Only meaningful when enable_htp_graph_splitting=1.
  {
    std::string num_threads_str;
    GetSessionConfigEntryOrDefault(ort_api,
                                   session_options_,
                                   FormatEPConfigKey("htp_graph_splitting_num_prepare_threads"),
                                   "",
                                   num_threads_str);
    if (!num_threads_str.empty()) {
      htp_graph_splitting_num_prepare_threads_ = static_cast<uint32_t>(std::stoul(num_threads_str));
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE,
                  ("htp_graph_splitting_num_prepare_threads: " + num_threads_str).c_str());
#ifndef QNN_HTP_GRAPH_SPLITTING_NUM_THREADS_AVAILABLE
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING,
                  "htp_graph_splitting_num_prepare_threads was set but this build was compiled against "
                  "QAIRT SDK < 2.51. The option will be ignored.");
      htp_graph_splitting_num_prepare_threads_ = UINT32_MAX;
#endif
    }
  }

  // Option to skip QNN API interface version check to use other QNN library other than default.
  static const std::string SKIP_QNN_VERSION_CHECK = "skip_qnn_version_check";
  auto skip_qnn_version_check = ParseBoolOption(ort_api,
                                                session_options,
                                                FormatEPConfigKey(SKIP_QNN_VERSION_CHECK),
                                                false,
                                                logger_);

  // When dumping to DLC on a device-less host, the target backend validates op configs against an
  // arch-agnostic op table and over-rejects arch-specific (v73+) ops. This option skips that
  // validation, falling back to the serializer's generic checks. Default false keeps validation on.
  static const std::string SKIP_BACKEND_OP_VALIDATION = "skip_backend_op_validation";
  auto skip_backend_op_validation = ParseBoolOption(ort_api,
                                                    session_options,
                                                    FormatEPConfigKey(SKIP_BACKEND_OP_VALIDATION),
                                                    false,
                                                    logger_);

  // For context binary generation with weight sharing enabled, use the QnnBackendManager from the shared context if it exits
  // So that all graphs from later sessions will be compiled into the same QNN context
  if (
      ((context_cache_enabled_ && share_ep_contexts_) || htp_share_resource_optimization_ == 1) &&
      SharedContext::GetInstance().GetSharedQnnBackendManager()) {
    qnn_backend_manager_ = SharedContext::GetInstance().GetSharedQnnBackendManager();
    // Reset QnnBackendManager's logger to the one in current session as original one could be deleted along with the
    // previous session.
    qnn_backend_manager_->ResetLogger(logger_);
    // Clear the QnnBackendManager from singleton to stop the resource share
    if (stop_share_ep_contexts_) {
      SharedContext::GetInstance().ResetSharedQnnBackendManager();
    }
  } else {
    qnn_backend_manager_ = qnn::QnnBackendManager::Create(
        qnn::QnnBackendManagerConfig{backend_path,
                                     profiling_level_etw,
                                     profiling_level,
                                     profiling_file_path,
                                     context_priority,
                                     std::move(qnn_serializer_config),
                                     device_id_,
                                     htp_arch,
                                     soc_model,
                                     op_packages,
                                     skip_qnn_version_check,
                                     enable_framework_op_trace_,
                                     skip_backend_op_validation},
        ApiPtrs{ort_api, ep_api, model_editor_api}, logger_);
    if (htp_share_resource_optimization_ == 1) {
      SharedContext::GetInstance().SetSharedQnnBackendManager(qnn_backend_manager_);
    }
  }

  // Initialize compatibility manager with backend manager.
  qnn_cache_compatibility_manager_ = std::make_shared<qnn::QnnCacheCompatibilityManager>(qnn_backend_manager_.get());

  // Choose EP allocator. Must be done after creating the backend manager.
  static const std::string QNN_HTP_SHARED_MEMORY_ALLOCATOR_ENABLED = "enable_htp_shared_memory_allocator";
  if (ParseBoolOption(ort_api,
                      session_options_,
                      FormatEPConfigKey(QNN_HTP_SHARED_MEMORY_ALLOCATOR_ENABLED),
                      false,
                      logger_)) {
    // Initialize rpcmem_library_.
    // This library is only necessary for the inference (for the shared memory allocator), if we are in context
    // generation stage, there is no need to load it as no allocations will be made.
    if (!context_cache_enabled_) {
      rpcmem_library_ = std::make_shared<qnn::RpcMemLibrary>();
      qnn_allocator_type_ = qnn::QnnAllocatorType::HTP_SHARED;
    } else {
      ORT_CXX_LOGF(logger_,
                   ORT_LOGGING_LEVEL_INFO,
                   "Context cache is enabled in this session (via %s); the HTP shared memory allocator will be disabled"
                   " as no allocations are expected to be made.",
                   kOrtSessionOptionEpContextEnable);
    }
    model_settings_.htp_shared_memory = true;
  }

  static const std::string QNN_DX12_SHARED_MEMORY_ALLOCATOR_ENABLED = "enable_dx12_shared_memory_allocator";
  if (ParseBoolOption(ort_api,
                      session_options_,
                      FormatEPConfigKey(QNN_DX12_SHARED_MEMORY_ALLOCATOR_ENABLED),
                      false,
                      logger_)) {
    if (qnn_allocator_type_ != qnn::QnnAllocatorType::NONE) {
      ORT_CXX_LOGF(logger_,
                   ORT_LOGGING_LEVEL_WARNING,
                   "QNN allocator already set to type: %s. Option '%s' will be ignored.",
                   qnn::QnnAllocatorTypeToString(qnn_allocator_type_).data(),
                   QNN_DX12_SHARED_MEMORY_ALLOCATOR_ENABLED.c_str());
    } else {
      if (qnn_backend_manager_->IsDx12SharedMemoryAllocatorSupported()) {
        qnn_allocator_type_ = qnn::QnnAllocatorType::DX12_SHARED;
      } else {
        ORT_CXX_LOGF(logger_,
                     ORT_LOGGING_LEVEL_WARNING,
                     "Dx12SharedMemoryAllocator was requested, but it is not supported.");
      }
    }
  }

  qnn_backend_manager_->SetQnnAllocatorType(qnn_allocator_type_);
  if (qnn_allocator_type_ != qnn::QnnAllocatorType::NONE) {
    ORT_CXX_LOGF(logger_,
                 ORT_LOGGING_LEVEL_VERBOSE,
                 "QNN allocator set to type: %s.",
                 qnn::QnnAllocatorTypeToString(qnn_allocator_type_).data());
  }

#if defined(_WIN32)
  if (qnn::QnnTelemetry::SupportsETW()) {
    auto& etwRegistrationManager = qnn::QnnTelemetry::Instance();
    // Register callback for ETW capture state (rundown)
    callback_ETWSink_provider_ = qnn::QnnTelemetry::EtwInternalCallback(
        [&etwRegistrationManager, this](
            LPCGUID SourceId,
            ULONG IsEnabled,
            UCHAR Level,
            ULONGLONG MatchAnyKeyword,
            ULONGLONG MatchAllKeyword,
            PEVENT_FILTER_DESCRIPTOR FilterData,
            PVOID CallbackContext) {
          ORT_UNUSED_PARAMETER(SourceId);
          ORT_UNUSED_PARAMETER(MatchAnyKeyword);
          ORT_UNUSED_PARAMETER(MatchAllKeyword);
          ORT_UNUSED_PARAMETER(FilterData);
          ORT_UNUSED_PARAMETER(CallbackContext);

          if (IsEnabled == EVENT_CONTROL_CODE_ENABLE_PROVIDER) {
            if ((MatchAnyKeyword & static_cast<ULONGLONG>(qnn::ORTTraceLoggingKeyword::Logs)) != 0) {
              auto ortETWSeverity = etwRegistrationManager.MapLevelToOrtLoggingLevel();
              (void)qnn_backend_manager_->ResetQnnLogLevel(ortETWSeverity);
            }
            if ((MatchAnyKeyword & static_cast<ULONGLONG>(qnn::ORTTraceLoggingKeyword::Profiling)) != 0) {
              if (Level != 0) {
                // Commenting out Dynamic QNN Profiling for now
                // There seems to be a crash in 3rd party QC QnnHtp.dll with this.
                // Repro Scenario - start ETW tracing prior to session creation.
                //    Then disable/enable ETW Tracing with the code below uncommented a few times
                // auto profiling_level_etw = GetProfilingLevelFromETWLevel(Level);
                // (void)qnn_backend_manager_->SetProfilingLevelETW(profiling_level_etw);
                //
                // NOTE(1/2/2025): It is possible that the above was not working in part because it is using the
                // *logging ETW* subsystem to modify profiling, which should use an entirely different
                // ETW provider (see QnnTelemetry). Should add callbacks for profiling to the QnnTelemetry ETW provider.
              }
            }
          }

          if (IsEnabled == EVENT_CONTROL_CODE_DISABLE_PROVIDER) {
            // (void)qnn_backend_manager_->SetProfilingLevelETW(qnn::ProfilingLevel::INVALID);
            (void)qnn_backend_manager_->ResetQnnLogLevel(std::nullopt);
          }
        });
    etwRegistrationManager.RegisterInternalCallback(callback_ETWSink_provider_);
  }
#endif

  // Owns the EPContext callbacks resolved from session_options. On pre-v28 ORT this is a no-op
  // stub with HasReadCallback()/HasWriteCallback() returning false.
  io_dispatch_ = std::make_unique<qnn::EpContextIoDispatch>(&session_options_, &logger_);

  // File mapping and encryption are mutually exclusive: an App-provided read callback replaces
  // the on-disk read, so file mapping must be off when a read callback is registered.
  if (enable_file_mapped_weights_ && io_dispatch_->HasReadCallback()) {
    enable_file_mapped_weights_ = false;
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING,
                "EPContext read callback registered; disabling file mapping for this session.");
  }
}

QnnEp::~QnnEp() {
  if (qnn_backend_manager_) {
    auto thread_id = std::this_thread::get_id();
    qnn_backend_manager_->RemovePerThreadHtpPowerConfigMapping(thread_id);
    // NOTE: do NOT tear down the release timer here. The QnnBackendManager may be
    // shared across sessions (htp_share_resource_optimization_/weight sharing);
    // killing the timer on the first session's destruction would break the others.
    // The timer is released with the manager itself (QnnBackendManager::ReleaseResources).
    std::lock_guard<std::mutex> lock(config_id_mutex_);
    if (htp_power_config_id_.has_value()) {
      // Drop this id from the (possibly still-live shared) timer's boosted set
      // before destroying it, so the timer never relaxes a destroyed id.
      qnn_backend_manager_->DropBoostedPowerConfigId(*htp_power_config_id_);
      qnn_backend_manager_->DestroyHtpPowerConfigId(*htp_power_config_id_);
    }
  }

  // Explicitly clear the QNN models map to ensure proper cleanup
  if (!qnn_models_.empty()) {
    qnn_models_.clear();
  }

#if defined(_WIN32)
  // // Clean up ETW callback if registered
  if (callback_ETWSink_provider_) {
    if (qnn::QnnTelemetry::SupportsETW()) {
      auto& etwRegistrationManager = qnn::QnnTelemetry::Instance();
      etwRegistrationManager.UnregisterInternalCallback(callback_ETWSink_provider_);
      callback_ETWSink_provider_ = nullptr;
    }
  }
#endif
}

const char* ORT_API_CALL QnnEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* qnn_ep = static_cast<const QnnEp*>(this_ptr);
  return qnn_ep->name_.c_str();
}

// Logs information about the supported/unsupported nodes.
static void LogNodeSupport(const Ort::Logger& logger,
                           const qnn::IQnnNodeGroup& qnn_node_group,
                           const Ort::Status& support_status) {
  size_t num_nodes = 0;
  std::ostringstream oss;
  for (const OrtNodeUnit* node_unit : qnn_node_group.GetNodeUnits()) {
    for (const OrtNode* node : node_unit->GetAllNodesInGroup()) {
      Ort::ConstNode const_node(node);
      size_t node_id = const_node.GetId();
      const std::string& op_type = const_node.GetOperatorType();
      const std::string& name = const_node.GetName();

      oss << "\tOperator type: " << op_type
          << " Node name: " << name
          << " Node index: " << node_id << std::endl;
      num_nodes += 1;
    }
  }
  if (!support_status.IsOK()) {
    oss << "\tREASON : " << support_status.GetErrorMessage() << std::endl;
  }

  std::string msg = std::string((support_status.IsOK() ? "Validation PASSED " : "Validation FAILED ")) +
                    "for " + std::to_string(num_nodes) +
                    " nodes in " + std::string(qnn_node_group.Type()) +
                    " (" + qnn_node_group.GetTargetNodeUnit()->OpType() + ") :\n" +
                    oss.str();
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, msg.c_str());
}

OrtStatus* QnnEp::GetSupportedNodes(const OrtGraph* graph,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const size_t node_unit_size,
                                    std::vector<const OrtNode*>& supported_nodes,
                                    std::vector<qnn::UnsupportedNodeInfo>& unsupported_nodes) const {
  size_t num_graph_inputs = 0;
  size_t num_graph_outputs = 0;
  RETURN_IF_NOT_NULL(ort_api.Graph_GetNumInputs(graph, &num_graph_inputs));
  RETURN_IF_NOT_NULL(ort_api.Graph_GetNumOutputs(graph, &num_graph_outputs));

  std::vector<const OrtValueInfo*> graph_inputs(num_graph_inputs);
  std::vector<const OrtValueInfo*> graph_outputs(num_graph_outputs);
  RETURN_IF_NOT_NULL(ort_api.Graph_GetInputs(graph, graph_inputs.data(), graph_inputs.size()));
  RETURN_IF_NOT_NULL(ort_api.Graph_GetOutputs(graph, graph_outputs.data(), graph_outputs.size()));

  // Util function that initializes name vector and index map for graph inputs/outputs.
  auto init_input_output_info = [&](std::vector<std::string>& names,
                                    std::unordered_map<std::string, size_t>& index_map,
                                    const std::vector<const OrtValueInfo*>& inouts) {
    for (size_t idx = 0; idx < inouts.size(); ++idx) {
      const char* name = nullptr;
      auto lambda_status = ort_api.GetValueInfoName(inouts[idx], &name);
      if (lambda_status != nullptr) {
        ort_api.ReleaseStatus(lambda_status);
        return;  // Skip on error
      }
      names.push_back(name);
      index_map.emplace(name, idx);
    }
  };

  // Note that ort_api.GraphGetInputs includes initializers that are included in the list of graph inputs but
  // in previous non-ABI, graph_viewer.GetInputs does not include them. Fortunately, this index map is only used in
  // QnnModelWrapper::GetTensorType to determine APP_WRITE tensor type. Since STATIC tensor type is checked before
  // APP_WRITE in current implementation, there should be no impact including initializers here.
  qnn::GraphInputOutputInfo model_inputs;
  init_input_output_info(model_inputs.names, model_inputs.indices, graph_inputs);

  qnn::GraphInputOutputInfo model_outputs;
  init_input_output_info(model_outputs.names, model_outputs.indices, graph_outputs);

  auto qnn_model_wrapper = qnn::QnnModelWrapper(*graph,
                                                ApiPtrs{ort_api, ep_api, model_editor_api},
                                                logger_,
                                                *qnn_backend_manager_,
                                                model_inputs,
                                                model_outputs,
                                                model_settings_,
                                                &tensor_name_overrides_,
                                                /*op_trace_collector=*/nullptr,
                                                /*is_post_layout_transform=*/is_post_layout_transform_);

  std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> qnn_node_groups;
  qnn_node_groups.reserve(node_unit_size);

  Ort::Status qnn_status = qnn::GetQnnNodeGroups(qnn_node_groups, qnn_model_wrapper, node_unit_map, node_unit_size, logger_);
  if (!qnn_status.IsOK()) {
    return qnn_status.release();
  }

  for (const std::unique_ptr<qnn::IQnnNodeGroup>& qnn_node_group : qnn_node_groups) {
    Ort::Status support_status = qnn_node_group->IsSupported(qnn_model_wrapper, logger_);
    const bool supported = support_status.IsOK();

    LogNodeSupport(logger_, *qnn_node_group, support_status);
    if (supported) {
      for (const OrtNodeUnit* node_unit : qnn_node_group->GetNodeUnits()) {
        for (const OrtNode* node : node_unit->GetAllNodesInGroup()) {
          supported_nodes.push_back(node);
        }
      }
    } else if (enable_framework_op_trace_) {
      std::vector<qnn::UnsupportedNodeInfo> batch;
      for (const OrtNodeUnit* node_unit : qnn_node_group->GetNodeUnits()) {
        for (const OrtNode* node : node_unit->GetAllNodesInGroup()) {
          Ort::ConstNode const_node(node);
          batch.push_back({
              std::string(const_node.GetName()),
              std::string(const_node.GetOperatorType()),
              const_node.GetId(),
              std::string(support_status.GetErrorMessage()),
          });
        }
      }
      if (!batch.empty()) {
        unsupported_nodes.insert(unsupported_nodes.end(),
                                 std::make_move_iterator(batch.begin()),
                                 std::make_move_iterator(batch.end()));
      }
    }
  }
  return nullptr;
}

OrtStatus* QnnEp::GetMultiSocSupportedNodes(const OrtGraph* graph,
                                            const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                            const size_t node_unit_size,
                                            std::vector<const OrtNode*>& supported_nodes,
                                            std::vector<qnn::UnsupportedNodeInfo>& unsupported_nodes) const {
  // Collect each SoC's rejections tagged with its SoC label, then merge into one
  // row per ONNX node via MergePerSocUnsupportedNodes.
  std::vector<qnn::PerSocUnsupportedNodes> per_soc_unsupported;
  if (enable_framework_op_trace_) {
    per_soc_unsupported.reserve(htp_arch_per_soc_.size());
  }

  // Iterate each SoC and intersect supported nodes.
  for (size_t idx = 0; idx < htp_arch_per_soc_.size(); ++idx) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Getting supported nodes for HTP arch " + std::to_string(htp_arch_per_soc_[idx]) +
                 " and SoC model " + std::to_string(soc_model_per_soc_[idx]) + ".")
                    .c_str());

    // Complete setup for device and context.
    ScopedPerSocQnnBackendSetup scoped_backend_setup(*this);
    RETURN_IF_NOT_OK(scoped_backend_setup.Init(idx));

    // Get SoC-specific supported and unsupported nodes.
    std::vector<const OrtNode*> supported_nodes_per_soc;
    std::vector<qnn::UnsupportedNodeInfo> unsupported_nodes_per_soc;
    RETURN_IF_NOT_NULL(GetSupportedNodes(graph,
                                         node_unit_map,
                                         node_unit_size,
                                         supported_nodes_per_soc,
                                         unsupported_nodes_per_soc));

    // Tag this SoC's rejections with its (htp_arch, soc_model) label for merging.
    // Identical (arch, soc_model) configs share a label and merge — acceptable,
    // they are indistinguishable targets.
    if (enable_framework_op_trace_) {
      per_soc_unsupported.emplace_back(htp_arch_per_soc_[idx], soc_model_per_soc_[idx],
                                       std::move(unsupported_nodes_per_soc));
    }

    if (idx == 0) {
      // Directly move for the first SoC.
      supported_nodes = std::move(supported_nodes_per_soc);
    } else {
      // Incrementally remove nodes not supported in later SoC from the list.
      std::unordered_set<size_t> supported_node_ids_per_soc;
      supported_node_ids_per_soc.reserve(supported_nodes_per_soc.size());
      for (const OrtNode* node : supported_nodes_per_soc) {
        supported_node_ids_per_soc.insert(Ort::ConstNode(node).GetId());
      }

      auto not_contains = [&supported_node_ids_per_soc](const OrtNode* node) {
        return supported_node_ids_per_soc.find(Ort::ConstNode(node).GetId()) == supported_node_ids_per_soc.end();
      };
      supported_nodes.erase(std::remove_if(supported_nodes.begin(), supported_nodes.end(), not_contains),
                            supported_nodes.end());
    }
  }

  if (enable_framework_op_trace_) {
    std::vector<qnn::UnsupportedNodeInfo> merged = qnn::MergePerSocUnsupportedNodes(per_soc_unsupported);
    unsupported_nodes.insert(unsupported_nodes.end(),
                             std::make_move_iterator(merged.begin()),
                             std::make_move_iterator(merged.end()));
  }

  return nullptr;
}

void QnnEp::InitQnnHtpGraphConfigs(
    const qnn::HtpGraphConfigs_t& configs,
    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const {
  if (qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::HTP) {
    if (configs.htp_graph_finalization_opt_mode != qnn::HtpGraphFinalizationOptimizationMode::kDefault) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config = configs_builder.PushCustomConfig();
      htp_graph_opt_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
      htp_graph_opt_config->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
      htp_graph_opt_config->optimizationOption.floatValue = static_cast<float>(configs.htp_graph_finalization_opt_mode);

      gsl::not_null<QnnGraph_Config_t*> graph_opt_config = configs_builder.PushConfig();
      graph_opt_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config->customConfig = htp_graph_opt_config;
    }

    if (configs.vtcm_size_in_mb > 0) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config_vtcm = configs_builder.PushCustomConfig();
      htp_graph_opt_config_vtcm->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
      htp_graph_opt_config_vtcm->vtcmSizeInMB = static_cast<uint32_t>(configs.vtcm_size_in_mb);

      gsl::not_null<QnnGraph_Config_t*> graph_opt_config_vtcm = configs_builder.PushConfig();
      graph_opt_config_vtcm->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_opt_config_vtcm->customConfig = htp_graph_opt_config_vtcm;
    }

    if (configs.enable_htp_fp16_precision) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_precision_config = configs_builder.PushCustomConfig();
      htp_graph_precision_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
      htp_graph_precision_config->precision = QNN_PRECISION_FLOAT16;

      gsl::not_null<QnnGraph_Config_t*> graph_precision_config = configs_builder.PushConfig();
      graph_precision_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_precision_config->customConfig = htp_graph_precision_config;
    }

    if (configs.enable_htp_monolithic_lstm) {
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_monolithic_lstm_config = configs_builder.PushCustomConfig();
      htp_graph_monolithic_lstm_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_MONOLITHIC_LSTM;
      htp_graph_monolithic_lstm_config->monolithicLstm = true;

      gsl::not_null<QnnGraph_Config_t*> graph_config = configs_builder.PushConfig();
      graph_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_config->customConfig = htp_graph_monolithic_lstm_config;
    }

    if (configs.enable_htp_fp16_clamp_overflow) {
#ifdef QNN_HTP_FP16_CLAMP_OVERFLOW_AVAILABLE
      gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_fp16_clamp_config = configs_builder.PushCustomConfig();
      htp_fp16_clamp_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_FP16_CLAMP_OVERFLOW;
      htp_fp16_clamp_config->fp16ClampOverflow = true;

      gsl::not_null<QnnGraph_Config_t*> graph_config = configs_builder.PushConfig();
      graph_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
      graph_config->customConfig = htp_fp16_clamp_config;
#endif
    }
  }
}

static bool EpSharedContextsHasAllGraphs(const OrtGraph* graph, const OrtApi& ort_api, const Ort::Logger& logger) {
  size_t num_nodes = 0;
  if (ort_api.Graph_GetNumNodes(graph, &num_nodes) != nullptr) {
    return false;
  }

  std::vector<const OrtNode*> graph_nodes(num_nodes);
  if (ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size()) != nullptr) {
    return false;
  }

  for (const OrtNode* node : graph_nodes) {
    const char* op_type = nullptr;
    if (ort_api.Node_GetOperatorType(node, &op_type) != nullptr) {
      return false;
    }

    OrtNodeAttrHelper node_helper(*node);
    std::string cache_source = qnn::utils::GetLowercaseString(node_helper.Get(qnn::SOURCE, ""));

    if (op_type == qnn::EPCONTEXT_OP && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
      const char* node_name = nullptr;
      if (ort_api.Node_GetName(node, &node_name) != nullptr) {
        return false;
      }

      if (!SharedContext::GetInstance().HasQnnModel(node_name)) {
        ORT_CXX_LOG(logger,
                    ORT_LOGGING_LEVEL_VERBOSE,
                    ("Graph: " +
                     std::string(node_name) +
                     " from EpContext node not found from shared EP contexts.")
                        .c_str());
        return false;
      }
    }
  }

  return true;
}

static void GetMainEPCtxNodes(const OrtGraph* graph,
                              const OrtApi& ort_api,
                              std::unordered_set<const OrtNode*>& ep_context_nodes,
                              const Ort::Logger& logger) {
  size_t num_nodes = 0;
  if (ort_api.Graph_GetNumNodes(graph, &num_nodes) != nullptr) {
    return;
  }

  std::vector<const OrtNode*> graph_nodes(num_nodes);
  if (ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size()) != nullptr) {
    return;
  }

  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = graph_nodes[node_idx];

    const char* op_type = nullptr;
    if (ort_api.Node_GetOperatorType(node, &op_type) != nullptr) {
      continue;
    }

    OrtNodeAttrHelper node_helper(*node);
    bool is_main_context = node_helper.Get(qnn::MAIN_CONTEXT, static_cast<int64_t>(0));
    std::string cache_source = qnn::utils::GetLowercaseString(node_helper.Get(qnn::SOURCE, ""));

    if (is_main_context &&
        op_type == qnn::EPCONTEXT_OP &&
        (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
      const char* node_name = nullptr;
      auto node_status = ort_api.Node_GetName(node, &node_name);
      if (node_status != nullptr) {
        ort_api.ReleaseStatus(node_status);
        continue;
      }
      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_VERBOSE,
                  ("EPContext Node found: [1] index: [" +
                   std::to_string(node_idx) +
                   "] name: [" +
                   std::string(node_name) +
                   "]")
                      .c_str());

      ep_context_nodes.insert(node);
    }
  }
}

void QnnEp::PartitionCtxModel(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info) {
  // Get all nodes from the graph
  size_t num_nodes = 0;
  if (ort_api.Graph_GetNumNodes(graph, &num_nodes) != nullptr) {
    return;
  }

  std::vector<const OrtNode*> graph_nodes(num_nodes);
  if (ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size()) != nullptr) {
    return;
  }

  std::vector<std::vector<const OrtNode*>> supported_groups;
  size_t num_supported_groups = 0;

  for (size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const OrtNode* node = graph_nodes[node_idx];

    const char* op_type = nullptr;
    if (ort_api.Node_GetOperatorType(node, &op_type) != nullptr) {
      continue;
    }

    OrtNodeAttrHelper node_helper(*node);
    std::string cache_source = qnn::utils::GetLowercaseString(node_helper.Get(qnn::SOURCE, ""));

    if (op_type == qnn::EPCONTEXT_OP && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
      const char* node_name = nullptr;
      auto partition_status = ort_api.Node_GetName(node, &node_name);
      if (partition_status != nullptr) {
        ort_api.ReleaseStatus(partition_status);
        continue;
      }

      std::string log_message = "Node supported: [1] index: [" + std::to_string(node_idx) +
                                "] name: [" + (node_name ? node_name : std::string("unknown")) +
                                "] Operator type: [EPContext] index: [" + std::to_string(node_idx) + "]";
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, log_message.c_str());

      std::vector<const OrtNode*> supported_group{node};
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;

      auto add_status = ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                 supported_group.data(),
                                                                 supported_group.size(),
                                                                 &node_fusion_options);
      if (add_status != nullptr) {
        ort_api.ReleaseStatus(add_status);
        continue;
      }

      ++num_supported_groups;
    }
  }

  std::string summary_msg = "Number of partitions supported by QNN EP: " + std::to_string(num_supported_groups) +
                            ", number of nodes in the graph: " + std::to_string(num_nodes) +
                            ", number of nodes supported by QNN: " + std::to_string(num_supported_groups);
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO, summary_msg.c_str());
}

static void GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
                                        const std::basic_string<ORTCHAR_T>& model_path_string,
                                        std::basic_string<ORTCHAR_T>& context_model_path) {
  // always try the path set by user first, it's the only way to set it if load model from memory
  if (!user_context_cache_path.empty()) {
    context_model_path = FILEPATH_TO_STRING(std::filesystem::path(user_context_cache_path));
  } else if (!model_path_string.empty()) {  // model loaded from file
    context_model_path = model_path_string;
  }
}

OrtStatus* ORT_API_CALL QnnEp::GetGenieCapability(OrtEp* this_ptr,
                                                  const OrtGraph* graph,
                                                  OrtEpGraphSupportInfo* graph_support_info) {
  // CREATE GENIE_BACKEND_MANAGER
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);
  if (!ep->genie_backend_manager_) {
    ep->genie_backend_manager_ = qnn::GenieBackendManager::Create(
        qnn::GenieBackendManagerConfig{kDefaultGenieBackendPath}, ep->logger_);
    auto setup_st = ep->genie_backend_manager_->SetupBackend();
    if (!setup_st.IsOK()) {
      return ep->ort_api.CreateStatus(ORT_EP_FAIL, setup_st.GetErrorMessage().c_str());
    }
  }
  ep->genie_api_loader_ = std::make_shared<GenieApiLoader>((ep->genie_backend_manager_)->GetGenieBackendHandle());
  // Get all nodes from the graph
  size_t num_nodes = 0;
  if (ep->ort_api.Graph_GetNumNodes(graph, &num_nodes) != nullptr) {
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Graph_GetNumNodes failed");
  }
  if (num_nodes != 1) {
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Number of nodes must be 1 for Genie");
  }
  std::vector<const OrtNode*> graph_nodes(num_nodes);
  if (ep->ort_api.Graph_GetNodes(graph, graph_nodes.data(), graph_nodes.size()) != nullptr) {
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Graph Creation error");
  }

  // Identify the single node in the graph (which should be the only node)
  const OrtNode* node = graph_nodes[0];
  std::vector<const OrtNode*> supported_group{node};
  OrtNodeFusionOptions node_fusion_options = {};
  node_fusion_options.ort_version_supported = ORT_API_VERSION;
  auto add_status = ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                 supported_group.data(),
                                                                 supported_group.size(),
                                                                 &node_fusion_options);
  if (add_status != nullptr) {
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Error adding Node.");
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::GetCapabilityImpl(OrtEp* this_ptr,
                                                 const OrtGraph* graph,
                                                 OrtEpGraphSupportInfo* graph_support_info) noexcept {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  // Best-effort diagnostic dump of the ONNX graph the EP just received.
  // Fires before any subgraph / EPContext / Genie / backend-setup early
  // return below so every GetCapability invocation produces a dump file —
  // including subgraphs and sessions that later fail to set up a backend
  // (precisely the cases where a "what did the EP see?" artifact is most
  // useful). Filename is sanitized graph name + per-EP counter; the
  // matcher's recommended consumption is "highest counter per unique
  // sanitized name" (see docs/execution_providers/QNN-ExecutionProvider.md).
  if (ep->dump_qnn_ep_input_graph_) {
    Ort::ConstGraph dump_graph{graph};
    // SanitizeGraphNameForFilename returns "graph" when the input is empty
    // or sanitizes to empty, so a single call covers both the present-name
    // and missing-name paths.
    std::string graph_name = qnn::SanitizeGraphNameForFilename(std::string(dump_graph.GetName()));
    size_t count = ep->dump_qnn_ep_input_graph_count_++;
    std::filesystem::path out_path =
        std::filesystem::path(ep->dump_qnn_ep_input_graph_dir_) /
        (graph_name + "." + std::to_string(count) + "_qnn_ep_input_graph.json");
    // Best-effort diagnostic: failures are already logged inside the dumper,
    // so the bool return is intentionally discarded.
    qnn::DumpQnnEpInputGraphToJson(graph, out_path, ep->logger_);
  }

  const OrtNode* parent_node = nullptr;
  RETURN_IF_NOT_NULL(ep->ort_api.Graph_GetParentNode(graph, &parent_node));
  if (parent_node != nullptr) {
    return nullptr;
  }

  size_t num_nodes_in_graph = 0;
  RETURN_IF_NOT_NULL(ep->ort_api.Graph_GetNumNodes(graph, &num_nodes_in_graph));
  if (num_nodes_in_graph == 0) {
    return nullptr;
  }

  ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_VERBOSE,
              ep->is_post_layout_transform_
                  ? "GetCapability pass 2 (post-Layout-Transform; post-LT-gated fusions enabled)"
                  : "GetCapability pass 1 (pre-Layout-Transform; post-LT-gated fusions dormant)");
  auto post_lt_marker = gsl::finally([ep] { ep->is_post_layout_transform_ = true; });

  // Genie Pathway
  if (qnn::GraphHasDlcContextNode(graph, ep->ort_api)) {
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
    return ep->GetGenieCapability(this_ptr, graph, graph_support_info);
#else
    return ep->ort_api.CreateStatus(ORT_EP_FAIL,
                                    "Genie execution pathway is unsupported on this platform.");
#endif
  }

  bool is_qnn_ctx_model = qnn::GraphHasEpContextNode(graph, ep->ort_api);

  if (is_qnn_ctx_model && ep->share_ep_contexts_ && SharedContext::GetInstance().HasSharedQnnModels()) {
    if (EpSharedContextsHasAllGraphs(graph, ep->ort_api, ep->logger_)) {
      ep->PartitionCtxModel(graph, graph_support_info);
      return nullptr;
    }
  }

  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> context_bin_map;
  if (ep->htp_share_resource_optimization_ == 1) {
    std::unordered_set<const OrtNode*> ep_ctx_nodes;
    GetMainEPCtxNodes(graph, ep->ort_api, ep_ctx_nodes, ep->logger_);

    std::basic_string<ORTCHAR_T> model_path = GetModelPathString(graph, ep->ort_api);
    std::basic_string<ORTCHAR_T> context_model_path;
    GetContextOnnxModelFilePath(ep->context_cache_path_cfg_, model_path, context_model_path);

    std::filesystem::path parent_path = std::filesystem::path(context_model_path).parent_path();

    for (auto& ep_ctx_node : ep_ctx_nodes) {
      // Get the ep_cache_context attribute from the node
      const OrtOpAttr* ep_cache_context_attr = nullptr;
      if (ep->ort_api.Node_GetAttributeByName(ep_ctx_node, "ep_cache_context", &ep_cache_context_attr) == nullptr && ep_cache_context_attr != nullptr) {
        char context_buffer[512] = {0};
        size_t context_len = 0;
        if (ep->ort_api.ReadOpAttr(ep_cache_context_attr, ORT_OP_ATTR_STRING, context_buffer, sizeof(context_buffer) - 1, &context_len) == nullptr) {
          std::string context_bin_filepath(parent_path.string());
          context_bin_filepath.append("/").append(std::string(context_buffer, context_len));

          if (context_bin_map.find(context_bin_filepath) == context_bin_map.end()) {
            context_bin_map.emplace(context_bin_filepath, std::make_unique<std::vector<std::string>>());
            // Push context bin filepath for lookup between sessions
            context_bin_map.at(context_bin_filepath)->push_back(context_bin_filepath);
          }

          // Add the node name to the context bin map
          const char* node_name = nullptr;
          if (ep->ort_api.Node_GetName(ep_ctx_node, &node_name) == nullptr && node_name != nullptr) {
            context_bin_map.at(context_bin_filepath)->push_back(std::string(node_name));
          }
        }
      }
    }
  }

  Ort::Status rt;
  if (!ep->enable_multi_soc_ep_context_) {
    rt = ep->qnn_backend_manager_->SetupBackend(is_qnn_ctx_model,
                                                ep->context_cache_enabled_ || ep->prepare_and_load_,
                                                ep->share_ep_contexts_,
                                                ep->htp_share_resource_optimization_,
                                                ep->enable_file_mapped_weights_,
                                                ep->rpcmem_library_,
                                                context_bin_map,
                                                *ep->io_dispatch_,
                                                ep->enable_htp_extended_udma_mode_,
                                                ep->prepare_only_,
                                                ep->enable_htp_graph_splitting_,
                                                ep->htp_graph_splitting_num_prepare_threads_);
  } else {
    rt = ep->qnn_backend_manager_->SetupBackendExceptDeviceAndContext();
  }

  context_bin_map.clear();

  if (!rt.IsOK()) {
    const std::string message = "QNN SetupBackend failed " + rt.GetErrorMessage();
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_ERROR, message.c_str());
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  // op_affinity: the GQA builder may trigger the known performance regression on HTP,
  // so HTP sessions default to keeping GQA on CPU
  // -- users can opt into HTP via the op_affinity config once they've validated it.
  const qnn::QnnBackendType resolved_backend = ep->qnn_backend_manager_->GetQnnBackendType();
  if (resolved_backend == qnn::QnnBackendType::HTP) {
    ep->model_settings_.op_affinity.SeedDefaultIfAbsent("GroupQueryAttention", qnn::QnnBackendType::CPU);
  }

  rt = ep->model_settings_.op_affinity.ValidateForSessionBackend(resolved_backend);

  if (!rt.IsOK()) {
    const std::string message = "Op Affinity validation failed: " + rt.GetErrorMessage();
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_ERROR, message.c_str());
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  if (qnn::IsNpuBackend(ep->qnn_backend_manager_->GetQnnBackendType())) {
    // Create the HTP power config id (and its release timer) for the main thread.
    // The perf mode itself is not voted here: it is applied around graph compile
    // via the INIT_START/INIT_DONE power guard in CompileImpl, and per run via
    // SetPerThreadHtpPowerConfigs. The id is torn down with the session, so a
    // session created but never run ends compile in the relaxed state and frees
    // its vote on destruction.
    ep->CreateHtpPowerConfigId();

    ep->WarnIfHnrdPathActive();
  }

  // Report error if QNN CPU backend is loaded while CPU fallback is disabled
  if (ep->disable_cpu_ep_fallback_ && ep->qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::CPU) {
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Qnn CPU backend is loaded while CPU fallback is disabled.");
  }

  if ((ep->context_cache_enabled_ || is_qnn_ctx_model) && !qnn::IsQpuBackend(ep->qnn_backend_manager_->GetQnnBackendType())) {
    return ep->ort_api.CreateStatus(ORT_EP_FAIL, "Qnn context cache only works for HTP/DSP/GPU backend.");
  }

  if (is_qnn_ctx_model) {
    ep->PartitionCtxModel(graph, graph_support_info);
    return nullptr;
  }

  // GetCapability runs up to twice. Clear before pass 2 so its collection
  // supersedes pass 1; pass 1's data is kept for the all-unsupported flush below
  // (reached only on pass 1, since pass 2 doesn't run when nothing is supported).
  if (ep->enable_framework_op_trace_ && ep->is_post_layout_transform_) {
    ep->op_trace_builder_.Reset();
  }

  // Store original graph I/O order for use in Compile.
  // The partitioning process may reorder inputs, so we capture the original order here.
  {
    Ort::ConstGraph ort_graph{graph};

    std::vector<std::string> input_order;
    for (const auto& input : ort_graph.GetInputs()) {
      // Skip initializers - only track true model inputs
      if (!input.IsConstantInitializer()) {
        input_order.push_back(input.GetName());
      }
    }

    std::vector<std::string> output_order;
    for (const auto& output : ort_graph.GetOutputs()) {
      output_order.push_back(output.GetName());
    }

    if (!ep->onnx_graph_io_names_.has_value()) {
      ep->onnx_graph_io_names_.emplace(std::move(input_order), std::move(output_order));
    }
  }

  // Get node units for the ABI layer
  std::vector<std::unique_ptr<OrtNodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;

  std::tie(node_unit_holder, node_unit_map) = GetAllOrtNodeUnits(ep->ort_api, graph, ep->logger_);

  // Analyze nodes for QNN support
  std::vector<const OrtNode*> supported_nodes;
  if (!ep->enable_multi_soc_ep_context_) {
    RETURN_IF_NOT_NULL(ep->GetSupportedNodes(graph,
                                             node_unit_map,
                                             node_unit_holder.size(),
                                             supported_nodes,
                                             ep->op_trace_builder_.UnsupportedNodes()));
  } else {
    RETURN_IF_NOT_NULL(ep->GetMultiSocSupportedNodes(graph,
                                                     node_unit_map,
                                                     node_unit_holder.size(),
                                                     supported_nodes,
                                                     ep->op_trace_builder_.UnsupportedNodes()));
  }

  // Helper function that returns a string that lists all unsupported nodes.
  // Ex: { name: mul_123, type: Mul }, {}, ...
  auto get_unsupported_node_names = [&node_unit_holder, &supported_nodes]() -> std::string {
    std::stringstream ss;
    const size_t num_node_units = node_unit_holder.size();

    for (size_t i = 0; i < num_node_units; ++i) {
      const auto& node_unit = node_unit_holder[i];

      auto it = std::find(supported_nodes.begin(), supported_nodes.end(), &node_unit->GetNode());
      if (it == supported_nodes.end()) {
        ss << "{ name: " << node_unit->Name() << ", type: " << node_unit->OpType() << " }";
        if (i == num_node_units - 1) {
          ss << ", ";
        }
      }
    }

    return ss.str();
  };

  // If no supported nodes, return empty
  if (supported_nodes.empty()) {
    // ORT does not invoke CompileImpl when no nodes are supported, so the
    // unsupported list collected in GetSupportedNodes would otherwise be
    // discarded. Flush a trace here so the diagnostic record (why every node
    // was rejected) is preserved.
    if (ep->enable_framework_op_trace_ && !ep->op_trace_builder_.UnsupportedNodes().empty()) {
      // CompileImpl is skipped here, so record every configured SoC (multi-SoC)
      // or the live target (single-SoC) before writing.
      if (ep->enable_multi_soc_ep_context_) {
        ep->AppendMultiSocTraces();
      } else {
        ep->AppendSingleSocTrace();
      }
      ep->WriteFrameworkOpTrace(graph);
    }
    return nullptr;
  }

  size_t num_of_supported_nodes = 0;
  std::vector<std::vector<const OrtNode*>> partitions = utils::CreateSupportedPartitionNodeGroups(graph,
                                                                                                  ep->ort_api,
                                                                                                  supported_nodes,
                                                                                                  ep->name_,
                                                                                                  node_unit_map);

  // Filter out partitions that consist of a single QuantizeLinear or DequantizeLinear node.
  // We also count the number of supported nodes in all valid partitions.
  for (const std::vector<const OrtNode*>& partition : partitions) {
    size_t nodes_in_partition = partition.size();
    if (nodes_in_partition == 1 && !is_qnn_ctx_model) {
      const char* op_type = nullptr;
      auto partition_op_status = ep->ort_api.Node_GetOperatorType(partition[0], &op_type);
      if (partition_op_status != nullptr) {
        ep->ort_api.ReleaseStatus(partition_op_status);
        continue;
      }
      if (std::string(op_type) == "QuantizeLinear" || std::string(op_type) == "DequantizeLinear") {
        ORT_CXX_LOG(ep->logger_,
                    ORT_LOGGING_LEVEL_WARNING,
                    "QNN EP does not support a single Quantize/Dequantize node in a partition.");
        continue;
      }
    }

    OrtNodeFusionOptions node_fusion_options = {};
    node_fusion_options.ort_version_supported = ORT_API_VERSION;
    RETURN_IF_NOT_NULL(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                    partition.data(),
                                                                    partition.size(),
                                                                    &node_fusion_options));

    num_of_supported_nodes += nodes_in_partition;
  }

  // Print list of unsupported nodes to the ERROR logger if the CPU EP
  // has been disabled for this inference session.
  if (!is_qnn_ctx_model && ep->disable_cpu_ep_fallback_ && num_nodes_in_graph != num_of_supported_nodes) {
    ORT_CXX_LOG(ep->logger_,
                ORT_LOGGING_LEVEL_ERROR,
                ("Unsupported nodes in QNN EP: " + get_unsupported_node_names()).c_str());
  }

  return nullptr;
}

OrtStatus* QnnEp::CompileOnnxModel(const OrtGraph** graphs,
                                   const OrtNode** fused_nodes,
                                   size_t count,
                                   OrtNodeComputeInfo** node_compute_infos,
                                   const qnn::HtpGraphConfigs_t& htp_graph_configs,
                                   bool collect_subgraph_traces) {
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // Initialize now for possible reuse in loop
  auto finalize_start = std::chrono::steady_clock::time_point::min();
  auto end = std::chrono::steady_clock::time_point::min();
  std::chrono::milliseconds total_finalize_time{0};

  auto compile_start = std::chrono::steady_clock::now();
  std::vector<GraphFinalizationInfo_t> model_infos;

  bool use_multithreaded_prepare = count >= 5 || num_graph_prepare_threads_ > 1;
  if (use_multithreaded_prepare) {
    model_infos.reserve(count);
  } else {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Only using single thread for graph prepare due to graph count (" + std::to_string(count) + ") or user request.").c_str());
  }
#endif

  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    const OrtGraph* graph = graphs[graph_idx];
    const OrtNode* fused_node = fused_nodes[graph_idx];
    const std::string fused_node_name = Ort::ConstNode(fused_node).GetName();

    std::unique_ptr<qnn::QnnModel> qnn_model = std::make_unique<qnn::QnnModel>(
        qnn_backend_manager_.get(), ApiPtrs{ort_api, ep_api, model_editor_api});

    qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t> htp_graph_configs_builder(
        QNN_GRAPH_CONFIG_INIT, QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
    InitQnnHtpGraphConfigs(htp_graph_configs, htp_graph_configs_builder);

    std::vector<const QnnGraph_Config_t*> all_graph_configs;
    const QnnGraph_Config_t** htp_configs = htp_graph_configs_builder.GetQnnConfigs();
    if (htp_configs) {
      // Reserve enough for configs + nullptr
      all_graph_configs.reserve(htp_graph_configs_builder.GetSize() + 1);
      for (const QnnGraph_Config_t** config = htp_configs; *config; ++config) {
        all_graph_configs.push_back(*config);
      }
    }

    qnn::QnnSerializerConfig* qnn_serializer_config = qnn_backend_manager_->GetQnnSerializerConfig();
    if (qnn_serializer_config) {
      // We don't bother reserving here to keep the API simpler. Also note that if we're here,
      // we're likely debugging and not waiting for inference.
      qnn_serializer_config->SetGraphName(fused_node_name);
      const QnnGraph_Config_t** serializer_configs = qnn_serializer_config->Configure();
      if (serializer_configs) {
        for (const QnnGraph_Config_t** config = serializer_configs; *config; ++config) {
          all_graph_configs.push_back(*config);
        }
      }
    }

    const QnnGraph_Config_t** all_graph_configs_ptr = nullptr;
    if (!all_graph_configs.empty()) {
      all_graph_configs.push_back(nullptr);
      all_graph_configs_ptr = all_graph_configs.data();
    }

    // Get original input/output order captured in GetCapability
    if (!onnx_graph_io_names_.has_value()) {
      return ort_api.CreateStatus(ORT_EP_FAIL,
                                  "ONNX I/O names not found. GetCapability must be called before Compile.");
    }
    const auto& onnx_input_names = onnx_graph_io_names_->first;
    const auto& onnx_output_names = onnx_graph_io_names_->second;

    // Build context for graph composition
    qnn::QnnModelContext context{
        /*ort_graph=*/*graph,
        /*fused_node=*/*fused_node,
        /*logger=*/logger_,
        /*onnx_input_names=*/&onnx_input_names,
        /*onnx_output_names=*/&onnx_output_names,
        /*model_settings=*/&model_settings_,
        /*graph_configs=*/all_graph_configs_ptr,
        /*tensor_name_overrides=*/&tensor_name_overrides_,
        /*json_qnn_graph_path=*/std::string{}};

    // subgraph_traces are SoC-agnostic, so the FCB multi-SoC loop collects them
    // only on its first iteration (collect_subgraph_traces=false thereafter).
    if (enable_framework_op_trace_ && collect_subgraph_traces) {
      context.op_trace_output = op_trace_builder_.NewSubgraphSlot();
    }

    if (dump_json_qnn_graph_) {
      namespace fs = std::filesystem;
      context.json_qnn_graph_path =
          (fs::path(json_qnn_graph_dir_) / fs::path(fused_node_name + ".json")).string();
    }

    RETURN_IF_NOT_OK(qnn_model->ComposeGraph(context));
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
    if (use_multithreaded_prepare) {
      auto& model_info = model_infos.emplace_back();
      model_info.model_name = fused_node_name;
      model_info.model = std::move(qnn_model);
      model_info.graph_idx = graph_idx;
    } else {
      finalize_start = std::chrono::steady_clock::now();
#endif
      {
        // Boost the HTP only around graphFinalize() -- the accelerator-side graph
        // compilation. ComposeGraph above is host-side (graphAddNode), so the HTP is
        // deliberately left relaxed during it. The perf config id is valid here in both
        // the single-SoC path (created at GetCapability) and the multi-SoC path (created
        // per-SoC by ScopedPerSocQnnBackendSetup::Init before CompileOnnxModel runs).
        uint32_t htp_power_config_id = 0;
        bool valid_power_config_id = GetHtpPowerConfigId(htp_power_config_id);
        qnn::power::HtpPerfConfig_t perf_config{htp_power_config_id, default_htp_performance_mode_,
                                                default_rpc_polling_time_, default_rpc_control_latency_};
        qnn::HtpPowerStateGuard power_guard(
            &qnn_backend_manager_->GetHtpPowerConfigManager(),
            valid_power_config_id,
            qnn::power::GraphState::INIT_START, qnn::power::GraphState::INIT_DONE,
            perf_config,
            logger_);
        RETURN_IF_NOT_OK(power_guard.SetPreRunHtpPerfStatus());
        RETURN_IF_NOT_OK(qnn_model->FinalizeGraphs(logger_));
        RETURN_IF_NOT_OK(power_guard.SetPostRunHtpPerf());
      }
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
      end = std::chrono::steady_clock::now();
      total_finalize_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - finalize_start);
#endif

      // SetupQnnInputOutput populates qnn_input_infos_/qnn_output_infos_ which are only consumed
      // in ExecuteGraph during inference. In prepare_only/prepare_and_load mode these compile-time
      // models are discarded after binary extraction, so skip.
      if (!prepare_only_ && !prepare_and_load_) {
        RETURN_IF_NOT_OK(qnn_model->SetupQnnInputOutput(logger_));
      }

      qnn_models_.emplace(fused_node_name, std::move(qnn_model));

      auto node_compute_info = std::make_unique<QnnNodeComputeInfo>(*this);
      node_compute_infos[graph_idx] = node_compute_info.release();
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
    }
#endif
  }

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  if (use_multithreaded_prepare) {
    qnn::thread::QnnJobThreadPool tp(num_graph_prepare_threads_);
    tp.Start();
    finalize_start = std::chrono::steady_clock::now();

    // Boost the HTP once around the whole finalize batch. graphFinalize() is the
    // accelerator-side work; the preceding ComposeGraph loop is host-side and runs
    // relaxed. The boost is a single global vote on the perf config id, so one guard
    // for the batch is correct -- do not vote from inside the per-thread jobs.
    uint32_t htp_power_config_id = 0;
    bool valid_power_config_id = GetHtpPowerConfigId(htp_power_config_id);
    qnn::power::HtpPerfConfig_t perf_config{htp_power_config_id, default_htp_performance_mode_,
                                            default_rpc_polling_time_, default_rpc_control_latency_};
    qnn::HtpPowerStateGuard power_guard(
        &qnn_backend_manager_->GetHtpPowerConfigManager(),
        valid_power_config_id,
        qnn::power::GraphState::INIT_START, qnn::power::GraphState::INIT_DONE,
        perf_config,
        logger_);
    RETURN_IF_NOT_OK(power_guard.SetPreRunHtpPerfStatus());

    for (auto& model_info : model_infos) {
      tp.SubmitJob([qnn_model = model_info.model.get(), &logger = logger_, res = &model_info.result] {
        *res = qnn_model->FinalizeGraphs(logger);
      });
    }
    tp.WaitForQueuedJobsToFinish();
    RETURN_IF_NOT_OK(power_guard.SetPostRunHtpPerf());
    end = std::chrono::steady_clock::now();
    total_finalize_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - finalize_start);

    for (auto& model_info : model_infos) {
      RETURN_IF_NOT_OK(std::move(model_info.result));

      auto qnn_model = std::move(model_info.model);
      if (!prepare_only_ && !prepare_and_load_) {
        RETURN_IF_NOT_OK(qnn_model->SetupQnnInputOutput(logger_));
      }

      qnn_models_.emplace(model_info.model_name, std::move(qnn_model));

      auto node_compute_info = std::make_unique<QnnNodeComputeInfo>(*this);
      node_compute_infos[model_info.graph_idx] = node_compute_info.release();
    }
  }

  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("Total finalize time for all fused nodes: " + std::to_string(total_finalize_time.count()) + " ms").c_str());
#endif  // _WIN32

  return nullptr;
}

OrtStatus* QnnEp::CompileMultiSocOnnxModel(const OrtGraph** graphs,
                                           const OrtNode** fused_nodes,
                                           size_t count,
                                           OrtNodeComputeInfo** node_compute_infos) {
  // Iterate each SoC and compile.
  for (size_t idx = 0; idx < htp_arch_per_soc_.size(); ++idx) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Compiling model for HTP arch " + std::to_string(htp_arch_per_soc_[idx]) +
                 " and SoC model " + std::to_string(soc_model_per_soc_[idx]) + ".")
                    .c_str());

    // Complete setup for device and context.
    ScopedPerSocQnnBackendSetup scoped_backend_setup(*this);
    RETURN_IF_NOT_OK(scoped_backend_setup.Init(idx));

    // Collect subgraph_traces once (on idx 0): the ONNX->QNN op mapping is the
    // same across SoCs, since which QNN op a node lowers to does not depend on arch.
    RETURN_IF_NOT_NULL(CompileOnnxModel(graphs,
                                        fused_nodes,
                                        count,
                                        node_compute_infos,
                                        htp_graph_configs_per_soc_[idx],
                                        /*collect_subgraph_traces=*/idx == 0));

    if (idx != htp_arch_per_soc_.size() - 1) {
      // `qnn_models_` and `node_compute_infos` are repeatedly set in each `CompileOnnxModel` call, where previous
      // values are not properly freed. Since we know multi-SoC preparation usecase could not run inference, these
      // objects are in fact useless.
      // Note that the objects in the last iteration are deliberately kept and guarded by `prepare_only` flag that
      // they will never be used.
      for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
        delete static_cast<QnnNodeComputeInfo*>(node_compute_infos[graph_idx]);
        node_compute_infos[graph_idx] = nullptr;
        qnn_models_.clear();
      }
    }
    RETURN_IF_NOT_NULL(qnn_backend_manager_->AddContextToDlc());
  }

  return nullptr;
}

OrtStatus* QnnEp::CompileContextModel(const OrtGraph** graphs,
                                      const OrtNode** fused_nodes,
                                      size_t count,
                                      OrtNodeComputeInfo** node_compute_infos) {
  if (enable_framework_op_trace_) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO,
                "Framework op tracing requested but graph is loaded from EPContext "
                "(no fresh ComposeGraph) - no trace file will be written.");
  }

  // Collect graph and fused nodes names.
  std::vector<std::pair<std::string, std::string>> names;
  names.reserve(count);
  std::vector<std::unordered_map<std::string, std::string>> io_name_overrides_per_graph(count);

  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    const char* graph_name = nullptr;
    auto compile_status = ort_api.Node_GetName(fused_nodes[graph_idx], &graph_name);
    if (compile_status != nullptr) {
      ort_api.ReleaseStatus(compile_status);
      return ort_api.CreateStatus(ORT_EP_FAIL, "Failed to get fused node name.");
    }

    size_t num_nodes = 0;
    compile_status = ort_api.Graph_GetNumNodes(graphs[graph_idx], &num_nodes);
    if (compile_status != nullptr) {
      ort_api.ReleaseStatus(compile_status);
      return ort_api.CreateStatus(ORT_EP_FAIL, "Failed to get number of nodes in graph.");
    }

    std::vector<const OrtNode*> graph_nodes(num_nodes);
    compile_status = ort_api.Graph_GetNodes(graphs[graph_idx], graph_nodes.data(), graph_nodes.size());
    if (compile_status != nullptr) {
      ort_api.ReleaseStatus(compile_status);
      return ort_api.CreateStatus(ORT_EP_FAIL, "Failed to get graph nodes.");
    }

    const OrtNode* ep_context_node = graph_nodes[0];
    const char* ep_context_node_name = nullptr;
    compile_status = ort_api.Node_GetName(ep_context_node, &ep_context_node_name);
    if (compile_status != nullptr) {
      ort_api.ReleaseStatus(compile_status);
      return ort_api.CreateStatus(ORT_EP_FAIL, "Failed to get EP context node name.");
    }

    names.push_back(std::pair<std::string, std::string>(graph_name, ep_context_node_name));
    io_name_overrides_per_graph[graph_idx] = qnn::ParseIoNameOverrides(ep_context_node);
  }

  // Get QnnModel from EP shared contexts
  if (share_ep_contexts_ && SharedContext::GetInstance().HasSharedQnnModels()) {
    bool has_all_graphs = true;
    for (const auto& name_pair : names) {
      if (!SharedContext::GetInstance().HasQnnModel(name_pair.second)) {
        has_all_graphs = false;
        ORT_CXX_LOG(logger_,
                    ORT_LOGGING_LEVEL_VERBOSE,
                    ("Graph: " + name_pair.second + " from EpContext node not found from shared EP contexts.").c_str());
        break;
      }
    }

    if (has_all_graphs) {
      for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
        auto qnn_model_shared = SharedContext::GetInstance().GetSharedQnnModel(names[graph_idx].second);
        if (qnn_model_shared == nullptr) {
          return ort_api.CreateStatus(ORT_EP_FAIL,
                                      ("Graph: " + names[graph_idx].second +
                                       " not found from shared EP contexts.")
                                          .c_str());
        }

        qnn::QnnModelContext context{
            /*ort_graph=*/*graphs[graph_idx],
            /*fused_node=*/*fused_nodes[graph_idx],
            /*logger=*/logger_,
            /*onnx_input_names=*/nullptr,
            /*onnx_output_names=*/nullptr,
            /*model_settings=*/nullptr,
            /*graph_configs=*/nullptr,
            /*tensor_name_overrides=*/io_name_overrides_per_graph[graph_idx].empty()
                ? nullptr
                : &io_name_overrides_per_graph[graph_idx],
            /*json_qnn_graph_path=*/{}};
        RETURN_IF_NOT_OK(qnn_model_shared->SetGraphInputOutputInfo(context));
        RETURN_IF_NOT_OK(qnn_model_shared->SetupQnnInputOutput(logger_));
        RETURN_IF_NOT_OK(qnn_model_shared->ApplyRuntimeGraphConfigs(htp_graph_configs_, logger_));
        qnn_models_.emplace(names[graph_idx].first, std::move(qnn_model_shared));

        auto node_compute_info = std::make_unique<QnnNodeComputeInfo>(*this);
        node_compute_infos[graph_idx] = node_compute_info.release();
      }

      return nullptr;
    }
  }

  // Table<EPContext node name, QnnModel>, the node name is the graph_meta_id (old) created from user model which used
  // to generate the EP context model for this session (created from an EP context model), the graph_meta_id is new
  qnn::QnnModelLookupTable qnn_models;

  std::vector<int> main_context_pos_list;
  RETURN_IF_NOT_OK(qnn::GetMainContextNode(graphs, count, ort_api, main_context_pos_list));
  uint32_t total_context_size = SafeInt<uint32_t>(main_context_pos_list.size());

  int64_t max_spill_fill_size = 0;

  // Adjust the main_context_pos_list, move the one with max spill fill buffer to the beginning
  // HTP spill fill buffer only works for multiple QNN contexts generated after QNN v2.28
  if (total_context_size > 1) {
    RETURN_IF_NOT_OK(qnn::TryGetMaxSpillFillSize(graphs,
                                                 ort_api,
                                                 total_context_size,
                                                 max_spill_fill_size,
                                                 main_context_pos_list));
  }

  // Figure out the EP context model path from session option
  std::basic_string<ORTCHAR_T> model_path = GetModelPathString(graphs[0], ort_api);
  std::basic_string<ORTCHAR_T> context_model_path;
  GetContextOnnxModelFilePath(context_cache_path_cfg_, model_path, context_model_path);

  // AOT Phase 2 sidecar discovery for profiling enrichment. Loaded here
  // (before any context binary is restored) so the lookup is on the backend
  // manager when the first profile extraction runs, ensuring InitCsvFile()
  // emits the `ONNX Source Ops` column and every NODE event row is annotated.
  if (enable_framework_op_trace_) {
    auto trace_path = qnn::DeriveTracePathFromContextModel(std::filesystem::path(context_model_path));
    std::error_code ec;
    if (std::filesystem::exists(trace_path, ec) && !ec) {
      qnn::OpTraceLookup loaded;
      if (qnn::LoadTraceLookupFromFile(trace_path, loaded, logger_)) {
        qnn_backend_manager_->SetOpTraceLookup(std::move(loaded));
      }
    } else {
      ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO,
                  ("No sidecar op trace found at: " + trace_path.string() +
                   " - the `ONNX Source Ops` profiling CSV column will be present "
                   "(framework op trace was requested) but every NODE row's annotation "
                   "will be empty.")
                      .c_str());
    }
  }

  for (auto main_context_pos : main_context_pos_list) {
    // Create QNN context from the cached binary, deserialize the QNN graph from the binary
    RETURN_IF_NOT_OK(qnn::LoadQnnCtxFromOnnxGraph(graphs[main_context_pos],
                                                  ort_api,
                                                  context_model_path,
                                                  qnn_backend_manager_.get(),
                                                  qnn_models,
                                                  logger_,
                                                  max_spill_fill_size,
                                                  *io_dispatch_));
  }

  std::string graph_name;
  std::string ep_context_node_name;
  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    std::tie(graph_name, ep_context_node_name) = names[graph_idx];

    // Try to find the model using ep_context_node_name first, then try the main context node name
    auto qnn_model_it = qnn_models.find(ep_context_node_name);
    if (qnn_model_it == qnn_models.end()) {
      // For single graph contexts, the model might be stored with the main context node name
      qnn_model_it = qnn_models.find(names[main_context_pos_list[0]].second);
    }

    if (qnn_model_it == qnn_models.end()) {
      return ort_api.CreateStatus(ORT_EP_FAIL,
                                  (ep_context_node_name +
                                   " context node name not exists in table qnn_models.")
                                      .c_str());
    }

    auto qnn_model = std::move(qnn_model_it->second);
    qnn::QnnModelContext context{
        /*ort_graph=*/*graphs[graph_idx],
        /*fused_node=*/*fused_nodes[graph_idx],
        /*logger=*/logger_,
        /*onnx_input_names=*/nullptr,
        /*onnx_output_names=*/nullptr,
        /*model_settings=*/nullptr,
        /*graph_configs=*/nullptr,
        /*tensor_name_overrides=*/io_name_overrides_per_graph[graph_idx].empty()
            ? nullptr
            : &io_name_overrides_per_graph[graph_idx],
        /*json_qnn_graph_path=*/{}};
    RETURN_IF_NOT_OK(qnn_model->SetGraphInputOutputInfo(context));
    RETURN_IF_NOT_OK(qnn_model->SetupQnnInputOutput(logger_));
    RETURN_IF_NOT_OK(qnn_model->ApplyRuntimeGraphConfigs(htp_graph_configs_, logger_));

    // fused node name is QNNExecutionProvider_QNN_[hash_id]_[id]
    // the name here must be same with context->node_name in compute_info
    qnn_models_.emplace(graph_name, std::move(qnn_model));
    qnn_models.erase(qnn_model_it->first);

    auto node_compute_info = std::make_unique<QnnNodeComputeInfo>(*this);
    node_compute_infos[graph_idx] = node_compute_info.release();
  }

  if (share_ep_contexts_ && qnn_models.size() > 0) {
    std::vector<std::unique_ptr<qnn::QnnModel>> shared_qnn_models;
    for (auto& [key, value] : qnn_models) {
      shared_qnn_models.push_back(std::move(qnn_models[key]));
    }
    std::string duplicate_graph_names;
    bool has_duplicate_graph = SharedContext::GetInstance().SetSharedQnnModel(std::move(shared_qnn_models),
                                                                              duplicate_graph_names);
    if (has_duplicate_graph) {
      return ort_api.CreateStatus(ORT_EP_FAIL,
                                  ("Duplicate graph names detect across sessions: " + duplicate_graph_names).c_str());
    }
  }

  return nullptr;
}

OrtStatus* QnnEp::CreateEPContextNodes(const OrtGraph* graph,
                                       const OrtNode** fused_nodes,
                                       size_t count,
                                       OrtNode** ep_context_nodes) {
  // All partitioned graph share single QNN context, included in the same context binary
  unsigned char* raw_context_buffer = nullptr;
  uint64_t buffer_size = 0;
  RETURN_IF_NOT_OK(qnn_backend_manager_->GetContextBinaryBuffer(enable_multi_soc_ep_context_,
                                                                &raw_context_buffer,
                                                                buffer_size));
  std::unique_ptr<unsigned char[]> context_buffer(raw_context_buffer);
  // Get max spill fill buffer size
  uint64_t max_spill_fill_buffer_size = 0;
  if (enable_spill_fill_buffer_) {
    RETURN_IF_NOT_OK(qnn_backend_manager_->GetMaxSpillFillBufferSize(context_buffer.get(),
                                                                     buffer_size,
                                                                     enable_multi_soc_ep_context_,
                                                                     max_spill_fill_buffer_size));
  }

  // Figure out the EP context model path from session option
  std::basic_string<ORTCHAR_T> model_path = GetModelPathString(graph, ort_api);
  std::basic_string<ORTCHAR_T> context_model_path;
  GetContextOnnxModelFilePath(context_cache_path_cfg_, model_path, context_model_path);

  RETURN_IF_NOT_OK(qnn::CreateEPContextNodes(fused_nodes,
                                             count,
                                             ep_context_nodes,
                                             ort_api,
                                             model_editor_api,
                                             context_buffer.get(),
                                             buffer_size,
                                             qnn_backend_manager_->GetSdkVersion(),
                                             qnn_models_,
                                             context_model_path,
                                             qnn_context_embed_mode_,
                                             max_spill_fill_buffer_size,
                                             logger_,
                                             share_ep_contexts_,
                                             stop_share_ep_contexts_,
                                             name_,
                                             tensor_name_overrides_,
                                             enable_multi_soc_ep_context_,
                                             *io_dispatch_));

  // Get V2 compatibility info for later query in GetCompiledModelCompatibilityInfo.
  qnn::QnnCompatibilityInfoV2& info_v2 = std::get<qnn::QnnCompatibilityInfoV2>(compatibility_info_.info);
  if (enable_multi_soc_ep_context_) {
    // Manually set per-SoC configs here as QnnBackendManager did not record this info.
    info_v2.htp_archs.reserve(htp_arch_per_soc_.size());
    std::transform(htp_arch_per_soc_.begin(),
                   htp_arch_per_soc_.end(),
                   std::back_inserter(info_v2.htp_archs),
                   [](QnnHtpDevice_Arch_t htp_arch) { return static_cast<uint32_t>(htp_arch); });

    info_v2.soc_models = soc_model_per_soc_;

    info_v2.vtcm_mbs.reserve(htp_graph_configs_per_soc_.size());
    std::transform(htp_graph_configs_per_soc_.begin(),
                   htp_graph_configs_per_soc_.end(),
                   std::back_inserter(info_v2.vtcm_mbs),
                   [](const qnn::HtpGraphConfigs_t& config) { return static_cast<uint32_t>(config.vtcm_size_in_mb); });
  } else {
    // Manually set VTCM size here as it is not passed into QnnBackendManager.
    info_v2.vtcm_mbs.push_back(static_cast<uint32_t>(htp_graph_configs_.vtcm_size_in_mb));
  }

  Ort::Status status = qnn_cache_compatibility_manager_->GetCompatibilityInfo(compatibility_info_);
  if (!status.IsOK()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING,
                ("Failed to get compatibility info. " + status.GetErrorMessage()).c_str());
  }

  if (share_ep_contexts_ &&
      !stop_share_ep_contexts_ &&
      nullptr == SharedContext::GetInstance().GetSharedQnnBackendManager()) {
    if (!SharedContext::GetInstance().SetSharedQnnBackendManager(qnn_backend_manager_)) {
      return ort_api.CreateStatus(ORT_EP_FAIL, "Failed to set shared QnnBackendManager.");
    }
  }

  return nullptr;
}

OrtStatus* QnnEp::CompileDlcContextModel(OrtEp* this_ptr,
                                         const OrtGraph** graphs,
                                         const OrtNode** fused_nodes,
                                         size_t count,
                                         OrtNodeComputeInfo** node_compute_infos) {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);
  if (ep->enable_framework_op_trace_) {
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                "Framework op tracing requested but graph is loaded from EPContext "
                "(no fresh ComposeGraph) - no trace file will be written.");
  }

  std::basic_string<ORTCHAR_T> model_path = GetModelPathString(graphs[0], ep->ort_api);
  std::basic_string<ORTCHAR_T> context_model_path;
  GetContextOnnxModelFilePath(ep->context_cache_path_cfg_, model_path, context_model_path);
  std::filesystem::path parent_path = std::filesystem::path(context_model_path).parent_path();

  // Extract the DLC information
  std::string dlc_path;
  RETURN_IF_NOT_OK(qnn::GetEpContextDlcPath(graphs, count, ep->ort_api, dlc_path));
  std::filesystem::path dlc_extracted_path(parent_path / dlc_path);

  // Populate the Genie APIs
  const GenieApi& genie_api_ = ep->genie_api_loader_->Get();

  // NOTE: We will have only one node in the fused graph --> Ep_context_node
  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    const OrtNode* fused_node = fused_nodes[graph_idx];

    // Build everything you can at compile time
    auto builder = std::make_shared<GenieNodeBuilder>();
    builder->api = &genie_api_;
    builder->dlc_path = dlc_extracted_path.string();
    RETURN_IF_NOT_NULL(ep->ort_api.Node_GetNumInputs(fused_node, &(builder->num_inputs)));
    RETURN_IF_NOT_NULL(ep->ort_api.Node_GetNumOutputs(fused_node, &(builder->num_outputs)));

    auto node_compute_info = std::make_unique<GenieNodeComputeInfo>(*ep, builder);
    node_compute_infos[graph_idx] = node_compute_info.release();
  }
  return nullptr;
}

void QnnEp::AppendSingleSocTrace() {
  // Single-SoC path: encode from live backend state.
  if (qnn::IsNpuBackend(qnn_backend_manager_->GetQnnBackendType())) {
    op_trace_builder_.AppendSoc(qnn_backend_manager_->GetHtpArch(),
                                qnn_backend_manager_->GetSocModel(),
                                device_id_);
  } else {
    // Non-NPU backends have no arch/soc_model; pass the QNN "unknown" sentinels.
    op_trace_builder_.AppendSoc(QNN_HTP_DEVICE_ARCH_NONE, QNN_SOC_MODEL_UNKNOWN, device_id_);
  }
}

void QnnEp::AppendMultiSocTraces() {
  // One SocTrace per configured (htp_arch, soc_model) entry, from the member arrays.
  for (size_t idx = 0; idx < htp_arch_per_soc_.size(); ++idx) {
    op_trace_builder_.AppendSoc(htp_arch_per_soc_[idx], soc_model_per_soc_[idx], device_id_);
  }
}

void QnnEp::WriteFrameworkOpTrace(const OrtGraph* primary_graph) {
  std::string model_name = std::filesystem::path(GetModelPathString(primary_graph, ort_api)).filename().u8string();
  op_trace_builder_.FinalizeAndWrite(model_name,
                                     qnn::QnnBackendTypeToString(qnn_backend_manager_->GetQnnBackendType()),
                                     std::filesystem::path(framework_op_trace_dir_) / "qnn_op_trace.json",
                                     logger_);
}

OrtStatus* ORT_API_CALL QnnEp::CompileImpl(_In_ OrtEp* this_ptr,
                                           _In_ const OrtGraph** graphs,
                                           _In_ const OrtNode** fused_nodes,
                                           _In_ size_t count,
                                           _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                           _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  if (qnn::IsOrtGraphHasCtxNode(graphs, count, ep->ort_api)) {
    if (ep->prepare_and_load_) {
      ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_WARNING,
                  "prepare_and_load=1 is ignored because the input model is already a pre-compiled context model. "
                  "The model will be loaded directly via the AOT path.");
    }
    if (ep->enable_htp_graph_splitting_) {
      ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_WARNING,
                  "enable_htp_graph_splitting=1 has no effect because the input model is already a pre-compiled "
                  "context binary. Graph splitting is applied at context creation time, not at load time.");
    }
    uint32_t htp_power_config_id = 0;
    bool power_config_valid = ep->GetHtpPowerConfigId(htp_power_config_id);
    qnn::power::HtpPerfConfig_t perf_config{htp_power_config_id, ep->default_htp_performance_mode_, ep->default_rpc_polling_time_, ep->default_rpc_control_latency_};
    qnn::HtpPowerStateGuard power_guard(
        &ep->qnn_backend_manager_->GetHtpPowerConfigManager(),
        power_config_valid,
        qnn::power::GraphState::INIT_START, qnn::power::GraphState::INIT_DONE,
        perf_config,
        ep->logger_);
    RETURN_IF_NOT_OK(power_guard.SetPreRunHtpPerfStatus());
    auto status = ep->CompileContextModel(graphs, fused_nodes, count, node_compute_infos);
    RETURN_IF_NOT_OK(power_guard.SetPostRunHtpPerf());
    return status;
  } else if (qnn::IsOrtGraphHasDlcCtxNode(graphs, count, ep->ort_api)) {
    uint32_t htp_power_config_id = 0;
    bool power_config_valid = ep->GetHtpPowerConfigId(htp_power_config_id);
    qnn::power::HtpPerfConfig_t perf_config{htp_power_config_id, ep->default_htp_performance_mode_, ep->default_rpc_polling_time_, ep->default_rpc_control_latency_};
    qnn::HtpPowerStateGuard power_guard(
        &ep->qnn_backend_manager_->GetHtpPowerConfigManager(),
        power_config_valid,
        qnn::power::GraphState::INIT_START, qnn::power::GraphState::INIT_DONE,
        perf_config,
        ep->logger_);
    RETURN_IF_NOT_OK(power_guard.SetPreRunHtpPerfStatus());
    auto status = ep->CompileDlcContextModel(this_ptr, graphs, fused_nodes, count, node_compute_infos);
    RETURN_IF_NOT_OK(power_guard.SetPostRunHtpPerf());
    return status;
  }

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  auto compile_start = std::chrono::steady_clock::now();
#endif

  // NOTE: the HTP perf boost is applied inside CompileOnnxModel, scoped tightly around
  // graphFinalize() (the accelerator-side work) rather than the whole compile. This keeps
  // the HTP relaxed during host-side graph composition, and -- unlike a guard here -- works
  // for the multi-SoC path, where the valid perf config id only exists per-SoC (created by
  // ScopedPerSocQnnBackendSetup::Init), not at this point.
  OrtStatus* compile_status = nullptr;
  if (!ep->enable_multi_soc_ep_context_) {
    compile_status = ep->CompileOnnxModel(graphs,
                                          fused_nodes,
                                          count,
                                          node_compute_infos,
                                          ep->htp_graph_configs_,
                                          /*collect_subgraph_traces=*/true);
  } else {
    compile_status = ep->CompileMultiSocOnnxModel(graphs, fused_nodes, count, node_compute_infos);
  }

  RETURN_IF_NOT_NULL(compile_status);

  // NOTE: onnx_graph_io_names_ cleanup is deferred to after the prepare_and_load reload block
  // (which needs the names for I/O mapping). tensor_name_overrides_ must NOT be cleared here;
  // it is read by CreateEPContextNodes below to serialize io_name_overrides into the EPContext model.

  // Framework op trace: record SoC trace(s), then finalize and write.
  if (ep->enable_framework_op_trace_) {
    if (ep->enable_multi_soc_ep_context_) {
      ep->AppendMultiSocTraces();
    } else {
      ep->AppendSingleSocTrace();
    }
    ep->WriteFrameworkOpTrace(graphs[0]);
  }

  if (ep->context_cache_enabled_) {
    RETURN_IF_NOT_NULL(ep->CreateEPContextNodes(graphs[0], fused_nodes, count, ep_context_nodes));
  }

  // Clean up tensor_name_overrides_ now that CreateEPContextNodes has serialized it.
  ep->tensor_name_overrides_.clear();

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  auto end = std::chrono::steady_clock::now();
  auto total_compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - compile_start);
  ORT_CXX_LOG(ep->logger_,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("Total compile time for all fused nodes: " + std::to_string(total_compile_time.count()) + " ms").c_str());
#endif

  // After compilation timing: reload compiled context for multi-PD inference if requested.
  // Runs after the timing window intentionally — reload is post-compile work.
  if (ep->prepare_and_load_) {
    RETURN_IF_NOT_NULL(ep->ReloadCompiledContext(graphs, fused_nodes, count));
  }

  // Clean up transient GetCapability→Compile state.
  // onnx_graph_io_names_ is deferred to here because ReloadCompiledContext needs it.
  ep->onnx_graph_io_names_.reset();

  if (ep->prepare_only_) {
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_INFO,
                "prepare_only mode: context saved. Releasing QNN device resources.");
    ep->qnn_models_.clear();
    ep->qnn_backend_manager_.reset();
  }

  return nullptr;
}

OrtStatus* QnnEp::ReloadCompiledContext(const OrtGraph** graphs,
                                        const OrtNode** fused_nodes,
                                        size_t count) {
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO,
              "prepare_and_load mode: reloading compiled context for multi-PD inference.");

  // 1. Extract the compiled context binary (while compile context is still alive).
  //    Wrap immediately in unique_ptr so the buffer is freed on all return paths (RAII).
  uint64_t buffer_size = 0;
  unsigned char* raw_context_buffer = nullptr;
  Ort::Status get_buf_status = qnn_backend_manager_->GetContextBinaryBuffer(
      /*is_multi_soc_buffer=*/false, &raw_context_buffer, buffer_size);
  if (!get_buf_status.IsOK()) {
    return ort_api.CreateStatus(ORT_EP_FAIL,
                                ("prepare_and_load: Failed to extract context binary buffer: " +
                                 std::string(get_buf_status.GetErrorMessage()))
                                    .c_str());
  }
  std::unique_ptr<unsigned char[]> context_buffer(raw_context_buffer);
  if (buffer_size == 0) {
    return ort_api.CreateStatus(ORT_EP_FAIL,
                                "prepare_and_load: Context binary buffer is empty.");
  }

  // 2. Collect fused node names before clearing models.
  InlinedVector<std::string> fused_node_names;
  fused_node_names.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const char* node_name = nullptr;
    auto st = ort_api.Node_GetName(fused_nodes[i], &node_name);
    if (st != nullptr) {
      ort_api.ReleaseStatus(st);
      return ort_api.CreateStatus(ORT_EP_FAIL, "prepare_and_load: Failed to get fused node name.");
    }
    fused_node_names.emplace_back(node_name);
  }

  // 3. Clear compile-time QNN models (drops compile-time graph handles).
  qnn_models_.clear();

  // 4. Release the compile-time QNN context (frees single-PD HW allocation).
  Ort::Status release_status = qnn_backend_manager_->ReleaseContext();
  if (!release_status.IsOK()) {
    return ort_api.CreateStatus(ORT_EP_FAIL,
                                ("prepare_and_load: ReleaseContext failed: " +
                                 std::string(release_status.GetErrorMessage()))
                                    .c_str());
  }

  // 5. Load from the in-memory binary buffer via AOT path (multi-PD).
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> loaded_models;
  Ort::Status load_status = qnn_backend_manager_->LoadCachedQnnContextFromBuffer(
      reinterpret_cast<char*>(context_buffer.get()),
      buffer_size,
      "",  // no file path needed for in-memory load
      fused_node_names[0],
      loaded_models,
      0,
      *io_dispatch_);
  if (!load_status.IsOK()) {
    return ort_api.CreateStatus(ORT_EP_FAIL,
                                ("prepare_and_load: Failed to reload context: " +
                                 std::string(load_status.GetErrorMessage()))
                                    .c_str());
  }

  // 6. Wire up loaded models: SetGraphInputOutputInfo + SetupQnnInputOutput.
  for (size_t graph_idx = 0; graph_idx < count; ++graph_idx) {
    const std::string& fused_node_name = fused_node_names[graph_idx];

    auto model_it = loaded_models.find(fused_node_name);
    if (model_it == loaded_models.end()) {
      // For single-graph case or when QNN uses different internal names,
      // take the only available model if there is exactly one.
      if (loaded_models.size() == 1 && count == 1) {
        model_it = loaded_models.begin();
      } else {
        return ort_api.CreateStatus(ORT_EP_FAIL,
                                    ("prepare_and_load: No loaded model found for fused node: " +
                                     fused_node_name)
                                        .c_str());
      }
    }

    auto qnn_model = std::move(model_it->second);
    loaded_models.erase(model_it);

    const auto& onnx_input_names = onnx_graph_io_names_->first;
    const auto& onnx_output_names = onnx_graph_io_names_->second;

    RETURN_IF_NOT_OK(qnn_model->SetGraphInputOutputInfo(
        qnn::QnnModelContext{*graphs[graph_idx], *fused_nodes[graph_idx], logger_,
                             &onnx_input_names, &onnx_output_names,
                             nullptr, nullptr, nullptr, std::string{}}));
    RETURN_IF_NOT_OK(qnn_model->SetupQnnInputOutput(logger_));

    qnn_models_.emplace(fused_node_name, std::move(qnn_model));
  }

  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_INFO,
              "prepare_and_load mode: context reload complete. Session is ready for inference.");
  return nullptr;
}

void ORT_API_CALL QnnEp::ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                     OrtNodeComputeInfo** node_compute_infos,
                                                     size_t num_node_compute_infos) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);
  for (size_t idx = 0; idx < num_node_compute_infos; ++idx) {
    // All derived types share QnnNodeComputeInfoBase, which provides the virtual destructor
    // required to safely delete through the OrtNodeComputeInfo C-API base.
    delete static_cast<QnnNodeComputeInfoBase*>(node_compute_infos[idx]);
  }
}

OrtStatus* ORT_API_CALL QnnEp::GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                          _Out_ OrtEpDataLayout* preferred_data_layout) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);
  *preferred_data_layout = OrtEpDataLayout::OrtEpDataLayout_NHWC;
  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                                _In_z_ const char* domain,
                                                                _In_z_ const char* op_type,
                                                                _In_ OrtEpDataLayout target_data_layout,
                                                                _Outptr_ int* should_convert) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);
  ORT_UNUSED_PARAMETER(target_data_layout);

  *should_convert = -1;

  if (std::string(domain) == kOnnxDomain && std::string(op_type) == "Upsample") {
    // Upsample is translated to QNN's Resize, which requires the NHWC layout for processing.
    *should_convert = 1;
  }

  if (std::string(domain) == kOnnxDomain && std::string(op_type) == "GroupNormalization") {
    // GroupNormalization is translated to QNN's GroupNorm, which requires the NHWC layout for processing.
    *should_convert = 1;
  }

  if (std::string(domain) == kOnnxDomain && std::string(op_type) == "RoiAlign") {
    // RoiAlign is translated to QNN's RoiAlign, which requires the NHWC layout for processing.
    *should_convert = 1;
  }

  if (std::string(domain) == kOnnxDomain && std::string(op_type) == "MaxRoiPool") {
    // MaxRoiPool is decomposed into StridedSlice/ReduceMax/Concat, which require the NHWC layout
    // for processing.
    *should_convert = 1;
  }

  if (std::string(domain) == kOnnxDomain && std::string(op_type) == "LpPool") {
    // LpPool is translated to a QNN AvgPool-based decomposition, which requires the NHWC layout
    // for processing.
    *should_convert = 1;
  }

  if (std::string(domain) == kOnnxDomain && std::string(op_type) == "QLinearConv") {
    // QLinearConv's activation is a bare quantized integer graph input whose quant params live in
    // the x_scale/x_zero_point op inputs, not as tensor metadata. If ORT's layout transformer were
    // allowed to insert a Transpose on that activation, the Transpose would be a plain uint8/int8
    // node with no quant params, which HTP rejects. So suppress ORT's layout transform here; the
    // QLinearConv builder performs the NCHW<->NHWC transposition internally with quant params
    // attached to every tensor (same strategy as DQConvIntegerFusion for ConvInteger).
    *should_convert = 0;
  }

  if (std::string(domain) == kOnnxDomain && std::string(op_type) == "ConvInteger") {
    // DQConvIntegerFusion handles NCHW->NHWC transposition internally via QNN Transpose ops.
    // Suppress ORT's layout transformer so it does not rewrite ConvInteger to kMSInternalNHWCDomain,
    // which would break the fusion's pattern-matching on the second GetCapability pass.
    *should_convert = 0;
  }

  return nullptr;
}

void QnnEp::GetPerThreadHtpPowerConfigs(qnn::PerThreadHtpPowerConfigs_t& per_thread_htp_power_configs,
                                        const ::OrtRunOptions* run_options) {
  qnn::HtpPerformanceMode pre_run_htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
  qnn::HtpPerformanceMode post_run_htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;

  const char* htp_perf_mode = nullptr;
  htp_perf_mode = ort_api.GetRunConfigEntry(run_options, kOrtRunOptionsConfigQnnPerfMode);
  if (htp_perf_mode != nullptr) {
    ParseHtpPerformanceMode(htp_perf_mode, pre_run_htp_performance_mode, logger_);
  }

  htp_perf_mode = nullptr;
  htp_perf_mode = ort_api.GetRunConfigEntry(run_options, kOrtRunOptionsConfigQnnPerfModePostRun);
  if (htp_perf_mode != nullptr) {
    ParseHtpPerformanceMode(htp_perf_mode, post_run_htp_performance_mode, logger_);
  }

  const char* rpc_latency = nullptr;
  rpc_latency = ort_api.GetRunConfigEntry(run_options, kOrtRunOptionsConfigQnnRpcControlLatency);
  uint32_t rpc_control_latency = 0;
  if (rpc_latency != nullptr) {
    rpc_control_latency = static_cast<uint32_t>(std::stoul(rpc_latency));
    per_thread_htp_power_configs.rpc_control_latency = rpc_control_latency;

    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, (std::string("rpc_control_latency: ") + rpc_latency).c_str());
  } else {
    per_thread_htp_power_configs.rpc_control_latency = default_rpc_control_latency_;
  }

  // This ensures that rpc polling time is always set to a value
  per_thread_htp_power_configs.rpc_polling_time = qnn::kDisableRpcPolling;

  if (qnn::HtpPerformanceMode::kHtpDefault != dynamic_htp_performance_mode_) {
    // reset perf mode, rpc control latency and rpc polling time to dynamic perf mode values
    per_thread_htp_power_configs.default_perf_mode = dynamic_htp_performance_mode_;
    per_thread_htp_power_configs.rpc_polling_time = dynamic_rpc_polling_time_;
  } else if (qnn::HtpPerformanceMode::kHtpDefault != default_htp_performance_mode_) {
    per_thread_htp_power_configs.default_perf_mode = default_htp_performance_mode_;
    per_thread_htp_power_configs.rpc_polling_time = default_rpc_polling_time_;
  }

  if (qnn::HtpPerformanceMode::kHtpDefault != pre_run_htp_performance_mode) {
    per_thread_htp_power_configs.pre_run_perf_mode = pre_run_htp_performance_mode;
    // rpc polling time will only be updated with perf mode changes
    if (qnn::HtpPerformanceMode::kHtpBurst == pre_run_htp_performance_mode) {
      per_thread_htp_power_configs.rpc_polling_time = 9999;
    }
  }

  if (qnn::HtpPerformanceMode::kHtpDefault != post_run_htp_performance_mode) {
    per_thread_htp_power_configs.post_run_perf_mode = post_run_htp_performance_mode;
  }
}

OrtStatus* ORT_API_CALL QnnEp::OnRunStartImpl(_In_ OrtEp* this_ptr, _In_ const ::OrtRunOptions* run_options) noexcept {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  if (ep->prepare_only_) {
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_VERBOSE,
                "Skipping OnRunStart in enable_htp_prepare_only mode.");
    return nullptr;
  }

  auto backend_type = ep->qnn_backend_manager_->GetQnnBackendType();
  if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
    return nullptr;
  }

  uint32_t htp_power_config_id = 0;
  if (ep->GetHtpPowerConfigId(htp_power_config_id)) {
    auto thread_id = std::this_thread::get_id();
    qnn::PerThreadHtpPowerConfigs_t per_thread_htp_power_configs;
    ep->GetPerThreadHtpPowerConfigs(per_thread_htp_power_configs, run_options);
    per_thread_htp_power_configs.power_config_id = htp_power_config_id;
    RETURN_IF_ERROR(ep->qnn_backend_manager_->AddPerThreadHtpPowerConfigMapping(thread_id,
                                                                                per_thread_htp_power_configs));
  }

  const char* lora_config = nullptr;
  lora_config = ep->ort_api.GetRunConfigEntry(run_options, kOrtRunOptionsConfigQnnLoraConfig);
  if (lora_config != nullptr) {
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_VERBOSE, (std::string("lora_config: ") + lora_config).c_str());
    RETURN_IF_NOT_OK(ep->qnn_backend_manager_->ParseLoraConfig(lora_config));
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::OnRunEndImpl(_In_ OrtEp* this_ptr,
                                            _In_ const ::OrtRunOptions* /*run_options*/,
                                            _In_ bool /*sync_stream*/) noexcept {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  if (ep->prepare_only_) {
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_VERBOSE,
                "Skipping OnRunEnd in enable_htp_prepare_only mode.");
    return nullptr;
  }

  auto backend_type = ep->qnn_backend_manager_->GetQnnBackendType();
  if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
    return nullptr;
  }

  uint32_t htp_power_config_id;
  if (ep->GetHtpPowerConfigId(htp_power_config_id)) {
    auto thread_id = std::this_thread::get_id();
    ep->qnn_backend_manager_->RemovePerThreadHtpPowerConfigMapping(thread_id);
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                   _In_ const OrtMemoryInfo* memory_info,
                                                   _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  *allocator = nullptr;
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  if (qnn::IsHtpSharedMemoryAllocator(ep->qnn_allocator_type_)) {
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_INFO, "Creating HtpSharedMemoryAllocator.");

    auto htp_allocator = std::make_unique<qnn::HtpSharedMemoryAllocator>(memory_info, ep->rpcmem_library_);
    *allocator = htp_allocator.release();
  }
#ifdef _WIN32
  else if (qnn::IsDx12SharedMemoryAllocator(ep->qnn_allocator_type_)) {
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_INFO, "Creating Dx12SharedMemoryAllocator.");

    OrtStatus* status = nullptr;
    auto dx12_allocator = std::make_unique<qnn::Dx12SharedMemoryAllocator>(memory_info, status);

    if (status != nullptr) {
      return status;
    }

    *allocator = dx12_allocator.release();
  }
#endif  // _WIN32
  return nullptr;
}

OrtStatus* ORT_API_CALL QnnEp::SetDynamicOptionsImpl(_In_ OrtEp* this_ptr,
                                                     _In_reads_(num_options) const char* const* option_keys,
                                                     _In_reads_(num_options) const char* const* option_values,
                                                     _In_ size_t num_options) noexcept {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  if (ep->prepare_only_) {
    ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_WARNING,
                "SetDynamicOptions is a no-op in enable_htp_prepare_only mode.");
    return nullptr;
  }

  for (size_t opt_idx = 0; opt_idx < num_options; ++opt_idx) {
    std::string key(option_keys[opt_idx]);
    std::string value(option_values[opt_idx]);

    if (key == "kvcache_rewind") {
      uint64_t rewind_value = std::stoull(value);
      if (!(ep->genie_backend_manager_)) {
        ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_ERROR, ("Invalid EP Workload Type: " + value).c_str());
        return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Genie Execution Not Set.");
      }
      ep->genie_kv_cache_rewind_.store(rewind_value, std::memory_order_release);
    } else if (key == kOrtEpDynamicOptionsWorkloadType) {
      if (value == "Default") {
        RETURN_IF_NOT_OK(ep->qnn_backend_manager_->ResetContextPriority());
      } else if (value == "Efficient") {
        RETURN_IF_NOT_OK(ep->qnn_backend_manager_->SetContextPriority(qnn::ContextPriority::LOW));
      } else {
        ORT_CXX_LOG(ep->logger_, ORT_LOGGING_LEVEL_ERROR, ("Invalid EP Workload Type: " + value).c_str());
        return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid EP Workload Type.");
      }
    } else if (key == kOrtEpDynamicOptionsQnnHtpPerformanceMode) {
      auto backend_type = ep->qnn_backend_manager_->GetQnnBackendType();
      if (qnn::QnnBackendType::HTP != backend_type && qnn::QnnBackendType::DSP != backend_type) {
        return nullptr;
      }
      qnn::HtpPerformanceMode htp_performance_mode = qnn::HtpPerformanceMode::kHtpDefault;
      ParseHtpPerformanceMode(value, htp_performance_mode, ep->logger_);
      // Dynamic HTP performance mode is used for performance setting for execute so it will be set in OnRunStart.
      if (htp_performance_mode != qnn::HtpPerformanceMode::kHtpDefault) {
        ep->dynamic_htp_performance_mode_ = htp_performance_mode;
        if (htp_performance_mode == qnn::HtpPerformanceMode::kHtpBurst) {
          ep->dynamic_rpc_polling_time_ = 9999;
        } else {
          ep->dynamic_rpc_polling_time_ = 0;
        }
      } else {
        // Reset the dynamic override so a caller can revert to default. OnRunStart
        // prefers dynamic_htp_performance_mode_ over the session default, so without
        // this reset a previously set non-default mode would latch permanently. With
        // it cleared, OnRunStart falls back to the session default perf mode (or
        // applies nothing if that is also default).
        ep->dynamic_htp_performance_mode_ = qnn::HtpPerformanceMode::kHtpDefault;
        ep->dynamic_rpc_polling_time_ = 0;
      }
    } else {
      ORT_CXX_LOG(ep->logger_,
                  ORT_LOGGING_LEVEL_ERROR,
                  ("EP Dynamic Option \"" + key + "\" is not currently supported.").c_str());
      return ep->ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported EP Dynamic Option");
    }
  }  // end for loop

  return nullptr;
}
const char* ORT_API_CALL QnnEp::GetCompiledModelCompatibilityInfoImpl(_In_ OrtEp* this_ptr,
                                                                      _In_ const OrtGraph* /*graph*/) noexcept {
  QnnEp* ep = static_cast<QnnEp*>(this_ptr);

  Ort::Status status = ep->qnn_cache_compatibility_manager_->SerializeCompatibilityInfo(ep->compatibility_info_,
                                                                                        ep->compatibility_info_string_);
  if (!status.IsOK()) {
    ORT_CXX_LOG(ep->logger_,
                ORT_LOGGING_LEVEL_WARNING,
                ("Failed to serialize compatibility info. " + status.GetErrorMessage()).c_str());
    return "";
  }

  ORT_CXX_LOG(ep->logger_,
              ORT_LOGGING_LEVEL_INFO,
              ("Model compatibility info: " + ep->compatibility_info_string_).c_str());
  return ep->compatibility_info_string_.c_str();
}

OrtStatus* QnnEp::ValidateCompiledModelCompatibilityInfo(const OrtHardwareDevice* const* /*devices*/,
                                                         size_t /*num_devices*/,
                                                         const char* compatibility_info,
                                                         OrtCompiledModelCompatibility* model_compatibility) noexcept {
  std::string info_string(compatibility_info);
  if (info_string.empty()) {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING, "No compatibility info to be validated.");
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  } else {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_VERBOSE, ("Validating compatibility info: " + info_string).c_str());
  }

#if !defined(__aarch64__) && !defined(_M_ARM64) && !defined(_M_ARM64EC)
  ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_WARNING, "Skip compatibility validation on x86 platforms.");
  *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  return nullptr;
#endif  // !defined(__aarch64__) && !defined(_M_ARM64) && !defined(_M_ARM64EC)

  qnn::QnnCompatibilityInfo info;
  Ort::Status status = qnn_cache_compatibility_manager_->DeserializeCompatibilityInfo(info_string, info);
  if (!status.IsOK()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING,
                ("Skip compatibility validation due to deserialization failure: " + status.GetErrorMessage()).c_str());
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }

  // Backend is only setup in GetCapability. However, at this point, it is possible that this function is invoked
  // before any GetCapability call, and thus backend is not ready. Here, backend is setup and released with basic
  // settings for actual setup in GetCapability later. If the duplicate setup introduces significant overhead in the
  // future, we may consider setup actual backend here.
  bool is_backend_setup = qnn_backend_manager_->IsBackendSetup();
  if (!is_backend_setup) {
    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
    qnn_backend_manager_->SetupBackend(true, true, false, false, false, nullptr, dummy_map);
  }

  status = qnn_cache_compatibility_manager_->ValidateCompatibilityInfo(info, *model_compatibility);
  if (!status.IsOK()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_WARNING,
                ("Skip compatibility validation due to runtime failure: " + status.GetErrorMessage()).c_str());
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  }

  if (!is_backend_setup) {
    // Release backend to avoid interfering later usage.
    qnn_backend_manager_->ReleaseResources();
  }

  return nullptr;
}

OrtStatus* QnnEp::GetHardwareDeviceIncompatibilityDetails(const OrtHardwareDevice* /*hw*/,
                                                          OrtDeviceEpIncompatibilityDetails* details) noexcept {
  // This function is always called by temporary QnnEp, so no need to check if backend is already setup.
  std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> dummy_map;
  Ort::Status status = qnn_backend_manager_->SetupBackend(false, false, false, false, false, nullptr, dummy_map);

  if (!status.IsOK()) {
    const std::string error_message = status.GetErrorMessage();
    OrtDeviceEpIncompatibilityReason reasons = OrtDeviceEpIncompatibility_UNKNOWN;
    int32_t error_code = QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;

    // Classify the failure based on the error message produced by each SetupBackend step.
    if (error_message.find("Unable to load backend") != std::string::npos ||
        error_message.find("Failed to get QNN providers") != std::string::npos) {
      // LoadBackend: GetQnnInterfaceProvider() failed.
      // The QNN backend shared library (e.g., QnnHtp.dll) or one of its dependencies could
      // not be found or loaded, or the required symbol was not present in the library.
      reasons = OrtDeviceEpIncompatibility_MISSING_DEPENDENCY;
    } else if (error_message.find("Unable to find a valid interface") != std::string::npos) {
      // LoadBackend: GetQnnInterfaceProvider() failed.
      // The library was loaded but no interface version compatible with the required QNN
      // API version was found. The installed QNN driver is too old or too new relative to this build of ORT.
      reasons = OrtDeviceEpIncompatibility_DRIVER_INCOMPATIBLE;
    } else if (error_message.find("Failed to initialize backend") != std::string::npos) {
      // InitializeBackend: QNN backendCreate() failed.
      // The backend library loaded successfully but the driver could not be initialised.
      reasons = OrtDeviceEpIncompatibility_DRIVER_INCOMPATIBLE;
    } else if (error_message.find("Failed to create device") != std::string::npos) {
      // CreateDevice: QNN deviceCreate() failed.
      // The hardware device is not present, not accessible, or not supported by the driver.
      reasons = OrtDeviceEpIncompatibility_DEVICE_INCOMPATIBLE;
    }

    return ep_api.DeviceEpIncompatibilityDetails_SetDetails(
        details,
        reasons,
        error_code,
        error_message.c_str());
  }

  // Since this function is always called by temporary QnnEp, so no need to release resource.

  return ep_api.DeviceEpIncompatibilityDetails_SetDetails(
      details,
      OrtDeviceEpIncompatibility_NONE,
      QNN_SUCCESS,
      nullptr);
}

bool QnnEp::GetHtpPowerConfigId(uint32_t& htp_power_config_id) {
  std::lock_guard<std::mutex> lock(config_id_mutex_);
  if (!htp_power_config_id_.has_value()) {
    return false;
  }

  htp_power_config_id = *htp_power_config_id_;
  return true;
}

void QnnEp::CreateHtpPowerConfigId() const {
  std::lock_guard<std::mutex> lock(config_id_mutex_);
  if (htp_power_config_id_.has_value()) {
    return;
  }

  constexpr uint32_t core_id = 0;
  uint32_t htp_power_config_id;

  Ort::Status rt = qnn_backend_manager_->InitializePowerCfgId(device_id_, core_id, htp_power_config_id);

  if (rt.IsOK()) {
    htp_power_config_id_ = htp_power_config_id;
  } else {
    ORT_CXX_LOG(logger_, ORT_LOGGING_LEVEL_ERROR, "Failed to create HTP power config id.");
  }
}

void QnnEp::WarnIfHnrdPathActive() {
  if (hnrd_warning_emitted_) {
    return;
  }
  hnrd_warning_emitted_ = true;
  const uint32_t htp_arch = static_cast<uint32_t>(qnn_backend_manager_->GetHtpArch());
  if (htp_arch == static_cast<uint32_t>(QNN_HTP_DEVICE_ARCH_NONE)) {
    return;
  }
  bool hnrd_enabled = false;
  Ort::Status status = qnn::htp_usr_drv::IsHtpUsrDrvEnabled(
      qnn_backend_manager_->GetBackendLibDir(), htp_arch, hnrd_enabled);
  if (!status.IsOK()) {
    ORT_CXX_LOG(logger_,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("HNRD detection skipped: " + status.GetErrorMessage()).c_str());
    return;
  }
  if (!hnrd_enabled) {
    return;
  }
  ORT_CXX_LOG(logger_,
              ORT_LOGGING_LEVEL_WARNING,
              "QNN EP fell back to HTP user-driver (HNRD) path; "
              "QnnHtpPrepare/Stub/Skel libs missing from backend lib dir.");
}

QnnEp::QnnNodeComputeInfo::QnnNodeComputeInfo(QnnEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* QnnEp::QnnNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                      OrtNodeComputeContext* compute_context,
                                                      void** compute_state) {
  auto* node_compute_info = static_cast<QnnNodeComputeInfo*>(this_ptr);
  QnnEp& ep = node_compute_info->ep;

  if (ep.prepare_only_) {
    ORT_CXX_LOG(ep.logger_, ORT_LOGGING_LEVEL_VERBOSE,
                "Skipping CreateState in enable_htp_prepare_only mode.");
    *compute_state = nullptr;
    return nullptr;
  }

  std::string fused_node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  auto qnn_model_it = ep.qnn_models_.find(fused_node_name);

  // If not found with the fused_node_name, try to find with any available key
  // This handles the case where context models might have different naming
  if (qnn_model_it == ep.qnn_models_.end() && !ep.qnn_models_.empty()) {
    // For context models, there might be only one model, so use the first available
    qnn_model_it = ep.qnn_models_.begin();
  }

  if (qnn_model_it == ep.qnn_models_.end()) {
    std::string message = "Unable to get QnnModel with name " + fused_node_name;
    return ep.ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
  }

  *compute_state = qnn_model_it->second.get();
  return nullptr;
}

OrtStatus* QnnEp::QnnNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr,
                                                  void* compute_state,
                                                  OrtKernelContext* kernel_context) {
  auto* node_compute_info = static_cast<QnnNodeComputeInfo*>(this_ptr);
  QnnEp& ep = node_compute_info->ep;

  if (ep.prepare_only_) {
    return ep.ort_api.CreateStatus(ORT_EP_FAIL,
                                   "QNN EP is in prepare_only mode. Session.Run() is not supported. "
                                   "Load the generated context model for inference.");
  }

  qnn::QnnModel* model = reinterpret_cast<qnn::QnnModel*>(compute_state);
  RETURN_IF_NOT_OK(model->ExecuteGraph(kernel_context, ep.logger_, *ep.io_dispatch_));

  return nullptr;
}

void QnnEp::QnnNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  // The 'state' is a qnn::QnnModel managed by unique_ptr.
  ORT_UNUSED_PARAMETER(this_ptr);
  ORT_UNUSED_PARAMETER(compute_state);
}

Ort::Status QnnEp::ScopedPerSocQnnBackendSetup::Init(size_t per_soc_idx) {
  RETURN_IF_ERROR(ep_.qnn_backend_manager_->SetupDeviceAndContext(ep_.htp_arch_per_soc_[per_soc_idx],
                                                                  ep_.soc_model_per_soc_[per_soc_idx],
                                                                  ep_.enable_htp_extended_udma_mode_,
                                                                  ep_.prepare_only_,
                                                                  ep_.enable_htp_ref_weight_sharing_,
                                                                  ep_.enable_htp_graph_splitting_,
                                                                  ep_.htp_graph_splitting_num_prepare_threads_));

  if (qnn::IsNpuBackend(ep_.qnn_backend_manager_->GetQnnBackendType())) {
    ep_.CreateHtpPowerConfigId();
  }

  return Ort::Status();
}

QnnEp::ScopedPerSocQnnBackendSetup::~ScopedPerSocQnnBackendSetup() {
  // Safe to run even if Init() was never called or failed partway: the power config release is guarded by
  // has_value(), and ReleaseDeviceAndContext() is idempotent (SetupDeviceAndContext() also self-cleans on failure).
  // Hence no separate "initialized" flag is needed here.
  if (qnn::IsNpuBackend(ep_.qnn_backend_manager_->GetQnnBackendType()) && ep_.htp_power_config_id_.has_value()) {
    ep_.qnn_backend_manager_->DropBoostedPowerConfigId(*ep_.htp_power_config_id_);
    ep_.qnn_backend_manager_->DestroyHtpPowerConfigId(*ep_.htp_power_config_id_);
    ep_.htp_power_config_id_.reset();
  }
  ep_.qnn_backend_manager_->ReleaseDeviceAndContext();
}

}  // namespace onnxruntime
