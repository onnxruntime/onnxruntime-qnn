// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/builder/qnn_htp_power_config_manager.h"

#include <algorithm>
#include <vector>

#include <QnnInterface.h>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {
namespace power {

HtpPowerConfigManager::HtpPowerConfigManager() {
  constexpr int kMaxNumConfigs = 3;
  power_configs_.reserve(kMaxNumConfigs);
}

HtpPowerConfigManager::~HtpPowerConfigManager() {
  // Defensive: join and tear down the timer here so correctness does not depend
  // on an external caller (QnnBackendManager::ReleaseResources) invoking
  // ReleaseTimerThread() first. Members destruct in reverse declaration order,
  // which would destroy timer_callback_arg_ before timer_; a callback firing
  // during ~Timer's join would then dereference freed args. ReleaseTimerThread()
  // joins the timer thread before returning, so no callback is in flight past
  // this point. Safe to call twice: the second call finds timer_ already null.
  ReleaseTimerThread();
}

Ort::Status HtpPowerConfigManager::AddRpcPollingTime(uint32_t rpc_polling_time, const Ort::Logger& logger) {
  RETURN_IF(rpc_polling_time > kMaxRpcPolling,
            ("Cannot set RPC polling time to " + std::to_string(rpc_polling_time) +
             ". Max allowable RPC polling time is: " + std::to_string(kMaxRpcPolling))
                .c_str());

  RETURN_IF(rpc_polling_time_set_, "There is already a pending RPC polling time config");

  if (rpc_polling_time == last_set_rpc_polling_time_) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Requested rpc polling time is the same as last set (" + std::to_string(last_set_rpc_polling_time_) + "). Ignoring request").c_str());
  } else {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Updating rpc polling time to: " + std::to_string(rpc_polling_time) + "us.").c_str());
    auto& rpc_polling_time_cfg = power_configs_.emplace_back();
    rpc_polling_time_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpc_polling_time_cfg.rpcPollingTimeConfig = rpc_polling_time;

    last_set_rpc_polling_time_ = rpc_polling_time;
    rpc_polling_time_set_ = true;
  }
  return Ort::Status();
}

Ort::Status HtpPowerConfigManager::AddRpcControlLatency(uint32_t rpc_control_latency, const Ort::Logger& logger) {
  RETURN_IF(rpc_control_latency_set_, "There is already a pending RPC control latency config");
  if (rpc_control_latency == last_set_rpc_control_latency_) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Requested rpc control latency is the same as last set (" +
                 std::to_string(last_set_rpc_control_latency_) + "). Ignoring request")
                    .c_str());
  } else {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Updating rpc control latency to: " + std::to_string(rpc_control_latency) + "us.").c_str());
    auto& rpc_control_latency_cfg = power_configs_.emplace_back();
    rpc_control_latency_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc_control_latency_cfg.rpcControlLatencyConfig = rpc_control_latency;

    last_set_rpc_control_latency_ = rpc_control_latency;
    rpc_control_latency_set_ = true;
  }

  return Ort::Status();
}

static std::string_view PerformanceModeToString(HtpPerformanceMode htp_performance_mode) {
  constexpr std::array<std::pair<HtpPerformanceMode, std::string_view>, 10> perf_string_map = {{{HtpPerformanceMode::kHtpDefault, "default"},
                                                                                                {HtpPerformanceMode::kHtpSustainedHighPerformance, "sustained_high_performance"},
                                                                                                {HtpPerformanceMode::kHtpBurst, "burst"},
                                                                                                {HtpPerformanceMode::kHtpHighPerformance, "high_performance"},
                                                                                                {HtpPerformanceMode::kHtpPowerSaver, "power_saver"},
                                                                                                {HtpPerformanceMode::kHtpLowPowerSaver, "low_power_saver"},
                                                                                                {HtpPerformanceMode::kHtpHighPowerSaver, "high_power_saver"},
                                                                                                {HtpPerformanceMode::kHtpLowBalanced, "low_balanced"},
                                                                                                {HtpPerformanceMode::kHtpBalanced, "balanced"},
                                                                                                {HtpPerformanceMode::kHtpExtremePowerSaver, "extreme_power_saver"}}};

  auto it = std::find_if(perf_string_map.begin(), perf_string_map.end(),
                         [htp_performance_mode](const auto& mapping) {
                           return mapping.first == htp_performance_mode;
                         });

  if (it != perf_string_map.end()) {
    return it->second;
  }

  return "UNKNOWN";
}

Ort::Status HtpPowerConfigManager::AddHtpPerformanceConfig(QnnHtpPerfInfrastructure_PowerConfig_t htp_performance_cfg) {
  power_configs_.emplace_back(std::move(htp_performance_cfg));
  htp_performance_mode_set_ = true;
  return Ort::Status();
}

Ort::Status HtpPowerConfigManager::AddHtpPerformanceMode(HtpPerformanceMode htp_performance_mode,
                                                         uint32_t htp_power_config_client_id,
                                                         const Ort::Logger& logger) {
  RETURN_IF(htp_performance_mode_set_, "There is already a pending HTP performance mode config");
  ORT_CXX_LOG(logger,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("Updating htp performance mode to: " +
               std::string(PerformanceModeToString(htp_performance_mode)) + ".")
                  .c_str());

  QnnHtpPerfInfrastructure_PowerConfig_t htp_performance_cfg{};
  RETURN_IF_ERROR(SetHtpPerformancePowerConfig(htp_performance_cfg,
                                               htp_power_config_client_id,
                                               htp_performance_mode));

  power_configs_.emplace_back(std::move(htp_performance_cfg));
  htp_performance_mode_set_ = true;

  return Ort::Status();
}

Ort::Status HtpPowerConfigManager::SetPowerConfig(uint32_t htp_power_config_client_id,
                                                  const QNN_INTERFACE_VER_TYPE& qnn_interface,
                                                  const Ort::Logger& logger) {
  if (!power_configs_.empty()) {
    QnnDevice_Infrastructure_t qnn_device_infra = nullptr;
    auto status = qnn_interface.deviceGetInfrastructure(&qnn_device_infra);
    RETURN_IF(QNN_SUCCESS != status, "backendGetPerfInfrastructure failed.");

    auto* htp_infra = static_cast<QnnHtpDevice_Infrastructure_t*>(qnn_device_infra);
    RETURN_IF(QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF != htp_infra->infraType,
              ("HTP infra type = " + std::to_string(htp_infra->infraType) + ", which is not perf infra type.").c_str());
    QnnHtpDevice_PerfInfrastructure_t& htp_perf_infra = htp_infra->perfInfra;

    std::vector<const QnnHtpPerfInfrastructure_PowerConfig_t*> perf_power_configs_ptr;

    for (const auto& power_config : power_configs_) {
      perf_power_configs_ptr.push_back(&power_config);
    }
    perf_power_configs_ptr.push_back(nullptr);

    status = htp_perf_infra.setPowerConfig(htp_power_config_client_id, perf_power_configs_ptr.data());
    if (QNN_SUCCESS != status) {
      // Best-effort: on targets without a functioning on-chip HTP (e.g. HTP arch
      // could not be resolved during SetupBackend) the DCVS perf-infrastructure
      // vote is rejected. This is not a correctness gate for graph execution, so
      // warn and continue rather than failing the run with an exception. The
      // pending state below is still reset so the next request starts clean.
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING, "SetPowerConfig failed; continuing without applying HTP power config.");
    }

    rpc_polling_time_set_ = false;
    rpc_control_latency_set_ = false;
    htp_performance_mode_set_ = false;
    power_configs_.clear();
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "SetPowerConfig called but no configs to be set.");
  }

  return Ort::Status();
}

Ort::Status HtpPowerConfigManager::SetHtpPerformancePowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config,
                                                                uint32_t htp_power_config_client_id,
                                                                const HtpPerformanceMode& htp_performance_mode) {
  power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = power_config.dcvsV3Config;
  dcvs_v3.contextId = htp_power_config_client_id;
  dcvs_v3.setSleepDisable = 0;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  // choose performance mode
  switch (htp_performance_mode) {
    case HtpPerformanceMode::kHtpBurst:
    case HtpPerformanceMode::kHtpSustainedHighPerformance:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMinLatency;
      dcvs_v3.dcvsEnable = kDcvsDisable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      break;
    case HtpPerformanceMode::kHtpHighPerformance:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepLowLatency;
      dcvs_v3.dcvsEnable = kDcvsDisable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
      break;
    case HtpPerformanceMode::kHtpBalanced:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      break;
    case HtpPerformanceMode::kHtpLowBalanced:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
      break;
    case HtpPerformanceMode::kHtpHighPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      break;
    case HtpPerformanceMode::kHtpPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
      break;
    case HtpPerformanceMode::kHtpLowPowerSaver:
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
      break;
    case HtpPerformanceMode::kHtpExtremePowerSaver:
      dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
      dcvs_v3.setSleepLatency = 1;  // true
      dcvs_v3.sleepLatency = kSleepMediumLatency;
      dcvs_v3.dcvsEnable = kDcvsEnable;
      dcvs_v3.setBusParams = 1;
      dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.setCoreParams = 1;
      dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_CORNER_DISABLE;
      dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
      break;
    default:
      ORT_CXX_API_THROW(("Invalid performance profile " +
                         std::to_string(static_cast<uint8_t>(htp_performance_mode)))
                            .c_str(),
                        ORT_EP_FAIL);
      break;
  }

  return Ort::Status();
}

void HtpPowerConfigManager::SetRelaxedPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id) {
  power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = power_config.dcvsV3Config;
  dcvs_v3.contextId = htp_power_config_client_id;
  dcvs_v3.dcvsEnable = 1;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.sleepLatency = kSleepHighLatency;
  dcvs_v3.setSleepLatency = 1;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setSleepDisable = 0;
  dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
  dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
  dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
  dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
  dcvs_v3.setBusParams = 1;
  dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
  dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
  dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
  dcvs_v3.setCoreParams = 1;
}

void HtpPowerConfigManager::SetExtremeLowPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id) {
  power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = power_config.dcvsV3Config;
  dcvs_v3.contextId = htp_power_config_client_id;
  dcvs_v3.dcvsEnable = 1;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.sleepLatency = kSleepHigherLatency;
  dcvs_v3.setSleepLatency = 1;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setSleepDisable = 0;
  dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
  dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_CORNER_DISABLE;
  dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_CORNER_DISABLE;
  dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
  dcvs_v3.setBusParams = 1;
  dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_CORNER_DISABLE;
  dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_CORNER_DISABLE;
  dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_CORNER_DISABLE;
  dcvs_v3.setCoreParams = 1;
}

void HtpPowerConfigManager::SetReleasedPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id) {
  power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = power_config.dcvsV3Config;
  dcvs_v3.contextId = htp_power_config_client_id;
  dcvs_v3.dcvsEnable = 1;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.sleepLatency = kSleepHigherLatency;
  dcvs_v3.setSleepLatency = 1;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setSleepDisable = 0;
  dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
  dcvs_v3.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  dcvs_v3.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  dcvs_v3.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  dcvs_v3.setBusParams = 1;
  dcvs_v3.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  dcvs_v3.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  dcvs_v3.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
  dcvs_v3.setCoreParams = 1;
}

void HtpPowerConfigManager::CreateTimerThread(uint32_t htp_power_config_client_id) {
  // timer_ / timer_callback_arg_ are guarded by perf_mutex_ (see header).
  std::lock_guard<std::mutex> lk(perf_mutex_);
  const Ort::Logger& logger = OrtLoggingManager::GetDefaultLogger();
  if (timer_ == nullptr) {
    std::shared_ptr<Timer> temp = std::make_shared<Timer>();
    if (temp != nullptr) {
      timer_ = std::move(temp);
      timer_callback_arg_ = std::make_unique<TimerCallbackArg>(htp_power_config_client_id, this);
      if (!timer_->Initialize(TimerCallback, timer_callback_arg_.get())) {
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Failed to create timer to set performance");
        timer_callback_arg_.reset();
        timer_.reset();
      } else {
        timer_resource_.timer_active_ = true;
      }
    } else {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Failed: Timer is nullptr");
    }
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Timer already created");
  }
}

void HtpPowerConfigManager::ReleaseTimerThread() {
  std::shared_ptr<Timer> local_timer;
  std::unique_ptr<TimerCallbackArg> local_callback_arg;
  // Reset run/graph state under state_mutex_. timer_active_ is atomic; clearing
  // it first means any timer callback that fires from here on early-returns.
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    timer_resource_.timer_active_ = false;
    graph_state_ = GraphState::NONE;
    timer_resource_.inflight_run_count_ = 0;
  }
  // Move the timer/callback-arg ownership out under perf_mutex_ (which now guards
  // the timer lifecycle) and clear the boosted-id set. timer_ becomes nullptr so
  // CreateTimerThread can safely recreate. We hold the only remaining references
  // in the locals. Follows the state_mutex_-before-perf_mutex_ order (state scope
  // already closed).
  {
    std::lock_guard<std::mutex> lk(perf_mutex_);
    local_timer = std::move(timer_);
    local_callback_arg = std::move(timer_callback_arg_);
    boosted_config_ids_.clear();
  }
  // Deinitialize outside the lock to avoid deadlock: an in-flight
  // TimerCallback calls SetState() which acquires state_mutex_/perf_mutex_.
  // Note: DeInitialize()->join() ensures any in-flight callback completes
  // before the timer and callback_arg are destroyed, so no additional
  // synchronization is needed to protect callback access to these objects.
  if (local_timer != nullptr) {
    local_timer->DeInitialize();
    local_callback_arg.reset();
    local_timer.reset();
  }
}

Ort::Status HtpPowerConfigManager::SetSustainedPerformance(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger) {
  // The TIMEOUT case runs on the timer thread (via TimerCallback -> SetState).
  // Other cases (RUN_*/INIT_*) run on caller threads that may hold perf_mutex_
  // across a blocking AbortTimer(). If the timer fires in the tiny window between
  // a caller's IsTimerThreadRunning() check and its AbortTimer() call, the caller
  // waits for the timer to reach IDLE while holding perf_mutex_, and the timer's
  // callback would block acquiring perf_mutex_ here -> deadlock. To break the
  // cycle the TIMEOUT path uses try_lock and bails on contention: contention means
  // a run is either starting (we must not relax) or finishing (it will re-arm the
  // timer), so skipping this relax is always safe.
  std::unique_lock<std::mutex> lk(perf_mutex_, std::defer_lock);
  if (state == GraphState::TIMEOUT) {
    if (!lk.try_lock()) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                  "Release timer contended with a run transition; skipping relax (will re-arm).");
      return Ort::Status();
    }
  } else {
    lk.lock();
  }

  Ort::Status status = Ort::Status();

  std::chrono::microseconds sustainedDurationUs(timer_resource_.sustained_timer_duration_);

  switch (state) {
    case GraphState::RUN_DONE:
      // This id holds a burst/sustained vote; the release timer owns relaxing it.
      RegisterBoostedId(config.htp_power_config_client_id);
      if (IsTimerThreadRunning()) {
        timer_->AbortTimer();
      }
      // (Re)arm the release timer. Because RUN_DONE aborts any pending timer and
      // relaunches, the release is always measured from the most recent done.
      // Whether the timer is actually allowed to relax performance when it fires
      // is gated on the in-flight run count in the TIMEOUT case below, so a
      // still-running concurrent graph is never dropped to SVS mid-computation.
      //
      // Launch() can fail if the timer is momentarily in the CALLING state (a
      // just-fired timeout whose callback has not yet returned to IDLE): in that
      // narrow race IsTimerThreadRunning() reports false (remaining duration is 0
      // while CALLING), so the abort above is skipped and Launch() sees a non-IDLE
      // timer. This is a best-effort power-down optimization, not a correctness
      // gate: the graph has already executed. Log and continue rather than failing
      // the run; the timer is re-armed on the next RUN_DONE.
      if (!timer_->Launch(sustainedDurationUs)) {
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                    "Release timer busy (concurrent timeout); power-down will be re-armed on next run.");
      }
      break;
    case GraphState::RUN_START:
      // This id holds a burst/sustained vote; the release timer owns relaxing it.
      RegisterBoostedId(config.htp_power_config_client_id);
      if (IsTimerThreadRunning()) {
        timer_->AbortTimer();
      } else {
        status = SetHtpPowerConfigs(config, logger);
      }
      break;
    case GraphState::INIT_DONE: {
      QnnHtpPerfInfrastructure_PowerConfig_t init_done_htp_performance_cfg{};
      SetRelaxedPerfPowerConfig(init_done_htp_performance_cfg, config.htp_power_config_client_id);
      status = SetHtpPowerCustomConfigs(config.htp_power_config_client_id, init_done_htp_performance_cfg, config.rpc_polling_time, config.rpc_control_latency, logger);
      break;
    }
    case GraphState::INIT_START:
      if (IsTimerThreadRunning()) {
        timer_->AbortTimer();
      } else {
        status = SetHtpPowerConfigs(config, logger);
      }
      break;
    case GraphState::TIMEOUT: {
      // Only relax to sustained/released performance once the last concurrent
      // run across all graphs sharing this manager has finished. If any run is
      // still in flight the drop is skipped, leaving performance boosted.
      if (timer_resource_.inflight_run_count_.load() == 0) {
        // Relax every id that currently holds a burst/sustained vote — not just
        // the timer's construction-time id. Under a shared manager each session
        // owns a distinct non-zero power-config id, and each is an independent
        // DCVS voter (QnnHtpPerfInfrastructure.h: setPowerConfig is per client
        // id). Relaxing only one would leave the other sessions' boost votes
        // standing and keep the HTP elevated while idle. Ids that moved to a
        // low-power/default vote were removed from the set, so their intentional
        // low state is preserved. Best-effort: log and continue so one id's
        // failure does not skip the rest.
        //
        // Invariant: the set is non-empty whenever the timer fires. The timer is
        // only armed by a burst/sustained RUN_DONE, which calls RegisterBoostedId
        // first; every path that calls RemoveBoostedId also aborts the timer
        // (AbortTimer blocks until the callback is idle). So there is no need to
        // fall back to config.htp_power_config_client_id here.
        for (uint32_t id : boosted_config_ids_) {
          QnnHtpPerfInfrastructure_PowerConfig_t timeout_htp_performance_cfg{};
          SetRelaxedPerfPowerConfig(timeout_htp_performance_cfg, id);
          Ort::Status relax_status = SetHtpPowerCustomConfigs(id, timeout_htp_performance_cfg,
                                                              config.rpc_polling_time,
                                                              config.rpc_control_latency, logger);
          if (!relax_status.IsOK()) {
            ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                        "Failed to relax HTP perf for a power-config id on timeout.");
          }
        }
        // All boosted ids have been relaxed; the next RUN_START re-registers.
        boosted_config_ids_.clear();
      }
      break;
    }
    default:
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Invalid graph state");
      break;
  }
  return status;
}

Ort::Status HtpPowerConfigManager::SetPerformance(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger) {
  std::lock_guard<std::mutex> lk(perf_mutex_);
  Ort::Status status = Ort::Status();
  switch (state) {
    case GraphState::RUN_DONE:
    case GraphState::INIT_DONE:
      switch (config.perf_mode) {
        case qnn::HtpPerformanceMode::kHtpLowBalanced:
        case qnn::HtpPerformanceMode::kHtpBalanced:
        case qnn::HtpPerformanceMode::kHtpHighPerformance: {
          QnnHtpPerfInfrastructure_PowerConfig_t relaxed_htp_performance_cfg{};
          SetRelaxedPerfPowerConfig(relaxed_htp_performance_cfg, config.htp_power_config_client_id);
          status = SetHtpPowerCustomConfigs(config.htp_power_config_client_id, relaxed_htp_performance_cfg, config.rpc_polling_time, config.rpc_control_latency, logger);
          break;
        }
        case qnn::HtpPerformanceMode::kHtpExtremePowerSaver: {
          QnnHtpPerfInfrastructure_PowerConfig_t extreme_power_saver_htp_performance_cfg{};
          SetExtremeLowPerfPowerConfig(extreme_power_saver_htp_performance_cfg, config.htp_power_config_client_id);
          status = SetHtpPowerCustomConfigs(config.htp_power_config_client_id, extreme_power_saver_htp_performance_cfg, config.rpc_polling_time, config.rpc_control_latency, logger);
          break;
        }
        case qnn::HtpPerformanceMode::kHtpLowPowerSaver:
        case qnn::HtpPerformanceMode::kHtpHighPowerSaver:
        case qnn::HtpPerformanceMode::kHtpPowerSaver: {
          QnnHtpPerfInfrastructure_PowerConfig_t released_htp_performance_cfg{};
          SetReleasedPerfPowerConfig(released_htp_performance_cfg, config.htp_power_config_client_id);
          status = SetHtpPowerCustomConfigs(config.htp_power_config_client_id, released_htp_performance_cfg, config.rpc_polling_time, config.rpc_control_latency, logger);
          break;
        }
        default:
          ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Invalid performance mode");
          break;
      }
      break;
    case GraphState::RUN_START:
    case GraphState::INIT_START:
      status = SetHtpPowerConfigs(config, logger);
      break;
    default:
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Invalid graph state");
      break;
  }
  return status;
}

Ort::Status HtpPowerConfigManager::SetState(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger) {
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    graph_state_ = state;

    if (config.perf_mode == qnn::HtpPerformanceMode::kHtpSustainedHighPerformance || config.perf_mode == qnn::HtpPerformanceMode::kHtpBurst) {
      // timer_active_ (atomic) is only set true after timer_ is successfully
      // created and cleared before timer_ is released, so it implies a live
      // timer. We deliberately do NOT read timer_ here: timer_ is guarded by
      // perf_mutex_ and reading it under state_mutex_ would both race and violate
      // the state-before-perf lock order.
      RETURN_IF(timer_resource_.timer_active_ == false, "Timer is not active. Cannot apply sustained/burst performance config.");
    }

    // Track the number of runs in flight across ALL graphs that share this
    // manager. A single session can drive several QnnModels (each with its own
    // graph_exec_mutex_) through this one manager concurrently. We must key the
    // count on the graph state rather than on the perf mode, because a run's
    // start and done may use different perf modes (e.g. burst pre-run and
    // power_saver post-run), which would otherwise leak the count.
    //
    // Note: unlike the previous single busy flag, we intentionally do NOT
    // early-return when state == graph_state_. Two different graphs can both be
    // in RUN_START at once; deduping would drop the second graph's start (and,
    // symmetrically, prevent the last concurrent RUN_DONE from arming the
    // release timer). For a single graph the states strictly alternate, so this
    // has no effect there. The counter is updated after the sustained/burst
    // validity checks above so a rejected start does not leak a count.
    switch (state) {
      case GraphState::RUN_START:
      case GraphState::INIT_START:
        timer_resource_.inflight_run_count_.fetch_add(1);
        break;
      case GraphState::RUN_DONE:
      case GraphState::INIT_DONE:
        // Clamp at 0 to stay robust against any unpaired done transition.
        if (timer_resource_.inflight_run_count_.load() > 0) {
          timer_resource_.inflight_run_count_.fetch_sub(1);
        }
        break;
      default:
        break;
    }
  }

  // Dispatch to performance setters outside state_mutex_ to avoid deadlock:
  // AbortTimer() blocks until the timer thread is idle, but the timer thread
  // (inside TimerCallback) calls SetState() which acquires state_mutex_.
  // Holding state_mutex_ across AbortTimer() would therefore deadlock.
  // The same pattern is already applied in ReleaseTimerThread().
  Ort::Status status;
  if (config.perf_mode == qnn::HtpPerformanceMode::kHtpSustainedHighPerformance || config.perf_mode == qnn::HtpPerformanceMode::kHtpBurst) {
    status = SetSustainedPerformance(state, config, logger);
  } else if (config.perf_mode == qnn::HtpPerformanceMode::kHtpDefault) {
    // No longer a boosted vote: abort any pending timer and drop this id so the
    // release timer will not relax it. Snapshot timer_ and remove the id under
    // perf_mutex_, then abort on the snapshot outside the lock (AbortTimer blocks
    // on the timer thread, which itself needs perf_mutex_ via the callback).
    AbortActiveTimerAndDropBoostedId(config.htp_power_config_client_id);
    status = Ort::Status();
  } else {
    // This id is being set to an intentional low-power/non-boosted vote. Same
    // snapshot-then-abort discipline; SetPerformance re-acquires perf_mutex_ after
    // our critical section has closed.
    AbortActiveTimerAndDropBoostedId(config.htp_power_config_client_id);
    status = SetPerformance(state, config, logger);
  }

  return status;
}

void HtpPowerConfigManager::TimerCallback(void* user_data) {
  TimerCallbackArg* args = static_cast<TimerCallbackArg*>(user_data);
  if (args == nullptr) {
    return;
  }
  HtpPowerConfigManager* instance = args->instance_;
  if (instance == nullptr) {
    return;
  }
  if (instance->timer_resource_.timer_active_) {
    const Ort::Logger& logger = OrtLoggingManager::GetDefaultLogger();
    auto rt = instance->SetState(GraphState::TIMEOUT, {args->power_config_id_, qnn::HtpPerformanceMode::kHtpSustainedHighPerformance, 0, 0}, logger);
    if (!rt.IsOK()) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "State update failed");
    }
  }
}

bool HtpPowerConfigManager::IsTimerThreadRunning() {
  // Caller holds perf_mutex_ (SetSustainedPerformance), which guards timer_.
  std::chrono::microseconds remainUs = std::chrono::microseconds::zero();
  uint64_t remaining_duration = 0;
  if (timer_ && timer_->TimerInUse() && timer_->RemainingDuration(remainUs)) {
    remaining_duration = static_cast<uint64_t>(remainUs.count());
    return remaining_duration > 0 && remaining_duration < timer_resource_.sustained_timer_duration_;
  }
  return false;
}

void HtpPowerConfigManager::AbortActiveTimerAndDropBoostedId(uint32_t htp_power_config_client_id) {
  // Snapshot the timer and drop the id under perf_mutex_, then abort on the
  // snapshot OUTSIDE the lock. AbortTimer() blocks until the timer thread reaches
  // IDLE, and that thread (TimerCallback -> SetState -> SetSustainedPerformance)
  // needs perf_mutex_; holding it across AbortTimer() would deadlock. The
  // shared_ptr snapshot keeps the Timer alive even if ReleaseTimerThread resets
  // timer_ concurrently, and Timer::AbortTimer() early-returns if the timer was
  // already deinitialized, so the abort can never hang.
  std::shared_ptr<Timer> timer_snapshot;
  {
    std::lock_guard<std::mutex> lk(perf_mutex_);
    timer_snapshot = timer_;
    RemoveBoostedId(htp_power_config_client_id);
  }
  if (timer_snapshot && timer_snapshot->TimerInUse()) {
    timer_snapshot->AbortTimer();
  }
}

void HtpPowerConfigManager::RegisterBoostedId(uint32_t htp_power_config_client_id) {
  // Caller holds perf_mutex_.
  if (std::find(boosted_config_ids_.begin(), boosted_config_ids_.end(),
                htp_power_config_client_id) == boosted_config_ids_.end()) {
    boosted_config_ids_.push_back(htp_power_config_client_id);
  }
}

void HtpPowerConfigManager::RemoveBoostedId(uint32_t htp_power_config_client_id) {
  // Caller holds perf_mutex_.
  boosted_config_ids_.erase(
      std::remove(boosted_config_ids_.begin(), boosted_config_ids_.end(), htp_power_config_client_id),
      boosted_config_ids_.end());
}

void HtpPowerConfigManager::DropBoostedPowerConfigId(uint32_t htp_power_config_client_id) {
  std::lock_guard<std::mutex> lk(perf_mutex_);
  RemoveBoostedId(htp_power_config_client_id);
}

Ort::Status HtpPowerConfigManager::SetHtpPowerConfigs(const HtpPerfConfig_t& config, const Ort::Logger& logger) {
  RETURN_IF(qnn_interface_ == nullptr, "QNN interface is not initialized. Call Init() first.");
  RETURN_IF_ERROR(AddRpcPollingTime(config.rpc_polling_time, logger));
  RETURN_IF_ERROR(AddRpcControlLatency(config.rpc_control_latency, logger));
  RETURN_IF_ERROR(AddHtpPerformanceMode(config.perf_mode,
                                        config.htp_power_config_client_id, logger));
  RETURN_IF_ERROR(SetPowerConfig(config.htp_power_config_client_id,
                                 *qnn_interface_, logger));

  return Ort::Status();
}

Ort::Status HtpPowerConfigManager::SetHtpPowerCustomConfigs(uint32_t htp_power_config_client_id,
                                                            const QnnHtpPerfInfrastructure_PowerConfig_t& power_config,
                                                            uint32_t rpc_polling_time,
                                                            uint32_t rpc_control_latency,
                                                            const Ort::Logger& logger) {
  RETURN_IF(qnn_interface_ == nullptr, "QNN interface is not initialized. Call Init() first.");
  RETURN_IF_ERROR(AddRpcPollingTime(rpc_polling_time, logger));
  RETURN_IF_ERROR(AddRpcControlLatency(rpc_control_latency, logger));
  RETURN_IF_ERROR(AddHtpPerformanceConfig(power_config));
  RETURN_IF_ERROR(SetPowerConfig(htp_power_config_client_id, *qnn_interface_, logger));

  return Ort::Status();
}

}  // namespace power
}  // namespace qnn
}  // namespace onnxruntime
