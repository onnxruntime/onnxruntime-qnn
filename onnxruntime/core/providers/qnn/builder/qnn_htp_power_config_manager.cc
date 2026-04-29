// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "core/providers/qnn/builder/qnn_htp_power_config_manager.h"

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

HtpPowerConfigManager::~HtpPowerConfigManager() {}

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
    RETURN_IF(QNN_SUCCESS != status, "SetPowerConfig failed.");

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

void HtpPowerConfigManager::SetRelaxedPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id, DcvsState dcvsState) {
  power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = power_config.dcvsV3Config;
  dcvs_v3.contextId = htp_power_config_client_id;
  dcvs_v3.dcvsEnable = 1;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.sleepLatency = kSleepHighLatency;
  dcvs_v3.setSleepLatency = 1;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setSleepDisable = 0;
  if (dcvsState == DcvsState::DCVS_ENABLE) {
    dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;
  } else {
    dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
  }
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

void HtpPowerConfigManager::SetReleasedPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id, DcvsState dcvsState) {
  power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  QnnHtpPerfInfrastructure_DcvsV3_t& dcvs_v3 = power_config.dcvsV3Config;
  dcvs_v3.contextId = htp_power_config_client_id;
  dcvs_v3.dcvsEnable = 1;
  dcvs_v3.setDcvsEnable = 1;
  dcvs_v3.sleepLatency = kSleepHigherLatency;
  dcvs_v3.setSleepLatency = 1;
  dcvs_v3.sleepDisable = 0;
  dcvs_v3.setSleepDisable = 0;
  if (dcvsState == DcvsState::DCVS_ENABLE) {
    dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;
  } else {
    dcvs_v3.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
  }
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
  std::lock_guard<std::mutex> lk(state_mutex_);
  const Ort::Logger& logger = OrtLoggingManager::GetDefaultLogger();
  if (timer_ == nullptr) {
    std::unique_ptr<Timer> temp(new Timer());
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
  std::unique_ptr<Timer> local_timer;
  std::unique_ptr<TimerCallbackArg> local_callback_arg;
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    if (timer_ != nullptr) {
      timer_resource_.timer_active_ = false;
      graph_state_ = GraphState::NONE;
      timer_resource_.caller_busy_ = false;
      // Move ownership out while holding the lock: timer_ becomes nullptr
      // atomically, so CreateTimerThread sees null and can safely create
      // a new timer. We hold exclusive ownership in the locals.
      local_timer = std::move(timer_);
      local_callback_arg = std::move(timer_callback_arg_);
    }
  }
  // Deinitialize outside the lock to avoid deadlock: an in-flight
  // TimerCallback calls SetState() which acquires state_mutex_.
  if (local_timer != nullptr) {
    local_timer->DeInitialize();
    local_callback_arg.reset();
    local_timer.reset();
  }
}

Ort::Status HtpPowerConfigManager::SetSustainedPerformance(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger) {
  std::lock_guard<std::mutex> lk(perf_mutex_);
  Ort::Status status = Ort::Status();

  std::chrono::microseconds sustainedDurationUs(timer_resource_.sustained_timer_duration_);

  switch (state) {
    case GraphState::RUN_DONE:
      if (IsTimerThreadRunning()) {
        timer_->AbortTimer();
      }
      RETURN_IF_NOT(timer_->Launch(sustainedDurationUs), "Not able to launch timer thread.");
      graph_state_ = GraphState::NONE;
      timer_resource_.caller_busy_ = false;
      break;
    case GraphState::RUN_START:
      if (IsTimerThreadRunning()) {
        timer_->AbortTimer();
      } else {
        status = SetHtpPowerConfigs(config, logger);
      }
      graph_state_ = GraphState::NONE;
      timer_resource_.caller_busy_ = true;
      break;
    case GraphState::INIT_DONE: {
      QnnHtpPerfInfrastructure_PowerConfig_t init_done_htp_performance_cfg{};
      SetRelaxedPerfPowerConfig(init_done_htp_performance_cfg, config.htp_power_config_client_id, DcvsState::DCVS_DEFAULT);
      status = SetHtpPowerCustomConfigs(config.htp_power_config_client_id, init_done_htp_performance_cfg, config.rpc_polling_time, config.rpc_control_latency, logger);
      graph_state_ = GraphState::NONE;
      timer_resource_.caller_busy_ = false;
      break;
    }
    case GraphState::INIT_START:
      if (IsTimerThreadRunning()) {
        timer_->AbortTimer();
      } else {
        status = SetHtpPowerConfigs(config, logger);
      }
      graph_state_ = GraphState::NONE;
      timer_resource_.caller_busy_ = true;
      break;
    case GraphState::TIMEOUT: {
      if (!timer_resource_.caller_busy_) {
        QnnHtpPerfInfrastructure_PowerConfig_t timeout_htp_performance_cfg{};
        SetRelaxedPerfPowerConfig(timeout_htp_performance_cfg, config.htp_power_config_client_id, DcvsState::DCVS_DEFAULT);
        status = SetHtpPowerCustomConfigs(config.htp_power_config_client_id, timeout_htp_performance_cfg, config.rpc_polling_time, config.rpc_control_latency, logger);
        graph_state_ = GraphState::NONE;
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
          SetRelaxedPerfPowerConfig(relaxed_htp_performance_cfg, config.htp_power_config_client_id, DcvsState::DCVS_DEFAULT);
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
          SetReleasedPerfPowerConfig(released_htp_performance_cfg, config.htp_power_config_client_id, DcvsState::DCVS_DEFAULT);
          status = SetHtpPowerCustomConfigs(config.htp_power_config_client_id, released_htp_performance_cfg, config.rpc_polling_time, config.rpc_control_latency, logger);
          break;
        }
        default:
          ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Invalid performance mode");
          break;
      }
      graph_state_ = GraphState::NONE;
      break;
    case GraphState::RUN_START:
    case GraphState::INIT_START:
      status = SetHtpPowerConfigs(config, logger);
      graph_state_ = GraphState::NONE;
      break;
    default:
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "Invalid graph state");
      break;
  }
  return status;
}

Ort::Status HtpPowerConfigManager::SetState(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger) {
  std::lock_guard<std::mutex> lk(state_mutex_);
  if (state != graph_state_) {
    graph_state_ = state;
  } else {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "State is the same as current. Ignoring request.");
    return Ort::Status();
  }
  if (config.perf_mode == qnn::HtpPerformanceMode::kHtpSustainedHighPerformance || config.perf_mode == qnn::HtpPerformanceMode::kHtpBurst) {
    RETURN_IF(timer_resource_.timer_active_ == false, "Timer is not active. Cannot set state.");
    RETURN_IF(timer_ == nullptr, "timer is not started");
    return SetSustainedPerformance(state, config, logger);
  } else if (config.perf_mode == qnn::HtpPerformanceMode::kHtpDefault) {
    if (timer_ && timer_->TimerInUse()) {
      timer_->AbortTimer();
    }
    return Ort::Status();
  } else {
    if (timer_ && timer_->TimerInUse()) {
      timer_->AbortTimer();
    }
    return SetPerformance(state, config, logger);
  }
}

void HtpPowerConfigManager::TimerCallback(void* user_data) {
  TimerCallbackArg* args = static_cast<TimerCallbackArg*>(user_data);
  if (args == nullptr) {
    return;
  }
  HtpPowerConfigManager* instance = args->instance_;
  if (instance->timer_resource_.timer_active_) {
    const Ort::Logger& logger = OrtLoggingManager::GetDefaultLogger();
    auto rt = instance->SetState(GraphState::TIMEOUT, {args->power_config_id_, qnn::HtpPerformanceMode::kHtpSustainedHighPerformance, 0, 0}, logger);
    if (!rt.IsOK()) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "State update failed");
    }
  }
}

bool HtpPowerConfigManager::IsTimerThreadRunning() {
  std::chrono::microseconds remainUs = std::chrono::microseconds::zero();
  uint64_t remaining_duration = 0;
  if (timer_ && timer_->TimerInUse() && timer_->RemainingDuration(remainUs)) {
    remaining_duration = static_cast<uint64_t>(remainUs.count());
    return remaining_duration > 0 && remaining_duration < timer_resource_.sustained_timer_duration_;
  }
  return false;
}

Ort::Status HtpPowerConfigManager::SetHtpPowerConfigs(const HtpPerfConfig_t& config, const Ort::Logger& logger) {
  RETURN_IF(qnn_interface_ == nullptr, "QNN interface is not initialized");
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
  RETURN_IF(qnn_interface_ == nullptr, "QNN interface is not initialized");
  RETURN_IF_ERROR(AddRpcPollingTime(rpc_polling_time, logger));
  RETURN_IF_ERROR(AddRpcControlLatency(rpc_control_latency, logger));
  RETURN_IF_ERROR(AddHtpPerformanceConfig(power_config));
  RETURN_IF_ERROR(SetPowerConfig(htp_power_config_client_id, *qnn_interface_, logger));

  return Ort::Status();
}

}  // namespace power
}  // namespace qnn
}  // namespace onnxruntime
