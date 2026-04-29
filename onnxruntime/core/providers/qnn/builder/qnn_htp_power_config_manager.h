// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <vector>

#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <QnnInterface.h>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/timer.h"

namespace onnxruntime {
namespace qnn {
namespace power {

// Manages staging of any new power configurations and
// updates power configurations for the HTP backend
class HtpPowerConfigManager {
 public:
  HtpPowerConfigManager();
  ~HtpPowerConfigManager();

  // Stages a new rpc polling time for next power config update
  // If the value is the same as the last previously set, then
  // there will be no new rpc polling time staged
  Ort::Status AddRpcPollingTime(uint32_t rpc_polling_time, const Ort::Logger& logger);

  // Stages a new rpc control latency for next power config update
  // If the value is the same as the last previously set, then
  // there will be no new rpc control latency staged
  Ort::Status AddRpcControlLatency(uint32_t rpc_control_latency, const Ort::Logger& logger);

  // Stages a new performance mode for next power config update
  Ort::Status AddHtpPerformanceMode(HtpPerformanceMode htp_performance_mode,
                                    uint32_t htp_power_config_client_id,
                                    const Ort::Logger& logger);

  // Stages a new HTP power configuration for next power config update
  // performance mode is set to default after setting the power config
  Ort::Status AddHtpPerformanceConfig(QnnHtpPerfInfrastructure_PowerConfig_t);

  // Takes all configs staged for update and attempts to update
  // the HTP power configurations. If there is nothing staged,
  // then no attempt will be made.
  Ort::Status SetPowerConfig(uint32_t htp_power_config_client_id,
                             const QNN_INTERFACE_VER_TYPE& qnn_interface,
                             const Ort::Logger& logger);

  void CreateTimerThread(uint32_t htp_power_config_client_id, const Ort::Logger& logger);

  void ReleaseTimerThread();

  Ort::Status SetState(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger);

  void Init(const QNN_INTERFACE_VER_TYPE& qnn_interface) { qnn_interface_ = &qnn_interface; }

 private:
  // Sets voltage corner votes for HTP based on the given performance mode
  Ort::Status SetHtpPerformancePowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config,
                                           uint32_t htp_power_config_client_id,
                                           const HtpPerformanceMode& htp_performance_mode);

  Ort::Status SetSustainedPerformance(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger);

  Ort::Status SetPerformance(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger);

  static void TimerCallback(void* user_data);

  bool IsTimerThreadRunning();

  Ort::Status SetHtpPowerConfigs(const HtpPerfConfig_t& config, const Ort::Logger& logger);

  Ort::Status SetHtpPowerCustomConfigs(uint32_t htp_power_config_client_id, const QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t rpc_polling_time, uint32_t rpc_control_latency, const Ort::Logger& logger);

  enum class DcvsState {
    DCVS_DEFAULT = 0,
    DCVS_DISABLE = 1,
    DCVS_ENABLE = 2
  };

  // Sets power config for relaxed performance mode based on DCVS state
  void SetRelaxedPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config,
                                 uint32_t htp_power_config_client_id,
                                 DcvsState dcvsState);

  // Sets power config for released performance mode based on DCVS state
  void SetReleasedPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id, DcvsState dcvsState);

  // Sets power config for extreme low performance mode
  void SetExtremeLowPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id);

  uint32_t last_set_rpc_polling_time_ = kDisableRpcPolling;
  uint32_t last_set_rpc_control_latency_ = kDisableRpcControlLatency;

  bool rpc_polling_time_set_ = false;
  bool rpc_control_latency_set_ = false;
  bool htp_performance_mode_set_ = false;

  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs_;

  const QNN_INTERFACE_VER_TYPE* qnn_interface_ = nullptr;

  std::mutex perf_mutex_;
  std::mutex state_mutex_;
  std::unique_ptr<Timer> timer_;
  struct TimerResource {
    static constexpr uint64_t sustained_timer_duration_ = kDefaultTimerTimeoutUs;  // in microseconds
    std::atomic<bool> caller_busy_ = false;
    std::atomic<bool> timer_active_ = false;
  };
  TimerResource timer_resource_;
  std::atomic<GraphState> graph_state_ = GraphState::NONE;
  struct TimerCallbackArg {
    uint32_t power_config_id_;
    HtpPowerConfigManager* instance_;
    const Ort::Logger* logger_ptr_;
    TimerCallbackArg(uint32_t id, HtpPowerConfigManager* manager, const Ort::Logger& logger)
        : power_config_id_(id), instance_(manager), logger_ptr_(&logger) {}
  };
  std::unique_ptr<TimerCallbackArg> timer_callback_arg_;
};

}  // namespace power
}  // namespace qnn
}  // namespace onnxruntime
