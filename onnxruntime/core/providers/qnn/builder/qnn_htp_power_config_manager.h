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

// Graph states to tune the power/performance configurations
enum class GraphState {
  INIT_START,
  INIT_DONE,
  RUN_START,
  RUN_DONE,
  TIMEOUT,
  NONE
};

typedef struct HtpPerfConfig {
  uint32_t htp_power_config_client_id;
  HtpPerformanceMode perf_mode;
  uint32_t rpc_polling_time;
  uint32_t rpc_control_latency;
} HtpPerfConfig_t;

// Manages staging of any new power configurations and
// updates power configurations for the HTP backend.
//
// IMPORTANT: Init() must be called before any other methods that access
// the QNN interface (SetState, SetPowerConfig, etc.), typically during
// backend initialization. Failure to call Init() will result in errors
// when attempting to set power configurations.
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

  void CreateTimerThread(uint32_t htp_power_config_client_id);

  void ReleaseTimerThread();

  Ort::Status SetState(GraphState state, const HtpPerfConfig_t& config, const Ort::Logger& logger);

  void Init(const QNN_INTERFACE_VER_TYPE& qnn_interface) { qnn_interface_ = &qnn_interface; }

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(HtpPowerConfigManager);
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

  // Sets power config for relaxed performance mode
  void SetRelaxedPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config,
                                 uint32_t htp_power_config_client_id);

  // Sets power config for released performance mode
  void SetReleasedPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id);

  // Sets power config for extreme low performance mode
  void SetExtremeLowPerfPowerConfig(QnnHtpPerfInfrastructure_PowerConfig_t& power_config, uint32_t htp_power_config_client_id);

  uint32_t last_set_rpc_polling_time_ = kDisableRpcPolling;
  uint32_t last_set_rpc_control_latency_ = kDisableRpcControlLatency;

  bool rpc_polling_time_set_ = false;
  bool rpc_control_latency_set_ = false;
  bool htp_performance_mode_set_ = false;

  std::vector<QnnHtpPerfInfrastructure_PowerConfig_t> power_configs_;

  const QNN_INTERFACE_VER_TYPE* qnn_interface_ = nullptr;

  // Lock acquisition order: state_mutex_ must always be acquired before perf_mutex_
  // to prevent deadlocks. Never acquire state_mutex_ while already holding perf_mutex_.
  //
  // state_mutex_ guards both graph_state_ and the timer lifecycle (timer_,
  // timer_callback_arg_, timer_resource_.timer_active_).
  std::mutex perf_mutex_;
  std::mutex state_mutex_;
  std::unique_ptr<Timer> timer_;
  struct TimerResource {
    static constexpr uint64_t sustained_timer_duration_ = kDefaultTimerTimeoutUs;  // in microseconds
    std::atomic<bool> caller_busy_ = false;
    std::atomic<bool> timer_active_ = false;
  };
  TimerResource timer_resource_;
  GraphState graph_state_ = GraphState::NONE;
  struct TimerCallbackArg {
    uint32_t power_config_id_;
    HtpPowerConfigManager* instance_;
    TimerCallbackArg(uint32_t id, HtpPowerConfigManager* manager)
        : power_config_id_(id), instance_(manager) {}
  };
  std::unique_ptr<TimerCallbackArg> timer_callback_arg_;
};

}  // namespace power
}  // namespace qnn
}  // namespace onnxruntime
