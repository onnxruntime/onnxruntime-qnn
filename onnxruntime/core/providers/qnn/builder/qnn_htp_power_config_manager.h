// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <memory>
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

  // Drops a power-config id from the boosted set (public entry; acquires
  // perf_mutex_). Call when a session/id is being destroyed while the shared
  // timer may still be alive, so the timer does not later relax a destroyed id.
  void DropBoostedPowerConfigId(uint32_t htp_power_config_client_id);

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

  // Registers/removes a power-config id in the set of ids currently holding a
  // burst/sustained (boosted) vote. The release timer relaxes exactly this set
  // on timeout. Both helpers assume the caller already holds perf_mutex_.
  void RegisterBoostedId(uint32_t htp_power_config_client_id);
  void RemoveBoostedId(uint32_t htp_power_config_client_id);

  // Removes the id from the boosted set and aborts any pending release timer.
  // Acquires perf_mutex_ internally to snapshot the timer, then aborts outside
  // the lock (see the implementation for the deadlock rationale). Callers must
  // NOT hold perf_mutex_.
  void AbortActiveTimerAndDropBoostedId(uint32_t htp_power_config_client_id);

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

  // ---------------------------------------------------------------------------
  // Concurrency contract (read before touching any timer/lock code)
  // ---------------------------------------------------------------------------
  // This manager may be shared by multiple sessions (htp_share_resource_optimization_
  // / weight sharing), each driving runs concurrently, plus a background timer
  // thread that fires TimerCallback -> SetState(TIMEOUT). Two mutexes:
  //
  //   state_mutex_ : guards graph_state_ and timer_resource_.inflight_run_count_.
  //   perf_mutex_  : guards the timer lifecycle (timer_, timer_callback_arg_) and
  //                  boosted_config_ids_.
  //
  // Lock order: state_mutex_ BEFORE perf_mutex_. Never acquire state_mutex_ while
  // holding perf_mutex_. (In practice SetState takes state_mutex_ in a scope that
  // closes before it dispatches to the perf_mutex_-taking setters, so the two are
  // never nested.)
  //
  // The AbortTimer() hazard: Timer::AbortTimer() BLOCKS until the timer thread
  // reaches IDLE, and the timer thread, inside TimerCallback, calls
  // SetState(TIMEOUT) -> SetSustainedPerformance which needs perf_mutex_.
  // Therefore a thread must NEVER hold perf_mutex_ (or state_mutex_) across a
  // blocking AbortTimer() call, or the callback can never make progress -> deadlock.
  // This is enforced two ways:
  //   1. Callers that abort while holding perf_mutex_ (SetSustainedPerformance
  //      RUN_*/INIT_* cases) are made safe by rule 2, which guarantees the callback
  //      does not block on perf_mutex_.
  //   2. The re-entrant TIMEOUT path (SetSustainedPerformance under TimerCallback)
  //      acquires perf_mutex_ with try_lock and BAILS on contention. Contention
  //      means a run transition holds the lock (starting -> must not relax;
  //      finishing -> will re-arm), so skipping the relax is always safe, and the
  //      callback returns promptly so AbortTimer() can complete.
  //   3. Paths that abort outside any lock (AbortActiveTimerAndDropBoostedId)
  //      snapshot timer_ under perf_mutex_, release it, then AbortTimer() on the
  //      snapshot.
  //
  // timer_ is a shared_ptr so a snapshot taken under perf_mutex_ keeps the Timer
  // alive even if ReleaseTimerThread concurrently resets timer_; Timer::AbortTimer()
  // also early-returns if the timer is already deinitialized, so aborting a stale
  // snapshot can never hang.
  //
  // NOTE: today ReleaseTimerThread runs only from QnnBackendManager::ReleaseResources
  // (manager teardown), which cannot overlap an in-flight SetState (run callers hold
  // a shared_ptr to the manager; the timer thread is joined first via DeInitialize).
  // So the shared_ptr is defensive rather than strictly required now — but it makes
  // the "abort on a snapshot outside perf_mutex_" pattern safe by construction, so do
  // NOT downgrade it to unique_ptr without re-proving that release can never race a
  // snapshot holder.
  //
  // timer_resource_.timer_active_ is atomic and used as an advisory "a live timer
  // exists" flag readable under state_mutex_ without touching perf_mutex_-guarded
  // timer_ (see SetState's sustained/burst validity check).
  //
  // Timer teardown (ReleaseTimerThread) is tied to the MANAGER's lifetime
  // (QnnBackendManager::ReleaseResources), NOT to an individual QnnEp destructor,
  // so a shared timer is not killed while other sessions are still using it.
  // ---------------------------------------------------------------------------
  std::mutex perf_mutex_;
  std::mutex state_mutex_;
  std::shared_ptr<Timer> timer_;
  struct TimerResource {
    static constexpr uint64_t sustained_timer_duration_ = kDefaultTimerTimeoutUs;  // in microseconds
    // Number of runs currently in flight across ALL graphs that share this
    // manager. A single session can drive several QnnModels (each with its own
    // graph_exec_mutex_) through this one manager concurrently, so a single
    // busy flag is insufficient: the HTP release timer may only be allowed to
    // relax performance once this reaches 0 (i.e. the last concurrent run has
    // finished). Otherwise a still-running graph would be dropped to SVS
    // mid-computation. Atomic because it is written under state_mutex_ but read
    // under perf_mutex_.
    std::atomic<int> inflight_run_count_ = 0;
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

  // Distinct power-config client ids that currently hold a burst/sustained
  // (boosted) vote and are therefore this timer's responsibility to relax when
  // it fires. Under a shared manager each concurrent session contributes its
  // own non-zero id; each is an independent DCVS voter
  // (QnnHtpPerfInfrastructure.h: setPowerConfig associates settings per client
  // id), so the single release timer must relax every boosted id — not only the
  // construction-time id. Ids that move to a low-power/default vote (via the
  // dynamic, pre-run, or post-run perf paths) are removed so their intentional
  // low state is never stomped. Only ever accessed under perf_mutex_.
  std::vector<uint32_t> boosted_config_ids_;
};

}  // namespace power
}  // namespace qnn
}  // namespace onnxruntime
