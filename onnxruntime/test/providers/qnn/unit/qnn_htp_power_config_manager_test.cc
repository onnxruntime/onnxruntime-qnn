// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Host-side unit tests for HtpPowerConfigManager::SetState's in-flight run
// counter (qnn_htp_power_config_manager.cc).
//
// These tests need no QNN device. The manager is driven entirely through its
// existing public API (Init / CreateTimerThread / SetState / ReleaseTimerThread)
// and a stub QNN interface whose deviceGetInfrastructure returns an in-process
// perf-infrastructure table. The timer is a plain host thread (Timer::Initialize
// just spawns std::thread).
//
// Regression coverage for the leak fixed in qnn_model.cc: SetState(RUN_START)
// increments inflight_run_count_ before dispatching to the perf setter, so if a
// pre-run call fails (or unwinds) without a paired RUN_DONE, the counter is left
// elevated. A leaked count permanently gates the release timer's TIMEOUT relax,
// which is only allowed to fire when inflight_run_count_ == 0 (see SetState's
// TIMEOUT case). That relax is the counter's ONLY externally observable effect,
// so these tests assert on it directly: they drive SetState(TIMEOUT) (the same
// entry point the timer thread's callback uses) and check whether the stubbed
// setPowerConfig was invoked. This proves the count actually gates the relax,
// not merely that a private integer changed.
//
// The counter is a single manager-wide value (not per-id): every RUN_START /
// INIT_START increments it and every RUN_DONE / INIT_DONE decrements it (clamped
// at 0). So a leak is modelled by a RUN_START with no matching done, and the
// gate clears only once starts and dones balance out.

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <QnnInterface.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>

#include "core/providers/qnn/builder/qnn_htp_power_config_manager.h"
#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

namespace onnxruntime {
namespace test {

using qnn::HtpPerformanceMode;
using qnn::power::GraphState;
using qnn::power::HtpPerfConfig_t;
using qnn::power::HtpPowerConfigManager;

namespace {

// ---------------------------------------------------------------------------
// StubHtpPerfInterface
//
// Backs a QNN interface whose deviceGetInfrastructure returns a perf-infra table
// pointing at a process-static counter. setPowerConfig increments that counter
// so a test can observe whether the manager actually pushed a power vote. This
// is the only signal the in-flight run counter produces to the outside world:
// SetState(TIMEOUT) relaxes (calls setPowerConfig) only when the count is 0.
//
// gtest runs tests sequentially in one process, so a single file-static counter
// is safe; each test zeroes it via the RAII guard below.
// ---------------------------------------------------------------------------
struct StubHtpPerfInterface {
  static int set_power_config_calls;
  static QnnHtpDevice_Infrastructure_t infra;

  static Qnn_ErrorHandle_t SetPowerConfig(uint32_t /*power_config_id*/,
                                          const QnnHtpPerfInfrastructure_PowerConfig_t** /*config*/) {
    ++set_power_config_calls;
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t CreatePowerConfigId(uint32_t /*device_id*/, uint32_t /*core_id*/,
                                               uint32_t* power_config_id) {
    if (power_config_id != nullptr) {
      *power_config_id = 1;
    }
    return QNN_SUCCESS;
  }

  static Qnn_ErrorHandle_t GetInfrastructure(const QnnDevice_Infrastructure_t* device_infra) {
    // The manager casts the returned handle to QnnHtpDevice_Infrastructure_t*;
    // hand back the process-static table.
    *const_cast<QnnDevice_Infrastructure_t*>(device_infra) =
        reinterpret_cast<QnnDevice_Infrastructure_t>(&infra);
    return QNN_SUCCESS;
  }

  // Builds a zero-initialised QNN interface wired to the stubs above.
  static QNN_INTERFACE_VER_TYPE MakeInterface() {
    infra = QnnHtpDevice_Infrastructure_t{};
    infra.infraType = QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF;
    infra.perfInfra.createPowerConfigId = &CreatePowerConfigId;
    infra.perfInfra.setPowerConfig = &SetPowerConfig;

    QNN_INTERFACE_VER_TYPE qnn_interface = QNN_INTERFACE_VER_TYPE_INIT;
    qnn_interface.deviceGetInfrastructure = &GetInfrastructure;
    return qnn_interface;
  }
};

int StubHtpPerfInterface::set_power_config_calls = 0;
QnnHtpDevice_Infrastructure_t StubHtpPerfInterface::infra{};

// Zeroes the observable call counter for a test's duration.
struct ResetStubCounters {
  ResetStubCounters() { StubHtpPerfInterface::set_power_config_calls = 0; }
  ~ResetStubCounters() { StubHtpPerfInterface::set_power_config_calls = 0; }
};

// A burst perf config. Burst/sustained is the mode that (a) requires an active
// timer and (b) drives the inflight_run_count_ increment in SetState.
HtpPerfConfig_t MakeBurstConfig(uint32_t client_id = 1) {
  HtpPerfConfig_t config{};
  config.htp_power_config_client_id = client_id;
  config.perf_mode = HtpPerformanceMode::kHtpBurst;
  config.rpc_polling_time = 0;
  config.rpc_control_latency = 0;
  return config;
}

// Fires SetState(TIMEOUT) synchronously — the exact call the timer thread's
// callback makes — and returns how many setPowerConfig calls the relax issued.
// TIMEOUT does not require the timer to be *running*, only timer_active_ (set by
// CreateTimerThread) so the sustained/burst validity check passes. A non-zero
// return means the manager saw inflight_run_count_ == 0 and relaxed the boosted
// ids; zero means the count gated (skipped) the relax.
int FireTimeoutRelaxCalls(HtpPowerConfigManager& manager, const Ort::Logger& logger) {
  int before = StubHtpPerfInterface::set_power_config_calls;
  // TIMEOUT is delivered with the sustained perf mode, mirroring TimerCallback.
  HtpPerfConfig_t timeout_config = MakeBurstConfig();
  timeout_config.perf_mode = HtpPerformanceMode::kHtpSustainedHighPerformance;
  manager.SetState(GraphState::TIMEOUT, timeout_config, logger);
  return StubHtpPerfInterface::set_power_config_calls - before;
}

// Common fixture wiring: stub interface + active timer.
struct ManagerFixture {
  HtpPowerConfigManager manager;
  Ort::Logger logger = MakeNullLogger();
  QNN_INTERFACE_VER_TYPE qnn_interface = StubHtpPerfInterface::MakeInterface();

  ManagerFixture() {
    manager.Init(qnn_interface);
    manager.CreateTimerThread(/*htp_power_config_client_id=*/1);
  }
  ~ManagerFixture() { manager.ReleaseTimerThread(); }
};

}  // namespace

// A balanced RUN_START / RUN_DONE pair leaves inflight_run_count_ at 0, so a
// following TIMEOUT relaxes the HTP (fires setPowerConfig). This is the healthy
// baseline the leak breaks.
TEST(QnnUnit_HtpPowerConfigManagerTest, SetState_BalancedRun_TimeoutRelaxesHtp) {
  ResetStubCounters reset;
  ManagerFixture fx;
  HtpPerfConfig_t config = MakeBurstConfig();

  fx.manager.SetState(GraphState::RUN_START, config, fx.logger);
  fx.manager.SetState(GraphState::RUN_DONE, config, fx.logger);

  EXPECT_GT(FireTimeoutRelaxCalls(fx.manager, fx.logger), 0)
      << "With inflight_run_count_ back at 0 the TIMEOUT relax must fire.";
}

// The leak scenario: a RUN_START increments the count, but its paired RUN_DONE
// never runs (e.g. the pre-run setter failed and — before the qnn_model.cc fix —
// the scope guard did not issue RUN_DONE). The count stays at 1, so a TIMEOUT
// finds count != 0 and SKIPS the relax, leaving the HTP pinned boosted for the
// rest of the session. Draining the leaked start with a RUN_DONE then reopens
// the relax path — proving the gate was the count, not some other state.
TEST(QnnUnit_HtpPowerConfigManagerTest, SetState_LeakedRunStart_TimeoutSkipsRelaxUntilDrained) {
  ResetStubCounters reset;
  ManagerFixture fx;
  HtpPerfConfig_t config = MakeBurstConfig();

  // Leaked start, no paired done: count == 1.
  fx.manager.SetState(GraphState::RUN_START, config, fx.logger);

  EXPECT_EQ(FireTimeoutRelaxCalls(fx.manager, fx.logger), 0)
      << "A leaked inflight_run_count_ must gate the TIMEOUT relax (count != 0).";

  // Drain the leaked start; count returns to 0 and the relax path reopens.
  fx.manager.SetState(GraphState::RUN_DONE, config, fx.logger);

  EXPECT_GT(FireTimeoutRelaxCalls(fx.manager, fx.logger), 0)
      << "Once the leaked count drains to 0 the TIMEOUT relax must fire again.";
}

// RUN_DONE clamps the counter at 0: an unpaired done (e.g. the qnn_model.cc
// scope guard firing after a RUN_START that was rejected before it incremented)
// can never drive the count negative. If it did, it would take extra RUN_STARTs
// to climb back to 0 and the next TIMEOUT would wrongly skip the relax. Two
// unpaired dones followed by a relaxing TIMEOUT proves the clamp holds.
TEST(QnnUnit_HtpPowerConfigManagerTest, SetState_UnpairedRunDone_ClampsAndStillRelaxes) {
  ResetStubCounters reset;
  ManagerFixture fx;
  HtpPerfConfig_t config = MakeBurstConfig();

  fx.manager.SetState(GraphState::RUN_DONE, config, fx.logger);
  fx.manager.SetState(GraphState::RUN_DONE, config, fx.logger);

  EXPECT_GT(FireTimeoutRelaxCalls(fx.manager, fx.logger), 0)
      << "Clamped-at-0 counter must let the TIMEOUT relax fire.";
}

// Concurrent runs across graphs sharing one manager accumulate the count, and
// the TIMEOUT relax stays gated until the LAST run finishes. This is the core
// invariant the counter exists for: a still-running graph must not be dropped
// to SVS mid-computation.
TEST(QnnUnit_HtpPowerConfigManagerTest, SetState_ConcurrentRuns_RelaxGatedUntilLastDone) {
  ResetStubCounters reset;
  ManagerFixture fx;
  HtpPerfConfig_t config_a = MakeBurstConfig(/*client_id=*/1);
  HtpPerfConfig_t config_b = MakeBurstConfig(/*client_id=*/2);

  // Two concurrent starts: count == 2.
  fx.manager.SetState(GraphState::RUN_START, config_a, fx.logger);
  fx.manager.SetState(GraphState::RUN_START, config_b, fx.logger);

  // First done drops count to 1; TIMEOUT must still skip the relax.
  fx.manager.SetState(GraphState::RUN_DONE, config_a, fx.logger);
  EXPECT_EQ(FireTimeoutRelaxCalls(fx.manager, fx.logger), 0)
      << "A still-in-flight concurrent run must keep the relax gated.";

  // Last done drops count to 0; TIMEOUT now relaxes.
  fx.manager.SetState(GraphState::RUN_DONE, config_b, fx.logger);
  EXPECT_GT(FireTimeoutRelaxCalls(fx.manager, fx.logger), 0)
      << "Once the last concurrent run finishes the TIMEOUT relax must fire.";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
