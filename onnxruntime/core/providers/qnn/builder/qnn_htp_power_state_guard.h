// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/providers/qnn/builder/qnn_htp_power_config_manager.h"

namespace onnxruntime {
namespace qnn {
// RAII guard for HtpPowerConfigManager::SetState.
//
// Calls SetState(start_state, ...) when SetPreRunHtpPerfStatus() is invoked and
// SetState(done_state, ...) on destruction, ensuring the done state is always reached
// even on early returns.
//
// Typical usage (INIT_START / INIT_DONE pair):
//
//   power::HtpPowerConfigManager* power_manager = ...;
//   bool valid_power_config_id = ...;  // determined by caller based on whether power config id was successfully created
//   power::HtpPerfConfig_t config = ...;  // configured as needed for the operation
//   HtpPowerStateGuard power_guard(power_manager, valid_power_config_id, power::GraphState::INIT_START, power::GraphState::INIT_DONE,
//                                  config);
//   // ... optional setup work ...
//   RETURN_IF_NOT_OK(power_guard.SetPreRunHtpPerfStatus());  // Sets the pre-run state here
//   auto status = DoWork(...);
//   RETURN_IF_NOT_OK(power_guard.SetPostRunHtpPerf());  // optional: capture post-run perf error
//   return status;
//
// Passing nullptr as power_manager creates a no-op guard (all calls succeed immediately).
class HtpPowerStateGuard {
 public:
  HtpPowerStateGuard(power::HtpPowerConfigManager* power_manager,
                     bool valid_power_config_id,
                     power::GraphState start_state,
                     power::GraphState done_state,
                     const power::HtpPerfConfig_t& config,
                     const Ort::Logger& logger)
      : power_manager_(power_manager),
        valid_power_config_id_(valid_power_config_id),
        start_state_(start_state),
        done_state_(done_state),
        config_(config),
        logger_(logger),
        pre_run_called_(false),
        finalized_(false) {
  }
  ~HtpPowerStateGuard() {
    if (pre_run_called_ && !finalized_ && power_manager_ && valid_power_config_id_) {
      // Error cannot be propagated from a destructor; silently ignore.
      power_manager_->SetState(done_state_, config_, logger_);
    }
  }
  // Sets HTP performance state before work begins and returns the status.
  // Should be called after construction and before the actual work starts.
  // This provides flexibility to perform other setup between construction and state setting.
  Ort::Status SetPreRunHtpPerfStatus() {
    pre_run_called_ = true;
    if (power_manager_ && valid_power_config_id_) {
      return power_manager_->SetState(start_state_, config_, logger_);
    }
    return Ort::Status();
  }
  // Explicitly sets HTP performance after work is done and returns its status.
  // After this call the destructor will not invoke SetState again.
  Ort::Status SetPostRunHtpPerf() {
    if (power_manager_ && valid_power_config_id_) {
      Ort::Status status = power_manager_->SetState(done_state_, config_, logger_);
      // Only mark finalized on success; on failure leave finalized_ false so the
      // destructor retries the done-state transition and the HTP perf state is relaxed.
      finalized_ = status.IsOK();
      return status;
    }
    finalized_ = true;
    return Ort::Status();
  }
  HtpPowerStateGuard(const HtpPowerStateGuard&) = delete;
  HtpPowerStateGuard& operator=(const HtpPowerStateGuard&) = delete;

 private:
  power::HtpPowerConfigManager* power_manager_;
  bool valid_power_config_id_;
  power::GraphState start_state_;
  power::GraphState done_state_;
  power::HtpPerfConfig_t config_;
  const Ort::Logger& logger_;
  bool pre_run_called_;
  bool finalized_;
};
}  // namespace qnn
}  // namespace onnxruntime
