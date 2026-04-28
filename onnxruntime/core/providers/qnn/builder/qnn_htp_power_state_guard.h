// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/providers/qnn/builder/qnn_htp_power_config_manager.h"

namespace onnxruntime {
namespace qnn {

// RAII guard for HtpPowerConfigManager::SetState.
//
// Calls SetState(start_state, ...) on construction and SetState(done_state, ...)
// on destruction, ensuring the done state is always reached even on early returns.
//
// Typical usage (INIT_START / INIT_DONE pair):
//
//   power::HtpPowerConfigManager* power_manager = ...;
//   bool valid_power_config_id = ...;  // determined by caller based on whether power config id was successfully created
//   HtpPerfConfig_t config = ...;  // configured as needed for the operation
//   HtpPowerStateGuard power_guard(power_manager, valid_power_config_id, GraphState::INIT_START, GraphState::INIT_DONE,
//                                  config);
//   RETURN_IF_NOT_OK(power_guard.SetPreRunHtpPerfStatus());
//   auto status = DoWork(...);
//   RETURN_IF_NOT_OK(power_guard.SetPostRunHtpPerf());  // optional: capture post-run perf error
//   return status;
//
// Passing nullptr as power_manager creates a no-op guard (all calls succeed immediately).
class HtpPowerStateGuard {
 public:
  HtpPowerStateGuard(power::HtpPowerConfigManager* power_manager,
                     bool valid_power_config_id,
                     GraphState start_state,
                     GraphState done_state,
                     const HtpPerfConfig_t& config)
      : power_manager_(power_manager),
        valid_power_config_id_(valid_power_config_id),
        done_state_(done_state),
        config_(config),
        finalized_(false) {
    if (power_manager_ && valid_power_config_id_) {
      start_status_ = power_manager_->SetState(start_state, config_);
    }
  }
  ~HtpPowerStateGuard() {
    if (!finalized_ && power_manager_ && valid_power_config_id_) {
      // Error cannot be propagated from a destructor; silently ignore.
      power_manager_->SetState(done_state_, config_);
    }
  }
  // Returns (by move) the status of setting HTP performance before work begins.
  // Should be checked immediately after construction.
  Ort::Status SetPreRunHtpPerfStatus() { return std::move(start_status_); }
  // Explicitly sets HTP performance after work is done and returns its status.
  // After this call the destructor will not invoke SetState again.
  Ort::Status SetPostRunHtpPerf() {
    finalized_ = true;
    if (power_manager_ && valid_power_config_id_) {
      return power_manager_->SetState(done_state_, config_);
    }
    return Ort::Status();
  }
  HtpPowerStateGuard(const HtpPowerStateGuard&) = delete;
  HtpPowerStateGuard& operator=(const HtpPowerStateGuard&) = delete;

 private:
  power::HtpPowerConfigManager* power_manager_;
  bool valid_power_config_id_;
  GraphState done_state_;
  HtpPerfConfig_t config_;
  Ort::Status start_status_;
  bool finalized_;
};

}  // namespace qnn
}  // namespace onnxruntime
