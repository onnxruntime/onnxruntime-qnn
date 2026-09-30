// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <memory>

#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"
#include "test/providers/qnn/infra/qnn_test_logger.h"

namespace onnxruntime {
namespace test {

struct StubApiEnv {
  OrtApi stub_ort_api{};
  OrtEpApi stub_ep_api{};
  OrtModelEditorApi stub_editor_api{};
  Ort::Logger logger{MakeNullLogger()};
  ApiPtrs api_ptrs{stub_ort_api, stub_ep_api, stub_editor_api};

  StubApiEnv() = default;
  StubApiEnv(const StubApiEnv&) = delete;
  StubApiEnv& operator=(const StubApiEnv&) = delete;
};

// Owns a QnnBackendManager created through the public Create() factory with no
// backend library loaded, and exposes the pieces QnnModelWrapper reads as
// mutable references so component tests can stub them. The accessors are defined
// in qnn_unit_test_utils.cc because reaching the private members uses a
// single-TU explicit-instantiation helper.
class StubBackendManager {
 public:
  StubBackendManager(const ApiPtrs& api_ptrs, const Ort::Logger& logger) {
    qnn::QnnBackendManagerConfig cfg{};
    cfg.profiling_level = qnn::ProfilingLevel::OFF;
    cfg.profiling_level_etw = qnn::ProfilingLevel::OFF;
    cfg.context_priority = qnn::ContextPriority::NORMAL;
    cfg.htp_arch = QNN_HTP_DEVICE_ARCH_NONE;
    cfg.soc_model = QNN_SOC_MODEL_UNKNOWN;
    cfg.skip_qnn_version_check = true;
    manager_ = qnn::QnnBackendManager::Create(cfg, api_ptrs, logger);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StubBackendManager);

  const qnn::QnnBackendManager* Get() const { return manager_.get(); }

  QNN_INTERFACE_VER_TYPE& QnnInterface();
  Qnn_BackendHandle_t& BackendHandle();
  QNN_INTERFACE_VER_TYPE& ValidatorInterface();
  Qnn_BackendHandle_t& ValidatorBackendHandle();
  qnn::QnnBackendType& BackendType();
  QnnHtpDevice_Arch_t& HtpArch();

 private:
  std::shared_ptr<qnn::QnnBackendManager> manager_;
};

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
