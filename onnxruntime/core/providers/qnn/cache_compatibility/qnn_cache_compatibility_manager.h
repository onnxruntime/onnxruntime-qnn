// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/providers/qnn/cache_compatibility/qnn_cache_compatibility_info.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Forward declaration.
class QnnBackendManager;

class QnnCacheCompatibilityManager {
 public:
  QnnCacheCompatibilityManager(QnnBackendManager* qnn_backend_manager)
      : qnn_backend_manager_(qnn_backend_manager) {}

  Ort::Status DeserializeCompatibilityInfo(const std::string& info_string, /*out*/ QnnCompatibilityInfo& info);

  Ort::Status GetCompatibilityInfo(/*out*/ QnnCompatibilityInfo& info);

  Ort::Status SerializeCompatibilityInfo(const QnnCompatibilityInfo& info, /*out*/ std::string& info_string);

  Ort::Status ValidateCompatibilityInfo(const QnnCompatibilityInfo& info,
                                        /*out*/ OrtCompiledModelCompatibility& compatibility);

 private:
  Ort::Status DeserializeCompatibilityInfoV1(const std::string& info_string, /*out*/ QnnCompatibilityInfoV1& info);

  Ort::Status DeserializeCompatibilityInfoV2(const std::string& info_string, /*out*/ QnnCompatibilityInfoV2& info);

  Ort::Status GetRuntimeCompatibilityInfoV1(/*out*/ QnnCompatibilityInfoV1& info);

  Ort::Status GetRuntimeCompatibilityInfoV2(/*out*/ QnnCompatibilityInfoV2& info);

  Ort::Status SerializeCompatibilityInfoV1(const QnnCompatibilityInfoV1& info, /*out*/ std::string& info_string);

  Ort::Status SerializeCompatibilityInfoV2(const QnnCompatibilityInfoV2& info, /*out*/ std::string& info_string);

  Ort::Status ValidateCompatibilityInfoV1(const QnnCompatibilityInfoV1& info,
                                          /*out*/ OrtCompiledModelCompatibility& compatibility);

  Ort::Status ValidateCompatibilityInfoV2(const QnnCompatibilityInfoV2& info,
                                          /*out*/ OrtCompiledModelCompatibility& compatibility);

 private:
  QnnBackendManager* qnn_backend_manager_;
};

}  // namespace qnn
}  // namespace onnxruntime
