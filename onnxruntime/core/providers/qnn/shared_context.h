// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_model.h"

namespace onnxruntime {

class SharedContext {
 public:
  static SharedContext& GetInstance() {
    static SharedContext instance_;
    return instance_;
  }

  bool HasSharedQnnModels(const std::shared_ptr<qnn::QnnBackendManager>& producer) {
    const std::lock_guard<std::mutex> lock(mtx_);
    return std::any_of(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                       [&producer](const SharedQnnModel& entry) { return entry.producer == producer; });
  }

  bool HasQnnModel(const std::string& model_name,
                   const std::shared_ptr<qnn::QnnBackendManager>& producer) {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto it = std::find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                           [&model_name, &producer](const SharedQnnModel& entry) {
                             return entry.producer == producer && entry.qnn_model->Name() == model_name;
                           });
    return it != shared_qnn_models_.end();
  }

  std::unique_ptr<qnn::QnnModel> GetSharedQnnModel(
      const std::string& model_name,
      const std::shared_ptr<qnn::QnnBackendManager>& producer) {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto it = std::find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                           [&model_name, &producer](const SharedQnnModel& entry) {
                             return entry.producer == producer && entry.qnn_model->Name() == model_name;
                           });
    if (it == shared_qnn_models_.end()) {
      return nullptr;
    }
    auto qnn_model = std::move(it->qnn_model);
    shared_qnn_models_.erase(it);
    return qnn_model;
  }

  bool SetSharedQnnModel(std::vector<std::unique_ptr<qnn::QnnModel>>&& shared_qnn_models,
                         const std::shared_ptr<qnn::QnnBackendManager>& producer,
                         std::string& duplicate_graph_names) {
    const std::lock_guard<std::mutex> lock(mtx_);
    bool graph_exist = false;
    for (auto& shared_qnn_model : shared_qnn_models) {
      auto& model_name = shared_qnn_model->Name();
      auto it = std::find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                             [&model_name, &producer](const SharedQnnModel& entry) {
                               return entry.producer == producer && entry.qnn_model->Name() == model_name;
                             });
      if (it == shared_qnn_models_.end()) {
        shared_qnn_models_.push_back(SharedQnnModel{std::move(shared_qnn_model), producer});
      } else {
        duplicate_graph_names.append(model_name + " ");
        graph_exist = true;
      }
    }

    return graph_exist;
  }

  void RemoveSharedQnnModels(const std::shared_ptr<qnn::QnnBackendManager>& producer) {
    const std::lock_guard<std::mutex> lock(mtx_);
    shared_qnn_models_.erase(
        std::remove_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                       [&producer](const SharedQnnModel& entry) { return entry.producer == producer; }),
        shared_qnn_models_.end());
  }

  bool SetSharedQnnBackendManager(std::shared_ptr<qnn::QnnBackendManager>& qnn_backend_manager) {
    const std::lock_guard<std::mutex> lock(mtx_);

    if (qnn_backend_manager_ != nullptr) {
      if (qnn_backend_manager_ == qnn_backend_manager) {
        return true;
      }
      return false;
    }
    qnn_backend_manager_ = qnn_backend_manager;
    return true;
  }

  std::shared_ptr<qnn::QnnBackendManager> GetSharedQnnBackendManager() {
    const std::lock_guard<std::mutex> lock(mtx_);
    return qnn_backend_manager_;
  }

  void ResetSharedQnnBackendManager() {
    const std::lock_guard<std::mutex> lock(mtx_);
    qnn_backend_manager_.reset();
  }

  void SetSharedCtxBinFileName(std::string& shared_ctx_bin_file_name) {
    const std::lock_guard<std::mutex> lock(mtx_);
    shared_ctx_bin_file_name_ = shared_ctx_bin_file_name;
  }

  const std::string& GetSharedCtxBinFileName() {
    const std::lock_guard<std::mutex> lock(mtx_);
    return shared_ctx_bin_file_name_;
  }

  void ResetSharedCtxBinFileName() {
    const std::lock_guard<std::mutex> lock(mtx_);
    shared_ctx_bin_file_name_.clear();
  }

 private:
  struct SharedQnnModel {
    std::unique_ptr<qnn::QnnModel> qnn_model;
    std::shared_ptr<qnn::QnnBackendManager> producer;
  };

  SharedContext() = default;
  ~SharedContext() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SharedContext);

  // Used for passing through QNN models (deserialized from context binary) across sessions
  std::vector<SharedQnnModel> shared_qnn_models_;
  // Used for compiling multiple models into same QNN context binary
  std::shared_ptr<qnn::QnnBackendManager> qnn_backend_manager_;
  // Track the shared ctx binary .bin file name, all _ctx.onnx point to this .bin file
  // only the last session generate the .bin file since it contains all graphs from all sessions.
  std::string shared_ctx_bin_file_name_;
  // Producer sessions can be in parallel
  // Consumer sessions have to be after producer sessions initialized
  std::mutex mtx_;
};

}  // namespace onnxruntime
