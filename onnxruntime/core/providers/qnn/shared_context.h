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
  SharedContext() = default;
  ~SharedContext() = default;

  // Keep manager selection and model consumption together during initialization.
  std::unique_lock<std::mutex> GetInitializationLock() {
    return std::unique_lock<std::mutex>(initialization_mutex_);
  }

  std::shared_ptr<qnn::QnnBackendManager> FindSharedQnnBackendManager(
      const std::string& context_key, const std::vector<std::string>& names,
      const std::shared_ptr<qnn::QnnBackendManager>& candidate) {
    const std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& entry : shared_qnn_models_) {
      if (entry.context_key == context_key && entry.producer->IsContextSharingCompatible(*candidate) &&
          std::all_of(names.begin(), names.end(), [&](const std::string& name) {
            return FindSharedQnnModel(name, entry.producer, context_key) != shared_qnn_models_.end();
          })) {
        return entry.producer;
      }
    }
    return nullptr;
  }

  static SharedContext& GetInstance() {
    static SharedContext instance_;
    return instance_;
  }

  bool HasSharedQnnModels(const std::shared_ptr<qnn::QnnBackendManager>& producer) {
    const std::lock_guard<std::mutex> lock(mtx_);
    return std::any_of(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                       [&producer](const SharedQnnModel& entry) { return entry.producer == producer; });
  }

  bool HasSharedQnnModel(const std::string& model_name,
                         const std::shared_ptr<qnn::QnnBackendManager>& producer,
                         const std::string& context_key) {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto it = FindSharedQnnModel(model_name, producer, context_key);
    return it != shared_qnn_models_.end();
  }

  std::unique_ptr<qnn::QnnModel> GetSharedQnnModel(
      const std::string& model_name,
      const std::shared_ptr<qnn::QnnBackendManager>& producer,
      const std::string& context_key) {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto it = FindSharedQnnModel(model_name, producer, context_key);
    if (it == shared_qnn_models_.end()) {
      return nullptr;
    }
    auto qnn_model = std::move(it->qnn_model);
    shared_qnn_models_.erase(it);
    return qnn_model;
  }

  // The caller holds the initialization lock until it consumes the model.
  qnn::QnnModel* PeekSharedQnnModel(const std::string& model_name,
                                    const std::shared_ptr<qnn::QnnBackendManager>& producer,
                                    const std::string& context_key) {
    const std::lock_guard<std::mutex> lock(mtx_);
    const auto it = FindSharedQnnModel(model_name, producer, context_key);
    return it != shared_qnn_models_.end() ? it->qnn_model.get() : nullptr;
  }

  bool SetSharedQnnModel(std::vector<std::unique_ptr<qnn::QnnModel>>&& shared_qnn_models,
                         const std::shared_ptr<qnn::QnnBackendManager>& producer,
                         const std::string& context_key,
                         std::string& duplicate_graph_names) {
    const std::lock_guard<std::mutex> lock(mtx_);
    bool graph_exist = false;
    for (auto& shared_qnn_model : shared_qnn_models) {
      auto& model_name = shared_qnn_model->Name();
      auto it = FindSharedQnnModel(model_name, producer, context_key);
      if (it == shared_qnn_models_.end()) {
        shared_qnn_models_.push_back(SharedQnnModel{std::move(shared_qnn_model), producer, context_key});
      } else {
        duplicate_graph_names.append(model_name + " ");
        graph_exist = true;
      }
    }

    return graph_exist;
  }

  void RemoveSharedQnnModels(const std::shared_ptr<qnn::QnnBackendManager>& producer,
                             const std::string& context_key) {
    std::vector<std::unique_ptr<qnn::QnnModel>> removed;
    {
      const std::lock_guard<std::mutex> lock(mtx_);
      for (auto it = shared_qnn_models_.begin(); it != shared_qnn_models_.end();) {
        if (it->producer == producer && it->context_key == context_key) {
          removed.push_back(std::move(it->qnn_model));
          it = shared_qnn_models_.erase(it);
        } else {
          ++it;
        }
      }
    }
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
    std::string context_key;
  };

  // The caller must hold mtx_ while finding or using the returned iterator.
  std::vector<SharedQnnModel>::iterator FindSharedQnnModel(
      const std::string& model_name,
      const std::shared_ptr<qnn::QnnBackendManager>& producer,
      const std::string& context_key) {
    return std::find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                        [&model_name, &producer, &context_key](const SharedQnnModel& entry) {
                          return entry.producer == producer && entry.qnn_model->Name() == model_name &&
                                 entry.context_key == context_key;
                        });
  }

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
  std::mutex initialization_mutex_;
};

}  // namespace onnxruntime
