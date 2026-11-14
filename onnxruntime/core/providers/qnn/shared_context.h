// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_model.h"

namespace onnxruntime {

class SharedContext {
 public:
  struct SharedQnnModel {
    std::string name;
    std::string provenance;
    std::unique_ptr<qnn::QnnModel> model;
  };

  static SharedContext& GetInstance() {
    static SharedContext instance_;
    return instance_;
  }

  bool HasSharedQnnModels() {
    const std::lock_guard<std::mutex> lock(mtx_);
    return !shared_qnn_models_.empty();
  }

  bool HasQnnModel(const std::string& model_name, const std::string& provenance) {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto it = find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                      [&model_name, &provenance](const SharedQnnModel& entry) {
                        return entry.name == model_name && entry.provenance == provenance;
                      });
    return it != shared_qnn_models_.end();
  }

  std::vector<std::unique_ptr<qnn::QnnModel>> TakeSharedQnnModels(
      const std::vector<std::string>& model_names,
      const std::vector<std::string>& provenances) {
    const std::lock_guard<std::mutex> lock(mtx_);
    if (model_names.size() != provenances.size()) {
      return {};
    }

    for (size_t i = 0; i < model_names.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        if (model_names[i] == model_names[j] && provenances[i] == provenances[j]) {
          return {};
        }
      }

      auto it = find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                        [&model_names, &provenances, i](const SharedQnnModel& entry) {
                          return entry.name == model_names[i] && entry.provenance == provenances[i];
                        });
      if (it == shared_qnn_models_.end()) {
        return {};
      }
    }

    std::vector<std::unique_ptr<qnn::QnnModel>> models;
    models.reserve(model_names.size());
    for (size_t i = 0; i < model_names.size(); ++i) {
      auto it = find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                        [&model_names, &provenances, i](const SharedQnnModel& entry) {
                          return entry.name == model_names[i] && entry.provenance == provenances[i];
                        });
      models.push_back(std::move(it->model));
      shared_qnn_models_.erase(it);
    }
    return models;
  }

  bool SetSharedQnnModel(std::vector<SharedQnnModel>&& shared_qnn_models,
                         std::string& duplicate_graph_names) {
    const std::lock_guard<std::mutex> lock(mtx_);
    bool graph_exist = false;
    for (auto& shared_qnn_model : shared_qnn_models) {
      auto it = find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                        [&shared_qnn_model](const SharedQnnModel& entry) {
                          return entry.name == shared_qnn_model.name &&
                                 entry.provenance == shared_qnn_model.provenance;
                        });
      if (it == shared_qnn_models_.end()) {
        shared_qnn_models_.push_back(std::move(shared_qnn_model));
      } else {
        duplicate_graph_names.append(shared_qnn_model.name + " ");
        graph_exist = true;
      }
    }

    return graph_exist;
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

  std::string GetOrSetSharedCtxBinFileName(const std::string& candidate) {
    const std::lock_guard<std::mutex> lock(mtx_);
    if (shared_ctx_bin_file_name_.empty()) {
      shared_ctx_bin_file_name_ = candidate;
    }
    return shared_ctx_bin_file_name_;
  }

  void ResetSharedCtxBinFileName() {
    const std::lock_guard<std::mutex> lock(mtx_);
    shared_ctx_bin_file_name_.clear();
  }

 private:
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
