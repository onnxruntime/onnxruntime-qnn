// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
#ifdef USE_QAIRT_API
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/runtime/qairt/qairt_profiling_manager.h"
#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing.h"
#include "core/providers/qnn/builder/qdq_node_group_selector.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/qnn_ep_profiler.h"
#include <fstream>
namespace onnxruntime::qnn {
Ort::Status QnnModel::LoadQairtGraph(QairtBackendManager::CachedGraph&& graph) {
  RETURN_IF_ERROR(QairtGraph().RetrieveGraph(std::move(graph.context), graph.info->Name()));
  graph_info_ = std::move(graph.info);
  return Ort::Status();
}
Ort::Status QnnModel::FinalizeQairtGraph(const Ort::Logger&) {
  QairtProfilingManager::Scope scope;
  QnnEpProfiler* profiler = nullptr;
#if QNN_ORT_EP_PROFILING_API_ENABLED
  profiler = QnnEpProfiler::Current();
#endif
  RETURN_IF_ERROR(qairt_profiler_->Begin(scope, graph_info_->Name(), OrtProfilingOperation::FINALIZE, profiler));
  const QairtGraphProfile graph_profile(scope.profile);
  RETURN_IF_ERROR(graph_emitter_->Finalize(&graph_profile));
  return qairt_profiler_->Complete(scope);
}
Ort::Status QnnModel::ExecuteQairtGraph(OrtKernelContext* context, const Ort::Logger& logger, QnnEpProfiler* profiler) {
  size_t num_inputs = 0, num_outputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetInputCount(context, &num_inputs));
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetOutputCount(context, &num_outputs));
  RETURN_IF(qnn_input_infos_.size() > num_inputs || qnn_output_infos_.size() != num_outputs,
            "Inconsistent QAIRT graph input/output counts.");
  std::lock_guard<std::mutex> lock(graph_exec_mutex_);
  std::vector<Qnn_Tensor_t> qnn_inputs, qnn_outputs;
  RETURN_IF_ERROR(BindGraphInputOutput(context, logger, qnn_inputs, qnn_outputs));
  QairtProfilingManager::Scope scope;
  RETURN_IF_ERROR(qairt_profiler_->Begin(scope, graph_info_->Name(), OrtProfilingOperation::EXECUTE, profiler));
  const QairtGraphProfile graph_profile(scope.profile);
  GraphExecutionResult execution_result;
  RETURN_IF_ERROR(graph_emitter_->Execute(qnn_inputs.data(), static_cast<uint32_t>(qnn_inputs.size()),
                                          qnn_outputs.data(), static_cast<uint32_t>(qnn_outputs.size()),
                                          &graph_profile, execution_result));
  return qairt_profiler_->Complete(scope);
}
}  // namespace onnxruntime::qnn
#endif
