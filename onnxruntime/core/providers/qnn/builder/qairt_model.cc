// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#ifdef USE_QAIRT_API
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qairt_profiling_manager.h"
#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing.h"
#include "core/providers/qnn/builder/qdq_node_group_selector.h"
#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/qnn_ep_profiler.h"
#include <fstream>
namespace onnxruntime::qnn {
Ort::Status QnnModel::LoadQairtGraph(QairtBackendManager::CachedGraph&& graph) {
  RETURN_IF_ERROR(qairt_graph_->RetrieveGraph(std::move(graph.context), graph.info->Name()));
  graph_info_ = std::move(graph.info);
  return Ort::Status();
}
Ort::Status QnnModel::ComposeQairtGraph(const QnnModelContext& context) {
  utils::UniqueNameGenerator().Reset();

  RETURN_IF(context.onnx_input_names == nullptr, "onnx_input_names is required for ComposeGraph");
  RETURN_IF(context.onnx_output_names == nullptr, "onnx_output_names is required for ComposeGraph");
  RETURN_IF(context.model_settings == nullptr, "model_settings is required for ComposeGraph");

  const OrtGraph& ort_graph = context.ort_graph;
  const OrtNode& fused_node = context.fused_node;
  const Ort::Logger& logger = context.logger;

  ORT_CXX_LOG(logger,
              ORT_LOGGING_LEVEL_VERBOSE,
              ("ComposeGraph Graph name: " + Ort::ConstGraph(&ort_graph).GetName()).c_str());

  // Holder for the OrtNodes in the graph, this will guarantee the OrtNodes is
  // valid throughout the lifetime of the ModelBuilder
  std::vector<std::unique_ptr<OrtNodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const OrtNodeUnit*> node_unit_map;
  // GetQDQNodeUnits
  std::tie(node_unit_holder, node_unit_map) = GetAllOrtNodeUnits(api_ptrs_.ort_api, &ort_graph, logger);

  // This name must be same with the EPContext node name
  const auto& graph_name = Ort::ConstNode(&fused_node).GetName();
  RETURN_IF_ERROR(SetGraphInputOutputInfo(context));

  // Framework op trace: create collector before QnnModelWrapper so it can be
  // passed at construction time. nullptr when tracing is disabled.
  std::unique_ptr<OpTraceCollector> trace_collector;
  if (context.op_trace_output) {
    trace_collector = std::make_unique<OpTraceCollector>();
  }

  QnnModelWrapper qnn_model_wrapper = QnnModelWrapper(ort_graph,
                                                      api_ptrs_,
                                                      logger,
                                                      *qairt_backend_manager_,
                                                      graph_inputs_,
                                                      graph_outputs_,
                                                      *context.model_settings,
                                                      context.tensor_name_overrides,
                                                      trace_collector.get(),
                                                      /*is_post_layout_transform=*/true, qairt_graph_.get());

  QairtProfilingManager::Scope scope;
  QnnEpProfiler* profiler = nullptr;
#if QNN_ORT_EP_PROFILING_API_ENABLED
  profiler = QnnEpProfiler::Current();
#endif
  RETURN_IF_ERROR(qairt_profiler_->Begin(scope, graph_name, OrtProfilingOperation::COMPOSE, profiler));
  bool rt = qnn_model_wrapper.CreateQnnGraph(nullptr, graph_name);

  RETURN_IF_NOT(rt, "Failed to initialize qnn_model_wrapper.");

  std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> qnn_node_groups;
  qnn_node_groups.reserve(node_unit_holder.size());

  RETURN_IF_ERROR(qnn::GetQnnNodeGroups(qnn_node_groups, qnn_model_wrapper, node_unit_map,
                                        node_unit_holder.size(), logger));

  for (const std::unique_ptr<qnn::IQnnNodeGroup>& qnn_node_group : qnn_node_groups) {
    NodeGroupGuard guard(trace_collector.get(), qnn_node_group.get());

    Ort::Status status = qnn_node_group->AddToModelBuilder(qnn_model_wrapper, logger);

    if (!status.IsOK()) {
      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_ERROR,
                  ("[QNN EP] Failed to add supported node to QNN graph during EP's compile call: " +
                   status.GetErrorMessage())
                      .c_str());
      return status;
    }
  }

  const bool build_json_graph = !context.json_qnn_graph_path.empty();
  RETURN_IF_NOT(qnn_model_wrapper.ComposeQnnGraph(build_json_graph), "Failed to compose Qnn graph.");

  RETURN_IF_ERROR(qairt_profiler_->Complete(scope));

  // Collect framework op trace after graph composition
  if (trace_collector) {
    OpTraceLookup per_graph_lookup;
    trace_collector->Finalize(graph_name, qnn_model_wrapper, *context.op_trace_output, per_graph_lookup);
    // Hand the per-graph lookup off to the profiling manager, which owns the
    // session-wide lookup that extraction reads from.
    qairt_profiler_->MergeOpTraceLookup(std::move(per_graph_lookup));
  }

  LogTensorDetails(qnn_model_wrapper, graph_name, context.json_qnn_graph_path, logger);

  if (build_json_graph) {
    const nlohmann::json& json_graph = qnn_model_wrapper.GetQnnJSONGraph();
    std::ofstream ofs(context.json_qnn_graph_path);

    if (ofs.is_open()) {
      ofs << json_graph.dump();
      ofs.close();
    } else {
      ORT_CXX_LOG(logger,
                  ORT_LOGGING_LEVEL_WARNING,
                  ("Could not open JSON graph file: " + context.json_qnn_graph_path).c_str());
    }
  }

  RETURN_IF_NOT(GetGraphInfoFromModel(qnn_model_wrapper, logger), "GetGraphInfoFromModel failed.");
  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, "GetGraphInfoFromModel completed.");
  return Ort::Status();
}

Ort::Status QnnModel::FinalizeQairtGraph(const Ort::Logger&) {
  QairtProfilingManager::Scope scope;
  QnnEpProfiler* profiler = nullptr;
#if QNN_ORT_EP_PROFILING_API_ENABLED
  profiler = QnnEpProfiler::Current();
#endif
  RETURN_IF_ERROR(qairt_profiler_->Begin(scope, graph_info_->Name(), OrtProfilingOperation::FINALIZE, profiler));
  RETURN_IF_ERROR(qairt_graph_->Finalize(scope.profile.get()));
  return qairt_profiler_->Complete(scope);
}
Ort::Status QnnModel::ExecuteQairtGraph(OrtKernelContext* context, const Ort::Logger& logger, QnnEpProfiler* profiler) {
  size_t num_inputs = 0, num_outputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetInputCount(context, &num_inputs));
  ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetOutputCount(context, &num_outputs));
  RETURN_IF(qnn_input_infos_.size() > num_inputs || qnn_output_infos_.size() != num_outputs,
            "Inconsistent QAIRT graph input/output counts.");
  std::lock_guard<std::mutex> lock(graph_exec_mutex_);
  auto BindRaw = [&](const OrtMemoryInfo* memory_info, void* data, uint32_t size, Qnn_Tensor_t& tensor) -> Ort::Status {
    OrtMemoryInfoDeviceType device;
    api_ptrs_.ort_api.MemoryInfoGetDeviceType(memory_info, &device);
    RETURN_IF(device != OrtMemoryInfoDeviceType_CPU,
              "QAIRT registered/imported-memory binding is not implemented.");
    SetQnnTensorMemType(tensor, QNN_TENSORMEMTYPE_RAW);
    SetQnnTensorClientBuf(tensor, data, size);
    return Ort::Status();
  };
  using namespace qnn::utils;
  auto TensorDataSize = [&ort_api = api_ptrs_.ort_api](auto ort_tensor) -> size_t {
    OrtTensorTypeAndShapeInfo* tensor_type_and_shape = nullptr;
    OrtStatusPtr tensor_status = ort_api.GetTensorTypeAndShape(ort_tensor, &tensor_type_and_shape);
    if (tensor_status != nullptr) {
      ort_api.ReleaseStatus(tensor_status);
      return 0;
    }
    size_t length;
    tensor_status = ort_api.GetTensorShapeElementCount(tensor_type_and_shape, &length);
    if (tensor_status != nullptr) {
      ort_api.ReleaseTensorTypeAndShapeInfo(tensor_type_and_shape);
      ort_api.ReleaseStatus(tensor_status);
      return 0;
    }
    ONNXTensorElementDataType element_type;
    tensor_status = ort_api.GetTensorElementType(tensor_type_and_shape, &element_type);
    if (tensor_status != nullptr) {
      ort_api.ReleaseTensorTypeAndShapeInfo(tensor_type_and_shape);
      ort_api.ReleaseStatus(tensor_status);
      return 0;
    }
    size_t element_size = GetElementSizeByType(element_type);
    ort_api.ReleaseTensorTypeAndShapeInfo(tensor_type_and_shape);
    return element_size * length;
  };

  std::vector<Qnn_Tensor_t> qnn_inputs;
  qnn_inputs.reserve(qnn_input_infos_.size());

  for (const auto& qnn_input_info : qnn_input_infos_) {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("model_input = " + qnn_input_info.tensor_wrapper->GetName() +
                 " index = " + std::to_string(qnn_input_info.ort_index))
                    .c_str());
    const OrtValue* ort_input_tensor = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetInput(context, qnn_input_info.ort_index, &ort_input_tensor));
    auto ort_tensor_size = TensorDataSize(ort_input_tensor);
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Qnn tensor size: " + std::to_string(qnn_input_info.tensor_byte_size) +
                 " Ort tensor size: " + std::to_string(ort_tensor_size))
                    .c_str());
    RETURN_IF_NOT(qnn_input_info.tensor_byte_size == ort_tensor_size,
                  "ORT Tensor data size does not match QNN tensor data size.");

    qnn_inputs.push_back(qnn_input_info.tensor_wrapper->GetQnnTensor());

    const OrtMemoryInfo* input_tensor_mem_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorMemoryInfo(ort_input_tensor, &input_tensor_mem_info));

    const void* raw_data;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorData(ort_input_tensor, &raw_data));

    RETURN_IF_ERROR(BindRaw(
        static_cast<const OrtMemoryInfo*>(input_tensor_mem_info),
        const_cast<void*>(raw_data), qnn_input_info.tensor_byte_size,
        qnn_inputs.back()));
  }

  std::vector<Qnn_Tensor_t> qnn_outputs;
  qnn_outputs.reserve(qnn_output_infos_.size());

  for (auto& qnn_output_info : qnn_output_infos_) {
    const std::string& model_output_name = qnn_output_info.tensor_wrapper->GetName();
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("model_output = " + model_output_name +
                 " index = " + std::to_string(qnn_output_info.ort_index))
                    .c_str());
    const auto& ort_output_info = GetOutputInfo(model_output_name);
    const std::vector<int64_t>& output_shape = ort_output_info->shape_;
    OrtValue* ort_output_tensor = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.KernelContext_GetOutput(context,
                                                                         qnn_output_info.ort_index,
                                                                         output_shape.data(),
                                                                         output_shape.size(),
                                                                         &ort_output_tensor));

    auto ort_tensor_size = TensorDataSize(ort_output_tensor);
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Qnn tensor size: " + std::to_string(qnn_output_info.tensor_byte_size) +
                 " Ort tensor size: " + std::to_string(ort_tensor_size))
                    .c_str());
    RETURN_IF_NOT(qnn_output_info.tensor_byte_size == ort_tensor_size,
                  "ORT Tensor data size does not match QNN tensor data size");

    qnn_outputs.push_back(qnn_output_info.tensor_wrapper->GetQnnTensor());

    const OrtMemoryInfo* output_tensor_mem_info = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorMemoryInfo(ort_output_tensor, &output_tensor_mem_info));

    void* mutable_data;
    ORT_CXX_RETURN_ON_API_FAIL(api_ptrs_.ort_api.GetTensorMutableData(ort_output_tensor, &mutable_data));

    RETURN_IF_ERROR(BindRaw(
        static_cast<const OrtMemoryInfo*>(output_tensor_mem_info),
        mutable_data, qnn_output_info.tensor_byte_size,
        qnn_outputs.back()));
  }

  QairtProfilingManager::Scope scope;
  RETURN_IF_ERROR(qairt_profiler_->Begin(scope, graph_info_->Name(), OrtProfilingOperation::EXECUTE, profiler));
  RETURN_IF_ERROR(qairt_graph_->Execute(qnn_inputs.data(), static_cast<uint32_t>(qnn_inputs.size()),
                                        qnn_outputs.data(), static_cast<uint32_t>(qnn_outputs.size()), scope.profile));
  return qairt_profiler_->Complete(scope);
}
}  // namespace onnxruntime::qnn
#endif
