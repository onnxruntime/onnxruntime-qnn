// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#ifdef USE_QAIRT_API
#include "core/providers/qnn/qnn_execution_provider.h"
#include "core/providers/qnn/builder/qairt_backend_manager.h"
#include "core/providers/qnn/builder/qairt_profiling_manager.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/qnn_ep_profiler.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/builder/ep_context_io_dispatch.h"

namespace onnxruntime {
namespace {
std::filesystem::path ContextPath(const OrtGraph* graph, const OrtApi& api, const std::string& configured) {
  return configured.empty() ? std::filesystem::path(GetModelPathString(graph, api)) : std::filesystem::path(configured);
}
Ort::Status ReadContext(const OrtNode* node, const std::filesystem::path& model_path,
                        const qnn::EpContextIoDispatch& io, std::vector<char>& bytes) {
  OrtNodeAttrHelper attrs(*node);
  RETURN_IF(attrs.Get(qnn::IS_MULTI_SOC_EP_CONTEXT, false), "QAIRT multi-SoC contexts require backend extensions.");
  const auto payload = attrs.Get(qnn::EP_CACHE_CONTEXT, std::string{});
  if (attrs.Get(qnn::EMBED_MODE, true)) {
    bytes.assign(payload.begin(), payload.end());
  } else {
    const auto relative = std::filesystem::path(payload).lexically_normal();
    RETURN_IF(payload.empty() || relative.has_root_path(), "EPContext requires a relative binary path.");
    for (const auto& component : relative) RETURN_IF(component == "..", "EPContext path escapes the model directory.");
    if (io.HasReadCallback()) return io.Read(payload, bytes);
    std::ifstream file(model_path.parent_path() / relative, std::ios::binary | std::ios::ate);
    RETURN_IF(!file, "Cannot open EPContext binary.");
    const auto size = file.tellg();
    RETURN_IF(size <= 0, "Empty or unreadable EPContext binary.");
    bytes.resize(static_cast<size_t>(size));
    file.seekg(0);
    RETURN_IF(!file.read(bytes.data(), size), "Cannot read EPContext binary.");
  }
  RETURN_IF(bytes.empty(), "Empty EPContext binary.");
  return Ort::Status();
}
}  // namespace

OrtStatus* QnnEp::GetQairtCapability(const OrtGraph* graph, OrtEpGraphSupportInfo* support, size_t count) try {
  // These options have no QAIRT implementation yet. Check explicit requests here,
  // rather than letting shared option parsing imply that the SDK applied them.
  const bool is_htp = qairt_backend_manager_->GetBackendType() == qnn::QnnBackendType::HTP;
  for (const auto& option : {std::pair<const char*, const char*>{"htp_arch", "0"},
                            {"soc_model", "0"}, {"device_id", "0"},
                            {"qnn_context_priority", "normal"},
                            {"enable_htp_cross_device_prepare", "0"},
                            {"htp_share_resource_optimization", "0"},
                            {"num_graph_prepare_threads", "1"},
                            {"enable_htp_matmul_lut", ""},
                            {"disable_file_mapped_weights", "1"}}) {
    if (!is_htp && (std::strcmp(option.first, "htp_arch") == 0 ||
                    std::strcmp(option.first, "soc_model") == 0)) {
      continue;
    }
    std::string value;
    RETURN_IF_NOT_NULL(GetSessionConfigEntryOrDefault(ort_api, session_options_, FormatEPConfigKey(option.first), "", value));
    if (!value.empty() && value != option.second) {
      const auto message = std::string("QAIRT option is not implemented: ") + option.first;
      return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }
  }
  RETURN_IF_NOT_OK(qairt_backend_manager_->Setup());
  if (qnn::GraphHasDlcContextNode(graph, ort_api))
    return ort_api.CreateStatus(ORT_EP_FAIL, "QAIRT sessions do not support the Genie DLC execution pathway.");
  if (is_htp &&
      (htp_graph_configs_.enable_htp_monolithic_lstm || htp_graph_configs_.enable_htp_fp16_precision ||
       htp_graph_configs_.enable_htp_fp16_clamp_overflow || htp_graph_configs_.vtcm_size_in_mb > 0 ||
       htp_graph_configs_.htp_graph_finalization_opt_mode != qnn::HtpGraphFinalizationOptimizationMode::kDefault ||
       htp_graph_configs_.htp_num_cores > 0 || enable_htp_graph_splitting_ || enable_htp_extended_udma_mode_ ||
       model_settings_.htp_bf16_enable || default_htp_performance_mode_ != qnn::HtpPerformanceMode::kHtpDefault ||
       default_rpc_control_latency_ || default_rpc_polling_time_)) {
    return ort_api.CreateStatus(ORT_EP_FAIL, "Requested QAIRT HTP graph or performance options require backend-extension support.");
  }
  if (qairt_requires_shared_memory_ || qairt_has_op_packages_ || enable_multi_soc_ep_context_ ||
      share_ep_contexts_ || enable_spill_fill_buffer_ || prepare_only_ || prepare_and_load_) {
    return ort_api.CreateStatus(ORT_EP_FAIL,
                                "QAIRT shared memory, op packages, shared/multi-SoC contexts and prepare modes are not implemented.");
  }
  if (qnn::GraphHasEpContextNode(graph, ort_api)) {
    PartitionCtxModel(graph, support);
    return nullptr;
  }
  return GetOnnxGraphCapability(graph, support, count, false);
} catch (const std::exception& e) {
  return ort_api.CreateStatus(ORT_EP_FAIL, e.what());
} catch (...) {
  return ort_api.CreateStatus(ORT_EP_FAIL, "Unknown QAIRT capability error.");
}
OrtStatus* QnnEp::CompileQairtModels(const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count,
                                     OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) try {
  if (count && qnn::GraphHasEpContextNode(graphs[0], ort_api)) {
    if (enable_framework_op_trace_) {
      auto trace_path = qnn::DeriveTracePathFromContextModel(ContextPath(graphs[0], ort_api, context_cache_path_cfg_));
      qnn::OpTraceLookup lookup;
      if (qnn::LoadTraceLookupFromFile(trace_path, lookup, logger_))
        qairt_profiling_manager_->MergeOpTraceLookup(std::move(lookup));
    }
    std::vector<int> main_positions;
    RETURN_IF_NOT_OK(qnn::GetMainContextNode(graphs, count, ort_api, main_positions));
    std::unordered_map<std::string, qnn::QairtBackendManager::CachedGraph> cached;
    for (int position : main_positions) {
      const OrtNode* node = nullptr;
      RETURN_IF_NOT_NULL(ort_api.Graph_GetNodes(graphs[position], &node, 1));
      std::vector<char> bytes;
      RETURN_IF_NOT_OK(ReadContext(node, ContextPath(graphs[position], ort_api, context_cache_path_cfg_), *io_dispatch_, bytes));
      std::vector<qnn::QairtBackendManager::CachedGraph> loaded;
      qnn::QairtProfilingManager::Scope scope;
      qnn::QnnEpProfiler* profiler = nullptr;
#if QNN_ORT_EP_PROFILING_API_ENABLED
      profiler = qnn::QnnEpProfiler::Current();
#endif
      RETURN_IF_NOT_OK(qairt_profiling_manager_->Begin(scope, Ort::ConstNode(node).GetName(),
                                                       qnn::OrtProfilingOperation::CONTEXT_LOAD, profiler));
      RETURN_IF_NOT_OK(qairt_backend_manager_->LoadContext(bytes, loaded, scope.profile.get()));
      RETURN_IF_NOT_OK(qairt_profiling_manager_->Complete(scope));
      for (auto& entry : loaded) {
        auto name = entry.info->Name();
        if (!cached.emplace(name, std::move(entry)).second)
          return ort_api.CreateStatus(ORT_INVALID_GRAPH, "Duplicate QAIRT context graph name.");
      }
    }
    for (size_t i = 0; i < count; ++i) {
      const OrtNode* node = nullptr;
      RETURN_IF_NOT_NULL(ort_api.Graph_GetNodes(graphs[i], &node, 1));
      auto name = Ort::ConstNode(node).GetName();
      auto entry = cached.find(name);
      if (entry == cached.end()) return ort_api.CreateStatus(ORT_INVALID_GRAPH, "QAIRT context graph was not found.");
      auto model = std::make_unique<qnn::QnnModel>(qairt_backend_manager_, qairt_profiling_manager_, static_cast<const ApiPtrs&>(*this));
      RETURN_IF_NOT_OK(model->LoadQairtGraph(std::move(entry->second)));
      cached.erase(entry);
      auto overrides = qnn::ParseIoNameOverrides(node);
      qnn::QnnModelContext context{*graphs[i], *fused_nodes[i], logger_, nullptr, nullptr, nullptr, nullptr, &overrides, {}};
      RETURN_IF_NOT_OK(model->SetGraphInputOutputInfo(context));
      RETURN_IF_NOT_OK(model->SetupQnnInputOutput(logger_));
      qnn_models_.emplace(Ort::ConstNode(fused_nodes[i]).GetName(), std::move(model));
      node_compute_infos[i] = new QnnNodeComputeInfo(*this);
    }
    return nullptr;
  }
  if (!onnx_graph_io_names_) return ort_api.CreateStatus(ORT_EP_FAIL, "Missing ONNX graph I/O names.");
  for (size_t i = 0; i < count; ++i) {
    const auto name = Ort::ConstNode(fused_nodes[i]).GetName();
    auto model = std::make_unique<qnn::QnnModel>(qairt_backend_manager_, qairt_profiling_manager_,
                                                 static_cast<const ApiPtrs&>(*this));
    qnn::QnnModelContext context{*graphs[i], *fused_nodes[i], logger_,
                                 &onnx_graph_io_names_->first, &onnx_graph_io_names_->second, &model_settings_,
                                 nullptr, &tensor_name_overrides_, std::string{}};
    if (enable_framework_op_trace_) context.op_trace_output = op_trace_builder_.NewSubgraphSlot();
    if (dump_json_qnn_graph_) context.json_qnn_graph_path =
                                  (std::filesystem::path(json_qnn_graph_dir_) / (name + ".json")).string();
    RETURN_IF_NOT_OK(model->ComposeGraph(context));
    RETURN_IF_NOT_OK(model->FinalizeGraphs(logger_));
    RETURN_IF_NOT_OK(model->SetupQnnInputOutput(logger_));
    qnn_models_.emplace(name, std::move(model));
    node_compute_infos[i] = new QnnNodeComputeInfo(*this);
  }
  if (context_cache_enabled_ && count) {
    std::vector<unsigned char> binary;
    RETURN_IF_NOT_OK(qairt_backend_manager_->ExportContext(binary));
    const auto context_path = FILEPATH_TO_STRING(ContextPath(graphs[0], ort_api, context_cache_path_cfg_));
    RETURN_IF_NOT_OK(qnn::CreateEPContextNodes(fused_nodes, count, ep_context_nodes, ort_api, model_editor_api,
                                               binary.data(), binary.size(), qairt_backend_manager_->GetSdkVersion(), qnn_models_,
                                               context_path, qnn_context_embed_mode_,
                                               0, logger_, false, false, name_, tensor_name_overrides_, false, *io_dispatch_));
  }
  tensor_name_overrides_.clear();
  if (enable_framework_op_trace_ && count) {
    AppendSingleSocTrace();
    WriteFrameworkOpTrace(graphs[0]);
  }
  return nullptr;
} catch (const std::exception& e) {
  return ort_api.CreateStatus(ORT_EP_FAIL, e.what());
} catch (...) {
  return ort_api.CreateStatus(ORT_EP_FAIL, "Unknown QAIRT compilation error.");
}
OrtStatus* QnnEp::OnQairtRunStart(const OrtRunOptions* options) {
  for (const char* key : {kOrtRunOptionsConfigQnnLoraConfig, kOrtRunOptionsConfigQnnPerfMode,
                          kOrtRunOptionsConfigQnnPerfModePostRun, kOrtRunOptionsConfigQnnRpcControlLatency}) {
    if (ort_api.GetRunConfigEntry(options, key))
      return ort_api.CreateStatus(ORT_EP_FAIL, "QAIRT runtime LoRA and power overrides require backend-extension support.");
  }
  return nullptr;
}
#if QNN_ORT_EP_PROFILING_API_ENABLED
OrtStatus* QnnEp::CreateQairtProfiler(OrtEpProfilerImpl** profiler) {
  try {
    auto manager = qairt_profiling_manager_;
    qnn::QnnEpProfiler::BackendCallbacks callbacks{
        [manager](const Ort::Logger& logger) { return manager->Acquire(logger); },
        [manager]() { return manager->Release(); },
        [manager]() { return manager->Lock(); },
        [](qnn::profile::ProfilingInfo&, const Ort::Logger&) { return Ort::Status(); }};
    *profiler = new qnn::QnnEpProfiler(ep_api, ort_api, logger_, std::move(callbacks));
    return nullptr;
  } catch (const std::exception& e) {
    return ort_api.CreateStatus(ORT_FAIL, e.what());
  }
}
#endif
}  // namespace onnxruntime
#endif
