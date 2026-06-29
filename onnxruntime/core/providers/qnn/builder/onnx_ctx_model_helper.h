// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {

namespace qnn {

class QnnModel;
class QnnBackendManager;
using QnnModelLookupTable = std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>>;

static const std::string EPCONTEXT_OP = "EPContext";
static const std::string EP_CONTEXT_TYPE = "ep_context_type";
static const std::string MAIN_CONTEXT = "main_context";
static const std::string EMBED_MODE = "embed_mode";
static const std::string EP_CACHE_CONTEXT = "ep_cache_context";
static const std::string EP_SDK_VER = "ep_sdk_version";
static const std::string PARTITION_NAME = "partition_name";
static const std::string SOURCE = "source";
static const std::string MAX_SIZE = "max_size";
// Serialized internal->external tensor-name overrides produced by offload_graph_io_quantization.
// Persisted into the EPContext node so the cached-context load path can resolve graph I/O by name.
static const std::string IO_NAME_OVERRIDES = "io_name_overrides";

// EP_CONTEXT_TYPES
static const std::string EP_CONTEXT_TYPE_BIN = "bin";
static const std::string EP_CONTEXT_TYPE_DLC = "dlc";

bool GraphHasEpContextNode(const OrtGraph* graph, const OrtApi& ort_api,
                           const std::string& ep_context_type = EP_CONTEXT_TYPE_BIN);

bool GraphHasDlcContextNode(const OrtGraph* graph, const OrtApi& ort_api);

bool IsOrtGraphHasDlcCtxNode(const OrtGraph** graphs, size_t count, const OrtApi& ort_api);

bool IsOrtGraphHasCtxNode(const OrtGraph** graphs, size_t count, const OrtApi& ort_api,
                          const std::string& ep_context_type = EP_CONTEXT_TYPE_BIN);

Ort::Status GetEpContextDlcPath(const OrtGraph** graphs, size_t count, const OrtApi& ort_api,
                                std::string& dlc_path);

Ort::Status GetMainContextNode(const OrtGraph** graphs,
                               size_t count,
                               const OrtApi& ort_api,
                               std::vector<int>& main_context_pos);

// Parses the IO_NAME_OVERRIDES attribute (if present) from an EPContext node into an
// internal->external tensor-name map. Returns an empty map when the attribute is absent
// (offload disabled, or a context model generated before this attribute existed).
std::unordered_map<std::string, std::string> ParseIoNameOverrides(const OrtNode* ep_context_node);

Ort::Status GetEpContextFromMainNode(const OrtNode* main_context_node,
                                     const OrtApi& ort_api,
                                     const std::basic_string<ORTCHAR_T>& ctx_onnx_model_path,
                                     QnnBackendManager* qnn_backend_manager,
                                     QnnModelLookupTable& qnn_models,
                                     int64_t max_spill_fill_size);

Ort::Status TryGetMaxSpillFillSize(const OrtGraph** graphs,
                                   const OrtApi& ort_api,
                                   uint32_t total_context_size,
                                   int64_t& max_spill_fill_size,
                                   std::vector<int>& main_context_pos_list);

Ort::Status LoadQnnCtxFromOnnxGraph(const OrtGraph* graph,
                                    const OrtApi& ort_api,
                                    const std::basic_string<ORTCHAR_T>& ctx_onnx_model_path,
                                    QnnBackendManager* qnn_backend_manager,
                                    QnnModelLookupTable& qnn_models,
                                    const Ort::Logger& logger,
                                    int64_t max_spill_fill_size);

Ort::Status CreateEPContextNodes(const OrtNode** fused_nodes,
                                 size_t count,
                                 OrtNode** ep_context_nodes,
                                 const OrtApi& ort_api,
                                 const OrtModelEditorApi& model_editor_api,
                                 unsigned char* buffer,
                                 uint64_t buffer_size,
                                 const std::string& sdk_build_version,
                                 const QnnModelLookupTable& qnn_models,
                                 const std::basic_string<ORTCHAR_T>& context_model_path,
                                 bool qnn_context_embed_mode,
                                 uint64_t max_spill_fill_buffer_size,
                                 const Ort::Logger& logger,
                                 bool share_ep_contexts,
                                 bool stop_share_ep_contexts,
                                 const std::string& ep_name,
                                 const std::unordered_map<std::string, std::string>& tensor_name_overrides);

}  // namespace qnn
}  // namespace onnxruntime
