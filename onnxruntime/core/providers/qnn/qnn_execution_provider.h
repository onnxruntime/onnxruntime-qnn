// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "HTP/QnnHtpGraph.h"

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/ep_context_io_dispatch.h"
#include "core/providers/qnn/builder/qnn_configs_helper.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing.h"
#include "core/providers/qnn/builder/op_tracing/qnn_op_tracing_types.h"
#include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
#include "core/providers/qnn/cache_compatibility/qnn_cache_compatibility_info.h"
#include "core/providers/qnn/cache_compatibility/qnn_cache_compatibility_manager.h"
#include "core/providers/qnn/qnn_telemetry.h"
#include "core/providers/qnn/rpcmem_library.h"
#include "core/providers/qnn/qnn_node_compute_info_base.h"
#include "core/providers/qnn/genie/genie_api_loader.h"
#include "core/providers/qnn/genie/genie_node.h"
#include "core/providers/qnn/genie/genie_node_compute_info.h"
#include "core/providers/qnn/builder/qnn_htp_power_state_guard.h"

namespace onnxruntime {
class QnnEpFactory;

// Forward declaration for QnnBackendManager, GenieBackendManager
namespace qnn {
class QnnBackendManager;
class GenieBackendManager;
}  // namespace qnn

class QnnEp : public OrtEp, public ApiPtrs {
 public:
  QnnEp(QnnEpFactory& factory,
        const std::string& name,
        const OrtSessionOptions& session_options,
        const OrtLogger* logger);
  ~QnnEp();

  OrtStatus* ValidateCompiledModelCompatibilityInfo(const OrtHardwareDevice* const* devices,
                                                    size_t num_devices,
                                                    const char* compatibility_info,
                                                    OrtCompiledModelCompatibility* model_compatibility) noexcept;
  OrtStatus* GetHardwareDeviceIncompatibilityDetails(const OrtHardwareDevice* hw,
                                                     OrtDeviceEpIncompatibilityDetails* details) noexcept;

  friend struct GenieNodeComputeInfo;
  friend class QnnEpFactory;

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr,
                                                   const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;
  static OrtStatus* ORT_API_CALL GetGenieCapability(OrtEp* this_ptr,
                                                    const OrtGraph* graph,
                                                    OrtEpGraphSupportInfo* graph_support_info);
  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr,
                                             _In_ const OrtGraph** graphs,
                                             _In_ const OrtNode** fused_nodes,
                                             _In_ size_t count,
                                             _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                             _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;
  static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                            _Out_ OrtEpDataLayout* preferred_data_layout) noexcept;
  static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                                  _In_z_ const char* domain,
                                                                  _In_z_ const char* op_type,
                                                                  _In_ OrtEpDataLayout target_data_layout,
                                                                  _Outptr_ int* should_convert) noexcept;
  static OrtStatus* ORT_API_CALL OnRunStartImpl(_In_ OrtEp* this_ptr, _In_ const ::OrtRunOptions* run_options) noexcept;
  static OrtStatus* ORT_API_CALL OnRunEndImpl(_In_ OrtEp* this_ptr,
                                              _In_ const ::OrtRunOptions* run_options,
                                              _In_ bool sync_stream) noexcept;
  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept;
  static OrtStatus* ORT_API_CALL SetDynamicOptionsImpl(_In_ OrtEp* this_ptr,
                                                       _In_reads_(num_options) const char* const* option_keys,
                                                       _In_reads_(num_options) const char* const* option_values,
                                                       _In_ size_t num_options) noexcept;
  static const char* ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(_In_ OrtEp* this_ptr,
                                                                        _In_ const OrtGraph* graph) noexcept;

  OrtStatus* ReloadCompiledContext(const OrtGraph** graphs,
                                   const OrtNode** fused_nodes,
                                   size_t count);

  OrtStatus* GetSupportedNodes(const OrtGraph* graph,
                               const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                               const size_t node_unit_size,
                               std::vector<const OrtNode*>& supported_nodes,
                               std::vector<qnn::UnsupportedNodeInfo>& unsupported_nodes) const;

  OrtStatus* GetMultiSocSupportedNodes(const OrtGraph* graph,
                                       const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                       const size_t node_unit_size,
                                       std::vector<const OrtNode*>& supported_nodes,
                                       std::vector<qnn::UnsupportedNodeInfo>& unsupported_nodes) const;

  void PartitionCtxModel(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info);

  OrtStatus* CompileOnnxModel(const OrtGraph** graphs,
                              const OrtNode** fused_nodes,
                              size_t count,
                              OrtNodeComputeInfo** node_compute_infos,
                              const qnn::HtpGraphConfigs_t& htp_graph_configs,
                              bool collect_subgraph_traces = true);

  OrtStatus* CompileMultiSocOnnxModel(const OrtGraph** graphs,
                                      const OrtNode** fused_nodes,
                                      size_t count,
                                      OrtNodeComputeInfo** node_compute_infos);

  OrtStatus* CompileContextModel(const OrtGraph** graphs,
                                 const OrtNode** fused_nodes,
                                 size_t count,
                                 OrtNodeComputeInfo** node_compute_infos);

  OrtStatus* CompileDlcContextModel(OrtEp* this_ptr,
                                    const OrtGraph** graphs,
                                    const OrtNode** fused_nodes,
                                    size_t count,
                                    OrtNodeComputeInfo** node_compute_infos);

  OrtStatus* CreateEPContextNodes(const OrtGraph* graph,
                                  const OrtNode** fused_nodes,
                                  size_t count,
                                  OrtNode** ep_context_nodes);

  // Helper functions
  void ParsePerSocHtpConfigs();

  // Append a SocTrace: from live backend state (single-SoC), or one per configured SoC.
  void AppendSingleSocTrace();
  void AppendMultiSocTraces();
  // Feed EP-side inputs (model name, backend type, dir) to op_trace_builder_ and write.
  void WriteFrameworkOpTrace(const OrtGraph* primary_graph);

  // Emit a one-shot WARNING when the QNN EP is running on the HTP user-driver (HNRD) fallback path.
  void WarnIfHnrdPathActive();

  bool IsHtpSharedMemoryAllocatorAvailable() const { return rpcmem_library_ != nullptr; }

  void InitQnnHtpGraphConfigs(
      const qnn::HtpGraphConfigs_t& configs,
      qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const;

  std::unique_ptr<qnn::QnnSerializerConfig> InitQnnSerializerConfig() const;

  std::string FormatEPConfigKey(const std::string& key) const {
    return GetProviderOptionPrefix(name_) + key;
  }

  struct QnnNodeComputeInfo : QnnNodeComputeInfoBase {
    explicit QnnNodeComputeInfo(QnnEp& ep);

    static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                   OrtNodeComputeContext* compute_context,
                                                   void** compute_state);
    static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                               OrtKernelContext* kernel_context);
    static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

    QnnEp& ep;
  };

  typedef struct GraphFinalizationInfo {
    std::string model_name;
    std::unique_ptr<qnn::QnnModel> model;
    Ort::Status result;
    size_t graph_idx;

  } GraphFinalizationInfo_t;

  // RAII guard to complete backend setup and release resource during exit.
  // This is expected to be used in GetCapability and Compile for multi-SoC EP context.
  struct ScopedPerSocQnnBackendSetup {
    explicit ScopedPerSocQnnBackendSetup(const QnnEp& ep) : ep_(ep) {};
    ~ScopedPerSocQnnBackendSetup();
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedPerSocQnnBackendSetup);

    Ort::Status Init(size_t per_soc_idx);

   private:
    const QnnEp& ep_;
  };

  // Retrieves per-thread HTP power configurations from run options
  void GetPerThreadHtpPowerConfigs(qnn::PerThreadHtpPowerConfigs_t& per_thread_htp_power_configs,
                                   const ::OrtRunOptions* run_options);

  void CreateHtpPowerConfigId() const;
  // Will return false if htp_power_config_id_ has no value
  bool GetHtpPowerConfigId(uint32_t& htp_power_config_id);

  // htp_power_config_id_ must be created during GetCapability() as it is the only
  // step during setup/initialization in which QNNBackendManager is setup and ready.
  // GetCapability() is a const function, so these options must be mutable
  mutable std::optional<uint32_t> htp_power_config_id_;
  mutable std::mutex config_id_mutex_;

  // const QnnEpFactory& factory_;
  std::string name_;
  const Ort::Logger logger_;
  bool context_cache_enabled_ = false;
  bool share_ep_contexts_ = false;
  int htp_share_resource_optimization_ = -1;
  std::string context_node_name_prefix_ = "";
  std::string context_cache_path_cfg_ = "";
  const OrtSessionOptions& session_options_;

  bool disable_cpu_ep_fallback_ = false;  // True if CPU EP fallback has been disabled for this session.
  bool qnn_context_embed_mode_ = true;
  bool stop_share_ep_contexts_ = false;
  bool prepare_only_ = false;
  bool prepare_and_load_ = false;
  bool enable_spill_fill_buffer_ = false;
  bool enable_file_mapped_weights_ = true;
#if defined(_WIN32)
  uint8_t num_graph_prepare_threads_ = 8;
  qnn::QnnTelemetry::EtwInternalCallback callback_ETWSink_provider_ = nullptr;
#endif

  // Metadef ID generation state
  mutable std::unordered_map<uint64_t, uint64_t> main_graph_hash_;
  mutable std::unordered_map<uint64_t, int> model_metadef_id_;

  // QNN-related.
  std::shared_ptr<qnn::QnnBackendManager> qnn_backend_manager_;
  std::unordered_map<std::string, std::unique_ptr<qnn::QnnModel>> qnn_models_;

  // Configurations for HTP backend.
  uint32_t device_id_{0};
  qnn::HtpPerformanceMode default_htp_performance_mode_{qnn::HtpPerformanceMode::kHtpDefault};
  qnn::HtpPerformanceMode dynamic_htp_performance_mode_{qnn::HtpPerformanceMode::kHtpDefault};
  uint32_t default_rpc_control_latency_ = 0;
  uint32_t default_rpc_polling_time_ = 0;
  uint32_t dynamic_rpc_polling_time_ = 0;
  qnn::ModelSettings model_settings_ = {};
  qnn::HtpGraphConfigs_t htp_graph_configs_;
  qnn::HtpContextConfigs_t htp_context_configs_;

  bool dump_json_qnn_graph_ = false;
  std::string json_qnn_graph_dir_ = "";

  // === Qnn Ep Input Graph ===
  // Dumps the ONNX graph the EP receives in GetCapabilityImpl (compile-time,
  // pre-partition) as a QNN-Netron-schema JSON. Distinct from
  // dump_json_qnn_graph_ above, which dumps the post-compile QNN graph.
  bool dump_qnn_ep_input_graph_ = false;
  std::string dump_qnn_ep_input_graph_dir_;
  // Per-EP counter that disambiguates output filenames. Always appended to
  // the filename so two graphs whose names sanitize to the same string still
  // produce two distinct files. ORT calls GetCapabilityImpl sequentially
  // from a single thread today, so plain size_t is sufficient.
  size_t dump_qnn_ep_input_graph_count_ = 0;

  // === Framework op trace ===
  bool enable_framework_op_trace_ = false;
  std::string framework_op_trace_dir_;
  // Owns trace assembly; the EP only feeds it data during GetCapability/Compile.
  qnn::FrameworkOpTraceBuilder op_trace_builder_;

  bool enable_htp_extended_udma_mode_ = false;

  // HTP Graph Splitting (Graph Program Executor). Requires QAIRT SDK 2.49+ at runtime.
  bool enable_htp_graph_splitting_ = false;

  // === Multi-SoC context binary (a.k.a. Flexible Context Binary) ===
  bool enable_multi_soc_ep_context_ = false;
  // Per-SoC HTP backend configurations. The size of below arrays are enforced to be the same.
  std::vector<QnnHtpDevice_Arch_t> htp_arch_per_soc_;
  std::vector<uint32_t> soc_model_per_soc_;
  std::vector<qnn::HtpGraphConfigs_t> htp_graph_configs_per_soc_;
  // Internal option, could not be speicified by provider options.
  bool enable_htp_ref_weight_sharing_ = true;

  // Whether this is set depends on a session option enabling it and if the RPCMEM dynamic library is available.
  // This is potentially shared with HtpSharedMemoryAllocator which may be returned by CreatePreferredAllocators().
  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library_ = nullptr;

  qnn::QnnAllocatorType qnn_allocator_type_ = qnn::QnnAllocatorType::NONE;

  // Model compatibility.
  std::shared_ptr<qnn::QnnCacheCompatibilityManager> qnn_cache_compatibility_manager_ = nullptr;
  qnn::QnnCompatibilityInfo compatibility_info_ = QNN_COMPATIBILITY_INFO_INIT;
  std::string compatibility_info_string_ = "";

  // One-shot guard so the HNRD fallback warning emitted by WarnIfHnrdPathActive()
  // fires at most once per session, even when GetCapability is invoked multiple
  // times (e.g. initial pass + EPContext re-pass).
  bool hnrd_warning_emitted_ = false;

  // A flag to indicate whether current GetCapability call is after layout transform, flipped to true once exiting the first call.
  bool is_post_layout_transform_ = false;

  // Transient state captured in GetCapability() and consumed in Compile().
  // Only one model is ever in-flight per EP instance (one EP per session).
  mutable std::unordered_map<std::string, std::string> tensor_name_overrides_;
  // ONNX graph I/O names in declaration order: {input_names, output_names}.
  mutable std::optional<std::pair<std::vector<std::string>, std::vector<std::string>>> onnx_graph_io_names_;

  // Genie pathway-specific variables
  std::shared_ptr<qnn::GenieBackendManager> genie_backend_manager_;
  mutable std::shared_ptr<GenieApiLoader> genie_api_loader_;
  GenieLog_Level_t genie_log_level_ = GENIE_LOG_LEVEL_ERROR;
  mutable std::atomic<uint64_t> genie_kv_cache_rewind_{1};

  // Owns the App-provided EPContext read/write callbacks.
  // On pre-v28 ORT it degrades to a no-op stub with HasReadCallback() / HasWriteCallback() returning false.
  std::unique_ptr<qnn::EpContextIoDispatch> io_dispatch_;
};

}  // namespace onnxruntime
