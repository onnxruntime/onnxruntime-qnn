// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "qnn_execution_provider.h"

#include <filesystem>
#include <optional>
#include <string_view>
#include <unordered_set>

#include "core/common/string_utils.h"
#include "core/providers/qnn-abi/qnn_ep_factory.h"
// #include "core/providers/qnn/builder/onnx_ctx_model_helper.h"
// #include "core/providers/qnn/builder/op_builder_factory.h"
// #include "core/providers/qnn/builder/qnn_def.h"
// #include "core/providers/qnn/builder/qnn_model_wrapper.h"
// #include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
// #include "core/providers/qnn/builder/qnn_utils.h"
// #include "core/providers/qnn/ort_api.h"
// #include "core/providers/qnn/qnn_allocator.h"
// #include "core/providers/qnn/qnn_telemetry.h"
// #include "core/providers/qnn/rpcmem_library.h"
// #include "core/providers/qnn/shared_context.h"

namespace onnxruntime {


// Declaration of the external C function CreateEpFactories from the QNN ABI library
extern "C" OrtStatus* CreateEpFactories(const char* registration_name,
                                        const OrtApiBase* ort_api_base,
                                        const OrtLogger* default_logger,
                                        OrtEpFactory** factories,
                                        size_t max_factories,
                                        size_t* p_num_ep_factories);

extern "C" OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);

constexpr const char* QNN = "QNN";

static std::string MakeSharedLibraryPath(std::string_view name) {
#if defined(_WIN32)
  return MakeString(name, ".dll");
#else
  return MakeString("lib", name, ".so");
#endif
}

const std::string kDefaultCpuBackendPath = MakeSharedLibraryPath("QnnCpu");
const std::string kDefaultGpuBackendPath = MakeSharedLibraryPath("QnnGpu");
const std::string kDefaultHtpBackendPath = MakeSharedLibraryPath("QnnHtp");
const std::string kDefaultSaverBackendPath = MakeSharedLibraryPath("QnnSaver");
const std::string kDefaultIrBackendPath = MakeSharedLibraryPath("QnnIr");

QNNExecutionProvider::QNNExecutionProvider(const ProviderOptions& provider_options_map,
                                           const ConfigOptions* config_options,
                                           const OrtSessionOptions* session_options,
                                           const OrtLogger* session_logger)
    : IExecutionProvider{onnxruntime::kQnnExecutionProvider} {

    provider_options_map; config_options;

    size_t num_factories = 0;

    onnxruntime::CreateEpFactories(kQnnExecutionProvider, OrtGetApiBase(),
                                   session_logger,
                                   factories.data(), factories.size(), &num_factories);

    factories[0]->CreateEp(factories[0], nullptr, nullptr, 0, session_options, session_logger, &qnn_external_ep);

    this->SetLogger(reinterpret_cast<const logging::Logger*>(session_logger));
    external_logger_ = session_logger;
}

QNNExecutionProvider::~QNNExecutionProvider() {
  onnxruntime::ReleaseEpFactory(factories[0]);
}

// Logs information about the supported/unsupported nodes.
// static void LogNodeSupport(const logging::Logger& logger,
//                            logging::Severity log_severity,
//                            logging::DataType log_data_type,
//                            const onnxruntime::CodeLocation& call_site,
//                            const qnn::IQnnNodeGroup& qnn_node_group,
//                            Status support_status) {
//   if (!logger.OutputIsEnabled(log_severity, log_data_type)) {
//     return;
//   }

//   size_t num_nodes = 0;
//   std::ostringstream oss;
//   for (const NodeUnit* node_unit : qnn_node_group.GetNodeUnits()) {
//     for (const Node* node : node_unit->GetAllNodesInGroup()) {
//       oss << "\tOperator type: " << node->OpType()
//           << " Node name: " << node->Name()
//           << " Node index: " << node->Index() << std::endl;
//       num_nodes += 1;
//     }
//   }
//   if (!support_status.IsOK()) {
//     oss << "\tREASON : " << support_status.ErrorMessage() << std::endl;
//   }

//   auto log_capture = Factory<logging::Capture>::Create(logger, log_severity,
//                                                        logging::Category::onnxruntime,
//                                                        log_data_type, call_site);
//   log_capture->Stream()
//       << (support_status.IsOK() ? "Validation PASSED " : "Validation FAILED ") << "for " << num_nodes
//       << " nodes in " << qnn_node_group.Type() << " (" << qnn_node_group.GetTargetNodeUnit()->OpType() << ") :"
//       << std::endl
//       << oss.str();
// }

// std::unordered_set<const Node*>
// QNNExecutionProvider::GetSupportedNodes(const GraphViewer& graph_viewer,
//                                         const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
//                                         const size_t node_unit_size,
//                                         const logging::Logger& logger) const {
//   std::unordered_set<const Node*> supported_nodes{};
//   logger;
// node_unit_size;
// node_unit_map;
// graph_viewer;

//   // // Util function that initializes a table that maps a graph input or output name to its index.
//   // auto init_input_output_index_map = [](std::unordered_map<std::string, size_t>& index_map,
//   //                                       const std::vector<const NodeArg*>& node_args) {
//   //   const size_t num_args = node_args.size();
//   //   for (size_t i = 0; i < num_args; i++) {
//   //     index_map.emplace(node_args[i]->Name(), i);
//   //   }
//   // };

//   // std::unordered_map<std::string, size_t> model_input_index_map;
//   // init_input_output_index_map(model_input_index_map, graph_viewer.GetInputs());  // GetInputs excludes initializers.

//   // std::unordered_map<std::string, size_t> model_output_index_map;
//   // init_input_output_index_map(model_output_index_map, graph_viewer.GetOutputs());

//   // auto qnn_model_wrapper = qnn::QnnModelWrapper(graph_viewer, logger,
//   //                                               qnn_backend_manager_->GetQnnInterface(),
//   //                                               qnn_backend_manager_->GetQnnBackendHandle(),
//   //                                               model_input_index_map,
//   //                                               model_output_index_map,
//   //                                               qnn_backend_manager_->GetQnnBackendType(),
//   //                                               model_settings_);

//   // std::vector<std::unique_ptr<qnn::IQnnNodeGroup>> qnn_node_groups;
//   // qnn_node_groups.reserve(node_unit_size);

//   // if (Status status = qnn::GetQnnNodeGroups(qnn_node_groups, qnn_model_wrapper,
//   //                                           node_unit_map, node_unit_size, logger);
//   //     !status.IsOK()) {
//   //   LOGS(logger, ERROR) << status.ErrorMessage();
//   //   return {};
//   // }

//   // for (const std::unique_ptr<qnn::IQnnNodeGroup>& qnn_node_group : qnn_node_groups) {
//   //   Status status = qnn_node_group->IsSupported(qnn_model_wrapper, logger);
//   //   const bool supported = status.IsOK();

//   //   constexpr auto log_severity = logging::Severity::kINFO;
//   //   constexpr auto log_data_type = logging::DataType::SYSTEM;
//   //   // if (logger.OutputIsEnabled(log_severity, log_data_type)) {
//   //   //   LogNodeSupport(logger, log_severity, log_data_type, ORT_WHERE, *qnn_node_group, status);
//   //   // }

//   //   if (supported) {
//   //     for (const NodeUnit* node_unit : qnn_node_group->GetNodeUnits()) {
//   //       for (const Node* node : node_unit->GetAllNodesInGroup()) {
//   //         supported_nodes.insert(node);
//   //       }
//   //     }
//   //   }
//   // }

//   return supported_nodes;
// }

// static bool EpSharedContextsHasAllGraphs(const onnxruntime::GraphViewer& graph_viewer,
//                                          const logging::Logger& logger) {
//                                           graph_viewer;logger;
//   // for (const auto& node : graph_viewer.Nodes()) {
//   //   NodeAttrHelper node_helper(node);
//   //   std::string cache_source = node_helper.Get(qnn::SOURCE, "");

//   //   std::transform(cache_source.begin(),
//   //                  cache_source.end(),
//   //                  cache_source.begin(),
//   //                  [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

//   //   if (qnn::EPCONTEXT_OP == node.OpType() && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
//   //     const std::string& graph_name = node.Name();
//   //     // bool has_shared_qnn_model = SharedContext::GetInstance().HasQnnModel(graph_name);
//   //     // if (!has_shared_qnn_model) {
//   //     //   LOGS(logger, VERBOSE) << "Graph: " << graph_name << " from EpContext node not found from shared EP contexts.";
//   //     //   return false;
//   //     // }
//   //   }
//   // }

//   return true;
// }

// static bool EpSharedContextsHasAllGraphs(const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
//                                          const logging::Logger& logger) {
//   for (auto fused_node_and_graph : fused_nodes_and_graphs) {
//     const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);
//     const Node& ep_context_node = *graph_viewer.Nodes().begin();
//     NodeAttrHelper node_helper(ep_context_node);
//     std::string cache_source = node_helper.Get(qnn::SOURCE, "");

//     const std::string& graph_name = ep_context_node.Name();
//     // bool has_shared_qnn_model = SharedContext::GetInstance().HasQnnModel(graph_name);
//     // if (!has_shared_qnn_model) {
//     //   LOGS(logger, VERBOSE) << "Graph: " << graph_name << " from EpContext node not found from shared EP contexts.";
//     //   return false;
//     // }
//   }

//   return true;
// }

// static void GetMainEPCtxNodes(const onnxruntime::GraphViewer& graph_viewer,
//                               std::unordered_set<const Node*>& ep_context_nodes,
//                               const logging::Logger& logger) {
//   for (const auto& node : graph_viewer.Nodes()) {
//     NodeAttrHelper node_helper(node);
//     bool is_main_context = node_helper.Get(qnn::MAIN_CONTEXT, static_cast<int64_t>(0));
//     std::string cache_source = node_helper.Get(qnn::SOURCE, "");

//     std::transform(cache_source.begin(),
//                    cache_source.end(),
//                    cache_source.begin(),
//                    [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

//     if (is_main_context && qnn::EPCONTEXT_OP == node.OpType() && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
//       LOGS(logger, VERBOSE) << "EPContext Node found: [1] index: [" << node.Index()
//                             << "] name: [" << node.Name();
//       ep_context_nodes.insert(&node);
//     }
//   }
// }

// For model with EPContext, filter in EPContext nodes only, and make sure each partition only has one single EPContext node
// static void PartitionCtxModel(const onnxruntime::GraphViewer& graph_viewer,
//                               const size_t num_nodes_in_graph,
//                               std::vector<std::unique_ptr<ComputeCapability>>& result,
//                               const std::function<std::string()>& gen_metadef_name,
//                               const logging::Logger& logger) {
//   std::unordered_set<const Node*> supported_nodes{};
//   std::vector<std::vector<const Node*>> supported_groups{};

//   for (const auto& node : graph_viewer.Nodes()) {
//     NodeAttrHelper node_helper(node);
//     std::string cache_source = node_helper.Get(qnn::SOURCE, "");

//     std::transform(cache_source.begin(),
//                    cache_source.end(),
//                    cache_source.begin(),
//                    [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

//     if (qnn::EPCONTEXT_OP == node.OpType() && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
//       LOGS(logger, VERBOSE) << "Node supported: [1] index: [" << node.Index()
//                             << "] name: [" << node.Name()
//                             << "] Operator type: [EPContext"
//                             << "] index: [" << node.Index() << "]";
//       supported_nodes.insert(&node);

//       std::vector<const Node*> supported_group{&node};
//       supported_groups.emplace_back(std::move(supported_group));
//     }
//   }

//   result.reserve(supported_groups.size());

//   std::transform(
//       supported_groups.begin(), supported_groups.end(),
//       std::back_inserter(result),
//       [&](const auto& supported_partition) {
//         return utils::MakeComputeCapability(graph_viewer, supported_partition, gen_metadef_name, QNN,
//                                             /*drop_constant_initializers*/ false);  // TODO: could this be set to true?
//       });

//   const size_t num_of_partitions = result.size();
//   const auto summary_msg = MakeString("Number of partitions supported by QNN EP: ", num_of_partitions,
//                                       ", number of nodes in the graph: ", num_nodes_in_graph,
//                                       ", number of nodes supported by QNN: ", num_of_partitions);
//   LOGS(logger, INFO) << summary_msg;

//   return;
// }

// // Figure out the context cache Onnx file path to decide the folder location
// static void GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
//                                         const onnxruntime::PathString& model_path_string,
//                                         onnxruntime::PathString& context_model_path) {
//   // always try the path set by user first, it's the only way to set it if load model from memory
//   if (!user_context_cache_path.empty()) {
//     context_model_path = ToPathString(user_context_cache_path);
//   } else if (!model_path_string.empty()) {  // model loaded from file
//     context_model_path = model_path_string;
//   }
// }

std::vector<std::unique_ptr<ComputeCapability>>
QNNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                    const IKernelLookup& /*kernel_lookup*/,
                                    const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                    IResourceAccountant* /* resource_accountant */) const {

  std::cout << "GetCapability in old: " << std::endl;
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Call the external QNN EP's GetCapability method
  #if 0
    qnn_external_ep->GetCapability(qnn_external_ep, utils::GraphViewerToOrtGraph(graph_viewer), utils::GraphViewerToOrtEpGraphSupportInfo(graph_viewer));
  #else
    // step-1
    OrtGraph* ort_graph = utils::GraphViewerToOrtGraph(graph_viewer);

    //step-2
    // GetSupportedNode = use the logic between lines 1155 - 1253 in GetCapabilityImpl in qnn_ep.cc file to get supported nodes.

    //step3:
    // use the logic between lines 1010 - 1062 in QNNExecutionProvider::GetCapability() in file: qnn_execution_provider.cc


  #endif
  // Create a dummy ComputeCapability to prevent graph_partitioner from failing
  // This allows the graph partitioner to continue processing without crashing
  if (graph_viewer.NumberOfNodes() > 0) {
    // Get the first node as a dummy capability
    const auto& nodes = graph_viewer.GetNodesInTopologicalOrder();
    if (!nodes.empty()) {
      auto indexed_sub_graph = IndexedSubGraph::Create();
      indexed_sub_graph->Nodes().push_back(nodes[0]);

      auto compute_capability = ComputeCapability::Create(std::move(indexed_sub_graph));
      result.push_back(std::move(compute_capability));
    }
  }

  return result;
}

DataLayout QNNExecutionProvider::GetPreferredLayout() const {
  return DataLayout::NHWC;
}

// std::optional<bool> QNNExecutionProvider::ShouldConvertDataLayoutForOp(std::string_view node_domain,
//                                                                        std::string_view node_op_type,
//                                                                        DataLayout target_data_layout) const {
//   if (target_data_layout != DataLayout::NHWC) {
//     return std::nullopt;
//   }

//   if (node_domain == kOnnxDomain && node_op_type == "Upsample") {
//     // Upsample is translated to QNN's Resize, which requires the NHWC layout for processing.
//     return true;
//   }

//   return std::nullopt;
// }

// Status QNNExecutionProvider::CreateComputeFunc(std::vector<NodeComputeInfo>& node_compute_funcs,
//                                                const logging::Logger& logger) {
//                                                 node_compute_funcs;logger;
  // NodeComputeInfo compute_info;
  // compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
  //   LOGS(logger, VERBOSE) << "compute_info.create_state_func context->node_name: " << context->node_name;
  //   *state = qnn_models_[context->node_name].get();
  //   return 0;
  // };

  // compute_info.release_state_func = [](FunctionState state) {
  //   // the 'state' is a qnn::QnnModel managed by unique_ptr
  //   ORT_UNUSED_PARAMETER(state);
  // };

  // compute_info.compute_func = [&logger](FunctionState state, const OrtApi*, OrtKernelContext* context) {
  //   Ort::KernelContext ctx(context);
  //   qnn::QnnModel* model = reinterpret_cast<qnn::QnnModel*>(state);
  //   Status result = model->ExecuteGraph(ctx, logger);
  //   return result;
  // };

  // node_compute_funcs.push_back(compute_info);

//   return Status::OK();
// }

// void QNNExecutionProvider::InitQnnHtpGraphConfigs(qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t>& configs_builder) const {
//   if (qnn_backend_manager_->GetQnnBackendType() == qnn::QnnBackendType::HTP) {
//     if (htp_graph_finalization_opt_mode_ != qnn::HtpGraphFinalizationOptimizationMode::kDefault) {
//       gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config = configs_builder.PushCustomConfig();
//       htp_graph_opt_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
//       htp_graph_opt_config->optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
//       htp_graph_opt_config->optimizationOption.floatValue = static_cast<float>(htp_graph_finalization_opt_mode_);

//       gsl::not_null<QnnGraph_Config_t*> graph_opt_config = configs_builder.PushConfig();
//       graph_opt_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
//       graph_opt_config->customConfig = htp_graph_opt_config;
//     }

//     if (vtcm_size_in_mb_ > 0) {
//       gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_opt_config_vtcm = configs_builder.PushCustomConfig();
//       htp_graph_opt_config_vtcm->option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
//       htp_graph_opt_config_vtcm->vtcmSizeInMB = static_cast<uint32_t>(vtcm_size_in_mb_);

//       gsl::not_null<QnnGraph_Config_t*> graph_opt_config_vtcm = configs_builder.PushConfig();
//       graph_opt_config_vtcm->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
//       graph_opt_config_vtcm->customConfig = htp_graph_opt_config_vtcm;
//     }

//     if (enable_HTP_FP16_precision_) {
//       gsl::not_null<QnnHtpGraph_CustomConfig_t*> htp_graph_precision_config = configs_builder.PushCustomConfig();
//       htp_graph_precision_config->option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
//       htp_graph_precision_config->precision = QNN_PRECISION_FLOAT16;

//       gsl::not_null<QnnGraph_Config_t*> graph_precision_config = configs_builder.PushConfig();
//       graph_precision_config->option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
//       graph_precision_config->customConfig = htp_graph_precision_config;
//     }
//   }
// }

// Status QNNExecutionProvider::CompileFromOrtGraph(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
//                                                  std::vector<NodeComputeInfo>& node_compute_funcs,
//                                                  const logging::Logger& logger) {
//                                                   fused_nodes_and_graphs;
//                                                   node_compute_funcs;
//                                                   logger;
  // for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
  //   Node& fused_node = fused_node_and_graph.fused_node;
  //   const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

  //   std::unique_ptr<qnn::QnnModel> qnn_model = std::make_unique<qnn::QnnModel>(qnn_backend_manager_.get());

  //   std::vector<const QnnGraph_Config_t*> all_graph_configs;

  //   qnn::QnnConfigsBuilder<QnnGraph_Config_t, QnnHtpGraph_CustomConfig_t> htp_graph_configs_builder(QNN_GRAPH_CONFIG_INIT,
  //                                                                                                   QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
  //   InitQnnHtpGraphConfigs(htp_graph_configs_builder);

  //   const QnnGraph_Config_t** htp_configs = htp_graph_configs_builder.GetQnnConfigs();
  //   if (htp_configs) {
  //     // Reserve enough for configs + nullptr
  //     all_graph_configs.reserve(htp_graph_configs_builder.GetSize() + 1);
  //     for (const QnnGraph_Config_t** config = htp_configs; *config; ++config) {
  //       all_graph_configs.push_back(*config);
  //     }
  //   }

  //   qnn::QnnSerializerConfig* qnn_serializer_config = qnn_backend_manager_->GetQnnSerializerConfig();
  //   if (qnn_serializer_config) {
  //     // We don't bother reserving here to keep the API simpler. Also note that if we're here,
  //     // we're likely debugging and not waiting for inference.
  //     qnn_serializer_config->SetGraphName(fused_node.Name());
  //     const QnnGraph_Config_t** serializer_configs = qnn_serializer_config->Configure();
  //     if (serializer_configs) {
  //       for (const QnnGraph_Config_t** config = serializer_configs; *config; ++config) {
  //         all_graph_configs.push_back(*config);
  //       }
  //     }
  //   }

  //   const QnnGraph_Config_t** all_graph_configs_ptr = nullptr;
  //   if (!all_graph_configs.empty()) {
  //     all_graph_configs.push_back(nullptr);
  //     all_graph_configs_ptr = all_graph_configs.data();
  //   }

  //   std::string json_graph_filepath;

  //   if (dump_json_qnn_graph_) {
  //     namespace fs = std::filesystem;
  //     fs::path path = fs::path(json_qnn_graph_dir_) / fs::path(fused_node.Name() + ".json");
  //     json_graph_filepath = path.string();
  //   }

  //   ORT_RETURN_IF_ERROR(qnn_model->ComposeGraph(graph_viewer, fused_node, model_settings_, logger,
  //                                               all_graph_configs_ptr, json_graph_filepath));
  //   ORT_RETURN_IF_ERROR(qnn_model->FinalizeGraphs(logger));
  //   ORT_RETURN_IF_ERROR(qnn_model->SetupQnnInputOutput(logger));

  //   LOGS(logger, VERBOSE) << "fused node name: " << fused_node.Name();
  //   qnn_models_.emplace(fused_node.Name(), std::move(qnn_model));

  //   ORT_RETURN_IF_ERROR(CreateComputeFunc(node_compute_funcs, logger));
  // }
//   return Status::OK();
// }

Status QNNExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                     std::vector<NodeComputeInfo>& node_compute_funcs) {
  // Check if there are any nodes to compile
  if (fused_nodes_and_graphs.empty()) {
    return Status::OK();
  }

  // Verify that qnn_external_ep is valid
  if (qnn_external_ep == nullptr || qnn_external_ep->Compile == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN external EP or Compile function is not available");
  }

  const size_t count = fused_nodes_and_graphs.size();
  const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  // Prepare data structures for conversion
  std::vector<const OrtGraph*> ort_graphs;
  std::vector<const OrtNode*> ort_fused_nodes;
  std::vector<OrtNodeComputeInfo*> ort_node_compute_infos(count, nullptr);
  std::vector<OrtNode*> ort_ep_context_nodes(count, nullptr);

  ort_graphs.reserve(count);
  ort_fused_nodes.reserve(count);

  // Convert FusedNodeAndGraph to OrtGraph and OrtNode
  // We need to keep the OrtGraph instances alive for the duration of the Compile call
  std::vector<OrtGraph*> owned_ort_graphs;
  owned_ort_graphs.reserve(count);

  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    // Use the utility function to convert GraphViewer to OrtGraph
    // This creates an EpGraph internally and returns it as OrtGraph*
    OrtGraph* ort_graph = utils::GraphViewerToOrtGraph(fused_node_and_graph.filtered_graph);
    ort_graphs.push_back(ort_graph);
    owned_ort_graphs.push_back(ort_graph);

    // For the fused node, we need to get it from the OrtGraph
    // The fused node should be retrievable from the graph's nodes
    // Get the node index from the original fused node (use .get() to unwrap reference_wrapper)
    const Node& fused_node_ref = fused_node_and_graph.fused_node.get();
    NodeIndex node_index = fused_node_ref.Index();

    // Get the number of nodes in the graph
    size_t num_nodes = 0;
    OrtStatus* num_nodes_status = ort_api->Graph_GetNumNodes(ort_graph, &num_nodes);
    if (num_nodes_status != nullptr) {
      const char* error_msg = ort_api->GetErrorMessage(num_nodes_status);
      ort_api->ReleaseStatus(num_nodes_status);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get number of nodes: ", error_msg ? error_msg : "Unknown error");
    }

    // Get all nodes from the graph
    std::vector<const OrtNode*> graph_nodes(num_nodes);
    OrtStatus* get_nodes_status = ort_api->Graph_GetNodes(ort_graph, graph_nodes.data(), graph_nodes.size());
    if (get_nodes_status != nullptr) {
      const char* error_msg = ort_api->GetErrorMessage(get_nodes_status);
      ort_api->ReleaseStatus(get_nodes_status);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get graph nodes: ", error_msg ? error_msg : "Unknown error");
    }

    // Find the node with matching index
    const OrtNode* ort_fused_node = nullptr;
    for (const OrtNode* graph_node : graph_nodes) {
      size_t node_id = 0;
      OrtStatus* get_id_status = ort_api->Node_GetId(graph_node, &node_id);
      if (get_id_status != nullptr) {
        ort_api->ReleaseStatus(get_id_status);
        continue;
      }

      if (node_id == node_index) {
        ort_fused_node = graph_node;
        break;
      }
    }

    if (ort_fused_node == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to find fused node with index ", node_index, " in OrtGraph");
    }

    ort_fused_nodes.push_back(ort_fused_node);
  }

  // Call the external QNN EP's CompileImpl
  OrtStatus* compile_status = qnn_external_ep->Compile(
      qnn_external_ep,
      ort_graphs.data(),
      ort_fused_nodes.data(),
      count,
      ort_node_compute_infos.data(),
      ort_ep_context_nodes.data());

  // Check compilation status
  if (compile_status != nullptr) {
    const char* error_message = ort_api->GetErrorMessage(compile_status);
    std::string error_str = error_message ? error_message : "Unknown error during QNN EP compilation";
    ort_api->ReleaseStatus(compile_status);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_str);
  }

  // Convert OrtNodeComputeInfo back to NodeComputeInfo
  node_compute_funcs.clear();
  node_compute_funcs.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    OrtNodeComputeInfo* ort_compute_info = ort_node_compute_infos[i];

    if (ort_compute_info == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create compute info for node at index ", i);
    }

    NodeComputeInfo compute_info;

    // Set create_state_func - creates state from OrtNodeComputeInfo
    compute_info.create_state_func = [ort_compute_info, ort_api, this](
                                         ComputeContext* context, FunctionState* state) -> int {
      // Convert ComputeContext to OrtNodeComputeContext
      OrtNodeComputeContext* ort_context = reinterpret_cast<OrtNodeComputeContext*>(context);

      void* compute_state = nullptr;
      OrtStatus* status = ort_compute_info->CreateState(ort_compute_info, ort_context, &compute_state);

      if (status != nullptr) {
        const char* error_msg = ort_api->GetErrorMessage(status);
        LOGS(*GetLogger(), ERROR) << "Failed to create compute state: " << (error_msg ? error_msg : "Unknown error");
        ort_api->ReleaseStatus(status);
        return -1;
      }

      *state = compute_state;
      return 0;
    };

    // Set release_state_func - releases the state
    compute_info.release_state_func = [ort_compute_info](FunctionState state) {
      if (ort_compute_info != nullptr && ort_compute_info->ReleaseState != nullptr) {
        ort_compute_info->ReleaseState(ort_compute_info, state);
      }
    };

    // Set compute_func - executes the computation
    compute_info.compute_func = [ort_compute_info, ort_api](
                                    FunctionState state, const OrtApi* /*api*/, OrtKernelContext* context) -> Status {
      OrtStatus* status = ort_compute_info->Compute(ort_compute_info, state, context);

      if (status != nullptr) {
        const char* error_msg = ort_api->GetErrorMessage(status);
        std::string error_str = error_msg ? error_msg : "Unknown error during compute";
        ort_api->ReleaseStatus(status);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_str);
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

// const InlinedVector<const Node*> QNNExecutionProvider::GetEpContextNodes() const {
//   InlinedVector<const Node*> ep_context_nodes;
//   if (qnn_ep_context_model_) {
//     const auto& graph = qnn_ep_context_model_->MainGraph();
//     for (gsl::not_null<const Node*> node : Graph__Nodes(graph)) {
//       ep_context_nodes.push_back(graph.GetNode(node->Index()));
//     }
//   }

//   return ep_context_nodes;
// }

// QNNExecutionProvider::PerThreadContext::PerThreadContext(qnn::QnnBackendManager* qnn_backend_manager,
//                                                          uint32_t device_id,
//                                                          uint32_t core_id,
//                                                          qnn::HtpPerformanceMode default_htp_performance_mode,
//                                                          uint32_t default_rpc_control_latency,
//                                                          uint32_t default_rpc_polling_time)
//     : qnn_backend_manager_(qnn_backend_manager) {
//   Status rt = qnn_backend_manager_->CreateHtpPowerCfgId(device_id, core_id, htp_power_config_id_);
//   is_htp_power_config_id_valid_ = rt.IsOK();
//   // default_htp_performance_mode and default_rpc_control_latency are from QNN EP option.
//   // set it once only for each thread as default so user don't need to set it for every session run
//   if (is_htp_power_config_id_valid_) {
//     if (qnn::HtpPerformanceMode::kHtpDefault != default_htp_performance_mode) {
//       ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->SetHtpPowerConfig(htp_power_config_id_,
//                                                                       default_htp_performance_mode));
//     }
//     if (default_rpc_control_latency > 0 || default_rpc_polling_time > 0) {
//       ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->SetRpcPowerConfigs(htp_power_config_id_,
//                                                                        default_rpc_control_latency,
//                                                                        default_rpc_polling_time));
//     }
//   }
// }

// QNNExecutionProvider::PerThreadContext::~PerThreadContext() {
//   if (is_htp_power_config_id_valid_) {
//     ORT_IGNORE_RETURN_VALUE(qnn_backend_manager_->DestroyHTPPowerConfigID(htp_power_config_id_));
//   }
// }

// QNNExecutionProvider::PerThreadContext& QNNExecutionProvider::GetPerThreadContext() const {
//   const auto& per_thread_context_cache = PerThreadContextCache();

//   // try to use cached context
//   auto cached_context_it = per_thread_context_cache->find(this);
//   if (cached_context_it != per_thread_context_cache->end()) {
//     auto cached_context = cached_context_it->second.lock();
//     ORT_ENFORCE(cached_context);
//     return *cached_context;
//   }

//   // get context and update cache
//   std::shared_ptr<PerThreadContext> context;
//   {
//     std::lock_guard<std::mutex> lock(context_state_.mutex);

//     // get or create a context
//     if (context_state_.retired_context_pool.empty()) {
//       uint32_t core_id = 0;
//       context = std::make_shared<PerThreadContext>(qnn_backend_manager_.get(), device_id_, core_id,
//                                                    default_htp_performance_mode_, default_rpc_control_latency_,
//                                                    default_rpc_polling_time_);
//     } else {
//       context = context_state_.retired_context_pool.back();
//       context_state_.retired_context_pool.pop_back();
//     }

//     // insert into active_contexts, should not already be present
//     const auto active_contexts_insert_result = context_state_.active_contexts.insert(context);
//     ORT_ENFORCE(active_contexts_insert_result.second);

//     // insert into caches_to_update_on_destruction, may already be present
//     ORT_IGNORE_RETURN_VALUE(context_state_.caches_to_update_on_destruction.insert(per_thread_context_cache));
//   }

//   per_thread_context_cache->insert(std::make_pair(this, context));

//   return *context;
// }

// void QNNExecutionProvider::ReleasePerThreadContext() const {
//   const auto& per_thread_context_cache = PerThreadContextCache();

//   auto cached_context_it = per_thread_context_cache->find(this);
//   ORT_ENFORCE(cached_context_it != per_thread_context_cache->end());
//   auto cached_context = cached_context_it->second.lock();
//   ORT_ENFORCE(cached_context);

//   {
//     std::lock_guard<std::mutex> lock(context_state_.mutex);
//     context_state_.active_contexts.erase(cached_context);
//     context_state_.retired_context_pool.push_back(cached_context);
//   }

//   per_thread_context_cache->erase(cached_context_it);
// }

// static bool TryGetConfigEntry(const ConfigOptions& config_options, const std::string& key, std::string& value) {
//   std::optional<std::string> new_value = config_options.GetConfigEntry(key);
//   if (!new_value.has_value()) {
//     return false;
//   }

//   value = *new_value;
//   return true;
// }

Status QNNExecutionProvider::OnRunStart(const onnxruntime::RunOptions& run_options) {
  // Verify that qnn_external_ep is valid
  if (qnn_external_ep == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN external EP is not available");
  }

  // Verify that OnRunStart function pointer is valid
  if (qnn_external_ep->OnRunStart == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN external EP OnRunStart function is not available");
  }

  // Convert onnxruntime::RunOptions to ::OrtRunOptions*
  // In execution_provider.h: using RunOptions = ::OrtRunOptions;
  // Cast to the global namespace type explicitly
  const ::OrtRunOptions& ort_run_options_ref = static_cast<const ::OrtRunOptions&>(run_options);
  const ::OrtRunOptions* ort_run_options = &ort_run_options_ref;

  // Call the external QNN EP's OnRunStart implementation
  OrtStatus* status = qnn_external_ep->OnRunStart(qnn_external_ep, ort_run_options);

  // Handle the returned status
  if (status != nullptr) {
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const char* error_message = ort_api->GetErrorMessage(status);
    std::string error_str = error_message ? error_message : "Unknown error during QNN EP OnRunStart";
    ort_api->ReleaseStatus(status);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_str);
  }

  return Status::OK();
}

Status QNNExecutionProvider::OnRunEnd(bool sync_stream, const onnxruntime::RunOptions& run_options) {
  // Verify that qnn_external_ep is valid
  if (qnn_external_ep == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN external EP is not available");
  }

  // Verify that OnRunEnd function pointer is valid
  if (qnn_external_ep->OnRunEnd == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN external EP OnRunEnd function is not available");
  }

  // Convert onnxruntime::RunOptions to ::OrtRunOptions*
  // In execution_provider.h: using RunOptions = ::OrtRunOptions;
  // Cast to the global namespace type explicitly
  const ::OrtRunOptions& ort_run_options_ref = static_cast<const ::OrtRunOptions&>(run_options);
  const ::OrtRunOptions* ort_run_options = &ort_run_options_ref;

  // Call the external QNN EP's OnRunEnd implementation
  // Note: Parameter order is different - OnRunEndImpl expects (this_ptr, run_options, sync_stream)
  OrtStatus* status = qnn_external_ep->OnRunEnd(qnn_external_ep, ort_run_options, sync_stream);

  // Handle the returned status
  if (status != nullptr) {
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const char* error_message = ort_api->GetErrorMessage(status);
    std::string error_str = error_message ? error_message : "Unknown error during QNN EP OnRunEnd";
    ort_api->ReleaseStatus(status);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_str);
  }

  return Status::OK();
}

std::vector<AllocatorPtr> QNNExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators{};

  // if (IsHtpSharedMemoryAllocatorAvailable()) {
  //   LOGS_DEFAULT(INFO) << "Creating HtpSharedMemoryAllocator.";

  //   AllocatorFactory rpcmem_allocator_factory = [this](OrtDevice::DeviceId) {
  //     return std::make_unique<qnn::HtpSharedMemoryAllocator>(rpcmem_library_);
  //   };

  //   AllocatorCreationInfo rpcmem_allocator_creation_info{rpcmem_allocator_factory,
  //                                                        /* device_id */ 0,
  //                                                        /* use_arena */ false};

  //   allocators.emplace_back(CreateAllocator(rpcmem_allocator_creation_info));
  // }

  return allocators;
}

OrtDevice QNNExecutionProvider::GetOrtDeviceByMemType(OrtMemType /* em_type */) const {
  // We are disabling the HTP shared memory allocator for intermediate values
  // until we learn how to deal with memhandle costs.
  // if (IsHtpSharedMemoryAllocatorAvailable()) {
  //  return qnn::HtpSharedMemoryAllocator::AssociatedMemoryInfo().device;
  //}
  // Default CPU allocator
  return default_device_;
}

Status QNNExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                 gsl::span<const char* const> values) {
                                                  keys;values;
  // if (keys.size() != values.size()) {
  //   LOGS_DEFAULT(ERROR) << "SetEpDynamicOptions: number of keys (" << keys.size()
  //                       << ") does not equal number of values (" << values.size() << ").";
  // }
  // auto key_it = keys.begin();
  // auto value_it = values.begin();

  // while (key_it != keys.end() && value_it != values.end()) {
  //   std::string key(*key_it);
  //   std::string value(*value_it);

  //   if (key == kOrtEpDynamicOptionsWorkloadType) {
  //     if (value == "Default") {
  //       ORT_RETURN_IF_ERROR(qnn_backend_manager_->ResetContextPriority());
  //     } else if (value == "Efficient") {
  //       ORT_RETURN_IF_ERROR(qnn_backend_manager_->SetContextPriority(qnn::ContextPriority::LOW));
  //     } else {
  //       LOGS_DEFAULT(ERROR) << "Invalid EP Workload Type: " << value;
  //       return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid EP Workload Type.");
  //     }
  //   } else {
  //     LOGS_DEFAULT(ERROR) << "EP Dynamic Option \"" << key << "\" is not currently supported.";
  //     return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported EP Dynamic Option");
  //   }

  //   key_it++;
  //   value_it++;
  // }

  return Status::OK();
}

}  // namespace onnxruntime
