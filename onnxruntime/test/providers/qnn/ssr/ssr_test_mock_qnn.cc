#include <vector>
#include "C:/Users/hungjuiw/workspace/qairt/2.38.0.250812/include/QNN/QnnInterface.h"
// #include "QnnInterface.h"

QNN_API
Qnn_ErrorHandle_t QnnGraph_create(Qnn_ContextHandle_t contextHandle,
                                  const char* graphName,
                                  const QnnGraph_Config_t** config,
                                  Qnn_GraphHandle_t* graphHandle) {
  Qnn_ErrorHandle_t status = static_cast<Qnn_ErrorHandle_t>(QNN_GRAPH_NO_ERROR);
  return status;
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_getBuildId(const char** id) {
  if (!id) {
    return QNN_BACKEND_ERROR_INVALID_ARGUMENT;
  }
  // *id = QNN_SDK_BUILD_ID;
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnLog_create(QnnLog_Callback_t callback,
                                QnnLog_Level_t maxLogLevel,
                                Qnn_LogHandle_t* logger) {
  Qnn_ErrorHandle_t status = QNN_LOG_NO_ERROR;
  return status;
}

extern "C"
Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList,
                                            uint32_t* numProviders) {
  static QnnInterface_t interface;
  interface.backendId = 0;
  interface.providerName = "MockSSR";
  interface.apiVersion = QNN_API_VERSION_INIT;
  interface.QNN_INTERFACE_VER_NAME = QNN_INTERFACE_VER_TYPE_INIT;
  interface.QNN_INTERFACE_VER_NAME.backendGetBuildId = QnnBackend_getBuildId;
  interface.QNN_INTERFACE_VER_NAME.logCreate = QnnLog_create;
  static std::vector<const QnnInterface_t*> m_providerPtrs = {&interface};
  *providerList = m_providerPtrs.data(),
  *numProviders = 1;
  return QNN_SUCCESS;
}

// Qnn_ErrorHandle_t QnnGraph_createSubgraph(Qnn_GraphHandle_t graphHandle,
//                                           const char* graphName,
//                                           Qnn_GraphHandle_t* subgraphHandle) {
//   return QNN_COMMON_ERROR_NOT_SUPPORTED;
// }

// Qnn_ErrorHandle_t QnnGraph_setConfig(Qnn_GraphHandle_t graphHandle,
//                                      const QnnGraph_Config_t** config) {
//   return QNN_COMMON_ERROR_NOT_SUPPORTED;
// }

// Qnn_ErrorHandle_t QnnGraph_getProperty(Qnn_GraphHandle_t graphHandle,
//                                        QnnGraph_Property_t** properties) {
//   if (NULL == graphHandle) {
//   return QNN_GRAPH_ERROR_INVALID_HANDLE;
//   }

//   if (NULL == properties) {
//   return QNN_GRAPH_ERROR_INVALID_ARGUMENT;
//   }

//   Qnn_ErrorHandle_t status = QNN_GRAPH_NO_ERROR;
//   while (*properties) {
//     switch ((*properties)->option) {
//     case QNN_GRAPH_PROPERTY_OPTION_CUSTOM:
//       status = QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE;
//       break;
//     default:
//       status = QNN_GRAPH_ERROR_INVALID_ARGUMENT;
//       break;
//     }
//     if (status != QNN_GRAPH_NO_ERROR) {
//       break;
//     }
//     (*properties)++;
//   }
//   return status;
// }


// Qnn_ErrorHandle_t QnnGraph_addNode(Qnn_GraphHandle_t graph, Qnn_OpConfig_t opConfig) {
//   // Default Value of status
//   Qnn_ErrorHandle_t status = static_cast<Qnn_ErrorHandle_t>(QNN_GRAPH_NO_ERROR);
//   return status;
// }

// Qnn_ErrorHandle_t QnnGraph_finalize(Qnn_GraphHandle_t graph,
//                                     Qnn_ProfileHandle_t profileHandle,
//                                     Qnn_SignalHandle_t signalHandle) {
//   Qnn_ErrorHandle_t status = static_cast<Qnn_ErrorHandle_t>(QNN_GRAPH_NO_ERROR);
//   return status;
// }

// Qnn_ErrorHandle_t QnnGraph_retrieve(Qnn_ContextHandle_t contextHandle,
//                                     const char* graphName,
//                                     Qnn_GraphHandle_t* graphHandle) {
//   return QNN_COMMON_ERROR_NOT_SUPPORTED;
// }

// Qnn_ErrorHandle_t QnnGraph_prepareExecutionEnvironment(Qnn_GraphHandle_t graphHandle,
//   QnnGraph_ExecuteEnvironment_t** envs,
//           uint32_t envSize) {
//   return QNN_SUCCESS;
// }

// Qnn_ErrorHandle_t QnnGraph_execute(Qnn_GraphHandle_t graphHandle,
//                                    const Qnn_Tensor_t* inputs,
//                                    uint32_t numInputs,
//                                    Qnn_Tensor_t* outputs,
//                                    uint32_t numOutputs,
//                                    Qnn_ProfileHandle_t profileHandle,
//                                    Qnn_SignalHandle_t signalHandle) {
//   return QNN_COMMON_ERROR_NOT_SUPPORTED;
// }

// Qnn_ErrorHandle_t QnnGraph_executeAsync(Qnn_GraphHandle_t graphHandle,
//                                         const Qnn_Tensor_t* inputs,
//                                         uint32_t numInputs,
//                                         Qnn_Tensor_t* outputs,
//                                         uint32_t numOutputs,
//                                         Qnn_ProfileHandle_t profileHandle,
//                                         Qnn_SignalHandle_t signalHandle,
//                                         Qnn_NotifyFn_t notifyFn,
//                                         void* notifyParam) {
//   return QNN_COMMON_ERROR_NOT_SUPPORTED;
// }

// Qnn_ErrorHandle_t QnnGraph_releaseExecutionEnvironment(Qnn_GraphHandle_t graphHandle,
//                                                        const QnnGraph_ExecuteEnvironment_t** envs,
//                                                        uint32_t envSize) {
//   return QNN_SUCCESS;
// }
