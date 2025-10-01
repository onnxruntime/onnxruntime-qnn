#include <vector>
#include "QnnInterface.h"

QNN_API
Qnn_ErrorHandle_t QnnGraph_create(Qnn_ContextHandle_t contextHandle,
                                  const char* graphName,
                                  const QnnGraph_Config_t** config,
                                  Qnn_GraphHandle_t* graphHandle) {
  static int mock_graph = 3;
  *graphHandle = reinterpret_cast<Qnn_GraphHandle_t>(&mock_graph);
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_getBuildId(const char** id) {
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnLog_create(QnnLog_Callback_t callback,
                                QnnLog_Level_t maxLogLevel,
                                Qnn_LogHandle_t* logger) {
  Qnn_ErrorHandle_t status = QNN_LOG_NO_ERROR;
  return status;
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_create(Qnn_LogHandle_t logHandle,
                                    const QnnBackend_Config_t** config,
                                    Qnn_BackendHandle_t* backend) {
  return QNN_BACKEND_NO_ERROR;
}

QNN_API
Qnn_ErrorHandle_t QnnContext_create(Qnn_BackendHandle_t backend,
                                    Qnn_DeviceHandle_t device,
                                    const QnnContext_Config_t** config,
                                    Qnn_ContextHandle_t* context) {
  // Qnn_ContextHandle_t itself is a pointer (void*)
  // We use an (int*) to mock it (void*). Accessing its value will crash since
  // there is no real memory allocation
  static int mock_context = 5;
  *context = reinterpret_cast<Qnn_ContextHandle_t>(&mock_context);
  return QNN_CONTEXT_NO_ERROR;
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_validateOpConfig(Qnn_BackendHandle_t backend,
                                              Qnn_OpConfig_t opConfig) {
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_addNode(Qnn_GraphHandle_t graph, Qnn_OpConfig_t opConfig) {
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnTensor_createGraphTensor(Qnn_GraphHandle_t graph, Qnn_Tensor_t* tensor) {
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_retrieve(Qnn_ContextHandle_t contextHandle,
                                    const char* graphName,
                                    Qnn_GraphHandle_t* graphHandle) {
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_finalize(Qnn_GraphHandle_t graph,
                                    Qnn_ProfileHandle_t profileHandle,
                                    Qnn_SignalHandle_t signalHandle) {
  return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
}

QNN_API
Qnn_ErrorHandle_t QnnContext_getBinarySize(Qnn_ContextHandle_t context,
                                           Qnn_ContextBinarySize_t* binaryBufferSize) {
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnContext_getBinary(Qnn_ContextHandle_t context,
                                       void* binaryBuffer,
                                       Qnn_ContextBinarySize_t binaryBufferSize,
                                       Qnn_ContextBinarySize_t* writtenBufferSize) {
  return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_execute(Qnn_GraphHandle_t graphHandle,
                                   const Qnn_Tensor_t* inputs,
                                   uint32_t numInputs,
                                   Qnn_Tensor_t* outputs,
                                   uint32_t numOutputs,
                                   Qnn_ProfileHandle_t profileHandle,
                                   Qnn_SignalHandle_t signalHandle) {
  return QNN_SUCCESS;
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
  interface.QNN_INTERFACE_VER_NAME.backendCreate = QnnBackend_create;
  interface.QNN_INTERFACE_VER_NAME.contextCreate = QnnContext_create;
  interface.QNN_INTERFACE_VER_NAME.backendValidateOpConfig = QnnBackend_validateOpConfig;
  interface.QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor = QnnTensor_createGraphTensor;
  interface.QNN_INTERFACE_VER_NAME.graphCreate = QnnGraph_create;
  interface.QNN_INTERFACE_VER_NAME.graphRetrieve = QnnGraph_retrieve;
  interface.QNN_INTERFACE_VER_NAME.graphAddNode = QnnGraph_addNode;
  interface.QNN_INTERFACE_VER_NAME.graphFinalize = QnnGraph_finalize;
  interface.QNN_INTERFACE_VER_NAME.contextGetBinarySize = QnnContext_getBinarySize;
  interface.QNN_INTERFACE_VER_NAME.contextGetBinary = QnnContext_getBinary;
  interface.QNN_INTERFACE_VER_NAME.graphExecute = QnnGraph_execute;
  static std::vector<const QnnInterface_t*> m_providerPtrs = {&interface};
  *providerList = m_providerPtrs.data(),
  *numProviders = 1;
  return QNN_SUCCESS;
}
