#include <cassert>
#include <iostream>
#include <vector>
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "ssr_controller.h"

extern "C"
QnnSSRController* GetQnnSSRController() { return &QnnSSRController::Instance(); }

#if defined(_WIN32)
#include <windows.h>
HMODULE lib_handle = LoadLibraryW(L"QnnHtp.dll");
FARPROC addr = GetProcAddress(lib_handle, "QnnInterface_getProviders");
typedef Qnn_ErrorHandle_t (*QnnApiFnType_t)(const QnnInterface_t***, uint32_t*);
QnnApiFnType_t real_QnnInterface_getProviders = reinterpret_cast<QnnApiFnType_t>(addr);
const QnnInterface_t** real_providerList{nullptr};
uint32_t real_numProviders{0};
auto res = real_QnnInterface_getProviders((const QnnInterface_t***)&real_providerList, &real_numProviders);
#endif  // defined(_WIN32)

QNN_API
Qnn_ErrorHandle_t QnnGraph_create(Qnn_ContextHandle_t contextHandle,
                                  const char* graphName,
                                  const QnnGraph_Config_t** config,
                                  Qnn_GraphHandle_t* graphHandle) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphCreate(
      contextHandle, graphName, config, graphHandle);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_getBuildId(const char** id) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.backendGetBuildId(id);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnLog_create(QnnLog_Callback_t callback,
                                QnnLog_Level_t maxLogLevel,
                                Qnn_LogHandle_t* logger) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.logCreate(
      callback, maxLogLevel, logger);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_create(Qnn_LogHandle_t logHandle,
                                    const QnnBackend_Config_t** config,
                                    Qnn_BackendHandle_t* backend) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.backendCreate(
      logHandle, config, backend);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnContext_create(Qnn_BackendHandle_t backend,
                                    Qnn_DeviceHandle_t device,
                                    const QnnContext_Config_t** config,
                                    Qnn_ContextHandle_t* context) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.contextCreate(
      backend, device, config, context);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnBackend_validateOpConfig(Qnn_BackendHandle_t backend,
                                              Qnn_OpConfig_t opConfig) {
#if defined(_WIN32)
  typedef Qnn_ErrorHandle_t (*QnnApiFnType_t)(Qnn_BackendHandle_t, Qnn_OpConfig_t);
  FARPROC addr = GetProcAddress(lib_handle, "QnnBackend_validateOpConfig");
  QnnApiFnType_t real_QnnBackend_validateOpConfig = reinterpret_cast<QnnApiFnType_t>(addr);
#endif  // defined(_WIN32)
  return real_QnnBackend_validateOpConfig(backend, opConfig);
  // return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_addNode(Qnn_GraphHandle_t graph, Qnn_OpConfig_t opConfig) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphAddNode(
      graph, opConfig);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnTensor_createGraphTensor(Qnn_GraphHandle_t graph, Qnn_Tensor_t* tensor) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(
      graph, tensor);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_retrieve(Qnn_ContextHandle_t contextHandle,
                                    const char* graphName,
                                    Qnn_GraphHandle_t* graphHandle) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphRetrieve(
      contextHandle, graphName, graphHandle);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_finalize(Qnn_GraphHandle_t graph,
                                    Qnn_ProfileHandle_t profileHandle,
                                    Qnn_SignalHandle_t signalHandle) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphFinalize(
      graph, profileHandle, signalHandle);
  }
}

QNN_API
Qnn_ErrorHandle_t QnnContext_getBinarySize(Qnn_ContextHandle_t context,
                                           Qnn_ContextBinarySize_t* binaryBufferSize) {
#if defined(_WIN32)
  typedef Qnn_ErrorHandle_t (*QnnApiFnType_t)(Qnn_ContextHandle_t, Qnn_ContextBinarySize_t*);
  FARPROC addr = GetProcAddress(lib_handle, "QnnContext_getBinarySize");
  QnnApiFnType_t real_QnnContext_getBinarySize = reinterpret_cast<QnnApiFnType_t>(addr);
#endif  // defined(_WIN32)
  return real_QnnContext_getBinarySize(context, binaryBufferSize);
  // return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnContext_getBinary(Qnn_ContextHandle_t context,
                                       void* binaryBuffer,
                                       Qnn_ContextBinarySize_t binaryBufferSize,
                                       Qnn_ContextBinarySize_t* writtenBufferSize) {
#if defined(_WIN32)
  typedef Qnn_ErrorHandle_t (*QnnApiFnType_t)(Qnn_ContextHandle_t, void*, Qnn_ContextBinarySize_t, Qnn_ContextBinarySize_t*);
  FARPROC addr = GetProcAddress(lib_handle, "QnnContext_getBinary");
  QnnApiFnType_t real_QnnContext_getBinary = reinterpret_cast<QnnApiFnType_t>(addr);
#endif  // defined(_WIN32)
  return real_QnnContext_getBinary(context, binaryBuffer, binaryBufferSize, writtenBufferSize);
  // return QNN_SUCCESS;
}

QNN_API
Qnn_ErrorHandle_t QnnGraph_execute(Qnn_GraphHandle_t graphHandle,
                                   const Qnn_Tensor_t* inputs,
                                   uint32_t numInputs,
                                   Qnn_Tensor_t* outputs,
                                   uint32_t numOutputs,
                                   Qnn_ProfileHandle_t profileHandle,
                                   Qnn_SignalHandle_t signalHandle) {
  static int call_cnt = 0;
  if (call_cnt == 0) {
    call_cnt += 1;
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  } else {
    return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphExecute(graphHandle,
      inputs, numInputs, outputs, numOutputs, profileHandle, signalHandle);
  }
}

extern "C"
Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList,
                                            uint32_t* numProviders) {
  static QnnInterface_t interface;
  interface.backendId = 0;
  interface.providerName = "MockSSR";
  interface.apiVersion = real_providerList[0]->apiVersion;
  interface.QNN_INTERFACE_VER_NAME = real_providerList[0]->QNN_INTERFACE_VER_NAME;
  std::cout << "QnnSSRController::Instance().GetTiming() " << static_cast<int>(QnnSSRController::Instance().GetTiming()) << std::endl;
  switch(QnnSSRController::Instance().GetTiming()) {
    case QnnSSRController::Timing::TensorCreateGraphTensor:
      interface.QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor = QnnTensor_createGraphTensor;
      break;
    case QnnSSRController::Timing::GraphAddNode:
      interface.QNN_INTERFACE_VER_NAME.graphAddNode = QnnGraph_addNode;
      break;
    case QnnSSRController::Timing::GraphFinalize:
      interface.QNN_INTERFACE_VER_NAME.graphFinalize = QnnGraph_finalize;
      break;
    case QnnSSRController::Timing::GraphExecute:
      interface.QNN_INTERFACE_VER_NAME.graphExecute = QnnGraph_execute;
      break;
    default:
      break;
  }
  // interface.QNN_INTERFACE_VER_NAME.backendGetBuildId = QnnBackend_getBuildId;
  // interface.QNN_INTERFACE_VER_NAME.logCreate = QnnLog_create;
  // interface.QNN_INTERFACE_VER_NAME.backendCreate = QnnBackend_create;
  // interface.QNN_INTERFACE_VER_NAME.contextCreate = QnnContext_create;
  // interface.QNN_INTERFACE_VER_NAME.backendValidateOpConfig = QnnBackend_validateOpConfig;
  // interface.QNN_INTERFACE_VER_NAME.graphCreate = QnnGraph_create;
  // interface.QNN_INTERFACE_VER_NAME.graphRetrieve = QnnGraph_retrieve;
  // interface.QNN_INTERFACE_VER_NAME.contextGetBinarySize = QnnContext_getBinarySize;
  // interface.QNN_INTERFACE_VER_NAME.contextGetBinary = QnnContext_getBinary;
  static std::vector<const QnnInterface_t*> m_providerPtrs = {&interface};
  *providerList = m_providerPtrs.data(),
  *numProviders = 1;
  return QNN_SUCCESS;
}
