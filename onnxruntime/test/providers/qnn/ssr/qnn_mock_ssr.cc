// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// QnnMockSSR.dll — wraps QnnHtp.dll and injects one QNN_COMMON_ERROR_SYSTEM_COMMUNICATION
// error on the FIRST graphExecute call to simulate an NPU Subsystem Restart (SSR).
// Subsequent calls are forwarded to the real HTP backend unchanged.

#include <vector>
#include <thread>
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "rpcmem_utils.h"

const QnnInterface_t** real_providerList{nullptr};
uint32_t real_numProviders{0};

namespace {
#if defined(_WIN32)
// Load QnnHtp.dll at DLL startup and resolve the real QnnInterface_getProviders.
// Register a deleter so QnnHtp.dll is released before QnnMockSSR.dll destructs.
auto free_qnn_htp_fn = [](HMODULE m) {
  if (m) FreeLibrary(m);
};

std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(free_qnn_htp_fn)> qnn_htp(
    LoadLibraryW(L"QnnHtp.dll"), free_qnn_htp_fn);

FARPROC addr = GetProcAddress(qnn_htp.get(), "QnnInterface_getProviders");
typedef Qnn_ErrorHandle_t (*QnnApiFnType_t)(const QnnInterface_t***, uint32_t*);
QnnApiFnType_t real_QnnInterface_getProviders = reinterpret_cast<QnnApiFnType_t>(addr);
auto res = real_QnnInterface_getProviders((const QnnInterface_t***)&real_providerList, &real_numProviders);
#endif  // defined(_WIN32)
}  // namespace

#if defined(_WIN32)

// Intercepts graphExecute: triggers a real NPU PD reset on the first call, then
// delegates to the real HTP backend (which returns QNN_COMMON_ERROR_SYSTEM_COMMUNICATION
// while the NPU is resetting, and succeeds on subsequent calls after recovery).
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
    onnxruntime::test::TriggerPDReset();
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphExecute(
      graphHandle, inputs, numInputs, outputs, numOutputs, profileHandle, signalHandle);
}

#endif  // defined(_WIN32)

// 'interface' is #defined as 'struct' in <objbase.h> (pulled in via <windows.h>).
// Use a different name to avoid that macro collision.
extern "C" Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList,
                                                       uint32_t* numProviders) {
  static QnnInterface_t mock_interface;
#if defined(_WIN32)
  mock_interface.backendId = real_providerList[0]->backendId;
#else
  mock_interface.backendId = 0;
#endif
  mock_interface.providerName = "MockSSR";
#if defined(_WIN32)
  mock_interface.apiVersion = real_providerList[0]->apiVersion;
  mock_interface.QNN_INTERFACE_VER_NAME = real_providerList[0]->QNN_INTERFACE_VER_NAME;
  // Always intercept graphExecute to simulate SSR.
  mock_interface.QNN_INTERFACE_VER_NAME.graphExecute = QnnGraph_execute;
#endif  // defined(_WIN32)
  static std::vector<const QnnInterface_t*> m_providerPtrs = {&mock_interface};
  *providerList = m_providerPtrs.data();
  *numProviders = 1;
  return QNN_SUCCESS;
}
