// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// QnnMockSSR.dll — wraps QnnHtp.dll and injects QNN_COMMON_ERROR_SYSTEM_COMMUNICATION
// on the FIRST graphExecute call to simulate an NPU Subsystem Restart (SSR).
// Subsequent calls are forwarded to the real HTP backend unchanged.
//
// This mock does NOT perform a real PD reset — it simply returns the SSR error code.
// This makes the test portable across devices regardless of CDSP driver configuration.

#include <memory>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#endif
#include "QnnCommon.h"
#include "QnnInterface.h"

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

struct StaticInit {
  StaticInit() {
    if (!qnn_htp.get()) return;
    FARPROC addr = GetProcAddress(qnn_htp.get(), "QnnInterface_getProviders");
    if (!addr) return;
    typedef Qnn_ErrorHandle_t (*QnnApiFnType_t)(const QnnInterface_t***, uint32_t*);
    QnnApiFnType_t real_QnnInterface_getProviders = reinterpret_cast<QnnApiFnType_t>(addr);
    real_QnnInterface_getProviders((const QnnInterface_t***)&real_providerList, &real_numProviders);
  }
} static_init;
#endif  // defined(_WIN32)
}  // namespace

#if defined(_WIN32)

// Intercepts graphExecute: returns QNN_COMMON_ERROR_SYSTEM_COMMUNICATION on the first call
// to simulate SSR, then forwards to the real HTP backend on subsequent calls.
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
  }
  if (!real_providerList) {
    return QNN_COMMON_ERROR_GENERAL;
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
  if (real_providerList) {
    mock_interface.backendId = real_providerList[0]->backendId;
  } else {
    mock_interface.backendId = 0;
  }
#else
  mock_interface.backendId = 0;
#endif
  mock_interface.providerName = "MockSSR";
#if defined(_WIN32)
  if (real_providerList) {
    mock_interface.apiVersion = real_providerList[0]->apiVersion;
    mock_interface.QNN_INTERFACE_VER_NAME = real_providerList[0]->QNN_INTERFACE_VER_NAME;
    // Always intercept graphExecute to simulate SSR.
    mock_interface.QNN_INTERFACE_VER_NAME.graphExecute = QnnGraph_execute;
  }
#endif  // defined(_WIN32)
  static std::vector<const QnnInterface_t*> m_providerPtrs = {&mock_interface};
  *providerList = m_providerPtrs.data();
  *numProviders = 1;
  return QNN_SUCCESS;
}
