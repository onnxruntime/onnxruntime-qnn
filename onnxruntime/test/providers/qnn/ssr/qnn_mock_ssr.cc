// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// QnnMockSSR.dll — wraps QnnHtp.dll and injects QNN_COMMON_ERROR_SYSTEM_COMMUNICATION
// on the FIRST graphExecute call to simulate an NPU Subsystem Restart (SSR).
// Subsequent calls are forwarded to the real HTP backend unchanged.
//
// In multi-partition models, sibling QnnModel instances detect the stale context
// proactively (via HasContextHandle check in ExecuteGraph) and recover without
// needing the backend to return SSR for each graph individually.
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

// SSR mock state: fire SSR on the first graphExecute call per session.
// g_ssr_fired is reset in QnnInterface_getProviders (called once per new session/test).
// Declared at file scope so QnnInterface_getProviders can reset it on all platforms.
static bool g_ssr_fired = false;

// Forces every graphExecute to fail, so a retry after SSR recovery crashes again.
// Exported (see .def) because no header exists for this DLL.
static bool g_always_crash_mode = false;

extern "C" void SetMockSSRAlwaysCrash(bool always_crash) {
  g_always_crash_mode = always_crash;
}

// Tracks whether contextCreateFromBinaryWithCallback was invoked during SSR recovery
// (i.e., after g_ssr_fired became true).  Reset alongside g_ssr_fired each session.
static bool g_file_mapping_used_for_recovery = false;

// When true, contextCreateFromBinaryWithCallback returns QNN_COMMON_ERROR_NOT_SUPPORTED
// during SSR recovery without forwarding to the real backend, simulating the behavior
// seen with a pre-3.3.3 context binary (file mapping not supported).
// NOT reset by QnnInterface_getProviders; tests must set/clear this explicitly.
static bool g_simulate_old_blob_version = false;

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
// per session, then forwards to real HTP on subsequent calls.
QNN_API
Qnn_ErrorHandle_t QnnGraph_execute(Qnn_GraphHandle_t graphHandle,
                                   const Qnn_Tensor_t* inputs,
                                   uint32_t numInputs,
                                   Qnn_Tensor_t* outputs,
                                   uint32_t numOutputs,
                                   Qnn_ProfileHandle_t profileHandle,
                                   Qnn_SignalHandle_t signalHandle) {
  if (g_always_crash_mode) {
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  }
  if (!g_ssr_fired) {
    g_ssr_fired = true;
    g_file_mapping_used_for_recovery = false;  // Reset: anything after this is recovery.
    return QNN_COMMON_ERROR_SYSTEM_COMMUNICATION;
  }
  if (!real_providerList) {
    return QNN_COMMON_ERROR_GENERAL;
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.graphExecute(
      graphHandle, inputs, numInputs, outputs, numOutputs, profileHandle, signalHandle);
}

#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
// Intercepts contextCreateFromBinaryWithCallback: records whether it was called during
// SSR recovery (g_ssr_fired == true).  Calls before SSR are the initial load and are
// not counted.
QNN_API
Qnn_ErrorHandle_t QnnContext_createFromBinaryWithCallback(
    Qnn_BackendHandle_t backendHandle,
    Qnn_DeviceHandle_t deviceHandle,
    const QnnContext_Config_t** config,
    const Qnn_ContextBinaryCallback_t* callbacks,
    const void* binaryBuffer,
    Qnn_ContextBinarySize_t binaryBufferSize,
    Qnn_ContextHandle_t* context,
    Qnn_ProfileHandle_t profileHandle,
    void* reserved) {
  if (g_ssr_fired) {
    if (g_simulate_old_blob_version) {
      // Simulate a pre-3.3.3 context binary: reject the callback API and do NOT record
      // this as a successful file-mapping recovery.  The caller should fall back to
      // contextCreateFromBinary (direct read).
      return QNN_COMMON_ERROR_NOT_SUPPORTED;
    }
    g_file_mapping_used_for_recovery = true;
  }
  if (!real_providerList) {
    return QNN_COMMON_ERROR_GENERAL;
  }
  return real_providerList[0]->QNN_INTERFACE_VER_NAME.contextCreateFromBinaryWithCallback(
      backendHandle, deviceHandle, config, callbacks, binaryBuffer, binaryBufferSize,
      context, profileHandle, reserved);
}
#endif  // QNN_FILE_MAPPED_WEIGHTS_AVAILABLE

#endif  // defined(_WIN32)

// Returns true if contextCreateFromBinaryWithCallback was invoked during SSR recovery
// in the current session.  Can be queried by tests after triggering SSR to verify that
// file mapping was used for recovery.
extern "C" bool QnnMockSSR_WasFileMappingUsedForRecovery() {
  return g_file_mapping_used_for_recovery;
}

// When set to true, the contextCreateFromBinaryWithCallback interceptor will return
// QNN_COMMON_ERROR_NOT_SUPPORTED during SSR recovery instead of forwarding to the real
// backend, simulating a pre-3.3.3 context binary where the callback API is unsupported.
// This flag is NOT reset by QnnInterface_getProviders; callers must reset it explicitly.
extern "C" void QnnMockSSR_SetSimulateOldBlobVersion(bool simulate) {
  g_simulate_old_blob_version = simulate;
}

// 'interface' is #defined as 'struct' in <objbase.h> (pulled in via <windows.h>).
// Use a different name to avoid that macro collision.
extern "C" Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList,
                                                       uint32_t* numProviders) {
  // Reset mock state for each new session that loads this provider.
  g_ssr_fired = false;
  g_file_mapping_used_for_recovery = false;

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
    // Intercept graphExecute to simulate SSR.
    mock_interface.QNN_INTERFACE_VER_NAME.graphExecute = QnnGraph_execute;
#ifdef QNN_FILE_MAPPED_WEIGHTS_AVAILABLE
    // Intercept contextCreateFromBinaryWithCallback to detect file mapping during recovery.
    mock_interface.QNN_INTERFACE_VER_NAME.contextCreateFromBinaryWithCallback =
        QnnContext_createFromBinaryWithCallback;
#endif
  }
#endif  // defined(_WIN32)
  static std::vector<const QnnInterface_t*> m_providerPtrs = {&mock_interface};
  *providerList = m_providerPtrs.data();
  *numProviders = 1;
  return QNN_SUCCESS;
}
