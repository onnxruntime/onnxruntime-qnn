// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

package ai.onnxruntime.qnnpluginep

private const val QNN_EP_LIBRARY_NAME = "onnxruntime_providers_qnn"
private const val QNN_EP_NAME = "QNNExecutionProvider"

/**
 * Returns the filename of the QNN Execution Provider shared library.
 * This can be passed to `OrtEnvironment.registerExecutionProviderLibrary()`.
 */
fun getLibraryPath(): String {
    return "lib${QNN_EP_LIBRARY_NAME}.so"
}

/**
 * Returns the EP name exposed by the QNN Execution Provider library.
 * Use this to filter `OrtEnvironment.epDevices` when selecting the QNN EP.
 */
fun getEpName(): String {
    return QNN_EP_NAME
}

/**
 * Returns the EP names exposed by the QNN Execution Provider library. There is only one.
 * Use this to filter `OrtEnvironment.epDevices` when selecting the QNN EP.
 */
fun getEpNames() : Array<String> {
    return arrayOf(getEpName())
}
