// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Intentionally re-includable action header. Include after Windows/ORT headers
// that require the COM `interface` macro and before QNN headers that use
// `interface` as a regular C++ identifier.
#if defined(_WIN32) && defined(interface)
#undef interface
#endif
