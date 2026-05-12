// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Wrapper around <onnx/onnx_pb.h> that suppresses warnings from third-party
// protobuf / ONNX headers — most notably -Wshorten-64-to-32, which fires
// inside protobuf's parse_context.h on the aarch64_oe_gcc11_2 clang toolchain
// that builds with -Werror.
//
// This file mirrors core/graph/onnx_protobuf.h from ORT Core. We keep a local
// copy under core/providers/qnn/common/ rather than including the ORT-internal
// header directly, because qcom/linters/check_private_ort_headers.py forbids
// QNN EP source from #including paths that start with "core/" outside of
// "core/providers/qnn/".

#pragma once

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4244)  // possible loss of data on conversion
#endif

#if defined(__clang__)
// -Wshorten-64-to-32 only exists on clang. Suppressing it on GCC would itself
// be an unknown-pragma error.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include "onnx/onnx_pb.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef _WIN32
#pragma warning(pop)
#endif
