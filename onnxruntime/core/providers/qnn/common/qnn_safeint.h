// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

// SafeInt wrapper for the QNN EP plugin.
//
// We intentionally do NOT use core/common/safeint.h (which uses ORT_THROW →
// ORT_WHERE_WITH_STACK → GetStackTrace()) because onnxruntime::GetStackTrace is
// a LOCAL symbol in libonnxruntime.so and is not available to plugin EPs at
// runtime. Using std::runtime_error here avoids the undefined-symbol dependency
// while still throwing on integer overflow/divide-by-zero.
//
// In test builds, core/common/safeint.h may be included transitively before this
// header. In that case SafeIntDefaultExceptionHandler is already defined (using
// ORT_THROW), so we skip our definition to avoid a redefinition conflict. The
// ORT_THROW-based handler is safe in tests because GetStackTrace is available
// when linking against ort_core.
//
// DO NOT replace this header with core/common/safeint.h — doing so will introduce
// a hidden runtime dependency on GetStackTrace that only manifests when the EP is
// loaded as a plugin (not in test builds).
//
// An equivalent handler is also defined locally in qnn_test_utils.cc for test
// builds. It cannot be consolidated here because ORT's core/common/safeint.h
// (included transitively in test builds) declares SafeIntExceptionHandler as a
// class template, which is incompatible with this concrete class definition.
#ifndef SafeIntDefaultExceptionHandler
#include <exception>
#include <stdexcept>
class SafeIntExceptionHandler : public std::exception {
 public:
  [[noreturn]] static void SafeIntOnOverflow() { throw std::runtime_error("Integer overflow"); }
  [[noreturn]] static void SafeIntOnDivZero() { throw std::runtime_error("Divide by zero"); }
};

#define SAFEINT_EXCEPTION_HANDLER_CPP 1
#define SafeIntDefaultExceptionHandler SafeIntExceptionHandler
#endif  // !defined(SafeIntDefaultExceptionHandler)

#if defined(__GNUC__)
#include "onnxruntime_config.h"
#pragma GCC diagnostic push
#ifdef HAS_UNUSED_BUT_SET_PARAMETER
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#endif
#endif
#include "SafeInt.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
