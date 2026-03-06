// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/test_environment.h"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "google/protobuf/stubs/common.h"

#include "core/session/ort_env.h"
#include "core/session/environment.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

const ::onnxruntime::Environment& GetEnvironment() {
  return ((OrtEnv*)*ort_env.get())->GetEnvironment();
}

Ort::Env* GetOrtEnv() {
  return ort_env.get();
}

}  // namespace test
}  // namespace onnxruntime
