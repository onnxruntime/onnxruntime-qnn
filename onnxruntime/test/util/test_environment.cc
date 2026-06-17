// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <memory>

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

Ort::Env* GetOrtEnv() {
  return ort_env.get();
}

}  // namespace test
}  // namespace onnxruntime
