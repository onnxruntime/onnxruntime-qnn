// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared helpers for prepare_app.cc and run_app.cc.
//
// Example only: the cipher is a single-byte XOR, just to make the write/read
// callback mechanism obvious. Swap WriteCb/ReadCb in prepare_app.cc/run_app.cc
// for your own encryption to use this for real.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

#if ORT_API_VERSION >= 28
#include "onnxruntime_experimental_c_api.h"
#include "onnxruntime_experimental_cxx_api.h"
#endif

namespace encapp {

// Default XOR key. Both apps must agree; overridable on the command line.
constexpr uint8_t kDefaultKey = 0x5A;

// The registration name the QNN EP is published under.
inline const char* QnnEpName() { return "QNNExecutionProvider"; }

// Checks that onnxruntime.dll actually supports ORT_API_VERSION (1.28+),
// before any Ort C++ call that would dereference a null API pointer on an
// older runtime. Call this first thing in main().
inline bool EnsureRuntimeApiAvailable() {
  const OrtApiBase* base = OrtGetApiBase();
  if (base == nullptr) {
    std::fprintf(stderr,
                 "[error] OrtGetApiBase() returned null — onnxruntime.dll is missing or "
                 "corrupt.\n");
    return false;
  }
  if (base->GetApi(ORT_API_VERSION) == nullptr) {
    const char* ver = (base->GetVersionString != nullptr) ? base->GetVersionString() : "<unknown>";
    std::fprintf(stderr,
                 "[error] This build requires ORT API version %d (ONNX Runtime 1.28+), but the "
                 "loaded onnxruntime.dll only reports version \"%s\" and does not provide it.\n"
                 "        The EPContext encryption callbacks are a 1.28 feature. Replace "
                 "onnxruntime.dll (and onnxruntime_providers_shared.dll) with a 1.28+ build.\n",
                 static_cast<int>(ORT_API_VERSION), ver);
    return false;
  }
  return true;
}

// XOR a buffer in place.
inline void XorInPlace(uint8_t* data, size_t n, uint8_t key) {
  for (size_t i = 0; i < n; ++i) data[i] ^= key;
}

// ---------------------------------------------------------------------------
// Real-input / answer-output support
//
// Both apps can optionally run on a real float32 input and dump their output
// to a flat .raw file — prepare_app from the plaintext model, run_app from
// the decrypted context model. Comparing the two is left to a separate tool
// (compare_answers.py); neither app judges pass/fail on its own output.
// ---------------------------------------------------------------------------

// Read a whole file as raw bytes.
inline bool ReadFileBytes(const char* path, std::vector<uint8_t>& out) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.good()) {
    std::fprintf(stderr, "[error] cannot open file: %s\n", path);
    return false;
  }
  const auto size = static_cast<size_t>(in.tellg());
  in.seekg(0);
  out.resize(size);
  if (size > 0 && !in.read(reinterpret_cast<char*>(out.data()),
                           static_cast<std::streamsize>(size))) {
    std::fprintf(stderr, "[error] failed reading file: %s\n", path);
    return false;
  }
  return true;
}

// Load a flat float32 .raw file into `out`.
inline bool ReadFloatRaw(const char* path, std::vector<float>& out) {
  std::vector<uint8_t> bytes;
  if (!ReadFileBytes(path, bytes)) return false;
  if (bytes.empty() || (bytes.size() % sizeof(float)) != 0) {
    std::fprintf(stderr,
                 "[error] %s is %zu bytes — not a whole number of float32 values.\n",
                 path, bytes.size());
    return false;
  }
  out.resize(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return true;
}

// Write a float32 vector as a flat .raw file.
inline bool WriteFloatRaw(const char* path, const std::vector<float>& v) {
  std::ofstream o(path, std::ios::binary);
  if (!o.is_open()) {
    std::fprintf(stderr, "[error] cannot open for write: %s\n", path);
    return false;
  }
  o.write(reinterpret_cast<const char*>(v.data()),
          static_cast<std::streamsize>(v.size() * sizeof(float)));
  if (!o) {
    std::fprintf(stderr, "[error] failed writing: %s\n", path);
    return false;
  }
  return true;
}

// Run `session` on a single float32 input loaded from `input_raw` and return
// the first output as float32. Dynamic input dims are pinned to 1.
inline bool RunWithFloatInput(Ort::Session& session,
                              const char* input_raw_path,
                              std::vector<float>& out_values) {
  std::vector<float> input_data;
  if (!ReadFloatRaw(input_raw_path, input_data)) return false;

  Ort::AllocatorWithDefaultOptions allocator;
  if (session.GetInputCount() != 1 || session.GetOutputCount() < 1) {
    std::fprintf(stderr,
                 "[error] expected a single-input model with at least one output, got "
                 "%zu input(s) / %zu output(s).\n",
                 session.GetInputCount(), session.GetOutputCount());
    return false;
  }

  Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, allocator);
  Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, allocator);

  Ort::TypeInfo ti = session.GetInputTypeInfo(0);
  auto tsi = ti.GetTensorTypeAndShapeInfo();
  if (tsi.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::fprintf(stderr,
                 "[error] model input 0 is not float32 — this app's real-input path "
                 "supports float32 input models only.\n");
    return false;
  }
  std::vector<int64_t> shape = tsi.GetShape();
  size_t expected = 1;
  for (auto& d : shape) {
    if (d < 0) d = 1;  // pin dynamic dims to 1
    expected *= static_cast<size_t>(d);
  }
  if (expected != input_data.size()) {
    std::fprintf(stderr,
                 "[error] %s holds %zu float(s) but model input 0 needs %zu.\n",
                 input_raw_path, input_data.size(), expected);
    return false;
  }

  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                            shape.data(), shape.size()));
  const char* in_names[] = {in_name.get()};
  const char* out_names[] = {out_name.get()};
  std::vector<Ort::Value> outputs =
      session.Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 1, out_names, 1);

  if (outputs.empty() || !outputs[0].IsTensor()) {
    std::fprintf(stderr, "[error] model produced no output tensor.\n");
    return false;
  }
  auto oi = outputs[0].GetTensorTypeAndShapeInfo();
  if (oi.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::fprintf(stderr, "[error] model output 0 is not float32.\n");
    return false;
  }
  const float* data = outputs[0].GetTensorData<float>();
  const size_t n = oi.GetElementCount();
  out_values.assign(data, data + n);
  return true;
}

// Appends the QNN EP (HTP/NPU device if present) to `session_options`.
// Assumes the plugin library is already registered on `env`.
// Returns true if an NPU (HTP) device was selected, false if it fell back to
// a compile-only CPU device.
inline bool AppendQnnEp(Ort::Env& env,
                        Ort::SessionOptions& session_options,
                        const std::unordered_map<std::string, std::string>& ep_options) {
  std::vector<Ort::ConstEpDevice> all = env.GetEpDevices();
  size_t chosen = SIZE_MAX;
  size_t npu = SIZE_MAX;
  for (size_t i = 0; i < all.size(); ++i) {
    if (std::string(all[i].EpName()) != QnnEpName()) continue;
    if (chosen == SIZE_MAX) chosen = i;
    if (all[i].Device().Type() == OrtHardwareDeviceType_NPU) npu = i;
  }
  size_t target = (npu != SIZE_MAX) ? npu : chosen;
  if (target == SIZE_MAX) {
    throw std::runtime_error("QNN EP registered but no matching Ep device was found");
  }

  if (npu != SIZE_MAX) {
    std::fprintf(stderr, "[enc_common] Selected device: NPU (HTP) — hardware execution available\n");
  } else {
    std::fprintf(stderr,
                 "[enc_common] Selected device: CPU (compile-only, no HTP hardware) — "
                 "expected on x86_64 prepare hosts; run_app on this device would fail\n");
  }

  std::vector<Ort::ConstEpDevice> devices{all[target]};
  session_options.AppendExecutionProvider_V2(env, devices, ep_options);
  return npu != SIZE_MAX;
}

// Registers the QNN EP plugin library on `env` (once per process) and appends
// it to `session_options`. Returns true if an NPU (HTP) device was selected.
inline bool RegisterAndAppendQnnEp(Ort::Env& env,
                                   Ort::SessionOptions& session_options,
                                   const std::unordered_map<std::string, std::string>& ep_options) {
  static bool registered = false;
  if (!registered) {
#if defined(_WIN32)
    const std::basic_string<ORTCHAR_T> library_path = ORT_TSTR("onnxruntime_providers_qnn.dll");
#else
    const std::basic_string<ORTCHAR_T> library_path = ORT_TSTR("libonnxruntime_providers_qnn.so");
#endif
    env.RegisterExecutionProviderLibrary(QnnEpName(), library_path);
    registered = true;
  }
  return AppendQnnEp(env, session_options, ep_options);
}

}  // namespace encapp
