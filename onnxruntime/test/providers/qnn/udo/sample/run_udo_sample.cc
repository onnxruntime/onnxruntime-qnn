// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Standalone C++ reference sample for running a QNN UDO (User-Defined Op).
//
// Demonstrates the MyAdd UDO (output = input + constant) on:
//   - QNN CPU backend  (fp32 model, no QDQ wrapping)
//   - QNN HTP backend  (QDQ uint8 model, DQ -> MyAdd -> Q fusion)
//
// Build (after building ORT and the MyAdd op package):
//   g++ -std=c++17 run_udo_sample.cc \
//       -I<ort_include_dir> \
//       -L<ort_lib_dir> -lonnxruntime \
//       -Wl,-rpath,<ort_lib_dir> \
//       -o run_udo_sample
//
// Usage:
//   # CPU backend
//   ./run_udo_sample cpu myadd_fp32.onnx /path/to/libMyAddOpPackage_cpu.so
//
//   # HTP backend (x86 simulator or on-device)
//   ./run_udo_sample htp myadd_qdq.onnx /path/to/libMyAddOpPackage_htp.so

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"

static constexpr const char* kDomain = "example";
static constexpr const char* kQnnEpName = "QNNExecutionProvider";
static constexpr float kConstant = 2.0f;
static constexpr int kElements = 32;

// ── CPU fallback kernel (provides the schema + shape-inference that ORT
//    needs to parse the model; QNN EP runs the actual kernel via the package) ──
struct MyAdd {
  MyAdd(const OrtApi* api, const OrtKernelInfo* info) {
    OrtStatus* st = api->KernelInfoGetAttribute_float(info, "constant", &constant_);
    if (st) api->ReleaseStatus(st);
  }
  Ort::Status Compute(const Ort::Custom::Tensor<float>& X,
                      Ort::Custom::Tensor<float>* Y) {
    auto shape = X.Shape();
    const float* in = X.Data();
    float* out = Y->Allocate(shape);
    for (int i = 0; i < X.NumberOfElement(); ++i)
      out[i] = in[i] + constant_;
    return Ort::Status{nullptr};
  }
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    ctx.SetOutputShape(0, ctx.GetInputShape(0));
    return Ort::Status{nullptr};
  }
  float constant_ = 1.0f;
};

// Register the QNN EP plugin library and append it to session options using
// the v2 plugin API (RegisterExecutionProviderLibrary + AppendExecutionProvider_V2).
static void AppendQnnEp(Ort::Env& env,
                        Ort::SessionOptions& so,
                        const std::unordered_map<std::string, std::string>& ep_options) {
  // Register the QNN EP shared library with the environment.
  env.RegisterExecutionProviderLibrary(kQnnEpName, "libonnxruntime_providers_qnn.so");

  // Query all registered EP devices and filter to those from the QNN EP.
  std::vector<Ort::ConstEpDevice> all_devices = env.GetEpDevices();
  std::vector<Ort::ConstEpDevice> qnn_devices;
  for (const auto& dev : all_devices) {
    if (std::string(dev.EpName()) == kQnnEpName)
      qnn_devices.push_back(dev);
  }
  if (qnn_devices.empty())
    throw std::runtime_error("No QNN EP device found after registration.");

  so.AppendExecutionProvider_V2(env, qnn_devices, ep_options);
}

static std::vector<float> make_input() {
  std::vector<float> v(kElements);
  for (int i = 0; i < kElements; ++i)
    v[i] = -1.0f + 2.0f * i / (kElements - 1);
  return v;
}

static void run_cpu(const std::string& model_path, const std::string& pkg_path) {
  printf("\n=== QNN CPU backend ===\n");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "UDOSampleCPU");
  Ort::SessionOptions so;

  // 1. Register custom-op schema so ORT can load the "example" domain model.
  Ort::CustomOpDomain domain{kDomain};
  auto op = Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider");
  domain.Add(op);
  so.Add(domain);

  // 2. Register QNN EP and configure with op_packages.
  std::string op_packages = "MyAdd:" + pkg_path + ":MyAddOpPackageInterfaceProvider";
  AppendQnnEp(env, so, {{"backend_type", "cpu"}, {"op_packages", op_packages}});

  Ort::Session session(env, model_path.c_str(), so);

  // 3. Run inference with float32 input in [-1, 1].
  auto input = make_input();
  std::vector<int64_t> shape = {1, kElements};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto in_tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                   shape.data(), shape.size());
  const char* in_names[] = {"input"};
  const char* out_names[] = {"output"};
  auto outputs = session.Run(Ort::RunOptions{}, in_names, &in_tensor, 1, out_names, 1);

  // 4. Verify output ≈ input + constant.
  const float* out = outputs[0].GetTensorData<float>();
  float max_err = 0.0f;
  for (int i = 0; i < kElements; ++i)
    max_err = std::max(max_err, std::fabs(out[i] - (input[i] + kConstant)));

  printf("Max absolute error vs (input + %.1f): %.2e\n", kConstant, max_err);
  printf(max_err <= 1e-4f ? "PASS\n" : "FAIL: error exceeds threshold\n");
}

static void run_htp(const std::string& model_path, const std::string& pkg_path) {
  printf("\n=== QNN HTP backend ===\n");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "UDOSampleHTP");
  Ort::SessionOptions so;

  // 1. Register custom-op schema.
  Ort::CustomOpDomain domain{kDomain};
  auto op = Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider");
  domain.Add(op);
  so.Add(domain);

  // 2. Register QNN HTP EP.
  std::string op_packages = "MyAdd:" + pkg_path + ":MyAddOpPackageInterfaceProvider:CPU";
  AppendQnnEp(env, so, {{"backend_type", "htp"},
                        {"offload_graph_io_quantization", "0"},
                        {"op_packages", op_packages}});

  Ort::Session session(env, model_path.c_str(), so);

  // 3. Run inference.
  auto input = make_input();
  std::vector<int64_t> shape = {1, kElements};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto in_tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                   shape.data(), shape.size());
  const char* in_names[] = {"input"};
  const char* out_names[] = {"output"};
  auto outputs = session.Run(Ort::RunOptions{}, in_names, &in_tensor, 1, out_names, 1);

  // 4. Verify within QDQ tolerance (~2 quantization steps for uint8 over [-1,1]).
  const float* out = outputs[0].GetTensorData<float>();
  float max_err = 0.0f;
  for (int i = 0; i < kElements; ++i)
    max_err = std::max(max_err, std::fabs(out[i] - (input[i] + kConstant)));

  // QDQ tolerance: 2x the output quantization scale (4/255, covering [0,4] output range).
  const float qdq_tol = 4.0f / 255.0f * 2;
  printf("Max absolute error vs (input + %.1f): %.4f  (QDQ tol: %.4f)\n",
         kConstant, max_err, qdq_tol);
  printf(max_err <= qdq_tol ? "PASS\n" : "FAIL: error exceeds QDQ tolerance\n");
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage:\n"
            "  %s cpu  <model.onnx> <libMyAddOpPackage_cpu.so>\n"
            "  %s htp  <model.onnx> <libMyAddOpPackage_htp.so>\n",
            argv[0], argv[0]);
    return 1;
  }

  std::string backend = argv[1];
  std::string model   = argv[2];
  std::string pkg     = argv[3];

  try {
    if (backend == "cpu")       run_cpu(model, pkg);
    else if (backend == "htp")  run_htp(model, pkg);
    else {
      fprintf(stderr, "Unknown backend '%s'. Use 'cpu' or 'htp'.\n", backend.c_str());
      return 1;
    }
  } catch (const Ort::Exception& e) {
    fprintf(stderr, "ORT error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }

  return 0;
}
