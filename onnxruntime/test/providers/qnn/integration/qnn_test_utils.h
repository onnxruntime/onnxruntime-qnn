// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared test utilities for QNN EP pipeline integration tests.
//
// Provides:
//   - RegisteredQnnEp: RAII helper that registers / unregisters the QNN EP plugin.
//   - MakeQnnHtpSessionOptions: builds Ort::SessionOptions targeting the QNN HTP backend.
//   - MakeValueInfo1D / 2D / 3D / 4D: builders for Ort::ValueInfo with the given rank.
//
// Guarded the same way as integration test files: only compiled on Linux non-minimal
// builds. See integration/README.md for the file-structure conventions.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && defined(__linux__)

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// RAII: registers the QNN EP plugin on construction, unregisters on destruction.
struct RegisteredQnnEp {
  std::string name;
  bool valid = false;

  explicit RegisteredQnnEp(const std::string& registration_name) : name(registration_name) {
    const ORTCHAR_T* kLibPath = ORT_TSTR("libonnxruntime_providers_qnn.so");
    try {
      ort_env->RegisterExecutionProviderLibrary(name.c_str(), kLibPath);
      valid = true;
    } catch (const Ort::Exception&) {
    }
  }

  ~RegisteredQnnEp() {
    if (valid) {
      try {
        ort_env->UnregisterExecutionProviderLibrary(name.c_str());
      } catch (const Ort::Exception&) {
      }
    }
  }

  RegisteredQnnEp(const RegisteredQnnEp&) = delete;
  RegisteredQnnEp& operator=(const RegisteredQnnEp&) = delete;
};

// Build Ort::SessionOptions targeting the QNN HTP backend.
// On Linux x86-64, libQnnHtp.so runs as a simulator and supports full graph
// compilation and execution. On Linux AArch64, real HTP hardware is used.
// Returns false if the HTP device is not found (libQnnHtp.so unavailable).
inline bool MakeQnnHtpSessionOptions(const RegisteredQnnEp& ep, Ort::SessionOptions& out_opts) {
  const OrtApi& api = Ort::GetApi();
  std::vector<Ort::ConstEpDevice> ep_devices = ort_env->GetEpDevices();

  // On Linux x86-64, the HTP simulator registers as OrtHardwareDeviceType_CPU.
  const OrtEpDevice* target = nullptr;
  for (const Ort::ConstEpDevice& dev : ep_devices) {
    if (api.EpDevice_EpName(dev) != ep.name) continue;
    if (api.HardwareDevice_Type(api.EpDevice_Device(dev)) != OrtHardwareDeviceType_CPU) continue;
    target = dev;
    break;
  }
  if (!target) return false;

  const std::unordered_map<std::string, std::string> provider_opts{{"backend_path", "libQnnHtp.so"}};
  try {
    out_opts.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(target)}, provider_opts);
  } catch (const Ort::Exception&) {
    return false;
  }
  return true;
}

// Build an Ort::ValueInfo for a 1D tensor of the given element type and size.
inline Ort::ValueInfo MakeValueInfo1D(const char* name, ONNXTensorElementDataType elem_type, int64_t dim) {
  const OrtModelEditorApi* ed = Ort::GetApi().GetModelEditorApi();

  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorTypeAndShapeInfo(&shape_info));
  Ort::ThrowOnError(Ort::GetApi().SetTensorElementType(shape_info, elem_type));
  Ort::ThrowOnError(Ort::GetApi().SetDimensions(shape_info, &dim, 1));

  OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ed->CreateTensorTypeInfo(shape_info, &type_info));
  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(shape_info);

  OrtValueInfo* vi = nullptr;
  Ort::ThrowOnError(ed->CreateValueInfo(name, type_info, &vi));
  Ort::GetApi().ReleaseTypeInfo(type_info);
  return Ort::ValueInfo(vi);
}

// Build an Ort::ValueInfo for a 2D tensor.
inline Ort::ValueInfo MakeValueInfo2D(const char* name, ONNXTensorElementDataType elem_type,
                                      int64_t d0, int64_t d1) {
  const OrtModelEditorApi* ed = Ort::GetApi().GetModelEditorApi();

  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorTypeAndShapeInfo(&shape_info));
  Ort::ThrowOnError(Ort::GetApi().SetTensorElementType(shape_info, elem_type));
  int64_t dims[] = {d0, d1};
  Ort::ThrowOnError(Ort::GetApi().SetDimensions(shape_info, dims, 2));

  OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ed->CreateTensorTypeInfo(shape_info, &type_info));
  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(shape_info);

  OrtValueInfo* vi = nullptr;
  Ort::ThrowOnError(ed->CreateValueInfo(name, type_info, &vi));
  Ort::GetApi().ReleaseTypeInfo(type_info);
  return Ort::ValueInfo(vi);
}

// Build an Ort::ValueInfo for a 3D tensor.
inline Ort::ValueInfo MakeValueInfo3D(const char* name, ONNXTensorElementDataType elem_type,
                                      int64_t d0, int64_t d1, int64_t d2) {
  const OrtModelEditorApi* ed = Ort::GetApi().GetModelEditorApi();

  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorTypeAndShapeInfo(&shape_info));
  Ort::ThrowOnError(Ort::GetApi().SetTensorElementType(shape_info, elem_type));
  int64_t dims[] = {d0, d1, d2};
  Ort::ThrowOnError(Ort::GetApi().SetDimensions(shape_info, dims, 3));

  OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ed->CreateTensorTypeInfo(shape_info, &type_info));
  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(shape_info);

  OrtValueInfo* vi = nullptr;
  Ort::ThrowOnError(ed->CreateValueInfo(name, type_info, &vi));
  Ort::GetApi().ReleaseTypeInfo(type_info);
  return Ort::ValueInfo(vi);
}

// Build an Ort::ValueInfo for a 4D tensor.
inline Ort::ValueInfo MakeValueInfo4D(const char* name, ONNXTensorElementDataType elem_type,
                                      int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  const OrtModelEditorApi* ed = Ort::GetApi().GetModelEditorApi();

  OrtTensorTypeAndShapeInfo* shape_info = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorTypeAndShapeInfo(&shape_info));
  Ort::ThrowOnError(Ort::GetApi().SetTensorElementType(shape_info, elem_type));
  int64_t dims[] = {d0, d1, d2, d3};
  Ort::ThrowOnError(Ort::GetApi().SetDimensions(shape_info, dims, 4));

  OrtTypeInfo* type_info = nullptr;
  Ort::ThrowOnError(ed->CreateTensorTypeInfo(shape_info, &type_info));
  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(shape_info);

  OrtValueInfo* vi = nullptr;
  Ort::ThrowOnError(ed->CreateValueInfo(name, type_info, &vi));
  Ort::GetApi().ReleaseTypeInfo(type_info);
  return Ort::ValueInfo(vi);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && defined(__linux__)
