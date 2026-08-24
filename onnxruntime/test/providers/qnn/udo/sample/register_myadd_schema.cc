// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Minimal RegisterCustomOps shared library for Python users.
//
// ORT needs a schema for the custom-domain "example::MyAdd" node just to *load*
// the ONNX model.  In C++ this is done inline via Ort::CustomOpDomain; Python
// cannot do that, so this companion library provides the same schema + CPU
// fallback compute and is loaded via:
//
//   session_options.register_custom_ops_library("libMyAddSchema.so")
//
// The actual QNN execution comes from the separately-built QNN op-package
// (.so), supplied through the "op_packages" provider option.  This library
// only handles the CPU (fallback) path.

// ORT_API_MANUAL_INIT must be defined before including onnxruntime_cxx_api.h
// in a shared library that exports RegisterCustomOps, so that Ort::InitApi()
// is available to initialize the global API pointer at load time.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"

// Domain that matches MyAddOpPackageCpu.xml / MyAddOpPackageHtp.xml
static constexpr const char* kDomain = "example";

struct MyAdd {
  MyAdd(const OrtApi* ort_api, const OrtKernelInfo* info) {
    // 'constant' is optional: keep default 1.0 when attribute is absent.
    OrtStatus* status = ort_api->KernelInfoGetAttribute_float(info, "constant", &constant_);
    if (status != nullptr) {
      ort_api->ReleaseStatus(status);
    }
  }

  Ort::Status Compute(const Ort::Custom::Tensor<float>& X,
                      Ort::Custom::Tensor<float>* Y) {
    const std::vector<int64_t>& shape = X.Shape();
    const float* in = X.Data();
    float* out = Y->Allocate(shape);
    for (int i = 0; i < X.NumberOfElement(); i++) {
      out[i] = in[i] + constant_;
    }
    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    ctx.SetOutputShape(0, ctx.GetInputShape(0));
    return Ort::Status{nullptr};
  }

  float constant_ = 1.0f;
};

// Called by OrtSessionOptions::RegisterCustomOpsLibrary.
// Signature must match the ORT_API_CALL convention used by register_custom_ops_library.
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
  Ort::InitApi(api_base->GetApi(ORT_API_VERSION));

  // Keep the op object alive for the process lifetime — ORT holds a pointer to it
  // through the domain, so it must not be freed after RegisterCustomOps returns.
  static Ort::Custom::OrtLiteCustomOp* op =
      Ort::Custom::CreateLiteCustomOp<MyAdd>("MyAdd", "CPUExecutionProvider");

  Ort::CustomOpDomain domain{kDomain};
  domain.Add(op);
  Ort::ThrowOnError(Ort::GetApi().AddCustomOpDomain(options, domain));
  // Transfer domain ownership to ORT; do not let the RAII destructor free it.
  domain.release();
  return nullptr;
}

