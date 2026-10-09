// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for the HTP v68 FP32/FP16 blanket rejection guard
// in BaseOpBuilder::IsOpSupported.
//
// The guard lives in base_op_builder.cc and fires before any per-op validation:
//
//   if (IsNpuBackend(backend_type) && htp_arch == QNN_HTP_DEVICE_ARCH_V68) {
//     for each input/output io_def:
//       if qnn_data_type == FLOAT_32 || qnn_data_type == FLOAT_16 => reject
//   }
//
// We exercise the guard via IOpBuilder::IsOpSupported using a real registered
// op builder (Relu — single FP32 input → single FP32 output, no attributes,
// no constants, no QDQ overhead).  The stub wrapper is configured with different
// backend types and HTP arch values to hit every branch of the guard.
//
// Test suite: QnnUnit_BaseOpBuilder_V68Rejection
//
//   V68_HTP_FP32Input_IsRejected      — FP32 input  → rejected
//   V68_HTP_FP16Input_IsRejected      — FP16 input  → rejected
//   V68_HTP_FP32Output_IsRejected     — FP32 output → rejected (INT8 in, FP32 out)
//   V68_HTP_INT8_PassesThrough        — INT8 input  → passes the v68 guard
//   V73_HTP_FP32_PassesThrough        — v73 + FP32  → passes the v68 guard
//   V68_CPU_FP32_PassesThrough        — CPU backend + FP32 → passes (guard is NPU-only)
//
// Note on "passes through": IsOpSupported calls ProcessDataTypes after the v68
// guard.  With a stub wrapper, CheckHtpDataTypes / CheckCpuDataTypes both return
// OK, but AddToModelBuilder (called next) fails when it tries to compose a real
// QNN node through null function pointers.  The tests therefore only assert the
// correct error-source, not that IsOpSupported returns OK end-to-end.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include "gtest/gtest.h"

#include "HTP/QnnHtpDevice.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_def.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

namespace {

// ---------------------------------------------------------------------------
// Helper: build a stub wrapper with a specific backend type and HTP arch.
// ---------------------------------------------------------------------------
std::unique_ptr<QnnModelWrapper> MakeStubWrapper(
    OpBuilderTestContext& ctx,
    QnnBackendType backend_type,
    QnnHtpDevice_Arch_t htp_arch) {
  ctx.backend_manager.BackendType() = backend_type;
  ctx.backend_manager.HtpArch() = htp_arch;
  ModelSettings settings{};
  // CreateWrapper sets BackendType a second time, so we override HtpArch after.
  auto wrapper = ctx.CreateWrapper(settings, backend_type);
  // HtpArch is read from the BackendManager through the wrapper's GetHtpArch().
  // StubBackendManager::HtpArch() gives us the mutable reference written during
  // Create(), so setting it here (after CreateWrapper) is authoritative.
  ctx.backend_manager.HtpArch() = htp_arch;
  return wrapper;
}

// ---------------------------------------------------------------------------
// Helper: run IsOpSupported on a single-input/single-output node via Relu.
//
// Relu is an ideal vehicle: registered, no mandatory attributes, no
// constant inputs, no explicit op check beyond BaseOpBuilder's.
// input_type and output_type are ONNX element data types; they must be
// consistent with a valid FP32→FP32, FP16→FP16, or INT8→INT8 Relu graph.
// ---------------------------------------------------------------------------
Ort::Status RunIsOpSupported(
    OpBuilderTestContext& ctx,
    std::unique_ptr<QnnModelWrapper>& wrapper,
    ONNXTensorElementDataType input_type,
    ONNXTensorElementDataType output_type) {
  const IOpBuilder* builder = GetOpBuilder("Relu");
  if (builder == nullptr) {
    // If Relu is absent in this build configuration the test is meaningless.
    return Ort::Status("Relu op builder not found", ORT_EP_FAIL);
  }

  auto input = MakeMockIODef("data", input_type, std::vector<int64_t>{1, 4, 4, 3});
  auto output = MakeMockIODef("result", output_type, std::vector<int64_t>{1, 4, 4, 3});
  auto node_unit = MakeMockNodeUnit("Relu", {input}, {output}, "relu_node");

  return builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
}

// ---------------------------------------------------------------------------
// Helper: check that status message contains the v68 rejection string.
// ---------------------------------------------------------------------------
bool IsV68RejectionError(const Ort::Status& status) {
  if (status.IsOK()) return false;
  return std::string(status.GetErrorMessage()).find("HTP v68") != std::string::npos;
}

}  // namespace

// ---------------------------------------------------------------------------
// V68 + HTP + FP32 input → rejected
// ---------------------------------------------------------------------------
TEST(QnnUnit_BaseOpBuilder_V68Rejection, V68_HTP_FP32Input_IsRejected) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeStubWrapper(ctx, QnnBackendType::HTP, QNN_HTP_DEVICE_ARCH_V68);

  Ort::Status status = RunIsOpSupported(
      ctx, wrapper,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  EXPECT_FALSE(status.IsOK()) << "Expected rejection for FP32 input on HTP v68";
  EXPECT_TRUE(IsV68RejectionError(status))
      << "Expected v68 rejection message, got: " << status.GetErrorMessage();
}

// ---------------------------------------------------------------------------
// V68 + HTP + FP16 input → rejected
// ---------------------------------------------------------------------------
TEST(QnnUnit_BaseOpBuilder_V68Rejection, V68_HTP_FP16Input_IsRejected) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeStubWrapper(ctx, QnnBackendType::HTP, QNN_HTP_DEVICE_ARCH_V68);

  Ort::Status status = RunIsOpSupported(
      ctx, wrapper,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  EXPECT_FALSE(status.IsOK()) << "Expected rejection for FP16 input on HTP v68";
  EXPECT_TRUE(IsV68RejectionError(status))
      << "Expected v68 rejection message, got: " << status.GetErrorMessage();
}

// ---------------------------------------------------------------------------
// V68 + HTP + INT8 input / FP32 output → rejected (guard also checks outputs)
//
// The guard iterates inputs then outputs.  To isolate the output-side check we
// use INT8 input (which passes the input loop) and FP32 output (which the
// output loop must catch).  Relu itself is only registered for FP32, not INT8,
// so we use Cast here — it accepts INT8 input and FP32 output in the ONNX spec
// and is also registered in the BaseOpBuilder map.
// ---------------------------------------------------------------------------
TEST(QnnUnit_BaseOpBuilder_V68Rejection, V68_HTP_FP32Output_IsRejected) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeStubWrapper(ctx, QnnBackendType::HTP, QNN_HTP_DEVICE_ARCH_V68);

  const IOpBuilder* builder = GetOpBuilder("Cast");
  ASSERT_NE(builder, nullptr) << "Cast op builder not found";

  // INT8 input passes the input scan; FP32 output must be caught by the output scan.
  auto input = MakeMockIODef("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                              std::vector<int64_t>{1, 4});
  auto output = MakeMockIODef("y", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                               std::vector<int64_t>{1, 4});
  auto node_unit = MakeMockNodeUnit("Cast", {input}, {output}, "cast_node");

  Ort::Status status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);

  EXPECT_FALSE(status.IsOK()) << "Expected rejection for FP32 output on HTP v68";
  EXPECT_TRUE(IsV68RejectionError(status))
      << "Expected v68 rejection message, got: " << status.GetErrorMessage();
}

// ---------------------------------------------------------------------------
// V68 + HTP + INT8 input/output → passes the v68 guard
//
// The guard only rejects FLOAT_32 and FLOAT_16.  INT8 must not be rejected.
// IsOpSupported will fail later (ProcessDataTypes or AddToModelBuilder with
// null QNN pointers), but the failure message must NOT be the v68 string.
// ---------------------------------------------------------------------------
TEST(QnnUnit_BaseOpBuilder_V68Rejection, V68_HTP_INT8_PassesThrough) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeStubWrapper(ctx, QnnBackendType::HTP, QNN_HTP_DEVICE_ARCH_V68);

  const IOpBuilder* builder = GetOpBuilder("Relu");
  ASSERT_NE(builder, nullptr) << "Relu op builder not found";

  // INT8 input (no quant params here — stub wrapper returns a raw INT8 TensorInfo).
  auto input = MakeMockIODef("x", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                              std::vector<int64_t>{1, 4});
  auto output = MakeMockIODef("y", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                               std::vector<int64_t>{1, 4});
  auto node_unit = MakeMockNodeUnit("Relu", {input}, {output}, "relu_node");

  Ort::Status status = builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);

  // The v68 guard must NOT have fired — any failure is from a downstream stage.
  EXPECT_FALSE(IsV68RejectionError(status))
      << "v68 guard fired unexpectedly for INT8 tensor on HTP v68; "
      << "message: " << (status.IsOK() ? "(ok)" : status.GetErrorMessage());
}

// ---------------------------------------------------------------------------
// Non-V68 arch (V73) + HTP + FP32 → passes the v68 guard
//
// The guard is conditional on arch == QNN_HTP_DEVICE_ARCH_V68.  Any other arch
// must not trigger it, regardless of data type.
// ---------------------------------------------------------------------------
TEST(QnnUnit_BaseOpBuilder_V68Rejection, V73_HTP_FP32_PassesThrough) {
  OpBuilderTestContext ctx;
  auto wrapper = MakeStubWrapper(ctx, QnnBackendType::HTP, QNN_HTP_DEVICE_ARCH_V73);

  Ort::Status status = RunIsOpSupported(
      ctx, wrapper,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  EXPECT_FALSE(IsV68RejectionError(status))
      << "v68 guard fired unexpectedly for FP32 on HTP v73; "
      << "message: " << (status.IsOK() ? "(ok)" : status.GetErrorMessage());
}

// ---------------------------------------------------------------------------
// V68 arch value + CPU backend + FP32 → passes the v68 guard (NPU-only guard)
//
// The guard is further gated on IsNpuBackend().  CPU backend must not be
// affected even if someone set htp_arch to V68 on the backend manager.
// ---------------------------------------------------------------------------
TEST(QnnUnit_BaseOpBuilder_V68Rejection, V68_CPU_FP32_PassesThrough) {
  OpBuilderTestContext ctx;
  // Force arch to V68 even though the backend type is CPU, to verify the
  // IsNpuBackend() gate correctly suppresses the guard.
  auto wrapper = MakeStubWrapper(ctx, QnnBackendType::CPU, QNN_HTP_DEVICE_ARCH_V68);

  Ort::Status status = RunIsOpSupported(
      ctx, wrapper,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  EXPECT_FALSE(IsV68RejectionError(status))
      << "v68 guard fired unexpectedly for FP32 on CPU backend; "
      << "message: " << (status.IsOK() ? "(ok)" : status.GetErrorMessage());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
