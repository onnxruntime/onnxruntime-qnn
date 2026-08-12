// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level unit tests for ClipOpBuilder.
//
//   QnnUnit_Clip_ComponentTest — ClipOpBuilder white-box tests:
//                                 * partition reject path (dynamic min/max,
//                                   do_op_validation=true, no QNN backend)
//                                 * ProcessClipMinMax dtype dispatch, driven
//                                   via AddToModelBuilder so we don't touch
//                                   production op-builder code
//
// Coverage rationale: component tests fill ProcessClipMinMax switch arms that
// the snapshot tier cannot reach (each dtype dispatch arm needs one case).

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstring>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/test_infra/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Build a minimal wrapper using stub ORT API (no real backend).
std::unique_ptr<QnnModelWrapper> MakeStubWrapper(OpBuilderTestContext& ctx) {
  ModelSettings settings{};
  return ctx.CreateWrapper(settings);
}

}  // namespace

// ---------------------------------------------------------------------------
// Component tests — partition reject path (QnnUnit_Clip_ComponentTest)
//
// Verify that ClipOpBuilder rejects inputs that QNN EP cannot handle at
// partitioning time. No QNN backend session is needed — the check is purely
// in ExplicitOpCheck (called from ProcessInputs with do_op_validation=true).
// Paired with the dtype-dispatch cases below: same builder, accept path.
// ---------------------------------------------------------------------------

// Dynamic (non-initializer) min or max input must be rejected.
// Each sub-assertion simulates ExplicitOpCheck for one unsupported configuration.
TEST(QnnUnit_Clip_ComponentTest, Clip_Dynamic_MinMax_Unsupported) {
  const IOpBuilder* builder = GetOpBuilder("Clip");
  ASSERT_NE(builder, nullptr);

  // Case 1: dynamic min input (non-empty name, not an initializer).
  // With the stub graph, IsConstantInput always returns false, so any named
  // min/max input is treated as dynamic.
  {
    OpBuilderTestContext ctx;
    auto wrapper = MakeStubWrapper(ctx);

    auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 3, 4, 4});
    auto min_dyn = MakeMockIODef("min_dynamic", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                 std::vector<int64_t>{});  // scalar
    auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                std::vector<int64_t>{1, 3, 4, 4});

    auto node_unit = MakeMockNodeUnit("Clip", {data, min_dyn}, {output}, "clip_node");

    // do_op_validation=true triggers ExplicitOpCheck — should fail.
    auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger,
                                             /*do_op_validation=*/true);
    EXPECT_FALSE(status.IsOK()) << "Expected rejection for dynamic min input";
  }

  // Case 2: dynamic max input (min is absent/empty, max is dynamic).
  {
    OpBuilderTestContext ctx;
    auto wrapper = MakeStubWrapper(ctx);

    auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 3, 4, 4});
    // min is absent (empty name = optional input not provided).
    auto min_absent = MakeMockIODef("", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, std::nullopt);
    auto max_dyn = MakeMockIODef("max_dynamic", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                 std::vector<int64_t>{});  // scalar
    auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                std::vector<int64_t>{1, 3, 4, 4});

    auto node_unit = MakeMockNodeUnit("Clip", {data, min_absent, max_dyn}, {output}, "clip_node");

    auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger,
                                             /*do_op_validation=*/true);
    EXPECT_FALSE(status.IsOK()) << "Expected rejection for dynamic max input";
  }
}

// Backend-validation success path (real HTP). Complements the reject test above:
// drives AddToModelBuilder(do_op_validation=true) against a live HTP validator so
// both ExplicitOpCheck's accept path (constant min/max) and QNN's
// backendValidateOpConfig are exercised end-to-end. Constant min/max come from the
// mock init registry (its IsConstantInitializer stub reports true), so they are not
// treated as dynamic. Skipped when libQnnHtp.so is unavailable (a missing SDK on a
// local box degrades to skip rather than fail; CI images always ship it).
//
// Kept separate from the dtype-dispatch tests below, which run
// do_op_validation=false against a stub wrapper — validation and dispatch are
// orthogonal, so coupling them would obscure which concern a failure belongs to.
TEST(QnnUnit_Clip_ComponentTest, Clip_BackendValidation_Htp) {
  QnnRealHtpBackendContext backend;
  if (!backend.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";

  const IOpBuilder* builder = GetOpBuilder("Clip");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarFloat("min", -1.0f);
  g_mock_init_reg.AddScalarFloat("max", 1.0f);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  ctx.qnn_validator_interface = backend.qnn_interface;
  ctx.validator_backend_handle = backend.backend_handle;
  ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings);

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{1, 3, 4, 4});
  auto min = MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{});
  auto max = MakeMockIODef("max", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, std::vector<int64_t>{});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 3, 4, 4});
  auto node_unit = MakeMockNodeUnit("Clip", {data, min, max}, {output}, "clip_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger,
                                           /*do_op_validation=*/true);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

// ---------------------------------------------------------------------------
// Component tests — ProcessClipMinMax dtype dispatch (QnnUnit_Clip_ComponentTest)
//
// Drive the static helper ProcessClipMinMax (clip_op_builder.cc:36) end-to-end
// via AddToModelBuilder(do_op_validation=false). Each switch arm needs one
// test — too many dtype combinations to express via snapshot tests alone.
//
// Setup pattern: data is plain FP32 (Inputs()[0], processed by ProcessInput),
// min is a scalar of the target dtype (Inputs()[1], processed by
// ProcessClipMinMax during ProcessAttributesAndOutputs). Quantized variants
// pass scale/zp via MakeMockQDQIODef.
//
// Coverage scope: switch arms in clip_op_builder.cc lines 50-94 (quantized)
// and 98-141 (non-quantized). UFIXED_POINT_8/16 and FLOAT_32 are also
// covered by the snapshot suite — keep here for completeness.
//
// Coverage gap: ExplicitOpCheck success path (line 159) — only the failure
// path is exercised (Clip_Dynamic_MinMax_Unsupported partition test). Driving
// it via AddToModelBuilder(do_op_validation=true) crashes with the stub
// wrapper; driving it via the snapshot wrapper (real backend) would work
// but mixing fallback assertions into snapshot tests muddies test intent.
// Documented gap; revisit when a cheap mock for ProcessInput exists.
// ---------------------------------------------------------------------------

namespace {

// Run ProcessClipMinMax dispatch for the given min IODef. Caller must register
// referenced initializer names in g_mock_init_reg before calling.
void RunClipMinMaxDispatchPass(OrtNodeUnitIODef min_iodef) {
  const IOpBuilder* builder = GetOpBuilder("Clip");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  auto wrapper = MakeStubWrapper(ctx);

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{1, 3, 4, 4});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 3, 4, 4});
  auto node_unit = MakeMockNodeUnit("Clip", {data, min_iodef}, {output}, "clip_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

// Variant that expects rejection (default-error switch arms).
void RunClipMinMaxDispatchFail(OrtNodeUnitIODef min_iodef) {
  const IOpBuilder* builder = GetOpBuilder("Clip");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  auto wrapper = MakeStubWrapper(ctx);

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{1, 3, 4, 4});
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 3, 4, 4});
  auto node_unit = MakeMockNodeUnit("Clip", {data, min_iodef}, {output}, "clip_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  EXPECT_FALSE(status.IsOK()) << "Expected rejection for unsupported min dtype";
}

// Quantized dispatch helper: register float scale + dtype-matched zp + dtype-matched min raw,
// then drive ProcessClipMinMax via AddToModelBuilder. Covers the SFIXED/UFIXED 8/16/32 arms.
//
// Unlike the non-quant path, the quantized min must reach ProcessClipMinMax with
// its quant_param intact. A SingleNode NodeUnit ctor drops quant_param, so this
// builds a QDQGroup NodeUnit (MakeMockQDQNodeUnit) — the min input is fed by a
// synthesized DequantizeLinear, exactly as in a real QDQ Clip partition — which
// preserves the scale/zp the op-builder dispatches on.
void RunClipQuantDispatchPass(ONNXTensorElementDataType dtype,
                              float scale, uint32_t zp, uint32_t raw_min) {
  g_mock_init_reg.clear();
  auto scale_vi = g_mock_init_reg.AddScalarFloat("scale", scale);
  const OrtValueInfo* zp_vi = nullptr;
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      zp_vi = g_mock_init_reg.AddScalarInt8("zp", static_cast<int8_t>(zp));
      g_mock_init_reg.AddScalarFixedPoint8("min", dtype, static_cast<uint8_t>(raw_min));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      zp_vi = g_mock_init_reg.AddScalarInt16("zp", static_cast<int16_t>(zp));
      g_mock_init_reg.AddScalarFixedPoint16("min", dtype, static_cast<uint16_t>(raw_min));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      zp_vi = g_mock_init_reg.AddScalarInt32("zp", static_cast<int32_t>(zp));
      g_mock_init_reg.AddScalarFixedPoint32("min", dtype, raw_min);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      zp_vi = g_mock_init_reg.AddScalarUint8("zp", static_cast<uint8_t>(zp));
      g_mock_init_reg.AddScalarUint8("min", static_cast<uint8_t>(raw_min));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      zp_vi = g_mock_init_reg.AddScalarUint16("zp", static_cast<uint16_t>(zp));
      g_mock_init_reg.AddScalarUint16("min", static_cast<uint16_t>(raw_min));
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      zp_vi = g_mock_init_reg.AddScalarUint32("zp", zp);
      g_mock_init_reg.AddScalarUint32("min", raw_min);
      break;
    default:
      FAIL() << "Unsupported quant dtype";
  }

  const IOpBuilder* builder = GetOpBuilder("Clip");
  ASSERT_NE(builder, nullptr);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  auto wrapper = MakeStubWrapper(ctx);

  auto data = MakeMockIODef("data", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                            std::vector<int64_t>{1, 3, 4, 4});
  auto min_quant = MakeMockQDQIODef("min", dtype, std::vector<int64_t>{}, scale_vi, zp_vi);
  auto output = MakeMockIODef("output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                              std::vector<int64_t>{1, 3, 4, 4});
  auto node_unit = MakeMockQDQNodeUnit("Clip", {data, min_quant}, {output}, "clip_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
}

}  // namespace

// --- Non-quantized dispatch (lines 96-143 in clip_op_builder.cc) ---

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_FP32) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarFloat("min", -2.5f);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_FP16) {
  // FP16 bit pattern for 1.5: 0 01111 1000000000 = 0x3E00.
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarFloat16("min", 0x3E00);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_INT8) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarInt8("min", -7);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_INT16) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarInt16("min", -300);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_INT32) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarInt32("min", -100000);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_INT64) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarInt64("min", -123456789LL);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_UINT8) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarUint8("min", 250);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_UINT16) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarUint16("min", 60000);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_UINT32) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarUint32("min", 4000000000U);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
                                          std::vector<int64_t>{}));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_UINT64) {
  g_mock_init_reg.clear();
  g_mock_init_reg.AddScalarUint64("min", 1234567890123ULL);
  RunClipMinMaxDispatchPass(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
                                          std::vector<int64_t>{}));
}

// --- Quantized dispatch (lines 50-94 in clip_op_builder.cc) ---

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_QUANT_INT8) {
  RunClipQuantDispatchPass(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, /*scale=*/0.1f,
                           /*zp=*/0, /*raw_min=*/static_cast<uint32_t>(-50));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_QUANT_INT16) {
  RunClipQuantDispatchPass(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, /*scale=*/0.01f,
                           /*zp=*/0, /*raw_min=*/static_cast<uint32_t>(-200));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_QUANT_INT32) {
  RunClipQuantDispatchPass(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, /*scale=*/0.001f,
                           /*zp=*/0, /*raw_min=*/static_cast<uint32_t>(-1500));
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_QUANT_UINT8) {
  RunClipQuantDispatchPass(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, /*scale=*/0.1f,
                           /*zp=*/128, /*raw_min=*/178);
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_QUANT_UINT16) {
  RunClipQuantDispatchPass(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, /*scale=*/0.01f,
                           /*zp=*/32768, /*raw_min=*/33068);
}

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_QUANT_UINT32) {
  RunClipQuantDispatchPass(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, /*scale=*/0.001f,
                           /*zp=*/2147483648U, /*raw_min=*/2147485148U);
}

// --- Default-error path ---

TEST(QnnUnit_Clip_ComponentTest, ProcessClipMinMax_NonQuant_UnsupportedDtype) {
  // BOOL is not in the non-quantized switch → default error.
  g_mock_init_reg.clear();
  MockInitSpec spec;
  spec.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  spec.raw_bytes = {1};
  g_mock_init_reg.Add("min", spec);
  RunClipMinMaxDispatchFail(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
                                          std::vector<int64_t>{}));
}

// Note: the quantized default-error path (lines 91-94) is unreachable — the
// `if (input.quant_param.has_value())` branch is gated by
// MakeMockQDQIODef, and GetTensorInfo only assigns SFIXED/UFIXED_POINT_{8,16,32}
// when a quant_param is present. All six are explicitly handled above.

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
