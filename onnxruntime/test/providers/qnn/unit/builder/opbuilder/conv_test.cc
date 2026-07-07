// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Component-level tests for ConvOpBuilder — QDQ / HTP paths.
//
// Suite:
//   QnnUnit_Component_ConvTest  — IsOpSupported pass/fail tests (stub HTP wrapper,
//                                 no snapshot).
//
// All Conv QDQ snapshot coverage is in conv_session_test.cc (Phase B):
// session-level snapshots correctly exercise the NCHW→NHWC layout transform
// that op-builder-level snapshots would bypass.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/unit/qnn_unit_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

namespace {

// ---------------------------------------------------------------------------
// IsOpSupported helper (stub HTP wrapper, no real QNN backend needed)
// ---------------------------------------------------------------------------

// Returns the Ort::Status from IsOpSupported for a 2D Conv/ConvTranspose with
// per-channel quantized weights at the given axis.
// Uses a stub (non-real) HTP wrapper — no libQnnHtp.so connection required.
Ort::Status RunIsOpSupportedPerChanAxis(const std::string& op_type,
                                         int64_t weight_quant_axis,
                                         int num_out_channels = 8) {
  g_mock_init_reg.clear();

  // activation (U8): scalar scale (per-tensor)
  auto act_scale_vi = g_mock_init_reg.AddScalarFloat("act_scale", 0.05f);
  auto act_zp_vi = g_mock_init_reg.AddScalarUint8("act_zp", 128);

  // weights (S8): per-channel scale tensor
  std::vector<float> w_scales(num_out_channels, 0.01f);
  MockInitSpec ws;
  ws.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  ws.dims = {static_cast<int64_t>(num_out_channels)};
  ws.raw_bytes.resize(num_out_channels * sizeof(float));
  std::memcpy(ws.raw_bytes.data(), w_scales.data(), num_out_channels * sizeof(float));
  auto w_scale_vi = g_mock_init_reg.Add("w_scale", std::move(ws));
  auto w_zp_vi = g_mock_init_reg.AddScalarInt8("w_zp", 0);

  SnapshotTestContext ctx;
  SetupMockInitRegistryStubs(ctx);

  qnn::ModelSettings settings{};
  auto wrapper = ctx.CreateWrapper(settings, qnn::QnnBackendType::HTP);
  EXPECT_NE(wrapper, nullptr);

  // input0: U8 NCHW [1, 4, 5, 5] (rank 4 — pre-NHWC for IsOpSupported NCHW path)
  auto input0 = MakeMockQDQIODef("input", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                                  {{1, 4, 5, 5}}, act_scale_vi, act_zp_vi);
  // input1: S8 per-channel weight NCHW [out_C, in_C=4, kH=3, kW=3]
  auto input1 = MakeMockQDQIODef("weights", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                                  {{static_cast<int64_t>(num_out_channels), 4, 3, 3}},
                                  w_scale_vi, w_zp_vi,
                                  weight_quant_axis);

  auto node_unit = MakeMockNodeUnit(op_type, {input0, input1}, {}, "conv_node");

  const IOpBuilder* builder = GetOpBuilder("Conv");
  EXPECT_NE(builder, nullptr);
  return builder->IsOpSupported(*wrapper, node_unit, ctx.ort_logger);
}

}  // namespace

// ---------------------------------------------------------------------------
// QnnUnit_Component_ConvTest — IsOpSupported per-channel axis validation
// Covers conv_op_builder.cc lines 117-129.
//
// For Conv:         per-channel weight axis must be 0 (output channels in NCHW).
// For ConvTranspose: per-channel weight axis must be 1 (output channels in CNHW).
// ---------------------------------------------------------------------------

TEST(QnnUnit_Component_ConvTest, IsOpSupported_NPU_PerChannelWeights_Conv_AxisZero_OK) {
  auto status = RunIsOpSupportedPerChanAxis("Conv", /*axis=*/0);
  EXPECT_TRUE(status.IsOK()) << "Expected OK for Conv per-channel axis=0, got: "
                              << status.GetErrorMessage();
}

TEST(QnnUnit_Component_ConvTest, IsOpSupported_NPU_PerChannelWeights_Conv_AxisOne_Fails) {
  auto status = RunIsOpSupportedPerChanAxis("Conv", /*axis=*/1);
  EXPECT_FALSE(status.IsOK()) << "Expected error for Conv per-channel axis=1";
}

TEST(QnnUnit_Component_ConvTest, IsOpSupported_NPU_PerChannelWeights_ConvTranspose_AxisOne_OK) {
  auto status = RunIsOpSupportedPerChanAxis("ConvTranspose", /*axis=*/1);
  EXPECT_TRUE(status.IsOK()) << "Expected OK for ConvTranspose per-channel axis=1, got: "
                              << status.GetErrorMessage();
}

TEST(QnnUnit_Component_ConvTest, IsOpSupported_NPU_PerChannelWeights_ConvTranspose_AxisZero_Fails) {
  auto status = RunIsOpSupportedPerChanAxis("ConvTranspose", /*axis=*/0);
  EXPECT_FALSE(status.IsOK()) << "Expected error for ConvTranspose per-channel axis=0";
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
