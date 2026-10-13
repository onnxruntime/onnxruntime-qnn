// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Op-builder snapshot tests for ClipOpBuilder.
//
//   QnnSnapshot_Clip_OpBuilderTest — Snapshot tests: JSON golden compare via
//                               QnnJSONGraph (real QNN backend, no finalize,
//                               no fusion).
//
// Compare QnnJSONGraph (populated by ComposeQnnGraph(true) via real QNN
// graphAddNode, no finalize) against a stored golden .json file. Each case's
// name (and golden basename) is `spec.name` from clip_specs.h — the single
// source of truth shared with the paired accuracy suite, so snapshot ↔
// accuracy case names cannot desync.
//
// Lives in its own translation unit (separate from the component tier's
// clip_test.cc) so the snapshot harness (snapshot.h: MakeSnapshotWrapper*,
// AssertSnapshotJson) is only pulled where it is used. Component tests that
// exercise ProcessClipMinMax dispatch don't need it.
//
// Golden files: $QNN_UT_SNAPSHOT_GOLDEN_DIR/snapshot/builder/opbuilder/clip/<name>.json
// To generate/update (both env vars on one line):
//   QNN_UT_SNAPSHOT_GOLDEN_DIR=<dir> QNN_UT_SNAPSHOT_GOLDEN_UPDATE=1 ./onnxruntime_provider_test --gtest_filter="QnnSnapshot_Clip_OpBuilder*"

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "test/providers/qnn/infra/specs/builder/opbuilder/clip_specs.h"
#include "test/providers/qnn/infra/qnn_unit_test_utils.h"
#include "test/providers/qnn/snapshot/snapshot.h"

using namespace onnxruntime;
using namespace onnxruntime::qnn;

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Snapshot tests — value-parameterized, one suite per spec kind:
//   QnnSnapshot_Clip_OpBuilderTest             (plain Clip inputs)
//   QnnSnapshot_Clip_OpBuilder_QDQFloatTest     (QDQ data + float min/max)
//   QnnSnapshot_Clip_OpBuilder_QDQQuantTest     (QDQ data + quantized min/max)
//
// Three helpers cover the test patterns:
//   RunClipSnapshot               — Plain dtype data, optional float min/max
//   RunClipSnapshotQDQFloatMinMax — QDQ data + optional float min/max scalars
//   RunClipSnapshotQDQQuantMinMax — QDQ data + optional quantized min/max scalars
//
// Every case runs on HTP — QnnCpu is no longer shipped with the QNN EP
// wheel. The `SnapshotBackend` enum is kept (with only HTP as a value) so
// per-spec `snapshot_backend` / `accuracy_backend` fields stay in place for
// future backend flexibility.
//
// QDQ data with default min/max is covered by the session-snapshot tier
// (session_snapshot/builder/opbuilder/clip_test.cc), not this op-builder
// snapshot tier.
// ---------------------------------------------------------------------------

namespace {

// SnapshotBackend + QdqDataSpec + QuantScalarSpec lifted to clip_specs.h
// (shared with the accuracy tier).

// Register `<prefix>_scale` (float) + `<prefix>_zp` (uint8 or uint16 by dtype)
// in g_mock_init_reg. Returns the OrtValueInfo* sentinels.
std::pair<const OrtValueInfo*, const OrtValueInfo*>
RegisterQdqScaleZp(const std::string& prefix,
                   ONNXTensorElementDataType qdq_dtype,
                   float scale, uint32_t zp) {
  auto scale_vi = g_mock_init_reg.AddScalarFloat(prefix + "_scale", scale);
  const OrtValueInfo* zp_vi = nullptr;
  if (qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    zp_vi = g_mock_init_reg.AddScalarUint8(prefix + "_zp", static_cast<uint8_t>(zp));
  } else if (qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    zp_vi = g_mock_init_reg.AddScalarUint16(prefix + "_zp", static_cast<uint16_t>(zp));
  } else {
    EXPECT_TRUE(false) << "Unsupported QDQ dtype for ZP setup";
  }
  return {scale_vi, zp_vi};
}

// Register `<name>` as a quantized scalar of width matching qdq_dtype.
void RegisterQuantScalar(const std::string& name,
                         ONNXTensorElementDataType qdq_dtype, uint32_t raw) {
  if (qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    g_mock_init_reg.AddScalarUint8(name, static_cast<uint8_t>(raw));
  } else if (qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    g_mock_init_reg.AddScalarUint16(name, static_cast<uint16_t>(raw));
  } else {
    EXPECT_TRUE(false) << "Unsupported QDQ dtype for quant scalar";
  }
}

// Build a wrapper backed by a real HTP backend. GTEST_SKIP is invoked when
// the backend library is unavailable. Inlined into each Run* helper (cannot
// be extracted into a non-void function — GTEST_SKIP only works in
// void-returning callers).
//
// `backend` is currently always SnapshotBackend::HTP (QnnCpu is no longer
// shipped with the wheel) but is threaded through from the spec for
// interface stability if another backend is reintroduced later.
//
// Pattern (callers):
//   OpBuilderTestContext ctx;
//   ... mock registry setup if needed ...
//   QnnRealHtpBackendManagerContext htp;
//   if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
//   auto wrapper = MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
//   ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph";

// Plain dtype data with optional float min/max scalar input(s).
//
// min/max stored in spec as `std::optional<float>` regardless of dtype; this
// helper casts to the actual dtype when registering the scalar initializer
// (INT32 truncates, FP16 rounds via Ort::Float16_t). Mirrors the integration-tier
// `Clip_*` graph structure so snapshot ↔ accuracy ↔ integration test the same
// graph.
void RunClipSnapshot([[maybe_unused]] SnapshotBackend backend,
                     ONNXTensorElementDataType dtype,
                     std::vector<int64_t> shape,
                     std::optional<float> min_val,
                     std::optional<float> max_val,
                     const char* golden_basename) {
  const IOpBuilder* builder = GetOpBuilder("Clip");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();

  // Register min/max scalars as initializers of the graph data dtype (so the
  // mock NodeUnit IODef references them correctly). `min` slot may be absent
  // when only `max` is set — represented downstream as an empty-name input.
  auto register_dtype_scalar = [dtype](const std::string& name, float v) {
    switch (dtype) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        g_mock_init_reg.AddScalarFloat(name, v);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        g_mock_init_reg.AddScalarInt32(name, static_cast<int32_t>(v));
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
        const Ort::Float16_t f16(v);
        uint16_t bits;
        std::memcpy(&bits, &f16, sizeof(uint16_t));
        g_mock_init_reg.AddScalarFloat16(name, bits);
        break;
      }
      default:
        FAIL() << "RunClipSnapshot: unsupported dtype " << dtype;
    }
  };
  if (min_val) register_dtype_scalar("min", *min_val);
  if (max_val) register_dtype_scalar("max", *max_val);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
  std::unique_ptr<qnn::QnnModelWrapper> wrapper =
      MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
  ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph for snapshot test";

  auto data = MakeMockIODef("data", dtype, shape);
  auto output = MakeMockIODef("output", dtype, shape);

  std::vector<OrtNodeUnitIODef> inputs{data};
  if (min_val) {
    inputs.push_back(MakeMockIODef("min", dtype, std::vector<int64_t>{}));
  } else if (max_val) {
    // ONNX Clip min slot absent (only max provided) → empty-name optional input.
    inputs.push_back(MakeMockIODef("", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, std::nullopt));
  }
  if (max_val) {
    inputs.push_back(MakeMockIODef("max", dtype, std::vector<int64_t>{}));
  }
  auto node_unit = MakeMockNodeUnit("Clip", inputs, {output}, "clip_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true)) << "ComposeQnnGraph failed";
  AssertSnapshotJson(*wrapper, golden_basename);
}

// QDQ data with optional float min/max scalars. Pass nullopt for both min_val
// and max_val to test default min/max.
void RunClipSnapshotQDQFloatMinMax([[maybe_unused]] SnapshotBackend backend,
                                   ONNXTensorElementDataType qdq_dtype,
                                   QdqDataSpec data,
                                   std::vector<int64_t> shape,
                                   std::optional<float> min_val,
                                   std::optional<float> max_val,
                                   const char* golden_basename) {
  const IOpBuilder* builder = GetOpBuilder("Clip");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  auto [data_scale_vi, data_zp_vi] = RegisterQdqScaleZp("data", qdq_dtype, data.scale, data.zp);
  if (min_val) g_mock_init_reg.AddScalarFloat("min", *min_val);
  if (max_val) g_mock_init_reg.AddScalarFloat("max", *max_val);

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
  std::unique_ptr<qnn::QnnModelWrapper> wrapper =
      MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
  ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph for snapshot test";

  auto data_iodef = MakeMockQDQIODef("data", qdq_dtype, shape, data_scale_vi, data_zp_vi);
  auto output_iodef = MakeMockQDQIODef("output", qdq_dtype, shape, data_scale_vi, data_zp_vi);

  std::vector<OrtNodeUnitIODef> inputs{data_iodef};
  if (min_val) {
    inputs.push_back(MakeMockIODef("min", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                   std::vector<int64_t>{}));
  } else if (max_val) {
    inputs.push_back(MakeMockIODef("", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, std::nullopt));
  }
  if (max_val) {
    inputs.push_back(MakeMockIODef("max", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                   std::vector<int64_t>{}));
  }
  auto node_unit = MakeMockQDQNodeUnit("Clip", inputs, {output_iodef}, "clip_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true)) << "ComposeQnnGraph failed";
  AssertSnapshotJson(*wrapper, golden_basename);
}

// QDQ data with quantized min/max scalars, each with its own scale/zp.
void RunClipSnapshotQDQQuantMinMax([[maybe_unused]] SnapshotBackend backend,
                                   ONNXTensorElementDataType qdq_dtype,
                                   QdqDataSpec data,
                                   std::vector<int64_t> shape,
                                   std::optional<QuantScalarSpec> min_spec,
                                   std::optional<QuantScalarSpec> max_spec,
                                   const char* golden_basename) {
  const IOpBuilder* builder = GetOpBuilder("Clip");
  ASSERT_NE(builder, nullptr);

  g_mock_init_reg.clear();
  auto [data_scale_vi, data_zp_vi] = RegisterQdqScaleZp("data", qdq_dtype, data.scale, data.zp);
  auto [out_scale_vi, out_zp_vi] = RegisterQdqScaleZp("out", qdq_dtype, data.scale, data.zp);

  const OrtValueInfo* min_scale_vi = nullptr;
  const OrtValueInfo* min_zp_vi = nullptr;
  if (min_spec) {
    std::tie(min_scale_vi, min_zp_vi) = RegisterQdqScaleZp("min", qdq_dtype,
                                                           min_spec->scale, min_spec->zp);
    RegisterQuantScalar("min_quant", qdq_dtype, min_spec->raw);
  }
  const OrtValueInfo* max_scale_vi = nullptr;
  const OrtValueInfo* max_zp_vi = nullptr;
  if (max_spec) {
    std::tie(max_scale_vi, max_zp_vi) = RegisterQdqScaleZp("max", qdq_dtype,
                                                           max_spec->scale, max_spec->zp);
    RegisterQuantScalar("max_quant", qdq_dtype, max_spec->raw);
  }

  OpBuilderTestContext ctx;
  SetupMockInitRegistryStubs(ctx);
  QnnRealHtpBackendManagerContext htp;
  if (!htp.IsValid()) GTEST_SKIP() << "libQnnHtp.so not available";
  std::unique_ptr<qnn::QnnModelWrapper> wrapper =
      MakeSnapshotWrapperHtpJson(ctx, htp, {"data"}, {"output"});
  ASSERT_NE(wrapper, nullptr) << "Failed to initialize QNN graph for snapshot test";

  auto data_iodef = MakeMockQDQIODef("data", qdq_dtype, shape, data_scale_vi, data_zp_vi);
  auto output_iodef = MakeMockQDQIODef("output", qdq_dtype, shape, out_scale_vi, out_zp_vi);

  std::vector<OrtNodeUnitIODef> inputs{data_iodef};
  if (min_spec) {
    inputs.push_back(MakeMockQDQIODef("min_quant", qdq_dtype, std::vector<int64_t>{},
                                      min_scale_vi, min_zp_vi));
  } else if (max_spec) {
    inputs.push_back(MakeMockIODef("", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, std::nullopt));
  }
  if (max_spec) {
    inputs.push_back(MakeMockQDQIODef("max_quant", qdq_dtype, std::vector<int64_t>{},
                                      max_scale_vi, max_zp_vi));
  }
  auto node_unit = MakeMockQDQNodeUnit("Clip", inputs, {output_iodef}, "clip_node");

  auto status = builder->AddToModelBuilder(*wrapper, node_unit, ctx.ort_logger, false);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(wrapper->ComposeQnnGraph(/*build_json_qnn_graph=*/true)) << "ComposeQnnGraph failed";
  AssertSnapshotJson(*wrapper, golden_basename);
}

}  // namespace

// Value-parameterized suites — one per spec kind. Case name = spec.name.

class QnnSnapshot_Clip_OpBuilderTest : public ::testing::TestWithParam<ClipSpec> {};
class QnnSnapshot_Clip_OpBuilder_QDQFloatTest : public ::testing::TestWithParam<ClipQDQFloatSpec> {};
class QnnSnapshot_Clip_OpBuilder_QDQQuantTest : public ::testing::TestWithParam<ClipQDQQuantSpec> {};

TEST_P(QnnSnapshot_Clip_OpBuilderTest, Case) {
  const ClipSpec& s = GetParam();
  RunClipSnapshot(s.snapshot_backend, s.dtype, s.shape, s.min_val, s.max_val, s.name);
}

TEST_P(QnnSnapshot_Clip_OpBuilder_QDQFloatTest, Case) {
  const ClipQDQFloatSpec& s = GetParam();
  RunClipSnapshotQDQFloatMinMax(s.snapshot_backend, s.qdq_dtype, s.data, s.shape,
                                s.min_val, s.max_val, s.name);
}

TEST_P(QnnSnapshot_Clip_OpBuilder_QDQQuantTest, Case) {
  const ClipQDQQuantSpec& s = GetParam();
  RunClipSnapshotQDQQuantMinMax(s.snapshot_backend, s.qdq_dtype, s.data, s.shape,
                                s.min_spec, s.max_spec, s.name);
}

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_Clip_OpBuilderTest, ::testing::ValuesIn(kClipSpecs),
    [](const ::testing::TestParamInfo<ClipSpec>& i) { return std::string(i.param.name); });

// Op-builder snapshot exercises the explicit float min/max QDQ cases.
// Default-min/max QDQ cases are session-snapshot-only sentinels.
INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_Clip_OpBuilder_QDQFloatTest, ::testing::ValuesIn(kClipQDQFloatOpBuilderSpecs),
    [](const ::testing::TestParamInfo<ClipQDQFloatSpec>& i) { return std::string(i.param.name); });

INSTANTIATE_TEST_SUITE_P(
    , QnnSnapshot_Clip_OpBuilder_QDQQuantTest, ::testing::ValuesIn(kClipQDQQuantSpecs),
    [](const ::testing::TestParamInfo<ClipQDQQuantSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
