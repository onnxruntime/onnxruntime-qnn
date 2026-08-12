// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Op-builder-paired accuracy tests for Clip.
//
// 1:1 mapping with the snapshot tiers — every op-builder snapshot case
// (QnnUnit_Clip_Snapshot[_{QDQFloat,QDQQuant,FoldedConst}]Test) and
// session-snapshot case (QnnUnit_Clip_SessionSnapshot_QDQFloatTest) has a
// paired accuracy case here
// (QnnUnit_Clip_Accuracy[_{QDQFloat,QDQQuant,FoldedConst}]Test.Case/<name>)
// that runs the same ONNX graph end-to-end through ORT and diffs outputs
// against an ORT-CPU EP reference within tolerance. Case names come from
// spec.name so the mapping is aligned by construction (see clip_specs.h).
//
// The graph each accuracy test runs is structurally identical to the
// op-builder snapshot graph for the same case (shape / dtype / min/max
// pattern read from the shared `kClip<Case>Spec` literal in clip_specs.h).
// Per-case input data range and accuracy backend mirror the integration-tier
// counterpart (onnxruntime/test/providers/qnn/clip_test.cc) so all three
// layers verify the same production path.
//
// Lives in its own translation unit (separate from the component tier's
// clip_test.cc) because
// qnn_test_utils.h pulls full ORT internal headers (core/graph/constants.h)
// which conflict with the QNN-EP-internal kOnnxDomain copy that
// clip_test.cc receives via test_infra/qnn_unit_test_utils.h → ort_api.h.
//
// Whole file gated on QNN_EP_ACCURACY_UT (enabled together with
// ENABLE_COVERAGE on Linux x86_64 — see cmake/onnxruntime_unittests.cmake).
// When OFF, file compiles to nothing.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT

#include <cstdint>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/test_infra/specs/builder/opbuilder/clip_specs.h"

namespace onnxruntime {
namespace test {

namespace {

ProviderOptions MakeAccuracyProviderOptions(SnapshotBackend backend) {
  ProviderOptions po;
  po["backend_type"] = (backend == SnapshotBackend::CPU) ? "cpu" : "htp";
  if (backend == SnapshotBackend::HTP) {
    po["offload_graph_io_quantization"] = "0";
  }
  return po;
}

// FP32 builder — input range mirrors integration tier (-10..10, sized by
// std::accumulate of spec.shape). Used for both the FP32 accuracy cases
// and as the FP32 reference for FP16 cases (TestFp16ModelAccuracy needs
// a FP32 builder + FP16 builder pair).
GetTestModelFn BuildClipOnnxF32(const ClipSpec& spec) {
  const int64_t total = std::accumulate(spec.shape.begin(), spec.shape.end(),
                                        int64_t{1}, std::multiplies<>{});
  auto input_def = TestInputDef<float>(spec.shape, false,
                                       GetFloatDataInRange(-10.0f, 10.0f, total));
  std::vector<TestInputDef<float>> mn;
  if (spec.min_val) {
    mn.push_back(TestInputDef<float>({}, true, {*spec.min_val}));
  } else if (spec.max_val) {
    // Only max set — emit an absent slot so max ends up at input index 2.
    mn.push_back(TestInputDef<float>(/*is_optional=*/true));
  }
  if (spec.max_val) {
    mn.push_back(TestInputDef<float>({}, true, {*spec.max_val}));
  }
  return BuildOpTestCase<float, float>("Clip_node", "Clip", {input_def}, mn, /*attrs=*/{});
}

GetTestModelFn BuildClipOnnxInt32(const ClipSpec& spec) {
  // Mirror integration QnnHTPBackendTests.Clip_int32 specific values.
  auto input_def = TestInputDef<int32_t>(spec.shape, false, {1, 2, -5, 3, -10, 25});
  std::vector<TestInputDef<int32_t>> mn;
  if (spec.min_val) {
    mn.push_back(TestInputDef<int32_t>({}, true, {static_cast<int32_t>(*spec.min_val)}));
  } else if (spec.max_val) {
    mn.push_back(TestInputDef<int32_t>(/*is_optional=*/true));
  }
  if (spec.max_val) {
    mn.push_back(TestInputDef<int32_t>({}, true, {static_cast<int32_t>(*spec.max_val)}));
  }
  return BuildOpTestCase<int32_t, int32_t>("Clip_node", "Clip", {input_def}, mn, /*attrs=*/{});
}

GetTestModelFn BuildClipOnnxFp16(const ClipSpec& spec) {
  // Mirror integration QnnHTPBackendTests.Clip_FP16 specific values.
  const std::vector<float> f32_vals = {
      -10.0f, -8.0f, -3.5f, 2.2f,
      1.3f, 1.5f, 3.2f, 5.8f,
      5.8f, 9.7f, 8.5f, 8.9f};
  std::vector<Ort::Float16_t> f16_data;
  f16_data.reserve(f32_vals.size());
  for (float f : f32_vals) {
    f16_data.push_back(static_cast<Ort::Float16_t>(f));
  }
  auto input_def = TestInputDef<Ort::Float16_t>(spec.shape, false, f16_data);
  std::vector<TestInputDef<Ort::Float16_t>> mn;
  if (spec.min_val) {
    mn.push_back(TestInputDef<Ort::Float16_t>(
        {}, true, {static_cast<Ort::Float16_t>(*spec.min_val)}));
  } else if (spec.max_val) {
    mn.push_back(TestInputDef<Ort::Float16_t>(/*is_optional=*/true));
  }
  if (spec.max_val) {
    mn.push_back(TestInputDef<Ort::Float16_t>(
        {}, true, {static_cast<Ort::Float16_t>(*spec.max_val)}));
  }
  return BuildOpTestCase<Ort::Float16_t, Ort::Float16_t>(
      "Clip_node", "Clip", {input_def}, mn, /*attrs=*/{});
}

// Per-dtype dispatch. FP32/INT32 → RunQnnModelTest (integration's standard
// helper, ORT-CPU oracle). FP16 → TestFp16ModelAccuracy (integration's
// FP16 helper that runs FP32 reference on ORT-CPU + FP16 actual on QNN).
void RunClipAccuracy(const ClipSpec& spec) {
  ProviderOptions po = MakeAccuracyProviderOptions(spec.accuracy_backend);
  switch (spec.dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      // HTP since QAIRT 2.35 internally converts FP32→FP16, requiring
      // 5e-3f tolerance even for "exact" ops. CPU backend is precision-clean.
      const float tol = (spec.accuracy_backend == SnapshotBackend::HTP) ? 5e-3f : 1e-5f;
      RunQnnModelTest(BuildClipOnnxF32(spec), po,
                      /*opset=*/13,
                      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(tol)});
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      // Integer math — exact.
      RunQnnModelTest(BuildClipOnnxInt32(spec), po,
                      /*opset=*/13,
                      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.0f)});
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      // FP16 needs FP32 reference (ORT-CPU may not have native FP16 Clip).
      // Build FP32-ref spec from same shape/min/max but FLOAT dtype.
      ClipSpec f32_ref = spec;
      f32_ref.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      TestFp16ModelAccuracy(BuildClipOnnxF32(f32_ref),
                            BuildClipOnnxFp16(spec),
                            po, /*opset=*/13, ExpectedEPNodeAssignment::All);
      break;
    }
    default:
      ADD_FAILURE() << "RunClipAccuracy: unsupported dtype " << spec.dtype;
  }
}

// ---------- Group B+C: QDQ data + optional float min/max scalars ----------

// Float-data input + min/max for the f32 reference / qdq input pre-quant.
// Mirrors integration `RunQDQClipTestOnHTP` line 109-110.
template <typename QType>
void RunClipQDQFloatHelper(const ClipQDQFloatSpec& spec,
                           const TestInputDef<float>& input_def,
                           const std::vector<TestInputDef<float>>& min_max_defs) {
  ProviderOptions po = MakeAccuracyProviderOptions(spec.accuracy_backend);
  auto f32_model_builder = BuildOpTestCase<float, float>(
      "Clip_node", "Clip", {input_def}, min_max_defs, /*attrs=*/{});
  auto qdq_model_builder = BuildQDQOpTestCase<QType, float>(
      "Clip_node", "Clip", {input_def}, min_max_defs, /*attrs=*/{},
      kOnnxDomain, spec.use_contrib_qdq);
  TestQDQModelAccuracy(f32_model_builder, qdq_model_builder, po,
                       spec.opset, ExpectedEPNodeAssignment::All);
}

// Build the float-side input + min/max defs used by both f32 reference and
// qdq pre-quant. Input data range mirrors integration tier (-10..10 for
// rank-4 cases, custom for rank-5).
GetTestModelFn BuildClipQDQFloatRank5HandRolled(const ClipQDQFloatSpec& spec) {
  // Mirror integration QnnHTPBackendTests.Clip_U8_Rank5 (line 207-245):
  // input(u8) -> DQ -> Clip(min,max) -> Q -> output(u8). No boundary Q/DQ.
  // Specific raw u8 values: {0, 1, 6, 10, 20, 100, 128, 255}.
  // min=5.0f, max=100.0f as scalar float initializers (not Q/DQ-wrapped).
  const float scale = spec.data.scale;
  const uint8_t zp = static_cast<uint8_t>(spec.data.zp);
  const float min_v = spec.min_val.value();
  const float max_v = spec.max_val.value();
  return [scale, zp, min_v, max_v](ModelTestBuilder& builder) {
    builder.MakeInput<uint8_t>("X", {1, 1, 2, 2, 2},
                               {0, 1, 6, 10, 20, 100, 128, 255});
    builder.AddDequantizeLinearNode<uint8_t>("dq", "X", scale, zp, "dq_out");
    builder.MakeScalarInitializer("min", min_v);
    builder.MakeScalarInitializer("max", max_v);
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    builder.AddNode("clip", "Clip", {"dq_out", "min", "max"}, {"clip_out"}, "", attrs);
    builder.AddQuantizeLinearNode<uint8_t>("q", "clip_out", scale, zp, "Y");
    builder.MakeOutput("Y");
  };
}

void RunClipQDQFloatAccuracy(const ClipQDQFloatSpec& spec) {
  // Rank > 4 needs hand-rolled DQ → Clip → Q because QNN's Quantize/Dequantize
  // don't support rank 5+ (see integration tier comment line 209-212).
  if (spec.shape.size() > 4) {
    ProviderOptions po = MakeAccuracyProviderOptions(spec.accuracy_backend);
    RunQnnModelTest(BuildClipQDQFloatRank5HandRolled(spec), po,
                    spec.opset, EPVerificationParams{ExpectedEPNodeAssignment::All});
    return;
  }

  // Rank ≤ 4 — use integration's standard QDQ helper pair.
  const int64_t total = std::accumulate(spec.shape.begin(), spec.shape.end(),
                                        int64_t{1}, std::multiplies<>{});
  auto input_def = TestInputDef<float>(spec.shape, false,
                                       GetFloatDataInRange(-10.0f, 10.0f, total));
  std::vector<TestInputDef<float>> min_max_defs;
  if (spec.min_val) {
    min_max_defs.push_back(TestInputDef<float>({}, true, {*spec.min_val}));
  } else if (spec.max_val) {
    min_max_defs.push_back(TestInputDef<float>(/*is_optional=*/true));
  }
  if (spec.max_val) {
    min_max_defs.push_back(TestInputDef<float>({}, true, {*spec.max_val}));
  }

  switch (spec.qdq_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      RunClipQDQFloatHelper<uint8_t>(spec, input_def, min_max_defs);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      RunClipQDQFloatHelper<uint16_t>(spec, input_def, min_max_defs);
      break;
    default:
      ADD_FAILURE() << "RunClipQDQFloatAccuracy: unsupported qdq_dtype " << spec.qdq_dtype;
  }
}

// ---------- Group D: QDQ data + Q+DQ-wrapped quantized min/max scalars ----------
//
// Hand-rolled — integration tier doesn't have a helper for "Q+DQ-wrapped
// scalar min/max". Each builder mirrors the corresponding integration tier
// model_fn lambda exactly (see onnxruntime/test/providers/qnn/clip_test.cc
// lines 165-205, 208-245, 249-285, 290-326, 330-end).

template <typename QType>
GetTestModelFn BuildClipU8QDQQuantHandRolled(const ClipQDQQuantSpec& spec,
                                             std::vector<QType> input_data) {
  const auto data = spec.data;
  const auto min_spec_opt = spec.min_spec;
  const auto max_spec_opt = spec.max_spec;
  const auto shape = spec.shape;
  return [data, min_spec_opt, max_spec_opt, shape, input_data](ModelTestBuilder& builder) {
    const QType data_zp = static_cast<QType>(data.zp);
    builder.MakeInput<QType>("data_quantized", shape, input_data);
    builder.AddDequantizeLinearNode<QType>("data_dq", "data_quantized", data.scale, data_zp,
                                           "data_dq_out");

    std::string min_dq_out;
    std::string max_dq_out;
    if (min_spec_opt) {
      const QType min_zp_q = static_cast<QType>(min_spec_opt->zp);
      const QType min_raw = static_cast<QType>(min_spec_opt->raw);
      builder.template MakeInitializer<QType>("min_quantized", {}, {min_raw});
      builder.AddDequantizeLinearNode<QType>("min_dq", "min_quantized", min_spec_opt->scale,
                                             min_zp_q, "min_dq_out");
      min_dq_out = "min_dq_out";
    }
    if (max_spec_opt) {
      const QType max_zp_q = static_cast<QType>(max_spec_opt->zp);
      const QType max_raw = static_cast<QType>(max_spec_opt->raw);
      builder.template MakeInitializer<QType>("max_quantized", {}, {max_raw});
      builder.AddDequantizeLinearNode<QType>("max_dq", "max_quantized", max_spec_opt->scale,
                                             max_zp_q, "max_dq_out");
      max_dq_out = "max_dq_out";
    }

    std::vector<std::string> clip_inputs{"data_dq_out"};
    if (min_dq_out.empty() && !max_dq_out.empty()) {
      clip_inputs.push_back("");  // min absent
    } else if (!min_dq_out.empty()) {
      clip_inputs.push_back(min_dq_out);
    }
    if (!max_dq_out.empty()) {
      clip_inputs.push_back(max_dq_out);
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    builder.AddNode("clip", "Clip", clip_inputs, {"clip_output"}, "", attrs);
    builder.AddQuantizeLinearNode<QType>("q", "clip_output", data.scale, data_zp, "Y");
    builder.MakeOutput("Y");
  };
}

GetTestModelFn BuildClipQDQQuantOnnxFromSpec(const ClipQDQQuantSpec& spec) {
  // Each Group D case has a specific input data pattern mirroring its
  // integration-tier counterpart's hand-rolled model_fn. We dispatch on
  // (qdq_dtype, shape, scale/zp) — practical to encode per-case.
  switch (spec.qdq_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      if (spec.shape == std::vector<int64_t>{1, 3, 4, 4}) {
        // Clip_U8_IndependentQDQ_MinMaxQDQ: input data = 0..47 (48 elems)
        std::vector<uint8_t> input_data(48);
        for (size_t i = 0; i < input_data.size(); ++i) {
          input_data[i] = static_cast<uint8_t>(i);
        }
        return BuildClipU8QDQQuantHandRolled<uint8_t>(spec, std::move(input_data));
      }
      if (spec.shape == std::vector<int64_t>{200}) {
        // Clip_U8_QuantizedMin / Clip_U8_QuantizedMinMax: input data = 28..227 (200 elems)
        std::vector<uint8_t> input_data(200);
        for (size_t i = 0; i < 200; ++i) {
          input_data[i] = static_cast<uint8_t>(28 + i);
        }
        return BuildClipU8QDQQuantHandRolled<uint8_t>(spec, std::move(input_data));
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      if (spec.shape == std::vector<int64_t>{200}) {
        // Clip_U16_QuantizedMax: input data = 32668..32867 (200 elems centered around zp)
        std::vector<uint16_t> input_data(200);
        for (size_t i = 0; i < 200; ++i) {
          input_data[i] = static_cast<uint16_t>(32768 - 100 + i);
        }
        return BuildClipU8QDQQuantHandRolled<uint16_t>(spec, std::move(input_data));
      }
      break;
    }
    default:
      break;
  }
  ADD_FAILURE() << "BuildClipQDQQuantOnnxFromSpec: no input data pattern for"
                << " qdq_dtype=" << spec.qdq_dtype << " shape.size=" << spec.shape.size();
  return GetTestModelFn{};
}

void RunClipQDQQuantAccuracy(const ClipQDQQuantSpec& spec) {
  ProviderOptions po = MakeAccuracyProviderOptions(spec.accuracy_backend);
  RunQnnModelTest(BuildClipQDQQuantOnnxFromSpec(spec), po,
                  spec.opset, EPVerificationParams{ExpectedEPNodeAssignment::All});
}

// ---------- Group E: bare-float data + Q+DQ-const-wrapped min/max ----------
//
// Mirrors integration `Clip_U*_FloatData_QDQConstMinMax` (clip_test.cc:375, 407):
// bare-float MakeInput → Clip → bare-float MakeOutput; min/max are u*_scalar
// initializers wrapped by DequantizeLinear. QDQ selector rejects (no output Q),
// so both DQ nodes remain standalone and are folded by QNN EP's
// qdq_constant_folding pass. Clip builder consumes them via the folded-constant
// fallback branch (Path A, clip_op_builder.cc:45-52).

GetTestModelFn BuildClipFoldedConstOnnx(const ClipFoldedConstSpec& spec) {
  const int64_t total = std::accumulate(spec.shape.begin(), spec.shape.end(),
                                        int64_t{1}, std::multiplies<>{});
  // Input range mirrors integration tier: U8 test uses -10..10, U16 test -20..20.
  const bool is_u8 = spec.qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  const float lo = is_u8 ? -10.0f : -20.0f;
  const std::vector<float> input_data = GetFloatDataInRange(lo, -lo, static_cast<size_t>(total));

  const auto min_spec = spec.min_spec;
  const auto max_spec = spec.max_spec;
  const auto shape = spec.shape;
  const auto qdq_dtype = spec.qdq_dtype;

  return [min_spec, max_spec, shape, qdq_dtype, input_data](ModelTestBuilder& builder) {
    builder.MakeInput<float>("X", shape, input_data);
    if (qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      builder.MakeInitializer<uint8_t>("min_q", {}, {static_cast<uint8_t>(min_spec.raw)});
      builder.AddDequantizeLinearNode<uint8_t>("min_dq", "min_q", min_spec.scale,
                                               static_cast<uint8_t>(min_spec.zp), "min_dq_out");
      builder.MakeInitializer<uint8_t>("max_q", {}, {static_cast<uint8_t>(max_spec.raw)});
      builder.AddDequantizeLinearNode<uint8_t>("max_dq", "max_q", max_spec.scale,
                                               static_cast<uint8_t>(max_spec.zp), "max_dq_out");
    } else {
      builder.MakeInitializer<uint16_t>("min_q", {}, {static_cast<uint16_t>(min_spec.raw)});
      builder.AddDequantizeLinearNode<uint16_t>("min_dq", "min_q", min_spec.scale,
                                                static_cast<uint16_t>(min_spec.zp), "min_dq_out");
      builder.MakeInitializer<uint16_t>("max_q", {}, {static_cast<uint16_t>(max_spec.raw)});
      builder.AddDequantizeLinearNode<uint16_t>("max_dq", "max_q", max_spec.scale,
                                                static_cast<uint16_t>(max_spec.zp), "max_dq_out");
    }
    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    builder.AddNode("clip", "Clip", {"X", "min_dq_out", "max_dq_out"}, {"Y"}, "", attrs);
    builder.MakeOutput("Y");
  };
}

void RunClipFoldedConstAccuracy(const ClipFoldedConstSpec& spec) {
  ProviderOptions po = MakeAccuracyProviderOptions(spec.accuracy_backend);
  // Integration uses ElementwiseAbsoluteVerifier{5e-3f}: HTP internally converts
  // FP32 → FP16 (QAIRT >= 2.35) so exact equality doesn't hold even for Clip.
  RunQnnModelTest(BuildClipFoldedConstOnnx(spec), po, spec.opset,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(5e-3f)});
}

}  // namespace

// Three value-parameterized sub-suites, one per spec struct type. Each case's
// name comes from spec.name (single source of truth in clip_specs.h), so
// accuracy case names align BY CONSTRUCTION with the paired snapshot /
// session-snapshot cases — a rename in the spec propagates to every tier.
//
// QDQFloat is instantiated over kClipQDQFloatAccuracySpecs = the union of the
// op-builder-snapshot set (Group C) and the session-snapshot set (Group B),
// so accuracy = snapshot ∪ session holds by construction (see clip_specs.h).
class QnnUnit_Clip_AccuracyTest
    : public ::testing::TestWithParam<ClipSpec> {};
class QnnUnit_Clip_Accuracy_QDQFloatTest
    : public ::testing::TestWithParam<ClipQDQFloatSpec> {};
class QnnUnit_Clip_Accuracy_QDQQuantTest
    : public ::testing::TestWithParam<ClipQDQQuantSpec> {};
class QnnUnit_Clip_Accuracy_FoldedConstTest
    : public ::testing::TestWithParam<ClipFoldedConstSpec> {};

TEST_P(QnnUnit_Clip_AccuracyTest, Case) {
  RunClipAccuracy(GetParam());
}

TEST_P(QnnUnit_Clip_Accuracy_QDQFloatTest, Case) {
  RunClipQDQFloatAccuracy(GetParam());
}

TEST_P(QnnUnit_Clip_Accuracy_QDQQuantTest, Case) {
  RunClipQDQQuantAccuracy(GetParam());
}

TEST_P(QnnUnit_Clip_Accuracy_FoldedConstTest, Case) {
  RunClipFoldedConstAccuracy(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    , QnnUnit_Clip_AccuracyTest,
    ::testing::ValuesIn(kClipSpecs),
    [](const ::testing::TestParamInfo<ClipSpec>& i) { return std::string(i.param.name); });

INSTANTIATE_TEST_SUITE_P(
    , QnnUnit_Clip_Accuracy_QDQFloatTest,
    ::testing::ValuesIn(kClipQDQFloatAccuracySpecs),
    [](const ::testing::TestParamInfo<ClipQDQFloatSpec>& i) { return std::string(i.param.name); });

INSTANTIATE_TEST_SUITE_P(
    , QnnUnit_Clip_Accuracy_QDQQuantTest,
    ::testing::ValuesIn(kClipQDQQuantSpecs),
    [](const ::testing::TestParamInfo<ClipQDQQuantSpec>& i) { return std::string(i.param.name); });

INSTANTIATE_TEST_SUITE_P(
    , QnnUnit_Clip_Accuracy_FoldedConstTest,
    ::testing::ValuesIn(kClipFoldedConstSpecs),
    [](const ::testing::TestParamInfo<ClipFoldedConstSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS && QNN_EP_ACCURACY_UT
