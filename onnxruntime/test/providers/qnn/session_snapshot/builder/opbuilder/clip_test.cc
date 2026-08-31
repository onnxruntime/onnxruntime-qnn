// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Session-level snapshot tests for Clip.
//
// Lives in its own translation unit (separate from the component tier's
// clip_test.cc) because
// session_snapshot.h pulls full ORT internal headers (qnn_test_utils.h ->
// core/graph/constants.h) which double-define kOnnxDomain etc. against the
// QNN-EP-internal copy in ort_api.h that the unit-layer aggregator brings in.
//
// What this catches: regressions in ORT pre-partition transformers (AOT
// inline, EnsureUniqueDQ, L1 — DoubleQDQPairsRemover etc.) and partition-time
// transforms (transform_layout_fn) that the op-builder snapshot
// (AssertSnapshotJson, qnn_unit_test_utils.h) cannot see — those drive
// op-builders directly without a session.
//
// Scope on Clip (layout-agnostic): the two default-min/max QDQ sentinel cases
// (kClipQDQFloatSessionSpecs in clip_specs.h) locking the observation that L1
// transformers do NOT collapse the QDQ-around-Clip pattern for QNN EP.
// ClipQuantFusion (L2, gated by IsSupportedProvider({CPU}) + blocked by
// FinalizeFuseSubGraph::RemoveNode) and DoubleQDQPairsRemover (L1, blocked
// because Clip sits between DQ1 and Q2) both do not fire — the golden captures
// the resulting Quantize -> ReluMinMax -> Dequantize graph. See
// migrate_clip_test.md §"Clip QDQ Fusion Never Fires for QNN EP" for the full
// pipeline analysis.
//
// Value-parameterized over kClipQDQFloatSessionSpecs; case name = spec.name,
// shared with the paired accuracy suite so names cannot desync. Accuracy
// covers these same two cases via kClipQDQFloatAccuracySpecs.
//
// L2+ regressions are intentionally out of scope (JSON dumps inside
// Partition::Compile before L2 runs). For layout-sensitive ops like Conv,
// this layer becomes the primary snapshot — see evolve_ut_infra_plan.md
// §Op-builder vs Session Snapshot.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/test_infra/specs/builder/opbuilder/clip_specs.h"
#include "test/providers/qnn/session_snapshot/session_snapshot.h"

namespace onnxruntime {
namespace test {

namespace {

// Default-min/max QDQ Clip built generically from a ClipQDQFloatSpec. Output
// qparams equal input qparams, matching the integration-layer setup. Even with
// same scale/zp on Q1 and Q2, DoubleQDQPairsRemover does not fire because Clip
// sits between DQ1 and Q2 and blocks the Q→DQ→Q→DQ pattern match; the golden
// therefore captures the unfused Quantize → ReluMinMax → Dequantize graph.
// `use_contrib_qdq` (spec) selects com.microsoft Q/DQ ops for the U16 case.
template <typename QType>
GetTestModelFn BuildClipDefaultMinMaxSessionFn(const ClipQDQFloatSpec& spec) {
  const int64_t total = std::accumulate(spec.shape.begin(), spec.shape.end(),
                                        int64_t{1}, std::multiplies<>{});
  auto input_def = TestInputDef<float>(spec.shape, false,
                                       GetFloatDataInRange(-10.0f, 10.0f, total));
  auto qdq_fn = BuildQDQOpTestCase<QType, float>(
      "Clip_node", "Clip",
      {input_def},
      /*non_quant_input_defs=*/{},
      /*attrs=*/{},
      kOnnxDomain,
      spec.use_contrib_qdq);
  const QuantParams<QType> qparams = GetTestInputQuantParams<QType>(input_def);
  return [qdq_fn = std::move(qdq_fn), qparams](ModelTestBuilder& builder) {
    std::vector<QuantParams<QType>> output_qparams = {qparams};
    qdq_fn(builder, output_qparams);
  };
}

}  // namespace

// Locks the observation that no L1 fusion (and no L2 ClipQuantFusion, since it
// is gated for CPU EP only and the original Clip is removed from the top-level
// graph by FinalizeFuseSubGraph before L2 runs) collapses the QDQ-around-Clip
// pattern for QNN EP. Golden captures Quantize → ReluMinMax → Dequantize. If a
// future ORT release loosens this behavior — e.g. DoubleQDQPairsRemover learns
// to span Clip, or the contrib-domain U16 path diverges — these goldens will
// diff and force review.
class QnnUnit_Clip_SessionSnapshot_QDQFloatTest
    : public ::testing::TestWithParam<ClipQDQFloatSpec> {};

TEST_P(QnnUnit_Clip_SessionSnapshot_QDQFloatTest, Case) {
  const ClipQDQFloatSpec& s = GetParam();

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestModelFn fn = (s.qdq_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
                          ? BuildClipDefaultMinMaxSessionFn<uint8_t>(s)
                          : BuildClipDefaultMinMaxSessionFn<uint16_t>(s);

  AssertSessionSnapshotJson(fn, provider_options, s.opset, s.name);
}

INSTANTIATE_TEST_SUITE_P(
    , QnnUnit_Clip_SessionSnapshot_QDQFloatTest,
    ::testing::ValuesIn(kClipQDQFloatSessionSpecs),
    [](const ::testing::TestParamInfo<ClipQDQFloatSpec>& i) { return std::string(i.param.name); });

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
