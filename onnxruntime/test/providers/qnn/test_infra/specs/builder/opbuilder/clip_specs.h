// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Shared spec literals for Clip op-builder tests.
//
// One literal per case is read by all tiers that exercise that case:
//   * op-builder snapshot tier (runs without an ORT session)
//   * session-snapshot tier (drives the full ORT session so L1/partition
//     transforms are visible)
//   * paired session-routed accuracy tier (runs the same case end-to-end
//     through an ORT session and diffs vs ORT-CPU EP)
//
// The `name` field is the single source of truth for the gtest case name in
// EVERY tier. Each tier instantiates a value-parameterized suite over the
// grouped spec lists below and derives its case name from `spec.name`, so
// snapshot / session-snapshot / accuracy case names are aligned BY
// CONSTRUCTION — a rename in one place is impossible to desync, and the
// gate's snapshot-case -> accuracy-case mapping (swap tier prefix, keep
// `Clip<Kind>Test.<name>`) is mechanical. `name` doubles as the golden
// file basename (`<name>.json`).
//
// This header intentionally pulls only the ONNX C API enum
// (ONNXTensorElementDataType) so it can be included from both the
// QNN-EP-internal world (component/ + snapshot/ clip_test.cc →
// test_infra/qnn_unit_test_utils.h) and the full-ORT-internals world
// (accuracy/ + session_snapshot/ clip_test.cc → qnn_test_utils.h) without the
// kOnnxDomain-style double-define that already split those into their own
// translation units.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "onnxruntime_c_api.h"

namespace onnxruntime {
namespace test {

// Backend the snapshot side initializes. Accuracy side mirrors the
// integration-tier counterpart's backend (see `accuracy_backend` field
// below), which is NOT always equal to `snapshot_backend`:
//   * `snapshot_backend` is rule-based — CPU keeps the wrapper IR clean of
//     BF16-conversion ops on FP32 inputs, HTP is required for dtypes
//     CPU rejects (FP16, U16).
//   * `accuracy_backend` matches what `onnxruntime/test/providers/qnn/clip_test.cc`
//     uses for the corresponding `QnnCPUBackendTests.<Case>` /
//     `QnnHTPBackendTests.<Case>` test, so all three layers exercise the
//     same production path.
enum class SnapshotBackend { CPU,
                             HTP };

// ---------- Group A: plain dtype data, optional float min/max ----------
//
// min/max stored as float regardless of `dtype` — the snapshot helper and
// accuracy builder cast to the actual dtype when registering the scalar
// initializer (INT32 truncates, FP16 rounds). This avoids templating the
// spec on dtype.
//
// Per-case shape, min/max, and `accuracy_backend` mirror the integration-tier
// counterpart (see onnxruntime/test/providers/qnn/clip_test.cc) so all three
// layers — op-builder snapshot, paired session-accuracy, original integration
// — exercise the SAME ONNX graph structure on the SAME QNN backend.
struct ClipPlainSpec {
  const char* name;  // gtest case name + golden basename (single source of truth)
  SnapshotBackend snapshot_backend;
  SnapshotBackend accuracy_backend;
  ONNXTensorElementDataType dtype;
  std::vector<int64_t> shape;
  std::optional<float> min_val;
  std::optional<float> max_val;
};

inline const ClipPlainSpec kClipFp32Spec = {
    "Clip_f32",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,  // integration: QnnHTPBackendTests.Clip_f32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    {1, 1, 3, 4},
    -5.0f,
    5.0f};

inline const ClipPlainSpec kClip4DFp32DefaultMinMaxSpec = {
    "Clip_4D_f32_DefaultMinMax",
    SnapshotBackend::CPU,
    SnapshotBackend::CPU,  // integration: QnnCPUBackendTests.Clip_4D_f32_DefaultMinMax
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    {1, 3, 4, 4},
    std::nullopt,
    std::nullopt};

inline const ClipPlainSpec kClip5DFp32Spec = {
    "Clip_5D_f32",
    SnapshotBackend::CPU,
    SnapshotBackend::CPU,  // integration: QnnCPUBackendTests.Clip_5D_f32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    {1, 1, 3, 4, 4},
    -5.0f,
    5.0f};

inline const ClipPlainSpec kClipInt32Spec = {
    "Clip_int32",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,  // integration: QnnHTPBackendTests.Clip_int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    {1, 1, 3, 2},
    -5.0f,
    5.0f};

inline const ClipPlainSpec kClipFp16Spec = {
    "Clip_FP16",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,  // integration: QnnHTPBackendTests.Clip_FP16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    {1, 3, 2, 2},
    1.2f,
    std::nullopt};

// ---------- Group B+C: QDQ data + optional float min/max scalars ----------
//
// Mirrors integration `RunQDQClipTestOnHTP<QType>(input_def, min_max_defs, ...)`
// pattern: input data + Q/DQ around input + optional float min/max scalar
// initializers passed direct to Clip op (NOT Q/DQ-wrapped — that's Group D).
//
// Per-case `qdq_dtype` (U8 vs U16) and `use_contrib_qdq` mirror integration
// (U16 needs com.microsoft Q/DQ ops because native U16 QDQ requires opset 21+).
//
// `Clip_U8_Rank5` is hand-rolled in integration tier (line 207-245 of
// `onnxruntime/test/providers/qnn/clip_test.cc`) because QNN's Quantize/Dequantize
// don't support rank 5 — the accuracy builder branches on `shape.size() > 4`.
struct QdqDataSpec {
  float scale;
  uint32_t zp;  // narrowed to uint8/uint16 by builder per qdq_dtype
};

struct ClipQDQFloatSpec {
  const char* name;  // gtest case name + golden basename (single source of truth)
  SnapshotBackend snapshot_backend;
  SnapshotBackend accuracy_backend;
  ONNXTensorElementDataType qdq_dtype;
  QdqDataSpec data;
  std::vector<int64_t> shape;
  std::optional<float> min_val;
  std::optional<float> max_val;
  int opset;
  bool use_contrib_qdq;
};

// Group B — default (no) min/max. These two cases live in the session-snapshot
// tier, NOT the op-builder snapshot tier: the QDQ-around
// -Clip pattern only becomes observable after the ORT session runs L1/partition
// transforms. Accuracy still covers them (kClipQDQFloatAccuracySpecs below).
inline const ClipQDQFloatSpec kClipU8DefaultMinMaxRank4Spec = {
    "Clip_U8_DefaultMinMax_Rank4",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,  // integration: QnnHTPBackendTests.Clip_U8_DefaultMinMax_Rank4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    {0.1f, 0},
    {1, 3, 4, 4},
    std::nullopt,
    std::nullopt,
    /*opset=*/13,
    /*use_contrib_qdq=*/false};

inline const ClipQDQFloatSpec kClipU16DefaultMinMaxRank4Spec = {
    "Clip_U16_DefaultMinMax_Rank4",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,  // integration: QnnHTPBackendTests.Clip_U16_DefaultMinMax_Rank4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    {0.1f, 0},
    {1, 3, 4, 4},
    std::nullopt,
    std::nullopt,
    /*opset=*/13,
    /*use_contrib_qdq=*/true};

// Group C — explicit float min/max
inline const ClipQDQFloatSpec kClipU8Rank4Spec = {
    "Clip_U8_Rank4",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,  // integration: QnnHTPBackendTests.Clip_U8_Rank4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    {0.1f, 0},
    {1, 3, 4, 4},
    -5.0f,
    5.0f,
    /*opset=*/13,
    /*use_contrib_qdq=*/false};

inline const ClipQDQFloatSpec kClipU16Rank4Spec = {
    "Clip_U16_Rank4",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,  // integration: QnnHTPBackendTests.Clip_U16_Rank4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    {0.1f, 0},
    {1, 3, 4, 4},
    -5.0f,
    5.0f,
    /*opset=*/13,
    /*use_contrib_qdq=*/true};

// Rank 5 — accuracy builder hand-rolls DQ → Clip → Q (no boundary Q/DQ).
// Integration: QnnHTPBackendTests.Clip_U8_Rank5
inline const ClipQDQFloatSpec kClipU8Rank5Spec = {
    "Clip_U8_Rank5",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    {1.0f, 0},
    {1, 1, 2, 2, 2},
    5.0f,
    100.0f,
    /*opset=*/13,
    /*use_contrib_qdq=*/false};

// ---------- Group D: QDQ data + Q+DQ-wrapped quantized min/max scalars ----------
//
// Each min/max input is a quantized scalar with its own (scale, zp), wrapped
// by a DequantizeLinear node before reaching Clip. Integration tier hand-rolls
// these (no helper exists in qnn_test_utils.h for "Q+DQ scalar input"); the
// accuracy builder mirrors that pattern.
struct QuantScalarSpec {
  float scale;
  uint32_t zp;
  uint32_t raw;  // pre-quantized representation (interpreted per qdq_dtype)
};

struct ClipQDQQuantSpec {
  const char* name;  // gtest case name + golden basename (single source of truth)
  SnapshotBackend snapshot_backend;
  SnapshotBackend accuracy_backend;
  ONNXTensorElementDataType qdq_dtype;
  QdqDataSpec data;
  std::vector<int64_t> shape;
  std::optional<QuantScalarSpec> min_spec;
  std::optional<QuantScalarSpec> max_spec;
  int opset;
};

// Integration: QnnHTPBackendTests.Clip_U8_IndependentQDQ_MinMaxQDQ
// (rank-4, scale=0.1/zp=128 shared across data + min + max, raw values 118/138)
inline const ClipQDQQuantSpec kClipU8IndependentQDQMinMaxQDQSpec = {
    "Clip_U8_IndependentQDQ_MinMaxQDQ",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    {0.1f, 128},
    {1, 3, 4, 4},
    QuantScalarSpec{0.1f, 128, 118},
    QuantScalarSpec{0.1f, 128, 138},
    /*opset=*/13};

// Integration: QnnHTPBackendTests.Clip_U8_QuantizedMin (only min, opset 11)
inline const ClipQDQQuantSpec kClipU8QuantizedMinSpec = {
    "Clip_U8_QuantizedMin",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    {0.001f, 128},
    {200},
    QuantScalarSpec{0.001f, 128, 128},
    std::nullopt,
    /*opset=*/11};

// Integration: QnnHTPBackendTests.Clip_U16_QuantizedMax (only max, opset 21)
inline const ClipQDQQuantSpec kClipU16QuantizedMaxSpec = {
    "Clip_U16_QuantizedMax",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    {0.001f, 32768},
    {200},
    std::nullopt,
    QuantScalarSpec{0.001f, 32768, 32768},
    /*opset=*/21};

// Integration: QnnHTPBackendTests.Clip_U8_QuantizedMinMax
// (both min/max with own qparams, opset 13)
inline const ClipQDQQuantSpec kClipU8QuantizedMinMaxSpec = {
    "Clip_U8_QuantizedMinMax",
    SnapshotBackend::HTP,
    SnapshotBackend::HTP,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    {0.001f, 128},
    {200},
    QuantScalarSpec{0.001f, 128, 78},
    QuantScalarSpec{0.001f, 128, 178},
    /*opset=*/13};

// ---------------------------------------------------------------------------
// Grouped spec lists — the parameter sources each tier's TEST_P instantiates
// over. Splitting by (kind, tier) is what makes accuracy = snapshot ∪ session
// hold by construction:
//   * op-builder snapshot: kClipPlainSpecs + kClipQDQFloatOpBuilderSpecs
//                          + kClipQDQQuantSpecs
//   * session snapshot   : kClipQDQFloatSessionSpecs (Group B only)
//   * accuracy           : kClipPlainSpecs + kClipQDQFloatAccuracySpecs
//                          + kClipQDQQuantSpecs
// Adding a case = add one literal to the right list; every consuming tier
// picks it up with a matching name automatically.
// ---------------------------------------------------------------------------

inline const std::vector<ClipPlainSpec> kClipPlainSpecs = {
    kClip4DFp32DefaultMinMaxSpec,
    kClip5DFp32Spec,
    kClipFp32Spec,
    kClipInt32Spec,
    kClipFp16Spec};

// Group C float-min/max cases — exercised by the op-builder snapshot tier.
inline const std::vector<ClipQDQFloatSpec> kClipQDQFloatOpBuilderSpecs = {
    kClipU8Rank4Spec,
    kClipU16Rank4Spec,
    kClipU8Rank5Spec};

// Group B default-min/max cases — exercised by the session-snapshot tier.
inline const std::vector<ClipQDQFloatSpec> kClipQDQFloatSessionSpecs = {
    kClipU8DefaultMinMaxRank4Spec,
    kClipU16DefaultMinMaxRank4Spec};

// Accuracy covers every QDQFloat case = op-builder-snapshot ∪ session-snapshot,
// built by concatenation so the union holds by construction (add to either
// source list and accuracy inherits it).
inline const std::vector<ClipQDQFloatSpec> kClipQDQFloatAccuracySpecs = [] {
  std::vector<ClipQDQFloatSpec> all = kClipQDQFloatOpBuilderSpecs;
  all.insert(all.end(), kClipQDQFloatSessionSpecs.begin(), kClipQDQFloatSessionSpecs.end());
  return all;
}();

inline const std::vector<ClipQDQQuantSpec> kClipQDQQuantSpecs = {
    kClipU8IndependentQDQMinMaxQDQSpec,
    kClipU8QuantizedMinSpec,
    kClipU16QuantizedMaxSpec,
    kClipU8QuantizedMinMaxSpec};

}  // namespace test
}  // namespace onnxruntime
