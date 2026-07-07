// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Mock NodeUnit helpers for QNN EP function-level / component-level unit tests.
//
// Lets op-builder tests construct lightweight OrtNodeUnit objects without a
// real OrtNode* or a live OrtApi:
//
//   auto input  = MakeMockIODef("data",   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4});
//   auto output = MakeMockIODef("result", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4});
//   auto node_unit = MakeMockNodeUnit("Clip", {input}, {output});
//   // node_unit.OpType() == "Clip", node_unit.Inputs()[0].name == "data", etc.
//
// The helpers produce a SingleNode-type OrtNodeUnit. Domain defaults to ""
// (default ONNX domain). Use OrtNodeUnit::MockSpec directly when you need
// non-default domain / since_version / index.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <optional>
#include <string>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

// Build a simple OrtNodeUnitIODef for use in mock NodeUnit construction.
// Quant params are absent (std::nullopt) — use MakeMockQDQIODef when testing
// quantized inputs (e.g. QDQ Clip min/max).
inline OrtNodeUnitIODef MakeMockIODef(
    std::string name,
    ONNXTensorElementDataType element_type,
    std::optional<std::vector<int64_t>> shape = std::nullopt) {
  OrtNodeUnitIODef def;
  def.name = std::move(name);
  def.type = element_type;
  def.shape = std::move(shape);
  def.quant_param = std::nullopt;
  return def;
}

// Build a mock OrtNodeUnitIODef for a quantized tensor (e.g. QDQ Clip min/max input).
// scale and zero_point are OrtValueInfo pointers — pass nullptr for zero_point if absent.
// axis is std::nullopt for per-tensor quantization.
inline OrtNodeUnitIODef MakeMockQDQIODef(
    std::string name,
    ONNXTensorElementDataType element_type,
    std::optional<std::vector<int64_t>> shape,
    const OrtValueInfo* scale,
    const OrtValueInfo* zero_point = nullptr,
    std::optional<int64_t> axis = std::nullopt) {
  OrtNodeUnitIODef def;
  def.name = std::move(name);
  def.type = element_type;
  def.shape = std::move(shape);
  def.quant_param = OrtNodeUnitIODef::QuantParam{scale, zero_point, axis};
  return def;
}

// Build an OrtNodeUnit backed entirely by mock data.
// Domain defaults to "" (default ONNX domain); since_version defaults to 1; index to 0.
// For non-default values, construct OrtNodeUnit::MockSpec directly.
inline OrtNodeUnit MakeMockNodeUnit(
    std::string op_type,
    std::vector<OrtNodeUnitIODef> inputs,
    std::vector<OrtNodeUnitIODef> outputs,
    std::string name = "",
    std::string domain = "",
    int since_version = 1,
    size_t index = 0) {
  OrtNodeUnit::MockSpec spec;
  spec.domain = std::move(domain);
  spec.op_type = std::move(op_type);
  spec.name = std::move(name);
  spec.since_version = since_version;
  spec.index = index;
  spec.inputs = std::move(inputs);
  spec.outputs = std::move(outputs);
  return OrtNodeUnit(std::move(spec));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
