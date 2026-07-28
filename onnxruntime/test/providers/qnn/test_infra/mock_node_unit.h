// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Mock NodeUnit helpers for QNN EP function-level / component-level unit tests.
//
// Lets op-builder tests construct real OrtNodeUnit objects — via the production
// OrtNodeUnit constructors — without a live ORT graph, by backing them with a
// fake OrtGraph / OrtNode / OrtValueInfo (see qnn_fake_ort_graph.h). This keeps
// the EP source untouched: no test-only constructor, no MockSpec.
//
// Two flavors:
//   MakeMockNodeUnit    — SingleNode NodeUnit (no quantization). Drives the
//                         production OrtNodeUnit(node, ort_api) ctor.
//   MakeMockQDQNodeUnit — QDQGroup NodeUnit. Synthesizes DQ nodes for quantized
//                         inputs and a Q node for a quantized output, then drives
//                         the production OrtNodeUnit(graph, node_group, ort_api)
//                         ctor. This is the path that preserves per-IO
//                         quant_param (scale/zp), which a SingleNode ctor would
//                         drop.
//
//   auto input  = MakeMockIODef("data",   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4});
//   auto output = MakeMockIODef("result", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4});
//   auto node_unit = MakeMockNodeUnit("Clip", {input}, {output});
//   builder->AddToModelBuilder(*wrapper, node_unit, logger, false);
//
// Lifetime / global-API contract
// ------------------------------
// OrtNodeUnit accessors (OpType/Name/Domain/SinceVersion/Index) dereference the
// target OrtNode* through the GLOBAL Ort::GetApi() (via Ort::ConstNode), not
// through the ort_api passed to the constructor. AddToModelBuilder calls those
// accessors, so the fake graph must remain decodable via the global API for the
// whole call. The returned holder therefore owns an OrtGlobalApiOverride that
// routes the global API to the fake-graph stubs for the holder's entire
// lifetime. Keep the holder alive across AddToModelBuilder (the natural
// `auto node_unit = MakeMockNodeUnit(...)` local does this). Do not hold two
// live holders in the same thread simultaneously (nested global overrides).
//
// An empty-name IODef (MakeMockIODef("", UNDEFINED, nullopt)) models an absent
// optional input (e.g. Clip with only `max` provided) — it becomes a nullptr
// input slot on the fake node.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"

#include "test/providers/qnn/test_infra/qnn_fake_ort_graph.h"

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

namespace detail {

// Owning backing store for a fake-graph OrtNodeUnit. Kept on the heap (behind a
// unique_ptr in MockNodeUnit) and never moved, so the FakeNode/FakeValueInfo
// pointers threaded into each other and into the OrtNodeUnit stay valid.
struct MockNodeUnitImpl {
  // deque: push_back never invalidates existing element addresses.
  std::deque<FakeValueInfo> vis;
  std::deque<FakeNode> nodes;
  OrtApi ctor_api{};
  std::unique_ptr<OrtNodeUnit> unit;
  // Constructed last, destroyed first: routes global Ort::GetApi() to ctor_api
  // so OrtNodeUnit accessors decode the fake node.
  std::unique_ptr<OrtGlobalApiOverride> global_guard;
};

// Create a FakeValueInfo in the holder and return a stable pointer to it.
inline FakeValueInfo* AddFakeVi(MockNodeUnitImpl& impl, std::string name,
                                ONNXTensorElementDataType type,
                                std::optional<std::vector<int64_t>> shape) {
  FakeValueInfo vi;
  vi.name = std::move(name);
  vi.elem_type = type;
  vi.shape = shape.value_or(std::vector<int64_t>{});
  impl.vis.push_back(std::move(vi));
  return &impl.vis.back();
}

// Reinterpret a registry OrtValueInfo* sentinel (scale/zp) as a FakeValueInfo*
// so it can live in a FakeNode's input list. The fake-graph stubs never decode
// these — GetQDQIODefs only stores them as scale/zp pointers in the quant_param
// (round-tripping back to the original sentinel via AsValueInfo()).
inline FakeValueInfo* AsFakeVi(const OrtValueInfo* sentinel) {
  return reinterpret_cast<FakeValueInfo*>(const_cast<OrtValueInfo*>(sentinel));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// MockNodeUnit
//
// Owning holder for a fake-graph-backed OrtNodeUnit. Movable (moves the heap
// Impl pointer). Implicitly convertible to const OrtNodeUnit& so it can be
// passed straight to IOpBuilder::AddToModelBuilder.
// ---------------------------------------------------------------------------
class MockNodeUnit {
 public:
  explicit MockNodeUnit(std::unique_ptr<detail::MockNodeUnitImpl> impl)
      : impl_(std::move(impl)) {}

  const OrtNodeUnit& Get() const { return *impl_->unit; }
  operator const OrtNodeUnit&() const { return *impl_->unit; }  // NOLINT(runtime/explicit)

 private:
  std::unique_ptr<detail::MockNodeUnitImpl> impl_;
};

// Build a SingleNode OrtNodeUnit backed by a fake graph. Ignores any quant_param
// on the IODefs (a SingleNode ctor re-derives IODefs from the fake node with no
// quant params) — use MakeMockQDQNodeUnit for quantized IO.
//
// Domain defaults to "" (default ONNX domain); since_version to 1; index to 0.
// index feeds Node_GetId → UniqueNameGenerator / QnnParamWrapper → QNN tensor
// names / snapshot goldens, so keep it deterministic across runs.
inline MockNodeUnit MakeMockNodeUnit(
    std::string op_type,
    std::vector<OrtNodeUnitIODef> inputs,
    std::vector<OrtNodeUnitIODef> outputs,
    std::string name = "",
    std::string domain = "",
    int since_version = 1,
    size_t index = 0) {
  auto impl = std::make_unique<detail::MockNodeUnitImpl>();
  InstallFakeGraphApiStubs(impl->ctor_api);

  auto make_slots = [&impl](const std::vector<OrtNodeUnitIODef>& defs) {
    std::vector<FakeValueInfo*> slots;
    slots.reserve(defs.size());
    for (const auto& d : defs) {
      // Empty-name IODef => absent optional input (nullptr slot).
      slots.push_back(d.name.empty() ? nullptr
                                     : detail::AddFakeVi(*impl, d.name, d.type, d.shape));
    }
    return slots;
  };

  FakeNode node;
  node.name = std::move(name);
  node.op_type = std::move(op_type);
  node.domain = std::move(domain);
  node.since_version = since_version;
  node.id = index;
  node.inputs = make_slots(inputs);
  node.outputs = make_slots(outputs);
  impl->nodes.push_back(std::move(node));

  const OrtNode* node_ptr = impl->nodes.back().AsNode();
  impl->unit = std::make_unique<OrtNodeUnit>(node_ptr, impl->ctor_api);
  impl->global_guard = std::make_unique<OrtGlobalApiOverride>(&impl->ctor_api);

  return MockNodeUnit(std::move(impl));
}

// Build a QDQGroup OrtNodeUnit backed by a fake graph. Each input IODef that
// carries a quant_param is wrapped in a synthesized DequantizeLinear node
// (DQ.input = {quant_value, scale, zp}); the target node consumes the DQ's
// float output. An output IODef with a quant_param is wrapped in a synthesized
// QuantizeLinear node (Q.input = {float_value, scale, zp}); the target node
// produces the float value the Q consumes. Plain (no-quant) IODefs feed / are
// fed by the target node directly. Empty-name input IODef => absent optional
// input (nullptr slot).
//
// This mirrors a real DQ*->target->Q* group closely enough that the production
// OrtNodeUnit(graph, node_group, ort_api) ctor reconstructs IODefs with the same
// name / type / shape / quant_param the caller specified — so the op-builder
// produces an identical QNN graph to a genuine QDQ partition.
inline MockNodeUnit MakeMockQDQNodeUnit(
    std::string op_type,
    std::vector<OrtNodeUnitIODef> inputs,
    std::vector<OrtNodeUnitIODef> outputs,
    std::string name = "",
    std::string domain = "",
    int since_version = 1,
    size_t index = 0) {
  auto impl = std::make_unique<detail::MockNodeUnitImpl>();
  InstallFakeGraphApiStubs(impl->ctor_api);

  std::vector<const OrtNode*> dq_nodes;
  std::vector<const OrtNode*> q_nodes;
  std::vector<FakeValueInfo*> target_inputs;
  std::vector<FakeValueInfo*> target_outputs;
  target_inputs.reserve(inputs.size());
  target_outputs.reserve(outputs.size());

  // ---- Inputs: quantized => DQ node; plain => direct; empty => nullptr slot ----
  for (const auto& d : inputs) {
    if (d.name.empty()) {
      target_inputs.push_back(nullptr);
      continue;
    }
    if (!d.quant_param.has_value()) {
      target_inputs.push_back(detail::AddFakeVi(*impl, d.name, d.type, d.shape));
      continue;
    }
    // DequantizeLinear: input[0] carries the caller's name/type/shape (this is
    // what GetQDQIODefs parses into the IODef); input[1,2] are the scale/zp
    // registry sentinels stored verbatim as the IODef's quant_param.
    FakeValueInfo* dq_in = detail::AddFakeVi(*impl, d.name, d.type, d.shape);
    FakeNode dq;
    dq.op_type = "DequantizeLinear";
    dq.since_version = 13;
    dq.inputs.push_back(dq_in);
    dq.inputs.push_back(detail::AsFakeVi(d.quant_param->scale));
    if (d.quant_param->zero_point != nullptr) {
      dq.inputs.push_back(detail::AsFakeVi(d.quant_param->zero_point));
    }
    impl->nodes.push_back(std::move(dq));
    const FakeNode* dq_ptr = &impl->nodes.back();
    dq_nodes.push_back(dq_ptr->AsNode());

    // The dequantized (float) value the target node actually consumes.
    FakeValueInfo* slot = detail::AddFakeVi(*impl, d.name + "_dq",
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, d.shape);
    slot->producer = dq_ptr;
    target_inputs.push_back(slot);
  }

  // ---- Outputs: quantized => Q node; plain => direct ----
  for (const auto& d : outputs) {
    if (!d.quant_param.has_value()) {
      target_outputs.push_back(detail::AddFakeVi(*impl, d.name, d.type, d.shape));
      continue;
    }
    // The float value the target node produces, consumed by a QuantizeLinear.
    FakeValueInfo* pre_q = detail::AddFakeVi(*impl, d.name + "_pre_q",
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, d.shape);
    // QuantizeLinear: output[0] carries the caller's name/type/shape (parsed by
    // GetQDQIODefs); input[1,2] are the scale/zp registry sentinels.
    FakeValueInfo* q_out = detail::AddFakeVi(*impl, d.name, d.type, d.shape);
    FakeNode q;
    q.op_type = "QuantizeLinear";
    q.since_version = 13;
    q.inputs.push_back(pre_q);
    q.inputs.push_back(detail::AsFakeVi(d.quant_param->scale));
    if (d.quant_param->zero_point != nullptr) {
      q.inputs.push_back(detail::AsFakeVi(d.quant_param->zero_point));
    }
    q.outputs.push_back(q_out);
    impl->nodes.push_back(std::move(q));
    const FakeNode* q_ptr = &impl->nodes.back();
    q_nodes.push_back(q_ptr->AsNode());

    pre_q->consumers.push_back(q_ptr);
    target_outputs.push_back(pre_q);
  }

  FakeNode target;
  target.name = std::move(name);
  target.op_type = std::move(op_type);
  target.domain = std::move(domain);
  target.since_version = since_version;
  target.id = index;
  target.inputs = std::move(target_inputs);
  target.outputs = std::move(target_outputs);
  impl->nodes.push_back(std::move(target));
  const OrtNode* target_ptr = impl->nodes.back().AsNode();

  QDQ::OrtNodeGroup node_group;
  node_group.dq_nodes = std::move(dq_nodes);
  node_group.q_nodes = std::move(q_nodes);
  node_group.target_node = target_ptr;
  node_group.redundant_clip_node = nullptr;

  // graph arg is unused by the ctor; pass nullptr.
  impl->unit = std::make_unique<OrtNodeUnit>(static_cast<const OrtGraph*>(nullptr),
                                             node_group, impl->ctor_api);
  impl->global_guard = std::make_unique<OrtGlobalApiOverride>(&impl->ctor_api);

  return MockNodeUnit(std::move(impl));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
