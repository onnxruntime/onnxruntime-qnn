// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Fake OrtGraph / OrtNode / OrtValueInfo for QNN EP unit tests.
//
// Background:
//   OrtGraph, OrtNode, OrtValueInfo, OrtTypeInfo, OrtTensorTypeAndShapeInfo
//   are opaque C handle types in the ORT public C API. The EP only accesses
//   them through OrtApi function pointers — never by dereferencing the
//   pointer directly. This means tests can freely reinterpret_cast a plain
//   POD struct pointer as any of these opaque types as long as our installed
//   stubs cast back to the same plain struct before reading fields.
//
//   This approach replaces the old qnn_mock_ort_graph.cc / qnn_mock_ort_node.cc
//   files which inherited from ORT's internal abstract classes
//   (core/graph/abi_graph_types.h) — those private headers are forbidden by
//   the new EP/ORT decoupling policy.
//
// Usage:
//   FakeValueInfo input_x{"x", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, 4}};
//   FakeNode identity{"identity_0", "Identity", "", 13, {&input_x}, {&output_y}};
//   FakeGraph graph{ {identity}, {&input_x}, {&output_y}, {} };
//   OrtApi stub_ort_api{};
//   InstallFakeGraphApiStubs(stub_ort_api);
//   // pass graph.AsGraph() / node.AsNode() / value_info.AsValueInfo() to EP code

#pragma once

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

// FakeOrtValue is declared here so FakeValueInfo can hold a pointer to it.
// Full definition follows after FakeValueInfo.
struct FakeOrtValue;

// ---------------------------------------------------------------------------
// FakeValueInfo
//
// Acts as OrtValueInfo*, OrtTypeInfo*, AND OrtTensorTypeAndShapeInfo*
// simultaneously — the installed stubs cast back to FakeValueInfo for every
// kind of opaque pointer. This works because the EP code only flows the
// pointers through OrtApi function pointers; it never inspects them as
// concrete ORT types.
// ---------------------------------------------------------------------------
struct FakeValueInfo {
  std::string name;
  ONNXTensorElementDataType elem_type;
  std::vector<int64_t> shape;
  // Non-owning pointer to a FakeOrtValue for constant initializers.
  // Null for non-initializer value infos. Used by ValueInfo_GetInitializerValue.
  FakeOrtValue* initializer_value = nullptr;

  const OrtValueInfo* AsValueInfo() const {
    return reinterpret_cast<const OrtValueInfo*>(this);
  }
  const OrtTypeInfo* AsTypeInfo() const {
    return reinterpret_cast<const OrtTypeInfo*>(this);
  }
  const OrtTensorTypeAndShapeInfo* AsTensorInfo() const {
    return reinterpret_cast<const OrtTensorTypeAndShapeInfo*>(this);
  }
};

// ---------------------------------------------------------------------------
// FakeOrtValue
//
// Acts as both OrtValue* and OrtTensorTypeAndShapeInfo* simultaneously.
//
// The first three fields (dummy_name, elem_type, shape) are layout-compatible
// with FakeValueInfo so the existing GetTensorElementType / GetDimensionsCount
// / GetDimensions stubs, which reinterpret_cast to FakeValueInfo, correctly
// decode a FakeOrtValue* that has been cast to OrtTensorTypeAndShapeInfo*.
// This avoids duplicating or overriding those stubs for the OrtValue code path.
//
// Keep FakeOrtValue objects on the stack (or as struct members with lifetimes
// that enclose EP code calls); ReleaseTensorTypeAndShapeInfo is a no-op stub,
// so no heap allocation or deallocation is performed.
//
// Usage:
//   FakeOrtValue scale = FakeOrtValue::MakeFloat(0.5f);
//   scale_vi.initializer_value = &scale;  // tie to a FakeValueInfo initializer
// ---------------------------------------------------------------------------
struct FakeOrtValue {
  // Layout-compatible fields (same type and order as FakeValueInfo[0..2])
  std::string dummy_name;                                                     // FakeValueInfo::name
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;  // FakeValueInfo::elem_type
  std::vector<int64_t> shape;                                                 // FakeValueInfo::shape (empty = scalar)
  // Scalar payload (read by GetTensorMutableData)
  float float_data = 0.0f;
  double double_data = 0.0;

  const OrtValue* AsOrtValue() const { return reinterpret_cast<const OrtValue*>(this); }

  static FakeOrtValue MakeFloat(float v) {
    FakeOrtValue r;
    r.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    r.float_data = v;
    return r;
  }
  static FakeOrtValue MakeDouble(double v) {
    FakeOrtValue r;
    r.elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    r.double_data = v;
    return r;
  }
};
static_assert(offsetof(FakeOrtValue, dummy_name) == offsetof(FakeValueInfo, name),
              "FakeOrtValue/FakeValueInfo layout drift: name offset mismatch");
static_assert(offsetof(FakeOrtValue, elem_type) == offsetof(FakeValueInfo, elem_type),
              "FakeOrtValue/FakeValueInfo layout drift: elem_type offset mismatch");
static_assert(offsetof(FakeOrtValue, shape) == offsetof(FakeValueInfo, shape),
              "FakeOrtValue/FakeValueInfo layout drift: shape offset mismatch");

// ---------------------------------------------------------------------------
// FakeOpAttr
//
// Acts as OrtOpAttr* in tests that exercise Node_GetAttributeByName /
// OpAttr_GetType / ReadOpAttr stubs (e.g., OrtNodeAttrHelper::Get(...)).
//
// Holds a name + type + value. Currently supports ORT_OP_ATTR_STRING and
// ORT_OP_ATTR_INT. Add more value fields if other attribute types are needed.
// ---------------------------------------------------------------------------
struct FakeOpAttr {
  // For debugging only. Node_GetAttributeByName looks up attrs by the map key in
  // FakeNode::attrs, not by this field — stubs never read FakeOpAttr::name.
  std::string name;
  OrtOpAttrType type = OrtOpAttrType::ORT_OP_ATTR_STRING;
  std::string string_value;
  int64_t int64_value = 0;

  const OrtOpAttr* AsOpAttr() const { return reinterpret_cast<const OrtOpAttr*>(this); }

  static FakeOpAttr MakeString(std::string name, std::string val) {
    FakeOpAttr a;
    a.name = std::move(name);
    a.type = OrtOpAttrType::ORT_OP_ATTR_STRING;
    a.string_value = std::move(val);
    return a;
  }
  static FakeOpAttr MakeInt64(std::string name, int64_t val) {
    FakeOpAttr a;
    a.name = std::move(name);
    a.type = OrtOpAttrType::ORT_OP_ATTR_INT;
    a.int64_value = val;
    return a;
  }
};

// ---------------------------------------------------------------------------
// FakeNode
//
// Pointers to inputs/outputs are observed (non-owning) — caller keeps
// FakeValueInfo objects alive for as long as the node is in use.
// `attrs` maps attribute name to a (non-owning) FakeOpAttr pointer used by
// Node_GetAttributeByName. An empty map represents a node with no attributes.
// ---------------------------------------------------------------------------
struct FakeNode {
  std::string name;
  std::string op_type;
  std::string domain;
  int since_version = 13;
  std::vector<FakeValueInfo*> inputs;
  std::vector<FakeValueInfo*> outputs;
  std::unordered_map<std::string, FakeOpAttr*> attrs;

  const OrtNode* AsNode() const {
    return reinterpret_cast<const OrtNode*>(this);
  }
};

// ---------------------------------------------------------------------------
// FakeGraph
//
// Nodes are owned by the FakeGraph; FakeValueInfo pointers are observed.
// ---------------------------------------------------------------------------
struct FakeGraph {
  std::vector<FakeNode> nodes;
  std::vector<FakeValueInfo*> inputs;
  std::vector<FakeValueInfo*> outputs;
  std::vector<FakeValueInfo*> initializers;

  const OrtGraph* AsGraph() const {
    return reinterpret_cast<const OrtGraph*>(this);
  }
};

// ---------------------------------------------------------------------------
// InstallFakeGraphApiStubs
//
// Installs OrtApi function-pointer stubs that decode opaque handles back to
// FakeGraph / FakeNode / FakeValueInfo via reinterpret_cast. Designed to be
// safe to call on a zero-initialized OrtApi{} — does not depend on or modify
// any other stub.
//
// Stub coverage (sufficient for SetGraphInputOutputInfo / ComposeGraph /
// LogTensorDetails / SetupTensors paths):
//   - GetValueInfoName, GetValueInfoTypeInfo
//   - CastTypeInfoToTensorInfo, GetTensorElementType
//   - GetDimensionsCount, GetDimensions, GetSymbolicDimensions
//   - TensorTypeAndShape_HasShape
//   - Graph_GetNum{Nodes,Inputs,Outputs,Initializers}, Graph_GetParentNode
//   - Graph_Get{Nodes,Inputs,Outputs,Initializers}
//   - Node_Get{Name,OperatorType,Domain,SinceVersion,EpName}
//   - Node_GetNum{Inputs,Outputs,ImplicitInputs,Attributes,Subgraphs}
//   - Node_Get{Inputs,Outputs,ImplicitInputs,Attributes,Subgraphs}
//   - Node_GetAttributeByName (looks up FakeOpAttr in FakeNode::attrs)
//   - OpAttr_GetType, ReadOpAttr (ORT_OP_ATTR_STRING and ORT_OP_ATTR_INT)
//
// Anything not in this list is left untouched and the caller can replace it
// with a more specific test stub.
// ---------------------------------------------------------------------------
inline void InstallFakeGraphApiStubs(OrtApi& api) {
  // ---- Release* no-ops ----
  // Fake objects are stack-owned by tests; never deallocated through ORT.
  // When tests override the global API with a stub (OrtGlobalApiOverride),
  // Ort::Status / OrtTypeInfo / OrtTensorTypeAndShapeInfo destructors call
  // these through Ort::GetApi(). Leaving them null would crash on every
  // RAII teardown.
  api.ReleaseTypeInfo = [](OrtTypeInfo*) noexcept {};
  api.ReleaseTensorTypeAndShapeInfo = [](OrtTensorTypeAndShapeInfo*) noexcept {};

  // ---- OrtStatus minimal heap-allocated wrapper ----
  // EP code constructs OrtStatus via Ort::Status(msg, code) which calls
  // CreateStatus, and inspects results via GetErrorCode / GetErrorMessage.
  struct FakeOrtStatus {
    OrtErrorCode code;
    std::string message;
  };
  api.CreateStatus = [](OrtErrorCode code, const char* msg) noexcept -> OrtStatus* {
    auto* s = new FakeOrtStatus{code, msg ? msg : ""};
    return reinterpret_cast<OrtStatus*>(s);
  };
  api.GetErrorCode = [](const OrtStatus* status) noexcept -> OrtErrorCode {
    return reinterpret_cast<const FakeOrtStatus*>(status)->code;
  };
  api.GetErrorMessage = [](const OrtStatus* status) noexcept -> const char* {
    return reinterpret_cast<const FakeOrtStatus*>(status)->message.c_str();
  };
  api.ReleaseStatus = [](OrtStatus* status) noexcept {
    delete reinterpret_cast<FakeOrtStatus*>(status);
  };

  // ---- OrtValueInfo / OrtTypeInfo / OrtTensorTypeAndShapeInfo ----
  api.GetValueInfoName = [](const OrtValueInfo* vi, const char** name) noexcept -> OrtStatus* {
    *name = reinterpret_cast<const FakeValueInfo*>(vi)->name.c_str();
    return nullptr;
  };
  api.GetValueInfoTypeInfo = [](const OrtValueInfo* vi, const OrtTypeInfo** out) noexcept -> OrtStatus* {
    *out = reinterpret_cast<const FakeValueInfo*>(vi)->AsTypeInfo();
    return nullptr;
  };
  api.CastTypeInfoToTensorInfo = [](const OrtTypeInfo* ti,
                                    const OrtTensorTypeAndShapeInfo** out) noexcept -> OrtStatus* {
    *out = reinterpret_cast<const FakeValueInfo*>(ti)->AsTensorInfo();
    return nullptr;
  };
  api.GetTensorElementType = [](const OrtTensorTypeAndShapeInfo* info,
                                ONNXTensorElementDataType* t) noexcept -> OrtStatus* {
    *t = reinterpret_cast<const FakeValueInfo*>(info)->elem_type;
    return nullptr;
  };
  api.GetDimensionsCount = [](const OrtTensorTypeAndShapeInfo* info, size_t* count) noexcept -> OrtStatus* {
    *count = reinterpret_cast<const FakeValueInfo*>(info)->shape.size();
    return nullptr;
  };
  api.GetDimensions = [](const OrtTensorTypeAndShapeInfo* info, int64_t* dims, size_t count) noexcept -> OrtStatus* {
    const auto& shape = reinterpret_cast<const FakeValueInfo*>(info)->shape;
    for (size_t i = 0; i < count && i < shape.size(); ++i) dims[i] = shape[i];
    return nullptr;
  };
  api.GetSymbolicDimensions = [](const OrtTensorTypeAndShapeInfo*,
                                 const char** dim_params, size_t count) noexcept -> OrtStatus* {
    for (size_t i = 0; i < count; ++i) dim_params[i] = "";
    return nullptr;
  };
  api.TensorTypeAndShape_HasShape = [](const OrtTensorTypeAndShapeInfo*) noexcept -> bool {
    return true;
  };

  // ---- OrtGraph ----
  api.Graph_GetNumNodes = [](const OrtGraph* g, size_t* n) noexcept -> OrtStatus* {
    *n = reinterpret_cast<const FakeGraph*>(g)->nodes.size();
    return nullptr;
  };
  api.Graph_GetNodes = [](const OrtGraph* g, const OrtNode** nodes, size_t count) noexcept -> OrtStatus* {
    const auto& fg = *reinterpret_cast<const FakeGraph*>(g);
    for (size_t i = 0; i < count && i < fg.nodes.size(); ++i) {
      nodes[i] = fg.nodes[i].AsNode();
    }
    return nullptr;
  };
  api.Graph_GetNumInputs = [](const OrtGraph* g, size_t* n) noexcept -> OrtStatus* {
    *n = reinterpret_cast<const FakeGraph*>(g)->inputs.size();
    return nullptr;
  };
  api.Graph_GetInputs = [](const OrtGraph* g, const OrtValueInfo** vis, size_t count) noexcept -> OrtStatus* {
    const auto& fg = *reinterpret_cast<const FakeGraph*>(g);
    for (size_t i = 0; i < count && i < fg.inputs.size(); ++i) {
      vis[i] = fg.inputs[i] ? fg.inputs[i]->AsValueInfo() : nullptr;
    }
    return nullptr;
  };
  api.Graph_GetNumOutputs = [](const OrtGraph* g, size_t* n) noexcept -> OrtStatus* {
    *n = reinterpret_cast<const FakeGraph*>(g)->outputs.size();
    return nullptr;
  };
  api.Graph_GetOutputs = [](const OrtGraph* g, const OrtValueInfo** vis, size_t count) noexcept -> OrtStatus* {
    const auto& fg = *reinterpret_cast<const FakeGraph*>(g);
    for (size_t i = 0; i < count && i < fg.outputs.size(); ++i) {
      vis[i] = fg.outputs[i] ? fg.outputs[i]->AsValueInfo() : nullptr;
    }
    return nullptr;
  };
  api.Graph_GetNumInitializers = [](const OrtGraph* g, size_t* n) noexcept -> OrtStatus* {
    *n = reinterpret_cast<const FakeGraph*>(g)->initializers.size();
    return nullptr;
  };
  api.Graph_GetInitializers = [](const OrtGraph* g, const OrtValueInfo** vis, size_t count) noexcept -> OrtStatus* {
    const auto& fg = *reinterpret_cast<const FakeGraph*>(g);
    for (size_t i = 0; i < count && i < fg.initializers.size(); ++i) {
      vis[i] = fg.initializers[i] ? fg.initializers[i]->AsValueInfo() : nullptr;
    }
    return nullptr;
  };
  api.Graph_GetParentNode = [](const OrtGraph*, const OrtNode** n) noexcept -> OrtStatus* {
    *n = nullptr;  // no parent (top-level graph)
    return nullptr;
  };

  // ---- OrtNode ----
  api.Node_GetId = [](const OrtNode* n, size_t* id) noexcept -> OrtStatus* {
    // Pointer address is unique per FakeNode — sufficient for "node id" semantics.
    *id = reinterpret_cast<size_t>(n);
    return nullptr;
  };
  api.Node_GetName = [](const OrtNode* n, const char** out) noexcept -> OrtStatus* {
    *out = reinterpret_cast<const FakeNode*>(n)->name.c_str();
    return nullptr;
  };
  api.Node_GetOperatorType = [](const OrtNode* n, const char** out) noexcept -> OrtStatus* {
    *out = reinterpret_cast<const FakeNode*>(n)->op_type.c_str();
    return nullptr;
  };
  api.Node_GetDomain = [](const OrtNode* n, const char** out) noexcept -> OrtStatus* {
    *out = reinterpret_cast<const FakeNode*>(n)->domain.c_str();
    return nullptr;
  };
  api.Node_GetSinceVersion = [](const OrtNode* n, int* v) noexcept -> OrtStatus* {
    *v = reinterpret_cast<const FakeNode*>(n)->since_version;
    return nullptr;
  };
  api.Node_GetEpName = [](const OrtNode*, const char** out) noexcept -> OrtStatus* {
    *out = nullptr;  // EP not yet assigned in fake graphs
    return nullptr;
  };
  api.Node_GetNumInputs = [](const OrtNode* n, size_t* count) noexcept -> OrtStatus* {
    *count = reinterpret_cast<const FakeNode*>(n)->inputs.size();
    return nullptr;
  };
  api.Node_GetInputs = [](const OrtNode* n, const OrtValueInfo** vis, size_t count) noexcept -> OrtStatus* {
    const auto& fn = *reinterpret_cast<const FakeNode*>(n);
    for (size_t i = 0; i < count && i < fn.inputs.size(); ++i) {
      vis[i] = fn.inputs[i] ? fn.inputs[i]->AsValueInfo() : nullptr;
    }
    return nullptr;
  };
  api.Node_GetNumOutputs = [](const OrtNode* n, size_t* count) noexcept -> OrtStatus* {
    *count = reinterpret_cast<const FakeNode*>(n)->outputs.size();
    return nullptr;
  };
  api.Node_GetOutputs = [](const OrtNode* n, const OrtValueInfo** vis, size_t count) noexcept -> OrtStatus* {
    const auto& fn = *reinterpret_cast<const FakeNode*>(n);
    for (size_t i = 0; i < count && i < fn.outputs.size(); ++i) {
      vis[i] = fn.outputs[i] ? fn.outputs[i]->AsValueInfo() : nullptr;
    }
    return nullptr;
  };
  api.Node_GetNumImplicitInputs = [](const OrtNode*, size_t* count) noexcept -> OrtStatus* {
    *count = 0;
    return nullptr;
  };
  api.Node_GetImplicitInputs = [](const OrtNode*, const OrtValueInfo**, size_t) noexcept -> OrtStatus* {
    return nullptr;
  };
  api.Node_GetNumAttributes = [](const OrtNode*, size_t* count) noexcept -> OrtStatus* {
    *count = 0;
    return nullptr;
  };
  api.Node_GetAttributes = [](const OrtNode*, const OrtOpAttr**, size_t) noexcept -> OrtStatus* {
    return nullptr;
  };
  api.Node_GetNumSubgraphs = [](const OrtNode*, size_t* count) noexcept -> OrtStatus* {
    *count = 0;
    return nullptr;
  };
  api.Node_GetSubgraphs = [](const OrtNode*, const OrtGraph**, size_t, const char**) noexcept -> OrtStatus* {
    return nullptr;
  };
  api.Node_GetAttributeByName = [](const OrtNode* n, const char* name, const OrtOpAttr** out) noexcept -> OrtStatus* {
    auto& fn = *reinterpret_cast<const FakeNode*>(n);
    auto it = fn.attrs.find(name);
    *out = (it == fn.attrs.end() || it->second == nullptr) ? nullptr : it->second->AsOpAttr();
    return nullptr;
  };
  api.Node_GetGraph = [](const OrtNode*, const OrtGraph** g) noexcept -> OrtStatus* {
    *g = nullptr;
    return nullptr;
  };

  // ---- OrtValue (scalar constant initializer) ----
  //
  // ValueInfo_GetInitializerValue: return the FakeOrtValue stored in
  // FakeValueInfo::initializer_value (null if not a constant initializer).
  api.ValueInfo_GetInitializerValue = [](const OrtValueInfo* vi, const OrtValue** out) noexcept -> OrtStatus* {
    auto* fov = reinterpret_cast<const FakeValueInfo*>(vi)->initializer_value;
    *out = fov ? fov->AsOrtValue() : nullptr;
    return nullptr;
  };
  // GetTensorTypeAndShape: the OrtValue* is a FakeOrtValue*; return it cast to
  // OrtTensorTypeAndShapeInfo*. The layout-compatible FakeOrtValue fields
  // (elem_type, shape) are read correctly by the existing GetTensorElementType
  // and GetDimensionsCount stubs without any modification.
  // ReleaseTensorTypeAndShapeInfo remains a no-op (no heap allocation here).
  api.GetTensorTypeAndShape = [](const OrtValue* v, OrtTensorTypeAndShapeInfo** out) noexcept -> OrtStatus* {
    *out = const_cast<OrtTensorTypeAndShapeInfo*>(
        reinterpret_cast<const OrtTensorTypeAndShapeInfo*>(v));
    return nullptr;
  };
  // GetTensorMutableData: production code (qnn_ep_utils.cc) calls the mutable variant via api_ptrs_.
  api.GetTensorMutableData = [](OrtValue* v, void** out) noexcept -> OrtStatus* {
    auto* fov = reinterpret_cast<FakeOrtValue*>(v);
    *out = (fov->elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
               ? static_cast<void*>(&fov->float_data)
               : static_cast<void*>(&fov->double_data);
    return nullptr;
  };

  // ---- ValueInfo producer/consumer queries ----
  // Return nullptr/empty by default. Tests that need a real producer can
  // override this stub after calling InstallFakeGraphApiStubs.
  api.ValueInfo_GetValueProducer = [](const OrtValueInfo*, const OrtNode** producer,
                                      size_t* output_index) noexcept -> OrtStatus* {
    if (producer) *producer = nullptr;
    if (output_index) *output_index = 0;
    return nullptr;
  };
  api.ValueInfo_GetValueConsumers = [](const OrtValueInfo*, const OrtNode** consumers,
                                       int64_t* indices, size_t count) noexcept -> OrtStatus* {
    for (size_t i = 0; i < count; ++i) {
      consumers[i] = nullptr;
      if (indices) indices[i] = 0;
    }
    return nullptr;
  };

  // ---- OrtOpAttr (read attribute type + value) ----
  // OpAttr_GetType: returns FakeOpAttr::type. Used by CheckAttrType in
  // ConstOpAttr::GetValue<T>() to verify the requested type matches.
  api.OpAttr_GetType = [](const OrtOpAttr* attr, OrtOpAttrType* out) noexcept -> OrtStatus* {
    *out = reinterpret_cast<const FakeOpAttr*>(attr)->type;
    return nullptr;
  };
  // ReadOpAttr: probe-then-read pattern used by ConstOpAttr::GetValue<T>().
  //   First call: buf=nullptr, buf_size=0 → write required size into *out_size.
  //   Second call: buf!=nullptr → copy at most buf_size bytes into buf, write
  //   actual size into *out_size.
  // Handles ORT_OP_ATTR_STRING (string_value) and ORT_OP_ATTR_INT (int64_value).
  api.ReadOpAttr = [](const OrtOpAttr* attr, OrtOpAttrType expected_type,
                      void* buf, size_t buf_size, size_t* out_size) noexcept -> OrtStatus* {
    auto* fa = reinterpret_cast<const FakeOpAttr*>(attr);
    if (fa->type != expected_type) {
      // Type mismatch. The scalar numeric path (Ort::ConstOpAttr::GetNumericValue)
      // calls ReadOpAttr directly with no preceding CheckAttrType, so a wrong-typed
      // numeric Get reaches here; the real API returns an error, which the caller
      // (OrtNodeAttrHelper::Get) maps to its default value. Return a real error to
      // match that contract. (STRING/array reads never hit this branch because
      // ConstOpAttr::GetValue<T>() calls CheckAttrType via OpAttr_GetType first.)
      *out_size = 0;
      return reinterpret_cast<OrtStatus*>(
          new FakeOrtStatus{ORT_INVALID_ARGUMENT, "ReadOpAttr: attribute type mismatch"});
    }
    if (expected_type == OrtOpAttrType::ORT_OP_ATTR_STRING) {
      // Byte count of string_value, NOT including a null terminator — matches
      // the real ReadOpAttr contract and how Ort::ConstOpAttr::GetValue<std::string>
      // consumes it (result.resize(size) + memcpy(size), no trailing '\0').
      *out_size = fa->string_value.size();
      if (buf != nullptr && buf_size >= fa->string_value.size()) {
        std::memcpy(buf, fa->string_value.data(), fa->string_value.size());
      }
    } else if (expected_type == OrtOpAttrType::ORT_OP_ATTR_INT) {
      *out_size = sizeof(int64_t);
      if (buf != nullptr && buf_size >= sizeof(int64_t)) {
        std::memcpy(buf, &fa->int64_value, sizeof(int64_t));
      }
    } else {
      *out_size = 0;
    }
    return nullptr;
  };
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
