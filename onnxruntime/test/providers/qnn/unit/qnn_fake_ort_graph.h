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
#include <string>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace test {

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
// FakeNode
//
// Pointers to inputs/outputs are observed (non-owning) — caller keeps
// FakeValueInfo objects alive for as long as the node is in use.
// ---------------------------------------------------------------------------
struct FakeNode {
  std::string name;
  std::string op_type;
  std::string domain;
  int since_version = 13;
  std::vector<FakeValueInfo*> inputs;
  std::vector<FakeValueInfo*> outputs;

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
  api.Node_GetAttributeByName = [](const OrtNode*, const char*, const OrtOpAttr** out) noexcept -> OrtStatus* {
    *out = nullptr;  // attribute "not found" — matches our 0-attribute FakeNode
    return nullptr;
  };
  api.Node_GetGraph = [](const OrtNode*, const OrtGraph** g) noexcept -> OrtStatus* {
    *g = nullptr;
    return nullptr;
  };
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_INTERNAL_SYMBOL_ACCESS
