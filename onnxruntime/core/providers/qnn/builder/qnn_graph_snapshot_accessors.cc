// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_graph_snapshot_accessors.h"

#include <cstring>
#include <string>

namespace onnxruntime {
namespace qnn {

// Cast helpers between fake opaque handles and snapshot objects.
namespace {

const SnapshotGraph* AsGraph(const OrtGraph* h) { return reinterpret_cast<const SnapshotGraph*>(h); }
const SnapshotNode* AsNode(const OrtNode* h) { return reinterpret_cast<const SnapshotNode*>(h); }
const SnapshotValueInfo* AsValueInfo(const OrtValueInfo* h) {
  return reinterpret_cast<const SnapshotValueInfo*>(h);
}
const SnapshotAttr* AsAttr(const OrtOpAttr* h) { return reinterpret_cast<const SnapshotAttr*>(h); }

const OrtGraph* Handle(const SnapshotGraph* g) { return reinterpret_cast<const OrtGraph*>(g); }
const OrtNode* Handle(const SnapshotNode* n) { return reinterpret_cast<const OrtNode*>(n); }
const OrtValueInfo* Handle(const SnapshotValueInfo* v) { return reinterpret_cast<const OrtValueInfo*>(v); }
const OrtOpAttr* Handle(const SnapshotAttr* a) { return reinterpret_cast<const OrtOpAttr*>(a); }

// Thread-local active shim, set by ActiveShimGuard around a compose call.
const SnapshotShim*& ActiveShim() {
  thread_local const SnapshotShim* active = nullptr;
  return active;
}

template <typename SnapT>
OrtStatus* CopyHandles(const std::vector<SnapT*>& src, const void** dst, size_t count) {
  if (count < src.size()) {
    return nullptr;
  }
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = Handle(src[i]);
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL Graph_GetNumNodes(const OrtGraph* graph, size_t* out) noexcept {
  *out = AsGraph(graph)->node_order.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL Graph_GetNodes(const OrtGraph* graph, const OrtNode** nodes, size_t count) noexcept {
  return CopyHandles(AsGraph(graph)->node_order, reinterpret_cast<const void**>(nodes), count);
}

OrtStatus* ORT_API_CALL Graph_GetNumInputs(const OrtGraph* graph, size_t* out) noexcept {
  *out = AsGraph(graph)->graph_inputs.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL Graph_GetInputs(const OrtGraph* graph, const OrtValueInfo** vis, size_t count) noexcept {
  return CopyHandles(AsGraph(graph)->graph_inputs, reinterpret_cast<const void**>(vis), count);
}

OrtStatus* ORT_API_CALL Graph_GetNumOutputs(const OrtGraph* graph, size_t* out) noexcept {
  *out = AsGraph(graph)->graph_outputs.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL Graph_GetOutputs(const OrtGraph* graph, const OrtValueInfo** vis, size_t count) noexcept {
  return CopyHandles(AsGraph(graph)->graph_outputs, reinterpret_cast<const void**>(vis), count);
}

OrtStatus* ORT_API_CALL Graph_GetNumInitializers(const OrtGraph* graph, size_t* out) noexcept {
  *out = AsGraph(graph)->initializers.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL Graph_GetInitializers(const OrtGraph* graph, const OrtValueInfo** vis, size_t count) noexcept {
  return CopyHandles(AsGraph(graph)->initializers, reinterpret_cast<const void**>(vis), count);
}

OrtStatus* ORT_API_CALL Graph_GetParentNode(const OrtGraph* /*graph*/, const OrtNode** node) noexcept {
  *node = nullptr;  // No parent; nested subgraphs not supported.
  return nullptr;
}

OrtStatus* ORT_API_CALL Graph_GetModelPath(const OrtGraph* graph, const ORTCHAR_T** out) noexcept {
  *out = AsGraph(graph)->model_path.c_str();
  return nullptr;
}

OrtStatus* ORT_API_CALL Graph_GetName(const OrtGraph* graph, const char** out) noexcept {
  *out = AsGraph(graph)->name.c_str();
  return nullptr;
}

// Defensive stubs for accessors the float path is not expected to call.
OrtStatus* ORT_API_CALL Graph_GetModelMetadata(const OrtGraph*, OrtModelMetadata**) noexcept {
  return ActiveShim()->ShimmedApi().CreateStatus(ORT_NOT_IMPLEMENTED, "shim: Graph_GetModelMetadata");
}
OrtStatus* ORT_API_CALL Graph_GetGraphView(const OrtGraph*, const OrtNode**, size_t, OrtGraph**) noexcept {
  return ActiveShim()->ShimmedApi().CreateStatus(ORT_NOT_IMPLEMENTED, "shim: Graph_GetGraphView");
}
OrtStatus* ORT_API_CALL Graph_GetNumOperatorSets(const OrtGraph*, size_t*) noexcept {
  return ActiveShim()->ShimmedApi().CreateStatus(ORT_NOT_IMPLEMENTED, "shim: Graph_GetNumOperatorSets");
}
OrtStatus* ORT_API_CALL Graph_GetOnnxIRVersion(const OrtGraph*, int64_t*) noexcept {
  return ActiveShim()->ShimmedApi().CreateStatus(ORT_NOT_IMPLEMENTED, "shim: Graph_GetOnnxIRVersion");
}
OrtStatus* ORT_API_CALL Graph_GetOperatorSets(const OrtGraph*, const char**, int64_t*, size_t) noexcept {
  return ActiveShim()->ShimmedApi().CreateStatus(ORT_NOT_IMPLEMENTED, "shim: Graph_GetOperatorSets");
}
OrtStatus* ORT_API_CALL Node_GetGraph(const OrtNode*, const OrtGraph** out) noexcept {
  *out = nullptr;
  return nullptr;
}
OrtStatus* ORT_API_CALL ValueInfo_IsFromOuterScope(const OrtValueInfo*, bool* out) noexcept {
  *out = false;
  return nullptr;
}
OrtStatus* ORT_API_CALL ValueInfo_IsRequiredGraphInput(const OrtValueInfo*, bool* out) noexcept {
  *out = false;
  return nullptr;
}
OrtStatus* ORT_API_CALL ValueInfo_IsOptionalGraphInput(const OrtValueInfo*, bool* out) noexcept {
  *out = false;
  return nullptr;
}
OrtStatus* ORT_API_CALL OpAttr_GetTensorAttributeAsOrtValue(const OrtOpAttr*, OrtValue**) noexcept {
  return ActiveShim()->ShimmedApi().CreateStatus(ORT_NOT_IMPLEMENTED, "shim: OpAttr_GetTensorAttributeAsOrtValue");
}

OrtStatus* ORT_API_CALL Node_GetId(const OrtNode* node, size_t* out) noexcept {
  *out = AsNode(node)->id;
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetName(const OrtNode* node, const char** out) noexcept {
  *out = AsNode(node)->name.c_str();
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetOperatorType(const OrtNode* node, const char** out) noexcept {
  *out = AsNode(node)->op_type.c_str();
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetDomain(const OrtNode* node, const char** out) noexcept {
  *out = AsNode(node)->domain.c_str();
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetSinceVersion(const OrtNode* node, int* out) noexcept {
  *out = AsNode(node)->since_version;
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetNumInputs(const OrtNode* node, size_t* out) noexcept {
  *out = AsNode(node)->inputs.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetInputs(const OrtNode* node, const OrtValueInfo** vis, size_t count) noexcept {
  return CopyHandles(AsNode(node)->inputs, reinterpret_cast<const void**>(vis), count);
}

OrtStatus* ORT_API_CALL Node_GetNumOutputs(const OrtNode* node, size_t* out) noexcept {
  *out = AsNode(node)->outputs.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetOutputs(const OrtNode* node, const OrtValueInfo** vis, size_t count) noexcept {
  return CopyHandles(AsNode(node)->outputs, reinterpret_cast<const void**>(vis), count);
}

OrtStatus* ORT_API_CALL Node_GetNumImplicitInputs(const OrtNode* node, size_t* out) noexcept {
  *out = AsNode(node)->implicit_inputs.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetImplicitInputs(const OrtNode* node, const OrtValueInfo** vis, size_t count) noexcept {
  return CopyHandles(AsNode(node)->implicit_inputs, reinterpret_cast<const void**>(vis), count);
}

OrtStatus* ORT_API_CALL Node_GetNumAttributes(const OrtNode* node, size_t* out) noexcept {
  *out = AsNode(node)->attributes.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetAttributes(const OrtNode* node, const OrtOpAttr** attrs, size_t count) noexcept {
  const SnapshotNode* sn = AsNode(node);
  if (count < sn->attributes.size()) {
    return nullptr;
  }
  for (size_t i = 0; i < sn->attributes.size(); ++i) {
    attrs[i] = Handle(&sn->attributes[i]);
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetAttributeByName(const OrtNode* node, const char* name,
                                                const OrtOpAttr** out) noexcept {
  const SnapshotNode* sn = AsNode(node);
  for (const SnapshotAttr& attr : sn->attributes) {
    if (attr.name == name) {
      *out = Handle(&attr);
      return nullptr;
    }
  }
  *out = nullptr;
  return ActiveShim()->ShimmedApi().CreateStatus(ORT_FAIL, "attribute not found");
}

OrtStatus* ORT_API_CALL Node_GetNumSubgraphs(const OrtNode* /*node*/, size_t* out) noexcept {
  *out = 0;
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetSubgraphs(const OrtNode* /*node*/, const OrtGraph** /*subgraphs*/,
                                          size_t /*num_subgraphs*/, const char** /*attribute_names*/) noexcept {
  return nullptr;
}

OrtStatus* ORT_API_CALL Node_GetEpName(const OrtNode* /*node*/, const char** out) noexcept {
  *out = nullptr;
  return nullptr;
}

OrtStatus* ORT_API_CALL GetValueInfoName(const OrtValueInfo* vi, const char** out) noexcept {
  *out = AsValueInfo(vi)->name.c_str();
  return nullptr;
}

OrtStatus* ORT_API_CALL GetValueInfoTypeInfo(const OrtValueInfo* vi, const OrtTypeInfo** out) noexcept {
  const OrtTypeInfo* ti = ActiveShim()->TypeInfoFor(AsValueInfo(vi));
  if (ti == nullptr) {
    return ActiveShim()->ShimmedApi().CreateStatus(ORT_FAIL, "no type info for value");
  }
  *out = ti;
  return nullptr;
}

OrtStatus* ORT_API_CALL ValueInfo_IsConstantInitializer(const OrtValueInfo* vi, bool* out) noexcept {
  *out = AsValueInfo(vi)->is_constant_initializer;
  return nullptr;
}

OrtStatus* ORT_API_CALL ValueInfo_IsGraphOutput(const OrtValueInfo* vi, bool* out) noexcept {
  *out = AsValueInfo(vi)->is_graph_output;
  return nullptr;
}

OrtStatus* ORT_API_CALL ValueInfo_GetValueProducer(const OrtValueInfo* vi, const OrtNode** node,
                                                   size_t* output_index) noexcept {
  const SnapshotValueInfo* svi = AsValueInfo(vi);
  *node = Handle(svi->producer_node);
  if (output_index != nullptr) {
    *output_index = svi->producer_output_index;
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL ValueInfo_GetValueNumConsumers(const OrtValueInfo* vi, size_t* out) noexcept {
  *out = AsValueInfo(vi)->consumers.size();
  return nullptr;
}

OrtStatus* ORT_API_CALL ValueInfo_GetValueConsumers(const OrtValueInfo* vi, const OrtNode** nodes,
                                                    int64_t* input_indices, size_t count) noexcept {
  const SnapshotValueInfo* svi = AsValueInfo(vi);
  if (count < svi->consumers.size()) {
    return nullptr;
  }
  for (size_t i = 0; i < svi->consumers.size(); ++i) {
    nodes[i] = Handle(svi->consumers[i].first);
    input_indices[i] = svi->consumers[i].second;
  }
  return nullptr;
}

OrtStatus* ORT_API_CALL ValueInfo_GetInitializerValue(const OrtValueInfo* vi, const OrtValue** out) noexcept {
  const OrtValue* v = ActiveShim()->InitializerValueFor(AsValueInfo(vi));
  if (v == nullptr) {
    return ActiveShim()->ShimmedApi().CreateStatus(ORT_FAIL, "no initializer value for value");
  }
  *out = v;
  return nullptr;
}

OrtStatus* ORT_API_CALL ValueInfo_GetExternalInitializerInfo(const OrtValueInfo* /*vi*/,
                                                             OrtExternalInitializerInfo** out) noexcept {
  // Snapshot stores initializer bytes in-line; null means UnpackInitializerData uses the OrtValue path.
  *out = nullptr;
  return nullptr;
}

OrtStatus* ORT_API_CALL OpAttr_GetName(const OrtOpAttr* attr, const char** out) noexcept {
  *out = AsAttr(attr)->name.c_str();
  return nullptr;
}

OrtStatus* ORT_API_CALL OpAttr_GetType(const OrtOpAttr* attr, OrtOpAttrType* out) noexcept {
  *out = AsAttr(attr)->type;
  return nullptr;
}

// ReadOpAttr implements the two-phase protocol: called with data=nullptr it returns byte size in
// `out`; called with a buffer it fills it.
OrtStatus* ORT_API_CALL ReadOpAttr(const OrtOpAttr* attr, OrtOpAttrType type, void* data, size_t len,
                                   size_t* out) noexcept {
  const SnapshotAttr* sa = AsAttr(attr);
  if (sa->type != type) {
    return ActiveShim()->ShimmedApi().CreateStatus(ORT_INVALID_ARGUMENT, "attribute type mismatch");
  }

  const auto fill = [&](const void* src, size_t bytes) -> OrtStatus* {
    *out = bytes;
    if (data == nullptr || len == 0) {
      return nullptr;
    }
    if (len < bytes) {
      return ActiveShim()->ShimmedApi().CreateStatus(ORT_INVALID_ARGUMENT, "buffer too small");
    }
    if (bytes > 0) {
      std::memcpy(data, src, bytes);
    }
    return nullptr;
  };

  switch (type) {
    case ORT_OP_ATTR_INT:
      return fill(&sa->i.value(), sizeof(int64_t));
    case ORT_OP_ATTR_INTS:
      return fill(sa->ints.data(), sa->ints.size() * sizeof(int64_t));
    case ORT_OP_ATTR_FLOAT:
      return fill(&sa->f.value(), sizeof(float));
    case ORT_OP_ATTR_FLOATS:
      return fill(sa->floats.data(), sa->floats.size() * sizeof(float));
    case ORT_OP_ATTR_STRING:
      return fill(sa->s.value().data(), sa->s.value().size());
    case ORT_OP_ATTR_STRINGS: {
      size_t total = 0;
      for (const std::string& str : sa->strings) {
        total += str.size() + 1;
      }
      *out = total;
      if (data == nullptr || len == 0) {
        return nullptr;
      }
      if (len < total) {
        return ActiveShim()->ShimmedApi().CreateStatus(ORT_INVALID_ARGUMENT, "buffer too small");
      }
      char* cursor = static_cast<char*>(data);
      for (const std::string& str : sa->strings) {
        std::memcpy(cursor, str.data(), str.size());
        cursor[str.size()] = '\0';
        cursor += str.size() + 1;
      }
      return nullptr;
    }
    default:
      return ActiveShim()->ShimmedApi().CreateStatus(ORT_NOT_IMPLEMENTED, "unsupported attribute type");
  }
}

}  // namespace

SnapshotShim::SnapshotShim(const OrtApi& real_api, const OrtModelEditorApi& model_editor_api,
                           const SnapshotGraph& snapshot)
    : real_api_(real_api),
      model_editor_api_(model_editor_api),
      snapshot_(snapshot),
      shimmed_api_(real_api) {
  shimmed_api_.Graph_GetNumNodes = &qnn::Graph_GetNumNodes;
  shimmed_api_.Graph_GetNodes = &qnn::Graph_GetNodes;
  shimmed_api_.Graph_GetNumInputs = &qnn::Graph_GetNumInputs;
  shimmed_api_.Graph_GetInputs = &qnn::Graph_GetInputs;
  shimmed_api_.Graph_GetNumOutputs = &qnn::Graph_GetNumOutputs;
  shimmed_api_.Graph_GetOutputs = &qnn::Graph_GetOutputs;
  shimmed_api_.Graph_GetNumInitializers = &qnn::Graph_GetNumInitializers;
  shimmed_api_.Graph_GetInitializers = &qnn::Graph_GetInitializers;
  shimmed_api_.Graph_GetParentNode = &qnn::Graph_GetParentNode;
  shimmed_api_.Graph_GetModelPath = &qnn::Graph_GetModelPath;
  shimmed_api_.Graph_GetName = &qnn::Graph_GetName;
  shimmed_api_.Graph_GetModelMetadata = &qnn::Graph_GetModelMetadata;
  shimmed_api_.Graph_GetGraphView = &qnn::Graph_GetGraphView;
  shimmed_api_.Graph_GetNumOperatorSets = &qnn::Graph_GetNumOperatorSets;
  shimmed_api_.Graph_GetOnnxIRVersion = &qnn::Graph_GetOnnxIRVersion;
  shimmed_api_.Graph_GetOperatorSets = &qnn::Graph_GetOperatorSets;

  shimmed_api_.Node_GetId = &qnn::Node_GetId;
  shimmed_api_.Node_GetName = &qnn::Node_GetName;
  shimmed_api_.Node_GetOperatorType = &qnn::Node_GetOperatorType;
  shimmed_api_.Node_GetDomain = &qnn::Node_GetDomain;
  shimmed_api_.Node_GetSinceVersion = &qnn::Node_GetSinceVersion;
  shimmed_api_.Node_GetNumInputs = &qnn::Node_GetNumInputs;
  shimmed_api_.Node_GetInputs = &qnn::Node_GetInputs;
  shimmed_api_.Node_GetNumOutputs = &qnn::Node_GetNumOutputs;
  shimmed_api_.Node_GetOutputs = &qnn::Node_GetOutputs;
  shimmed_api_.Node_GetNumImplicitInputs = &qnn::Node_GetNumImplicitInputs;
  shimmed_api_.Node_GetImplicitInputs = &qnn::Node_GetImplicitInputs;
  shimmed_api_.Node_GetNumAttributes = &qnn::Node_GetNumAttributes;
  shimmed_api_.Node_GetAttributes = &qnn::Node_GetAttributes;
  shimmed_api_.Node_GetAttributeByName = &qnn::Node_GetAttributeByName;
  shimmed_api_.Node_GetNumSubgraphs = &qnn::Node_GetNumSubgraphs;
  shimmed_api_.Node_GetSubgraphs = &qnn::Node_GetSubgraphs;
  shimmed_api_.Node_GetGraph = &qnn::Node_GetGraph;
  shimmed_api_.Node_GetEpName = &qnn::Node_GetEpName;

  shimmed_api_.GetValueInfoName = &qnn::GetValueInfoName;
  shimmed_api_.GetValueInfoTypeInfo = &qnn::GetValueInfoTypeInfo;
  shimmed_api_.ValueInfo_IsConstantInitializer = &qnn::ValueInfo_IsConstantInitializer;
  shimmed_api_.ValueInfo_IsGraphOutput = &qnn::ValueInfo_IsGraphOutput;
  shimmed_api_.ValueInfo_GetValueProducer = &qnn::ValueInfo_GetValueProducer;
  shimmed_api_.ValueInfo_GetValueNumConsumers = &qnn::ValueInfo_GetValueNumConsumers;
  shimmed_api_.ValueInfo_GetValueConsumers = &qnn::ValueInfo_GetValueConsumers;
  shimmed_api_.ValueInfo_GetInitializerValue = &qnn::ValueInfo_GetInitializerValue;
  shimmed_api_.ValueInfo_GetExternalInitializerInfo = &qnn::ValueInfo_GetExternalInitializerInfo;
  shimmed_api_.ValueInfo_IsFromOuterScope = &qnn::ValueInfo_IsFromOuterScope;
  shimmed_api_.ValueInfo_IsRequiredGraphInput = &qnn::ValueInfo_IsRequiredGraphInput;
  shimmed_api_.ValueInfo_IsOptionalGraphInput = &qnn::ValueInfo_IsOptionalGraphInput;

  shimmed_api_.OpAttr_GetName = &qnn::OpAttr_GetName;
  shimmed_api_.OpAttr_GetType = &qnn::OpAttr_GetType;
  shimmed_api_.ReadOpAttr = &qnn::ReadOpAttr;
  shimmed_api_.OpAttr_GetTensorAttributeAsOrtValue = &qnn::OpAttr_GetTensorAttributeAsOrtValue;

  real_api_.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory_info_);

  // Eagerly synthesize a real OrtTypeInfo per value-info and a real OrtValue per initializer.
  // When value_infos is empty (SnapshotSegmentView), iterate value_info_by_name instead.
  auto synthesize_for_vi = [&](SnapshotValueInfo* vi) {
    if (vi->elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      OrtTensorTypeAndShapeInfo* ts = nullptr;
      if (real_api_.CreateTensorTypeAndShapeInfo(&ts) == nullptr && ts != nullptr) {
        real_api_.SetTensorElementType(ts, vi->elem_type);
        if (vi->has_shape) {
          real_api_.SetDimensions(ts, vi->shape.data(), vi->shape.size());
        }
        OrtTypeInfo* ti = nullptr;
        if (model_editor_api_.CreateTensorTypeInfo(ts, &ti) == nullptr && ti != nullptr) {
          type_infos_.emplace(vi, ti);
        }
        real_api_.ReleaseTensorTypeAndShapeInfo(ts);
      }
    }

    if (vi->is_initializer && !vi->initializer_bytes.empty() && cpu_memory_info_ != nullptr) {
      OrtValue* value = nullptr;
      if (real_api_.CreateTensorWithDataAsOrtValue(
              cpu_memory_info_, vi->initializer_bytes.data(), vi->initializer_bytes.size(),
              vi->shape.data(), vi->shape.size(), vi->elem_type, &value) == nullptr &&
          value != nullptr) {
        initializer_values_.emplace(vi, value);
      }
    }
  };

  if (!snapshot_.value_infos.empty()) {
    for (const std::unique_ptr<SnapshotValueInfo>& vi_owned : snapshot_.value_infos) {
      synthesize_for_vi(vi_owned.get());
    }
  } else {
    for (auto& [_, vi] : snapshot_.value_info_by_name) {
      synthesize_for_vi(vi);
    }
  }
}

SnapshotShim::~SnapshotShim() {
  for (auto& [vi, value] : initializer_values_) {
    real_api_.ReleaseValue(value);
  }
  for (auto& [vi, ti] : type_infos_) {
    real_api_.ReleaseTypeInfo(ti);
  }
  if (cpu_memory_info_ != nullptr) {
    real_api_.ReleaseMemoryInfo(cpu_memory_info_);
  }
}

void SnapshotShim::ReleaseSynthesizedData() {
  for (auto& [vi, value] : initializer_values_) {
    real_api_.ReleaseValue(value);
  }
  initializer_values_.clear();
  for (auto& [vi, ti] : type_infos_) {
    real_api_.ReleaseTypeInfo(ti);
  }
  type_infos_.clear();
  if (cpu_memory_info_ != nullptr) {
    real_api_.ReleaseMemoryInfo(cpu_memory_info_);
    cpu_memory_info_ = nullptr;
  }
}

const OrtGraph* SnapshotShim::GraphHandle() const { return Handle(&snapshot_); }

const OrtNode* SnapshotShim::FusedNodeHandle() const { return Handle(snapshot_.fused_node.get()); }

const OrtTypeInfo* SnapshotShim::TypeInfoFor(const SnapshotValueInfo* vi) const {
  auto it = type_infos_.find(vi);
  return it == type_infos_.end() ? nullptr : it->second;
}

const OrtValue* SnapshotShim::InitializerValueFor(const SnapshotValueInfo* vi) const {
  auto it = initializer_values_.find(vi);
  return it == initializer_values_.end() ? nullptr : it->second;
}

ActiveShimGuard::ActiveShimGuard(const SnapshotShim* shim) : previous_(ActiveShim()) {
  ActiveShim() = shim;
}

ActiveShimGuard::~ActiveShimGuard() { ActiveShim() = previous_; }

}  // namespace qnn
}  // namespace onnxruntime
