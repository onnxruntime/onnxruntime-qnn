// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/onnx_subgraph_dumper.h"

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/providers/qnn/common/onnx_protobuf.h"

namespace onnxruntime {
namespace qnn {

namespace {

size_t ElementSizeBytes(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return 8;
    default:
      return 0;
  }
}

Ort::Status FillTensorTypeAndShape(const OrtApi& ort_api,
                                   const OrtTensorTypeAndShapeInfo& info,
                                   ONNX_NAMESPACE::TypeProto_Tensor* out_tensor_type) {
  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(&info, &elem_type));
  out_tensor_type->set_elem_type(static_cast<int>(elem_type));

  size_t num_dims = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensionsCount(&info, &num_dims));

  auto* shape_proto = out_tensor_type->mutable_shape();
  if (num_dims == 0) {
    return Ort::Status{nullptr};
  }

  std::vector<int64_t> dims(num_dims, 0);
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensions(&info, dims.data(), num_dims));

  std::vector<const char*> sym_dims(num_dims, nullptr);
  bool have_symbolic = (ort_api.GetSymbolicDimensions(&info, sym_dims.data(), num_dims) == nullptr);

  for (size_t i = 0; i < num_dims; ++i) {
    auto* dim_proto = shape_proto->add_dim();
    if (dims[i] >= 0) {
      dim_proto->set_dim_value(dims[i]);
    } else if (have_symbolic && sym_dims[i] != nullptr && sym_dims[i][0] != '\0') {
      dim_proto->set_dim_param(sym_dims[i]);
    }
  }
  return Ort::Status{nullptr};
}

Ort::Status ConvertValueInfoToProto(const OrtApi& ort_api,
                                    const OrtValueInfo* vi,
                                    ONNX_NAMESPACE::ValueInfoProto* out_proto) {
  const char* name = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoName(vi, &name));
  out_proto->set_name(name ? name : "");

  const OrtTypeInfo* type_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoTypeInfo(vi, &type_info));

  const OrtTensorTypeAndShapeInfo* tt_info = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.CastTypeInfoToTensorInfo(type_info, &tt_info));
  if (tt_info == nullptr) {
    return MAKE_EP_FAIL(("ValueInfo " + std::string(name ? name : "?") +
                         " is not a tensor type; non-tensor I/O is unsupported for ONNX subgraph dump.")
                            .c_str());
  }

  auto* tensor_type = out_proto->mutable_type()->mutable_tensor_type();
  RETURN_IF_ERROR(FillTensorTypeAndShape(ort_api, *tt_info, tensor_type));
  return Ort::Status{nullptr};
}

// RAII helper: ensures that if DumpPartitionAsOnnxModel returns a non-OK status for ANY
// reason after the output paths are known, both the `.onnx` and the `.onnx.data` are
// removed from disk before we propagate the error. Disarmed via commit() once both files
// have been fully written and flushed successfully. Without this guard, a mid-walk failure
// (or a serialize failure after the sidecar was already populated) could leave behind a
// partial/orphan pair on disk that round-trip tooling would then misread.
struct OutputFilesGuard {
  std::filesystem::path onnx_path;
  std::filesystem::path data_path;
  bool committed = false;

  ~OutputFilesGuard() {
    if (committed) {
      return;
    }
    std::error_code ec;
    std::filesystem::remove(onnx_path, ec);
    std::filesystem::remove(data_path, ec);
  }

  void commit() noexcept { committed = true; }
};

// Sink for ONNX external_data writes. Each partition gets one of these so all of its
// initializer bytes are streamed sequentially into a single sidecar file. Avoids protobuf's
// 2 GB single-message ceiling that fires when inlining gpt-oss-sized weight tensors as raw_data.
struct ExternalDataSink {
  std::ofstream ofs;
  std::string relative_filename;  // basename only — stored in TensorProto.external_data["location"]
  uint64_t offset = 0;
};

Ort::Status ConvertInitializerToTensorProto(const OrtApi& ort_api,
                                            const OrtValueInfo* vi,
                                            ONNX_NAMESPACE::TensorProto* out_proto,
                                            ExternalDataSink& sink) {
  const char* name = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoName(vi, &name));
  out_proto->set_name(name ? name : "");

  const OrtValue* tensor_value = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.ValueInfo_GetInitializerValue(vi, &tensor_value));
  if (tensor_value == nullptr) {
    return MAKE_EP_FAIL(("Initializer " + std::string(name ? name : "?") +
                         " has no underlying OrtValue.")
                            .c_str());
  }

  OrtTensorTypeAndShapeInfo* tt_info_raw = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorTypeAndShape(tensor_value, &tt_info_raw));
  std::unique_ptr<OrtTensorTypeAndShapeInfo, FuncDeleter<OrtTensorTypeAndShapeInfo>> tt_guard(
      tt_info_raw, FuncDeleter<OrtTensorTypeAndShapeInfo>{ort_api.ReleaseTensorTypeAndShapeInfo});

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(tt_info_raw, &elem_type));
  out_proto->set_data_type(static_cast<int>(elem_type));

  size_t num_dims = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensionsCount(tt_info_raw, &num_dims));
  std::vector<int64_t> dims(num_dims, 0);
  if (num_dims > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensions(tt_info_raw, dims.data(), num_dims));
  }
  for (auto d : dims) {
    out_proto->add_dims(d);
  }

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    return MAKE_EP_FAIL(("String initializer " + std::string(name ? name : "?") +
                         " is unsupported for ONNX subgraph dump.")
                            .c_str());
  }

  size_t element_size = ElementSizeBytes(elem_type);
  if (element_size == 0) {
    return MAKE_EP_FAIL(("Unsupported initializer element type " + std::to_string(static_cast<int>(elem_type)) +
                         " for " + std::string(name ? name : "?") + ".")
                            .c_str());
  }

  size_t element_count = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorShapeElementCount(tt_info_raw, &element_count));
  size_t bytes = element_count * element_size;
  if (bytes == 0) {
    return Ort::Status{nullptr};  // empty tensor: shape + dtype suffice, no data needed
  }

  void* data_ptr = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorMutableData(const_cast<OrtValue*>(tensor_value), &data_ptr));

  // Stream the bytes into the per-partition sidecar file and reference them via external_data.
  // This sidesteps the 2 GB protobuf serialization limit on the model proto and avoids a second
  // copy of the weight bytes that the inline raw_data path would otherwise force.
  const uint64_t this_offset = sink.offset;
  sink.ofs.write(static_cast<const char*>(data_ptr), static_cast<std::streamsize>(bytes));
  if (!sink.ofs.good()) {
    return MAKE_EP_FAIL(("Failed to write " + std::to_string(bytes) +
                         " bytes for initializer '" + std::string(name ? name : "?") +
                         "' to external-data sidecar.")
                            .c_str());
  }
  sink.offset += bytes;

  out_proto->set_data_location(ONNX_NAMESPACE::TensorProto::EXTERNAL);
  auto* loc = out_proto->add_external_data();
  loc->set_key("location");
  loc->set_value(sink.relative_filename);
  auto* off = out_proto->add_external_data();
  off->set_key("offset");
  off->set_value(std::to_string(this_offset));
  auto* len = out_proto->add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(bytes));

  return Ort::Status{nullptr};
}

Ort::Status ConvertAttributeToProto(const OrtApi& ort_api,
                                    const OrtOpAttr* attr,
                                    ONNX_NAMESPACE::AttributeProto* out_proto) {
  const char* attr_name = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.OpAttr_GetName(attr, &attr_name));
  std::string name = attr_name ? attr_name : "";
  out_proto->set_name(name);

  OrtOpAttrType attr_type = ORT_OP_ATTR_UNDEFINED;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.OpAttr_GetType(attr, &attr_type));

  using AP = ONNX_NAMESPACE::AttributeProto;

  switch (attr_type) {
    case ORT_OP_ATTR_INT: {
      out_proto->set_type(AP::INT);
      int64_t val = 0;
      size_t out_size = 0;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.ReadOpAttr(attr, attr_type, &val, sizeof(val), &out_size));
      out_proto->set_i(val);
      break;
    }
    case ORT_OP_ATTR_INTS: {
      out_proto->set_type(AP::INTS);
      size_t needed = 0;
      OrtStatus* probe = ort_api.ReadOpAttr(attr, attr_type, nullptr, 0, &needed);
      if (probe != nullptr) {
        ort_api.ReleaseStatus(probe);
      }
      if (needed > 0) {
        std::vector<int64_t> vals(needed / sizeof(int64_t));
        ORT_CXX_RETURN_ON_API_FAIL(ort_api.ReadOpAttr(attr, attr_type, vals.data(), needed, &needed));
        for (auto v : vals) out_proto->add_ints(v);
      }
      break;
    }
    case ORT_OP_ATTR_FLOAT: {
      out_proto->set_type(AP::FLOAT);
      float val = 0.0f;
      size_t out_size = 0;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.ReadOpAttr(attr, attr_type, &val, sizeof(val), &out_size));
      out_proto->set_f(val);
      break;
    }
    case ORT_OP_ATTR_FLOATS: {
      out_proto->set_type(AP::FLOATS);
      size_t needed = 0;
      OrtStatus* probe = ort_api.ReadOpAttr(attr, attr_type, nullptr, 0, &needed);
      if (probe != nullptr) {
        ort_api.ReleaseStatus(probe);
      }
      if (needed > 0) {
        std::vector<float> vals(needed / sizeof(float));
        ORT_CXX_RETURN_ON_API_FAIL(ort_api.ReadOpAttr(attr, attr_type, vals.data(), needed, &needed));
        for (auto v : vals) out_proto->add_floats(v);
      }
      break;
    }
    case ORT_OP_ATTR_STRING: {
      out_proto->set_type(AP::STRING);
      size_t needed = 0;
      OrtStatus* probe = ort_api.ReadOpAttr(attr, attr_type, nullptr, 0, &needed);
      if (probe != nullptr) {
        ort_api.ReleaseStatus(probe);
      }
      std::string str(needed, '\0');
      if (needed > 0) {
        ORT_CXX_RETURN_ON_API_FAIL(ort_api.ReadOpAttr(attr, attr_type, str.data(), needed, &needed));
      }
      out_proto->set_s(str);
      break;
    }
    case ORT_OP_ATTR_STRINGS: {
      out_proto->set_type(AP::STRINGS);
      size_t needed = 0;
      OrtStatus* probe = ort_api.ReadOpAttr(attr, attr_type, nullptr, 0, &needed);
      if (probe != nullptr) {
        ort_api.ReleaseStatus(probe);
      }
      if (needed > 0) {
        std::vector<char> buf(needed);
        ORT_CXX_RETURN_ON_API_FAIL(ort_api.ReadOpAttr(attr, attr_type, buf.data(), needed, &needed));
        const char* p = buf.data();
        const char* end = buf.data() + needed;
        while (p < end) {
          size_t len = std::strlen(p);
          out_proto->add_strings(std::string(p, len));
          p += len + 1;
          if (p > end) break;
        }
      }
      break;
    }
    case ORT_OP_ATTR_TENSOR: {
      out_proto->set_type(AP::TENSOR);
      OrtValue* tensor_value_raw = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.OpAttr_GetTensorAttributeAsOrtValue(attr, &tensor_value_raw));
      if (tensor_value_raw == nullptr) {
        return MAKE_EP_FAIL(("TENSOR attribute '" + name + "' produced no OrtValue.").c_str());
      }
      std::unique_ptr<OrtValue, FuncDeleter<OrtValue>> tensor_guard(
          tensor_value_raw, FuncDeleter<OrtValue>{ort_api.ReleaseValue});

      auto* tp = out_proto->mutable_t();
      tp->set_name(name);

      OrtTensorTypeAndShapeInfo* tt_raw = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorTypeAndShape(tensor_value_raw, &tt_raw));
      std::unique_ptr<OrtTensorTypeAndShapeInfo, FuncDeleter<OrtTensorTypeAndShapeInfo>> tt_guard(
          tt_raw, FuncDeleter<OrtTensorTypeAndShapeInfo>{ort_api.ReleaseTensorTypeAndShapeInfo});

      ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorElementType(tt_raw, &et));
      tp->set_data_type(static_cast<int>(et));

      size_t ndim = 0;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensionsCount(tt_raw, &ndim));
      std::vector<int64_t> dims(ndim, 0);
      if (ndim > 0) {
        ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetDimensions(tt_raw, dims.data(), ndim));
      }
      for (auto d : dims) tp->add_dims(d);

      size_t element_size = ElementSizeBytes(et);
      if (element_size == 0 && et != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        return MAKE_EP_FAIL(("TENSOR attribute '" + name +
                             "' has unsupported element type " + std::to_string(static_cast<int>(et)) + ".")
                                .c_str());
      }
      size_t count = 0;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorShapeElementCount(tt_raw, &count));
      size_t bytes = count * element_size;
      if (bytes > 0) {
        void* data = nullptr;
        ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetTensorMutableData(tensor_value_raw, &data));
        tp->set_raw_data(data, bytes);
      }
      break;
    }
    case ORT_OP_ATTR_GRAPH: {
      return MAKE_EP_FAIL(("Subgraph (GRAPH) attribute '" + name +
                           "' is not supported for ONNX subgraph dump (round-trip would be incomplete).")
                              .c_str());
    }
    case ORT_OP_ATTR_UNDEFINED:
    default: {
      return MAKE_EP_FAIL(("Unknown attribute type " + std::to_string(static_cast<int>(attr_type)) +
                           " on '" + name + "'.")
                              .c_str());
    }
  }
  return Ort::Status{nullptr};
}

Ort::Status ConvertNodeToProto(const OrtApi& ort_api,
                               const OrtNode* node,
                               ONNX_NAMESPACE::NodeProto* out_proto) {
  const char* name = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetName(node, &name));
  out_proto->set_name(name ? name : "");

  const char* op_type = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetOperatorType(node, &op_type));
  out_proto->set_op_type(op_type ? op_type : "");

  const char* domain = nullptr;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetDomain(node, &domain));
  if (domain && domain[0] != '\0') {
    out_proto->set_domain(domain);
  }

  size_t num_inputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumInputs(node, &num_inputs));
  std::vector<const OrtValueInfo*> inputs(num_inputs, nullptr);
  if (num_inputs > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetInputs(node, inputs.data(), num_inputs));
  }
  for (const OrtValueInfo* vi : inputs) {
    if (vi == nullptr) {
      out_proto->add_input("");
    } else {
      const char* in_name = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoName(vi, &in_name));
      out_proto->add_input(in_name ? in_name : "");
    }
  }

  size_t num_outputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumOutputs(node, &num_outputs));
  std::vector<const OrtValueInfo*> outputs(num_outputs, nullptr);
  if (num_outputs > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetOutputs(node, outputs.data(), num_outputs));
  }
  for (const OrtValueInfo* vi : outputs) {
    if (vi == nullptr) {
      out_proto->add_output("");
    } else {
      const char* out_name = nullptr;
      ORT_CXX_RETURN_ON_API_FAIL(ort_api.GetValueInfoName(vi, &out_name));
      out_proto->add_output(out_name ? out_name : "");
    }
  }

  size_t num_attrs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumAttributes(node, &num_attrs));
  if (num_attrs > 0) {
    std::vector<const OrtOpAttr*> attrs(num_attrs, nullptr);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetAttributes(node, attrs.data(), num_attrs));
    for (const OrtOpAttr* attr : attrs) {
      auto* ap = out_proto->add_attribute();
      RETURN_IF_ERROR(ConvertAttributeToProto(ort_api, attr, ap));
    }
  }

  return Ort::Status{nullptr};
}

}  // namespace

Ort::Status DumpPartitionAsOnnxModel(const OrtApi& ort_api,
                                     const OrtGraph& subgraph,
                                     const std::string& graph_name,
                                     const std::filesystem::path& output_path,
                                     const Ort::Logger& logger) {
  // ORT has already synthesized a per-partition OrtGraph for CompileImpl with the boundary
  // inputs/outputs computed. We can walk it directly using the standard Graph_* APIs.
  const OrtGraph* sg = &subgraph;

  // Compute output paths and ensure the parent directory exists *before* opening either
  // file or arming the cleanup guard, so a missing-dir failure can't leave half-written files.
  std::filesystem::path sidecar_path = output_path;
  sidecar_path += ".data";
  {
    std::error_code ec;
    std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent, ec)) {
      std::filesystem::create_directories(parent, ec);
      if (ec) {
        return MAKE_EP_FAIL(("Failed to create output directory '" + parent.string() +
                             "': " + ec.message())
                                .c_str());
      }
    }
  }

  // RAII: any non-OK return from this point on removes both files. Disarmed via commit()
  // at the very end after both writes have succeeded. Declared BEFORE `sink` and the .onnx
  // ofstream so its destructor runs LAST (after both file handles have been closed).
  OutputFilesGuard cleanup_guard{output_path, sidecar_path};

  ONNX_NAMESPACE::ModelProto model_proto;

  int64_t ir_version = 0;
  if (ort_api.Graph_GetOnnxIRVersion(sg, &ir_version) == nullptr && ir_version >= 4) {
    model_proto.set_ir_version(ir_version);
  } else {
    // Force IR version >= 4 so that initializers do not have to also appear in graph.input.
    // The dumped graph is intended for round-trip with QNN EP, so a modern IR is desired.
    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  }

  model_proto.set_producer_name("onnxruntime-qnn-ep");
  model_proto.set_producer_version("dump_onnx_subgraph");
  model_proto.set_doc_string("ONNX subgraph dumped from QNN EP partition '" + graph_name + "'");

  size_t num_opsets = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNumOperatorSets(sg, &num_opsets));
  if (num_opsets > 0) {
    std::vector<const char*> domains(num_opsets, nullptr);
    std::vector<int64_t> versions(num_opsets, 0);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetOperatorSets(sg, domains.data(), versions.data(), num_opsets));
    for (size_t i = 0; i < num_opsets; ++i) {
      auto* opset = model_proto.add_opset_import();
      opset->set_domain(domains[i] ? domains[i] : "");
      opset->set_version(versions[i]);
    }
  }

  auto* graph_proto = model_proto.mutable_graph();
  graph_proto->set_name(graph_name);

  // Open the per-partition external-data sidecar. Filename is `<onnx_filename>.data` so the
  // two files always live next to each other; ONNX loaders resolve the relative `location`
  // ref against the .onnx file's directory automatically.
  ExternalDataSink sink;
  sink.relative_filename = sidecar_path.filename().string();
  sink.ofs.open(sidecar_path, std::ios::binary | std::ios::trunc);
  if (!sink.ofs) {
    return MAKE_EP_FAIL(("Failed to open external-data sidecar '" + sidecar_path.string() +
                         "' for writing.")
                            .c_str());
  }

  // Initializers (must come before inputs walk because some inputs are also initializers
  // in IR<4 - we still emit them as initializers only).
  size_t num_initializers = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNumInitializers(sg, &num_initializers));
  std::vector<const OrtValueInfo*> initializers(num_initializers, nullptr);
  if (num_initializers > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetInitializers(sg, initializers.data(), num_initializers));
  }
  for (const OrtValueInfo* vi : initializers) {
    auto* tp = graph_proto->add_initializer();
    RETURN_IF_ERROR(ConvertInitializerToTensorProto(ort_api, vi, tp, sink));
  }

  // Inputs (skip constant initializers; non-constant initializers may also appear in inputs
  // and are kept as both an input and an initializer for IR>=4 compatibility).
  size_t num_inputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNumInputs(sg, &num_inputs));
  std::vector<const OrtValueInfo*> inputs(num_inputs, nullptr);
  if (num_inputs > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetInputs(sg, inputs.data(), num_inputs));
  }
  for (const OrtValueInfo* vi : inputs) {
    bool is_const_init = false;
    if (ort_api.ValueInfo_IsConstantInitializer(vi, &is_const_init) == nullptr && is_const_init) {
      continue;
    }
    auto* vip = graph_proto->add_input();
    RETURN_IF_ERROR(ConvertValueInfoToProto(ort_api, vi, vip));
  }

  // Outputs.
  size_t num_outputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNumOutputs(sg, &num_outputs));
  std::vector<const OrtValueInfo*> outputs(num_outputs, nullptr);
  if (num_outputs > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetOutputs(sg, outputs.data(), num_outputs));
  }
  for (const OrtValueInfo* vi : outputs) {
    auto* vip = graph_proto->add_output();
    RETURN_IF_ERROR(ConvertValueInfoToProto(ort_api, vi, vip));
  }

  // Nodes.
  size_t num_nodes = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNumNodes(sg, &num_nodes));
  std::vector<const OrtNode*> nodes(num_nodes, nullptr);
  if (num_nodes > 0) {
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.Graph_GetNodes(sg, nodes.data(), num_nodes));
  }
  for (const OrtNode* node : nodes) {
    auto* np = graph_proto->add_node();
    RETURN_IF_ERROR(ConvertNodeToProto(ort_api, node, np));
  }

  // Serialize.
  // Close the sidecar before serializing the model proto. If no bytes were written (partition
  // had no non-empty initializers), drop the empty file to avoid leaving 0-byte artifacts.
  sink.ofs.flush();
  const bool sidecar_has_data = sink.offset > 0 && sink.ofs.good();
  sink.ofs.close();
  if (!sidecar_has_data) {
    std::error_code rm_ec;
    std::filesystem::remove(sidecar_path, rm_ec);
  }

  std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
  if (!ofs) {
    return MAKE_EP_FAIL(("Failed to open '" + output_path.string() + "' for writing.").c_str());
  }
  if (!model_proto.SerializeToOstream(&ofs)) {
    return MAKE_EP_FAIL(("Failed to serialize ModelProto to '" + output_path.string() + "'.").c_str());
  }
  ofs.flush();
  if (!ofs.good()) {
    return MAKE_EP_FAIL(("Output stream not in good state after writing '" + output_path.string() + "'.").c_str());
  }

  // Both files are fully written and flushed — disarm the cleanup guard.
  cleanup_guard.commit();

  if (!IsNullLogger(logger)) {
    std::string msg = "Wrote ONNX subgraph dump: " + output_path.string();
    if (sidecar_has_data) {
      msg += " (+ " + sidecar_path.filename().string() + ", " +
             std::to_string(sink.offset) + " bytes external)";
    }
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO, msg.c_str());
  }
  return Ort::Status{nullptr};
}

}  // namespace qnn
}  // namespace onnxruntime
