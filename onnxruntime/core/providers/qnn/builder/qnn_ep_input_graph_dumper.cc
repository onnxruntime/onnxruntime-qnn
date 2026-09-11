// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#include "core/providers/qnn/builder/qnn_ep_input_graph_dumper.h"

#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// QNN tensor-type integers Netron uses to classify a tensor. Mirrors the
// values emitted by the post-compile QnnJSONGraph path (qnn_utils.cc), which
// serializes the raw Qnn_TensorType_t enum.
constexpr int kQnnTensorTypeAppWrite = QNN_TENSOR_TYPE_APP_WRITE;  // graph input  = 0
constexpr int kQnnTensorTypeAppRead = QNN_TENSOR_TYPE_APP_READ;    // graph output = 1
constexpr int kQnnTensorTypeNative = QNN_TENSOR_TYPE_NATIVE;       // intermediate = 3
constexpr int kQnnTensorTypeStatic = QNN_TENSOR_TYPE_STATIC;       // initializer  = 4

// Assigns a stable, incrementing integer id per tensor name. The public
// ValueInfo API exposes no per-value id, and Netron only needs uniqueness, so
// a name->int allocator suffices.
class TensorIdAllocator {
 public:
  TensorIdAllocator() = default;

  int GetOrAdd(const std::string& name) {
    auto it = ids_.find(name);
    if (it != ids_.end()) {
      return it->second;
    }
    int id = next_id_++;
    ids_.emplace(name, id);
    return id;
  }

 private:
  std::unordered_map<std::string, int> ids_;
  int next_id_ = 0;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TensorIdAllocator);
};

// Classify a value info into the QNN tensor-type integer Netron expects.
// Order matters: an initializer that is also an optional graph input is
// classified as STATIC (its data is what Netron should show).
int ClassifyTensorType(const Ort::ConstValueInfo& value_info) {
  if (value_info.IsConstantInitializer()) {
    return kQnnTensorTypeStatic;
  }
  if (value_info.IsRequiredGraphInput() || value_info.IsOptionalGraphInput()) {
    return kQnnTensorTypeAppWrite;
  }
  if (value_info.IsGraphOutput()) {
    return kQnnTensorTypeAppRead;
  }
  return kQnnTensorTypeNative;
}

// Build the per-tensor JSON entry. `raw_value_info` may be null (an optional,
// not-provided node input); the caller skips those before calling this.
// `type_info_failures` is incremented when the value's type info cannot be
// read; the dumper logs a single aggregated WARNING at the end so silent
// degradation (every tensor reported as FLOAT_32 + empty dims) is observable.
nlohmann::json BuildTensorJson(const Ort::ConstValueInfo& value_info, int id,
                               size_t& type_info_failures) {
  using json = nlohmann::json;
  json tensor_json = json::object();

  tensor_json["id"] = id;
  tensor_json["type"] = ClassifyTensorType(value_info);
  tensor_json["dataFormat"] = 0;
  tensor_json["src_axis_format"] = "NOT_YET_DEFINED";
  tensor_json["axis_format"] = "NOT_YET_DEFINED";

  // Non-quantized placeholder: this dumper records ONNX-level structure, not
  // QDQ scale/zero-point, so the encoding is reported as undefined.
  tensor_json["quant_params"] = {
      {"definition", QNN_DEFINITION_UNDEFINED},
      {"encoding", QNN_QUANTIZATION_ENCODING_UNDEFINED},
      {"scale_offset", {{"scale", 0.0}, {"offset", 0}}},
  };

  // data_type + dims are read from the value's type info, which may be absent.
  // Default to FLOAT_32 + empty dims when unavailable, and treat dynamic dims
  // as "no shape" (empty array) rather than emitting negative sizes.
  int data_type = QNN_DATATYPE_FLOAT_32;
  json dims = json::array();

  const OrtValueInfo* raw = value_info;
  if (raw != nullptr) {
    try {
      Ort::ConstTypeInfo type_info = value_info.TypeInfo();
      Ort::ConstTensorTypeAndShapeInfo shape_info = type_info.GetTensorTypeAndShapeInfo();

      ONNXTensorElementDataType elem_type = shape_info.GetElementType();
      Qnn_DataType_t qnn_dt = QNN_DATATYPE_FLOAT_32;
      if (utils::OnnxDataTypeToQnnDataType(elem_type, qnn_dt, /*is_quantized=*/false)) {
        data_type = qnn_dt;
      } else {
        // The mapping helper does not yet cover this ONNX dtype (e.g. some
        // 4-bit / brain-float variants). Reuse the same accumulator as the
        // type-info exception path so the user still sees a single
        // aggregated WARNING from the dumper instead of a silent FLOAT_32
        // substitution in the JSON.
        ++type_info_failures;
      }

      if (shape_info.HasShape()) {
        std::vector<int64_t> shape = shape_info.GetShape();
        bool has_dynamic = false;
        for (int64_t s : shape) {
          if (s < 0) {
            has_dynamic = true;
            break;
          }
        }
        if (!has_dynamic) {
          for (int64_t s : shape) {
            dims.push_back(s);
          }
        }
      }
    } catch (const Ort::Exception&) {
      ++type_info_failures;
    }
  }

  tensor_json["data_type"] = data_type;
  tensor_json["dims"] = std::move(dims);
  return tensor_json;
}

// Collect the input/output tensor-name array for a node, recording each tensor
// in `tensors_json` (deduped) and skipping null optional inputs.
nlohmann::json CollectNodeTensorNames(const std::vector<Ort::ConstValueInfo>& value_infos,
                                      TensorIdAllocator& id_alloc,
                                      std::unordered_set<std::string>& seen_tensors,
                                      nlohmann::json& tensors_json,
                                      size_t& type_info_failures) {
  nlohmann::json names = nlohmann::json::array();
  for (const Ort::ConstValueInfo& vi : value_infos) {
    const OrtValueInfo* raw = vi;
    if (raw == nullptr) {
      continue;  // optional input not provided
    }
    std::string name = std::string(vi.GetName());
    if (name.empty()) {
      continue;
    }
    names.push_back(name);
    if (!seen_tensors.count(name)) {
      seen_tensors.insert(name);
      tensors_json[name] = BuildTensorJson(vi, id_alloc.GetOrAdd(name), type_info_failures);
    }
  }
  return names;
}

}  // namespace

bool DumpQnnEpInputGraphToJson(const OrtGraph* graph,
                               const std::filesystem::path& output_path,
                               const Ort::Logger& logger) {
  using json = nlohmann::json;

  json root = {
      // Dummy model.cpp / model.bin: not required for Netron visualization.
      {"model.cpp", "N/A"},
      {"model.bin", "N/A"},
      {"converter_command", ""},
      {"copyright_str", "Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries."},
      {"op_types", json::array()},
      {"Total parameters", ""},
      {"Total MACs per inference", ""},
      {"graph", {{"tensors", json::object()}, {"nodes", json::object()}}},
  };

  json& tensors_json = root["graph"]["tensors"];
  json& nodes_json = root["graph"]["nodes"];

  TensorIdAllocator id_alloc;
  std::unordered_set<std::string> seen_tensors;
  // Sorted set so the serialized op_types array order is stable across runs
  // and platforms (downstream diff/hash tools will not see hash-iteration
  // noise).
  std::set<std::string> seen_op_types;
  // Tracks node names already used as JSON object keys so a collision does
  // not silently overwrite an earlier node.
  std::unordered_set<std::string> seen_node_names;
  size_t type_info_failures = 0;

  Ort::ConstGraph ort_graph{graph};

  // Nodes: one entry per ONNX node, keyed by node name (fall back to a
  // synthesized `unnamed_{op_type}_{index}` form when unnamed so the JSON
  // object key stays unique and matches the offline matcher's convention).
  std::vector<Ort::ConstNode> nodes = ort_graph.GetNodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    const Ort::ConstNode& node = nodes[i];

    std::string op_type = std::string(node.GetOperatorType());
    std::string node_name = std::string(node.GetName());
    if (node_name.empty()) {
      // Synthesize a name for an unnamed node.
      node_name = "unnamed_" + op_type + "_" + std::to_string(i);
    }
    // Disambiguate a name that has already been emitted (ONNX does not
    // guarantee unique node names; the synthesized fallback above can also
    // collide with an explicit name). Suffix with `__dup{i}` using the node's
    // position index, matching the offline matcher's convention.
    if (!seen_node_names.insert(node_name).second) {
      node_name = node_name + "__dup" + std::to_string(i);
      seen_node_names.insert(node_name);
    }

    // `package` distinguishes contrib / internal-domain ops (e.g.
    // `com.microsoft`, `com.ms.internal.nhwc` introduced by ORT's layout
    // transformer) from the default ONNX op set. Empty / `ai.onnx` is
    // normalized to "onnx" so the field matches what the post-compile
    // `dump_json_qnn_graph` path emits for native ONNX nodes.
    std::string node_domain = std::string(node.GetDomain());
    json node_json = json::object();
    if (node_domain.empty() || node_domain == "ai.onnx") {
      node_json["package"] = "onnx";
    } else {
      node_json["package"] = std::move(node_domain);
    }
    node_json["type"] = op_type;
    node_json["input_names"] =
        CollectNodeTensorNames(node.GetInputs(), id_alloc, seen_tensors, tensors_json, type_info_failures);
    node_json["output_names"] =
        CollectNodeTensorNames(node.GetOutputs(), id_alloc, seen_tensors, tensors_json, type_info_failures);
    node_json["tensor_params"] = json::object();
    node_json["scalar_params"] = json::object();
    node_json["macs_per_inference"] = "";

    nodes_json[node_name] = std::move(node_json);
    seen_op_types.insert(op_type);
  }

  // Initializers may not be referenced as a node input via GetInputs() in all
  // cases; gather them explicitly so weights still appear in Netron.
  for (const Ort::ConstValueInfo& init : ort_graph.GetInitializers()) {
    const OrtValueInfo* raw = init;
    if (raw == nullptr) {
      continue;
    }
    std::string name = std::string(init.GetName());
    if (name.empty() || seen_tensors.count(name)) {
      continue;
    }
    seen_tensors.insert(name);
    tensors_json[name] = BuildTensorJson(init, id_alloc.GetOrAdd(name), type_info_failures);
  }

  json op_types = json::array();
  for (const auto& t : seen_op_types) {
    op_types.push_back(t);
  }
  root["op_types"] = std::move(op_types);

  if (type_info_failures > 0) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("QNN EP input graph dump: " + std::to_string(type_info_failures) +
                 " tensor(s) had unreadable type info; their data_type/dims fell back to FLOAT_32 / empty.")
                    .c_str());
  }

  // File write: create parent dir, warn-on-overwrite, fail soft. Mirrors
  // WriteTraceToFile in op_tracing/qnn_op_tracing.cc.
  std::error_code ec;
  auto parent = output_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                  ("Failed to create QNN EP input graph dump directory: " + parent.string() +
                   " error: " + ec.message())
                      .c_str());
      return false;
    }
  }

  if (std::filesystem::exists(output_path)) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("Overwriting existing QNN EP input graph dump: " + output_path.string()).c_str());
  }

  std::ofstream ofs(output_path);
  if (!ofs.is_open()) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("Could not open QNN EP input graph dump file: " + output_path.string()).c_str());
    return false;
  }

  ofs << root.dump(2);
  ofs.close();
  if (!ofs) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_WARNING,
                ("QNN EP input graph dump write failed (disk full / I/O error): " + output_path.string()).c_str());
    std::error_code rm_ec;
    std::filesystem::remove(output_path, rm_ec);
    return false;
  }

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_INFO,
              ("QNN EP input graph dumped to: " + output_path.string()).c_str());
  return true;
}

}  // namespace qnn
}  // namespace onnxruntime
