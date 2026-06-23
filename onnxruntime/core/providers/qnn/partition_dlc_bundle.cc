// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// Licensed under the MIT License

#include "core/providers/qnn/partition_dlc_bundle.h"

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <utility>

#include "nlohmann/json.hpp"

#include "core/providers/qnn/builder/qnn_model.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

std::string SafeDtype(int32_t data_type) {
  try {
    return std::string(qnn::utils::GetElementNameByType(static_cast<ONNXTensorElementDataType>(data_type)));
  } catch (...) {
    return "elem_type_" + std::to_string(data_type);
  }
}

PartitionBundleTensor MakeTensor(const std::string& name,
                                 const std::unordered_map<std::string, OnnxTensorInfo>& info_map) {
  PartitionBundleTensor t;
  t.name = name;
  auto it = info_map.find(name);
  if (it != info_map.end()) {
    t.dtype = SafeDtype(it->second.data_type_);
    t.shape = it->second.shape_;
  }
  return t;
}

nlohmann::json TensorToJson(const PartitionBundleTensor& t) {
  nlohmann::json j;
  j["name"] = t.name;
  j["dtype"] = t.dtype;
  j["shape"] = t.shape;
  return j;
}

}  // namespace

PartitionBundleRecord RecordPartitionBundle(const QnnModel& qnn_model, std::string partition_name) {
  PartitionBundleRecord record;
  record.name = std::move(partition_name);
  const auto& inputs_info = qnn_model.GetInputsInfo();
  for (const auto& input_name : qnn_model.GetInputNames()) {
    record.inputs.push_back(MakeTensor(input_name, inputs_info));
  }
  const auto& outputs_info = qnn_model.GetOutputsInfo();
  for (const auto& output_name : qnn_model.GetOutputNames()) {
    record.outputs.push_back(MakeTensor(output_name, outputs_info));
  }
  return record;
}

void WritePartitionBundleManifest(const std::string& bundle_dir,
                                  const std::vector<PartitionBundleRecord>& records,
                                  const Ort::Logger& logger) {
  namespace fs = std::filesystem;
  fs::path dir(bundle_dir);
  std::error_code ec;
  fs::create_directories(dir, ec);

  nlohmann::json manifest;
  manifest["bundle_version"] = 1;
  manifest["partitions"] = nlohmann::json::array();

  std::unordered_map<std::string, std::string> producer_of;
  for (const auto& rec : records) {
    nlohmann::json p;
    p["name"] = rec.name;
    p["dlc_path"] = (fs::path("partitions") / (rec.name + ".dlc")).generic_string();
    p["inputs"] = nlohmann::json::array();
    for (const auto& in : rec.inputs) {
      p["inputs"].push_back(TensorToJson(in));
    }
    p["outputs"] = nlohmann::json::array();
    for (const auto& out : rec.outputs) {
      p["outputs"].push_back(TensorToJson(out));
    }
    manifest["partitions"].push_back(std::move(p));
    for (const auto& out : rec.outputs) {
      producer_of[out.name] = rec.name;
    }
  }

  manifest["edges"] = nlohmann::json::array();
  for (const auto& rec : records) {
    for (const auto& in : rec.inputs) {
      auto it = producer_of.find(in.name);
      if (it != producer_of.end()) {
        nlohmann::json edge;
        edge["producer_partition"] = it->second;
        edge["consumer_partition"] = rec.name;
        edge["tensor_name"] = in.name;
        manifest["edges"].push_back(std::move(edge));
      }
    }
  }

  fs::path manifest_path = dir / "manifest.json";
  std::ofstream ofs(manifest_path);
  if (ofs) {
    ofs << manifest.dump(2);
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_INFO,
                ("Wrote partition DLC bundle manifest: " + manifest_path.string()).c_str());
  } else {
    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_WARNING,
                ("Failed to write partition DLC bundle manifest: " + manifest_path.string()).c_str());
  }
}

}  // namespace qnn
}  // namespace onnxruntime
