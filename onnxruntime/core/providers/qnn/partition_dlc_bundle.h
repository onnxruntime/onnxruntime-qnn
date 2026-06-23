// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// Licensed under the MIT License

#pragma once

#include <string>
#include <vector>

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnModel;

struct PartitionBundleTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
};

struct PartitionBundleRecord {
  std::string name;
  std::vector<PartitionBundleTensor> inputs;
  std::vector<PartitionBundleTensor> outputs;
};

// Snapshot one partition's I/O for the bundle manifest. Called per partition,
// after ComposeGraph() has populated qnn_model's input/output info.
PartitionBundleRecord RecordPartitionBundle(const QnnModel& qnn_model, std::string partition_name);

// Write manifest.json to bundle_dir given all collected partition records.
// Creates bundle_dir if missing. Logs success/failure via logger.
void WritePartitionBundleManifest(const std::string& bundle_dir,
                                  const std::vector<PartitionBundleRecord>& records,
                                  const Ort::Logger& logger);

}  // namespace qnn
}  // namespace onnxruntime
