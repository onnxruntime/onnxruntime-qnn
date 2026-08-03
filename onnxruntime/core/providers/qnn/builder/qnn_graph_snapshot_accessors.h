// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/providers/qnn/builder/qnn_graph_snapshot.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Hot Migration (Phase 2) - OrtApi shim over a SnapshotGraph.
//
// SnapshotShim builds a copy of a real OrtApi whose graph/node/valueinfo/opattr read accessors are
// repointed at readers over a SnapshotGraph. The compose pipeline reads through this copy without
// knowing the OrtGraph*/OrtNode*/OrtValueInfo*/OrtOpAttr* handles are reinterpret_cast pointers
// into snapshot objects.
//
// Type/shape/tensor-data accessors are left pointing at the real OrtApi: GetValueInfoTypeInfo
// returns a synthesized real OrtTypeInfo and ValueInfo_GetInitializerValue returns a synthesized
// real OrtValue, so downstream calls operate on genuine ORT objects.
class SnapshotShim {
 public:
  // Builds the shimmed OrtApi over `snapshot`.
  SnapshotShim(const OrtApi& real_api, const OrtModelEditorApi& model_editor_api,
               const SnapshotGraph& snapshot);
  ~SnapshotShim();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SnapshotShim);

  // The shimmed copy. Feed this where ComposeGraph expects an OrtApi.
  const OrtApi& ShimmedApi() const { return shimmed_api_; }

  const OrtGraph* GraphHandle() const;

  // Null if the snapshot has no fused node.
  const OrtNode* FusedNodeHandle() const;

  // Release synthesized OrtValue/OrtTypeInfo objects early (same work the destructor does).
  void ReleaseSynthesizedData();

  const OrtTypeInfo* TypeInfoFor(const SnapshotValueInfo* vi) const;
  const OrtValue* InitializerValueFor(const SnapshotValueInfo* vi) const;

 private:
  const OrtApi& real_api_;
  const OrtModelEditorApi& model_editor_api_;
  const SnapshotGraph& snapshot_;
  OrtApi shimmed_api_;

  OrtMemoryInfo* cpu_memory_info_ = nullptr;
  std::unordered_map<const SnapshotValueInfo*, OrtTypeInfo*> type_infos_;
  std::unordered_map<const SnapshotValueInfo*, OrtValue*> initializer_values_;
};

// RAII scope that publishes `shim` as the thread-local active shim for accessor routing.
class ActiveShimGuard {
 public:
  explicit ActiveShimGuard(const SnapshotShim* shim);
  ~ActiveShimGuard();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ActiveShimGuard);

 private:
  const SnapshotShim* previous_;
};

}  // namespace qnn
}  // namespace onnxruntime
