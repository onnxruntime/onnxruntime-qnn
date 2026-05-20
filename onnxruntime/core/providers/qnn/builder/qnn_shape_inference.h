// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// For each output of node_unit whose ORT shape is dynamic (negative or missing dimensions),
// tries to compute a static shape from the (possibly overridden) input shapes and registers
// it in qmw via SetTensorShapeOverride. Called once per node in topological order so that
// shape overrides propagate transitively through the subgraph.
//
// Returns true if at least one output shape override was newly registered.
//
// Note: QNN EP is a plugin shared library and does not link against the ONNX schema
// registry. Shape formulas are implemented directly for the ops that appear downstream
// of NonZero in practice. Unknown op types are silently skipped.
bool TryPropagateShapeOverrides(QnnModelWrapper& qmw, const OrtNodeUnit& node_unit);

}  // namespace qnn
}  // namespace onnxruntime
