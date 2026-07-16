#pragma once

#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

enum class ModelClass { GenAI, NonGenAI, Unknown };

ModelClass ClassifyModel(const OrtGraph* graph,
                         const OrtApi& ort_api,
                         const Ort::Logger& logger);

}  // namespace qnn
}  // namespace onnxruntime
