#include "core/providers/qnn/builder/qnn_model_classifier.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace onnxruntime {
namespace qnn {

namespace {

constexpr float kMinDecisionScore = 10.0f;

// -- GenAI signal weights --
constexpr float kMatMulNBitsScore    = 10.0f;  // Int4-quantized LLM weights (triggers early return, not actually accumulated)
constexpr float kHeavyAttnOpScore    = 10.0f;  // MHA / GQA / Attention contrib ops
constexpr float kLightAttnOpScore    =  3.0f;  // RotaryEmbedding / SkipLayerNorm / GELU variants
constexpr float kKvCacheInputScore   =  5.0f;  // Per KV-cache input name match
constexpr float kDecomposedAttnScore =  5.0f;  // MatMul + Softmax co-occurrence
constexpr float kLayerNormDenseScore = 10.0f;  // LayerNorm density > 5%

// -- NonGenAI signal weights --
constexpr float kConvDenseScore      = 10.0f;  // Conv density > 30%
constexpr float kBatchNormDenseScore =  5.0f;  // BatchNorm density > 10%
constexpr float kClassifHeadScore    =  5.0f;  // GlobalAveragePool + (Softmax | Gemm)
constexpr float k4DImageScore        =  5.0f;  // 4D spatial input (H, W >= 224)

// -- Density thresholds --
constexpr float kConvDensityThreshold      = 0.30f;
constexpr float kBatchNormDensityThreshold = 0.10f;
constexpr float kLayerNormDensityThreshold = 0.05f;

// Minimum value of a spatial dimension to count as an image-shaped input
constexpr int64_t kMinSpatialDim = 224;

// -- Op tables --
constexpr std::string_view kMsDomain = "com.microsoft";
constexpr std::string_view kHeavyAttnOps[] = {
  "MultiHeadAttention",
  "GroupQueryAttention",
  "Attention"
};
constexpr std::string_view kLightAttnOps[] = {
  "RotaryEmbedding",
  "SkipLayerNormalization",
  "FastGelu",
  "BiasGelu"
};
// Standard HuggingFace LLM export naming for KV-cache inputs
constexpr std::string_view kKvCacheHints[] = {
  "past_key_values",
  "past_",
  "present",
  "input_ids",
  "attention_mask",
  "position_ids"
};

template <size_t N>
constexpr bool InArray(const std::string_view (&arr)[N], std::string_view v) {
  return std::find(std::begin(arr), std::end(arr), v) != std::end(arr);
}

bool MatchesKvCacheHint(std::string_view name) {
  for (std::string_view hint : kKvCacheHints)
    if (name.find(hint) != std::string_view::npos) return true;
  return false;
}

// Used for optional signals so that a failing call skips the signal rather than aborting classification
inline bool OkOrSkip(OrtStatus* s, const OrtApi& ort_api) {
  if (!s) return true;
  ort_api.ReleaseStatus(s);
  return false;
}

}  // namespace

ModelClass ClassifyModel(const OrtGraph* graph,
                         const OrtApi& ort_api,
                         const Ort::Logger& logger) {
  size_t num_nodes = 0;
  if (!OkOrSkip(ort_api.Graph_GetNumNodes(graph, &num_nodes), ort_api) ||
      num_nodes == 0)
    return ModelClass::Unknown;

  std::vector<const OrtNode*> nodes(num_nodes);
  if (!OkOrSkip(ort_api.Graph_GetNodes(graph, nodes.data(), num_nodes), ort_api))
    return ModelClass::Unknown;

  float genai_score     = 0.0f;
  float non_genai_score = 0.0f;

  // Pre-scan graph inputs before the node loop so KV-cache evidence can trigger early classification
  size_t num_inputs = 0;
  if (OkOrSkip(ort_api.Graph_GetNumInputs(graph, &num_inputs), ort_api) && num_inputs > 0) {
    std::vector<const OrtValueInfo*> inputs(num_inputs);
    if (OkOrSkip(ort_api.Graph_GetInputs(graph, inputs.data(), num_inputs), ort_api)) {

      // KV-cache / LLM input names
      for (const OrtValueInfo* vi : inputs) {
        const char* name = nullptr;
        if (!OkOrSkip(ort_api.GetValueInfoName(vi, &name), ort_api) || !name) continue;
        if (MatchesKvCacheHint(name)) genai_score += kKvCacheInputScore;
      }

      // 4D image-shaped first input
      const OrtTypeInfo* type_info = nullptr;
      const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
      if (OkOrSkip(ort_api.GetValueInfoTypeInfo(inputs[0], &type_info), ort_api) &&
          type_info &&
          OkOrSkip(ort_api.CastTypeInfoToTensorInfo(type_info, &shape_info), ort_api) &&
          shape_info) {
        size_t rank = 0;
        if (OkOrSkip(ort_api.GetDimensionsCount(shape_info, &rank), ort_api) && rank == 4) {
          int64_t dims[4] = {};
          if (OkOrSkip(ort_api.GetDimensions(shape_info, dims, 4), ort_api) &&
              dims[2] >= kMinSpatialDim && dims[3] >= kMinSpatialDim) {
            non_genai_score += k4DImageScore;
          }
        }
      }
    }
  }

  if (genai_score >= kMinDecisionScore && genai_score > non_genai_score) {
    ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                ("QNN ModelClassifier: GenAI (early return, kv-cache score: " + std::to_string(genai_score) + ")").c_str());
    return ModelClass::GenAI;
  }

  size_t conv_count       = 0;
  size_t layer_norm_count = 0;
  size_t batch_norm_count = 0;
  bool   has_matmul       = false;
  bool   has_softmax      = false;
  bool   has_global_avg   = false;
  bool   has_gemm         = false;

  for (const OrtNode* node : nodes) {
    const char* op_cstr = nullptr;
    if (!OkOrSkip(ort_api.Node_GetOperatorType(node, &op_cstr), ort_api) || !op_cstr)
      continue;
    const std::string_view op{op_cstr};

    const char* domain_cstr = nullptr;
    OkOrSkip(ort_api.Node_GetDomain(node, &domain_cstr), ort_api);
    const bool is_ms = domain_cstr && std::string_view{domain_cstr} == kMsDomain;

    if (is_ms) {
      if (InArray(kHeavyAttnOps, op) || op == "MatMulNBits") {
        ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE,
                    ("QNN ModelClassifier: GenAI (early return, decisive op: " + std::string(op) + ")").c_str());
        return ModelClass::GenAI;
      }

      if (InArray(kLightAttnOps, op))
        genai_score += kLightAttnOpScore;
      // Also count LayerNorm exported under the ms domain
      else if (op == "LayerNormalization" ||
               op == "SimplifiedLayerNormalization")
        ++layer_norm_count;
    } else {
      if      (op == "Conv" || op == "ConvTranspose" || op == "ConvInteger")
        ++conv_count;
      else if (op == "LayerNormalization" ||
               op == "SimplifiedLayerNormalization")
        ++layer_norm_count;
      else if (op == "BatchNormalization")
        ++batch_norm_count;
      else if (op == "GlobalAveragePool")
        has_global_avg = true;
      else if (op == "Softmax" || op == "LogSoftmax")
        has_softmax = true;
      else if (op == "Gemm")
        has_gemm   = true;
      else if (op == "MatMul")
        has_matmul = true;
    }
  }

  const float nf = static_cast<float>(num_nodes);

  // Density-based and co-occurrence signals
  if (conv_count       / nf > kConvDensityThreshold)      non_genai_score += kConvDenseScore;
  if (batch_norm_count / nf > kBatchNormDensityThreshold) non_genai_score += kBatchNormDenseScore;
  if (layer_norm_count / nf > kLayerNormDensityThreshold) genai_score     += kLayerNormDenseScore;

  // Decomposed attention: MatMul -> (Mask+Softmax) -> MatMul (appears in models exported without contrib ops)
  if (has_matmul && has_softmax)
    genai_score += kDecomposedAttnScore;

  // ImageNet-style classification tail: pool -> flatten -> linear -> softmax.
  if (has_global_avg && (has_softmax || has_gemm))
    non_genai_score += kClassifHeadScore;

  ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("QNN ModelClassifier:"
                                                  " genai_score=" + std::to_string(genai_score) +
                                                  " non_genai_score=" + std::to_string(non_genai_score)).c_str());

  if (genai_score >= kMinDecisionScore && genai_score > non_genai_score)
    return ModelClass::GenAI;
  if (non_genai_score >= kMinDecisionScore && non_genai_score > genai_score)
    return ModelClass::NonGenAI;
  return ModelClass::Unknown;
}

}  // namespace qnn
}  // namespace onnxruntime
