// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// All QNN EP GridSample tests, grouped by domain:
//   - "" (ai.onnx): string attributes, ONNX layout converted by ORT's layout transformer.
//   - "ai.onnx.contrib": INT attributes (mode 0/1, padding_mode 0/1/2, align_corners), NC(D)HW input, channel-last
//     grid. The builder inserts the layout transposes itself.
//   - "com.microsoft": string attributes, 4D only. The builder inserts the layout transposes itself.
// The "ai.onnx.contrib" op has no ORT CPU kernel, so a test-only reference kernel is registered through the lite
// custom-op API. It is used as the float CPU EP reference and as the fallback kernel for nodes the QNN EP rejects.
// The tests do not depend on BUILD_QNN_UDO_TEST (that flag only gates the Linux x86_64 UDO op-package tests).

namespace {

// Runs a non-QDQ model on the given QNN backend ("htp" or "cpu") and compares the output to the CPU EP.
template <typename InputType = float>
void RunOpTestOnBackend(const char* backend_type,
                        const std::string& op_type,
                        const std::vector<TestInputDef<InputType>>& input_defs,
                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                        int opset_version,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        const std::string& op_domain = kOnnxDomain,
                        float fp32_abs_err = 1e-5f,
                        bool enable_htp_fp16_precision = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_type;
  if (enable_htp_fp16_precision) {
#if defined(_WIN32)
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
#endif
#if defined(__linux__) && !defined(__aarch64__)
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  RunQnnModelTest(BuildOpTestCase<InputType>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain),
                  provider_options,
                  opset_version,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(fp32_abs_err)});
}

// Tests the accuracy of a QDQ model on the given QNN backend by comparing to the CPU EP, which runs both the fp32
// model and the QDQ model. The QNN CPU backend supports only 8-bit quantization.
template <typename InputQType = uint8_t>
void RunQDQOpTestOnBackend(const char* backend_type,
                           const std::string& op_type,
                           const std::vector<TestInputDef<float>>& input_defs,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           const std::string& op_domain = kOnnxDomain,
                           bool use_contrib_qdq = false,
                           QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_type;
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain),
                       BuildQDQOpTestCase<InputQType>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain,
                                                      use_contrib_qdq),
                       provider_options,
                       opset_version,
                       expected_ep_assignment,
                       tolerance);
}

template <typename InputType = float>
void RunOpTestOnCPU(const std::string& op_type,
                    const std::vector<TestInputDef<InputType>>& input_defs,
                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                    int opset_version,
                    ExpectedEPNodeAssignment expected_ep_assignment) {
  RunOpTestOnBackend<InputType>("cpu", op_type, input_defs, attrs, opset_version, expected_ep_assignment);
}

template <typename InputQType = uint8_t>
void RunQDQOpTestOnCPU(const std::string& op_type,
                       const std::vector<TestInputDef<float>>& input_defs,
                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                       int opset_version,
                       ExpectedEPNodeAssignment expected_ep_assignment,
                       const std::string& op_domain = kOnnxDomain,
                       bool use_contrib_qdq = false) {
  RunQDQOpTestOnBackend<InputQType>("cpu", op_type, input_defs, attrs, opset_version, expected_ep_assignment,
                                    op_domain, use_contrib_qdq);
}

}  // namespace

// -------------------------------------------------------------------------------------------------------------
// QNN CPU backend.
// -------------------------------------------------------------------------------------------------------------

// GridSample float, bilinear + zeros padding on the QNN CPU backend (rank 4).
TEST_F(QnnCPUBackendTests, GridSample_Float_Bilinear_Zeros) {
  RunOpTestOnCPU("GridSample",
                 {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                  TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-1.5f, 1.5f, 48))},
                 {test::MakeAttribute("mode", "bilinear"), test::MakeAttribute("padding_mode", "zeros")},
                 /*opset_version=*/16,
                 ExpectedEPNodeAssignment::All);
}

// GridSample float, nearest + reflection padding + align_corners on the QNN CPU backend (rank 4).
TEST_F(QnnCPUBackendTests, GridSample_Float_Nearest_Reflection_AlignCorners) {
  RunOpTestOnCPU("GridSample",
                 {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                  TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-2.5f, 2.5f, 48))},
                 {test::MakeAttribute("mode", "nearest"),
                  test::MakeAttribute("padding_mode", "reflection"),
                  test::MakeAttribute("align_corners", static_cast<int64_t>(1))},
                 /*opset_version=*/20,
                 ExpectedEPNodeAssignment::All);
}

// 8-bit QDQ GridSample, linear + border padding on the QNN CPU backend.
TEST_F(QnnCPUBackendTests, GridSample_U8_Linear_Border) {
  RunQDQOpTestOnCPU<uint8_t>("GridSample",
                             {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                              TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-1.5f, 1.5f, 48))},
                             {test::MakeAttribute("mode", "linear"), test::MakeAttribute("padding_mode", "border")},
                             /*opset_version=*/20,
                             ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, GridSample_U16_Unsupported) {
  RunQDQOpTestOnCPU<uint16_t>("GridSample",
                              {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                               TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-1.5f, 1.5f, 48))},
                              {test::MakeAttribute("mode", "linear"), test::MakeAttribute("padding_mode", "border")},
                              /*opset_version=*/20,
                              ExpectedEPNodeAssignment::None,
                              kOnnxDomain,
                              /*use_contrib_qdq=*/true);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__linux__)

namespace {

constexpr const char* kOrtContribDomain = "ai.onnx.contrib";
constexpr const char* kUnknownDomain = "foo";

// Test-only CPU reference for GridSample with ai.onnx.contrib semantics.
//   X:    [N, C, spatial...]  (rank 4 or 5)
//   grid: [N, out_spatial..., S] where S = rank(X) - 2, coordinates ordered (x, y[, z]) i.e. (W, H[, D]).
//   Y:    [N, C, out_spatial...]
// Attributes (INT): mode (0 = bilinear, 1 = nearest), padding_mode (0 = zeros, 1 = border, 2 = reflection),
// align_corners (0/1). Unknown mode values are treated as bilinear; unknown padding values as zeros. Those
// values are only used by tests that expect the QNN EP to reject the node.
struct GridSampleRef {
  GridSampleRef(const OrtApi* ort_api, const OrtKernelInfo* info) {
    ReadIntAttr(ort_api, info, "mode", mode_);
    ReadIntAttr(ort_api, info, "padding_mode", padding_mode_);
    ReadIntAttr(ort_api, info, "align_corners", align_corners_);
  }

  Ort::Status Compute(const Ort::Custom::Tensor<float>& X,
                      const Ort::Custom::Tensor<float>& grid,
                      Ort::Custom::Tensor<float>* Y) {
    const std::vector<int64_t>& x_shape = X.Shape();
    const std::vector<int64_t>& grid_shape = grid.Shape();
    if (x_shape.size() < 4 || x_shape.size() > 5 || grid_shape.size() != x_shape.size() ||
        grid_shape.back() != static_cast<int64_t>(x_shape.size()) - 2 || grid_shape[0] != x_shape[0]) {
      return Ort::Status("GridSampleRef: unsupported input/grid shapes", ORT_INVALID_ARGUMENT);
    }

    const size_t spatial_rank = x_shape.size() - 2;
    const int64_t batch = x_shape[0];
    const int64_t channels = x_shape[1];
    std::vector<int64_t> in_dims(x_shape.begin() + 2, x_shape.end());
    std::vector<int64_t> out_shape{batch, channels};
    out_shape.insert(out_shape.end(), grid_shape.begin() + 1, grid_shape.end() - 1);

    const int64_t in_total = std::accumulate(in_dims.begin(), in_dims.end(), int64_t{1}, std::multiplies<int64_t>());
    const int64_t out_total = std::accumulate(grid_shape.begin() + 1, grid_shape.end() - 1, int64_t{1},
                                              std::multiplies<int64_t>());

    const float* x_data = X.Data();
    const float* grid_data = grid.Data();
    float* y_data = Y->Allocate(out_shape);

    struct Tap {
      int64_t offset;
      float weight;
    };
    std::vector<Tap> taps;

    for (int64_t n = 0; n < batch; ++n) {
      for (int64_t p = 0; p < out_total; ++p) {
        const float* g = grid_data + (n * out_total + p) * static_cast<int64_t>(spatial_rank);

        // Spatial dim d (0 = outermost) reads grid component (spatial_rank - 1 - d).
        std::vector<float> coord(spatial_rank);
        for (size_t d = 0; d < spatial_rank; ++d) {
          coord[d] = ApplyPadding(Unnormalize(g[spatial_rank - 1 - d], in_dims[d]), in_dims[d]);
        }

        taps.clear();
        if (mode_ == 1) {
          int64_t offset = 0;
          bool valid = true;
          for (size_t d = 0; d < spatial_rank; ++d) {
            const int64_t idx = static_cast<int64_t>(std::nearbyint(coord[d]));
            if (idx < 0 || idx >= in_dims[d]) {
              valid = false;
              break;
            }
            offset = offset * in_dims[d] + idx;
          }
          if (valid) {
            taps.push_back({offset, 1.0f});
          }
        } else {
          std::vector<int64_t> lo(spatial_rank);
          std::vector<float> frac(spatial_rank);
          for (size_t d = 0; d < spatial_rank; ++d) {
            const float fl = std::floor(coord[d]);
            lo[d] = static_cast<int64_t>(fl);
            frac[d] = coord[d] - fl;
          }
          for (uint32_t corner = 0; corner < (1u << spatial_rank); ++corner) {
            int64_t offset = 0;
            float weight = 1.0f;
            bool valid = true;
            for (size_t d = 0; d < spatial_rank; ++d) {
              const bool upper = ((corner >> d) & 1u) != 0;
              const int64_t idx = lo[d] + (upper ? 1 : 0);
              weight *= upper ? frac[d] : (1.0f - frac[d]);
              if (idx < 0 || idx >= in_dims[d]) {
                valid = false;
                break;
              }
              offset = offset * in_dims[d] + idx;
            }
            if (valid && weight != 0.0f) {
              taps.push_back({offset, weight});
            }
          }
        }

        for (int64_t c = 0; c < channels; ++c) {
          const float* x_chan = x_data + (n * channels + c) * in_total;
          float acc = 0.0f;
          for (const Tap& tap : taps) {
            acc += tap.weight * x_chan[tap.offset];
          }
          y_data[(n * channels + c) * out_total + p] = acc;
        }
      }
    }

    return Ort::Status{nullptr};
  }

  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    const Ort::ShapeInferContext::Shape& x_shape = ctx.GetInputShape(0);
    const Ort::ShapeInferContext::Shape& grid_shape = ctx.GetInputShape(1);
    if (x_shape.size() < 4 || grid_shape.size() != x_shape.size()) {
      return Ort::Status("GridSampleRef: unsupported ranks", ORT_INVALID_ARGUMENT);
    }
    Ort::ShapeInferContext::Shape out_shape;
    out_shape.push_back(x_shape[0]);
    out_shape.push_back(x_shape[1]);
    for (size_t i = 1; i + 1 < grid_shape.size(); ++i) {
      out_shape.push_back(grid_shape[i]);
    }
    return ctx.SetOutputShape(0, out_shape);
  }

 private:
  static void ReadIntAttr(const OrtApi* ort_api, const OrtKernelInfo* info, const char* name, int64_t& value) {
    // Keep the default when the attribute is absent (or has a different type, e.g. a string).
    OrtStatus* status = ort_api->KernelInfoGetAttribute_int64(info, name, &value);
    if (status != nullptr) {
      ort_api->ReleaseStatus(status);
    }
  }

  float Unnormalize(float g, int64_t size) const {
    if (align_corners_ != 0) {
      return (g + 1.0f) * 0.5f * static_cast<float>(size - 1);
    }
    return ((g + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
  }

  float ApplyPadding(float x, int64_t size) const {
    const float max_idx = static_cast<float>(size - 1);
    if (padding_mode_ == 1) {
      return std::min(std::max(x, 0.0f), max_idx);
    }
    if (padding_mode_ == 2) {
      const float lo = align_corners_ != 0 ? 0.0f : -0.5f;
      const float hi = align_corners_ != 0 ? max_idx : static_cast<float>(size) - 0.5f;
      const float span = hi - lo;
      if (span <= 0.0f) {
        return 0.0f;
      }
      const float t = std::fmod(std::fabs(x - lo), 2.0f * span);
      const float r = t <= span ? lo + t : lo + 2.0f * span - t;
      return std::min(std::max(r, 0.0f), max_idx);
    }
    return x;  // zeros: out-of-bound taps are dropped when sampling.
  }

  int64_t mode_ = 0;
  int64_t padding_mode_ = 0;
  int64_t align_corners_ = 0;
};

// Owns a custom-op domain containing the GridSample reference kernel. Not movable: the domain stores a raw
// pointer to the op, so both must stay alive (and in place) for the lifetime of the session.
class GridSampleRefDomain {
 public:
  explicit GridSampleRefDomain(const char* domain)
      : domain_(domain),
        op_(Ort::Custom::CreateLiteCustomOp<GridSampleRef>("GridSample", "CPUExecutionProvider")) {
    domain_.Add(op_.get());
  }

  GridSampleRefDomain(const GridSampleRefDomain&) = delete;
  GridSampleRefDomain& operator=(const GridSampleRefDomain&) = delete;

  Ort::CustomOpDomain* Get() { return &domain_; }

 private:
  Ort::CustomOpDomain domain_;
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> op_;
};

// Adds an opset_import for `domain` (version 1) before running the wrapped builder. com.microsoft is added by
// RunQnnModelTest/TestQDQModelAccuracy, so it must not be added again here.
GetTestModelFn WithOpsetImport(GetTestModelFn build_fn, std::string domain) {
  return [build_fn = std::move(build_fn), domain = std::move(domain)](ModelTestBuilder& builder) {
    auto* opset = builder.model_.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(1);
    build_fn(builder);
  };
}

template <typename QuantType>
GetTestQDQModelFn<QuantType> WithOpsetImport(GetTestQDQModelFn<QuantType> build_fn, std::string domain) {
  return [build_fn = std::move(build_fn), domain = std::move(domain)](
             ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    auto* opset = builder.model_.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(1);
    build_fn(builder, output_qparams);
  };
}

bool NeedsOpsetImport(const std::string& domain) {
  return domain != kMSDomain && domain != kOnnxDomain;
}

size_t NumElements(const std::vector<int64_t>& shape) {
  return static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
}

std::vector<TestInputDef<float>> MakeGridSampleInputDefs(const std::vector<int64_t>& x_shape,
                                                         const std::vector<int64_t>& grid_shape,
                                                         float grid_min, float grid_max) {
  return {TestInputDef<float>(x_shape, false, GetFloatDataInRange(-10.0f, 10.0f, NumElements(x_shape))),
          TestInputDef<float>(grid_shape, false,
                              GetFloatDataInRange(grid_min, grid_max, NumElements(grid_shape)))};
}

// Deterministic grid for nearest mode. A random grid can land arbitrarily close to a rounding boundary (pixel .5) or
// a reflection edge, where HTP's fixed-point arithmetic may legitimately pick a different neighbour than the CPU
// reference. Every coordinate here is placed at pixel offset .3 or .7, i.e. >= 0.2 pixel from any rounding boundary
// and from any reflection edge for both align_corners settings, and a few are outside the image so the padding
// modes are exercised. Values are cycled deterministically over the grid points and axes.
std::vector<float> MakeNearestSafeGrid(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& grid_shape,
                                       int64_t align_corners) {
  const size_t spatial_rank = x_shape.size() - 2;
  std::vector<std::vector<float>> candidates(spatial_rank);
  for (size_t a = 0; a < spatial_rank; ++a) {
    // grid component a addresses the a-th innermost spatial axis: 0 -> W, 1 -> H, 2 -> D.
    const int64_t size = x_shape[x_shape.size() - 1 - a];
    if (align_corners != 0 && size == 1) {
      // Pixel position is degenerate (always 0) when align_corners = 1 and size = 1.
      candidates[a] = {-0.7f, -0.3f, 0.3f, 0.7f};
      continue;
    }
    for (int64_t k = -1; k < size; ++k) {
      for (float frac : {0.3f, 0.7f}) {
        const float pixel = static_cast<float>(k) + frac;
        candidates[a].push_back(align_corners != 0
                                    ? 2.0f * pixel / static_cast<float>(size - 1) - 1.0f
                                    : (2.0f * pixel + 1.0f) / static_cast<float>(size) - 1.0f);
      }
    }
  }

  const size_t num_points = NumElements(grid_shape) / spatial_rank;
  std::vector<float> grid;
  grid.reserve(num_points * spatial_rank);
  for (size_t i = 0; i < num_points; ++i) {
    for (size_t a = 0; a < spatial_rank; ++a) {
      const auto& c = candidates[a];
      grid.push_back(c[(i * (2 * a + 3) + a) % c.size()]);
    }
  }
  return grid;
}

// Random grid for bilinear (smooth, so boundary proximity is harmless); deterministic safe grid for nearest.
std::vector<TestInputDef<float>> MakeGridSampleInputDefsForMode(bool nearest,
                                                                const std::vector<int64_t>& x_shape,
                                                                const std::vector<int64_t>& grid_shape,
                                                                int64_t align_corners,
                                                                float grid_min, float grid_max) {
  if (!nearest) {
    return MakeGridSampleInputDefs(x_shape, grid_shape, grid_min, grid_max);
  }
  return {TestInputDef<float>(x_shape, false, GetFloatDataInRange(-10.0f, 10.0f, NumElements(x_shape))),
          TestInputDef<float>(grid_shape, false, MakeNearestSafeGrid(x_shape, grid_shape, align_corners))};
}

// ai.onnx.contrib attribute helpers (all INT).
std::vector<ONNX_NAMESPACE::AttributeProto> ContribAttrs(int64_t mode, int64_t padding_mode, int64_t align_corners) {
  return {test::MakeAttribute("mode", mode),
          test::MakeAttribute("padding_mode", padding_mode),
          test::MakeAttribute("align_corners", align_corners)};
}

// com.microsoft attribute helpers (string mode/padding_mode, INT align_corners).
std::vector<ONNX_NAMESPACE::AttributeProto> MsAttrs(const std::string& mode, const std::string& padding_mode,
                                                    int64_t align_corners) {
  return {test::MakeAttribute("mode", mode),
          test::MakeAttribute("padding_mode", padding_mode),
          test::MakeAttribute("align_corners", align_corners)};
}

// Runs a QDQ GridSample on HTP and compares against the CPU EP running the same QDQ model. When `custom_op_domain`
// is non-null, its reference kernel provides both the float reference and the CPU fallback.
template <typename QuantType>
void RunGridSampleQDQ(const std::string& domain,
                      const std::vector<TestInputDef<float>>& input_defs,
                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                      ExpectedEPNodeAssignment expected_ep_assignment,
                      Ort::CustomOpDomain* custom_op_domain,
                      QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestModelFn f32_fn = BuildOpTestCase<float>("GridSample_node", "GridSample", input_defs, {}, attrs, domain);
  // 16-bit Q/DQ requires the com.microsoft Q/DQ ops at opset 20.
  GetTestQDQModelFn<QuantType> qdq_fn = BuildQDQOpTestCase<QuantType>(
      "GridSample_node", "GridSample", input_defs, {}, attrs, domain,
      /*use_contrib_qdq=*/std::is_same_v<QuantType, uint16_t>);
  if (NeedsOpsetImport(domain)) {
    f32_fn = WithOpsetImport(std::move(f32_fn), domain);
    qdq_fn = WithOpsetImport<QuantType>(std::move(qdq_fn), domain);
  }

  TestQDQModelAccuracy<QuantType>(f32_fn, qdq_fn, provider_options, /*opset_version=*/20, expected_ep_assignment,
                                  tolerance, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                  /*qnn_ctx_model_path=*/"",
                                  /*session_option_pairs=*/{},
                                  /*graph_optimization_level=*/std::nullopt,
                                  /*qnn_ep_graph_checker=*/nullptr,
                                  custom_op_domain);
}

// Runs a float (non-QDQ) GridSample on HTP and compares against the CPU EP.
void RunGridSampleFloat(const std::string& domain,
                        const std::vector<TestInputDef<float>>& input_defs,
                        const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                        ExpectedEPNodeAssignment expected_ep_assignment,
                        Ort::CustomOpDomain* custom_op_domain,
                        float abs_err = 1e-2f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  if (expected_ep_assignment != ExpectedEPNodeAssignment::None) {
    provider_options["enable_htp_fp16_precision"] = "1";
  }

  GetTestModelFn f32_fn = BuildOpTestCase<float>("GridSample_node", "GridSample", input_defs, {}, attrs, domain);
  if (NeedsOpsetImport(domain)) {
    f32_fn = WithOpsetImport(std::move(f32_fn), domain);
  }

  RunQnnModelTest(f32_fn,
                  provider_options,
                  /*opset_version=*/20,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(abs_err)},
                  OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                  /*verify_outputs=*/true,
                  custom_op_domain);
}

// Runs mode x padding_mode x align_corners (ai.onnx.contrib INT enums) for the given shapes.
template <typename QuantType>
void RunContribMatrix(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& grid_shape) {
  GridSampleRefDomain ref(kOrtContribDomain);
  for (int64_t mode : {int64_t{0}, int64_t{1}}) {
    for (int64_t padding_mode : {int64_t{0}, int64_t{1}, int64_t{2}}) {
      for (int64_t align_corners : {int64_t{0}, int64_t{1}}) {
        SCOPED_TRACE("mode=" + std::to_string(mode) + " padding_mode=" + std::to_string(padding_mode) +
                     " align_corners=" + std::to_string(align_corners) +
                     " rank=" + std::to_string(x_shape.size()));
        RunGridSampleQDQ<QuantType>(kOrtContribDomain,
                                    MakeGridSampleInputDefsForMode(mode == 1, x_shape, grid_shape, align_corners,
                                                                   -3.5f, 3.5f),
                                    ContribAttrs(mode, padding_mode, align_corners),
                                    ExpectedEPNodeAssignment::All,
                                    ref.Get());
      }
    }
  }
}

}  // namespace

namespace {

template <typename InputQType = uint8_t>
void RunQDQOpTest(const std::string& op_type,
                  const std::vector<TestInputDef<float>>& input_defs,
                  const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                  int opset_version,
                  ExpectedEPNodeAssignment expected_ep_assignment,
                  const std::string& op_domain = kOnnxDomain,
                  bool use_contrib_qdq = false,
                  QDQTolerance tolerance = QDQTolerance()) {
  RunQDQOpTestOnBackend<InputQType>("htp", op_type, input_defs, attrs, opset_version, expected_ep_assignment,
                                    op_domain, use_contrib_qdq, tolerance);
}

template <typename InputType = float>
void RunOpTest(const std::string& op_type,
               const std::vector<TestInputDef<InputType>>& input_defs,
               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
               int opset_version,
               ExpectedEPNodeAssignment expected_ep_assignment,
               const std::string& op_domain = kOnnxDomain,
               float fp32_abs_err = 1e-5f,
               bool enable_htp_fp16_precision = false) {
  RunOpTestOnBackend<InputType>("htp", op_type, input_defs, attrs, opset_version, expected_ep_assignment,
                                op_domain, fp32_abs_err, enable_htp_fp16_precision);
}

}  // namespace

// -------------------------------------------------------------------------------------------------------------
// ai.onnx:GridSample, HTP.
// -------------------------------------------------------------------------------------------------------------

// Test 8-bit QDQ GridSample with bilinear
TEST_F(QnnHTPBackendTests, GridSample_Bilinear) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                         TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                        {test::MakeAttribute("align_corners", static_cast<int64_t>(0)),
                         test::MakeAttribute("mode", "bilinear"),
                         test::MakeAttribute("padding_mode", "zeros")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ GridSample with bilinear
TEST_F(QnnHTPBackendTests, GridSample_U16_Bilinear) {
  RunQDQOpTest<uint16_t>("GridSample",
                         {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                          TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                         {test::MakeAttribute("align_corners", static_cast<int64_t>(0)),
                          test::MakeAttribute("mode", "bilinear"),
                          test::MakeAttribute("padding_mode", "zeros")},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use com.microsoft Q/DQ ops
}

// Test 8-bit QDQ GridSample with align corners
TEST_F(QnnHTPBackendTests, GridSample_AlignCorners) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                         TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                        {test::MakeAttribute("align_corners", static_cast<int64_t>(1)),
                         test::MakeAttribute("mode", "bilinear"),
                         test::MakeAttribute("padding_mode", "zeros")},
                        17,
                        ExpectedEPNodeAssignment::All,
                        kOnnxDomain,
                        false,
                        QDQTolerance(0.008f));
}

// Test 16-bit QDQ GridSample with align corners
TEST_F(QnnHTPBackendTests, GridSample_U16_AlignCorners) {
  RunQDQOpTest<uint16_t>("GridSample",
                         {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                          TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                         {test::MakeAttribute("align_corners", static_cast<int64_t>(1)),
                          test::MakeAttribute("mode", "bilinear"),
                          test::MakeAttribute("padding_mode", "zeros")},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);  // Use com.microsoft Q/DQ ops
}

// Test QDQ GridSample with padding mode: border
// Inaccuracy detected for output 'output', element 0.
// Output quant params: scale=0.046370312571525574, zero_point=129.
// Expected val: 3.3620510101318359
// QNN QDQ val: 3.2922921180725098 (err 0.069758892059326172)
// CPU QDQ val: 3.3850328922271729 (err 0.022981882095336914)
// Issue fixed in 2.30
TEST_F(QnnHTPBackendTests, GridSample_BorderPadding) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                         TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f)},
                        {test::MakeAttribute("mode", "bilinear"),
                         test::MakeAttribute("padding_mode", "border")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 8-bit QDQ GridSample with nearest mode
TEST_F(QnnHTPBackendTests, GridSample_Nearest) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                         TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                        {test::MakeAttribute("mode", "nearest")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ GridSample with nearest mode
TEST_F(QnnHTPBackendTests, GridSample_U16_Nearest) {
  RunQDQOpTest<uint16_t>("GridSample",
                         {TestInputDef<float>({1, 1, 3, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 6)),
                          TestInputDef<float>({1, 2, 4, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 16))},
                         {test::MakeAttribute("mode", "nearest")},
                         17,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// Test QDQ GridSample with `linear` mode on opset 20+.
TEST_F(QnnHTPBackendTests, GridSample_Linear_ZerosPadding) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                         TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))},
                        {test::MakeAttribute("mode", "linear"), test::MakeAttribute("padding_mode", "zeros")},
                        /*opset_version=*/20,
                        /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GridSample_Linear_AlignCorners_BorderPadding) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                         TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))},
                        {test::MakeAttribute("align_corners", static_cast<int64_t>(1)),
                         test::MakeAttribute("mode", "linear"),
                         test::MakeAttribute("padding_mode", "border")},
                        /*opset_version=*/20,
                        /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GridSample_Linear_ReflectionPadding_U16) {
  RunQDQOpTest<uint16_t>("GridSample",
                         {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                          TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 48))},
                         {test::MakeAttribute("mode", "linear"), test::MakeAttribute("padding_mode", "reflection")},
                         /*opset_version=*/21,
                         /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                         /*op_domain=*/kOnnxDomain,
                         /*use_contrib_qdq=*/true);
}

// Rank-5 float GridSample exceeds QNN HTP's rank-4 limit and must fall back to CPU.
TEST_F(QnnHTPBackendTests, GridSample_Rank5_Float_Unsupported) {
  RunOpTest<float>("GridSample",
                   {TestInputDef<float>({1, 1, 2, 2, 2}, false, GetFloatDataInRange(-10.0f, 10.0f, 8)),
                    TestInputDef<float>({1, 2, 2, 2, 3}, false, GetFloatDataInRange(-1.0f, 1.0f, 24))},
                   {test::MakeAttribute("mode", "linear"), test::MakeAttribute("padding_mode", "zeros")},
                   /*opset_version=*/20,
                   ExpectedEPNodeAssignment::None,
                   kOnnxDomain);
}

// Test QDQ GridSample with reflection padding mode
// Inaccuracy detected for output 'output', element 2.
// Output quant params: scale=0.024269860237836838, zero_point=0.
// Expected val: 3.212885856628418
// QNN QDQ val: 3.1308119297027588 (err 0.08207392692565918)
// CPU QDQ val: 3.2036216259002686 (err 0.0092642307281494141)
// fixed by QNN 2.32
TEST_F(QnnHTPBackendTests, GridSample_ReflectionPaddingMode) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 1, 3, 2}, false, -10.0f, 10.0f),
                         TestInputDef<float>({1, 2, 4, 2}, false, -10.0f, 10.0f)},
                        {test::MakeAttribute("padding_mode", "reflection")},
                        17,
                        ExpectedEPNodeAssignment::All);
}

namespace {

// Runs a QDQ GridSample for every combination of mode x padding_mode x align_corners, except those for which
// `skip` returns true (combinations already covered by a dedicated test).
// The grid range [-3.5, 3.5] exercises far-border clamping and multi-bounce reflection.
using GridSampleComboFilter = bool (*)(const std::string& mode, const std::string& padding_mode,
                                       int64_t align_corners);

template <typename QuantType>
void RunGridSampleMatrix(const std::vector<int64_t>& input_shape,
                         const std::vector<int64_t>& grid_shape,
                         GridSampleComboFilter skip = nullptr,
                         bool use_contrib_qdq = false) {
  const size_t input_size = NumElements(input_shape);
  const size_t grid_size = NumElements(grid_shape);

  for (const char* mode : {"linear", "nearest"}) {
    for (const char* padding_mode : {"zeros", "border", "reflection"}) {
      for (int64_t align_corners : {int64_t{0}, int64_t{1}}) {
        if (skip != nullptr && skip(mode, padding_mode, align_corners)) {
          continue;
        }
        SCOPED_TRACE(std::string("mode=") + mode + " padding_mode=" + padding_mode +
                     " align_corners=" + std::to_string(align_corners) +
                     " rank=" + std::to_string(input_shape.size()));
        RunQDQOpTest<QuantType>("GridSample",
                                {TestInputDef<float>(input_shape, false, GetFloatDataInRange(-10.0f, 10.0f, input_size)),
                                 TestInputDef<float>(grid_shape, false, GetFloatDataInRange(-3.5f, 3.5f, grid_size))},
                                {test::MakeAttribute("mode", std::string(mode)),
                                 test::MakeAttribute("padding_mode", std::string(padding_mode)),
                                 test::MakeAttribute("align_corners", align_corners)},
                                /*opset_version=*/20,
                                ExpectedEPNodeAssignment::All,
                                kOnnxDomain,
                                use_contrib_qdq);
      }
    }
  }
}

// Builds a grid of (x, y) pairs covering every (y, x) combination of the given coordinates, row-major.
std::vector<float> MakeGridFromCoords(const std::vector<float>& coords) {
  std::vector<float> grid;
  grid.reserve(coords.size() * coords.size() * 2);
  for (float y : coords) {
    for (float x : coords) {
      grid.push_back(x);
      grid.push_back(y);
    }
  }
  return grid;
}

}  // namespace

// QNN parameter matrix for rank-4 QDQ GridSample (u8): every mode x padding_mode x align_corners combination that
// the dedicated tests above do not already cover (GridSample_Bilinear, _AlignCorners, _BorderPadding,
// _Linear_ZerosPadding, _Linear_AlignCorners_BorderPadding, _ReflectionPaddingMode and _Nearest).
TEST_F(QnnHTPBackendTests, GridSample_Matrix_Rank4_U8) {
  RunGridSampleMatrix<uint8_t>(
      {1, 3, 4, 6}, {1, 4, 6, 2},
      [](const std::string& mode, const std::string& padding_mode, int64_t align_corners) {
        if (mode == "linear") {
          // Covered: zeros and border for both align_corners values, reflection for align_corners = 0.
          return padding_mode != "reflection" || align_corners == 0;
        }
        return padding_mode == "zeros" && align_corners == 0;  // nearest + zeros + align 0.
      });
}

// Full QNN parameter matrix for rank-5 QDQ GridSample (u8). HTP uses a trilinear path for linear mode.
TEST_F(QnnHTPBackendTests, GridSample_Matrix_Rank5_U8) {
  RunGridSampleMatrix<uint8_t>({1, 1, 2, 3, 4}, {1, 2, 2, 2, 3});
}

// Rank-5 16-bit QDQ GridSample, nearest + border + align_corners.
TEST_F(QnnHTPBackendTests, GridSample_Rank5_U16_Nearest_Border) {
  RunQDQOpTest<uint16_t>("GridSample",
                         {TestInputDef<float>({1, 1, 2, 3, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 24)),
                          TestInputDef<float>({1, 2, 2, 2, 3}, false, GetFloatDataInRange(-3.5f, 3.5f, 24))},
                         {test::MakeAttribute("mode", "nearest"),
                          test::MakeAttribute("padding_mode", "border"),
                          test::MakeAttribute("align_corners", static_cast<int64_t>(1))},
                         /*opset_version=*/20,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         /*use_contrib_qdq=*/true);
}

// Batch > 1, non-square input, output spatial size differs from input, C > 4.
TEST_F(QnnHTPBackendTests, GridSample_Batch2_OutputResize) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({2, 4, 5, 7}, false, GetFloatDataInRange(-10.0f, 10.0f, 280)),
                         TestInputDef<float>({2, 3, 6, 2}, false, GetFloatDataInRange(-1.5f, 1.5f, 72))},
                        {test::MakeAttribute("mode", "linear"), test::MakeAttribute("padding_mode", "zeros")},
                        /*opset_version=*/20,
                        ExpectedEPNodeAssignment::All);
}

// Grid supplied as a constant initializer.
TEST_F(QnnHTPBackendTests, GridSample_ConstantGrid) {
  RunQDQOpTest<uint8_t>("GridSample",
                        {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                         TestInputDef<float>({1, 4, 6, 2}, true, GetFloatDataInRange(-1.5f, 1.5f, 48))},
                        {test::MakeAttribute("mode", "linear"), test::MakeAttribute("padding_mode", "zeros")},
                        /*opset_version=*/20,
                        ExpectedEPNodeAssignment::All);
}

// Nearest mode with grid positions just either side of pixel .5 boundaries, for align_corners = 0 and 1.
// Exact ties are intentionally avoided: HTP's tie-break differs from the CPU reference and is not specified.
TEST_F(QnnHTPBackendTests, GridSample_Nearest_HalfPixelTies) {
  // Coordinates sit 0.1 pixel on either side of the half-pixel ties rather than exactly on them. Exact ties are
  // backend-defined (HTP rounds in fixed point, and the quantized grid perturbs them anyway), so asserting one
  // tie-break would make the test non-portable. Offsets of 0.1 pixel are > 2 grid quantization steps.
  // align_corners = 0, W = H = 4: pixel = ((x + 1) * 4 - 1) / 2, so x in {-0.55, 0.05, 0.45} -> pixel {0.4, 1.6, 2.4}.
  {
    SCOPED_TRACE("align_corners=0");
    RunQDQOpTest<uint8_t>("GridSample",
                          {TestInputDef<float>({1, 2, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 32)),
                           TestInputDef<float>({1, 3, 3, 2}, false, MakeGridFromCoords({-0.55f, 0.05f, 0.45f}))},
                          {test::MakeAttribute("mode", "nearest"),
                           test::MakeAttribute("padding_mode", "zeros"),
                           test::MakeAttribute("align_corners", static_cast<int64_t>(0))},
                          /*opset_version=*/20,
                          ExpectedEPNodeAssignment::All);
  }

  // align_corners = 1, W = H = 5: pixel = (x + 1) * 2, so x in {-0.8, -0.2, 0.2, 0.8} -> pixel {0.4, 1.6, 2.4, 3.6}.
  {
    SCOPED_TRACE("align_corners=1");
    RunQDQOpTest<uint8_t>("GridSample",
                          {TestInputDef<float>({1, 2, 5, 5}, false, GetFloatDataInRange(-10.0f, 10.0f, 50)),
                           TestInputDef<float>({1, 4, 4, 2}, false, MakeGridFromCoords({-0.8f, -0.2f, 0.2f, 0.8f}))},
                          {test::MakeAttribute("mode", "nearest"),
                           test::MakeAttribute("padding_mode", "zeros"),
                           test::MakeAttribute("align_corners", static_cast<int64_t>(1))},
                          /*opset_version=*/20,
                          ExpectedEPNodeAssignment::All);
  }
}

// Float rank-4 bilinear + border on HTP with fp16 precision.
TEST_F(QnnHTPBackendTests, GridSample_Float_Bilinear_Border) {
  RunOpTest<float>("GridSample",
                   {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                    TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-1.5f, 1.5f, 48))},
                   {test::MakeAttribute("mode", "bilinear"), test::MakeAttribute("padding_mode", "border")},
                   /*opset_version=*/16,
                   ExpectedEPNodeAssignment::All,
                   kOnnxDomain,
                   /*fp32_abs_err=*/1e-2f,
                   /*enable_htp_fp16_precision=*/true);
}

// Bicubic mode (opset < 20 name) has no QNN equivalent and must fall back to CPU EP.
TEST_F(QnnHTPBackendTests, GridSample_Bicubic_Unsupported) {
  RunOpTest<float>("GridSample",
                   {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                    TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-1.5f, 1.5f, 48))},
                   {test::MakeAttribute("mode", "bicubic")},
                   /*opset_version=*/16,
                   ExpectedEPNodeAssignment::None);
}

// Cubic mode (opset 20+ name) has no QNN equivalent and must fall back to CPU EP.
TEST_F(QnnHTPBackendTests, GridSample_Cubic_Unsupported) {
  RunOpTest<float>("GridSample",
                   {TestInputDef<float>({1, 3, 4, 6}, false, GetFloatDataInRange(-10.0f, 10.0f, 72)),
                    TestInputDef<float>({1, 4, 6, 2}, false, GetFloatDataInRange(-1.5f, 1.5f, 48))},
                   {test::MakeAttribute("mode", "cubic")},
                   /*opset_version=*/20,
                   ExpectedEPNodeAssignment::None);
}

// -------------------------------------------------------------------------------------------------------------
// ai.onnx.contrib:GridSample (INT attributes), HTP QDQ.
// -------------------------------------------------------------------------------------------------------------

// Full mode x padding_mode x align_corners matrix, rank 4, u8.
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_Matrix_Rank4_U8) {
  RunContribMatrix<uint8_t>({1, 3, 4, 6}, {1, 4, 6, 2});
}

// Full matrix, rank 5, u8. HTP uses a trilinear path for bilinear mode.
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_Matrix_Rank5_U8) {
  RunContribMatrix<uint8_t>({1, 1, 2, 3, 4}, {1, 2, 2, 2, 3});
}

// u16 subset, rank 4.
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_Rank4_U16_Subset) {
  GridSampleRefDomain ref(kOrtContribDomain);
  const std::vector<int64_t> x_shape = {1, 3, 4, 6};
  const std::vector<int64_t> grid_shape = {1, 4, 6, 2};
  const std::vector<std::vector<int64_t>> combos = {{0, 0, 0}, {0, 1, 1}, {0, 2, 0}, {1, 1, 1}, {1, 0, 0}};
  for (const auto& combo : combos) {
    SCOPED_TRACE("mode=" + std::to_string(combo[0]) + " padding_mode=" + std::to_string(combo[1]) +
                 " align_corners=" + std::to_string(combo[2]));
    RunGridSampleQDQ<uint16_t>(kOrtContribDomain,
                               MakeGridSampleInputDefsForMode(combo[0] == 1, x_shape, grid_shape, combo[2],
                                                              -3.5f, 3.5f),
                               ContribAttrs(combo[0], combo[1], combo[2]),
                               ExpectedEPNodeAssignment::All,
                               ref.Get());
  }
}

// u16 subset, rank 5.
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_Rank5_U16_Subset) {
  GridSampleRefDomain ref(kOrtContribDomain);
  const std::vector<int64_t> x_shape = {1, 1, 2, 3, 4};
  const std::vector<int64_t> grid_shape = {1, 2, 2, 2, 3};
  const std::vector<std::vector<int64_t>> combos = {{0, 0, 0}, {0, 1, 1}, {1, 1, 1}, {1, 2, 0}};
  for (const auto& combo : combos) {
    SCOPED_TRACE("mode=" + std::to_string(combo[0]) + " padding_mode=" + std::to_string(combo[1]) +
                 " align_corners=" + std::to_string(combo[2]));
    RunGridSampleQDQ<uint16_t>(kOrtContribDomain,
                               MakeGridSampleInputDefsForMode(combo[0] == 1, x_shape, grid_shape, combo[2],
                                                              -3.5f, 3.5f),
                               ContribAttrs(combo[0], combo[1], combo[2]),
                               ExpectedEPNodeAssignment::All,
                               ref.Get());
  }
}

// Shape mirroring the stereo-depthmap model: 5D X with D = 1, bilinear, border padding, align_corners = 1.
// Scaled down from X [1,16,1,192,1024] / grid [1,64,192,256,3].
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_ZooShape_Rank5_U8) {
  GridSampleRefDomain ref(kOrtContribDomain);
  RunGridSampleQDQ<uint8_t>(kOrtContribDomain,
                            MakeGridSampleInputDefs({1, 2, 1, 6, 8}, {1, 4, 6, 8, 3}, -1.5f, 1.5f),
                            ContribAttrs(/*mode=*/0, /*padding_mode=*/1, /*align_corners=*/1),
                            ExpectedEPNodeAssignment::All,
                            ref.Get());
}

// Rank 4 with batch 2, C > 4 and an output spatial size different from the input.
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_Batch2_OutputResize_U8) {
  GridSampleRefDomain ref(kOrtContribDomain);
  RunGridSampleQDQ<uint8_t>(kOrtContribDomain,
                            MakeGridSampleInputDefs({2, 4, 5, 7}, {2, 3, 6, 2}, -1.5f, 1.5f),
                            ContribAttrs(/*mode=*/0, /*padding_mode=*/0, /*align_corners=*/0),
                            ExpectedEPNodeAssignment::All,
                            ref.Get());
}

// Float rank 4 on HTP with fp16 precision.
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_Float_Rank4) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  GridSampleRefDomain ref(kOrtContribDomain);
  RunGridSampleFloat(kOrtContribDomain,
                     MakeGridSampleInputDefs({1, 3, 4, 6}, {1, 4, 6, 2}, -1.5f, 1.5f),
                     ContribAttrs(/*mode=*/0, /*padding_mode=*/1, /*align_corners=*/0),
                     ExpectedEPNodeAssignment::All,
                     ref.Get());
}

// Negative: rank-5 float exceeds the HTP float rank limit (4). The node must fall back to the CPU kernel.
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_Float_Rank5_Unsupported) {
  GridSampleRefDomain ref(kOrtContribDomain);
  RunGridSampleFloat(kOrtContribDomain,
                     MakeGridSampleInputDefs({1, 2, 1, 6, 8}, {1, 4, 6, 8, 3}, -1.5f, 1.5f),
                     ContribAttrs(/*mode=*/0, /*padding_mode=*/1, /*align_corners=*/1),
                     ExpectedEPNodeAssignment::None,
                     ref.Get(),
                     /*abs_err=*/1e-5f);
}

// Negative: mode = 2 (bicubic in the PyTorch/QNN enum) has no QNN equivalent here.
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_Mode2_Unsupported) {
  GridSampleRefDomain ref(kOrtContribDomain);
  RunGridSampleFloat(kOrtContribDomain,
                     MakeGridSampleInputDefs({1, 3, 4, 6}, {1, 4, 6, 2}, -1.5f, 1.5f),
                     ContribAttrs(/*mode=*/2, /*padding_mode=*/0, /*align_corners=*/0),
                     ExpectedEPNodeAssignment::None,
                     ref.Get(),
                     /*abs_err=*/1e-5f);
}

// Negative: attribute type mismatch. ai.onnx.contrib expects INT mode/padding_mode; string attributes must be
// rejected rather than silently replaced by defaults. (The reference kernel keeps its defaults for non-INT attrs.)
TEST_F(QnnHTPBackendTests, GridSample_OrtContrib_StringAttrs_Unsupported) {
  GridSampleRefDomain ref(kOrtContribDomain);
  RunGridSampleFloat(kOrtContribDomain,
                     MakeGridSampleInputDefs({1, 3, 4, 6}, {1, 4, 6, 2}, -1.5f, 1.5f),
                     MsAttrs("nearest", "border", /*align_corners=*/0),
                     ExpectedEPNodeAssignment::None,
                     ref.Get(),
                     /*abs_err=*/1e-5f);
}

// Negative: a domain the builder does not recognize must not be claimed even though the op type is GridSample.
TEST_F(QnnHTPBackendTests, GridSample_UnknownDomain_Unsupported) {
  GridSampleRefDomain ref(kUnknownDomain);
  RunGridSampleFloat(kUnknownDomain,
                     MakeGridSampleInputDefs({1, 3, 4, 6}, {1, 4, 6, 2}, -1.5f, 1.5f),
                     ContribAttrs(/*mode=*/0, /*padding_mode=*/0, /*align_corners=*/0),
                     ExpectedEPNodeAssignment::None,
                     ref.Get(),
                     /*abs_err=*/1e-5f);
}

// -------------------------------------------------------------------------------------------------------------
// com.microsoft:GridSample (string attributes, 4D only). ORT's real CPU kernel is the reference, so no custom
// op domain is registered.
// -------------------------------------------------------------------------------------------------------------

// Representative mode x padding_mode x align_corners subset, rank 4, u8. The builder's transpose lowering is shared
// with ai.onnx.contrib (full matrix above) and the string-attribute parsing with ai.onnx (full matrix above), so
// this only needs to show that the com.microsoft domain reaches both correctly.
TEST_F(QnnHTPBackendTests, GridSample_MsDomain_Rank4_U8_Subset) {
  struct Combo {
    const char* mode;
    const char* padding_mode;
    int64_t align_corners;
  };
  const Combo combos[] = {{"bilinear", "zeros", 0}, {"nearest", "border", 1}, {"bilinear", "reflection", 0}};
  const std::vector<int64_t> x_shape = {1, 3, 4, 6};
  const std::vector<int64_t> grid_shape = {1, 4, 6, 2};
  for (const Combo& combo : combos) {
    SCOPED_TRACE(std::string("mode=") + combo.mode + " padding_mode=" + combo.padding_mode +
                 " align_corners=" + std::to_string(combo.align_corners));
    RunGridSampleQDQ<uint8_t>(kMSDomain,
                              MakeGridSampleInputDefsForMode(std::string(combo.mode) == "nearest", x_shape,
                                                             grid_shape, combo.align_corners, -3.5f, 3.5f),
                              MsAttrs(combo.mode, combo.padding_mode, combo.align_corners),
                              ExpectedEPNodeAssignment::All,
                              /*custom_op_domain=*/nullptr);
  }
}

// Note: a com.microsoft rank-5 negative is intentionally absent. The contrib schema and CPU kernel are 4D only, so
// ORT rejects such a model before the QNN EP sees it; there is nothing for the EP to decline.

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
