// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Quantization helpers
// ---------------------------------------------------------------------------

template <typename QType>
static QType QuantizeVal(float val, float scale, int32_t zero_point) {
  constexpr float qmin = static_cast<float>(std::numeric_limits<QType>::min());
  constexpr float qmax = static_cast<float>(std::numeric_limits<QType>::max());
  float q = std::round(val / scale) + static_cast<float>(zero_point);
  return static_cast<QType>(std::max(qmin, std::min(qmax, q)));
}

template <typename QType>
static std::vector<QType> QuantizeData(const std::vector<float>& data, float scale, int32_t zp) {
  std::vector<QType> out(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    out[i] = QuantizeVal<QType>(data[i], scale, zp);
  }
  return out;
}

template <typename QType>
static QuantParams<QType> ComputeQuantParams(float rmin, float rmax) {
  return QuantParams<QType>::Compute(rmin, rmax);
}

static size_t NumElems(const std::vector<int64_t>& shape) {
  return static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>()));
}

// Conv attributes bundle.
struct QLinearConvAttrs {
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  int64_t group = 1;
  std::string auto_pad = "NOTSET";
};

/**
 * Builds a QLinearConv graph and dequantizes its output to float:
 *   x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [B] -> QLinearConv -> y_q -> DequantizeLinear -> y
 *
 * Output is dequantized so accuracy is compared in float space with tolerance (HTP and CPU EP can
 * differ by ~1 LSB on quantized conv). per_channel_weight makes w_scale/w_zp 1-D of size M.
 * dynamic_x_scale makes x_scale a graph input (rejection test) and omits the DequantizeLinear.
 */
template <typename AType, typename WType, typename YType>
static GetTestModelFn BuildQLinearConvTestCase(const std::vector<int64_t>& x_shape,
                                               const std::vector<int64_t>& w_shape,
                                               const QLinearConvAttrs& attrs,
                                               bool has_bias = false,
                                               bool per_channel_weight = false,
                                               bool dynamic_x_scale = false) {
  return [x_shape, w_shape, attrs, has_bias, per_channel_weight, dynamic_x_scale](ModelTestBuilder& builder) {
    const size_t num_x = NumElems(x_shape);
    const size_t num_w = NumElems(w_shape);
    const int64_t M = w_shape[0];  // output channels

    const auto float_x = GetFloatDataInRange(-1.0f, 1.0f, num_x);
    const auto float_w = GetFloatDataInRange(-0.3f, 0.3f, num_w);

    const auto qp_x = ComputeQuantParams<AType>(-1.0f, 1.0f);
    const auto qp_y = ComputeQuantParams<YType>(-4.0f, 4.0f);

    const auto q_x = QuantizeData<AType>(float_x, qp_x.scale, static_cast<int32_t>(qp_x.zero_point));

    // x (dynamic activation)
    builder.MakeInput<AType>("x", x_shape, q_x);

    // x_scale / x_zp
    if (dynamic_x_scale) {
      builder.MakeInput<float>("x_scale", {}, std::vector<float>{qp_x.scale});
    } else {
      builder.MakeScalarInitializer<float>("x_scale", qp_x.scale);
    }
    builder.MakeScalarInitializer<AType>("x_zp", qp_x.zero_point);

    // w / w_scale / w_zp — per-tensor or per-channel (axis 0 = M).
    std::vector<WType> q_w(num_w);
    if (per_channel_weight) {
      const size_t per_oc = num_w / static_cast<size_t>(M);
      std::vector<float> w_scales(static_cast<size_t>(M));
      std::vector<WType> w_zps(static_cast<size_t>(M));
      for (int64_t oc = 0; oc < M; ++oc) {
        // Use distinct per-channel ranges so each scale/zp differs — this exercises
        // the per-channel quantization math rather than using uniform scales.
        const float range = 0.1f * static_cast<float>(oc + 1);
        const auto qp_w = ComputeQuantParams<WType>(-range, range);
        w_scales[oc] = qp_w.scale;
        w_zps[oc] = qp_w.zero_point;
        for (size_t i = 0; i < per_oc; ++i) {
          const size_t idx = static_cast<size_t>(oc) * per_oc + i;
          q_w[idx] = QuantizeVal<WType>(float_w[idx], qp_w.scale, static_cast<int32_t>(qp_w.zero_point));
        }
      }
      builder.MakeInitializer<WType>("w", w_shape, q_w);
      builder.Make1DInitializer<float>("w_scale", w_scales);
      builder.Make1DInitializer<WType>("w_zp", w_zps);
    } else {
      const auto qp_w = ComputeQuantParams<WType>(-0.3f, 0.3f);
      q_w = QuantizeData<WType>(float_w, qp_w.scale, static_cast<int32_t>(qp_w.zero_point));
      builder.MakeInitializer<WType>("w", w_shape, q_w);
      builder.MakeScalarInitializer<float>("w_scale", qp_w.scale);
      builder.MakeScalarInitializer<WType>("w_zp", qp_w.zero_point);
    }

    // y_scale / y_zp
    builder.MakeScalarInitializer<float>("y_scale", qp_y.scale);
    builder.MakeScalarInitializer<YType>("y_zp", qp_y.zero_point);

    std::vector<std::string> node_inputs = {"x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp"};

    // Optional int32 bias, quantized with scale = x_scale * w_scale (per-channel if weight is), zp = 0.
    if (has_bias) {
      const auto float_b = GetFloatDataInRange(-0.1f, 0.1f, static_cast<size_t>(M));
      std::vector<int32_t> q_b(static_cast<size_t>(M));
      if (per_channel_weight) {
        const auto qp_w = ComputeQuantParams<WType>(-0.3f, 0.3f);  // all channels share range here
        for (int64_t oc = 0; oc < M; ++oc) {
          const float bias_scale = qp_x.scale * qp_w.scale;
          q_b[oc] = static_cast<int32_t>(std::round(float_b[oc] / bias_scale));
        }
      } else {
        const auto qp_w = ComputeQuantParams<WType>(-0.3f, 0.3f);
        const float bias_scale = qp_x.scale * qp_w.scale;
        for (int64_t oc = 0; oc < M; ++oc) {
          q_b[oc] = static_cast<int32_t>(std::round(float_b[oc] / bias_scale));
        }
      }
      builder.MakeInitializer<int32_t>("B", {M}, q_b);
      node_inputs.push_back("B");
    }

    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", attrs.auto_pad));
    if (attrs.group != 1) {
      conv_attrs.push_back(builder.MakeScalarAttribute("group", attrs.group));
    }
    if (!attrs.pads.empty() && attrs.auto_pad == "NOTSET") {
      conv_attrs.push_back(builder.MakeIntsAttribute("pads", attrs.pads));
    }
    if (!attrs.strides.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("strides", attrs.strides));
    }
    if (!attrs.dilations.empty()) {
      conv_attrs.push_back(builder.MakeIntsAttribute("dilations", attrs.dilations));
    }

    if (dynamic_x_scale) {
      // Rejection test: emit the quantized output directly (no DequantizeLinear wrapper, which QNN
      // would otherwise grab and break an ExpectedEPNodeAssignment::None check).
      builder.MakeOutput("y_q");
      builder.AddNode("QLinearConv", "QLinearConv", node_inputs, {"y_q"}, kOnnxDomain, conv_attrs);
    } else {
      builder.AddNode("QLinearConv", "QLinearConv", node_inputs, {"y_q"}, kOnnxDomain, conv_attrs);
      builder.MakeOutput("y");
      builder.AddNode("DequantizeLinear", "DequantizeLinear", {"y_q", "y_scale", "y_zp"}, {"y"}, kOnnxDomain);
    }
  };
}

template <typename AType = uint8_t, typename WType = uint8_t, typename YType = uint8_t>
static void RunQLinearConvTest(const std::vector<int64_t>& x_shape,
                               const std::vector<int64_t>& w_shape,
                               const QLinearConvAttrs& attrs,
                               const std::string& backend_name,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               bool has_bias = false,
                               bool per_channel_weight = false,
                               bool dynamic_x_scale = false,
                               int opset = 10) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = expected_ep_assignment;
  // Output dequantized with y_scale = 8/255 ≈ 0.031 per LSB; allow ~1.5 LSB for HTP-vs-CPU.
  verification_params.tensor_verifier = ElementwiseAbsoluteVerifier{0.05f};

  RunQnnModelTest(
      BuildQLinearConvTestCase<AType, WType, YType>(x_shape, w_shape, attrs, has_bias,
                                                    per_channel_weight, dynamic_x_scale),
      provider_options, opset, verification_params);
}

// ---------------------------------------------------------------------------
// Note: QLinearConv is only supported on the QNN HTP backend. The QNN CPU
// backend does not implement quantized Conv2d/Conv3d, so all accuracy and
// IsOpSupported tests for QLinearConv live under the HTP guard below. (The
// existing conv_test.cc follows the same pattern: only float Conv runs on CPU,
// all quantized Conv tests are HTP-only.)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// HTP backend accuracy tests
// ---------------------------------------------------------------------------

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, QLinearConvOp_DynamicScale_Unsupported) {
  // x_scale is a graph input (not initializer) — must not be assigned to QNN EP.
  QLinearConvAttrs attrs;
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>(
      {1, 2, 5, 5}, {3, 2, 3, 3}, attrs, "htp", ExpectedEPNodeAssignment::None,
      /*has_bias=*/false, /*per_channel_weight=*/false, /*dynamic_x_scale=*/true);
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_DynamicZeroPoint_Unsupported) {
  // x_zero_point is a dynamic graph input — must not be assigned to QNN EP.
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    builder.MakeInput<uint8_t>("x", {1, 2, 5, 5}, std::vector<uint8_t>(50, 128u));
    builder.MakeScalarInitializer<float>("x_scale", 0.02f);
    builder.MakeInput<uint8_t>("x_zp", {}, std::vector<uint8_t>{128u});  // dynamic
    builder.MakeInitializer<uint8_t>("w", {3, 2, 3, 3}, std::vector<uint8_t>(54, 128u));
    builder.MakeScalarInitializer<float>("w_scale", 0.02f);
    builder.MakeScalarInitializer<uint8_t>("w_zp", 128u);
    builder.MakeScalarInitializer<float>("y_scale", 0.04f);
    builder.MakeScalarInitializer<uint8_t>("y_zp", 128u);
    builder.MakeOutput("y_q");
    builder.AddNode("QLinearConv", "QLinearConv",
                    {"x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp"},
                    {"y_q"}, kOnnxDomain);
  };

  EPVerificationParams vp;
  vp.ep_node_assignment = ExpectedEPNodeAssignment::None;
  RunQnnModelTest(model_fn, provider_options, 10, vp);
}

// --- uint8 ---

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_Basic2D) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_Bias) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp",
                                                ExpectedEPNodeAssignment::All, /*has_bias=*/true);
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_Strides) {
  QLinearConvAttrs attrs;
  attrs.strides = {2, 2};
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_Pads) {
  QLinearConvAttrs attrs;
  attrs.pads = {1, 1, 1, 1};
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_AutoPadSameUpper) {
  QLinearConvAttrs attrs;
  attrs.auto_pad = "SAME_UPPER";
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_Depthwise) {
  QLinearConvAttrs attrs;
  attrs.group = 4;
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 4, 8, 8}, {4, 1, 3, 3}, attrs, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_PerChannelWeight) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp",
                                                ExpectedEPNodeAssignment::All, /*has_bias=*/false,
                                                /*per_channel_weight=*/true);
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_PerChannelWeight_Bias) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp",
                                                ExpectedEPNodeAssignment::All, /*has_bias=*/true,
                                                /*per_channel_weight=*/true);
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_Conv1D) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 2, 8}, {3, 2, 3}, attrs, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_u8_Conv3D) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<uint8_t, uint8_t, uint8_t>({1, 2, 5, 5, 5}, {3, 2, 3, 3, 3}, attrs, "htp");
}

// --- int8 ---

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_s8_Basic2D) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<int8_t, int8_t, int8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp");
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_s8_PerChannelWeight) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<int8_t, int8_t, int8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp",
                                             ExpectedEPNodeAssignment::All, /*has_bias=*/false,
                                             /*per_channel_weight=*/true);
}

TEST_F(QnnHTPBackendTests, QLinearConvOp_HTP_Mixed_u8s8u8) {
  QLinearConvAttrs attrs;
  RunQLinearConvTest<uint8_t, int8_t, uint8_t>({1, 3, 8, 8}, {4, 3, 3, 3}, attrs, "htp");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
