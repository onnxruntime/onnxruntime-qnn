// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

/*
  ONNX GRU inputs:
  in[0]: X [seq_length, batch_size, input_size]
  in[1]: W [num_directions, 3*hidden_size, input_size]
  in[2]: R [num_directions, 3*hidden_size, hidden_size]

  ONNX GRU optional inputs:
  in[3]: B [num_directions, 6*hidden_size]
  in[4]: sequence_lens [batch_size]  (not used)
  in[5]: initial_h [num_directions, batch_size, hidden_size]

  ONNX GRU Parameters:
  - direction
  - hidden_size
  - linear_before_reset: When computing the output of the hidden gate, apply the
                         linear transformation before multiplying by the output of
                         the reset gate. Default: 0.
  - layout: The shape format of inputs X, initial_h and outputs Y, Y_h.
            If 0, the following shapes are expected:
                X.shape = [seq_length, batch_size, input_size],
                Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                initial_h.shape = Y_h.shape = [num_directions, batch_size, hidden_size].
            If 1, the following shapes are expected:
                X.shape = [batch_size, seq_length, input_size],
                Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                initial_h.shape = Y_h.shape = [batch_size, num_directions, hidden_size].

  ONNX GRU optional outputs:
  out[0]: Y [seq_length, num_directions, batch_size, hidden_size]
  out[1]: Y_h [num_directions, batch_size, hidden_size]
*/

template <typename InputType>
void _BuildGRUTestCase(ModelTestBuilder& builder,
                       const TestInputDef<float>& X_def,
                       const TestInputDef<float>& W_def,
                       const TestInputDef<float>& R_def,
                       const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                       const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                       const bool has_Y,
                       const bool has_Y_h,
                       const std::string direction,
                       const int64_t hidden_size,
                       const int64_t layout,
                       const int64_t linear_before_reset,
                       const std::vector<QuantParams<InputType>>& output_qparams,
                       const bool int32_bias = true) {
  static constexpr bool kIsFp16 = std::is_same<InputType, Ort::Float16_t>::value;
  static constexpr bool kIsU8 = std::is_same<InputType, uint8_t>::value;
  static constexpr bool kIsU16 = std::is_same<InputType, uint16_t>::value;

  auto add_input = [&](const char* name, const TestInputDef<float>& def) -> std::string {
    if constexpr (kIsFp16) {
      TestInputDef<Ort::Float16_t> fp16_def = ConvertToFP16InputDef(def);
      MakeTestInput(builder, name, fp16_def);
      return name;
    } else if constexpr (kIsU8 || kIsU16) {
      MakeTestInput(builder, name, def);
      QuantParams<InputType> qparams = GetTestInputQuantParams<InputType>(def);
      return AddQDQNodePair<InputType>(builder, std::string("qdq_") + name, name, qparams.scale, qparams.zero_point);
    } else {
      MakeTestInput(builder, name, def);
      return name;
    }
  };

  // Required inputs
  const std::string x_name = add_input("X", X_def);
  const std::string w_name = add_input("W", W_def);
  const std::string r_name = add_input("R", R_def);

  // Optional inputs are positional for GRU; represent missing values with empty string.
  std::vector<std::string> input_names;
  input_names.reserve(6);
  input_names.push_back(x_name);
  input_names.push_back(w_name);
  input_names.push_back(r_name);

  // B
  if (B_def) {
    // HTP's quantized Gru configs use an int32 (SFIXED_POINT_32) bias only -- the INT16 (u16) and INT8
    // (u8) configs both require it -- so int32 bias is the default. int32_bias=false forces an off-spec
    // u8 QDQ bias, used only by the GRU_QDQ_u8_bias_fp_degrade guard to prove that path fp-degrades.
    // Quantize to int32 with the usual input_scale * weight_scale bias scale. The float reference model
    // always takes a plain (non-QDQ) bias.
    if constexpr (kIsU16) {
      QuantParams<uint16_t> x_qparams = GetTestInputQuantParams<uint16_t>(X_def);
      QuantParams<uint16_t> w_qparams = GetTestInputQuantParams<uint16_t>(W_def);
      input_names.push_back(
          MakeTestQDQBiasInput(builder, "B", B_def->get(), x_qparams.scale * w_qparams.scale, false));
    } else if constexpr (kIsU8) {
      if (int32_bias) {
        QuantParams<uint8_t> x_qparams = GetTestInputQuantParams<uint8_t>(X_def);
        QuantParams<uint8_t> w_qparams = GetTestInputQuantParams<uint8_t>(W_def);
        input_names.push_back(
            MakeTestQDQBiasInput(builder, "B", B_def->get(), x_qparams.scale * w_qparams.scale, false));
      } else {
        input_names.push_back(add_input("B", B_def->get()));
      }
    } else {
      input_names.push_back(add_input("B", B_def->get()));
    }
  } else {
    input_names.push_back("");
  }

  // sequence_lens (not used)
  input_names.push_back("");

  // initial_h
  if (H_def) {
    input_names.push_back(add_input("initial_h", H_def->get()));
  } else {
    input_names.push_back("");
  }

  // Outputs
  auto make_output = [&](const char* name) -> std::string {
    if (name == nullptr || name[0] == '\0') return "";
    if constexpr (kIsU8 || kIsU16) {
      return std::string("gru_") + name;
    } else {
      builder.MakeOutput(name);
      return name;
    }
  };

  const std::string y_out = has_Y ? make_output("Y") : std::string("");
  const std::string y_h_out = has_Y_h ? make_output("Y_h") : std::string("");

  std::vector<std::string> output_names;
  output_names.push_back(y_out);
  output_names.push_back(y_h_out);

  // Attributes
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(builder.MakeStringAttribute("direction", direction));
  attrs.push_back(builder.MakeScalarAttribute("hidden_size", hidden_size));
  attrs.push_back(builder.MakeScalarAttribute("layout", layout));
  attrs.push_back(builder.MakeScalarAttribute("linear_before_reset", linear_before_reset));

  builder.AddNode("gru", "GRU", input_names, output_names, "", attrs);

  QNN_TEST_UNUSED_PARAMETER(output_qparams);
  QNN_TEST_UNUSED_PARAMETER(int32_bias);
  if constexpr (kIsU8 || kIsU16) {
    size_t i = 0;
    if (has_Y) {
      AddQDQNodePairWithOutputAsGraphOutput<InputType>(builder, "qdq_Y", y_out, output_qparams[i].scale,
                                                       output_qparams[i].zero_point);
      ++i;
    }
    if (has_Y_h) {
      AddQDQNodePairWithOutputAsGraphOutput<InputType>(builder, "qdq_Y_h", y_h_out, output_qparams[i].scale,
                                                       output_qparams[i].zero_point);
      ++i;
    }
  }
}

template <typename InputType>
static GetTestModelFn BuildGRUTestCase(const TestInputDef<float>& X_def,
                                       const TestInputDef<float>& W_def,
                                       const TestInputDef<float>& R_def,
                                       const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                       const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                       const bool has_Y,
                                       const bool has_Y_h,
                                       const std::string direction,
                                       const int64_t hidden_size,
                                       const int64_t layout,
                                       const int64_t linear_before_reset = 0) {
  return [X_def, W_def, R_def, B_def, H_def,
          has_Y, has_Y_h,
          direction, hidden_size, layout, linear_before_reset](ModelTestBuilder& builder) {
    _BuildGRUTestCase<InputType>(builder, X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                 direction, hidden_size, layout, linear_before_reset, {});
  };
}

template <typename InputQType>
static GetTestQDQModelFn<InputQType> BuildQDQGRUTestCase(const TestInputDef<float>& X_def,
                                                         const TestInputDef<float>& W_def,
                                                         const TestInputDef<float>& R_def,
                                                         const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                                         const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                                         const bool has_Y,
                                                         const bool has_Y_h,
                                                         const std::string direction,
                                                         const int64_t hidden_size,
                                                         const int64_t layout,
                                                         const int64_t linear_before_reset = 0,
                                                         const bool int32_bias = true) {
  return [X_def, W_def, R_def, B_def, H_def,
          has_Y, has_Y_h,
          direction, hidden_size, layout, linear_before_reset, int32_bias](ModelTestBuilder& builder,
                                                                           std::vector<QuantParams<InputQType>>& output_qparams) {
    _BuildGRUTestCase<InputQType>(builder, X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                  direction, hidden_size, layout, linear_before_reset, output_qparams, int32_bias);
  };
}

// Runs a GRU model on the QNN CPU backend with FP32.
static void RunCpuFP32GRUOpTest(const TestInputDef<float>& X_def,
                                const TestInputDef<float>& W_def,
                                const TestInputDef<float>& R_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                const bool has_Y,
                                const bool has_Y_h,
                                const std::string direction,
                                const int64_t hidden_size,
                                const int64_t layout,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                const int64_t linear_before_reset = 0,
                                float tolerance = 0.004f,
                                int opset = 22) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  RunQnnModelTest(BuildGRUTestCase<float>(X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                          direction, hidden_size, layout, linear_before_reset),
                  provider_options,
                  opset,
                  EPVerificationParams{expected_ep_assignment, ElementwiseAbsoluteVerifier(tolerance)});
}

// ============================================================
// CPU FP32 Tests
// ============================================================

TEST_F(QnnCPUBackendTests, GRU_fp32_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 6;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, GRU_fp32_sanity_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 6;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All);
}

// Y-only (has_Y=true, has_Y_h=false)
TEST_F(QnnCPUBackendTests, GRU_fp32_Y_only_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 6;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      false,                                                                                   // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All);
}

// Y_h-only (has_Y=false, has_Y_h=true)
TEST_F(QnnCPUBackendTests, GRU_fp32_Y_h_only_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 6;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      false,                                                                                   // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All);
}

// layout=1: ORT CPU EP does not support batchwise layout, so session initialization throws.
// Verify the expected failure so the test actively guards against silent regressions.
TEST_F(QnnCPUBackendTests, GRU_fp32_layout1_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 6;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  // layout=1: X [batch, seq, input], initial_h [batch, num_directions, hidden]
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({batch_size, num_direction, hidden_size}, false, -1.0f, 1.0f);
  EXPECT_THROW(
      RunCpuFP32GRUOpTest(TestInputDef<float>({batch_size, seq_len, input_size}, false, -1.0f, 1.0f),              // X
                          TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                          TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                          std::ref(B_def),                                                                         // B
                          std::ref(H_def),                                                                         // initial_h
                          true,                                                                                    // has_Y
                          true,                                                                                    // has_Y_h
                          direction,                                                                               // direction
                          hidden_size,                                                                             // hidden_size
                          1,                                                                                       // layout
                          ExpectedEPNodeAssignment::None),
      std::exception);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Runs a GRU model on the QNN HTP backend with QDQ quantization.
template <typename QuantType>
static void RunHtpQDQGRUOpTest(const TestInputDef<float>& X_def,
                               const TestInputDef<float>& W_def,
                               const TestInputDef<float>& R_def,
                               const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                               const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                               const bool has_Y,
                               const bool has_Y_h,
                               const std::string direction,
                               const int64_t hidden_size,
                               const int64_t layout,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               const int64_t linear_before_reset = 0,
                               QDQTolerance tolerance = QDQTolerance(),
                               int opset = 22,
                               bool int32_bias = true) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildGRUTestCase<float>(X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                               direction, hidden_size, layout, linear_before_reset),
                       BuildQDQGRUTestCase<QuantType>(X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                                      direction, hidden_size, layout, linear_before_reset, int32_bias),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

// Runs a GRU model on the QNN HTP backend with FP16 precision.
static void RunHtpFp16GRUOpTest(const TestInputDef<float>& X_def,
                                const TestInputDef<float>& W_def,
                                const TestInputDef<float>& R_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                const bool has_Y,
                                const bool has_Y_h,
                                const std::string direction,
                                const int64_t hidden_size,
                                const int64_t layout,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                const int64_t linear_before_reset = 0,
                                float tolerance = 0.004f,
                                int opset = 22) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  TestFp16ModelAccuracy(BuildGRUTestCase<float>(X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                                direction, hidden_size, layout, linear_before_reset),
                        BuildGRUTestCase<Ort::Float16_t>(X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                                         direction, hidden_size, layout, linear_before_reset),
                        provider_options,
                        opset,
                        expected_ep_assignment,
                        tolerance);
}

// ============================================================
// HTP QDQ Tests
// ============================================================

// u8 QDQ GRU, linear_before_reset=0. The selector is structural-only: it folds DQ -> GRU -> Q into a
// single QDQ group. The builder then fp-degrades LBR=0 (a fp-fallback trigger, because HTP can't
// finalize a u8 LBR=0 cell: the gate matmul widens u8 -> QUint16Crouton -> Code 1002 on v73/v81),
// emitting an explicit Dequantize -> fp32 GRU -> Quantize -- all on QNN, so the assignment stays All.
// Validates the LBR=0 fp-degrade + fp accuracy, not u8 HTP exec (genuine u8 = GRU_QDQ_linear_before_reset).
TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All);
}

// seq_len=1 variant of GRU_QDQ_sanity_forward. seq=1 still 1002s as u8 LBR=0, so the per-timestep
// unroll is not the discriminator -- linear_before_reset=0 is. fp-degraded by the builder, same as forward.
TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_forward_seq1) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 1;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All);
}

// LBR=0 u8 QDQ GRU, fp-degraded by the builder (see GRU_QDQ_sanity_forward). Validates fp-degrade, not u8 exec.
TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_reverse) {
  std::string direction = "reverse";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All);
}

// LBR=0 u8 QDQ GRU, fp-degraded by the builder (see GRU_QDQ_sanity_forward). Validates fp-degrade, not u8 exec.
TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All);
}

// LBR=0 u8 QDQ GRU, fp-degraded by the builder (see GRU_QDQ_sanity_forward). Validates fp-degrade, not u8 exec.
TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_bidirectional_wo_B) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::nullopt,                                                                            // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All);
}

// LBR=0 u8 QDQ GRU, fp-degraded by the builder (see GRU_QDQ_sanity_forward). Validates fp-degrade, not u8 exec.
TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_bidirectional_wo_H) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::nullopt,                                                                            // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All);
}

// LBR=0 u8 QDQ GRU, fp-degraded by the builder (see GRU_QDQ_sanity_forward). Validates fp-degrade, not u8 exec.
TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_bidirectional_all_initializer) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, true, -0.5f, 0.5f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -0.5f, 0.5f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -0.5f, 0.5f),             // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, true, -0.5f, 0.5f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, true, -0.5f, 0.5f),  // R
                              std::ref(B_def),                                                                        // B
                              std::ref(H_def),                                                                        // initial_h
                              true,                                                                                   // has_Y
                              true,                                                                                   // has_Y_h
                              direction,                                                                              // direction
                              hidden_size,                                                                            // hidden_size
                              0,                                                                                      // layout
                              ExpectedEPNodeAssignment::All,
                              0,
                              QDQTolerance(0.004f));
}

// Native u16 QDQ GRU with LBR=0. The LBR=0 fp-degrade exists only for the u8 combo -- its u8 cell
// widens to a mixed-width QUint16Crouton that fails HTP finalize (1002). The u16 combo is already
// 16-bit with no such widening, so LBR=0 runs genuine native u16 (X/W/R/initial_h u16 + int32 bias,
// forward, both outputs). Skipped on the x86 HTP emulator (no faithful native-INT16 Gru kernel; see
// GRU_QDQ_u16_linear_before_reset); validated on real silicon @3.0%.
TEST_F(QnnHTPBackendTests, GRU_QDQ_u16_sanity_forward) {
  QNN_SKIP_TEST_ON_LINUX_X86_64("native INT16 Gru kernel unsupported on linux x86_64 HTP emulator; requires real device.");
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  constexpr float kTolerance = 0.03f;
  RunHtpQDQGRUOpTest<uint16_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::ref(B_def),                                                                         // B
                               std::ref(H_def),                                                                         // initial_h
                               true,                                                                                    // has_Y
                               true,                                                                                    // has_Y_h
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All,
                               0,  // linear_before_reset
                               QDQTolerance(kTolerance));
}

// Native u16 QDQ GRU: HTP's INT16 Gru config with u16 X/W/R/initial_h, an int32 bias, forward direction,
// LBR=1. u16 mirror of GRU_QDQ_linear_before_reset; exercises the genuine native-u16 builder path (no
// builder-inserted Dequantize/Quantize) with LBR=1, as GRU_QDQ_u16_sanity_forward does with LBR=0. Like
// the u8 mirror it
// finalizes on HTP (no 1002) and runs genuine u16 on real silicon; measured vs qdq@CPU_EP on v73 (seed
// 2345): Y = 2.37%, Y_h = 2.52% (peak). That does not beat the u8 mirror's 2.24% despite 256x finer I/O
// quantization -- the per-timestep unrolled recurrence accumulation dominates the drift, so widening I/O
// 8->16 bit barely helps. Intrinsic quant drift, not an EP bug. 3.0% clears the 2.52% peak with headroom
// (mirrors the u8 silicon bound). The linux x86_64 HTP emulator has no faithful native-INT16 Gru kernel:
// this path degenerates to a constant output there (measured -- every mismatching element collapses to
// one value, err/output_range up to ~100%, which no tolerance can bracket), unlike the u8 mirror whose
// emulator INT8 kernel merely drifts (~1.9x its silicon peak and still bounded). So the test is skipped on
// that emulator; real silicon keeps the tight 3.0% bound.
TEST_F(QnnHTPBackendTests, GRU_QDQ_u16_linear_before_reset) {
  // No faithful native INT16 Gru kernel on the x86 HTP emulator (output degenerates to a constant -- see
  // the note above). Validated instead on real silicon; mirrors the x86-sim skips in cast_test.cc and
  // framework_op_trace_test.cc.
  QNN_SKIP_TEST_ON_LINUX_X86_64("native INT16 Gru kernel unsupported on linux x86_64 HTP emulator; requires real device.");
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  constexpr float kTolerance = 0.03f;
  RunHtpQDQGRUOpTest<uint16_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::ref(B_def),                                                                         // B
                               std::ref(H_def),                                                                         // initial_h
                               true,                                                                                    // has_Y
                               true,                                                                                    // has_Y_h
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All,
                               1,  // linear_before_reset
                               QDQTolerance(kTolerance));
}

// u16 QDQ GRU, bidirectional -> fp-degraded by the builder: HTP's native quantized Gru is forward-only,
// so !is_forward triggers fp-degrade regardless of dtype (see GRU_QDQ_sanity_forward). This is the only
// test that exercises the u16 fp-degrade boundary (builder-inserted AddDequantizeNode u16->fp32 /
// AddQuantizeNode fp32->u16); the native-u16 tests run genuine u16 and never insert those. Unlike them it
// needs no native-INT16 kernel (fp32 GRU + standard u16 Q/DQ), so it is NOT skipped on the x86 emulator
// and is the only live u16 GRU path there. Default tolerance suffices: the u16 input quant is shared with
// the CPU reference and cancels, leaving only the fine u16 output requantization.
TEST_F(QnnHTPBackendTests, GRU_QDQ_u16_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint16_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::ref(B_def),                                                                         // B
                               std::ref(H_def),                                                                         // initial_h
                               true,                                                                                    // has_Y
                               true,                                                                                    // has_Y_h
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All);
}

// Y-only GRU (Y_h absent), bidirectional. The structural-only selector folds the group even with an
// absent optional output; the builder then fp-degrades it. Two triggers fire here -- missing-output AND
// non-forward direction -- so this does NOT isolate the missing-output trigger (that is
// GRU_QDQ_Y_h_only_forward). A Y-only u8 fold was itself fine (~0.75% on v73); the Y_h-only mirror
// drifted (see below), which is why missing-output fp-degrades rather than folding to genuine u8.
TEST_F(QnnHTPBackendTests, GRU_QDQ_Y_only_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              false,                                                                                   // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All,
                              1);  // LBR=1 (bidirectional still fp-degrades via non-forward direction + missing-output)
}

// Y_h-only mirror of GRU_QDQ_Y_only_bidirectional, bidirectional. The selector folds the group; the
// builder fp-degrades it (missing-output AND non-forward direction both fire, so this does NOT isolate
// the missing-output trigger -- see GRU_QDQ_Y_h_only_forward). As a genuine-u8 fold this drifted ~8.8%
// on v73 (HTP requantizes the per-step recurrence at Y_h's tight final-step scale), which is why
// missing-output fp-degrades instead of folding to genuine u8.
TEST_F(QnnHTPBackendTests, GRU_QDQ_Y_h_only_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              false,                                                                                   // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All,
                              1);  // LBR=1 (bidirectional still fp-degrades via non-forward direction + missing-output)
}

// Y_h-only GRU (Y absent), forward, LBR=1, genuine-u8 dtype -- the isolation test for the missing-output
// fp-degrade trigger. LBR=1 and forward are both non-triggers and the dtype is genuine-u8, so missing-output
// is the ONLY term that makes use_fp_fallback fire (the bidirectional GRU_QDQ_Y*_only tests can't isolate it
// because non-forward direction fires too). The builder fp-degrades -- Dequantize -> fp32 GRU -> Quantize,
// all on QNN, so assignment stays All and the compute is fp32 -- it should clear the tight default 0.4%
// tolerance. A genuine-u8 Y_h-only fold instead drifted ~8.8% on v73 (HTP requantizes the per-step
// recurrence at Y_h's tight final-step scale); fp-degrading it is exactly why missing-output is a trigger.
TEST_F(QnnHTPBackendTests, GRU_QDQ_Y_h_only_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              false,                                                                                   // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All,
                              1);  // LBR=1: forward + genuine-u8, so missing-output is the sole fp-degrade trigger
}

// layout=1: ORT CPU EP does not support batchwise layout, so session initialization throws.
// Verify the expected failure so the test actively guards against silent regressions.
// On Linux aarch64 the test framework skips unsupported ops without throwing, so no EXPECT_THROW there.
TEST_F(QnnHTPBackendTests, GRU_QDQ_layout1_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  // layout=1: X [batch, seq, input], initial_h [batch, num_directions, hidden]
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({batch_size, num_direction, hidden_size}, false, -1.0f, 1.0f);
#if defined(__linux__) && defined(__aarch64__)
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({batch_size, seq_len, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              1,                                                                                       // layout
                              ExpectedEPNodeAssignment::None);
#else
  EXPECT_THROW(
      RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({batch_size, seq_len, input_size}, false, -1.0f, 1.0f),              // X
                                  TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                                  TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                                  std::ref(B_def),                                                                         // B
                                  std::ref(H_def),                                                                         // initial_h
                                  true,                                                                                    // has_Y
                                  true,                                                                                    // has_Y_h
                                  direction,                                                                               // direction
                                  hidden_size,                                                                             // hidden_size
                                  1,                                                                                       // layout
                                  ExpectedEPNodeAssignment::None),
      std::exception);
#endif
}

// linear_before_reset=1: the config customer models use. Unlike LBR=0 it finalizes on HTP (no u8 ->
// QUint16Crouton widening, no 1002) and runs genuine u8 on real silicon. The genuine INT8 Gru config's
// spec bias is int32 (SFIXED_POINT_32) -- HTP has no u8-bias config -- so the bias is int32 here; a u8
// bias would fp-degrade instead. Tolerance relaxed 0.4% -> 3.0% because the per-timestep unrolled
// recurrence accumulates u8 quant drift; measured peak vs qdq@CPU_EP = 2.24% (Y) on v81, seed 2345
// (measured with a u8 bias; int32's huge range makes the bias quant error negligible, so the drift is
// dominated by the u8 X/W/R recurrence and tracks that 2.24%). Intrinsic quant drift, not an EP bug;
// 3.0% clears 2.24% with headroom. The linux x86_64 HTP emulator's u8 GRU kernel is not bit-accurate to
// silicon and drifts further (observed peak ~4.29% vs f32@CPU_EP), so relax to 6.0% there while keeping
// the tight 3.0% bound on real silicon. TODO: Remove the platform-aware tolerance once the emulator u8
// kernel matches silicon.
TEST_F(QnnHTPBackendTests, GRU_QDQ_linear_before_reset) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
#if defined(__linux__) && defined(__x86_64__)
  constexpr float kTolerance = 0.06f;
#else
  constexpr float kTolerance = 0.03f;
#endif
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All,
                              1,  // linear_before_reset
                              QDQTolerance(kTolerance));
}

// Boundary guard for the u8-bias fp-degrade decision. HTP's INT8 Gru config takes an int32
// (SFIXED_POINT_32) bias only -- a u8 bias is off-spec, so genuine_u8_combo rejects it and this
// otherwise-genuine shape (u8 X/W/R, LBR=1, forward, both outputs) fp-degrades (Dequantize -> fp32 GRU
// -> Quantize). fp-degrade is accurate, so it passes at the tight default (~0.4%) tolerance; if a u8
// bias were ever (re)accepted as genuine, the u8 recurrence would drift ~2.24% (see
// GRU_QDQ_linear_before_reset) and blow this bound -- so a green run here proves the u8 bias fp-degraded.
TEST_F(QnnHTPBackendTests, GRU_QDQ_u8_bias_fp_degrade) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                              TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                              TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                              std::ref(B_def),                                                                         // B
                              std::ref(H_def),                                                                         // initial_h
                              true,                                                                                    // has_Y
                              true,                                                                                    // has_Y_h
                              direction,                                                                               // direction
                              hidden_size,                                                                             // hidden_size
                              0,                                                                                       // layout
                              ExpectedEPNodeAssignment::All,
                              1,                      // linear_before_reset
                              QDQTolerance(),         // tight default (~0.4%): fp-degrade is accurate
                              22,                     // opset
                              /*int32_bias=*/false);  // u8 bias -> off-spec -> fp-degrade
}

// ============================================================
// HTP FP16 Tests
// ============================================================

TEST_F(QnnHTPBackendTests, GRU_Fp16_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.04f);
}

TEST_F(QnnHTPBackendTests, GRU_Fp16_sanity_reverse) {
  std::string direction = "reverse";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  // Linux x86_64 accumulates larger FP16 rounding error in the reverse unroll
  // (observed: Y max-rel ~0.14, Y_h max-rel ~0.008).
  // TODO: Remove the platform-aware tolerance once the accuracy issue on Linux x86_64 is solved
#if defined(__linux__) && defined(__x86_64__)
  constexpr float kTolerance = 0.15f;
#else
  constexpr float kTolerance = 0.006f;
#endif
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      kTolerance);
}

TEST_F(QnnHTPBackendTests, GRU_Fp16_sanity_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.02f);
}

TEST_F(QnnHTPBackendTests, GRU_Fp16_sanity_bidirectional_wo_B) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::nullopt,                                                                            // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.03f);
}

TEST_F(QnnHTPBackendTests, GRU_Fp16_sanity_bidirectional_wo_H) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::nullopt,                                                                            // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.04f);
}

TEST_F(QnnHTPBackendTests, GRU_Fp16_sanity_bidirectional_all_initializer) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, true, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, true, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, true, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                        // B
                      std::ref(H_def),                                                                        // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.02f);
}

// Y-only (has_Y=true, has_Y_h=false) — exercises the bidirectional Concat path for Y
TEST_F(QnnHTPBackendTests, GRU_Fp16_Y_only_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      false,                                                                                   // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.02f);
}

// Y_h-only (has_Y=false, has_Y_h=true) — exercises the bidirectional Concat path for Y_h
TEST_F(QnnHTPBackendTests, GRU_Fp16_Y_h_only_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      false,                                                                                   // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.02f);
}

// layout=1: ORT CPU EP does not support batchwise layout, so session initialization throws.
// Verify the expected failure so the test actively guards against silent regressions.
// On Linux aarch64 the test framework skips unsupported ops without throwing, so no EXPECT_THROW there.
TEST_F(QnnHTPBackendTests, GRU_Fp16_layout1_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  // layout=1: X [batch, seq, input], initial_h [batch, num_directions, hidden]
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({batch_size, num_direction, hidden_size}, false, -1.0f, 1.0f);
#if defined(__linux__) && defined(__aarch64__)
  RunHtpFp16GRUOpTest(TestInputDef<float>({batch_size, seq_len, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      1,                                                                                       // layout
                      ExpectedEPNodeAssignment::None);
#else
  EXPECT_THROW(
      RunHtpFp16GRUOpTest(TestInputDef<float>({batch_size, seq_len, input_size}, false, -1.0f, 1.0f),              // X
                          TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                          TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                          std::ref(B_def),                                                                         // B
                          std::ref(H_def),                                                                         // initial_h
                          true,                                                                                    // has_Y
                          true,                                                                                    // has_Y_h
                          direction,                                                                               // direction
                          hidden_size,                                                                             // hidden_size
                          1,                                                                                       // layout
                          ExpectedEPNodeAssignment::None),
      std::exception);
#endif
}

TEST_F(QnnHTPBackendTests, GRU_Fp16_linear_before_reset) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  // Linux x86_64 accumulates larger FP16 rounding error with linear_before_reset
  // (observed: Y max-rel ~0.204, Y_h max-rel ~0.013).
  // TODO: Remove the platform-aware tolerance once the accuracy issue on Linux x86_64 is solved
#if defined(__linux__) && defined(__x86_64__)
  constexpr float kTolerance = 0.25f;
#else
  constexpr float kTolerance = 0.03f;
#endif
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                      std::ref(B_def),                                                                         // B
                      std::ref(H_def),                                                                         // initial_h
                      true,                                                                                    // has_Y
                      true,                                                                                    // has_Y_h
                      direction,                                                                               // direction
                      hidden_size,                                                                             // hidden_size
                      0,                                                                                       // layout
                      ExpectedEPNodeAssignment::All,
                      1,  // linear_before_reset
                      kTolerance);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
