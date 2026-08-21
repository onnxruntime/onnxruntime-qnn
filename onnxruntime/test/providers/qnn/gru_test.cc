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
                       const std::vector<QuantParams<InputType>>& output_qparams) {
  static constexpr bool kIsFp16 = std::is_same<InputType, Ort::Float16_t>::value;
  static constexpr bool kIsU8 = std::is_same<InputType, uint8_t>::value;

  auto add_input = [&](const char* name, const TestInputDef<float>& def) -> std::string {
    if constexpr (kIsFp16) {
      TestInputDef<Ort::Float16_t> fp16_def = ConvertToFP16InputDef(def);
      MakeTestInput(builder, name, fp16_def);
      return name;
    } else if constexpr (kIsU8) {
      MakeTestInput(builder, name, def);
      QuantParams<uint8_t> qparams = GetTestInputQuantParams<uint8_t>(def);
      return AddQDQNodePair<uint8_t>(builder, std::string("qdq_") + name, name, qparams.scale, qparams.zero_point);
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
    input_names.push_back(add_input("B", B_def->get()));
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
    if constexpr (kIsU8) {
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
  if constexpr (kIsU8) {
    size_t i = 0;
    if (has_Y) {
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_Y", y_out, output_qparams[i].scale,
                                                     output_qparams[i].zero_point);
      ++i;
    }
    if (has_Y_h) {
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_Y_h", y_h_out, output_qparams[i].scale,
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
                                                         const int64_t linear_before_reset = 0) {
  return [X_def, W_def, R_def, B_def, H_def,
          has_Y, has_Y_h,
          direction, hidden_size, layout, linear_before_reset](ModelTestBuilder& builder,
                                                               std::vector<QuantParams<InputQType>>& output_qparams) {
    _BuildGRUTestCase<InputQType>(builder, X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                  direction, hidden_size, layout, linear_before_reset, output_qparams);
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
                               int opset = 22) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildGRUTestCase<float>(X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                               direction, hidden_size, layout, linear_before_reset),
                       BuildQDQGRUTestCase<QuantType>(X_def, W_def, R_def, B_def, H_def, has_Y, has_Y_h,
                                                      direction, hidden_size, layout, linear_before_reset),
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

// u8 QDQ GRU, linear_before_reset=0. OrtGRUNodeGroupSelector declines LBR=0 (HTP can't finalize it:
// the gate matmul is widened u8 -> QUint16Crouton -> Code 1002 on v73/v81), so it runs fp32 + Q/DQ.
// This validates the LBR=0 decline + fp accuracy, not u8 HTP exec (genuine u8 = GRU_QDQ_linear_before_reset).
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
// unroll is not the discriminator -- linear_before_reset=0 is. Declined to fp, same as forward.
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

// LBR=0 u8 QDQ GRU, declined to fp32 + Q/DQ (see GRU_QDQ_sanity_forward). Validates decline, not u8 exec.
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

// LBR=0 u8 QDQ GRU, declined to fp32 + Q/DQ (see GRU_QDQ_sanity_forward). Validates decline, not u8 exec.
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

// LBR=0 u8 QDQ GRU, declined to fp32 + Q/DQ (see GRU_QDQ_sanity_forward). Validates decline, not u8 exec.
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

// LBR=0 u8 QDQ GRU, declined to fp32 + Q/DQ (see GRU_QDQ_sanity_forward). Validates decline, not u8 exec.
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

// LBR=0 u8 QDQ GRU, declined to fp32 + Q/DQ (see GRU_QDQ_sanity_forward). Validates decline, not u8 exec.
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

// u16 QDQ GRU. Declined for two reasons -- LBR=0 (see GRU_QDQ_sanity_forward) and u16 X outside the
// selector's UINT8-only in[0] allowlist -- so it runs fp. fp-path sanity check at u16 quant params.
TEST_F(QnnHTPBackendTests, GRU_QDQ_u16_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
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

// Y-only GRU (Y_h absent). The selector requires BOTH outputs (CheckQDQNodes declines an empty slot),
// so it runs fp. linear_before_reset=1 isolates the missing-output decline from the LBR=0 one. A Y-only
// u8 fold was itself fine (~0.75% on v73); it is the Y_h-only mirror that drifted (see below).
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
                              1);  // LBR=1: isolates the missing-output decline from LBR (fp path)
}

// Y_h-only mirror of GRU_QDQ_Y_only_bidirectional; declined to fp by the same both-outputs rule
// (LBR=1 isolates it from the LBR=0 decline). As a u8 fold this drifted ~8.8% on v73 -- HTP requantizes
// the per-step recurrence at Y_h's tight final-step scale -- which is why missing-output u8 is deferred.
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
                              1);  // LBR=1: isolates the missing-output decline from LBR (fp path)
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
// QUint16Crouton widening, no 1002) and runs genuine u8 on real silicon. Tolerance relaxed 0.4% -> 3.0%
// because the per-timestep unrolled recurrence accumulates u8 quant drift; measured peak vs qdq@CPU_EP
// = 2.24% (Y) on v81, seed 2345. Intrinsic quant drift, not an EP bug; 3.0% clears 2.24% with headroom.
// The linux x86_64 HTP emulator's u8 GRU kernel is not bit-accurate to silicon and drifts further
// (observed peak ~4.29% vs f32@CPU_EP), so relax to 6.0% there while keeping the tight 3.0% bound on
// real silicon. TODO: Remove the platform-aware tolerance once the emulator u8 kernel matches silicon.
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
