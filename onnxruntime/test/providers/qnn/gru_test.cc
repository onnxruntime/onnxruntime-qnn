// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "test/unittest_util/tester_types.h"

#include "core/graph/onnx_protobuf.h"

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

  ORT_UNUSED_PARAMETER(output_qparams);
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
                  expected_ep_assignment,
                  tolerance);
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
  RunCpuFP32GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                      std::ref(B_def),                                                                        // B
                      std::ref(H_def),                                                                        // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
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
  RunCpuFP32GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                      std::ref(B_def),                                                                        // B
                      std::ref(H_def),                                                                        // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
                      ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64)

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

TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                               std::ref(B_def),                                                                        // B
                               std::ref(H_def),                                                                        // initial_h
                               true,                                                                                   // has_Y
                               true,                                                                                   // has_Y_h
                               direction,                                                                              // direction
                               hidden_size,                                                                            // hidden_size
                               0,                                                                                      // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_reverse) {
  std::string direction = "reverse";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                               std::ref(B_def),                                                                        // B
                               std::ref(H_def),                                                                        // initial_h
                               true,                                                                                   // has_Y
                               true,                                                                                   // has_Y_h
                               direction,                                                                              // direction
                               hidden_size,                                                                            // hidden_size
                               0,                                                                                      // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                               std::ref(B_def),                                                                        // B
                               std::ref(H_def),                                                                        // initial_h
                               true,                                                                                   // has_Y
                               true,                                                                                   // has_Y_h
                               direction,                                                                              // direction
                               hidden_size,                                                                            // hidden_size
                               0,                                                                                      // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_bidirectional_wo_B) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                               std::nullopt,                                                                           // B
                               std::ref(H_def),                                                                        // initial_h
                               true,                                                                                   // has_Y
                               true,                                                                                   // has_Y_h
                               direction,                                                                              // direction
                               hidden_size,                                                                            // hidden_size
                               0,                                                                                      // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_bidirectional_wo_H) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                               std::ref(B_def),                                                                        // B
                               std::nullopt,                                                                           // initial_h
                               true,                                                                                   // has_Y
                               true,                                                                                   // has_Y_h
                               direction,                                                                              // direction
                               hidden_size,                                                                            // hidden_size
                               0,                                                                                      // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, GRU_QDQ_sanity_bidirectional_all_initializer) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, true, -0.5f, 0.5f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -0.5f, 0.5f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -0.5f, 0.5f),            // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, true, -0.5f, 0.5f),  // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, true, -0.5f, 0.5f), // R
                               std::ref(B_def),                                                                       // B
                               std::ref(H_def),                                                                       // initial_h
                               true,                                                                                  // has_Y
                               true,                                                                                  // has_Y_h
                               direction,                                                                             // direction
                               hidden_size,                                                                           // hidden_size
                               0,                                                                                     // layout
                               ExpectedEPNodeAssignment::All,
                               0,
                               QDQTolerance(0.008f));
}

TEST_F(QnnHTPBackendTests, GRU_QDQ_linear_before_reset) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQGRUOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                               TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                               TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                               std::ref(B_def),                                                                        // B
                               std::ref(H_def),                                                                        // initial_h
                               true,                                                                                   // has_Y
                               true,                                                                                   // has_Y_h
                               direction,                                                                              // direction
                               hidden_size,                                                                            // hidden_size
                               0,                                                                                      // layout
                               ExpectedEPNodeAssignment::All,
                               1);                                                                                     // linear_before_reset
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
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                      std::ref(B_def),                                                                        // B
                      std::ref(H_def),                                                                        // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
                      ExpectedEPNodeAssignment::All);
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
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                      std::ref(B_def),                                                                        // B
                      std::ref(H_def),                                                                        // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.27f);
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
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                      std::ref(B_def),                                                                        // B
                      std::ref(H_def),                                                                        // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.25f);
}

TEST_F(QnnHTPBackendTests, GRU_Fp16_sanity_bidirectional_wo_B) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                      std::nullopt,                                                                           // B
                      std::ref(H_def),                                                                        // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.07f);
}

TEST_F(QnnHTPBackendTests, GRU_Fp16_sanity_bidirectional_wo_H) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                      std::ref(B_def),                                                                        // B
                      std::nullopt,                                                                           // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.035f);
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
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),            // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, true, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, true, -1.0f, 1.0f), // R
                      std::ref(B_def),                                                                       // B
                      std::ref(H_def),                                                                       // initial_h
                      true,                                                                                  // has_Y
                      true,                                                                                  // has_Y_h
                      direction,                                                                             // direction
                      hidden_size,                                                                           // hidden_size
                      0,                                                                                     // layout
                      ExpectedEPNodeAssignment::All,
                      0,
                      0.14f);
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
  RunHtpFp16GRUOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
                      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),  // W
                      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f), // R
                      std::ref(B_def),                                                                        // B
                      std::ref(H_def),                                                                        // initial_h
                      true,                                                                                   // has_Y
                      true,                                                                                   // has_Y_h
                      direction,                                                                              // direction
                      hidden_size,                                                                            // hidden_size
                      0,                                                                                      // layout
                      ExpectedEPNodeAssignment::All,
                      1);                                                                                     // linear_before_reset
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
