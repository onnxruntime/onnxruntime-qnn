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
  qti_aisw StatefulLstm inputs (standard ONNX LSTM signature + trailing reset):
  in[0]: X [seq_length, batch_size, input_size]
  in[1]: W [num_directions, 4*hidden_size, input_size]
  in[2]: R [num_directions, 4*hidden_size, hidden_size]
  in[3]: B [num_directions, 8*hidden_size]            (optional)
  in[4]: sequence_lens                                 (unused / empty)
  in[5]: initial_h [num_directions, batch_size, hidden_size] (optional)
  in[6]: initial_c [num_directions, batch_size, hidden_size] (optional)
  in[7]: P [num_directions, 3*hidden_size]             (optional)
  in[8]: reset (BOOL scalar)                           (optional, stateful extension)

  qti_aisw StatefulLstm optional outputs:
  out[0]: Y [seq_length, num_directions, batch_size, hidden_size]
  out[1]: Y_h [num_directions, batch_size, hidden_size]
  out[2]: Y_c [num_directions, batch_size, hidden_size]

  HTP constraints:
  - Bidirectional is supported for all dtypes (float and quantized); HtpOpDefSupplement places
    no direction constraint on the Lstm op.

  QNN EP maps this to QNN_OP_LSTM; reset ONNX in[8] -> QNN LSTM in[24].
*/

template <typename InputType>
void _BuildStatefulLSTMTestCase(ModelTestBuilder& builder,
                                const TestInputDef<float>& X_def,
                                const TestInputDef<float>& W_def,
                                const TestInputDef<float>& R_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
                                const bool has_Y,
                                const bool has_Y_h,
                                const bool has_Y_c,
                                const std::string direction,
                                const int64_t hidden_size,
                                const std::vector<QuantParams<InputType>>& output_qparams) {
  static constexpr bool kIsFp16 = std::is_same<InputType, Ort::Float16_t>::value;
  static constexpr bool kIsU8 = std::is_same<InputType, uint8_t>::value;
  static constexpr bool kIsU16 = std::is_same<InputType, uint16_t>::value;

  auto add_input = [&](const char* name, const TestInputDef<float>& def) -> std::string {
    if constexpr (kIsFp16) {
      TestInputDef<Ort::Float16_t> fp16_def = ConvertToFP16InputDef(def);
      MakeTestInput(builder, name, fp16_def);
      return name;
    } else if constexpr (kIsU8) {
      MakeTestInput(builder, name, def);
      QuantParams<uint8_t> qparams = GetTestInputQuantParams<uint8_t>(def);
      return AddQDQNodePair<uint8_t>(builder, std::string("qdq_") + name, name, qparams.scale, qparams.zero_point);
    } else if constexpr (kIsU16) {
      MakeTestInput(builder, name, def);
      QuantParams<uint16_t> qparams = GetTestInputQuantParams<uint16_t>(def);
      return AddQDQNodePair<uint16_t>(builder, std::string("qdq_") + name, name, qparams.scale, qparams.zero_point);
    } else {
      MakeTestInput(builder, name, def);
      return name;
    }
  };

  // Required inputs
  const std::string x_name = add_input("X", X_def);
  const std::string w_name = add_input("W", W_def);
  const std::string r_name = add_input("R", R_def);

  // Optional inputs (positional — represent absent values with empty string)
  std::vector<std::string> input_names;
  input_names.reserve(9);
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

  // initial_c
  if (C_def) {
    input_names.push_back(add_input("initial_c", C_def->get()));
  } else {
    input_names.push_back("");
  }

  // P (peephole weights, optional)
  if (P_def) {
    input_names.push_back(add_input("P", P_def->get()));
  } else {
    input_names.push_back("");
  }

  // reset (in[8], BOOL scalar — leave absent in unit tests; the builder handles null_tensor)
  // Omitting it (no push_back for in[8]) means the ONNX node has 8 inputs, which is valid.

  // Compute output shapes from input and attributes.
  const int64_t num_dir = (direction == "bidirectional") ? 2 : 1;
  const int64_t seq_len = X_def.GetShape()[0];
  const int64_t batch_size_dim = X_def.GetShape()[1];

  // Outputs — declare typed+shaped outputs so QNN EP's GetTensorInfo succeeds during
  // ComposeGraph on real HTP hardware (where shape inference does not supply Y_h/Y_c shape).
  // FP16: use Float16_t type. QDQ: declare float32 value_info for the intermediate tensors
  // consumed by Q nodes (GetTensorInfo needs shape on both sides of Q/DQ).
  auto make_output = [&](const char* name, const std::vector<int64_t>& shape) -> std::string {
    if (name == nullptr || name[0] == '\0') return "";
    if constexpr (kIsFp16) {
      builder.MakeOutput<Ort::Float16_t>(name, {shape});
      return name;
    } else if constexpr (kIsU8 || kIsU16) {
      // Declare the intermediate float32 tensor with type+shape in value_info so QNN EP's
      // GetTensorInfo can resolve its shape during ComposeGraph.
      const std::string iname = std::string("lstm_") + name;
      auto* vi = builder.model_.mutable_graph()->add_value_info();
      vi->set_name(iname);
      auto* tt = vi->mutable_type()->mutable_tensor_type();
      tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      auto* sp = tt->mutable_shape();
      for (int64_t d : shape) {
        sp->add_dim()->set_dim_value(d);
      }
      return iname;
    } else {
      builder.MakeOutput<float>(name, {shape});
      return name;
    }
  };

  const std::string y_out = has_Y ? make_output("Y", {seq_len, num_dir, batch_size_dim, hidden_size}) : std::string("");
  const std::string y_h_out = has_Y_h ? make_output("Y_h", {num_dir, batch_size_dim, hidden_size}) : std::string("");
  const std::string y_c_out = has_Y_c ? make_output("Y_c", {num_dir, batch_size_dim, hidden_size}) : std::string("");

  std::vector<std::string> output_names;
  output_names.push_back(y_out);
  output_names.push_back(y_h_out);
  output_names.push_back(y_c_out);

  // Attributes
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(builder.MakeStringAttribute("direction", direction));
  attrs.push_back(builder.MakeScalarAttribute("hidden_size", hidden_size));

  builder.AddNode("slstm", "StatefulLstm", input_names, output_names, kQtiAiswDomain, attrs);

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
    if (has_Y_c) {
      AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_Y_c", y_c_out, output_qparams[i].scale,
                                                     output_qparams[i].zero_point);
      ++i;
    }
  } else if constexpr (kIsU16) {
    size_t i = 0;
    if (has_Y) {
      AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_Y", y_out, output_qparams[i].scale,
                                                      output_qparams[i].zero_point);
      ++i;
    }
    if (has_Y_h) {
      AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_Y_h", y_h_out, output_qparams[i].scale,
                                                      output_qparams[i].zero_point);
      ++i;
    }
    if (has_Y_c) {
      AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_Y_c", y_c_out, output_qparams[i].scale,
                                                      output_qparams[i].zero_point);
      ++i;
    }
  }
}

template <typename InputType>
static GetTestModelFn BuildStatefulLSTMTestCase(const TestInputDef<float>& X_def,
                                                const TestInputDef<float>& W_def,
                                                const TestInputDef<float>& R_def,
                                                const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                                const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                                const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
                                                const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
                                                const bool has_Y,
                                                const bool has_Y_h,
                                                const bool has_Y_c,
                                                const std::string direction,
                                                const int64_t hidden_size) {
  return [X_def, W_def, R_def, B_def, H_def, C_def, P_def,
          has_Y, has_Y_h, has_Y_c, direction, hidden_size](ModelTestBuilder& builder) {
    _BuildStatefulLSTMTestCase<InputType>(builder, X_def, W_def, R_def, B_def, H_def, C_def, P_def,
                                          has_Y, has_Y_h, has_Y_c, direction, hidden_size, {});
  };
}

template <typename InputQType>
static GetTestQDQModelFn<InputQType> BuildQDQStatefulLSTMTestCase(
    const TestInputDef<float>& X_def,
    const TestInputDef<float>& W_def,
    const TestInputDef<float>& R_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
    const bool has_Y,
    const bool has_Y_h,
    const bool has_Y_c,
    const std::string direction,
    const int64_t hidden_size) {
  return [X_def, W_def, R_def, B_def, H_def, C_def, P_def,
          has_Y, has_Y_h, has_Y_c, direction, hidden_size](
             ModelTestBuilder& builder, std::vector<QuantParams<InputQType>>& output_qparams) {
    _BuildStatefulLSTMTestCase<InputQType>(builder, X_def, W_def, R_def, B_def, H_def, C_def, P_def,
                                           has_Y, has_Y_h, has_Y_c, direction, hidden_size, output_qparams);
  };
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Builds the model from a GetTestModelFn and verifies QNN EP node assignment (no inference run).
// StatefulLstm is a qti_aisw custom op with no CPU kernel; RunQnnModelTest's mandatory CPU baseline
// is neither possible nor meaningful. The QNN EP factory supplies the qti_aisw placeholder schema
// so the model loads; we then check EP node assignment.
static void RunQnnOnlyStatefulLSTMModel(const GetTestModelFn& build_test_case,
                                        const ProviderOptions& provider_options,
                                        ExpectedEPNodeAssignment expected_ep_assignment,
                                        int opset) {
  // Positive cases need real HTP hardware to finalize the compiled graph; the x86_64 simulator
  // fails at graph finalize. Stateful ops require persistent state pool allocation — this fails
  // on HTP v68 due to insufficient VTCM. Negative cases only check IsOpSupported.
  if (expected_ep_assignment == ExpectedEPNodeAssignment::All) {
    QNN_SKIP_TEST_ON_LINUX_X86_64("qti_aisw StatefulLstm requires HTP hardware; not supported on the x86_64 simulator.");
    SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  }

  const std::unordered_map<std::string, int> domain_to_version = {
      {"", opset}, {kMSDomain, 1}, {kQtiAiswDomain, 1}};

  ModelTestBuilder helper;
  build_test_case(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  VerifyQnnEpModelAssignment(model_data, "StatefulLSTM_QNN", provider_options, expected_ep_assignment);
}

// Runs a StatefulLstm model on the QNN HTP backend with QDQ quantization.
template <typename QuantType>
static void RunHtpQDQStatefulLSTMOpTest(
    const TestInputDef<float>& X_def,
    const TestInputDef<float>& W_def,
    const TestInputDef<float>& R_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
    const bool has_Y,
    const bool has_Y_h,
    const bool has_Y_c,
    const std::string direction,
    const int64_t hidden_size,
    ExpectedEPNodeAssignment expected_ep_assignment,
    int opset = 21) {  // uint16 QuantizeLinear/DequantizeLinear in the default ONNX domain requires opset >= 21.
  // QDQ models wrap StatefulLstm in DQ/Q nodes that the QNN EP compiles into a QNN subgraph, which
  // needs real NPU hardware.
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // StatefulLstm is a qti_aisw custom op with no CPU kernel; run on QNN EP only.
  // Provide synthetic output qparams (one per active output) derived from X input range
  // to avoid needing a CPU reference session.
  QuantParams<QuantType> out_qp = GetTestInputQuantParams<QuantType>(X_def);
  size_t num_outputs = static_cast<size_t>(has_Y) + static_cast<size_t>(has_Y_h) + static_cast<size_t>(has_Y_c);
  std::vector<QuantParams<QuantType>> out_qparams_vec(num_outputs, out_qp);

  GetTestQDQModelFn<QuantType> qdq_fn = BuildQDQStatefulLSTMTestCase<QuantType>(
      X_def, W_def, R_def, B_def, H_def, C_def, P_def,
      has_Y, has_Y_h, has_Y_c, direction, hidden_size);

  GetTestModelFn model_fn = [qdq_fn, &out_qparams_vec](ModelTestBuilder& builder) {
    qdq_fn(builder, out_qparams_vec);
  };

  RunQnnOnlyStatefulLSTMModel(model_fn, provider_options, expected_ep_assignment, opset);
}

// Runs a StatefulLstm model on the QNN HTP backend with FP16 precision.
static void RunHtpFp16StatefulLSTMOpTest(
    const TestInputDef<float>& X_def,
    const TestInputDef<float>& W_def,
    const TestInputDef<float>& R_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
    const bool has_Y,
    const bool has_Y_h,
    const bool has_Y_c,
    const std::string direction,
    const int64_t hidden_size,
    ExpectedEPNodeAssignment expected_ep_assignment,
    int opset = 13) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  // StatefulLstm is a qti_aisw custom op with no CPU kernel; run on QNN EP only.
  RunQnnOnlyStatefulLSTMModel(BuildStatefulLSTMTestCase<Ort::Float16_t>(X_def, W_def, R_def, B_def, H_def, C_def, P_def,
                                                                        has_Y, has_Y_h, has_Y_c, direction, hidden_size),
                              provider_options,
                              expected_ep_assignment,
                              opset);
}

// ============================================================
// HTP FP16 Tests
// ============================================================

TEST_F(QnnHTPBackendTests, StatefulLSTM_Fp16_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16StatefulLSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      true,                                                                                    // has_Y_c
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, StatefulLSTM_Fp16_sanity_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16StatefulLSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      true,                                                                                    // has_Y_c
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, StatefulLSTM_Fp16_sanity_forward_wo_B) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16StatefulLSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::nullopt,                                                                            // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      true,                                                                                    // has_Y_c
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

// ============================================================
// HTP QDQ Tests (INT8 / INT16)
// ============================================================

TEST_F(QnnHTPBackendTests, StatefulLSTM_QDQ_u8_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQStatefulLSTMOpTest<uint8_t>(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      true,                                                                                    // has_Y_c
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

// INT16 forward (HTP supports INT16 for forward direction)
TEST_F(QnnHTPBackendTests, StatefulLSTM_QDQ_u16_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQStatefulLSTMOpTest<uint16_t>(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      true,                                                                                    // has_Y_c
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

// ============================================================
// Bidirectional quantized (INT8/INT16): accepted by QNN EP.
// HtpOpDefSupplement places no direction constraint on the Lstm op, so quantized
// bidirectional is supported.
// ============================================================

// Bidirectional INT8 is supported by QNN EP (HtpOpDefSupplement places no direction constraint on
// Lstm; the builder emits forward + reverse unroll + Concat for quantized inputs too).
TEST_F(QnnHTPBackendTests, StatefulLSTM_QDQ_u8_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQStatefulLSTMOpTest<uint8_t>(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      true,                                                                                    // has_Y_c
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

// Bidirectional INT16 is supported by QNN EP (see the INT8 case above).
TEST_F(QnnHTPBackendTests, StatefulLSTM_QDQ_u16_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQStatefulLSTMOpTest<uint16_t>(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      true,                                                                                    // has_Y_c
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

// ============================================================
// 6-activation bidirectional pattern must be accepted by QNN EP.
// The bidirectional branch validates the backward activations at indices [3],[4],[5]
// (sigmoid/tanh/tanh); the correct forward+backward default pattern is
// sigmoid/tanh/tanh/sigmoid/tanh/tanh. Note: because every activation here is a
// default value, this is a positive acceptance test, not a kill-test fence — it does
// not by itself distinguish the [4]-vs-[5] index bug (both indices hold "tanh").
// ============================================================

// Helper builder that includes an explicit activations attribute.
static GetTestModelFn BuildStatefulLSTMWithActivations(
    const TestInputDef<float>& X_def,
    const TestInputDef<float>& W_def,
    const TestInputDef<float>& R_def,
    const bool has_Y,
    const bool has_Y_h,
    const bool has_Y_c,
    const std::string direction,
    const int64_t hidden_size,
    const std::vector<std::string>& activations) {
  return [X_def, W_def, R_def, has_Y, has_Y_h, has_Y_c,
          direction, hidden_size, activations](ModelTestBuilder& builder) {
    MakeTestInput(builder, "X", X_def);
    MakeTestInput(builder, "W", W_def);
    MakeTestInput(builder, "R", R_def);

    std::vector<std::string> input_names = {"X", "W", "R"};

    const int64_t seq_len_d = X_def.GetShape()[0];
    const int64_t batch_d = X_def.GetShape()[1];
    const int64_t num_dir_d = (direction == "bidirectional") ? 2 : 1;
    builder.MakeOutput<float>("Y", {{{seq_len_d, num_dir_d, batch_d, hidden_size}}});
    builder.MakeOutput<float>("Y_h", {{{num_dir_d, batch_d, hidden_size}}});
    builder.MakeOutput<float>("Y_c", {{{num_dir_d, batch_d, hidden_size}}});

    std::vector<std::string> output_names = {"Y", "Y_h", "Y_c"};

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
    attrs.push_back(builder.MakeStringAttribute("direction", direction));
    attrs.push_back(builder.MakeScalarAttribute("hidden_size", hidden_size));
    attrs.push_back(MakeAttribute("activations", gsl::span<const std::string>(activations)));

    QNN_TEST_UNUSED_PARAMETER(has_Y);
    QNN_TEST_UNUSED_PARAMETER(has_Y_h);
    QNN_TEST_UNUSED_PARAMETER(has_Y_c);

    builder.AddNode("slstm_act", "StatefulLstm", input_names, output_names,
                    kQtiAiswDomain, attrs);
  };
}

TEST_F(QnnHTPBackendTests, StatefulLSTM_Fp16_bidir_6activations_accepted) {
  // Bidirectional pattern with 6 activations: sigmoid/tanh/tanh (forward) + sigmoid/tanh/tanh (backward).
  // The builder compares activation names against lowercase "sigmoid"/"tanh", so the attribute values
  // must be lowercase for IsOpSupported to accept the node.
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnOnlyStatefulLSTMModel(
      BuildStatefulLSTMWithActivations(
          TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
          TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
          TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
          true, true, true,                                                                        // has_Y, has_Y_h, has_Y_c
          direction, hidden_size,
          {"sigmoid", "tanh", "tanh", "sigmoid", "tanh", "tanh"}),
      provider_options,
      ExpectedEPNodeAssignment::All,
      13);
}

// ============================================================
// input_forget=1 must not be captured (IsOpSupported returns false).
// ============================================================

TEST_F(QnnHTPBackendTests, StatefulLSTM_Fp16_input_forget_Unsupported) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  uint32_t batch_size = 1, hidden_size = 4, input_size = 3, seq_len = 5;
  GetTestModelFn model_fn = [=](ModelTestBuilder& builder) {
    MakeTestInput(builder, "X", TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f));
    MakeTestInput(builder, "W", TestInputDef<float>({1, 4 * hidden_size, input_size}, false, -1.0f, 1.0f));
    MakeTestInput(builder, "R", TestInputDef<float>({1, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f));
    builder.MakeOutput("Y");

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
        builder.MakeStringAttribute("direction", "forward"),
        builder.MakeScalarAttribute("hidden_size", static_cast<int64_t>(hidden_size)),
        builder.MakeScalarAttribute("input_forget", static_cast<int64_t>(1))};
    builder.AddNode("slstm", "StatefulLstm", {"X", "W", "R"}, {"Y"}, kQtiAiswDomain, attrs);
  };

  RunQnnOnlyStatefulLSTMModel(model_fn, provider_options, ExpectedEPNodeAssignment::None, /*opset=*/13);
}

// ============================================================
// Reset-input tests: verify that the reset slot (ONNX in[8] -> QNN in[24]) is wired correctly.
// Tests leave reset=false (do not reset state) and only verify EP node assignment, since the
// stateful ops cannot execute on the x86 HTP emulator.
// ============================================================

// Builds a minimal StatefulLstm model with the reset input present as a BOOL false initializer.
static GetTestModelFn BuildStatefulLSTMWithResetCase(uint32_t seq_len, uint32_t batch_size,
                                                     uint32_t input_size, uint32_t hidden_size) {
  return [seq_len, batch_size, input_size, hidden_size](ModelTestBuilder& builder) {
    MakeTestInput(builder, "X", TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f));
    MakeTestInput(builder, "W", TestInputDef<float>({1, 4 * hidden_size, input_size}, false, -1.0f, 1.0f));
    MakeTestInput(builder, "R", TestInputDef<float>({1, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f));

    // in[0..7]: X, W, R, empty (B), empty (sequence_lens), empty (initial_h), empty (initial_c), empty (P)
    std::vector<std::string> input_names = {"X", "W", "R", "", "", "", "", ""};

    // in[8]: reset — BOOL scalar false. Exercises qnn_lstm_input_names[kQnnLstmResetInputIndex].
    builder.MakeInitializerBool("slstm_reset", {}, {false});
    input_names.push_back("slstm_reset");

    builder.MakeOutput<float>("Y", {{{static_cast<int64_t>(seq_len), 1LL, static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}}});
    builder.MakeOutput<float>("Y_h", {{{1LL, static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}}});
    builder.MakeOutput<float>("Y_c", {{{1LL, static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}}});

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
        builder.MakeStringAttribute("direction", "forward"),
        builder.MakeScalarAttribute("hidden_size", static_cast<int64_t>(hidden_size))};
    builder.AddNode("slstm_with_reset", "StatefulLstm", input_names, {"Y", "Y_h", "Y_c"}, kQtiAiswDomain, attrs);
  };
}

TEST_F(QnnHTPBackendTests, StatefulLSTM_Fp16_with_reset) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  RunQnnOnlyStatefulLSTMModel(BuildStatefulLSTMWithResetCase(5, 1, 3, 4),
                              provider_options, ExpectedEPNodeAssignment::All, 13);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
