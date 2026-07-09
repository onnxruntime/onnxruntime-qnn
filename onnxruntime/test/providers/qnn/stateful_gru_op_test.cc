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
  qti_aisw StatefulGru inputs (standard ONNX GRU signature + trailing reset):
  in[0]: X [seq_length, batch_size, input_size]
  in[1]: W [num_directions, 3*hidden_size, input_size]
  in[2]: R [num_directions, 3*hidden_size, hidden_size]
  in[3]: B [num_directions, 6*hidden_size]            (optional)
  in[4]: sequence_lens                                 (unused / empty)
  in[5]: initial_h [num_directions, batch_size, hidden_size] (optional)
  in[6]: reset (BOOL scalar)                           (optional, stateful extension)

  qti_aisw StatefulGru optional outputs:
  out[0]: Y [seq_length, num_directions, batch_size, hidden_size]
  out[1]: Y_h [num_directions, batch_size, hidden_size]

  HTP constraints:
  - Bidirectional is supported for FP16/FP32; quantized (INT8/INT16) bidirectional is rejected
    (IsOpSupported returns false), per HtpOpDefSupplement (Gru forward-only for INT8/INT16).
  - QNN EP maps this to QNN_OP_GRU; reset ONNX in[6] -> QNN GRU in[14].
*/

template <typename InputType>
void _BuildStatefulGRUTestCase(ModelTestBuilder& builder,
                               const TestInputDef<float>& X_def,
                               const TestInputDef<float>& W_def,
                               const TestInputDef<float>& R_def,
                               const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                               const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                               const bool has_Y,
                               const bool has_Y_h,
                               const std::string direction,
                               const int64_t hidden_size,
                               const int64_t linear_before_reset,
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

  // Optional inputs (positional)
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

  // reset (in[6], BOOL scalar — leave absent; the builder handles null_tensor)

  // Compute output shapes from input and attributes.
  const int64_t num_dir = (direction == "bidirectional") ? 2 : 1;
  const int64_t seq_len = X_def.GetShape()[0];
  const int64_t batch_size_dim = X_def.GetShape()[1];

  // Outputs — declare typed+shaped outputs so QNN EP's GetTensorInfo succeeds during
  // ComposeGraph on real HTP hardware (where shape inference does not supply Y_h shape).
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
      const std::string iname = std::string("gru_") + name;
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

  std::vector<std::string> output_names;
  output_names.push_back(y_out);
  output_names.push_back(y_h_out);

  // Attributes
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(builder.MakeStringAttribute("direction", direction));
  attrs.push_back(builder.MakeScalarAttribute("hidden_size", hidden_size));
  attrs.push_back(builder.MakeScalarAttribute("linear_before_reset", linear_before_reset));

  builder.AddNode("sgru", "StatefulGru", input_names, output_names, kQtiAiswDomain, attrs);

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
  }
}

template <typename InputType>
static GetTestModelFn BuildStatefulGRUTestCase(const TestInputDef<float>& X_def,
                                               const TestInputDef<float>& W_def,
                                               const TestInputDef<float>& R_def,
                                               const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                               const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                               const bool has_Y,
                                               const bool has_Y_h,
                                               const std::string direction,
                                               const int64_t hidden_size,
                                               const int64_t linear_before_reset = 0) {
  return [X_def, W_def, R_def, B_def, H_def,
          has_Y, has_Y_h, direction, hidden_size, linear_before_reset](ModelTestBuilder& builder) {
    _BuildStatefulGRUTestCase<InputType>(builder, X_def, W_def, R_def, B_def, H_def,
                                         has_Y, has_Y_h, direction, hidden_size, linear_before_reset, {});
  };
}

template <typename InputQType>
static GetTestQDQModelFn<InputQType> BuildQDQStatefulGRUTestCase(
    const TestInputDef<float>& X_def,
    const TestInputDef<float>& W_def,
    const TestInputDef<float>& R_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
    const bool has_Y,
    const bool has_Y_h,
    const std::string direction,
    const int64_t hidden_size,
    const int64_t linear_before_reset = 0) {
  return [X_def, W_def, R_def, B_def, H_def,
          has_Y, has_Y_h, direction, hidden_size, linear_before_reset](
             ModelTestBuilder& builder, std::vector<QuantParams<InputQType>>& output_qparams) {
    _BuildStatefulGRUTestCase<InputQType>(builder, X_def, W_def, R_def, B_def, H_def,
                                          has_Y, has_Y_h, direction, hidden_size, linear_before_reset,
                                          output_qparams);
  };
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Builds the model from a GetTestModelFn and verifies QNN EP node assignment (no inference run).
// StatefulGru is a qti_aisw custom op with no CPU kernel; RunQnnModelTest's mandatory CPU baseline
// is neither possible nor meaningful. The QNN EP factory supplies the qti_aisw placeholder schema
// so the model loads; we then check EP node assignment.
static void RunQnnOnlyStatefulGRUModel(const GetTestModelFn& build_test_case,
                                       const ProviderOptions& provider_options,
                                       ExpectedEPNodeAssignment expected_ep_assignment,
                                       int opset) {
  // Positive cases need real HTP hardware to finalize the compiled graph; the x86_64 simulator
  // fails at graph finalize. Stateful ops require persistent state pool allocation — this fails
  // on HTP v68 due to insufficient VTCM. Negative cases only check IsOpSupported.
  if (expected_ep_assignment == ExpectedEPNodeAssignment::All) {
    QNN_SKIP_TEST_ON_LINUX_X86_64("qti_aisw StatefulGru requires HTP hardware; not supported on the x86_64 simulator.");
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

  VerifyQnnEpModelAssignment(model_data, "StatefulGRU_QNN", provider_options, expected_ep_assignment);
}

// Runs a StatefulGru model on the QNN HTP backend with QDQ quantization.
template <typename QuantType>
static void RunHtpQDQStatefulGRUOpTest(
    const TestInputDef<float>& X_def,
    const TestInputDef<float>& W_def,
    const TestInputDef<float>& R_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
    const bool has_Y,
    const bool has_Y_h,
    const std::string direction,
    const int64_t hidden_size,
    ExpectedEPNodeAssignment expected_ep_assignment,
    const int64_t linear_before_reset = 0,
    int opset = 21) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  QuantParams<QuantType> out_qp = GetTestInputQuantParams<QuantType>(X_def);
  size_t num_outputs = static_cast<size_t>(has_Y) + static_cast<size_t>(has_Y_h);
  std::vector<QuantParams<QuantType>> out_qparams_vec(num_outputs, out_qp);

  GetTestQDQModelFn<QuantType> qdq_fn = BuildQDQStatefulGRUTestCase<QuantType>(
      X_def, W_def, R_def, B_def, H_def,
      has_Y, has_Y_h, direction, hidden_size, linear_before_reset);

  GetTestModelFn model_fn = [qdq_fn, &out_qparams_vec](ModelTestBuilder& builder) {
    qdq_fn(builder, out_qparams_vec);
  };

  RunQnnOnlyStatefulGRUModel(model_fn, provider_options, expected_ep_assignment, opset);
}

// Runs a StatefulGru model on the QNN HTP backend with FP16 precision.
static void RunHtpFp16StatefulGRUOpTest(
    const TestInputDef<float>& X_def,
    const TestInputDef<float>& W_def,
    const TestInputDef<float>& R_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
    const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
    const bool has_Y,
    const bool has_Y_h,
    const std::string direction,
    const int64_t hidden_size,
    ExpectedEPNodeAssignment expected_ep_assignment,
    const int64_t linear_before_reset = 0,
    int opset = 21) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnOnlyStatefulGRUModel(BuildStatefulGRUTestCase<Ort::Float16_t>(X_def, W_def, R_def, B_def, H_def,
                                                                      has_Y, has_Y_h, direction, hidden_size, linear_before_reset),
                             provider_options,
                             expected_ep_assignment,
                             opset);
}

// ============================================================
// HTP FP16 Tests
// StatefulGRU positive tests are prefixed DISABLED_ because QNN_COMMON_ERROR_MEM_ALLOC
// at graph finalization affects all known HTP devices (v68, v73, v81). Re-enable once
// QNN HTP firmware adds persistent-state support for StatefulGRU.
// ============================================================

TEST_F(QnnHTPBackendTests, DISABLED_StatefulGRU_Fp16_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16StatefulGRUOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All,
      0);
}

TEST_F(QnnHTPBackendTests, DISABLED_StatefulGRU_Fp16_sanity_forward_wo_B) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16StatefulGRUOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::nullopt,                                                                            // B
      std::ref(H_def),                                                                         // initial_h
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All,
      0);
}

// ============================================================
// HTP QDQ Tests (INT8 — forward only, per HTP constraint)
// ============================================================

TEST_F(QnnHTPBackendTests, DISABLED_StatefulGRU_QDQ_u8_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQStatefulGRUOpTest<uint8_t>(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, DISABLED_StatefulGRU_QDQ_u16_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQStatefulGRUOpTest<uint16_t>(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

// ============================================================
// Bidirectional: FP16/FP32 is supported (accepted by QNN EP); quantized (INT8/INT16)
// bidirectional is rejected, per HtpOpDefSupplement (Gru forward-only for INT8/INT16).
// ============================================================

TEST_F(QnnHTPBackendTests, DISABLED_StatefulGRU_Fp16_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 6 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16StatefulGRUOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 3 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      true,                                                                                    // has_Y
      true,                                                                                    // has_Y_h
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      ExpectedEPNodeAssignment::All);
}

// ============================================================
// Reset-input tests: verify that the reset slot (ONNX in[6] -> QNN in[14]) is wired correctly.
// Tests leave reset=false (do not reset state) and only verify EP node assignment, since the
// stateful ops cannot execute on the x86 HTP emulator.
// ============================================================

// Builds a minimal StatefulGru model with the reset input present as a BOOL false initializer.
static GetTestModelFn BuildStatefulGRUWithResetCase(uint32_t seq_len, uint32_t batch_size,
                                                    uint32_t input_size, uint32_t hidden_size) {
  return [seq_len, batch_size, input_size, hidden_size](ModelTestBuilder& builder) {
    MakeTestInput(builder, "X", TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f));
    MakeTestInput(builder, "W", TestInputDef<float>({1, 3 * hidden_size, input_size}, false, -1.0f, 1.0f));
    MakeTestInput(builder, "R", TestInputDef<float>({1, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f));

    // in[0..5]: X, W, R, empty (B), empty (sequence_lens), empty (initial_h)
    std::vector<std::string> input_names = {"X", "W", "R", "", "", ""};

    // in[6]: reset — BOOL scalar false. Exercises qnn_gru_input_names[kQnnGruResetInputIndex].
    builder.MakeInitializerBool("sgru_reset", {}, {false});
    input_names.push_back("sgru_reset");

    builder.MakeOutput<float>("Y", {{{static_cast<int64_t>(seq_len), 1LL, static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}}});
    builder.MakeOutput<float>("Y_h", {{{1LL, static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}}});

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
        builder.MakeStringAttribute("direction", "forward"),
        builder.MakeScalarAttribute("hidden_size", static_cast<int64_t>(hidden_size)),
        builder.MakeScalarAttribute("linear_before_reset", static_cast<int64_t>(0))};
    builder.AddNode("sgru_with_reset", "StatefulGru", input_names, {"Y", "Y_h"}, kQtiAiswDomain, attrs);
  };
}

TEST_F(QnnHTPBackendTests, StatefulGRU_Fp16_with_reset) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  RunQnnOnlyStatefulGRUModel(BuildStatefulGRUWithResetCase(5, 1, 3, 4),
                             provider_options, ExpectedEPNodeAssignment::All, 21);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
