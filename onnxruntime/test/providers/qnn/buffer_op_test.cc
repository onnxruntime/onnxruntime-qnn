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
  qti_aisw Buffer op (maps to QNN_OP_BUFFER "Buffer"):
  in[0]: activation (rank N)
  in[1]: reset (BOOL_8, 0D scalar, optional)
  out[0]: activation output; same rank as in[0], dim[buffer_dim] == buffer_size

  ONNX attributes (pass through to QNN params verbatim):
    buffer_size  (uint32, mandatory) — size of the sliding-window along buffer_dim
    buffer_dim   (uint32, mandatory) — axis of the sliding-window
    stride       (uint32, default 1)
    mode         (uint32) — 0=BLOCKING (unsupported on HTP), 1=NON_BLOCKING_LEFT, 2=NON_BLOCKING_RIGHT
    buffer_padding (uint32, default 0)

  Supported dtypes on HTP: FP16, INT16 (UFIXED_POINT_16), INT8 (UFIXED_POINT_8).
  mode=0 (BLOCKING) is not supported on HTP; IsOpSupported returns false for mode 0.

  Because Buffer is a stateful op with no CPU EP equivalent, accuracy comparison against
  a CPU reference is not meaningful — tests verify EP node assignment only
  (verify_outputs=false).
*/

// Build a minimal Buffer op test case.
//
// Input shape: (input_size,) [1D, buffer_dim=0 by default]
// Output shape: (buffer_size,) after the sliding window fills
// The model uses float (converted to fp16 for FP16 tests, or wrapped in QDQ for quantized tests).
template <typename InputType>
void _BuildBufferTestCase(ModelTestBuilder& builder,
                          const TestInputDef<float>& X_def,
                          const int64_t buffer_size,
                          const int64_t buffer_dim,
                          const int64_t mode,
                          const int64_t stride,
                          const std::vector<QuantParams<InputType>>& output_qparams) {
  static constexpr bool kIsFp16 = std::is_same<InputType, Ort::Float16_t>::value;
  static constexpr bool kIsU8 = std::is_same<InputType, uint8_t>::value;
  static constexpr bool kIsU16 = std::is_same<InputType, uint16_t>::value;

  std::string x_name;
  if constexpr (kIsFp16) {
    TestInputDef<Ort::Float16_t> fp16_def = ConvertToFP16InputDef(X_def);
    MakeTestInput(builder, "X", fp16_def);
    x_name = "X";
  } else if constexpr (kIsU8) {
    MakeTestInput(builder, "X", X_def);
    QuantParams<uint8_t> qparams = GetTestInputQuantParams<uint8_t>(X_def);
    x_name = AddQDQNodePair<uint8_t>(builder, "qdq_X", "X", qparams.scale, qparams.zero_point);
  } else if constexpr (kIsU16) {
    MakeTestInput(builder, "X", X_def);
    QuantParams<uint16_t> qparams = GetTestInputQuantParams<uint16_t>(X_def);
    x_name = AddQDQNodePair<uint16_t>(builder, "qdq_X", "X", qparams.scale, qparams.zero_point);
  } else {
    MakeTestInput(builder, "X", X_def);
    x_name = "X";
  }

  // Attributes
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(builder.MakeScalarAttribute("buffer_size", buffer_size));
  attrs.push_back(builder.MakeScalarAttribute("buffer_dim", buffer_dim));
  attrs.push_back(builder.MakeScalarAttribute("mode", mode));
  attrs.push_back(builder.MakeScalarAttribute("stride", stride));

  std::vector<int64_t> output_shape = X_def.GetShape();
  output_shape[static_cast<size_t>(buffer_dim)] = buffer_size;

  // Output
  std::string y_out;
  if constexpr (kIsU8 || kIsU16) {
    y_out = "buf_Y";
    // Declare the intermediate float32 tensor consumed by the Q node. This allows the QNN EP
    // to obtain the Buffer output shape without relying on custom-op shape inference.
    auto* vi = builder.model_.mutable_graph()->add_value_info();
    vi->set_name(y_out);
    auto* tt = vi->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* sp = tt->mutable_shape();
    for (const int64_t dim : output_shape) {
      sp->add_dim()->set_dim_value(dim);
    }
  } else {
    if constexpr (kIsFp16) {
      builder.MakeOutput<Ort::Float16_t>("Y", {output_shape});
    } else {
      builder.MakeOutput<float>("Y", {output_shape});
    }
    y_out = "Y";
  }

  builder.AddNode("buf", "Buffer", {x_name}, {y_out}, kQtiAiswDomain, attrs);

  QNN_TEST_UNUSED_PARAMETER(output_qparams);
  if constexpr (kIsU8) {
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_Y", y_out,
                                                   output_qparams[0].scale, output_qparams[0].zero_point);
  } else if constexpr (kIsU16) {
    AddQDQNodePairWithOutputAsGraphOutput<uint16_t>(builder, "qdq_Y", y_out,
                                                    output_qparams[0].scale, output_qparams[0].zero_point);
  }
}

template <typename InputType>
static GetTestModelFn BuildBufferTestCase(const TestInputDef<float>& X_def,
                                          const int64_t buffer_size,
                                          const int64_t buffer_dim,
                                          const int64_t mode,
                                          const int64_t stride = 1) {
  return [X_def, buffer_size, buffer_dim, mode, stride](ModelTestBuilder& builder) {
    _BuildBufferTestCase<InputType>(builder, X_def, buffer_size, buffer_dim, mode, stride, {});
  };
}

template <typename InputQType>
static GetTestQDQModelFn<InputQType> BuildQDQBufferTestCase(const TestInputDef<float>& X_def,
                                                            const int64_t buffer_size,
                                                            const int64_t buffer_dim,
                                                            const int64_t mode,
                                                            const int64_t stride = 1) {
  return [X_def, buffer_size, buffer_dim, mode, stride](
             ModelTestBuilder& builder, std::vector<QuantParams<InputQType>>& output_qparams) {
    _BuildBufferTestCase<InputQType>(builder, X_def, buffer_size, buffer_dim, mode, stride, output_qparams);
  };
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Builds the model from a GetTestModelFn and verifies QNN EP node assignment (no inference run).
// Buffer is a qti_aisw custom op with no CPU kernel; RunQnnModelTest's mandatory CPU baseline is
// neither possible nor meaningful. The QNN EP factory supplies the qti_aisw placeholder schema so
// the model loads; we then check EP node assignment.
static void RunQnnOnlyBufferModel(const GetTestModelFn& build_test_case,
                                  const ProviderOptions& provider_options,
                                  ExpectedEPNodeAssignment expected_ep_assignment,
                                  int opset) {
  // Positive cases need real HTP hardware to finalize the compiled graph; the x86_64 simulator
  // fails at graph finalize. Negative (Unsupported) cases only check IsOpSupported rejection and
  // run on all platforms.
  if (expected_ep_assignment == ExpectedEPNodeAssignment::All) {
    QNN_SKIP_TEST_ON_LINUX_X86_64("qti_aisw Buffer requires HTP hardware; not supported on the x86_64 simulator.");
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

  VerifyQnnEpModelAssignment(model_data, "Buffer_QNN", provider_options, expected_ep_assignment);
}

// Runs a Buffer model on the QNN HTP backend with FP16.
static void RunHtpFp16BufferOpTest(const TestInputDef<float>& X_def,
                                   const int64_t buffer_size,
                                   const int64_t buffer_dim,
                                   const int64_t mode,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   const int64_t stride = 1,
                                   int opset = 21) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  RunQnnOnlyBufferModel(BuildBufferTestCase<Ort::Float16_t>(X_def, buffer_size, buffer_dim, mode, stride),
                        provider_options,
                        expected_ep_assignment,
                        opset);
}

// Runs a Buffer model on the QNN HTP backend with QDQ quantization.
template <typename QuantType>
static void RunHtpQDQBufferOpTest(const TestInputDef<float>& X_def,
                                  const int64_t buffer_size,
                                  const int64_t buffer_dim,
                                  const int64_t mode,
                                  ExpectedEPNodeAssignment expected_ep_assignment,
                                  const int64_t stride = 1,
                                  int opset = 21) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Run the QDQ model on QNN only (no CPU reference available for Buffer).
  // Derive output qparams from the input range (Buffer output has same range as input).
  // This avoids needing a CPU reference session to compute output qparams.
  std::vector<QuantParams<QuantType>> out_qparams_vec = {GetTestInputQuantParams<QuantType>(X_def)};

  GetTestQDQModelFn<QuantType> qdq_fn = BuildQDQBufferTestCase<QuantType>(
      X_def, buffer_size, buffer_dim, mode, stride);

  GetTestModelFn model_fn = [qdq_fn, &out_qparams_vec](ModelTestBuilder& builder) {
    qdq_fn(builder, out_qparams_vec);
  };

  RunQnnOnlyBufferModel(model_fn, provider_options, expected_ep_assignment, opset);
}

// ============================================================
// HTP FP16 Tests
// ============================================================

// mode=1 (NON_BLOCKING_LEFT) — minimal 1D input
TEST_F(QnnHTPBackendTests, Buffer_Fp16_mode1_non_blocking_left) {
  // Input shape [8], buffer_size=4, buffer_dim=0, mode=1 (NON_BLOCKING_LEFT), stride=1
  // Output shape: [4] (dim[0] replaced by buffer_size)
  RunHtpFp16BufferOpTest(
      TestInputDef<float>({8}, false, -1.0f, 1.0f),  // X
      4,                                             // buffer_size
      0,                                             // buffer_dim
      1,                                             // mode = NON_BLOCKING_LEFT
      ExpectedEPNodeAssignment::All);
}

// mode=2 (NON_BLOCKING_RIGHT) — minimal 1D input
TEST_F(QnnHTPBackendTests, Buffer_Fp16_mode2_non_blocking_right) {
  RunHtpFp16BufferOpTest(
      TestInputDef<float>({8}, false, -1.0f, 1.0f),  // X
      4,                                             // buffer_size
      0,                                             // buffer_dim
      2,                                             // mode = NON_BLOCKING_RIGHT
      ExpectedEPNodeAssignment::All);
}

// mode=1 with 2D input, buffer_dim=1
TEST_F(QnnHTPBackendTests, Buffer_Fp16_mode1_2d_input) {
  // Input [3, 8], buffer_size=4, buffer_dim=1 → output [3, 4]
  RunHtpFp16BufferOpTest(
      TestInputDef<float>({3, 8}, false, -1.0f, 1.0f),  // X
      4,                                                // buffer_size
      1,                                                // buffer_dim
      1,                                                // mode = NON_BLOCKING_LEFT
      ExpectedEPNodeAssignment::All);
}

// ============================================================
// HTP QDQ Tests (INT8 / INT16)
// ============================================================

// INT8 (QDQ u8), mode=1
TEST_F(QnnHTPBackendTests, Buffer_QDQ_u8_mode1_non_blocking_left) {
  RunHtpQDQBufferOpTest<uint8_t>(
      TestInputDef<float>({8}, false, -1.0f, 1.0f),  // X
      4,                                             // buffer_size
      0,                                             // buffer_dim
      1,                                             // mode = NON_BLOCKING_LEFT
      ExpectedEPNodeAssignment::All);
}

// INT16 (QDQ u16), mode=1
TEST_F(QnnHTPBackendTests, Buffer_QDQ_u16_mode1_non_blocking_left) {
  RunHtpQDQBufferOpTest<uint16_t>(
      TestInputDef<float>({8}, false, -1.0f, 1.0f),  // X
      4,                                             // buffer_size
      0,                                             // buffer_dim
      1,                                             // mode = NON_BLOCKING_LEFT
      ExpectedEPNodeAssignment::All);
}

// INT8, mode=2
TEST_F(QnnHTPBackendTests, Buffer_QDQ_u8_mode2_non_blocking_right) {
  RunHtpQDQBufferOpTest<uint8_t>(
      TestInputDef<float>({8}, false, -1.0f, 1.0f),  // X
      4,                                             // buffer_size
      0,                                             // buffer_dim
      2,                                             // mode = NON_BLOCKING_RIGHT
      ExpectedEPNodeAssignment::All);
}

// ============================================================
// Negative test: mode=0 (BLOCKING) must not be captured by QNN EP.
// IsOpSupported returns false for mode=0.
// ============================================================

TEST_F(QnnHTPBackendTests, Buffer_Fp16_mode0_blocking_Unsupported) {
  RunHtpFp16BufferOpTest(
      TestInputDef<float>({8}, false, -1.0f, 1.0f),  // X
      4,                                             // buffer_size
      0,                                             // buffer_dim
      0,                                             // mode = BLOCKING (unsupported on HTP)
      ExpectedEPNodeAssignment::None);
}

// ============================================================
// Reset-input tests.
// ============================================================

// Reset is a BOOL scalar at Buffer input[1]. This verifies that the state-reset
// signal is included in the QNN Buffer node, rather than testing only the
// no-reset/default path above.
static GetTestModelFn BuildBufferWithResetCase() {
  return [](ModelTestBuilder& builder) {
    // Feed one frame per invocation. With buffer_size=4 this makes the second
    // (reset=false) result retain the first frame, while reset=true starts over.
    auto x_def = ConvertToFP16InputDef(TestInputDef<float>({1}, false, std::vector<float>{1.0f}));
    MakeTestInput(builder, "X", x_def);
    builder.MakeInput<bool>("buffer_reset", {}, {true});
    builder.MakeOutput<Ort::Float16_t>("Y", {{4}});

    std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
        builder.MakeScalarAttribute("buffer_size", static_cast<int64_t>(4)),
        builder.MakeScalarAttribute("buffer_dim", static_cast<int64_t>(0)),
        builder.MakeScalarAttribute("mode", static_cast<int64_t>(1)),
        builder.MakeScalarAttribute("stride", static_cast<int64_t>(1))};
    builder.AddNode("buf_with_reset", "Buffer", {"X", "buffer_reset"}, {"Y"}, kQtiAiswDomain, attrs);
  };
}

TEST_F(QnnHTPBackendTests, Buffer_Fp16_with_reset) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  RunQnnOnlyBufferModel(BuildBufferWithResetCase(), provider_options, ExpectedEPNodeAssignment::All, 21);
}

TEST_F(QnnHTPBackendTests, Buffer_Fp16_reset_restores_initial_state) {
  QNN_SKIP_TEST_ON_LINUX_X86_64("qti_aisw Buffer requires HTP hardware; not supported on the x86_64 simulator.");
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  VerifyQnnStatefulResetBehavior(BuildBufferWithResetCase(), "Buffer_ResetBehavior", provider_options, 21,
                                 "buffer_reset");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
