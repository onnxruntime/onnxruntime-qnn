// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

// GCC 14 with -O2 inlines TestInputDef construction deeply enough that its data-flow analyzer
// loses track of std::variant's initialization state, triggering a false-positive
// -Wmaybe-uninitialized. Same suppression as gemm_test.cc.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

namespace onnxruntime {
namespace test {

static void RunIrLSTMCompositionTest(uint32_t seq_len, uint32_t batch_size,
                                     uint32_t input_size, uint32_t hidden_size,
                                     bool has_B, bool has_H, bool has_C) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "ir";

  ModelTestBuilder builder;

  auto X = TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f);
  auto W = TestInputDef<float>({1, 4 * hidden_size, input_size}, false, -1.0f, 1.0f);
  auto R = TestInputDef<float>({1, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f);

  MakeTestInput(builder, "X", X);
  MakeTestInput(builder, "W", W);
  MakeTestInput(builder, "R", R);

  std::vector<std::string> input_names = {"X", "W", "R"};

  if (has_B) {
    auto B = TestInputDef<float>({1, 8 * hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "B", B);
    input_names.push_back("B");
  } else {
    input_names.push_back("");
  }

  input_names.push_back("");  // sequence_lens

  if (has_H) {
    auto H = TestInputDef<float>({1, batch_size, hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "initial_h", H);
    input_names.push_back("initial_h");
  } else {
    input_names.push_back("");
  }

  if (has_C) {
    auto C = TestInputDef<float>({1, batch_size, hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "initial_c", C);
    input_names.push_back("initial_c");
  } else {
    input_names.push_back("");
  }

  builder.MakeOutput("Y");
  builder.MakeOutput("Y_h");
  builder.MakeOutput("Y_c");

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      builder.MakeStringAttribute("direction", "forward"),
      builder.MakeScalarAttribute("hidden_size", static_cast<int64_t>(hidden_size)),
      builder.MakeScalarAttribute("layout", static_cast<int64_t>(0))};
  builder.AddNode("lstm", "LSTM", input_names, {"Y", "Y_h", "Y_c"}, "", attrs);

  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id{builder.model_.add_opset_import()};
  opset_id->set_domain("");
  opset_id->set_version(22);
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  builder.model_.SerializeToString(&model_data);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, "QNNExecutionProvider", provider_options);

  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);
}

static void RunIrGRUCompositionTest(uint32_t seq_len, uint32_t batch_size,
                                    uint32_t input_size, uint32_t hidden_size,
                                    bool has_B, bool has_H) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "ir";

  ModelTestBuilder builder;

  auto X = TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f);
  auto W = TestInputDef<float>({1, 3 * hidden_size, input_size}, false, -1.0f, 1.0f);
  auto R = TestInputDef<float>({1, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f);

  MakeTestInput(builder, "X", X);
  MakeTestInput(builder, "W", W);
  MakeTestInput(builder, "R", R);

  std::vector<std::string> input_names = {"X", "W", "R"};

  if (has_B) {
    auto B = TestInputDef<float>({1, 6 * hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "B", B);
    input_names.push_back("B");
  } else {
    input_names.push_back("");
  }

  input_names.push_back("");  // sequence_lens

  if (has_H) {
    auto H = TestInputDef<float>({1, batch_size, hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "initial_h", H);
    input_names.push_back("initial_h");
  } else {
    input_names.push_back("");
  }

  builder.MakeOutput("Y");
  builder.MakeOutput("Y_h");

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      builder.MakeStringAttribute("direction", "forward"),
      builder.MakeScalarAttribute("hidden_size", static_cast<int64_t>(hidden_size)),
      builder.MakeScalarAttribute("layout", static_cast<int64_t>(0)),
      builder.MakeScalarAttribute("linear_before_reset", static_cast<int64_t>(0))};
  builder.AddNode("gru", "GRU", input_names, {"Y", "Y_h"}, "", attrs);

  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id{builder.model_.add_opset_import()};
  opset_id->set_domain("");
  opset_id->set_version(22);
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  builder.model_.SerializeToString(&model_data);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, "QNNExecutionProvider", provider_options);

  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);
}

static void RunIrStatefulLSTMCompositionTest(uint32_t seq_len, uint32_t batch_size,
                                             uint32_t input_size, uint32_t hidden_size,
                                             bool has_B, bool has_H, bool has_C) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "ir";

  ModelTestBuilder builder;

  auto X = TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f);
  auto W = TestInputDef<float>({1, 4 * hidden_size, input_size}, false, -1.0f, 1.0f);
  auto R = TestInputDef<float>({1, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f);

  MakeTestInput(builder, "X", X);
  MakeTestInput(builder, "W", W);
  MakeTestInput(builder, "R", R);

  std::vector<std::string> input_names = {"X", "W", "R"};

  if (has_B) {
    auto B = TestInputDef<float>({1, 8 * hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "B", B);
    input_names.push_back("B");
  } else {
    input_names.push_back("");
  }

  input_names.push_back("");  // sequence_lens

  if (has_H) {
    auto H = TestInputDef<float>({1, batch_size, hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "initial_h", H);
    input_names.push_back("initial_h");
  } else {
    input_names.push_back("");
  }

  if (has_C) {
    auto C = TestInputDef<float>({1, batch_size, hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "initial_c", C);
    input_names.push_back("initial_c");
  } else {
    input_names.push_back("");
  }

  // reset (in[8]) absent — builder wires null_tensor for QNN in[24].
  // Declare typed+shaped outputs: StatefulLstm's placeholder schema sets output-0's shape from
  // input-0, but the QNN builder needs the declared shapes of all outputs for tensor setup.
  builder.MakeOutput<float>("Y", {{static_cast<int64_t>(seq_len), 1,
                                   static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}});
  builder.MakeOutput<float>("Y_h", {{1, static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}});
  builder.MakeOutput<float>("Y_c", {{1, static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}});

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      builder.MakeStringAttribute("direction", "forward"),
      builder.MakeScalarAttribute("hidden_size", static_cast<int64_t>(hidden_size))};
  builder.AddNode("slstm", "StatefulLstm", input_names, {"Y", "Y_h", "Y_c"}, kQtiAiswDomain, attrs);

  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> onnx_opset{builder.model_.add_opset_import()};
  onnx_opset->set_domain("");
  onnx_opset->set_version(13);
  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> qti_opset{builder.model_.add_opset_import()};
  qti_opset->set_domain(kQtiAiswDomain);
  qti_opset->set_version(1);
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  builder.model_.SerializeToString(&model_data);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, "QNNExecutionProvider", provider_options);

  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);
}

static void RunIrStatefulGRUCompositionTest(uint32_t seq_len, uint32_t batch_size,
                                            uint32_t input_size, uint32_t hidden_size,
                                            bool has_B, bool has_H) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "ir";

  ModelTestBuilder builder;

  auto X = TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f);
  auto W = TestInputDef<float>({1, 3 * hidden_size, input_size}, false, -1.0f, 1.0f);
  auto R = TestInputDef<float>({1, 3 * hidden_size, hidden_size}, false, -1.0f, 1.0f);

  MakeTestInput(builder, "X", X);
  MakeTestInput(builder, "W", W);
  MakeTestInput(builder, "R", R);

  std::vector<std::string> input_names = {"X", "W", "R"};

  if (has_B) {
    auto B = TestInputDef<float>({1, 6 * hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "B", B);
    input_names.push_back("B");
  } else {
    input_names.push_back("");
  }

  input_names.push_back("");  // sequence_lens

  if (has_H) {
    auto H = TestInputDef<float>({1, batch_size, hidden_size}, false, -1.0f, 1.0f);
    MakeTestInput(builder, "initial_h", H);
    input_names.push_back("initial_h");
  } else {
    input_names.push_back("");
  }

  // reset (in[6]) absent — builder wires null_tensor for QNN in[14].
  // Declare typed+shaped outputs: StatefulGru's placeholder schema sets output-0's shape from
  // input-0, but the QNN builder needs the declared shapes of all outputs for tensor setup.
  builder.MakeOutput<float>("Y", {{static_cast<int64_t>(seq_len), 1,
                                   static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}});
  builder.MakeOutput<float>("Y_h", {{1, static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)}});

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      builder.MakeStringAttribute("direction", "forward"),
      builder.MakeScalarAttribute("hidden_size", static_cast<int64_t>(hidden_size)),
      builder.MakeScalarAttribute("linear_before_reset", static_cast<int64_t>(0))};
  builder.AddNode("sgru", "StatefulGru", input_names, {"Y", "Y_h"}, kQtiAiswDomain, attrs);

  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> onnx_opset{builder.model_.add_opset_import()};
  onnx_opset->set_domain("");
  onnx_opset->set_version(21);
  const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> qti_opset{builder.model_.add_opset_import()};
  qti_opset->set_domain(kQtiAiswDomain);
  qti_opset->set_version(1);
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  builder.model_.SerializeToString(&model_data);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, "QNNExecutionProvider", provider_options);

  Ort::Session session(*GetOrtEnv(), model_data.data(), model_data.size(), session_options);
}

TEST_F(QnnIRBackendTests, LSTM_FP32_IR_forward_wo_P) {
  RunIrLSTMCompositionTest(5, 1, 3, 4, true, true, true);
}

TEST_F(QnnIRBackendTests, LSTM_FP32_IR_forward_wo_HC) {
  RunIrLSTMCompositionTest(5, 1, 3, 4, true, false, false);
}

TEST_F(QnnIRBackendTests, GRU_FP32_IR_forward_wo_H) {
  RunIrGRUCompositionTest(5, 1, 3, 4, true, false);
}

TEST_F(QnnIRBackendTests, StatefulLSTM_FP32_IR_forward_with_B_H_C) {
  RunIrStatefulLSTMCompositionTest(5, 1, 3, 4, /*has_B=*/true, /*has_H=*/true, /*has_C=*/true);
}

TEST_F(QnnIRBackendTests, StatefulLSTM_FP32_IR_forward_wo_B_H_C) {
  RunIrStatefulLSTMCompositionTest(5, 1, 3, 4, /*has_B=*/false, /*has_H=*/false, /*has_C=*/false);
}

TEST_F(QnnIRBackendTests, StatefulGRU_FP32_IR_forward_with_B_H) {
  RunIrStatefulGRUCompositionTest(5, 1, 3, 4, /*has_B=*/true, /*has_H=*/true);
}

TEST_F(QnnIRBackendTests, StatefulGRU_FP32_IR_forward_wo_H) {
  RunIrStatefulGRUCompositionTest(5, 1, 3, 4, /*has_B=*/true, /*has_H=*/false);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
