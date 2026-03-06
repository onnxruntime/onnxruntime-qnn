// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Runs a non-QDQ model on HTP and compares output to CPU EP.
template <typename SignalType = float, typename StepType = int64_t>
static void RunStftOpTest(const TestInputDef<SignalType>& signal_def,
                          const TestInputDef<StepType>& frame_step_def,
                          const std::optional<TestInputDef<SignalType>>& window_def = std::nullopt,
                          const std::optional<TestInputDef<StepType>>& frame_length_def = std::nullopt,
                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs = {},
                          int opset_version = 17,
                          ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                          float fp32_abs_err = 1e-3f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  // Create a model builder function
  auto build_test_case = [signal_def, frame_step_def, window_def, frame_length_def, attrs](ModelTestBuilder& builder) {
    // signal input
    MakeTestInput<SignalType>(builder, "signal", signal_def);

    // frame_step input
    MakeTestInput<StepType>(builder, "frame_step", frame_step_def);

    // optional window input
    if (window_def.has_value()) {
      MakeTestInput<SignalType>(builder, "window", window_def.value());
    }

    // optional frame_length input
    if (frame_length_def.has_value()) {
      MakeTestInput<StepType>(builder, "frame_length", frame_length_def.value());
    }

    // inputs for STFT op
    std::vector<std::string> stft_inputs;
    stft_inputs.reserve(4);
    stft_inputs.push_back("signal");
    stft_inputs.push_back("frame_step");
    if (window_def.has_value()) {
      stft_inputs.push_back("window");
    }
    if (frame_length_def.has_value()) {
      stft_inputs.push_back("frame_length");
    }

    builder.MakeOutput("Y");
    builder.AddNode("STFT",
                    "STFT",
                    stft_inputs,
                    {"Y"},
                    kOnnxDomain,
                    attrs);
  };

  // Run the model test
  RunQnnModelTest(build_test_case,
                  provider_options,
                  opset_version,
                  expected_ep_assignment,
                  fp32_abs_err);
}

TEST_F(QnnHTPBackendTests, StftOp_Float_WithWindowOnly) {
  std::vector<float> signal_data(128, 1.0f);  // Signal: shape [1, 128, 1]
  std::vector<float> window_data(16, 1.0f);   // Window: shape [16]

  RunStftOpTest<float, int32_t>(
      TestInputDef<float>({1, 128, 1}, false, signal_data),  // signal
      TestInputDef<int32_t>({}, true, {8}),                  // frame_step
      TestInputDef<float>({16}, true, window_data),          // window
      std::nullopt,                                          // no frame_length
      {test::MakeAttribute("onesided", static_cast<int64_t>(1))},
      17,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, StftOp_Float_WithOnesidedFalse) {
  std::vector<float> signal_data(128, 1.0f);  // Signal: shape [1, 128, 1]
  std::vector<float> window_data(16, 1.0f);   // Window: shape [16]

  RunStftOpTest<float, int32_t>(
      TestInputDef<float>({1, 128, 1}, false, signal_data),        // signal
      TestInputDef<int32_t>({}, true, {8}),                        // frame_step
      TestInputDef<float>({16}, true, window_data),                // window
      TestInputDef<int32_t>({}, true, {16}),                       // frame_length
      {test::MakeAttribute("onesided", static_cast<int64_t>(0))},  // full spectrum
      17,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, StftOp_Float_SimpleExample) {
  std::vector<float> signal_data(128, 1.0f);  // Signal: shape [1, 128, 1]
  std::vector<float> window_data(16, 1.0f);   // Window: shape [16]

  RunStftOpTest<float, int32_t>(
      TestInputDef<float>({1, 128, 1}, false, signal_data),  // signal
      TestInputDef<int32_t>({}, true, {8}),                  // frame_step
      TestInputDef<float>({16}, true, window_data),          // window
      TestInputDef<int32_t>({}, true, {16}),                 // frame_length
      {test::MakeAttribute("onesided", static_cast<int64_t>(1))},
      17,
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, StftOp_Float_Rank2Signal) {
  std::vector<float> signal_data(128, 1.0f);  // Signal: shape [1, 128]
  std::vector<float> window_data(16, 1.0f);   // Window: shape [16]

  RunStftOpTest<float, int32_t>(
      TestInputDef<float>({1, 128}, false, signal_data),  // signal with rank 2
      TestInputDef<int32_t>({}, true, {8}),               // frame_step
      TestInputDef<float>({16}, true, window_data),       // window
      TestInputDef<int32_t>({}, true, {16}),              // frame_length
      {test::MakeAttribute("onesided", static_cast<int64_t>(1))},
      17,
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
}  // namespace test
}  // namespace onnxruntime

#endif
