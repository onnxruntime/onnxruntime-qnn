// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
// Computes the mean and variance of inputs within a channel.
// Requires an input with rank >= 3
template <typename FLOAT_TYPE>
static void ComputeChannelMeanAndVar(const std::vector<FLOAT_TYPE>& input_data, const std::vector<int64_t>& input_shape,
                                     std::vector<FLOAT_TYPE>& mean_vals, std::vector<FLOAT_TYPE>& var_vals) {
  const size_t input_rank = input_shape.size();
  const size_t num_batches = input_shape[0];
  const size_t num_channels = input_shape[1];

  size_t batch_stride = 1;
  for (size_t i = 1; i < input_rank; i++) {
    batch_stride *= input_shape[i];
  }
  const size_t channel_stride = batch_stride / num_channels;

  QNN_ASSERT(mean_vals.size() == num_channels);
  QNN_ASSERT(var_vals.size() == num_channels);
  for (size_t i = 0; i < num_channels; i++) {
    mean_vals[i] = FLOAT_TYPE{};
    var_vals[i] = FLOAT_TYPE{};
  }

  // Compute running sum of elements within each channel. The running sum is stored in the mean_vals array directly.
  for (size_t b = 0; b < num_batches; b++) {
    const size_t batch_start = b * batch_stride;

    for (size_t c = 0; c < num_channels; c++) {
      const size_t chan_start = batch_start + (c * channel_stride);

      for (size_t i = chan_start; i < chan_start + channel_stride; i++) {
        // Avoid relying on implicit operator+ between FLOAT_TYPE and float16 wrappers (e.g., Ort::Float16_t).
        // Some toolchains (notably MSVC/ARM64) don't provide mixed-type operator overloads here.
        mean_vals[c] = static_cast<FLOAT_TYPE>(static_cast<float>(mean_vals[c]) + static_cast<float>(input_data[i]));
      }
    }
  }

  // Divide sums by the number of elements in a channel to get the mean.
  const float inv_count = 1.0f / static_cast<float>(num_batches * channel_stride);
  for (size_t c = 0; c < num_channels; c++) {
    mean_vals[c] = static_cast<FLOAT_TYPE>(static_cast<float>(mean_vals[c]) * inv_count);
  }

  // Compute running sum of deviations from mean within each channel. The running sum is stored in the var_vals array directly.
  for (size_t b = 0; b < num_batches; b++) {
    const size_t batch_start = b * batch_stride;

    for (size_t c = 0; c < num_channels; c++) {
      const size_t chan_start = batch_start + (c * channel_stride);

      for (size_t i = chan_start; i < chan_start + channel_stride; i++) {
        const FLOAT_TYPE deviation = static_cast<FLOAT_TYPE>(static_cast<float>(input_data[i]) -
                                                             static_cast<float>(mean_vals[c]));
        var_vals[c] = static_cast<FLOAT_TYPE>(static_cast<float>(var_vals[c]) +
                                              static_cast<float>(deviation) * static_cast<float>(deviation));
      }
    }
  }

  // Divide sums by the number of elements in a channel to get the variance.
  for (size_t c = 0; c < num_channels; c++) {
    var_vals[c] = static_cast<FLOAT_TYPE>(static_cast<float>(var_vals[c]) * inv_count);
  }
}

template <typename FLOAT_TYPE>
static GetTestModelFn BuildBatchNormTestCase(const TestInputDef<FLOAT_TYPE>& input_def,
                                             const TestInputDef<FLOAT_TYPE>& scale_def,
                                             const TestInputDef<FLOAT_TYPE>& bias_def) {
  QNN_ASSERT(input_def.IsRawData());  // Need raw data to compute mean and variance inputs.

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    std::vector<FLOAT_TYPE> mean_vals(num_channels);
    std::vector<FLOAT_TYPE> var_vals(num_channels);
    ComputeChannelMeanAndVar<FLOAT_TYPE>(input_data, input_shape, mean_vals, var_vals);

    builder.graph_->set_name("batch_norm_graph");

    MakeTestInput<FLOAT_TYPE>(builder, "X", input_def);
    MakeTestInput<FLOAT_TYPE>(builder, "scale", scale_def);
    MakeTestInput<FLOAT_TYPE>(builder, "bias", bias_def);
    builder.MakeInitializer<FLOAT_TYPE>("mean", {num_channels}, mean_vals);
    builder.MakeInitializer<FLOAT_TYPE>("var", {num_channels}, var_vals);

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    attributes.push_back(builder.MakeScalarAttribute("momentum", 0.9f));
    builder.AddNode(
        "bn",
        "BatchNormalization",
        {"X", "scale", "bias", "mean", "var"},
        {"Y"},
        "",
        attributes);

    builder.MakeOutput("Y");
  };
}

template <typename InputQType, typename ScaleQType>
GetTestQDQModelFn<InputQType> BuildQDQBatchNormTestCase(const TestInputDef<float>& input_def,
                                                        const TestInputDef<float>& scale_def,
                                                        const TestInputDef<float>& bias_def) {
  QNN_ASSERT(input_def.IsRawData());  // Need raw data to compute mean and variance inputs.

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder,
                                          std::vector<QuantParams<InputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];
    bool symmetric = sizeof(InputQType) == sizeof(uint16_t);
    MakeTestInput(builder, "X", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def, symmetric);
    std::string x_dq_name = AddQDQNodePair<InputQType>(builder, "qdq1", "X", input_qparams.scale, input_qparams.zero_point);

    MakeTestInput(builder, "scale", scale_def);
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);
    std::string scale_dq_name = AddQDQNodePair<ScaleQType>(builder, "qdq2", "scale", scale_qparams.scale, scale_qparams.zero_point);

    // bias (as int32) => DQ =>
    std::string bias_dq_name = MakeTestQDQBiasInput(builder, "bias", bias_def, input_qparams.scale * scale_qparams.scale, true);

    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);

    builder.MakeInitializer<float>("mean", {num_channels}, mean_vals);
    builder.MakeInitializer<float>("var", {num_channels}, var_vals);

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    attributes.push_back(builder.MakeScalarAttribute("momentum", 0.9f));
    builder.AddNode(
        "bn",
        "BatchNormalization",
        {x_dq_name.c_str(), scale_dq_name.c_str(), bias_dq_name.c_str(), "mean", "var"},
        {"Y"},
        "",
        attributes);

    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(
        builder, "qdq_out", "Y",
        output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

/**
 * Runs an BatchNormalization model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_shape The input's shape.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
template <typename InputQType, typename ScaleQType>
static void RunBatchNormQDQTestOnCPU(const TestInputDef<float>& input_def,
                                     const TestInputDef<float>& scale_def,
                                     const TestInputDef<float>& bias_def,
                                     ExpectedEPNodeAssignment expected_ep_assignment,
                                     QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildQDQBatchNormTestCase<InputQType, ScaleQType>(input_def, scale_def, bias_def),
                       provider_options,
                       21,
                       expected_ep_assignment,
                       tolerance);
}

TEST_F(QnnCPUBackendTests, BatchNorm2D_fp32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  RunQnnModelTest(
      BuildBatchNormTestCase(
          TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
          TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
          TestInputDef<float>({num_channels}, true, {1.1f, 2.1f})           // Bias initializer
          ),
      provider_options,
      13,
      EPVerificationParams{ExpectedEPNodeAssignment::All});
}

TEST_F(QnnCPUBackendTests, BatchNorm2D_int8) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTestOnCPU<uint8_t, uint8_t>(
      TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
      TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
      TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
      ExpectedEPNodeAssignment::All,
      QDQTolerance());
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Same shape as BuildQDQBatchNormTestCase, but the output Q/DQ pair can use a different quantized
// type than the input, so that dq_nodes[0]'s element type differs from q_nodes[0]'s. That mismatch
// is exactly what BatchNormalizationNodeGroupSelector::Check (ORT core) rejects when forming the
// BatchNorm QDQ NodeUnit, which degrades BatchNorm to a SingleNode NodeUnit with no DQ nodes attached.
template <typename InputQType, typename OutputQType, typename ScaleQType>
GetTestQDQModelFn<OutputQType> BuildQDQBatchNormMixedDtypeTestCase(const TestInputDef<float>& input_def,
                                                                   const TestInputDef<float>& scale_def,
                                                                   const TestInputDef<float>& bias_def) {
  QNN_ASSERT(input_def.IsRawData());  // Need raw data to compute mean and variance inputs.

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder,
                                          std::vector<QuantParams<OutputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];
    bool input_symmetric = sizeof(InputQType) == sizeof(uint16_t);
    MakeTestInput(builder, "X", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def, input_symmetric);
    std::string x_dq_name = AddQDQNodePair<InputQType>(builder, "qdq1", "X", input_qparams.scale, input_qparams.zero_point);

    MakeTestInput(builder, "scale", scale_def);
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);
    std::string scale_dq_name = AddQDQNodePair<ScaleQType>(builder, "qdq2", "scale", scale_qparams.scale, scale_qparams.zero_point);

    // bias (as int32) => DQ =>
    std::string bias_dq_name = MakeTestQDQBiasInput(builder, "bias", bias_def, input_qparams.scale * scale_qparams.scale, true);

    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);

    builder.MakeInitializer<float>("mean", {num_channels}, mean_vals);
    builder.MakeInitializer<float>("var", {num_channels}, var_vals);

    // Create attributes
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    attributes.push_back(builder.MakeScalarAttribute("momentum", 0.9f));
    builder.AddNode(
        "bn",
        "BatchNormalization",
        {x_dq_name.c_str(), scale_dq_name.c_str(), bias_dq_name.c_str(), "mean", "var"},
        {"Y"},
        "",
        attributes);

    AddQDQNodePairWithOutputAsGraphOutput<OutputQType>(
        builder, "qdq_out", "Y",
        output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

/**
 * Runs an BatchNormalization model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_shape The input's shape.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
template <typename InputQType, typename ScaleQType>
static void RunBatchNormQDQTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& scale_def,
                                const TestInputDef<float>& bias_def,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                QDQTolerance tolerance = QDQTolerance(),
                                const std::string& soc_model = "") {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  if (!soc_model.empty()) {
    provider_options["soc_model"] = soc_model;
  }

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildQDQBatchNormTestCase<InputQType, ScaleQType>(input_def, scale_def, bias_def),
                       provider_options,
                       21,
                       expected_ep_assignment,
                       tolerance);
}

static void RunBatchNormFP16Test(const TestInputDef<float>& input_def,
                                 const TestInputDef<float>& scale_def,
                                 const TestInputDef<float>& bias_def,
                                 ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestInputDef<Ort::Float16_t> input_fp16_def = ConvertToFP16InputDef(input_def);
  TestInputDef<Ort::Float16_t> scale_fp16_def = ConvertToFP16InputDef(scale_def);
  TestInputDef<Ort::Float16_t> bias_fp16_def = ConvertToFP16InputDef(bias_def);

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  TestFp16ModelAccuracy(BuildBatchNormTestCase<float>(input_def, scale_def, bias_def),
                        BuildBatchNormTestCase<Ort::Float16_t>(input_fp16_def, scale_fp16_def, bias_fp16_def),
                        provider_options,
                        11,
                        expected_ep_assignment);
}

// BatchNor QDQ model, input with rank 2.
TEST_F(QnnHTPBackendTests, BatchNormRank2) {
  constexpr int64_t num_channels = 2;

  RunBatchNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({4, num_channels}, false,
                                                            {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f}),  // Input data
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),                    // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),                    // Bias initializer
                                        ExpectedEPNodeAssignment::All);
}

// TODO: FIX TRANSLATION!!!
// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 3.
// Accuracy issue with Linux simulator, not sure with Android device
// Inaccuracy detected for output 'output_0', element 1
// output_range=4.8666362762451172, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): 1.0999999046325684
// qdq@QNN_EP val: -0.17176364362239838 (err: 1.2717635631561279, err/output_range: 26.132291793823242%)
// qdq@CPU_EP val: 1.1069211959838867 (err: 0.0069212913513183594, err/output_range: 0.14221921563148499%)
// abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = 25.990072250366211%
//
// Inaccuracy detected for output 'output_0', element 2
// output_range=4.8666362762451172, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): 2.3247356414794922
// qdq@QNN_EP val: -0.17176364362239838 (err: 2.4964993000030518, err/output_range: 51.298248291015625%)
// qdq@CPU_EP val: 2.3474364280700684 (err: 0.022700786590576172, err/output_range: 0.46645742654800415%)
#if defined(_WIN32)
TEST_F(QnnHTPBackendTests, BatchNorm1D) {
  constexpr int64_t num_channels = 2;

  RunBatchNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, num_channels, 3}, false,
                                                            {-5.0f, -4.0f, -3.0f, 0.0f, 2.0f, 5.0f}),  // Input data
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),       // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),       // Bias initializer
                                        ExpectedEPNodeAssignment::All);
}
#endif

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_U8U8S32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_U8S8S32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint8_t, int8_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                       TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                       TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                       ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_U8U16S32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint8_t, uint16_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                         TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                         TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                         ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_U8S16S32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint8_t, int16_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                        ExpectedEPNodeAssignment::None);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_U16U8S32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint16_t, uint8_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                         TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                         TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                         ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_U16U16S32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint16_t, uint16_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                          TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                          TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                          ExpectedEPNodeAssignment::All,
                                          QDQTolerance(),
#if defined(__linux__) && !defined(__aarch64__)
                                          std::to_string(QNN_SOC_MODEL_SM8550));
#else
                                          "");
#endif
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_U16S16S32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint16_t, int16_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                         TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                         TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                         ExpectedEPNodeAssignment::All,
                                         QDQTolerance(),
#if defined(__linux__) && !defined(__aarch64__)
                                         std::to_string(QNN_SOC_MODEL_SM8550));
#else
                                         "");
#endif
}

// Test FP16 BatchNormalization on the HTP backend.
TEST_F(QnnHTPBackendTests, BatchNorm_FP16) {
#if defined(_WIN32)
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
  }
#endif
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormFP16Test(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                       TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                       TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                       ExpectedEPNodeAssignment::All);
}

// Test FP32 BatchNormalization on the HTP backend with the enable_htp_fp16_precision option enabled
// to run it with fp16 precision.
TEST_F(QnnHTPBackendTests, BatchNorm_FP32_as_FP16) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
#if defined(_WIN32)
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "Test requires HTP FP16 support (arch > V68).";
  }
#endif
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  provider_options["enable_htp_fp16_precision"] = "1";

  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  auto input_def = TestInputDef<float>({2, num_channels, 2, 2}, false, input_data);
  auto scale_def = TestInputDef<float>({num_channels}, true, {1.0f, 2.0f});
  auto bias_def = TestInputDef<float>({num_channels}, true, {1.1f, 2.1f});
  auto model_fn = BuildBatchNormTestCase<float>(input_def, scale_def, bias_def);

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.01f)});
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 5. QNN BatchNormalization doesn't support 5D on HTP
TEST_F(QnnHTPBackendTests, BatchNorm3D) {
  constexpr int64_t num_channels = 2;
  constexpr int64_t num_elems = 1 * num_channels * 3 * 4 * 5;
  RunBatchNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, num_channels, 3, 4, 5}, false,
                                                            std::vector<float>(num_elems)),       // Input data (all zeros)
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),  // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),  // Bias initializer
                                        ExpectedEPNodeAssignment::None);
}

// Tests BatchNorm with Q->DQ structure commonly seen in quantized models
template <typename InputQType, typename ParamQType>
GetTestQDQModelFn<InputQType> BuildBatchNormQdqParamsTestCase(const TestInputDef<float>& input_def,
                                                              const TestInputDef<float>& scale_def,
                                                              const TestInputDef<float>& bias_def) {
  QNN_ASSERT(input_def.IsRawData());
  QNN_ASSERT(scale_def.IsRawData());

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder,
                                          std::vector<QuantParams<InputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    // Input: float -> Q -> DQ
    bool symmetric = sizeof(InputQType) == sizeof(uint16_t);
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def, symmetric);
    std::string input_qdq = AddQDQNodePair<InputQType>(builder, "input_qdq", "input", input_qparams.scale, input_qparams.zero_point);

    // Create axis attribute for per-channel quantization
    std::vector<ONNX_NAMESPACE::AttributeProto> axis_0_attrs;
    axis_0_attrs.push_back(builder.MakeScalarAttribute("axis", static_cast<int64_t>(0)));

    // Scale: float_init -> Q -> DQ (per-channel with axis=0, symmetric)
    const auto& scale_data = scale_def.GetRawData();
    std::vector<float> scale_scales(num_channels);
    std::vector<ParamQType> scale_zero_points(num_channels, static_cast<ParamQType>(0));
    for (int64_t c = 0; c < num_channels; ++c) {
      float abs_max = std::abs(scale_data[c]);
      if (abs_max == 0.0f) abs_max = 1.0f;
      scale_scales[c] = abs_max / static_cast<float>(std::numeric_limits<ParamQType>::max());
    }
    std::vector<int64_t> param_shape = {num_channels};
    builder.MakeInitializer<float>("scale_float_init", param_shape, scale_data);
    std::string scale_qdq = AddQDQNodePair<ParamQType>(builder, "scale_qdq", "scale_float_init", scale_scales, scale_zero_points,
                                                       axis_0_attrs, axis_0_attrs);

    builder.MakeInitializer<float>("bias", bias_def.GetShape(), bias_def.GetRawData());

    // Compute mean and var from input data
    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);

    // Mean: float_init -> Q -> DQ (per-channel with axis=0, symmetric)
    std::vector<float> mean_scales(num_channels);
    std::vector<ParamQType> mean_zero_points(num_channels, static_cast<ParamQType>(0));
    for (int64_t c = 0; c < num_channels; ++c) {
      float abs_max = std::abs(mean_vals[c]);
      if (abs_max == 0.0f) abs_max = 1.0f;
      mean_scales[c] = abs_max / static_cast<float>(std::numeric_limits<ParamQType>::max());
    }
    builder.MakeInitializer<float>("mean_float_init", param_shape, mean_vals);
    std::string mean_qdq = AddQDQNodePair<ParamQType>(builder, "mean_qdq", "mean_float_init", mean_scales, mean_zero_points,
                                                      axis_0_attrs, axis_0_attrs);

    // Var: float_init -> Q -> DQ (per-channel with axis=0, symmetric)
    std::vector<float> var_scales(num_channels);
    std::vector<ParamQType> var_zero_points(num_channels, static_cast<ParamQType>(0));
    for (int64_t c = 0; c < num_channels; ++c) {
      float abs_max = std::abs(var_vals[c]);
      if (abs_max == 0.0f) abs_max = 1.0f;
      var_scales[c] = abs_max / static_cast<float>(std::numeric_limits<ParamQType>::max());
    }
    builder.MakeInitializer<float>("var_float_init", param_shape, var_vals);
    std::string var_qdq = AddQDQNodePair<ParamQType>(builder, "var_qdq", "var_float_init", var_scales, var_zero_points,
                                                     axis_0_attrs, axis_0_attrs);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("batchnorm", "BatchNormalization", {input_qdq, scale_qdq, "bias", mean_qdq, var_qdq},
                    {"batchnorm_output"}, "", attributes);

    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "output_qdq", "batchnorm_output",
                                                      output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// Test BatchNorm with Q->DQ on input/scale/mean/var, float bias
TEST_F(QnnHTPBackendTests, BatchNorm2dQdqParams) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  TestInputDef<float> input_def({2, num_channels, 2, 2}, false, input_data);
  TestInputDef<float> scale_def({num_channels}, true, {1.0f, 2.0f});
  TestInputDef<float> bias_def({num_channels}, true, {1.1f, 2.1f});

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildBatchNormQdqParamsTestCase<uint16_t, int8_t>(input_def, scale_def, bias_def),
                       provider_options,
                       21,
                       ExpectedEPNodeAssignment::All);
}

// DQ on x (per-tensor) and scale (per-tensor), float bias/mean/var. 2 DQ nodes total.
// Pattern: DQ(x) + DQ(scale) -> BN(float bias, float mean, float var) -> Q(output)
template <typename InputQType, typename ScaleQType>
GetTestQDQModelFn<InputQType> BuildBatchNormDqOnXAndPerTensorScaleTestCase(const TestInputDef<float>& input_def,
                                                                           const TestInputDef<float>& scale_def,
                                                                           const TestInputDef<float>& bias_def) {
  QNN_ASSERT(input_def.IsRawData());
  QNN_ASSERT(scale_def.IsRawData());

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder,
                                          std::vector<QuantParams<InputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    bool symmetric = sizeof(InputQType) == sizeof(uint16_t);
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def, symmetric);
    std::string input_qdq = AddQDQNodePair<InputQType>(builder, "input_qdq", "input",
                                                       input_qparams.scale, input_qparams.zero_point);

    const auto& scale_data = scale_def.GetRawData();
    float scale_abs_max = 0.0f;
    for (auto v : scale_data) scale_abs_max = std::max(scale_abs_max, std::abs(v));
    if (scale_abs_max == 0.0f) scale_abs_max = 1.0f;
    float scale_qscale = scale_abs_max / static_cast<float>(std::numeric_limits<ScaleQType>::max());

    std::vector<int64_t> param_shape = {num_channels};
    builder.MakeInitializer<float>("scale_float_init", param_shape, scale_data);
    std::string scale_qdq = AddQDQNodePair<ScaleQType>(builder, "scale_qdq", "scale_float_init",
                                                       scale_qscale, static_cast<ScaleQType>(0));

    builder.MakeInitializer<float>("bias", bias_def.GetShape(), bias_def.GetRawData());

    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);
    builder.MakeInitializer<float>("mean", param_shape, mean_vals);
    builder.MakeInitializer<float>("var", param_shape, var_vals);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    builder.AddNode("batchnorm", "BatchNormalization",
                    {input_qdq, scale_qdq, "bias", "mean", "var"},
                    {"batchnorm_output"}, "", attributes);

    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "output_qdq", "batchnorm_output",
                                                      output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

TEST_F(QnnHTPBackendTests, BatchNorm2DU16S8FloatParams) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  TestInputDef<float> input_def({2, num_channels, 2, 2}, false, input_data);
  TestInputDef<float> scale_def({num_channels}, true, {1.0f, 2.0f});
  TestInputDef<float> bias_def({num_channels}, true, {1.1f, 2.1f});

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildBatchNormDqOnXAndPerTensorScaleTestCase<uint16_t, int8_t>(input_def, scale_def, bias_def),
                       provider_options,
                       21,
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, BatchNorm2DU8S8FloatParams) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  TestInputDef<float> input_def({2, num_channels, 2, 2}, false, input_data);
  TestInputDef<float> scale_def({num_channels}, true, {1.0f, 2.0f});
  TestInputDef<float> bias_def({num_channels}, true, {1.1f, 2.1f});

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildBatchNormDqOnXAndPerTensorScaleTestCase<uint8_t, int8_t>(input_def, scale_def, bias_def),
                       provider_options,
                       21,
                       ExpectedEPNodeAssignment::All);
}

// Test BatchNorm with U8 input, S8 scale (converted to U8), float bias (converted to S32)
TEST_F(QnnHTPBackendTests, BatchNorm2D_U8S8F32) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  TestInputDef<float> input_def({2, num_channels, 2, 2}, false, input_data);
  TestInputDef<float> scale_def({num_channels}, true, {1.0f, 2.0f});
  TestInputDef<float> bias_def({num_channels}, true, {1.1f, 2.1f});

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildBatchNormQdqParamsTestCase<uint8_t, int8_t>(input_def, scale_def, bias_def),
                       provider_options,
                       21,
                       ExpectedEPNodeAssignment::All);
}

// Test BatchNorm with low variance channels (U16 input). Channels with small variance produce
// large fused weights that overflow per-tensor uint8 quantization. Float-promotion path fixes this issue
TEST_F(QnnHTPBackendTests, BatchNorm2D_NearZeroVariance_U16) {
#if defined(__linux__) && !defined(__aarch64__)
  GTEST_SKIP() << "Skipped on x86_64 simulator due to flaky behavior";
#else
  constexpr int64_t batch = 1;
  constexpr int64_t channels = 4;
  constexpr int64_t H = 2;
  constexpr int64_t W = 2;
  // Channels 0,1: normal variance. Channels 2,3: low variance (but not zero)
  std::vector<float> input_data = {
      -4.0f, -2.0f, 2.0f, 4.0f,   // ch0: var ~= 10
      -3.0f, -1.0f, 1.0f, 3.0f,   // ch1: var ~= 5
      5.0f, 5.1f, 4.9f, 5.0f,     // ch2: var ~= 0.005
      -2.0f, -1.9f, -2.1f, -2.0f  // ch3: var ~= 0.005
  };

  std::vector<float> scale_data = {1.0f, 1.5f, 2.0f, 3.0f};
  std::vector<float> bias_data = {0.0f, 0.5f, 1.0f, -1.0f};

  TestInputDef<float> input_def({batch, channels, H, W}, false, input_data);
  TestInputDef<float> scale_def({channels}, true, scale_data);
  TestInputDef<float> bias_def({channels}, true, bias_data);

  RunBatchNormQDQTest<uint16_t, uint8_t>(input_def, scale_def, bias_def,
                                         ExpectedEPNodeAssignment::All,
                                         QDQTolerance(0.008f));
#endif
}

// QDQ BatchNormalization whose scale is quantized per channel (axis 0) and whose input/output are per-tensor.
//   DQ(x) + DQ(scale, per-channel) + DQ(bias) -> BN -> Q
template <typename InputQType>
static GetTestQDQModelFn<InputQType> BuildPerChannelScaleQDQBatchNormTestCase(const TestInputDef<float>& input_def,
                                                                              const TestInputDef<float>& scale_def,
                                                                              const TestInputDef<float>& bias_def) {
  QNN_ASSERT(input_def.IsRawData());  // Need raw data to compute mean and variance inputs.

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder,
                                          std::vector<QuantParams<InputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const int64_t num_channels = input_shape[1];

    MakeTestInput(builder, "X", input_def);
    const bool symmetric = sizeof(InputQType) == sizeof(uint16_t);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def, symmetric);
    std::string x_dq_name = AddQDQNodePair<InputQType>(builder, "qdq1", "X", input_qparams.scale,
                                                       input_qparams.zero_point);

    MakeTestInput(builder, "scale", scale_def);
    std::vector<float> scale_scales;
    std::vector<uint8_t> scale_zps;
    GetTestInputQuantParamsPerChannel<uint8_t>(scale_def, scale_scales, scale_zps, 0);
    std::vector<ONNX_NAMESPACE::AttributeProto> axis_attr = {builder.MakeScalarAttribute("axis", int64_t{0})};
    std::string scale_dq_name = AddQDQNodePair<uint8_t>(builder, "qdq2", "scale", scale_scales, scale_zps,
                                                        axis_attr, axis_attr);

    std::string bias_dq_name = MakeTestQDQBiasInput(builder, "bias", bias_def,
                                                    input_qparams.scale * scale_scales[0], true);

    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_def.GetRawData(), input_shape, mean_vals, var_vals);
    builder.MakeInitializer<float>("mean", {num_channels}, mean_vals);
    builder.MakeInitializer<float>("var", {num_channels}, var_vals);

    std::vector<ONNX_NAMESPACE::AttributeProto> attributes;
    attributes.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    attributes.push_back(builder.MakeScalarAttribute("momentum", 0.9f));
    builder.AddNode("bn", "BatchNormalization",
                    {x_dq_name.c_str(), scale_dq_name.c_str(), bias_dq_name.c_str(), "mean", "var"},
                    {"Y"}, "", attributes);

    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "qdq_out", "Y",
                                                      output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// Test BatchNorm with near-zero variance channels (U8 input). When gamma/sqrt(var+eps) produces
// outlier values, per-tensor uint8 quantization of fused weight overflows. Keeping the per-channel parameters
// avoids this accuracy loss, either as the float island (Convert U8->F32, BN in F32, Convert F32->U8) or, on
// HTP arch >= v73, as a per-channel depthwise conv.
TEST_F(QnnHTPBackendTests, BatchNorm2D_NearZeroVariance_U8) {
  constexpr int64_t batch = 1;
  constexpr int64_t channels = 4;
  constexpr int64_t H = 2;
  constexpr int64_t W = 2;
  // Channels 0,1: normal variance. Channels 2,3: near-zero variance (constant-ish values).
  std::vector<float> input_data = {
      -4.0f, -2.0f, 2.0f, 4.0f,   // ch0: var ~= 10
      -3.0f, -1.0f, 1.0f, 3.0f,   // ch1: var ~= 5
      5.0f, 5.0f, 5.0f, 5.0f,     // ch2: var ~= 0 (near-zero)
      -2.0f, -2.0f, -2.0f, -2.0f  // ch3: var ~= 0 (near-zero)
  };

  std::vector<float> scale_data = {1.0f, 1.5f, 2.0f, 3.0f};
  std::vector<float> bias_data = {0.0f, 0.5f, 1.0f, -1.0f};

  TestInputDef<float> input_def({batch, channels, H, W}, false, input_data);
  TestInputDef<float> scale_def({channels}, true, scale_data);
  TestInputDef<float> bias_def({channels}, true, bias_data);

  // The per-channel scale QDQ model exercises the per-channel promotion logic (float island or depthwise conv).
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildPerChannelScaleQDQBatchNormTestCase<uint8_t>(input_def, scale_def, bias_def),
                       provider_options, 21, ExpectedEPNodeAssignment::All);
}

// Builds the float (reference) form of the NASNet reduction-cell pattern:
//   X -> BatchNormalization -> Relu -> Conv -> Y
template <typename FLOAT_TYPE>
static GetTestModelFn BuildBatchNormFloatOutputTestCase(const TestInputDef<FLOAT_TYPE>& input_def,
                                                        const std::vector<FLOAT_TYPE>& scale_data,
                                                        const std::vector<FLOAT_TYPE>& bias_data,
                                                        const std::vector<FLOAT_TYPE>& conv_w_data) {
  QNN_ASSERT(input_def.IsRawData());
  return [input_def, scale_data, bias_data, conv_w_data](ModelTestBuilder& builder) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    std::vector<FLOAT_TYPE> mean_vals(num_channels);
    std::vector<FLOAT_TYPE> var_vals(num_channels);
    ComputeChannelMeanAndVar<FLOAT_TYPE>(input_data, input_shape, mean_vals, var_vals);

    MakeTestInput<FLOAT_TYPE>(builder, "X", input_def);
    builder.MakeInitializer<FLOAT_TYPE>("scale", {num_channels}, scale_data);
    builder.MakeInitializer<FLOAT_TYPE>("bias", {num_channels}, bias_data);
    builder.MakeInitializer<FLOAT_TYPE>("mean", {num_channels}, mean_vals);
    builder.MakeInitializer<FLOAT_TYPE>("var", {num_channels}, var_vals);

    std::vector<ONNX_NAMESPACE::AttributeProto> bn_attrs;
    bn_attrs.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    builder.AddNode("bn", "BatchNormalization", {"X", "scale", "bias", "mean", "var"}, {"bn_out"}, "", bn_attrs);
    builder.AddNode("relu", "Relu", {"bn_out"}, {"relu_out"});

    builder.MakeInitializer<FLOAT_TYPE>("conv_w", {num_channels, num_channels, 1, 1}, conv_w_data);
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    builder.AddNode("conv", "Conv", {"relu_out", "conv_w"}, {"Y"}, "", conv_attrs);
    builder.MakeOutput("Y");
  };
}

// Builds the QDQ form where BN has DQ on its inputs but no Q on its output, so its result flows in
// float through the Relu and Conv before the Conv's output Q re-quantizes it.
//
//   DQ(x) + DQ(scale) -> BN(float bias/mean/var) -> Relu -> Conv -> Q(graph output)
template <typename InputQType, typename ScaleQType>
static GetTestQDQModelFn<InputQType> BuildBatchNormFloatOutputQDQTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<float>& scale_data,
    const std::vector<float>& bias_data,
    const std::vector<float>& conv_w_data) {
  QNN_ASSERT(input_def.IsRawData());
  return [input_def, scale_data, bias_data, conv_w_data](ModelTestBuilder& builder,
                                                         std::vector<QuantParams<InputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    // Input: float -> Q -> DQ.
    bool symmetric = sizeof(InputQType) == sizeof(uint16_t);
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def, symmetric);
    std::string input_dq = AddQDQNodePair<InputQType>(builder, "input_qdq", "input",
                                                      input_qparams.scale, input_qparams.zero_point);

    // Scale: float_init -> Q -> DQ (per-tensor). Makes scale a DQ output, not a raw initializer,
    // which is what made the old selector treat it as "dynamic".
    float scale_abs_max = 0.0f;
    for (float v : scale_data) scale_abs_max = std::max(scale_abs_max, std::abs(v));
    if (scale_abs_max == 0.0f) scale_abs_max = 1.0f;
    float scale_qscale = scale_abs_max / static_cast<float>(std::numeric_limits<ScaleQType>::max());
    builder.MakeInitializer<float>("scale_init", {num_channels}, scale_data);
    std::string scale_dq = AddQDQNodePair<ScaleQType>(builder, "scale_qdq", "scale_init",
                                                      scale_qscale, static_cast<ScaleQType>(0));

    // Float bias/mean/var initializers.
    builder.MakeInitializer<float>("bias", {num_channels}, bias_data);
    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);
    builder.MakeInitializer<float>("mean", {num_channels}, mean_vals);
    builder.MakeInitializer<float>("var", {num_channels}, var_vals);

    std::vector<ONNX_NAMESPACE::AttributeProto> bn_attrs;
    bn_attrs.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    // BN output "bn_out" has NO Q after it, so it stays float.
    builder.AddNode("bn", "BatchNormalization",
                    {input_dq, scale_dq, "bias", "mean", "var"}, {"bn_out"}, "", bn_attrs);

    // Relu and Conv run in float; the graph re-enters quantized space only at the Conv's output Q.
    builder.AddNode("relu", "Relu", {"bn_out"}, {"relu_out"});
    builder.MakeInitializer<float>("conv_w", {num_channels, num_channels, 1, 1}, conv_w_data);
    std::vector<ONNX_NAMESPACE::AttributeProto> conv_attrs;
    conv_attrs.push_back(builder.MakeStringAttribute("auto_pad", std::string("NOTSET")));
    builder.AddNode("conv", "Conv", {"relu_out", "conv_w"}, {"conv_out"}, "", conv_attrs);

    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "output_qdq", "conv_out",
                                                      output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// A BatchNormalization with no Q on its output must be captured by QNN (entire graph), not fall back
// to CPU. Accuracy is compared against the float reference.
TEST_F(QnnHTPBackendTests, BatchNorm_NoOutputQ) {
  constexpr int64_t batch = 1;
  constexpr int64_t channels = 2;
  constexpr int64_t H = 4;
  constexpr int64_t W = 4;
  constexpr int64_t num_elems = batch * channels * H * W;

  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, num_elems);
  std::vector<float> scale_data = {1.0f, 2.0f};
  std::vector<float> bias_data = {1.1f, 2.1f};
  // 1x1 conv weights keep spatial dims unchanged: shape [out_ch, in_ch, 1, 1].
  // Use positive weights to avoid fp16 catastrophic cancellation: the float region executes in fp16
  // on HTP, so a near-zero cancelled output would be precision-dominated.
  std::vector<float> conv_w_data = {0.5f, 0.25f, 0.25f, 0.5f};

  TestInputDef<float> input_def({batch, channels, H, W}, false, input_data);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildBatchNormFloatOutputTestCase<float>(input_def, scale_data, bias_data, conv_w_data),
                       BuildBatchNormFloatOutputQDQTestCase<uint8_t, int8_t>(input_def, scale_data, bias_data,
                                                                             conv_w_data),
                       provider_options,
                       21,
                       ExpectedEPNodeAssignment::All);
}

// Float reference for two BatchNorm nodes that SHARE a single bias initializer but use different
// scales (hence different correct fused biases), summed into one output.
//   X -> BN_a(scale_a, SHARED_bias, mean1, var1) -> a
//   X -> BN_b(scale_b, SHARED_bias, mean2, var2) -> b
//   Add(a, b) -> Y
template <typename FLOAT_TYPE>
static GetTestModelFn BuildSharedBiasBatchNormFloatTestCase(const TestInputDef<FLOAT_TYPE>& input_def,
                                                            const std::vector<FLOAT_TYPE>& scale_a_data,
                                                            const std::vector<FLOAT_TYPE>& scale_b_data,
                                                            const std::vector<FLOAT_TYPE>& bias_data) {
  QNN_ASSERT(input_def.IsRawData());
  return [input_def, scale_a_data, scale_b_data, bias_data](ModelTestBuilder& builder) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    std::vector<FLOAT_TYPE> mean_vals(num_channels);
    std::vector<FLOAT_TYPE> var_vals(num_channels);
    ComputeChannelMeanAndVar<FLOAT_TYPE>(input_data, input_shape, mean_vals, var_vals);

    MakeTestInput<FLOAT_TYPE>(builder, "X", input_def);
    builder.MakeInitializer<FLOAT_TYPE>("scale_a", {num_channels}, scale_a_data);
    builder.MakeInitializer<FLOAT_TYPE>("scale_b", {num_channels}, scale_b_data);
    builder.MakeInitializer<FLOAT_TYPE>("shared_bias", {num_channels}, bias_data);
    builder.MakeInitializer<FLOAT_TYPE>("mean", {num_channels}, mean_vals);
    builder.MakeInitializer<FLOAT_TYPE>("var", {num_channels}, var_vals);

    std::vector<ONNX_NAMESPACE::AttributeProto> bn_attrs;
    bn_attrs.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    builder.AddNode("bn_a", "BatchNormalization", {"X", "scale_a", "shared_bias", "mean", "var"},
                    {"a_out"}, "", bn_attrs);
    std::vector<ONNX_NAMESPACE::AttributeProto> bn_attrs_b;
    bn_attrs_b.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    builder.AddNode("bn_b", "BatchNormalization", {"X", "scale_b", "shared_bias", "mean", "var"},
                    {"b_out"}, "", bn_attrs_b);

    builder.AddNode("add", "Add", {"a_out", "b_out"}, {"Y"});
    builder.MakeOutput("Y");
  };
}

// QDQ form: two BN nodes share one float bias initializer ("shared_bias") but consume different
// per-tensor scales. Each node's correct fused bias (bias - mean*scale/sqrt(var+eps)) differs.
// The QNN EP must emit a distinct fused-bias QNN tensor per node; if it instead caches the fused
// bias under the shared ONNX initializer name, the second BN silently reuses the first node's
// fused bias and produces wrong results. (Root cause of SiNet w8a8 bn_2/bn_3 bias collision.)
template <typename InputQType, typename ScaleQType>
static GetTestQDQModelFn<InputQType> BuildSharedBiasBatchNormQDQTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<float>& scale_a_data,
    const std::vector<float>& scale_b_data,
    const std::vector<float>& bias_data) {
  QNN_ASSERT(input_def.IsRawData());
  return [input_def, scale_a_data, scale_b_data, bias_data](
             ModelTestBuilder& builder, std::vector<QuantParams<InputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    bool symmetric = sizeof(InputQType) == sizeof(uint16_t);
    MakeTestInput<float>(builder, "input", input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def, symmetric);
    std::string input_dq = AddQDQNodePair<InputQType>(builder, "input_qdq", "input",
                                                      input_qparams.scale, input_qparams.zero_point);

    auto make_scale_dq = [&](const std::string& tag, const std::vector<float>& sdata) {
      float abs_max = 0.0f;
      for (float v : sdata) abs_max = std::max(abs_max, std::abs(v));
      if (abs_max == 0.0f) abs_max = 1.0f;
      float qscale = abs_max / static_cast<float>(std::numeric_limits<ScaleQType>::max());
      builder.MakeInitializer<float>(tag + "_init", {num_channels}, sdata);
      return AddQDQNodePair<ScaleQType>(builder, tag + "_qdq", tag + "_init", qscale,
                                        static_cast<ScaleQType>(0));
    };
    std::string scale_a_dq = make_scale_dq("scale_a", scale_a_data);
    std::string scale_b_dq = make_scale_dq("scale_b", scale_b_data);

    // Single shared float bias initializer feeding both BN nodes.
    builder.MakeInitializer<float>("shared_bias", {num_channels}, bias_data);

    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);
    builder.MakeInitializer<float>("mean", {num_channels}, mean_vals);
    builder.MakeInitializer<float>("var", {num_channels}, var_vals);

    std::vector<ONNX_NAMESPACE::AttributeProto> bn_attrs;
    bn_attrs.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    builder.AddNode("bn_a", "BatchNormalization", {input_dq, scale_a_dq, "shared_bias", "mean", "var"},
                    {"a_out"}, "", bn_attrs);
    std::vector<ONNX_NAMESPACE::AttributeProto> bn_attrs_b;
    bn_attrs_b.push_back(builder.MakeScalarAttribute("epsilon", 1e-5f));
    builder.AddNode("bn_b", "BatchNormalization", {input_dq, scale_b_dq, "shared_bias", "mean", "var"},
                    {"b_out"}, "", bn_attrs_b);

    builder.AddNode("add", "Add", {"a_out", "b_out"}, {"add_out"});

    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, "output_qdq", "add_out",
                                                      output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

// Two BatchNorm nodes sharing one bias initializer but with different scales must each get their own
// correct fused bias. Reproduces the SiNet w8a8 bug where the EP cached the fused bias by the shared
// initializer name, so the second BN reused the first node's bias and accuracy collapsed
TEST_F(QnnHTPBackendTests, BatchNorm2D_SharedBiasInitializer) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  TestInputDef<float> input_def({2, num_channels, 2, 2}, false, input_data);
  // Different scales per branch => different correct fused biases despite the shared bias initializer.
  std::vector<float> scale_a_data = {1.0f, 2.0f};
  std::vector<float> scale_b_data = {3.0f, 0.5f};
  std::vector<float> bias_data = {1.1f, 2.1f};

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildSharedBiasBatchNormFloatTestCase<float>(input_def, scale_a_data, scale_b_data, bias_data),
                       BuildSharedBiasBatchNormQDQTestCase<uint8_t, int8_t>(input_def, scale_a_data, scale_b_data,
                                                                            bias_data),
                       provider_options,
                       21,
                       ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Input whose channel c holds a fixed pattern scaled by amplitudes[c], laid out [batch, channel, spatial...].
static std::vector<float> MakeChannelAmplitudeData(int64_t batch, const std::vector<float>& amplitudes,
                                                   int64_t spatial_size) {
  const int64_t num_channels = static_cast<int64_t>(amplitudes.size());
  std::vector<float> data;
  data.reserve(static_cast<size_t>(batch * num_channels * spatial_size));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t i = 0; i < spatial_size; ++i) {
        const float pattern = static_cast<float>((i * 7 + c * 3 + b * 5) % 9 - 4) / 4.0f;  // [-1, 1]
        data.push_back(amplitudes[static_cast<size_t>(c)] * pattern);
      }
    }
  }
  return data;
}

// Geometric progression from lo to hi over n values.
static std::vector<float> GeometricRange(float lo, float hi, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = lo * std::pow(hi / lo, static_cast<float>(i) / static_cast<float>(n - 1));
  }
  return v;
}

// With no SoC given the EP does not know the simulator's HTP arch (the backend itself defaults to v68), and an
// unknown arch counts as older than v73. Pin SM8550 (v73) there, as the other BN tests do.
static std::string DefaultSocModelForSimulator() {
#if defined(__linux__) && !defined(__aarch64__)
  return std::to_string(QNN_SOC_MODEL_SM8550);
#else
  return "";
#endif
}

// Op counts expected in the compiled QNN graph. With offload_graph_io_quantization=0 the float graph input is
// quantized and the float graph output dequantized inside QNN, so a BN that stays on the quantized datapath leaves
// one Quantize and one Dequantize. A Dequantize -> BN(F32) -> Quantize island adds one more of each.
struct ExpectedBatchNormGraph {
  size_t batchnorm = 0;
  size_t depthwise_conv = 0;
  size_t quantize = 1;
  size_t dequantize = 1;
};

// Runs the per-channel-scale QDQ BatchNorm on HTP, compares it with the CPU EP running the same QDQ model, and checks
// the compiled QNN graph.
template <typename InputQType>
static void RunPerChannelScaleBatchNormTest(const std::string& test_name,
                                            const TestInputDef<float>& input_def,
                                            const TestInputDef<float>& scale_def,
                                            const TestInputDef<float>& bias_def,
                                            const ExpectedBatchNormGraph& expected_graph,
                                            QDQTolerance tolerance = QDQTolerance(),
                                            const std::string& soc_model = DefaultSocModelForSimulator()) {
  const std::filesystem::path json_qnn_graph_dir = test_name;
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  if (!soc_model.empty()) {
    provider_options["soc_model"] = soc_model;
  }

  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildPerChannelScaleQDQBatchNormTestCase<InputQType>(input_def, scale_def, bias_def),
                       provider_options, 21, ExpectedEPNodeAssignment::All, tolerance);

  // TestQDQModelAccuracy issues GTEST_SKIP() from the helper on unsupported HTP archs; without a dump there is
  // nothing to assert.
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpInQnnGraph(json_qnn_graph_dir, "Batchnorm", expected_graph.batchnorm);
  AssertOpInQnnGraph(json_qnn_graph_dir, "DepthWiseConv2d", expected_graph.depthwise_conv);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Quantize", expected_graph.quantize);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", expected_graph.dequantize);
}

// Per-channel scale params with per-channel fused scales spanning ~4 orders of magnitude, with both signs. Folding
// them into one per-tensor 8-bit scale zeroes the small channels and clips the large ones. The exact answer is a 1x1
// depthwise conv with per-channel weights, so BN must become one DepthWiseConv2d and no Q/DQ around it.
TEST_F(QnnHTPBackendTests, BatchNorm2D_PerChannelParams_DepthwiseConv_U8) {
  constexpr int64_t channels = 16;
  std::vector<float> amplitudes = GeometricRange(0.25f, 8.0f, channels);
  std::vector<float> gamma = GeometricRange(0.05f, 6.0f, channels);
  std::vector<float> beta(channels);
  for (int64_t c = 0; c < channels; ++c) {
    gamma[c] *= (c % 3 == 1) ? -1.0f : 1.0f;
    beta[c] = 0.1f * static_cast<float>(c % 5) - 0.2f;
  }

  TestInputDef<float> input_def({2, channels, 4, 4}, false, MakeChannelAmplitudeData(2, amplitudes, 16));
  TestInputDef<float> scale_def({channels}, true, gamma);
  TestInputDef<float> bias_def({channels}, true, beta);

  RunPerChannelScaleBatchNormTest<uint8_t>("BatchNorm2D_PerChannelParams_DepthwiseConv_U8", input_def, scale_def,
                                           bias_def, {/*batchnorm*/ 0, /*depthwise_conv*/ 1, /*quantize*/ 1,
                                                      /*dequantize*/ 1});
}

// uint16 activations keep the exact float island: a per-channel DepthWiseConv2d with uint16 activations fails
// graph finalization when it feeds the output Transpose, and validation cannot see it.
TEST_F(QnnHTPBackendTests, BatchNorm2D_PerChannelParams_KeepsFloatIsland_U16) {
  constexpr int64_t channels = 16;
  std::vector<float> amplitudes = GeometricRange(0.25f, 8.0f, channels);
  std::vector<float> gamma = GeometricRange(0.05f, 6.0f, channels);
  std::vector<float> beta(channels, 0.3f);
  for (int64_t c = 0; c < channels; c += 4) gamma[c] = -gamma[c];

  TestInputDef<float> input_def({1, channels, 4, 4}, false, MakeChannelAmplitudeData(1, amplitudes, 16));
  TestInputDef<float> scale_def({channels}, true, gamma);
  TestInputDef<float> bias_def({channels}, true, beta);

  RunPerChannelScaleBatchNormTest<uint16_t>("BatchNorm2D_PerChannelParams_KeepsFloatIsland_U16", input_def, scale_def,
                                            bias_def, {/*batchnorm*/ 1, /*depthwise_conv*/ 0, /*quantize*/ 2,
                                                       /*dequantize*/ 2});
}

#if defined(__linux__) && !defined(__aarch64__)
// On HTP v68 a per-channel DepthWiseConv2d that feeds the layout Transpose can fail graph finalization, so BN keeps
// the float island. SM8350 maps to v68; the arch of a real device is fixed by the hardware, so simulator only.
TEST_F(QnnHTPBackendTests, BatchNorm2D_PerChannelParams_KeepsFloatIsland_V68_U8) {
  constexpr int64_t channels = 16;
  std::vector<float> amplitudes = GeometricRange(0.25f, 8.0f, channels);
  std::vector<float> gamma = GeometricRange(0.05f, 6.0f, channels);
  std::vector<float> beta(channels, 0.1f);

  TestInputDef<float> input_def({1, channels, 4, 4}, false, MakeChannelAmplitudeData(1, amplitudes, 16));
  TestInputDef<float> scale_def({channels}, true, gamma);
  TestInputDef<float> bias_def({channels}, true, beta);

  RunPerChannelScaleBatchNormTest<uint8_t>("BatchNorm2D_PerChannelParams_KeepsFloatIsland_V68_U8", input_def,
                                           scale_def, bias_def,
                                           {/*batchnorm*/ 1, /*depthwise_conv*/ 0, /*quantize*/ 2,
                                            /*dequantize*/ 2},
                                           QDQTolerance(), std::to_string(QNN_SOC_MODEL_SM8350));
}
#endif

// Channels the fused scale cannot see: gamma == 0 (dead channel, output is the bias alone), a constant channel whose
// variance is ~0 so gamma / sqrt(var + eps) is huge, and a channel with a tiny gamma.
TEST_F(QnnHTPBackendTests, BatchNorm2D_PerChannelParams_DepthwiseConv_DeadAndExtremeChannels) {
  constexpr int64_t channels = 8;
  std::vector<float> amplitudes = {2.0f, 3.0f, 0.0f, 1.0f, 4.0f, 0.5f, 2.5f, 1.5f};  // channel 2 is constant zero
  std::vector<float> gamma = {1.0f, 0.0f, 0.5f, 1e-3f, -2.0f, 0.0f, 3.0f, -0.2f};
  std::vector<float> beta = {0.0f, 0.7f, 0.25f, -0.5f, 0.1f, -0.3f, 0.0f, 0.4f};

  TestInputDef<float> input_def({1, channels, 3, 3}, false, MakeChannelAmplitudeData(1, amplitudes, 9));
  TestInputDef<float> scale_def({channels}, true, gamma);
  TestInputDef<float> bias_def({channels}, true, beta);

  RunPerChannelScaleBatchNormTest<uint8_t>("BatchNorm2D_PerChannelParams_DepthwiseConv_DeadAndExtremeChannels",
                                           input_def, scale_def, bias_def, {0, 1, 1, 1});
}

// The decomposition needs the channel on a conv axis. A rank-3 BN keeps the exact float island.
TEST_F(QnnHTPBackendTests, BatchNorm1D_PerChannelParams_KeepsFloatIsland_U8) {
  constexpr int64_t channels = 8;
  std::vector<float> amplitudes = GeometricRange(0.5f, 4.0f, channels);
  std::vector<float> gamma = GeometricRange(0.1f, 3.0f, channels);
  std::vector<float> beta(channels, 0.1f);

  TestInputDef<float> input_def({2, channels, 6}, false, MakeChannelAmplitudeData(2, amplitudes, 6));
  TestInputDef<float> scale_def({channels}, true, gamma);
  TestInputDef<float> bias_def({channels}, true, beta);

  RunPerChannelScaleBatchNormTest<uint8_t>("BatchNorm1D_PerChannelParams_KeepsFloatIsland_U8", input_def, scale_def,
                                           bias_def, {/*batchnorm*/ 1, /*depthwise_conv*/ 0, /*quantize*/ 2,
                                                      /*dequantize*/ 2});
}

// Per-tensor params fold exactly into a single quantized QNN Batchnorm; it must not be rewritten.
TEST_F(QnnHTPBackendTests, BatchNorm2D_PerTensorParams_StaysQuantizedBatchnorm_U8) {
  constexpr int64_t num_channels = 2;
  const std::filesystem::path json_qnn_graph_dir = "BatchNorm2D_PerTensorParams_StaysQuantizedBatchnorm_U8";
  std::filesystem::remove_all(json_qnn_graph_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_qnn_graph_dir));
  auto cleanup = gsl::finally([&json_qnn_graph_dir]() { std::filesystem::remove_all(json_qnn_graph_dir); });

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_qnn_graph_dir.string();
  if (const std::string soc_model = DefaultSocModelForSimulator(); !soc_model.empty()) {
    provider_options["soc_model"] = soc_model;
  }

  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};
  TestInputDef<float> input_def({2, num_channels, 2, 2}, false, input_data);
  TestInputDef<float> scale_def({num_channels}, true, {1.0f, 2.0f});
  TestInputDef<float> bias_def({num_channels}, true, {1.1f, 2.1f});

  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildQDQBatchNormTestCase<uint8_t, uint8_t>(input_def, scale_def, bias_def),
                       provider_options, 21, ExpectedEPNodeAssignment::All);
  if (::testing::Test::IsSkipped()) {
    return;
  }

  AssertOpInQnnGraph(json_qnn_graph_dir, "Batchnorm", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "DepthWiseConv2d", 0);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Quantize", 1);
  AssertOpInQnnGraph(json_qnn_graph_dir, "Dequantize", 1);
}

}  // namespace test
}  // namespace onnxruntime

#endif
