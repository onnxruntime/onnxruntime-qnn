// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Helper to build a NonZero test model.
// Output shape is [rank, num_elements] where num_elements is the total number of elements
// in the input tensor (max possible non-zero count).
template <typename DataType>
static GetTestModelFn BuildNonZeroTestCase(const std::vector<int64_t>& input_shape,
                                           const std::vector<DataType>& input_data) {
  TestInputDef<DataType> input_def(input_shape, false, input_data);
  int64_t input_rank = static_cast<int64_t>(input_shape.size());
  int64_t num_elements = 1;
  for (int64_t dim : input_shape) {
    num_elements *= dim;
  }

  return [input_def, input_rank, num_elements](ModelTestBuilder& builder) {
    MakeTestInput<DataType>(builder, "X", input_def);
    builder.AddNode("nonzero_node", "NonZero", {"X"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput<int64_t>("Y", std::vector<int64_t>{input_rank, num_elements});
  };
}

static ProviderOptions HtpProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

// NonZero-specific test runner that handles padded output comparison.
// QNN returns [rank, num_elements] (padded to max), while CPU EP returns [rank, actual_count].
// This compares only the valid region of the QNN output against the CPU EP baseline.
static void RunNonZeroTest(const GetTestModelFn& build_test_case,
                           int opset_version,
                           ExpectedEPNodeAssignment expected_ep_assignment =
                               ExpectedEPNodeAssignment::All,
                           ProviderOptions provider_options = HtpProviderOptions()) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};
  ModelTestBuilder helper;
  build_test_case(helper);

  CONDITIONAL_SKIP_TEST_ON_LINUX_ARM64(provider_options, QNN_HTP_DEVICE_ARCH_V68, "FP16");

  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  // Run on CPU EP -> expected output [rank, actual_count]
  std::vector<Ort::Value> expected;
  InferenceModelCPU(model_data, "NonZero_CPU", helper.feeds_, expected);

  // Run on QNN EP -> actual output [rank, num_elements] (padded)
  std::vector<Ort::Value> actual;
  InferenceModel(model_data, "NonZero_QNN", provider_options,
                 expected_ep_assignment, helper.feeds_, actual);

  // NonZero has exactly one output
  uint32_t output_idx = 0;

  auto exp_shape = expected[output_idx].GetTensorTypeAndShapeInfo().GetShape();
  auto act_shape = actual[output_idx].GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(exp_shape.size(), 2) << "NonZero output must be 2D";
  ASSERT_EQ(act_shape.size(), 2) << "NonZero output must be 2D";

  int64_t rows = exp_shape[0];
  int64_t exp_cols = exp_shape[1];
  int64_t act_cols = act_shape[1];
  ASSERT_EQ(rows, act_shape[0]) << "Rank dimension mismatch";
  ASSERT_GE(act_cols, exp_cols) << "QNN output should be >= CPU output in column count";

  const int64_t* exp_data = expected[output_idx].GetTensorData<int64_t>();
  const int64_t* act_data = actual[output_idx].GetTensorData<int64_t>();
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < exp_cols; ++c) {
      EXPECT_EQ(exp_data[r * exp_cols + c], act_data[r * act_cols + c])
          << "Mismatch at [" << r << ", " << c << "]";
    }
  }
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Input: [0, 1, 0, 2, 0] -> non-zero at indices 1, 3
TEST_F(QnnHTPBackendTests, NonZero_1D_Float) {
  RunNonZeroTest(BuildNonZeroTestCase<float>({5}, {0.0f, 1.0f, 0.0f, 2.0f, 0.0f}), 11);
}

// Input: [[1, 0, 3], [0, 5, 0]] -> non-zero at (0,0), (0,2), (1,1)
TEST_F(QnnHTPBackendTests, NonZero_2D_Float) {
  RunNonZeroTest(BuildNonZeroTestCase<float>({2, 3}, {1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f}), 11);
}

// Input (QDQ uint8): [[1, 0, 3], [0, 5, 0]] -> non-zero at (0,0), (0,2), (1,1)
// Pattern: float input -> Q(u8) -> DQ(u8) -> NonZero -> int64 output
TEST_F(QnnHTPBackendTests, NonZero_2D_QDQ_Uint8) {
  std::vector<float> input_data = {1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f};
  TestInputDef<float> input_def({2, 3}, false, input_data);
  int64_t input_rank = 2;
  int64_t num_elements = 6;  // 2 * 3

  auto qdq_model_fn = [input_def, input_rank, num_elements](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", input_def);
    std::string dq_out = AddQDQNodePair<uint8_t>(builder, "qdq_in", "X", 0.02f, uint8_t(0));
    builder.AddNode("nonzero_node", "NonZero", {dq_out}, {"Y"}, kOnnxDomain);
    builder.MakeOutput<int64_t>("Y", std::vector<int64_t>{input_rank, num_elements});
  };

  RunNonZeroTest(qdq_model_fn, 11);
}

// Input (QDQ uint8): [[1, 0, 3], [0, 4.1, 0]] -> non-zero at (0,0), (0,2), (1,1)
// Pattern: float input -> Q(u16) -> DQ(u16) -> NonZero -> int64 output
TEST_F(QnnHTPBackendTests, NonZero_2D_QDQ_Uint16) {
  std::vector<float> input_data = {1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f};
  TestInputDef<float> input_def({2, 3}, false, input_data);
  int64_t input_rank = 2;
  int64_t num_elements = 6;  // 2 * 3

  auto qdq_model_fn = [input_def, input_rank, num_elements](ModelTestBuilder& builder) {
    MakeTestInput<float>(builder, "X", input_def);
    std::string dq_out = AddQDQNodePair<uint16_t>(builder, "qdq_in", "X", 0.02f, uint16_t(0));
    builder.AddNode("nonzero_node", "NonZero", {dq_out}, {"Y"}, kOnnxDomain);
    builder.MakeOutput<int64_t>("Y", std::vector<int64_t>{input_rank, num_elements});
  };

  RunNonZeroTest(qdq_model_fn, 21);
}

// Input (bool): [[true, false, true], [false, true, false]] -> non-zero at (0,0), (0,2), (1,1)
TEST_F(QnnHTPBackendTests, NonZero_2D_Bool) {
  RunNonZeroTest(BuildNonZeroTestCase<bool>({2, 3}, {true, false, true, false, true, false}), 11);
}

// Input (bool, 3D): [[[T,F,T],[F,T,F]], [[T,T,F],[F,F,T]]]
TEST_F(QnnHTPBackendTests, NonZero_3D_Bool) {
  RunNonZeroTest(BuildNonZeroTestCase<bool>({2, 2, 3},
                                            {true, false, true, false, true, false,
                                             true, true, false, false, false, true}),
                 11);
}

// Input (bool, 3D): [[[F,F,F],[F,F,F]], [[F,F,F],[F,F,F]]]
TEST_F(QnnHTPBackendTests, NonZero_3D_Bool_All_False) {
  RunNonZeroTest(BuildNonZeroTestCase<bool>({2, 2, 3},
                                            {false, false, false, false, false, false,
                                             false, false, false, false, false, false}),
                 11);
}

// All non-zero: [[1, 2], [3, 4]] -> every element is non-zero
TEST_F(QnnHTPBackendTests, NonZero_AllNonZero) {
  RunNonZeroTest(BuildNonZeroTestCase<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}), 11);
}

// All zero: [[0, 0], [0, 0]] -> every element is zero
TEST_F(QnnHTPBackendTests, NonZero_AllZero) {
  RunNonZeroTest(BuildNonZeroTestCase<float>({2, 2}, {0, 0, 0, 0}), 11);
}

// NonZero -> Gather pattern with two Gather consumers to verify multiple consumers are handled.
// Graph:
//   mask -> NonZero -> [1, N] -> Reshape -> [N] (int64) -> Cast -> [N] (int32) -+-> Gather(data1) -> output1
//                                                                                +-> Gather(data2) -> output2
// NonZero output is declared as a graph output with static shape [1, num_elements] so QNN EP can claim it.
// All mask elements are non-zero to avoid CPU/QNN shape mismatch from NonZero padding.
TEST_F(QnnHTPBackendTests, NonZero_Gather_1D_Int32) {
  std::vector<float> mask_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};  // non-zero at indices 0,1,2,3,4
  std::vector<int32_t> data1 = {10, 20, 30, 40, 50};
  std::vector<int32_t> data2 = {100, 200, 300, 400, 500};
  int64_t num_elements = 5;

  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  auto build_model = [mask_data, data1, data2, num_elements](ModelTestBuilder& builder) {
    TestInputDef<float> mask_def({num_elements}, false, mask_data);
    MakeTestInput<float>(builder, "mask", mask_def);

    // NonZero: output [1, num_elements] — declared as graph output with static shape
    // so QNN EP sees fixed dims and can claim the node
    builder.AddNode("nonzero_node", "NonZero", {"mask"}, {"nonzero_out"}, kOnnxDomain);
    builder.MakeOutput<int64_t>("nonzero_out", std::vector<int64_t>{1, num_elements});

    // Reshape [1, num_elements] -> [num_elements]
    builder.Make1DInitializer<int64_t>("reshape_shape", {num_elements});
    builder.AddNode("reshape_node", "Reshape", {"nonzero_out", "reshape_shape"}, {"indices_i64"}, kOnnxDomain);

    // Cast int64 -> int32 (HTP requires int32 indices for Gather)
    builder.AddNode("cast_node", "Cast", {"indices_i64"}, {"indices_i32"}, kOnnxDomain,
                    {test::MakeAttribute("to", int64_t(ONNX_NAMESPACE::TensorProto_DataType_INT32))});

    // First Gather: data1[indices]
    builder.MakeInitializer<int32_t>("data1", {num_elements}, data1);
    builder.AddNode("gather_node1", "Gather", {"data1", "indices_i32"}, {"output1"}, kOnnxDomain,
                    {test::MakeAttribute("axis", int64_t(0))});
    builder.MakeOutput<int32_t>("output1", std::vector<int64_t>{num_elements});

    // Second Gather: data2[indices]
    builder.MakeInitializer<int32_t>("data2", {num_elements}, data2);
    builder.AddNode("gather_node2", "Gather", {"data2", "indices_i32"}, {"output2"}, kOnnxDomain,
                    {test::MakeAttribute("axis", int64_t(0))});
    builder.MakeOutput<int32_t>("output2", std::vector<int64_t>{num_elements});
  };

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};
  ModelTestBuilder helper;
  build_model(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  // Run on CPU EP
  std::vector<Ort::Value> expected;
  InferenceModelCPU(model_data, "NonZero_Gather_CPU", helper.feeds_, expected);

  // Run on QNN EP (all nodes assigned)
  std::vector<Ort::Value> actual;
  InferenceModel(model_data, "NonZero_Gather_QNN", HtpProviderOptions(),
                 ExpectedEPNodeAssignment::All, helper.feeds_, actual);

  // Outputs: index 0 = NonZero graph output, index 1 = Gather1 output, index 2 = Gather2 output
  for (size_t out_idx : {1, 2}) {
    auto exp_shape = expected[out_idx].GetTensorTypeAndShapeInfo().GetShape();
    auto act_shape = actual[out_idx].GetTensorTypeAndShapeInfo().GetShape();
    ASSERT_EQ(exp_shape, act_shape) << "Shape mismatch for output " << out_idx;

    auto element_count = expected[out_idx].GetTensorTypeAndShapeInfo().GetElementCount();
    const int32_t* exp_data = expected[out_idx].GetTensorData<int32_t>();
    const int32_t* act_data = actual[out_idx].GetTensorData<int32_t>();
    for (size_t i = 0; i < element_count; ++i) {
      EXPECT_EQ(exp_data[i], act_data[i]) << "Mismatch at output " << out_idx << " index " << i;
    }
  }
}

// Negative test: NonZero with dynamic output shape should not be assigned to QNN EP.
TEST_F(QnnHTPBackendTests, NonZero_DynamicOutputShape_Negative) {
  auto build_model = [](ModelTestBuilder& builder) {
    TestInputDef<float> input_def({2, 3}, false, {1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f});
    MakeTestInput<float>(builder, "X", input_def);
    builder.AddNode("nonzero_node", "NonZero", {"X"}, {"Y"}, kOnnxDomain);
    // Dynamic output shape: [rank, -1]
    builder.MakeOutput<int64_t>("Y", std::vector<int64_t>{2, -1});
  };

  RunNonZeroTest(build_model, 11, ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
