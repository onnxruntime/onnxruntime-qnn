// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT
//
// Focused HTP QDQ accuracy and fallback-routing coverage for ArgMax and ArgMin.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename QType>
GetTestQDQModelFn<QType> BuildQDQArgMaxMinModel(
    const std::string& op_type, TestInputDef<float> input_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [op_type, input_def, attrs](ModelTestBuilder& builder,
                                     std::vector<QuantParams<QType>>& output_qparams) {
    QNN_TEST_UNUSED_PARAMETER(output_qparams);
    MakeTestInput(builder, "X", input_def);
    const QuantParams<QType> input_qparams = GetTestInputQuantParams<QType>(input_def);
    const std::string x_dq_name =
        AddQDQNodePair<QType>(builder, "qdq1", "X", input_qparams.scale, input_qparams.zero_point);
    builder.AddNode("argm", op_type, {x_dq_name.c_str()}, {"Y"}, "", attrs);
    builder.MakeOutput("Y");
  };
}

void RunArgMaxMinQdqU8(const std::string& op_type, TestInputDef<float> input_def,
                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                       ExpectedEPNodeAssignment expected_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(
      BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, attrs),
      BuildQDQArgMaxMinModel<uint8_t>(op_type, input_def, attrs), provider_options,
      /*opset=*/13, expected_assignment);
}

void RunBothArgMaxMin(TestInputDef<float> input_def,
                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                      ExpectedEPNodeAssignment expected_assignment) {
  RunArgMaxMinQdqU8("ArgMax", input_def, attrs, expected_assignment);
  RunArgMaxMinQdqU8("ArgMin", std::move(input_def), attrs, expected_assignment);
}

class QnnAcc_ArgMaxMin_AccuracyTest : public QnnHTPBackendTests {};

}  // namespace

TEST_F(QnnAcc_ArgMaxMin_AccuracyTest, DefaultAttrs_QdqU8) {
  RunArgMaxMinQdqU8("ArgMax", TestInputDef<float>({1, 3, 4}, false, -10.0f, 10.0f), {},
                    ExpectedEPNodeAssignment::All);
  RunArgMaxMinQdqU8("ArgMin", TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f), {},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnAcc_ArgMaxMin_AccuracyTest, AxisLast_QdqU8) {
  RunBothArgMaxMin(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                   {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                   ExpectedEPNodeAssignment::All);
}

TEST_F(QnnAcc_ArgMaxMin_AccuracyTest, KeepDimsFalse_QdqU8) {
  RunArgMaxMinQdqU8("ArgMax", TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                    {test::MakeAttribute("axis", static_cast<int64_t>(-1)),
                     test::MakeAttribute("keepdims", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
  RunArgMaxMinQdqU8("ArgMin", TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                    {test::MakeAttribute("keepdims", static_cast<int64_t>(0))},
                    ExpectedEPNodeAssignment::All);
}

TEST_F(QnnAcc_ArgMaxMin_AccuracyTest, SelectLastIndexNonZero_FallsBack) {
  RunBothArgMaxMin(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                   {test::MakeAttribute("select_last_index", static_cast<int64_t>(1))},
                   ExpectedEPNodeAssignment::None);
}

TEST_F(QnnAcc_ArgMaxMin_AccuracyTest, RankGreaterThanFour_FallsBack) {
  RunBothArgMaxMin(TestInputDef<float>({1, 3, 4, 4, 4}, false, -10.0f, 10.0f), {},
                   ExpectedEPNodeAssignment::None);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS
