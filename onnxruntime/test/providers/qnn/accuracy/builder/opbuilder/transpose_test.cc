// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// Focused HTP accuracy coverage for the standalone Transpose OpBuilder.

#if !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <gsl/util>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename DataType>
GetTestModelFn BuildTransposeModel(TestInputDef<DataType> input_def,
                                   std::vector<ONNX_NAMESPACE::AttributeProto> attrs) {
  return [input_def = std::move(input_def), attrs = std::move(attrs)](ModelTestBuilder& builder) {
    MakeTestInput<DataType>(builder, "input", input_def);
    builder.MakeOutput("Y");
    builder.AddNode("Transpose", "Transpose", {"input"}, {"Y"}, kOnnxDomain, attrs);
  };
}

template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQTransposeModel(
    const TestInputDef<float>& input_def, const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  return [input_def, attrs](
             ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) {
    QNN_TEST_UNUSED_PARAMETER(output_qparams);
    MakeTestInput(builder, "input", input_def);
    const QuantParams<QuantType> qparams = GetTestInputQuantParams<QuantType>(input_def);
    const std::string quantized_input =
        AddQDQNodePair<QuantType>(builder, "qdq_in", "input", qparams.scale, qparams.zero_point);
    builder.AddNode("Transpose", "Transpose", {quantized_input}, {"Y"}, kOnnxDomain, attrs);
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, "qdq_out", "Y",
                                                     qparams.scale, qparams.zero_point);
  };
}

ProviderOptions MakeHtpOptions() {
  ProviderOptions options;
  options["backend_type"] = "htp";
  return options;
}

template <typename DataType>
void RunTransposeAccuracy(TestInputDef<DataType> input_def,
                          const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                          float tolerance) {
  RunQnnModelTest(BuildTransposeModel(std::move(input_def), attrs), MakeHtpOptions(),
                  /*opset_version=*/13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All,
                                       ElementwiseAbsoluteVerifier(tolerance)});
}

void RunTransposeQdqU8Accuracy(TestInputDef<float> input_def,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs) {
  auto options = MakeHtpOptions();
  options["offload_graph_io_quantization"] = "0";
  TestQDQModelAccuracy(BuildTransposeModel(input_def, attrs),
                       BuildQDQTransposeModel<uint8_t>(input_def, attrs),
                       options, /*opset_version=*/18,
                       ExpectedEPNodeAssignment::All);
}

const std::vector<ONNX_NAMESPACE::AttributeProto>& NormalPerm() {
  static const std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      test::MakeAttribute("perm", std::vector<int64_t>{0, 2, 3, 1})};
  return attrs;
}

const std::vector<ONNX_NAMESPACE::AttributeProto>& UnitDimensionPerm() {
  static const std::vector<ONNX_NAMESPACE::AttributeProto> attrs = {
      test::MakeAttribute("perm", std::vector<int64_t>{1, 0, 2})};
  return attrs;
}

class QnnAcc_Transpose_AccuracyTest : public QnnHTPBackendTests {};

}  // namespace

TEST_F(QnnAcc_Transpose_AccuracyTest, NormalPerm_QdqU8) {
  RunTransposeQdqU8Accuracy(TestInputDef<float>({1, 3, 224, 128}, false, 0.0f, 1.0f), NormalPerm());
}

TEST_F(QnnAcc_Transpose_AccuracyTest, NormalPerm_Int32) {
  RunTransposeAccuracy(TestInputDef<int32_t>({1, 3, 224, 128}, false, -100, 100), NormalPerm(), 1e-5f);
}

TEST_F(QnnAcc_Transpose_AccuracyTest, NormalPerm_Fp32) {
  RunTransposeAccuracy(TestInputDef<float>({1, 3, 224, 128}, false, 0.0f, 10.0f), NormalPerm(), 5e-3f);
}

TEST_F(QnnAcc_Transpose_AccuracyTest, UnitDimensionPerm_LowersToReshape_Fp32) {
  const auto slot = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto directory = std::filesystem::temp_directory_path() / ("qnn_transpose_unit_dim_" + slot);
  ASSERT_TRUE(std::filesystem::create_directory(directory));
  auto cleanup = gsl::finally([&directory]() { std::filesystem::remove_all(directory); });
  auto options = MakeHtpOptions();
  options["dump_json_qnn_graph"] = "1";
  options["json_qnn_graph_dir"] = directory.string();

  RunQnnModelTest(BuildTransposeModel(TestInputDef<float>({128, 1, 128}, false, 0.0f, 1.0f),
                                      UnitDimensionPerm()),
                  options, 13,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(5e-3f)});
  AssertOpInQnnGraph(directory, "Transpose", 0);
  AssertOpInQnnGraph(directory, "Reshape", 1);
}

TEST_F(QnnAcc_Transpose_AccuracyTest, UnitDimensionPerm_LowersToReshape_QdqU8) {
  RunTransposeQdqU8Accuracy(TestInputDef<float>({128, 1, 128}, false, 0.0f, 1.0f), UnitDimensionPerm());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) && QNN_EP_ENABLE_ACCURACY_TESTS
