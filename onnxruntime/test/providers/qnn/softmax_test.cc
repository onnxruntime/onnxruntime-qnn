// Copyright (c) Qualcomm. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include <filesystem>

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gsl/gsl"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Regression: Softmax(axis=1) fed by a MatMul+Add that ORT's MatMulAddFusion rewrites to Gemm.
// The Softmax OpBuilder's transpose-insertion path must register a tensor wrapper for its
// input during GetCapability validation, since upstream node groups do not populate the
// shared QnnModelWrapper's tensor map in the validate phase. Without that, the op falls
// to CPU and fragments the graph.
namespace {
GetTestModelFn BuildMatMulAddSoftmaxNonLastAxisTestCase(int64_t K, int64_t N) {
  return [K, N](ModelTestBuilder& builder) {
    const std::vector<int64_t> input_shape = {1, 2, K};
    const std::vector<int64_t> weight_shape = {K, N};
    const std::vector<int64_t> bias_shape = {N};

    builder.MakeInput<float>("X", input_shape, -1.0f, 1.0f);
    builder.MakeInitializer<float>("W", weight_shape, -1.0f, 1.0f);
    builder.MakeInitializer<float>("B", bias_shape, -1.0f, 1.0f);

    builder.AddNode("node_MatMul", "MatMul", {"X", "W"}, {"val_mm"});
    builder.AddNode("node_linear_17", "Add", {"val_mm", "B"}, {"linear_17"});
    builder.AddNode("node_softmax", "Softmax", {"linear_17"}, {"softmax_out"},
                    /*domain=*/"",
                    {test::MakeAttribute("axis", static_cast<int64_t>(1))});

    builder.MakeOutput("softmax_out");
  };
}
}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

template <typename InputQType = uint8_t>
static void RunQDQOpTest(const std::string& op_type,
                         const std::vector<TestInputDef<float>>& input_defs,
                         const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                         int opset_version,
                         ExpectedEPNodeAssignment expected_ep_assignment,
                         const std::string& op_domain = kOnnxDomain,
                         bool use_contrib_qdq = false,
                         QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain),
                       BuildQDQOpTestCase<InputQType>(op_type + "_node", op_type, input_defs, {}, attrs, op_domain, use_contrib_qdq),
                       provider_options,
                       opset_version,
                       expected_ep_assignment,
                       tolerance);
}

TEST_F(QnnHTPBackendTests, Softmax13DefaultAxis) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Softmax13DefaultAxisU16) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint16_t>("Softmax",
                         {TestInputDef<float>({1, 2, 3}, false, input_data)},
                         {},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

// QNN EP wraps Softmax with transposes when axis != rank-1.
TEST_F(QnnHTPBackendTests, Softmax13NonLastAxis) {
  const std::vector<float> input_data = {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f, 100.0f, 110.0f, 120.0f,
                                         1.0856307f, 0.99734545f, 0.2829785f, 1.5062947f, 0.5786002f, 1.6514366f,
                                         2.4266791f, 0.42891264f, 1.2659363f};
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3, 3}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// Partner-model shape.
TEST_F(QnnHTPBackendTests, Softmax13NonLastAxisLargeInput) {
  const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 124, 1}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Softmax13NonLastAxisLargeInputU16) {
  const std::vector<float> input_data = GetFloatDataInRange(-50.0f, 50.0f, 124);
  RunQDQOpTest<uint16_t>("Softmax",
                         {TestInputDef<float>({1, 124, 1}, false, input_data)},
                         {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                         13,
                         ExpectedEPNodeAssignment::All,
                         kOnnxDomain,
                         true);
}

TEST_F(QnnHTPBackendTests, Softmax11DefaultAxis) {
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Softmax11LastAxis) {
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, -5.0f, 5.0f)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        11,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogSoftmax13DefaultAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogSoftmax13NonLastAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogSoftmax11DefaultAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {},
                        11,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LogSoftmax11LastAxis) {
  std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        11,
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Softmax13NonLastAxisAfterMatMulAddFusion) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildMatMulAddSoftmaxNonLastAxisTestCase(/*K=*/128, /*N=*/1),
                  provider_options,
                  /*opset=*/18,
                  EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(2e-3f)});
}

// Bounded-output split tests.
//
// A quantized Softmax output is in [0, 1] and is naturally encoded with zero-point 0. When the
// quantizer instead assigns a symmetric output encoding (zero-point != 0) -- e.g. because the
// output feeds an op that requires a symmetric input -- the QNN EP emits the Softmax with its
// natural (zero-point 0) encoding followed by a QNN_OP_CONVERT to the original encoding. These
// tests verify that the split is value-preserving (output matches the CPU reference) and does not
// fragment the graph. (The underlying HTP softmax-kernel accuracy bug only manifests on real
// hardware; here we guard that the fix is lossless and keeps the op on QNN.)
namespace {
// Builds a QDQ Softmax model whose output Q/DQ uses a forced SYMMETRIC encoding (zero-point at the
// mid-point of the quantized range), which triggers the bounded-output split in the QNN EP.
template <typename QType>
static GetTestQDQModelFn<QType> BuildQDQSoftmaxSymmetricOutputTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    bool use_contrib_qdq = false) {
  return [input_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                             std::vector<QuantParams<QType>>& output_qparams) {
    // input -> Q -> DQ
    MakeTestInput<float>(builder, "X", input_def);
    QuantParams<QType> input_qparams = GetTestInputQuantParams<QType>(input_def);
    const std::string input_qdq = AddQDQNodePair<QType>(builder, "qdq_in", "X", input_qparams.scale,
                                                        input_qparams.zero_point, use_contrib_qdq);

    // DQ -> Softmax
    builder.AddNode("Softmax_node", "Softmax", {input_qdq}, {"softmax_out"}, kOnnxDomain, attrs);

    // Force a symmetric output encoding (zero-point != 0) over the [0, 1] softmax range. This is
    // what triggers the EP's Softmax(natural) -> Convert -> output split.
    // softmax_out -> Q(symmetric) -> DQ -> graph output
    output_qparams[0] = QuantParams<QType>::Compute(0.0f, 1.0f, /*symmetric*/ true);
    AddQDQNodePairWithOutputAsGraphOutput<QType>(builder, "qdq_out", "softmax_out",
                                                 output_qparams[0].scale,
                                                 output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// Builds a QDQ Softmax model whose output Q/DQ uses a forced CALIBRATED-ASYMMETRIC encoding: a
// zero-point that is non-zero but NOT at the mid-point of the quantized range (e.g. derived from a
// slightly-negative calibrated min)
template <typename QType>
static GetTestQDQModelFn<QType> BuildQDQSoftmaxAsymmetricOutputTestCase(
    const TestInputDef<float>& input_def,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    bool use_contrib_qdq = false) {
  return [input_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                             std::vector<QuantParams<QType>>& output_qparams) {
    // input -> Q -> DQ
    MakeTestInput<float>(builder, "X", input_def);
    QuantParams<QType> input_qparams = GetTestInputQuantParams<QType>(input_def);
    const std::string input_qdq = AddQDQNodePair<QType>(builder, "qdq_in", "X", input_qparams.scale,
                                                        input_qparams.zero_point, use_contrib_qdq);

    // DQ -> Softmax
    builder.AddNode("Softmax_node", "Softmax", {input_qdq}, {"softmax_out"}, kOnnxDomain, attrs);

    // Force a calibrated-asymmetric output encoding: a slightly-negative calibrated min yields a
    // small non-zero zero-point that is NOT the symmetric mid-point (offset = -1, not -2^(bw-1)).
    // This mirrors a terminal classification softmax. The EP must treat it as a normal QDQ output
    // (no Convert split). softmax_out -> Q(asymmetric) -> DQ -> graph output.
    output_qparams[0] = QuantParams<QType>::Compute(-0.004f, 1.0f, /*symmetric*/ false);
    AddQDQNodePairWithOutputAsGraphOutput<QType>(builder, "qdq_out", "softmax_out",
                                                 output_qparams[0].scale,
                                                 output_qparams[0].zero_point, use_contrib_qdq);
  };
}

template <typename QType>
static void RunQDQSoftmaxAsymmetricOutputTest(const TestInputDef<float>& input_def,
                                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                              int opset_version,
                                              ExpectedEPNodeAssignment expected_ep_assignment,
                                              bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>("Softmax_node", "Softmax", {input_def}, {}, attrs),
                       BuildQDQSoftmaxAsymmetricOutputTestCase<QType>(input_def, attrs, use_contrib_qdq),
                       provider_options,
                       opset_version,
                       expected_ep_assignment);
}

template <typename QType>
static void RunQDQSoftmaxSymmetricOutputTest(const TestInputDef<float>& input_def,
                                             const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                             int opset_version,
                                             ExpectedEPNodeAssignment expected_ep_assignment,
                                             bool use_contrib_qdq = false,
                                             bool assert_convert_in_graph = false) {
  namespace fs = std::filesystem;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // When asserting, dump the lowered QNN graph JSON so we can check it contains exactly one
  // Convert, i.e. the bounded-output split emitted Softmax(natural) -> Convert -> output(symmetric).

  const bool check_graph = assert_convert_in_graph;
  const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  const fs::path graph_dir = fs::temp_directory_path() /
                             (std::string("softmax_split_qnn_graph_") + test_info->name());
  if (check_graph) {
    fs::remove_all(graph_dir);
    fs::create_directories(graph_dir);
    provider_options["dump_json_qnn_graph"] = "1";
    provider_options["json_qnn_graph_dir"] = graph_dir.string();
  }
  auto cleanup = gsl::finally([&]() {
    if (check_graph) fs::remove_all(graph_dir);
  });

  TestQDQModelAccuracy(BuildOpTestCase<float>("Softmax_node", "Softmax", {input_def}, {}, attrs),
                       BuildQDQSoftmaxSymmetricOutputTestCase<QType>(input_def, attrs, use_contrib_qdq),
                       provider_options,
                       opset_version,
                       expected_ep_assignment);

  if (check_graph && !::testing::Test::IsSkipped()) {
    AssertOpInQnnGraph(graph_dir, "Convert", 1);
  }
}
}  // namespace

// (1) Symmetric (zero-point != 0) uint8 Softmax output -> split path. Last-axis (direct path).
// Value-preserving + stays on QNN.
TEST_F(QnnHTPBackendTests, Softmax_SymmetricOutput_U8_Split) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQSoftmaxSymmetricOutputTest<uint8_t>({TestInputDef<float>({1, 2, 3}, false, input_data)},
                                            {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                            13,
                                            ExpectedEPNodeAssignment::All,
                                            /*use_contrib_qdq*/ false,
                                            /*assert_convert_in_graph*/ true);
}

// (2) Symmetric uint8 Softmax output that is also a graph output -> split path with the converted
// output tensor as the graph output (APP_READ).
TEST_F(QnnHTPBackendTests, Softmax_SymmetricOutput_U8_GraphOutput_Split) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 24);
  RunQDQSoftmaxSymmetricOutputTest<uint8_t>({TestInputDef<float>({1, 2, 3, 4}, false, input_data)},
                                            {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                            13,
                                            ExpectedEPNodeAssignment::All,
                                            /*use_contrib_qdq*/ false,
                                            /*assert_convert_in_graph*/ true);
}

// (3) Natural (zero-point 0) uint8 Softmax output -> NO split. Standard QDQ harness derives a
// zero-point-0 encoding from the [0, 1] range, so the EP must not insert a Convert. Stays on QNN.
TEST_F(QnnHTPBackendTests, Softmax_NaturalOutput_U8_NoSplit) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("Softmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// (4) Symmetric uint16 Softmax output -> split path also applies at 16-bit (no bitwidth gate).
TEST_F(QnnHTPBackendTests, Softmax_SymmetricOutput_U16_Split) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQSoftmaxSymmetricOutputTest<uint16_t>({TestInputDef<float>({1, 2, 3}, false, input_data)},
                                             {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                             13,
                                             ExpectedEPNodeAssignment::All,
                                             /*use_contrib_qdq*/ true);
}

// (10) LogSoftmax is excluded from the split (its output is (-inf, 0], not [0, 1]). Even with a
// symmetric output encoding it must NOT take the unit-range split path; it stays on QNN and is
// value-preserving via the normal path.
TEST_F(QnnHTPBackendTests, LogSoftmax_SymmetricOutput_U8_NoSplit) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQOpTest<uint8_t>("LogSoftmax",
                        {TestInputDef<float>({1, 2, 3}, false, input_data)},
                        {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                        13,
                        ExpectedEPNodeAssignment::All);
}

// (11) Calibrated-asymmetric uint8 Softmax output (zero-point != 0 but NOT the symmetric mid-point,
// e.g. a terminal classification softmax with a slightly-negative calibrated min) -> NO split. Only
// the symmetric mid-point encoding (zp at 2^(bw-1)) triggers the Convert; this guards against the
// over-trigger that perturbed classification-model numerics.
TEST_F(QnnHTPBackendTests, Softmax_AsymmetricOutput_U8_NoSplit) {
  const std::vector<float> input_data = GetFloatDataInRange(-5.0f, 5.0f, 6);
  RunQDQSoftmaxAsymmetricOutputTest<uint8_t>({TestInputDef<float>({1, 2, 3}, false, input_data)},
                                             {test::MakeAttribute("axis", static_cast<int64_t>(-1))},
                                             13,
                                             ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
