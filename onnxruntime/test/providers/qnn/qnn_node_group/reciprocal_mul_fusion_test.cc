// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

// =============================================================================
// Tests for ReciprocalMulFusion
// =============================================================================
//
// Verifies that the two-node ONNX sub-graph
//
//   [denominator] --> Reciprocal --+
//                                  v
//   [numerator]  ----------------> Mul --> [output]
//
// is fused into a single QNN ElementWiseDivide node on the HTP backend, and
// that the numerical output matches the CPU EP reference within tolerance.
//
// Test matrix
// -----------
//   Float32 (fp32)
//     - Basic 4-D input, numerator in Mul input[0]  (standard order)
//     - Basic 4-D input, numerator in Mul input[1]  (commuted order)
//
//   Float16 (fp16)
//     - Basic 4-D input, standard order  (HTP fp16 path)
//
//   QDQ (uint8) -- SingleNode Reciprocal (inputs quantized, Reciprocal bare)
//     - Basic 4-D input, standard order
//     - Basic 4-D input, commuted order
//
//   QDQ (uint16, contrib ops) -- SingleNode Reciprocal
//     - Basic 4-D input, standard order
//
//   QDQ (uint8) -- QDQGroup Reciprocal (DQ -> Reciprocal -> Q)
//     - Basic 4-D input, standard order  (LayerNorm rstd pattern)
//     - Basic 4-D input, commuted order
//
//   Negative / no-fusion cases (fusion blocked)
//     - Reciprocal output is a graph output (float32)
//                                                => blocked by GetChildNodeUnitAllowQdq
//                                                   (graph-output guard);
//                                                   no fusion; float32 Reciprocal is also
//                                                   unsupported by ReciprocalOpBuilder on HTP,
//                                                   so Reciprocal falls back to CPU EP;
//                                                   the Mul node runs independently on QNN EP
//                                                   as ElementWiseMultiply;
//                                                   0 ElementWiseDivide + 1 ElementWiseMultiply
//                                                   in the QNN graph
//     - QDQ-wrapped Reciprocal with two Mul consumers
//                                                => blocked by GetChildNodeUnitAllowQdq
//                                                   (single-consumer guard);
//                                                   no fusion; 1 ElementWiseDivide (op-builder)
//                                                   + 2 ElementWiseMultiply
// =============================================================================

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// ---------------------------------------------------------------------------
// Float32 / Float16 model builders
// ---------------------------------------------------------------------------

// Builds the canonical fusion pattern:
//
//   denominator --> Reciprocal --> recip_out --+
//                                              v
//   numerator   --------------------------------> Mul --> output
//
// When commute=false  =>  Mul(numerator, recip_out)   [recip in slot 1]
// When commute=true   =>  Mul(recip_out, numerator)   [recip in slot 0]
//
// Both orderings must produce the same fused ElementWiseDivide node because
// ONNX Mul is commutative and the fusion code handles both slots.
GetTestModelFn BuildReciprocalMulTestCase(const TestInputDef<float>& numerator_def,
                                          const TestInputDef<float>& denominator_def,
                                          bool commute = false) {
  return [numerator_def, denominator_def, commute](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_mul_fusion_graph");

    MakeTestInput<float>(builder, "numerator", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    // denominator -> Reciprocal -> recip_out
    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    // Mul(numerator, recip_out)  or  Mul(recip_out, numerator)
    std::vector<std::string> mul_inputs = commute
                                              ? std::vector<std::string>{"recip_out", "numerator"}
                                              : std::vector<std::string>{"numerator", "recip_out"};

    builder.AddNode("Mul_node",
                    "Mul",
                    mul_inputs,
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// ---------------------------------------------------------------------------
// Float16 model builder
// ---------------------------------------------------------------------------

// Builds the FP16 version of the fusion pattern by converting both inputs
// from float32 to float16.  Used with TestFp16ModelAccuracy which runs the
// fp32 reference on CPU EP and the fp16 model on QNN EP.
GetTestModelFn BuildReciprocalMulFP16TestCase(const TestInputDef<float>& numerator_def,
                                              const TestInputDef<float>& denominator_def,
                                              bool commute = false) {
  const TestInputDef<Ort::Float16_t> num_fp16_def = ConvertToFP16InputDef(numerator_def);
  const TestInputDef<Ort::Float16_t> den_fp16_def = ConvertToFP16InputDef(denominator_def);

  return [num_fp16_def, den_fp16_def, commute](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_mul_fp16_fusion_graph");

    MakeTestInput<Ort::Float16_t>(builder, "numerator", num_fp16_def);
    MakeTestInput<Ort::Float16_t>(builder, "denominator", den_fp16_def);

    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    std::vector<std::string> mul_inputs = commute
                                              ? std::vector<std::string>{"recip_out", "numerator"}
                                              : std::vector<std::string>{"numerator", "recip_out"};

    builder.AddNode("Mul_node",
                    "Mul",
                    mul_inputs,
                    {"output"},
                    kOnnxDomain);

    builder.MakeOutput("output");
  };
}

// ---------------------------------------------------------------------------
// QDQ model builders
// ---------------------------------------------------------------------------

// Builds the QDQ version of the fusion pattern.
//
// Each float input is wrapped in a Q -> DQ pair before being fed into the
// Reciprocal / Mul nodes, and the Mul output is wrapped in a Q -> DQ pair
// before being exposed as the graph output.  This mirrors the pattern used
// in gelu_fusion_test.cc and hardsigmoid_mul_fusion_test.cc.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQReciprocalMulTestCase(
    const TestInputDef<float>& numerator_def,
    const TestInputDef<float>& denominator_def,
    bool commute = false,
    bool use_contrib_qdq = false) {
  return [numerator_def, denominator_def, commute, use_contrib_qdq](
             ModelTestBuilder& builder,
             std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    builder.graph_->set_name("qdq_reciprocal_mul_fusion_graph");

    MakeTestInput<float>(builder, "numerator", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    const QuantParams<QuantType> num_qparams = GetTestInputQuantParams<QuantType>(numerator_def);
    const QuantParams<QuantType> den_qparams = GetTestInputQuantParams<QuantType>(denominator_def);

    // Wrap inputs in QDQ pairs.
    const std::string num_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_num", "numerator", num_qparams.scale, num_qparams.zero_point, use_contrib_qdq);
    const std::string den_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_den", "denominator", den_qparams.scale, den_qparams.zero_point, use_contrib_qdq);

    // denominator_qdq -> Reciprocal -> recip_out
    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {den_qdq},
                    {"recip_out"},
                    kOnnxDomain);

    // Wrap Reciprocal output in QDQ before feeding into Mul.
    const QuantParams<QuantType> recip_qparams = GetTestInputQuantParams<QuantType>(denominator_def);
    const std::string recip_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_recip", "recip_out", recip_qparams.scale, recip_qparams.zero_point, use_contrib_qdq);

    std::vector<std::string> mul_inputs = commute
                                              ? std::vector<std::string>{recip_qdq, num_qdq}
                                              : std::vector<std::string>{num_qdq, recip_qdq};

    builder.AddNode("Mul_node",
                    "Mul",
                    mul_inputs,
                    {"mul_out"},
                    kOnnxDomain);

    // Wrap Mul output in QDQ and expose as graph output.
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out", "mul_out",
        output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// ---------------------------------------------------------------------------
// QDQ model builder -- QDQGroup Reciprocal (DQ -> Reciprocal -> Q)
// ---------------------------------------------------------------------------

// Builds the fully-quantized version of the fusion pattern where the
// Reciprocal node itself is wrapped in a QDQ group:
//
//   denominator --> Q --> DQ --> Reciprocal --> Q --> DQ --> recip_qdq --+
//                                                                        v
//   numerator   --> Q --> DQ -----------------------------------------> Mul --> Q --> DQ --> output
//
// This is the pattern produced by quantization tools for models such as
// LayerNorm (rstd computation).  The ORT graph partitioner groups the
// DQ -> Reciprocal -> Q sequence into a single QDQGroup NodeUnit.
// ReciprocalMulFusion must accept QDQGroup Reciprocal units and fuse the
// whole sub-graph into a single ElementWiseDivide node.
//
// When commute=false  =>  Mul(numerator_qdq, recip_qdq)   [recip in slot 1]
// When commute=true   =>  Mul(recip_qdq, numerator_qdq)   [recip in slot 0]
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGroupReciprocalMulTestCase(
    const TestInputDef<float>& numerator_def,
    const TestInputDef<float>& denominator_def,
    bool commute = false,
    bool use_contrib_qdq = false) {
  return [numerator_def, denominator_def, commute, use_contrib_qdq](
             ModelTestBuilder& builder,
             std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    builder.graph_->set_name("qdq_group_reciprocal_mul_fusion_graph");

    MakeTestInput<float>(builder, "numerator", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    const QuantParams<QuantType> num_qparams = GetTestInputQuantParams<QuantType>(numerator_def);
    const QuantParams<QuantType> den_qparams = GetTestInputQuantParams<QuantType>(denominator_def);

    // Wrap inputs in QDQ pairs.
    const std::string num_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_num", "numerator", num_qparams.scale, num_qparams.zero_point, use_contrib_qdq);
    const std::string den_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_den", "denominator", den_qparams.scale, den_qparams.zero_point, use_contrib_qdq);

    // den_qdq -> Reciprocal -> recip_out
    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {den_qdq},
                    {"recip_out"},
                    kOnnxDomain);

    // Wrap the Reciprocal output in a QDQ pair.  This causes the ORT graph
    // partitioner to group the Q -> Reciprocal -> DQ sequence into a single
    // QDQGroup NodeUnit.  ReciprocalMulFusion now accepts QDQGroup Reciprocal
    // units and must fuse this pattern into a single ElementWiseDivide.
    const QuantParams<QuantType> recip_qparams = GetTestInputQuantParams<QuantType>(denominator_def);
    const std::string recip_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_recip", "recip_out", recip_qparams.scale, recip_qparams.zero_point, use_contrib_qdq);

    // recip_qdq feeds exactly ONE Mul node -- fusion must fire.
    std::vector<std::string> mul_inputs = commute
                                              ? std::vector<std::string>{recip_qdq, num_qdq}
                                              : std::vector<std::string>{num_qdq, recip_qdq};

    builder.AddNode("Mul_node",
                    "Mul",
                    mul_inputs,
                    {"mul_out"},
                    kOnnxDomain);

    // Wrap Mul output in QDQ and expose as graph output.
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out", "mul_out",
        output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
  };
}

// ---------------------------------------------------------------------------
// Negative-case model builders
// ---------------------------------------------------------------------------

// Builds a QDQ graph where the Reciprocal output is wrapped in a QDQ pair
// whose DQ output is then consumed by TWO Mul nodes.
//
// The fusion must NOT fire because GetChildNodeUnitAllowQdq's single-consumer
// guard detects that the Q node's output has two consumers (the two DQ nodes
// feeding the two Mul nodes) and returns nullptr.
//
// With the fusion blocked, the QDQ-wrapped Reciprocal is lowered by
// ReciprocalOpBuilder (reciprocal_op_builder.cc) as a standalone
// ElementWiseDivide(1.0, denominator) node.  Each Mul node is lowered
// independently as an ElementWiseMultiply node.
//
// Expected QNN graph:
//   1 x ElementWiseDivide  (from ReciprocalOpBuilder, constant-1 numerator)
//   2 x ElementWiseMultiply (Mul_A and Mul_B, lowered individually)
//
// Graph topology:
//
//   denominator --> Q --> DQ --> Reciprocal --> recip_out
//                                                  |
//                                                  v
//                                              Q --> DQ --> recip_qdq --+--> Mul_A --> Q --> DQ --> out_a
//                                                                       |
//   numerator_b --> Q --> DQ ------------------------------------------>+--> Mul_B --> Q --> DQ --> out_b
//
// All intermediate tensors are quantized, so QNN HTP can finalize the graph.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQReciprocalMulNoFusionTestCase(
    const TestInputDef<float>& numerator_def,
    const TestInputDef<float>& denominator_def,
    bool use_contrib_qdq = false) {
  return [numerator_def, denominator_def, use_contrib_qdq](
             ModelTestBuilder& builder,
             std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    builder.graph_->set_name("qdq_reciprocal_qdq_wrapped_no_fusion_graph");

    MakeTestInput<float>(builder, "numerator_a", numerator_def);
    MakeTestInput<float>(builder, "numerator_b", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    const QuantParams<QuantType> num_qparams = GetTestInputQuantParams<QuantType>(numerator_def);
    const QuantParams<QuantType> den_qparams = GetTestInputQuantParams<QuantType>(denominator_def);

    // Wrap all inputs in QDQ pairs.
    const std::string num_a_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_num_a", "numerator_a", num_qparams.scale, num_qparams.zero_point, use_contrib_qdq);
    const std::string num_b_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_num_b", "numerator_b", num_qparams.scale, num_qparams.zero_point, use_contrib_qdq);
    const std::string den_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_den", "denominator", den_qparams.scale, den_qparams.zero_point, use_contrib_qdq);

    // denominator_qdq -> Reciprocal -> recip_out
    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {den_qdq},
                    {"recip_out"},
                    kOnnxDomain);

    // Wrap the Reciprocal output in a QDQ pair.  This causes the ORT graph
    // partitioner to group the Q -> Reciprocal -> DQ sequence into a single
    // QDQGroup NodeUnit.  The fusion is blocked NOT by the unit-type check
    // (which now accepts QDQGroup) but by GetChildNodeUnitAllowQdq's
    // single-consumer guard: the Q node's output feeds TWO DQ nodes (one
    // for each Mul), so the guard returns nullptr and the fusion is skipped.
    // All intermediate tensors remain quantized, so QNN HTP can finalize
    // the graph without error.
    const QuantParams<QuantType> recip_qparams = GetTestInputQuantParams<QuantType>(denominator_def);
    const std::string recip_qdq = AddQDQNodePair<QuantType>(
        builder, "qdq_recip", "recip_out",
        recip_qparams.scale, recip_qparams.zero_point, use_contrib_qdq);

    // recip_qdq feeds TWO Mul nodes — two consumers of the DQ output.
    builder.AddNode("Mul_A",
                    "Mul",
                    {num_a_qdq, recip_qdq},
                    {"mul_out_a"},
                    kOnnxDomain);

    builder.AddNode("Mul_B",
                    "Mul",
                    {num_b_qdq, recip_qdq},
                    {"mul_out_b"},
                    kOnnxDomain);

    // Wrap both Mul outputs in QDQ and expose as graph outputs.
    // output_qparams[0] and output_qparams[1] are computed from the two
    // outputs of BuildReciprocalTwoConsumersTestCase (the f32 reference).
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out_a", "mul_out_a",
        output_qparams[0].scale, output_qparams[0].zero_point, use_contrib_qdq);
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(
        builder, "qdq_out_b", "mul_out_b",
        output_qparams[1].scale, output_qparams[1].zero_point, use_contrib_qdq);
  };
}

// Builds a graph where the Reciprocal output is consumed by TWO Mul nodes.
// The fusion must NOT fire because GetOnlyChildOfType() requires exactly one
// consumer.  Both Mul nodes should still be assigned to QNN individually.
//
//   denominator --> Reciprocal --> recip_out --+--> Mul_A --> out_a
//                                              |
//   numerator_b --------------------------------+--> Mul_B --> out_b
GetTestModelFn BuildReciprocalTwoConsumersTestCase(const TestInputDef<float>& numerator_def,
                                                   const TestInputDef<float>& denominator_def) {
  return [numerator_def, denominator_def](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_two_consumers_graph");

    MakeTestInput<float>(builder, "numerator_a", numerator_def);
    MakeTestInput<float>(builder, "numerator_b", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_A",
                    "Mul",
                    {"numerator_a", "recip_out"},
                    {"out_a"},
                    kOnnxDomain);

    builder.AddNode("Mul_B",
                    "Mul",
                    {"numerator_b", "recip_out"},
                    {"out_b"},
                    kOnnxDomain);

    builder.MakeOutput("out_a");
    builder.MakeOutput("out_b");
  };
}

// Builds a graph where the Reciprocal output is ALSO a graph output.
// The fusion must NOT fire because the intermediate tensor cannot be removed.
//
//   denominator --> Reciprocal --> recip_out (graph output)
//                                      |
//   numerator   -----------------------> Mul --> output
GetTestModelFn BuildReciprocalOutputIsGraphOutputTestCase(const TestInputDef<float>& numerator_def,
                                                          const TestInputDef<float>& denominator_def) {
  return [numerator_def, denominator_def](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("reciprocal_output_is_graph_output_graph");

    MakeTestInput<float>(builder, "numerator", numerator_def);
    MakeTestInput<float>(builder, "denominator", denominator_def);

    builder.AddNode("Reciprocal_node",
                    "Reciprocal",
                    {"denominator"},
                    {"recip_out"},
                    kOnnxDomain);

    builder.AddNode("Mul_node",
                    "Mul",
                    {"numerator", "recip_out"},
                    {"output"},
                    kOnnxDomain);

    // Expose the Reciprocal output as a graph output — this blocks fusion.
    builder.MakeOutput("recip_out");
    builder.MakeOutput("output");
  };
}

// ---------------------------------------------------------------------------
// Shared provider-options helper
// ---------------------------------------------------------------------------

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return provider_options;
}

}  // namespace

// =============================================================================
// Float32 tests
// =============================================================================

// Basic 4-D input, standard Mul input order: Mul(numerator, recip_out)
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_Float32_4D_StandardOrder) {
  const std::filesystem::path json_dir = "ReciprocalMulFusion_Float32_4D_StandardOrder";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  // Use non-zero denominator values to avoid division-by-zero.
  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  RunQnnModelTest(BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/false),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_dir, "ElementWiseDivide");
}

// Basic 4-D input, commuted Mul input order: Mul(recip_out, numerator)
// Verifies that the fusion handles both Mul input slot orderings correctly.
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_Float32_4D_CommutedOrder) {
  const std::filesystem::path json_dir = "ReciprocalMulFusion_Float32_4D_CommutedOrder";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  RunQnnModelTest(BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/true),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);

  AssertOpInQnnGraph(json_dir, "ElementWiseDivide");
}

// =============================================================================
// QDQ uint8 tests
// =============================================================================

// QDQ uint8, standard Mul input order
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_QDQ_U8_StandardOrder) {
  const std::filesystem::path json_dir = "ReciprocalMulFusion_QDQ_U8_StandardOrder";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  TestQDQModelAccuracy(
      BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/false),
      BuildQDQReciprocalMulTestCase<uint8_t>(numerator_def, denominator_def, /*commute=*/false),
      provider_options,
      /*opset_version=*/13,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  // QDQ Reciprocal is a SingleNode unit (no surrounding Q/DQ on the Reciprocal itself),
  // so the fusion fires and the compiled graph must contain a single ElementWiseDivide.
  AssertOpInQnnGraph(json_dir, "ElementWiseDivide", /*count=*/1);
}

// QDQ uint8, commuted Mul input order
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_QDQ_U8_CommutedOrder) {
  const std::filesystem::path json_dir = "ReciprocalMulFusion_QDQ_U8_CommutedOrder";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  TestQDQModelAccuracy(
      BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/true),
      BuildQDQReciprocalMulTestCase<uint8_t>(numerator_def, denominator_def, /*commute=*/true),
      provider_options,
      /*opset_version=*/13,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_dir, "ElementWiseDivide", /*count=*/1);
}

// =============================================================================
// QDQ uint16 tests (contrib ops, requires HTP v73+)
// =============================================================================

// QDQ uint16, standard Mul input order
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_QDQ_U16_StandardOrder) {
  if (QnnHTPBackendTests::ShouldSkipIfHtpArchIsLessThanOrEqualTo(QNN_HTP_DEVICE_ARCH_V68)) {
    GTEST_SKIP() << "uint16 QDQ requires HTP arch > v68";
  }

  const std::filesystem::path json_dir = "ReciprocalMulFusion_QDQ_U16_StandardOrder";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  TestQDQModelAccuracy(
      BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/false),
      BuildQDQReciprocalMulTestCase<uint16_t>(numerator_def, denominator_def,
                                              /*commute=*/false, /*use_contrib_qdq=*/true),
      provider_options,
      /*opset_version=*/13,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_dir, "ElementWiseDivide", /*count=*/1);
}

// =============================================================================
// Float16 tests
// =============================================================================

// FP16 Reciprocal->Mul fusion on HTP.
// Uses TestFp16ModelAccuracy: runs the fp32 reference on CPU EP and the fp16
// model on QNN EP, then checks that the fused graph contains a single
// ElementWiseDivide node (not a separate Reciprocal + Mul pair).
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_FP16) {
  if (QnnHTPBackendTests::ShouldSkipIfHtpFp16Unsupported()) {
    GTEST_SKIP() << "FP16 fusion requires HTP arch > V68";
  }

  const std::filesystem::path json_dir = "ReciprocalMulFusion_FP16";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  // fp32 reference model (run on CPU EP)
  const auto fp32_model_fn = BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/false);
  // fp16 model (run on QNN EP)
  const auto fp16_model_fn = BuildReciprocalMulFP16TestCase(numerator_def, denominator_def, /*commute=*/false);

  TestFp16ModelAccuracy(fp32_model_fn,
                        fp16_model_fn,
                        provider_options,
                        /*opset_version=*/13,
                        /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                        /*tolerance=*/0.004f);

  // The fusion must have fired: one ElementWiseDivide, no standalone Reciprocal.
  AssertOpInQnnGraph(json_dir, "ElementWiseDivide", /*count=*/1);
}

// =============================================================================
// QDQ uint8 tests -- QDQGroup Reciprocal (DQ -> Reciprocal -> Q)
// =============================================================================

// QDQ uint8, QDQGroup Reciprocal, standard Mul input order.
// Verifies that a fully-quantized Reciprocal (wrapped in DQ -> Reciprocal -> Q)
// is correctly fused into a single ElementWiseDivide node.  This is the
// pattern produced by quantization tools for LayerNorm rstd computation.
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_QDQGroup_U8_StandardOrder) {
  const std::filesystem::path json_dir = "ReciprocalMulFusion_QDQGroup_U8_StandardOrder";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  TestQDQModelAccuracy(
      BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/false),
      BuildQDQGroupReciprocalMulTestCase<uint8_t>(numerator_def, denominator_def, /*commute=*/false),
      provider_options,
      /*opset_version=*/13,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  // The QDQGroup Reciprocal fusion must have fired: one ElementWiseDivide,
  // no standalone Reciprocal or separate Mul.
  AssertOpInQnnGraph(json_dir, "ElementWiseDivide", /*count=*/1);
}

// QDQ uint8, QDQGroup Reciprocal, commuted Mul input order.
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_QDQGroup_U8_CommutedOrder) {
  const std::filesystem::path json_dir = "ReciprocalMulFusion_QDQGroup_U8_CommutedOrder";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  TestQDQModelAccuracy(
      BuildReciprocalMulTestCase(numerator_def, denominator_def, /*commute=*/true),
      BuildQDQGroupReciprocalMulTestCase<uint8_t>(numerator_def, denominator_def, /*commute=*/true),
      provider_options,
      /*opset_version=*/13,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  AssertOpInQnnGraph(json_dir, "ElementWiseDivide", /*count=*/1);
}

// =============================================================================
// Negative / no-fusion tests
// =============================================================================

// When the Reciprocal output is ALSO a graph output, GetChildNodeUnitAllowQdq's
// graph-output guard (outputs[0].IsGraphOutput()) detects the condition and
// returns nullptr, blocking the fusion.
//
// For float32 inputs on the HTP backend, ReciprocalOpBuilder::IsOpSupported
// also rejects the standalone Reciprocal node (unquantized float inputs are
// not supported by ElementWiseDivide(static_1.0, dynamic_x) on HTP).  As a
// result, the Reciprocal node falls back to CPU EP.
//
// The Mul node, however, is a valid standalone ElementWiseMultiply on QNN HTP:
// its inputs are a graph input (numerator) and recip_out, which is a graph
// output produced by CPU EP and passed to QNN EP as a cross-EP tensor.  The
// Mul node is therefore assigned to QNN EP and appears in the QNN graph as
// a single ElementWiseMultiply node.
//
// Expected QNN graph: 0 ElementWiseDivide, 1 ElementWiseMultiply.
// Expected EP assignment: Some (Reciprocal on CPU EP, Mul on QNN EP).
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_ReciprocalOutputIsGraphOutput_NoFusion) {
  const std::filesystem::path json_dir = "ReciprocalMulFusion_ReciprocalOutputIsGraphOutput_NoFusion";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  // Fusion is blocked (recip_out is a graph output) and ReciprocalOpBuilder
  // rejects float32 Reciprocal on HTP, so Reciprocal falls back to CPU EP.
  // The Mul node is a valid standalone ElementWiseMultiply on QNN HTP and
  // is assigned to QNN EP.
  RunQnnModelTest(BuildReciprocalOutputIsGraphOutputTestCase(numerator_def, denominator_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::Some,
                  /*fp32_abs_err=*/2e-3f);

  // No fused Div node; the Mul runs as a standalone ElementWiseMultiply on QNN EP.
  AssertOpInQnnGraph(json_dir, "ElementWiseDivide", /*count=*/0);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", /*count=*/1);
}

// When the Reciprocal output is wrapped in a QDQ pair, the ORT graph
// partitioner groups the Q -> Reciprocal -> DQ sequence into a QDQGroup
// NodeUnit.  ReciprocalMulFusion now accepts QDQGroup Reciprocal units, so
// the unit-type check no longer blocks the fusion.  However, when the DQ
// output feeds TWO Mul nodes, GetChildNodeUnitAllowQdq's single-consumer
// guard detects the fan-out and returns nullptr, blocking the fusion.
//
// With the fusion blocked, the QDQ-wrapped Reciprocal is lowered by
// ReciprocalOpBuilder as a standalone ElementWiseDivide(1.0, denominator)
// node.  Each of the two Mul nodes is lowered independently as an
// ElementWiseMultiply node.
//
// Structural assertions that distinguish the op-builder path from the fusion:
//   ElementWiseDivide   count=1  (ReciprocalOpBuilder: 1.0 / denominator)
//   ElementWiseMultiply count=2  (Mul_A and Mul_B lowered individually;
//                                 fusion did NOT absorb either of them)
//
// If the fusion were to fire incorrectly, one or both Mul nodes would be
// absorbed into a Div and ElementWiseMultiply count would drop below 2 --
// the second assertion would catch that regression.
TEST_F(QnnHTPBackendTests, ReciprocalMulFusion_QDQWrappedReciprocal_TwoConsumers_NoFusion) {
  const std::filesystem::path json_dir = "ReciprocalMulFusion_QDQWrappedReciprocal_TwoConsumers_NoFusion";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions provider_options = GetProviderOptions();
  provider_options["dump_json_qnn_graph"] = "1";
  provider_options["json_qnn_graph_dir"] = json_dir.string();

  const auto numerator_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);
  const auto denominator_def = TestInputDef<float>({1, 2, 3, 4}, false, 0.5f, 2.0f);

  // The f32 reference model must have the same number of outputs as the QDQ
  // model.  BuildQDQReciprocalMulNoFusionTestCase produces two outputs
  // (out_a, out_b), so we use BuildReciprocalTwoConsumersTestCase here.
  TestQDQModelAccuracy(
      BuildReciprocalTwoConsumersTestCase(numerator_def, denominator_def),
      BuildQDQReciprocalMulNoFusionTestCase<uint8_t>(numerator_def, denominator_def),
      provider_options,
      /*opset_version=*/13,
      /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  // Fusion did NOT fire: Reciprocal was lowered by ReciprocalOpBuilder as a
  // standalone ElementWiseDivide(1.0, denominator), and both Mul nodes were
  // lowered independently as ElementWiseMultiply nodes.
  AssertOpInQnnGraph(json_dir, "ElementWiseDivide", /*count=*/1);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", /*count=*/2);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
