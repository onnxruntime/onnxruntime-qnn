// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <string>
#include <vector>

#include "test/providers/qnn/qnn_node_group/qnn_graph_checker.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// ---------------------------------------------------------------------------
// Helpers: build SequenceConstruct -> SequenceAt ONNX sub-graphs.
// ---------------------------------------------------------------------------

// Float32 element tensors, static or dynamic int32/int64 position.
template <typename PosType>
static GetTestModelFn BuildSequenceAtTestCase(
    const std::vector<int64_t>& elem_shape,
    int num_elements,          // N, the sequence length
    bool is_dynamic_pos,       // true => position is a graph input
    PosType static_pos_value,  // used when is_dynamic_pos == false
    bool use_heterogeneous_shapes = false) {
  return [=](ModelTestBuilder& builder) -> void {
    std::vector<std::string> elem_names;
    for (int i = 0; i < num_elements; ++i) {
      const std::string name = "elem_" + std::to_string(i);
      std::vector<int64_t> shape_i = elem_shape;
      if (use_heterogeneous_shapes && i > 0) {
        shape_i[0] += 1;
      }
      auto input_def = TestInputDef<float>(shape_i, /*is_initializer=*/false, -2.0f, 2.0f);
      MakeTestInput(builder, name, input_def);
      elem_names.push_back(name);
    }

    builder.AddNode("seq_construct", "SequenceConstruct", elem_names, {"sequence"});

    std::string pos_name = "position";
    if (is_dynamic_pos) {
      builder.MakeInput<PosType>(pos_name, {}, std::vector<PosType>{static_pos_value});
    } else {
      builder.MakeScalarInitializer<PosType>(pos_name, static_pos_value);
    }

    builder.MakeOutput("output");
    builder.AddNode("seq_at", "SequenceAt", {"sequence", pos_name}, {"output"});
  };
}

// Float16 element tensors, static int64 position.
static GetTestModelFn BuildSequenceAtFloat16TestCase(
    const std::vector<int64_t>& elem_shape,
    int num_elements,
    int64_t static_pos_value) {
  return [=](ModelTestBuilder& builder) -> void {
    std::vector<std::string> elem_names;
    // Build FP32 input defs and convert to FP16.
    for (int i = 0; i < num_elements; ++i) {
      const std::string name = "elem_" + std::to_string(i);
      auto fp32_def = TestInputDef<float>(elem_shape, /*is_initializer=*/false, -1.0f, 1.0f);
      auto fp16_def = ConvertToFP16InputDef(fp32_def);
      MakeTestInput(builder, name, fp16_def);
      elem_names.push_back(name);
    }

    builder.AddNode("seq_construct", "SequenceConstruct", elem_names, {"sequence"});
    builder.MakeScalarInitializer<int64_t>("position", static_pos_value);
    builder.MakeOutput("output");
    builder.AddNode("seq_at", "SequenceAt", {"sequence", "position"}, {"output"});
  };
}

// ---------------------------------------------------------------------------
// Provider options shared across all HTP tests
// ---------------------------------------------------------------------------

static ProviderOptions GetHTPProviderOptions() {
  ProviderOptions opts;
  opts["backend_type"] = "htp";
  opts["offload_graph_io_quantization"] = "0";
#if defined(__linux__) && !defined(__aarch64__)
  opts["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
  return opts;
}

// ---------------------------------------------------------------------------
// HTP-only fusion tests
// ---------------------------------------------------------------------------

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Basic positive index: N=3 tensors of shape [3,4], select index 1.
TEST_F(QnnHTPBackendTests, SequenceAtFusion_PositiveIndex) {
  const std::filesystem::path json_dir = "SequenceAtFusion_PositiveIndex";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceAtTestCase<int64_t>({3, 4}, /*N=*/3, /*dynamic=*/false, /*pos=*/1LL),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "Gather", 1);
  AssertOpInQnnGraph(json_dir, "Concat", 1);
}

// Negative index: N=3, position=-1 selects the last element.
TEST_F(QnnHTPBackendTests, SequenceAtFusion_NegativeIndex) {
  const std::filesystem::path json_dir = "SequenceAtFusion_NegativeIndex";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceAtTestCase<int64_t>({3, 4}, /*N=*/3, /*dynamic=*/false, /*pos=*/-1LL),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "Gather", 1);
  AssertOpInQnnGraph(json_dir, "Concat", 1);
}

// First boundary index with int32 position: N=4, position=0.
TEST_F(QnnHTPBackendTests, SequenceAtFusion_FirstIndex_Int32) {
  const std::filesystem::path json_dir = "SequenceAtFusion_FirstIndex_Int32";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceAtTestCase<int32_t>({2, 5}, /*N=*/4, /*dynamic=*/false, /*pos=*/0),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "Gather", 1);
  AssertOpInQnnGraph(json_dir, "Concat", 1);
}

// Last boundary index with int32 position: N=4, position=3.
TEST_F(QnnHTPBackendTests, SequenceAtFusion_LastIndex_Int32) {
  const std::filesystem::path json_dir = "SequenceAtFusion_LastIndex_Int32";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceAtTestCase<int32_t>({2, 5}, /*N=*/4, /*dynamic=*/false, /*pos=*/3),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "Gather", 1);
  AssertOpInQnnGraph(json_dir, "Concat", 1);
}

// Dynamic (graph-input) position, int32: N=3, runtime position=1.
TEST_F(QnnHTPBackendTests, SequenceAtFusion_DynamicPosition_Int32) {
  const std::filesystem::path json_dir = "SequenceAtFusion_DynamicPosition_Int32";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceAtTestCase<int32_t>({2, 3}, /*N=*/3, /*dynamic=*/true, /*pos=*/1),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "Gather", 1);
  AssertOpInQnnGraph(json_dir, "Concat", 1);
  // Normalization subgraph must be present for dynamic position.
  AssertOpInQnnGraph(json_dir, "ElementWiseSelect", 1);
}

// Dynamic (graph-input) negative position: N=3, runtime position=-1 (last element).
TEST_F(QnnHTPBackendTests, SequenceAtFusion_DynamicNegativeIndex) {
  const std::filesystem::path json_dir = "SequenceAtFusion_DynamicNegativeIndex";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceAtTestCase<int32_t>({2, 3}, /*N=*/3, /*dynamic=*/true, /*pos=*/-1),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "Gather", 1);
  AssertOpInQnnGraph(json_dir, "ElementWiseSelect", 1);
}

// Float16 element tensors: N=2, shape [4], position=0.
TEST_F(QnnHTPBackendTests, SequenceAtFusion_Float16Elements) {
  const std::filesystem::path json_dir = "SequenceAtFusion_Float16Elements";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceAtFloat16TestCase({4}, /*N=*/2, /*pos=*/0LL),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "Gather", 1);
}

// N=1 edge case: single-element sequence, position=0.
TEST_F(QnnHTPBackendTests, SequenceAtFusion_SingleElementSequence) {
  const std::filesystem::path json_dir = "SequenceAtFusion_SingleElementSequence";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceAtTestCase<int32_t>({2, 2}, /*N=*/1, /*dynamic=*/false, /*pos=*/0),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(1e-2f)});

  AssertOpInQnnGraph(json_dir, "Gather", 1);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ---------------------------------------------------------------------------
// CPU fallback test (no architecture guard — runs on all platforms)
//
// A SequenceAt node fed by a SequenceConstruct whose element tensors have
// *heterogeneous* shapes must NOT be assigned to QNN EP.  ORT must fall it
// back to CPU EP.
// ---------------------------------------------------------------------------

TEST_F(QnnCPUBackendTests, SequenceAtFusion_CpuFallback_HeterogeneousShapes) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  // Two elements: [2,3] and [3,3] — different shapes → fusion must be declined.
  auto build_fn = [](ModelTestBuilder& builder) -> void {
    auto input0_def = TestInputDef<float>({2, 3}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_0", input0_def);
    // Deliberately different shape for elem_1.
    auto input1_def = TestInputDef<float>({3, 3}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_1", input1_def);

    builder.AddNode("seq_construct", "SequenceConstruct", {"elem_0", "elem_1"}, {"sequence"});
    builder.MakeScalarInitializer<int64_t>("position", 0LL);
    builder.MakeOutput("output");
    builder.AddNode("seq_at", "SequenceAt", {"sequence", "position"}, {"output"});
  };

  // The QNN EP does not support SequenceAt without the fusion (no standalone
  // SequenceAt op builder). With heterogeneous shapes the fusion is rejected,
  // so ExpectedEPNodeAssignment::None verifies the node falls back to CPU EP.
  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
