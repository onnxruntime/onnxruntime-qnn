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
// Helpers
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

// Returns true if at least one QNN JSON graph file exists in dump_dir.
// When the QNN JSON dump is absent (e.g., HTP unavailable on this host),
// structural AssertOpInQnnGraph checks are skipped to avoid vacuous passes.
static bool HasQnnJsonGraph(const std::filesystem::path& dump_dir) {
  if (!std::filesystem::exists(dump_dir)) return false;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".json" &&
        entry.path().filename().string().find("_tensor_log") == std::string::npos) {
      return true;
    }
  }
  return false;
}

// Build a SequenceConstruct(N float32 elements of elem_shape) → SequenceLength model.
static GetTestModelFn BuildSequenceLengthTestCase(const std::vector<int64_t>& elem_shape,
                                                  int num_elements) {
  return [=](ModelTestBuilder& builder) -> void {
    std::vector<std::string> elem_names;
    for (int i = 0; i < num_elements; ++i) {
      const std::string name = "elem_" + std::to_string(i);
      auto def = TestInputDef<float>(elem_shape, /*is_initializer=*/false, -1.0f, 1.0f);
      MakeTestInput(builder, name, def);
      elem_names.push_back(name);
    }

    builder.AddNode("seq_construct", "SequenceConstruct", elem_names, {"sequence"});

    // SequenceLength output: int64 scalar.
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("seq_len", "SequenceLength", {"sequence"}, {"length"});
  };
}

// Build a SequenceConstruct(N float16 elements of elem_shape) → SequenceLength model.
// SequenceLength output is always int64 regardless of element dtype.
static GetTestModelFn BuildSequenceLengthFP16TestCase(const std::vector<int64_t>& elem_shape,
                                                      int num_elements) {
  return [=](ModelTestBuilder& builder) -> void {
    std::vector<std::string> elem_names;
    for (int i = 0; i < num_elements; ++i) {
      const std::string name = "elem_" + std::to_string(i);
      auto fp32_def = TestInputDef<float>(elem_shape, /*is_initializer=*/false, -1.0f, 1.0f);
      auto fp16_def = ConvertToFP16InputDef(fp32_def);
      MakeTestInput(builder, name, fp16_def);
      elem_names.push_back(name);
    }

    builder.AddNode("seq_construct", "SequenceConstruct", elem_names, {"sequence"});

    // SequenceLength output is always int64.
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("seq_len", "SequenceLength", {"sequence"}, {"length"});
  };
}

// ---------------------------------------------------------------------------
// HTP-only fusion tests
// ---------------------------------------------------------------------------

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Basic case: N=3, float32 elements of shape [2,4].
//
// Emitted fusion (new ReduceSum-based design, 2N+3 = 9 nodes):
//   ReduceSum × 3, EWAdd × 3 (2 chain + 1 final), Reshape × 1,
//   ElementWiseMultiply × 1, Cast × 1.
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_N3) {
  const std::filesystem::path json_dir = "SequenceLengthFusion_N3";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceLengthTestCase({2, 4}, /*N=*/3),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "QNN HTP graph dump unavailable; structural assertions skipped.";
  }
  AssertOpInQnnGraph(json_dir, "ReduceSum", 3);            // per-element scalar reduction
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 3);       // N-1 chain + 1 final (0+N=N)
  AssertOpInQnnGraph(json_dir, "Reshape", 1);              // {1,...,1} → {1}
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);  // zero mask
  AssertOpInQnnGraph(json_dir, "Cast", 1);                 // float → int64
  // QNN registers the output as {1} (HTP rejects rank-0 tensors); ORT's buffer
  // copy is correct because both {} and {1} are 8 bytes for int64.
  AssertTensorShapeInQnnGraph(json_dir, "length", {1u});
}

// Edge case: N=1.
//
// No EWAdd chain step (N-1=0), but the final EWAdd(0, N) is still present.
// Total: 1+0+1+1+1+1 = 5 QNN nodes.
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_N1) {
  const std::filesystem::path json_dir = "SequenceLengthFusion_N1";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceLengthTestCase({3}, /*N=*/1),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "QNN HTP graph dump unavailable; structural assertions skipped.";
  }
  AssertOpInQnnGraph(json_dir, "ReduceSum", 1);            // single element
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);       // final only (no chain)
  AssertOpInQnnGraph(json_dir, "Reshape", 1);              // {1} → {1}
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);  // zero mask
  AssertOpInQnnGraph(json_dir, "Cast", 1);                 // float → int64
}

// Larger N: N=5, elements of shape [1,2,3].
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_N5) {
  const std::filesystem::path json_dir = "SequenceLengthFusion_N5";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceLengthTestCase({1, 2, 3}, /*N=*/5),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "QNN HTP graph dump unavailable; structural assertions skipped.";
  }
  AssertOpInQnnGraph(json_dir, "ReduceSum", 5);            // per-element scalar reduction
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 5);       // N-1=4 chain + 1 final
  AssertOpInQnnGraph(json_dir, "Reshape", 1);              // {1,...,1} → {1}
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);  // zero mask
  AssertOpInQnnGraph(json_dir, "Cast", 1);                 // float → int64
}

// `length` is intermediate: consumed by a Cast (int64 → float32) to exercise the
// NATIVE (non-APP_READ) output tensor path.
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_LengthFeedsOtherOp) {
  const std::filesystem::path json_dir = "SequenceLengthFusion_LengthFeedsOtherOp";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    for (int i = 0; i < 3; ++i) {
      auto def = TestInputDef<float>({2, 3}, /*is_initializer=*/false, -1.0f, 1.0f);
      MakeTestInput(builder, "elem_" + std::to_string(i), def);
    }
    builder.AddNode("seq_construct", "SequenceConstruct",
                    {"elem_0", "elem_1", "elem_2"}, {"sequence"});
    builder.AddNode("seq_len", "SequenceLength", {"sequence"}, {"length"});
    builder.MakeOutput<float>("length_f32", std::vector<int64_t>{});
    std::vector<ONNX_NAMESPACE::AttributeProto> cast_attrs;
    cast_attrs.push_back(builder.MakeScalarAttribute(
        "to", static_cast<int64_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)));
    builder.AddNode("cast", "Cast", {"length"}, {"length_f32"}, "", cast_attrs);
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});

  // Verify structure: fusion's EWAdd produces length (NATIVE), model's Cast consumes it.
  // N=3: ReduceSum=3, EWAdd=3 (2 chain + 1 final), Reshape=1, EWMul=1,
  //       Cast=2: one from the fusion (float→int64), one from the model (int64→float32).
  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "QNN HTP graph dump unavailable; structural assertions skipped.";
  }
  AssertOpInQnnGraph(json_dir, "ReduceSum", 3);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 3);
  AssertOpInQnnGraph(json_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);
  AssertOpInQnnGraph(json_dir, "Cast", 2);
}

// FP16 element tensors. Fusion supports FP16 (TryFusion Step 6 allows it).
// SequenceLength output is always int64 regardless of element dtype.
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_Float16Elements) {
  const std::filesystem::path json_dir = "SequenceLengthFusion_Float16Elements";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  RunQnnModelTest(
      BuildSequenceLengthFP16TestCase({2, 4}, /*N=*/3),
      opts,
      /*opset=*/11,
      EPVerificationParams{ExpectedEPNodeAssignment::All});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "QNN HTP graph dump unavailable; structural assertions skipped.";
  }
  AssertOpInQnnGraph(json_dir, "ReduceSum", 3);            // per-element scalar reduction
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 3);       // N-1 chain + 1 final
  AssertOpInQnnGraph(json_dir, "Reshape", 1);              // {1,...,1} → {1}
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);  // zero mask (float16)
  AssertOpInQnnGraph(json_dir, "Cast", 1);                 // float16 → int64
}

// One element is a static initializer; the other is a runtime graph input.
// Exercises the tensor-registration path in AddToModelBuilder where
// IsQnnTensorWrapperExist returns true for the initializer (already registered
// as STATIC by QnnModelWrapper before partition ops run).
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_InitializerElement) {
  const std::filesystem::path json_dir = "SequenceLengthFusion_InitializerElement";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    // elem_0: static initializer (STATIC tensor in QNN).
    auto init_def = TestInputDef<float>({2, 3}, /*is_initializer=*/true,
                                        std::vector<float>(6, 1.0f));
    MakeTestInput(builder, "elem_0", init_def);
    // elem_1: runtime graph input (APP_WRITE tensor in QNN).
    auto dyn_def = TestInputDef<float>({2, 3}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_1", dyn_def);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "QNN HTP graph dump unavailable; structural assertions skipped.";
  }
  AssertOpInQnnGraph(json_dir, "ReduceSum", 2);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 2);  // N-1=1 chain + 1 final
  AssertOpInQnnGraph(json_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);
  AssertOpInQnnGraph(json_dir, "Cast", 1);
}

// Element tensors come from upstream Relu ops in the same QNN partition.
// Exercises the NATIVE tensor path: AddToModelBuilder must not re-register
// tensors already emitted by an upstream op's AddToModelBuilder.
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_ElementFromUpstreamOp) {
  const std::filesystem::path json_dir = "SequenceLengthFusion_ElementFromUpstreamOp";
  std::filesystem::remove_all(json_dir);
  ASSERT_TRUE(std::filesystem::create_directory(json_dir));
  auto cleanup = gsl::finally([&json_dir]() { std::filesystem::remove_all(json_dir); });

  ProviderOptions opts = GetHTPProviderOptions();
  opts["dump_json_qnn_graph"] = "1";
  opts["json_qnn_graph_dir"] = json_dir.string();

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    for (int i = 0; i < 2; ++i) {
      const std::string in_name = "raw_" + std::to_string(i);
      const std::string relu_out = "elem_" + std::to_string(i);
      auto def = TestInputDef<float>({2, 3}, /*is_initializer=*/false, 0.0f, 1.0f);
      MakeTestInput(builder, in_name, def);
      builder.AddNode("relu_" + std::to_string(i), "Relu", {in_name}, {relu_out});
    }
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::All});

  if (!HasQnnJsonGraph(json_dir)) {
    GTEST_SKIP() << "QNN HTP graph dump unavailable; structural assertions skipped.";
  }
  AssertOpInQnnGraph(json_dir, "Relu", 2);
  AssertOpInQnnGraph(json_dir, "ReduceSum", 2);
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 2);  // N-1=1 chain + 1 final
  AssertOpInQnnGraph(json_dir, "Reshape", 1);
  AssertOpInQnnGraph(json_dir, "ElementWiseMultiply", 1);
  AssertOpInQnnGraph(json_dir, "Cast", 1);
}

// HTP-backend rejection: int64 element dtype falls back to CPU EP on HTP too.
// Distinguishes intentional guard rejection from CPU-only backend validation failure.
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_HTPFallback_UnsupportedDtype) {
  ProviderOptions opts = GetHTPProviderOptions();

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    auto def = TestInputDef<int64_t>({2, 3}, /*is_initializer=*/false, 0LL, 9LL);
    MakeTestInput(builder, "elem_0", def);
    MakeTestInput(builder, "elem_1", def);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// HTP-backend rejection: sequence fan-out falls back to CPU EP on HTP too.
TEST_F(QnnHTPBackendTests, SequenceLengthFusion_HTPFallback_SequenceFanOut) {
  ProviderOptions opts = GetHTPProviderOptions();

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    auto def = TestInputDef<float>({2, 3}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_0", def);
    MakeTestInput(builder, "elem_1", def);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("len1", std::vector<int64_t>{});
    builder.MakeOutput<int64_t>("len2", std::vector<int64_t>{});
    builder.AddNode("sl1", "SequenceLength", {"seq"}, {"len1"});
    builder.AddNode("sl2", "SequenceLength", {"seq"}, {"len2"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// ---------------------------------------------------------------------------
// CPU fallback: SequenceLength consuming a graph-input sequence (not from
// SequenceConstruct) must not be assigned to QNN EP.
// ---------------------------------------------------------------------------

TEST_F(QnnCPUBackendTests, SequenceLengthFusion_CpuFallback_GraphInputSequence) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  // A SequenceEmpty → SequenceLength pattern: SequenceLength's input is NOT a
  // SequenceConstruct, so TryFusion never fires and the node falls back to CPU EP.
  auto build_fn = [](ModelTestBuilder& builder) -> void {
    builder.AddNode("seq_empty", "SequenceEmpty", {}, {"sequence"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("seq_len", "SequenceLength", {"sequence"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// Non-FP32/FP16 element dtypes are rejected in TryFusion because the
// zero-tensor byte-size calculation assumes 4 bytes (FP32) or 2 bytes (FP16),
// and quantized types require a nonzero zero-point to encode real zero.
TEST_F(QnnCPUBackendTests, SequenceLengthFusion_CpuFallback_UnsupportedDtype) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    // int64 elements: unsupported dtype → TryFusion returns nullptr → CPU EP.
    auto def = TestInputDef<int64_t>({2, 3}, /*is_initializer=*/false, 0LL, 9LL);
    MakeTestInput(builder, "elem_0", def);
    MakeTestInput(builder, "elem_1", def);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// Heterogeneous element shapes are ONNX-valid (SequenceConstruct only
// requires a shared element dtype, not a shared shape) but unsupported by this
// fusion — AddToModelBuilder assumes every element has the same shape for the
// Concat axis.  TryFusion detects the mismatch and returns nullptr → CPU EP.
TEST_F(QnnCPUBackendTests, SequenceLengthFusion_CpuFallback_HeterogeneousShapes) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    auto def0 = TestInputDef<float>({2, 3}, /*is_initializer=*/false, -1.0f, 1.0f);
    auto def1 = TestInputDef<float>({4}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_0", def0);
    MakeTestInput(builder, "elem_1", def1);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// Scalar elements (rank-0, shape {}) must be rejected: ReduceSum requires
// at least one axis to reduce over.  TryFusion checks the raw shape before
// calling GetOnnxShape (which would silently convert {} to {1}) and returns
// nullptr → CPU EP.
TEST_F(QnnCPUBackendTests, SequenceLengthFusion_CpuFallback_ScalarElements) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    // rank-0 scalar inputs: shape {}
    auto def = TestInputDef<float>({}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_0", def);
    MakeTestInput(builder, "elem_1", def);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// The SequenceConstruct output is consumed by two SequenceLength nodes
// (fan-out).  FindUniqueSequenceConsumer finds more than one consumer and
// returns nullptr → TryFusion returns nullptr → CPU EP for all nodes.
TEST_F(QnnCPUBackendTests, SequenceLengthFusion_CpuFallback_SequenceFanOut) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    auto def = TestInputDef<float>({2, 3}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_0", def);
    MakeTestInput(builder, "elem_1", def);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("len1", std::vector<int64_t>{});
    builder.MakeOutput<int64_t>("len2", std::vector<int64_t>{});
    builder.AddNode("sl1", "SequenceLength", {"seq"}, {"len1"});
    builder.AddNode("sl2", "SequenceLength", {"seq"}, {"len2"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// elem_0 has shape {1}; elem_1 has shape {} (scalar).  Without a per-element
// raw-shape check, GetOnnxShape would convert {} to {1} and both shapes would
// compare equal, causing the fusion to fire incorrectly.  TryFusion must check
// the raw shape of every element before calling GetOnnxShape.
TEST_F(QnnCPUBackendTests, SequenceLengthFusion_CpuFallback_LaterScalarElement) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    // elem_0: shape {1} — non-scalar, passes the first raw-shape check.
    auto def1 = TestInputDef<float>({1}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_0", def1);
    // elem_1: shape {} (scalar) — must be rejected by the per-element raw-shape check.
    auto def2 = TestInputDef<float>({}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_1", def2);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// SequenceConstruct output is consumed by both SequenceLength and SequenceAt
// (mixed-type fan-out).  FindUniqueSequenceConsumer counts two raw consumers
// and returns nullptr for both fusion attempts → all nodes fall back to CPU EP.
TEST_F(QnnCPUBackendTests, SequenceLengthFusion_CpuFallback_MixedConsumerFanOut) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    auto def = TestInputDef<float>({2, 3}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_0", def);
    MakeTestInput(builder, "elem_1", def);
    MakeTestInput(builder, "elem_2", def);
    builder.AddNode("sc", "SequenceConstruct",
                    {"elem_0", "elem_1", "elem_2"}, {"seq"});
    builder.MakeScalarInitializer<int64_t>("pos", 0LL);
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.MakeOutput<float>("elem_out", std::vector<int64_t>{2, 3});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
    builder.AddNode("sa", "SequenceAt", {"seq", "pos"}, {"elem_out"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

// Elements with a zero in one dimension are rejected in TryFusion Step 5
// (zero-dim guard: `if (d == 0) return nullptr`).  The element tensor is ONNX-valid
// but an empty static buffer cannot satisfy HTP's tensor-connectivity requirement.
TEST_F(QnnCPUBackendTests, SequenceLengthFusion_CpuFallback_ZeroDimElement) {
  ProviderOptions opts;
  opts["backend_type"] = "cpu";

  auto build_fn = [](ModelTestBuilder& builder) -> void {
    // Shape {2, 0}: rank-2 but zero columns — element count is 0.
    auto def = TestInputDef<float>({2, 0}, /*is_initializer=*/false, -1.0f, 1.0f);
    MakeTestInput(builder, "elem_0", def);
    MakeTestInput(builder, "elem_1", def);
    builder.AddNode("sc", "SequenceConstruct", {"elem_0", "elem_1"}, {"seq"});
    builder.MakeOutput<int64_t>("length", std::vector<int64_t>{});
    builder.AddNode("sl", "SequenceLength", {"seq"}, {"length"});
  };

  RunQnnModelTest(build_fn, opts, /*opset=*/11,
                  EPVerificationParams{ExpectedEPNodeAssignment::None});
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
