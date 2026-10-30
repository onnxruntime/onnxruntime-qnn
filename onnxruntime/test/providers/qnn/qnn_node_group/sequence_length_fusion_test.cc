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
// Used to skip AssertOpInQnnGraph calls when RunQnnModelTest was silently
// skipped (e.g., HTP unavailable on this host), which would otherwise produce
// a vacuous green test that asserted nothing.
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

// ---------------------------------------------------------------------------
// HTP-only fusion tests
// ---------------------------------------------------------------------------

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Basic case: N=3, float32 elements.
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

  if (!HasQnnJsonGraph(json_dir)) return;
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);  // 0 + N = N (final output op)
  AssertOpInQnnGraph(json_dir, "Cast", 1);            // float → int64
}

// Edge case: N=1.
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

  if (!HasQnnJsonGraph(json_dir)) return;
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);  // 0 + N = N (final output op)
  AssertOpInQnnGraph(json_dir, "Cast", 1);            // float → int64
}

// Larger N.
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

  if (!HasQnnJsonGraph(json_dir)) return;
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);  // 0 + N = N (final output op)
  AssertOpInQnnGraph(json_dir, "Cast", 1);            // float → int64
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
  // Cast=2: one from the fusion (float→int64), one from the model (int64→float32).
  if (!HasQnnJsonGraph(json_dir)) return;
  AssertOpInQnnGraph(json_dir, "ElementWiseAdd", 1);
  AssertOpInQnnGraph(json_dir, "Cast", 2);
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

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
