// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

#if !defined(ORT_MINIMAL_BUILD)

#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "onnx/onnx_pb.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "gtest/gtest.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Builds a small fp32 Conv -> Relu test graph. Both ops are supported by the QNN CPU backend,
// so the whole graph becomes a single QNN partition (CI runs this on QnnCPUBackendTests, which
// doesn't have HTP available — QDQ would only get partially claimed there).
//
// Avoids onnxruntime::test::TestInputDef<float> deliberately: GCC 13 with -Werror flags a
// false-positive -Wmaybe-uninitialized inside std::variant when TestInputDef is captured by
// value into the returned lambda. See the same issue in qnn_basic_test.cc:2637.
GetTestModelFn BuildConvReluTestCase(const std::vector<int64_t>& input_shape) {
  return [input_shape](ModelTestBuilder& builder) -> void {
    builder.graph_->set_name("qnn_dump_test_conv_relu");

    builder.MakeInput<float>("input", input_shape, -1.0f, 1.0f);
    const int64_t c = input_shape[1];
    builder.MakeInitializer<float>("conv_weight", {c, c, 1, 1}, -1.f, 1.f);
    builder.AddNode("conv0", "Conv", {"input", "conv_weight"}, {"conv_out"}, kOnnxDomain);
    builder.AddNode("relu0", "Relu", {"conv_out"}, {"Y"}, kOnnxDomain);
    builder.MakeOutput("Y");
  };
}

ProviderOptions GetCpuProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

// Counts nodes whose op_type starts with "QNN_" — none should appear in the dump (the dump
// captures the pre-translation ONNX side, not the QNN op-builder output).
size_t CountQnnPrefixedOps(const onnx::GraphProto& graph) {
  size_t count = 0;
  for (const auto& node : graph.node()) {
    if (node.op_type().rfind("QNN_", 0) == 0) {
      ++count;
    }
  }
  return count;
}

}  // namespace

// Smoke test: dump_onnx_subgraph=1 emits a <fused_node_name>.onnx file with a parseable, well-formed ModelProto.
TEST_F(QnnCPUBackendTests, DumpOnnxSubgraph_SinglePartition_FileIsValidOnnx) {
  const std::filesystem::path dump_dir = "qnn_dump_test_single_partition";
  std::filesystem::remove_all(dump_dir);
  ASSERT_TRUE(std::filesystem::create_directory(dump_dir));
  const int uncaught_on_entry = std::uncaught_exceptions();
  auto cleanup = gsl::finally([uncaught_on_entry, dump_dir]() {
    if (std::uncaught_exceptions() > uncaught_on_entry) {
      return;  // keep dir on failure for inspection
    }
    std::filesystem::remove_all(dump_dir);
  });

  ProviderOptions provider_options = GetCpuProviderOptions();
  provider_options["dump_onnx_subgraph"] = "1";
  provider_options["onnx_subgraph_dir"] = dump_dir.string();

  RunQnnModelTest(BuildConvReluTestCase({1, 4, 4, 4}),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  // Find the dumped file. Filename matches the synthesized fused_node_name, e.g.
  // "QNNExecutionProvider_QNNExecutionProvider_<hash>_<id>_<part>.onnx" — so we glob.
  std::filesystem::path dumped;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
      dumped = entry.path();
      break;
    }
  }
  ASSERT_FALSE(dumped.empty()) << "No .onnx file emitted in " << dump_dir;
  ASSERT_GT(std::filesystem::file_size(dumped), 0u) << "Dumped ONNX file is empty.";

  // Filename should look like a QNN fused-node name so it correlates with QNN profiler output.
  EXPECT_NE(dumped.stem().string().find("QNN"), std::string::npos)
      << "Dumped filename '" << dumped.filename() << "' should match the QNN fused-node name pattern.";

  // Parse it back as an ONNX ModelProto.
  onnx::ModelProto model_proto;
  std::ifstream fin(dumped, std::ios::in | std::ios::binary);
  ASSERT_TRUE(fin.good()) << "Failed to open dumped file: " << dumped;
  ASSERT_TRUE(model_proto.ParseFromIstream(&fin)) << "Failed to parse: " << dumped;

  EXPECT_EQ(model_proto.producer_name(), "onnxruntime-qnn-ep");
  EXPECT_GE(model_proto.ir_version(), 4);

  // Graph name inside the proto matches the fused_node_name (= dumped filename stem).
  EXPECT_EQ(model_proto.graph().name(), dumped.stem().string());

  // No QNN_-prefixed op types — this is the pre-translation ONNX side.
  EXPECT_EQ(CountQnnPrefixedOps(model_proto.graph()), 0u)
      << "Dumped subgraph contains QNN_-prefixed op types — should be raw ONNX only.";

  // Must contain Conv + Relu (the actual ops in the test graph) and nothing else of interest.
  bool has_conv = false;
  bool has_relu = false;
  for (const auto& n : model_proto.graph().node()) {
    if (n.op_type() == "Conv") has_conv = true;
    if (n.op_type() == "Relu") has_relu = true;
  }
  EXPECT_TRUE(has_conv) << "Dumped subgraph missing Conv.";
  EXPECT_TRUE(has_relu) << "Dumped subgraph missing Relu.";

  // Boundary I/O must be typed.
  ASSERT_GT(model_proto.graph().input_size(), 0);
  ASSERT_GT(model_proto.graph().output_size(), 0);
  for (const auto& vi : model_proto.graph().input()) {
    ASSERT_TRUE(vi.has_type()) << "Graph input '" << vi.name() << "' missing type info.";
    ASSERT_TRUE(vi.type().has_tensor_type()) << "Graph input '" << vi.name() << "' is not a tensor type.";
    EXPECT_GT(vi.type().tensor_type().elem_type(), 0)
        << "Graph input '" << vi.name() << "' has undefined element type.";
  }

  // Initializers (Conv weight + Q/DQ scales/zero_points) must reference the external-data
  // sidecar — the dumper writes weight bytes to <fused_node_name>.onnx.data instead of
  // inlining as raw_data, to lift protobuf's 2 GB single-message ceiling on large models.
  ASSERT_GT(model_proto.graph().initializer_size(), 0);
  const std::filesystem::path sidecar = dumped.string() + ".data";
  EXPECT_TRUE(std::filesystem::exists(sidecar))
      << "Expected external-data sidecar at " << sidecar;
  EXPECT_GT(std::filesystem::file_size(sidecar), 0u)
      << "External-data sidecar is empty.";
  for (const auto& tp : model_proto.graph().initializer()) {
    EXPECT_EQ(tp.data_location(), onnx::TensorProto::EXTERNAL)
        << "Initializer '" << tp.name() << "' should reference external data, not be inlined.";
    EXPECT_TRUE(tp.raw_data().empty())
        << "Initializer '" << tp.name() << "' has inline raw_data — should be in the sidecar instead.";
    bool has_location = false;
    for (const auto& kv : tp.external_data()) {
      if (kv.key() == "location") {
        has_location = true;
        EXPECT_EQ(kv.value(), sidecar.filename().string())
            << "Initializer '" << tp.name() << "' points to wrong sidecar.";
      }
    }
    EXPECT_TRUE(has_location) << "Initializer '" << tp.name() << "' missing external_data location.";
  }
}

// Negative test: dump_onnx_subgraph=0 must not produce any files even if onnx_subgraph_dir is set.
TEST_F(QnnCPUBackendTests, DumpOnnxSubgraph_Disabled_NoFilesWritten) {
  const std::filesystem::path dump_dir = "qnn_dump_test_disabled";
  std::filesystem::remove_all(dump_dir);
  ASSERT_TRUE(std::filesystem::create_directory(dump_dir));
  const int uncaught_on_entry = std::uncaught_exceptions();
  auto cleanup = gsl::finally([uncaught_on_entry, dump_dir]() {
    if (std::uncaught_exceptions() > uncaught_on_entry) {
      return;
    }
    std::filesystem::remove_all(dump_dir);
  });

  ProviderOptions provider_options = GetCpuProviderOptions();
  provider_options["dump_onnx_subgraph"] = "0";
  provider_options["onnx_subgraph_dir"] = dump_dir.string();  // dir set but feature disabled.

  RunQnnModelTest(BuildConvReluTestCase({1, 4, 4, 4}),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);

  size_t onnx_files = 0;
  for (const auto& entry : std::filesystem::directory_iterator{dump_dir}) {
    if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
      ++onnx_files;
    }
  }
  EXPECT_EQ(onnx_files, 0u) << "dump_onnx_subgraph=0 produced unexpected .onnx files in " << dump_dir;
}

#endif  // arch / linux

}  // namespace test
}  // namespace onnxruntime

#endif  // !ORT_MINIMAL_BUILD
