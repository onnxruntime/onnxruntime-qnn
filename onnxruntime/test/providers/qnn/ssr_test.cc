// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <fstream>
#include <filesystem>
#include <string>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#if defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))

// Reads the EPContext ONNX skeleton file and extracts the relative path of the
// QNN context binary from the EPContext node with main_context=1.
static void GetContextBinaryFileName(const std::string& onnx_ctx_file,
                                     std::string& ctx_bin_file_out) {
  onnx::ModelProto ctx_model_proto;
  std::ifstream ifs(onnx_ctx_file, std::ios::in | std::ios::binary);
  ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << onnx_ctx_file;
  ASSERT_TRUE(ctx_model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << onnx_ctx_file;

  for (const auto& node : ctx_model_proto.graph().node()) {
    if (node.op_type() != "EPContext") continue;
    int64_t is_main_context = 0;
    std::string ep_cache_context;
    for (const auto& attr : node.attribute()) {
      if (attr.name() == "main_context") is_main_context = attr.i();
      else if (attr.name() == "ep_cache_context") ep_cache_context = attr.s();
    }
    if (is_main_context == 1) {
      ctx_bin_file_out = ep_cache_context;
      return;
    }
  }
}

// Removes both the EPContext skeleton .onnx file and its companion .bin file.
static void CleanUpCtxFile(const std::string& context_file_path) {
  std::string qnn_ctx_binary_file_name;
  GetContextBinaryFileName(context_file_path, qnn_ctx_binary_file_name);
  std::filesystem::path ctx_model_path(context_file_path);
  std::string bin_path = ctx_model_path.parent_path().string() + "/" + qnn_ctx_binary_file_name;
  ASSERT_EQ(std::remove(bin_path.c_str()), 0);
  ASSERT_EQ(std::remove(context_file_path.c_str()), 0);
}

class QnnMockSSRBackendTests : public QnnHTPBackendTests {
 protected:
  void SetUp() override;
  ProviderOptions provider_options;
};

void QnnMockSSRBackendTests::SetUp() {
  QnnHTPBackendTests::SetUp();
  provider_options = {
      {"backend_path", "QnnMockSSR.dll"},
      {"offload_graph_io_quantization", "0"},
  };
}

// Test that SSR is correctly recovered during graphExecute when loading a QNN context binary
// from an external file (embed_mode=0).
//
// Step 1 — Generate the embed_mode=0 context binary using the real HTP backend.
//           Because the real HTP backend is used here, QnnMockSSR's static graphExecute
//           counter is not touched, so call_cnt remains 0 when we reach step 2.
//
// Step 2 — Load the context binary through QnnMockSSR.dll. The first graphExecute call
//           triggers a PD reset then returns QNN_COMMON_ERROR_SYSTEM_COMMUNICATION.
//           QnnModel::ExecuteGraph detects the error, calls ReloadContextForModel() to
//           recreate the context from the .bin file on disk, and retries — which must
//           succeed and produce correct outputs.
TEST_F(QnnMockSSRBackendTests, SSRGraphExecuteEpContextNonEmbedMode) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::string context_model_file = "./ssr_ep_ctx_non_embed_test.onnx";
  std::remove(context_model_file.c_str());

  const std::string op_type = "Atan";
  const TestInputDef<float> input_def_qdq({1, 2, 3}, false, -10.0f, 10.0f);

  // -----------------------------------------------------------------------
  // Step 1: Generate the embed_mode=0 context binary with the real HTP backend.
  // -----------------------------------------------------------------------
  ProviderOptions htp_options;
  htp_options["backend_type"] = "htp";
  htp_options["offload_graph_io_quantization"] = "0";

  std::unordered_map<std::string, std::string> gen_session_opts;
  gen_session_opts.emplace(kOrtSessionOptionEpContextEnable, "1");
  gen_session_opts.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);
  gen_session_opts.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def_qdq}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def_qdq}, {}, {}),
                       htp_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",  // No pre-existing context model; generate it now
                       gen_session_opts);

  ASSERT_TRUE(std::filesystem::exists(context_model_file))
      << "Context model file was not generated: " << context_model_file;

  // -----------------------------------------------------------------------
  // Step 2: Load the context model via QnnMockSSR.dll. SSR fires on the first
  //         graphExecute; our recovery code reloads from the .bin file and retries.
  // -----------------------------------------------------------------------
  std::unordered_map<std::string, std::string> run_session_opts;
  run_session_opts.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def_qdq}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def_qdq}, {}, {}),
                       provider_options,  // QnnMockSSR.dll
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       context_model_file,  // Load from the generated context model
                       run_session_opts);

  CleanUpCtxFile(context_model_file);
}

// Test SSR recovery with a model containing 2 EPContext nodes, each backed by its own
// context binary (graph_count=1 per binary). This validates that ReloadContextForModel
// works for multi-partition AOT models where each partition has an independent .bin file.
//
// Approach:
//  1. Generate a single-partition embed_mode=0 context (Atan op) → produces 1 EPContext + 1 .bin.
//  2. Copy the .bin to create a second independent binary.
//  3. Build a combined ONNX model with 2 EPContext nodes chained together, each with
//     main_context=1 pointing to its own .bin file.
//  4. Load via QnnMockSSR.dll and run inference → SSR fires → recovery succeeds.
TEST_F(QnnMockSSRBackendTests, SSRGraphExecuteEpContextNonEmbedModeMultiPartition) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::string ctx_model_file = "./ssr_multi_part_ctx.onnx";
  const std::string bin_file_1 = "./ssr_multi_part_ctx_1_qnn.bin";
  const std::string bin_file_2 = "./ssr_multi_part_ctx_2_qnn.bin";
  const std::string combined_model_file = "./ssr_multi_part_combined.onnx";
  std::remove(ctx_model_file.c_str());
  std::remove(bin_file_1.c_str());
  std::remove(bin_file_2.c_str());
  std::remove(combined_model_file.c_str());

  const std::string op_type = "Atan";
  const TestInputDef<float> input_def_qdq({1, 2, 3}, false, -10.0f, 10.0f);

  // -----------------------------------------------------------------------
  // Step 1: Generate a single-partition embed_mode=0 context with real HTP.
  // -----------------------------------------------------------------------
  ProviderOptions htp_options;
  htp_options["backend_type"] = "htp";
  htp_options["offload_graph_io_quantization"] = "0";

  std::unordered_map<std::string, std::string> gen_session_opts;
  gen_session_opts.emplace(kOrtSessionOptionEpContextEnable, "1");
  gen_session_opts.emplace(kOrtSessionOptionEpContextFilePath, ctx_model_file);
  gen_session_opts.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def_qdq}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def_qdq}, {}, {}),
                       htp_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",
                       gen_session_opts);

  ASSERT_TRUE(std::filesystem::exists(ctx_model_file))
      << "Context model file was not generated: " << ctx_model_file;

  // -----------------------------------------------------------------------
  // Step 2: Extract the .bin filename, create 2 copies, and build a combined
  //         ONNX model with 2 chained EPContext nodes.
  // -----------------------------------------------------------------------
  std::string original_bin_name;
  GetContextBinaryFileName(ctx_model_file, original_bin_name);
  ASSERT_FALSE(original_bin_name.empty());

  std::filesystem::path original_bin_path =
      std::filesystem::path(ctx_model_file).parent_path() / original_bin_name;
  ASSERT_TRUE(std::filesystem::exists(original_bin_path));

  // Create 2 copies of the binary.
  std::filesystem::copy_file(original_bin_path, bin_file_1, std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(original_bin_path, bin_file_2, std::filesystem::copy_options::overwrite_existing);

  // Parse the generated context model to extract EPContext node attributes.
  onnx::ModelProto src_model;
  {
    std::ifstream ifs(ctx_model_file, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.good());
    ASSERT_TRUE(src_model.ParseFromIstream(&ifs));
  }

  // Find the EPContext node in the source model.
  const onnx::NodeProto* src_ep_node = nullptr;
  for (const auto& node : src_model.graph().node()) {
    if (node.op_type() == "EPContext") {
      src_ep_node = &node;
      break;
    }
  }
  ASSERT_NE(src_ep_node, nullptr);

  // Build combined model: input → EPContext_1 → EPContext_2 → output
  // Both EPContext nodes have identical I/O signatures (same Atan graph).
  onnx::ModelProto combined_model;
  combined_model.set_ir_version(src_model.ir_version());
  for (const auto& opset : src_model.opset_import()) {
    *combined_model.add_opset_import() = opset;
  }

  auto* graph = combined_model.mutable_graph();
  graph->set_name("ssr_multi_partition_graph");

  // Copy graph inputs/outputs from source (same shape for both nodes).
  // The source model has 1 input and 1 output for the EPContext node.
  std::string input_name = src_ep_node->input(0);
  std::string mid_name = "ep_ctx_mid_output";
  std::string output_name = src_ep_node->output(0);

  // Copy input/output type info from source graph.
  for (const auto& input : src_model.graph().input()) {
    if (input.name() == input_name) {
      *graph->add_input() = input;
      break;
    }
  }
  for (const auto& output : src_model.graph().output()) {
    if (output.name() == output_name) {
      // Use the same type info but with a different name for the final output.
      auto* out = graph->add_output();
      *out = output;
      out->set_name(output_name);
      break;
    }
  }

  // Helper to create an EPContext node with a specific bin file.
  auto make_ep_context_node = [&](const std::string& node_name,
                                  const std::string& in, const std::string& out,
                                  const std::string& bin_filename) {
    auto* node = graph->add_node();
    node->set_op_type("EPContext");
    node->set_name(node_name);
    node->add_input(in);
    node->add_output(out);
    // Copy domain from source
    node->set_domain(src_ep_node->domain());

    // Set attributes: main_context=1, embed_mode=0, ep_cache_context=bin_filename
    for (const auto& attr : src_ep_node->attribute()) {
      if (attr.name() == "main_context") {
        auto* a = node->add_attribute();
        a->set_name("main_context");
        a->set_type(onnx::AttributeProto::INT);
        a->set_i(1);  // Each node is its own main context.
      } else if (attr.name() == "ep_cache_context") {
        auto* a = node->add_attribute();
        a->set_name("ep_cache_context");
        a->set_type(onnx::AttributeProto::STRING);
        a->set_s(bin_filename);
      } else {
        *node->add_attribute() = attr;
      }
    }
  };

  make_ep_context_node("ep_ctx_node_1", input_name, mid_name,
                       std::filesystem::path(bin_file_1).filename().string());
  make_ep_context_node("ep_ctx_node_2", mid_name, output_name,
                       std::filesystem::path(bin_file_2).filename().string());

  // Add type info for the intermediate tensor so ORT can resolve it.
  for (const auto& output : src_model.graph().output()) {
    if (output.name() == src_ep_node->output(0)) {
      auto* vi = graph->add_value_info();
      *vi = output;
      vi->set_name(mid_name);
      break;
    }
  }

  // Save the combined model.
  {
    std::ofstream ofs(combined_model_file, std::ios::out | std::ios::binary);
    ASSERT_TRUE(ofs.good());
    ASSERT_TRUE(combined_model.SerializeToOstream(&ofs));
  }

  // -----------------------------------------------------------------------
  // Step 3: Load the combined model via QnnMockSSR.dll and run inference.
  //         SSR fires on the first graphExecute → recovery → success.
  // -----------------------------------------------------------------------
  {
    std::string model_data;
    combined_model.SerializeToString(&model_data);

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, combined_model_file.c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

    ScopedOrtSession scoped(std::move(registered_ep_device),
                            Ort::Session(*ort_env, model_data.data(), model_data.size(), so));

    // Run inference to trigger graphExecute (and SSR recovery).
    auto in_name = scoped.session().GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto out_name = scoped.session().GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

    Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::vector<int64_t> input_shape{1, 2, 3};
    std::vector<float> input_data(6, 1.0f);
    auto input_tensor = Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                 input_shape.data(), input_shape.size());

    const char* input_names[] = {in_name.get()};
    const char* output_names[] = {out_name.get()};
    auto outputs = scoped.session().Run(Ort::RunOptions{}, input_names, &input_tensor, 1,
                                        output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_TRUE(outputs[0].IsTensor());
  }

  // Cleanup.
  std::remove(ctx_model_file.c_str());
  std::remove(original_bin_path.string().c_str());
  std::remove(bin_file_1.c_str());
  std::remove(bin_file_2.c_str());
  std::remove(combined_model_file.c_str());
}

#endif  // defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
