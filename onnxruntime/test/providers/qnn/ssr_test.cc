// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <fstream>
#include <filesystem>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

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

#endif  // defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
