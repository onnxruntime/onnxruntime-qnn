// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: MIT

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
      if (attr.name() == "main_context")
        is_main_context = attr.i();
      else if (attr.name() == "ep_cache_context")
        ep_cache_context = attr.s();
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

// TODO: The test case is instable on CI devices and requires further investigation
// Test SSR recovery with a naturally-partitioned model: a CPU-only op (FusedGemm) forces
// the model into 2 QNN partitions, producing 2 EPContext nodes sharing one context binary.
// Each partition recovers independently via contextCreateFromBinary + graphRetrieve.
TEST_F(QnnMockSSRBackendTests, DISABLED_SSRGraphExecuteEpContextNonEmbedModeCpuFallbackPartition) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::string context_model_file = "./ssr_cpu_fallback_partition_ctx.onnx";
  std::remove(context_model_file.c_str());

  // Build a model: FusedGemm(CPU) → Add(QNN) → FusedGemm(CPU) → Add(QNN)
  // FusedGemm is a contrib op (kMSDomain) not supported by QNN, forcing 2 QNN partitions.
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto build_multi_partition_graph = [](ModelTestBuilder& builder) {
    std::vector<float> data(200 * 200, 1.0f);
    MakeTestInput(builder, "input1", TestInputDef<float>({200, 200}, false, data));
    MakeTestInput(builder, "gemm1_weight", TestInputDef<float>({200, 200}, true, data));

    std::vector<ONNX_NAMESPACE::AttributeProto> gemm_attrs;
    gemm_attrs.push_back(builder.MakeStringAttribute("activation", "Relu"));
    builder.AddNode("FusedGemm_0", "FusedGemm",
                    {"input1", "gemm1_weight"}, {"gemm1_out"},
                    kMSDomain, gemm_attrs);

    std::vector<float> add_data(12, 1.0f);
    gsl::span<float> data_range = gsl::make_span(add_data);
    QuantParams<uint8_t> qp = GetDataQuantParams<uint8_t>(data_range);

    std::string add1_in1 = AddQDQNodePair<uint8_t>(builder, "add1_qdq_in1", "gemm1_out", qp.scale, qp.zero_point);
    MakeTestInput(builder, "add1_weight", TestInputDef<float>({200, 200}, true, data));
    std::string add1_in2 = AddQDQNodePair<uint8_t>(builder, "add1_qdq_in2", "add1_weight", qp.scale, qp.zero_point);
    builder.AddNode("Add_0", "Add", {add1_in1, add1_in2}, {"add1_out"});

    std::string gemm2_in = AddQDQNodePair<uint8_t>(builder, "gemm2_qdq_in", "add1_out", qp.scale, qp.zero_point);
    MakeTestInput(builder, "gemm2_weight", TestInputDef<float>({200, 200}, true, data));

    std::vector<ONNX_NAMESPACE::AttributeProto> gemm_attrs2;
    gemm_attrs2.push_back(builder.MakeStringAttribute("activation", "Relu"));
    builder.AddNode("FusedGemm_1", "FusedGemm",
                    {gemm2_in, "gemm2_weight"}, {"gemm2_out"},
                    kMSDomain, gemm_attrs2);

    std::string add2_in1 = AddQDQNodePair<uint8_t>(builder, "add2_qdq_in1", "gemm2_out", qp.scale, qp.zero_point);
    MakeTestInput(builder, "add2_weight", TestInputDef<float>({200, 200}, true, data));
    std::string add2_in2 = AddQDQNodePair<uint8_t>(builder, "add2_qdq_in2", "add2_weight", qp.scale, qp.zero_point);
    builder.AddNode("Add_1", "Add", {add2_in1, add2_in2}, {"add2_out"});

    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "final_qdq", "add2_out", qp.scale, qp.zero_point);
  };

  ModelTestBuilder helper;
  build_multi_partition_graph(helper);

  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  // -----------------------------------------------------------------------
  // Step 1: Generate embed_mode=0 context with real HTP backend.
  // -----------------------------------------------------------------------
  ProviderOptions htp_options;
  htp_options["backend_type"] = "htp";
  htp_options["offload_graph_io_quantization"] = "0";

  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, htp_options);

    ScopedOrtSession scoped(std::move(registered_ep_device),
                            Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

    ASSERT_TRUE(std::filesystem::exists(context_model_file))
        << "Context model file was not generated: " << context_model_file;
  }

  // Verify 2 EPContext nodes were generated.
  {
    onnx::ModelProto ctx_model_proto;
    std::ifstream ifs(context_model_file, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.good());
    ASSERT_TRUE(ctx_model_proto.ParseFromIstream(&ifs));

    int ep_context_count = 0;
    for (const auto& node : ctx_model_proto.graph().node()) {
      if (node.op_type() == "EPContext") ++ep_context_count;
    }
    ASSERT_EQ(ep_context_count, 2) << "Expected 2 EPContext nodes for multi-partition model.";
  }

  // -----------------------------------------------------------------------
  // Step 2: Load context model via QnnMockSSR.dll, trigger SSR recovery,
  //         and verify output accuracy against CPU reference.
  // -----------------------------------------------------------------------
  std::unordered_map<std::string, std::string> run_session_opts;
  run_session_opts.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);

  // Wrap the model builder to match GetTestQDQModelFn signature (output_qparams unused when loading from context).
  auto qdq_model_fn = [&build_multi_partition_graph](ModelTestBuilder& builder,
                                                     std::vector<QuantParams<uint8_t>>& /*output_qparams*/) {
    build_multi_partition_graph(builder);
  };

  TestQDQModelAccuracy<uint8_t>(build_multi_partition_graph,
                                qdq_model_fn,
                                provider_options,  // QnnMockSSR.dll
                                13,
                                ExpectedEPNodeAssignment::Some,
                                QDQTolerance(),
                                OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                                context_model_file,
                                run_session_opts);

  CleanUpCtxFile(context_model_file);
}

// Test that SSR in JIT flow (no EPContext, no .bin file) returns ORT_ENGINE_ERROR to the user.
// In JIT mode, the graph is compiled at runtime with no external binary to reload from,
// so SSR recovery is impossible. The error should be ORT_ENGINE_ERROR (not ORT_EP_FAIL)
// to allow the application to distinguish NPU crashes from other EP failures.
TEST_F(QnnMockSSRBackendTests, SSRGraphExecuteJitReturnsEngineError) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::string op_type = "Atan";
  const TestInputDef<float> input_def_qdq({1, 2, 3}, false, -10.0f, 10.0f);

  // Use TestQDQModelAccuracy's approach: run f32 model on CPU first to get output ranges,
  // then build the QDQ model with proper quantization params.
  auto f32_model_fn = BuildOpTestCase<float>(op_type + "_node", op_type, {input_def_qdq}, {}, {});
  auto qdq_model_fn = BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def_qdq}, {}, {});

  const int opset_version = 14;
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};

  // Step 1: Run f32 model on CPU to get output ranges for quantization.
  ModelTestBuilder f32_helper;
  f32_model_fn(f32_helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{f32_helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  f32_helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string f32_model_data;
  f32_helper.model_.SerializeToString(&f32_model_data);

  // Run on CPU to get output ranges.
  Ort::SessionOptions cpu_so;
  Ort::Session cpu_session(*ort_env, f32_model_data.data(), f32_model_data.size(), cpu_so);

  auto in_name_alloc = cpu_session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto out_name_alloc = cpu_session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<int64_t> input_shape{1, 2, 3};
  std::vector<float> input_data(6, 1.0f);
  auto input_tensor = Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                               input_shape.data(), input_shape.size());

  const char* cpu_input_names[] = {in_name_alloc.get()};
  const char* cpu_output_names[] = {out_name_alloc.get()};
  auto cpu_outputs = cpu_session.Run(Ort::RunOptions{}, cpu_input_names, &input_tensor, 1,
                                     cpu_output_names, 1);

  // Compute output quantization params from CPU output.
  auto* output_data = cpu_outputs[0].GetTensorData<float>();
  auto output_count = cpu_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
  float out_min = *std::min_element(output_data, output_data + output_count);
  float out_max = *std::max_element(output_data, output_data + output_count);
  std::vector<QuantParams<uint8_t>> output_qparams = {QuantParams<uint8_t>::Compute(out_min, out_max)};

  // Step 2: Build QDQ model with proper quantization params.
  ModelTestBuilder qdq_helper;
  qdq_model_fn(qdq_helper, output_qparams);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{qdq_helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  qdq_helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  qdq_helper.model_.SerializeToString(&model_data);

  // Step 3: Create session in JIT mode using QnnMockSSR.dll (no EPContext caching).
  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device),
                          Ort::Session(*ort_env, model_data.data(), model_data.size(), so));

  // Step 4: Run inference — SSR fires, recovery is impossible, expect ORT_ENGINE_ERROR.
  auto qnn_in_name = scoped.session().GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  auto qnn_out_name = scoped.session().GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

  auto run_input_tensor = Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                   input_shape.data(), input_shape.size());

  const char* run_input_names[] = {qnn_in_name.get()};
  const char* run_output_names[] = {qnn_out_name.get()};

  try {
    scoped.session().Run(Ort::RunOptions{}, run_input_names, &run_input_tensor, 1, run_output_names, 1);
    FAIL() << "Expected ORT_ENGINE_ERROR exception but Run() succeeded.";
  } catch (const Ort::Exception& e) {
    EXPECT_EQ(e.GetOrtErrorCode(), ORT_ENGINE_ERROR)
        << "Expected ORT_ENGINE_ERROR for SSR in JIT flow, got error code: " << e.GetOrtErrorCode()
        << ", message: " << e.what();
  }
}

// Test that SSR in AOT flow with embed_mode=1 returns ORT_ENGINE_ERROR to the user.
// In embed_mode=1, the context binary is stored inside the ONNX model (no external .bin file),
// so SSR recovery is impossible. The error should be ORT_ENGINE_ERROR.
TEST_F(QnnMockSSRBackendTests, SSRGraphExecuteEpContextEmbedModeReturnsEngineError) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  const std::string context_model_file = "./ssr_ep_ctx_embed_mode_test.onnx";
  std::remove(context_model_file.c_str());

  const std::string op_type = "Atan";
  const TestInputDef<float> input_def_qdq({1, 2, 3}, false, -10.0f, 10.0f);

  // -----------------------------------------------------------------------
  // Step 1: Generate an embed_mode=1 context model with the real HTP backend.
  // -----------------------------------------------------------------------
  ProviderOptions htp_options;
  htp_options["backend_type"] = "htp";
  htp_options["offload_graph_io_quantization"] = "0";

  std::unordered_map<std::string, std::string> gen_session_opts;
  gen_session_opts.emplace(kOrtSessionOptionEpContextEnable, "1");
  gen_session_opts.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);
  gen_session_opts.emplace(kOrtSessionOptionEpContextEmbedMode, "1");

  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def_qdq}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def_qdq}, {}, {}),
                       htp_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",
                       gen_session_opts);

  ASSERT_TRUE(std::filesystem::exists(context_model_file))
      << "Context model file was not generated: " << context_model_file;

  // -----------------------------------------------------------------------
  // Step 2: Load the embed_mode=1 context model via QnnMockSSR.dll.
  //         SSR fires on graphExecute, recovery is impossible (no external .bin),
  //         expect ORT_ENGINE_ERROR.
  // -----------------------------------------------------------------------
  {
    onnx::ModelProto ctx_model_proto;
    std::ifstream ifs(context_model_file, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.good());
    ASSERT_TRUE(ctx_model_proto.ParseFromIstream(&ifs));

    std::string ctx_model_data;
    ctx_model_proto.SerializeToString(&ctx_model_data);

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

    ScopedOrtSession scoped(std::move(registered_ep_device),
                            Ort::Session(*ort_env, ctx_model_data.data(), ctx_model_data.size(), so));

    // Run inference — SSR fires, no .bin file to reload from, expect ORT_ENGINE_ERROR.
    auto in_name = scoped.session().GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto out_name = scoped.session().GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());

    Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::vector<int64_t> input_shape{1, 2, 3};
    std::vector<float> input_data(6, 1.0f);
    auto input_tensor = Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                 input_shape.data(), input_shape.size());

    const char* input_names[] = {in_name.get()};
    const char* output_names[] = {out_name.get()};

    try {
      scoped.session().Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);
      FAIL() << "Expected ORT_ENGINE_ERROR exception but Run() succeeded.";
    } catch (const Ort::Exception& e) {
      EXPECT_EQ(e.GetOrtErrorCode(), ORT_ENGINE_ERROR)
          << "Expected ORT_ENGINE_ERROR for SSR in embed_mode=1, got error code: " << e.GetOrtErrorCode()
          << ", message: " << e.what();
    }
  }

  std::remove(context_model_file.c_str());
}

// Build a simple QDQ Add model and save to file (for weight sharing tests).
static void CreateQdqAddModel(const std::string& model_file_name) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  gsl::span<float> data_range = gsl::make_span(data);
  QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);

  MakeTestInput(helper, "add_in1", TestInputDef<float>({2, 3}, false, data));
  std::string add_input1_qdq =
      AddQDQNodePair<uint8_t>(helper, "add_in1_qdq", "add_in1", q_parameter.scale, q_parameter.zero_point);

  MakeTestInput(helper, "add_in2", TestInputDef<float>({2, 3}, true, data));
  std::string add_input2_qdq =
      AddQDQNodePair<uint8_t>(helper, "add_in2_qdq", "add_in2", q_parameter.scale, q_parameter.zero_point);

  helper.AddNode("Add_node", "Add", {add_input1_qdq, add_input2_qdq}, {"add_out"});
  AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(helper, "qdq_out", "add_out",
                                                 q_parameter.scale, q_parameter.zero_point);

  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::ofstream model_ofs(model_file_name, std::ios::binary);
  ASSERT_TRUE(model_ofs.good());
  ASSERT_TRUE(helper.model_.SerializeToOstream(&model_ofs));
}

// Test SSR with weight sharing (htp_share_resource_optimization=1).
// Two models share the same QNN context binary. When SSR occurs, each model
// independently recovers by reloading the shared binary and retrieving its graph.
TEST_F(QnnMockSSRBackendTests, SSRGraphExecuteEpContextWeightSharing) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    !(QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 34))
  GTEST_SKIP() << "HTP weight sharing on ARM64 requires QNN API version >= 2.34.";
#else
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  // Create 2 identical QDQ models.
  std::string model_path1 = "./ssr_ws_model1.onnx";
  std::string model_path2 = "./ssr_ws_model2.onnx";
  std::string ctx_path1 = "./ssr_ws_model1_ctx.onnx";
  std::string ctx_path2 = "./ssr_ws_model2_ctx.onnx";
  std::remove(model_path1.c_str());
  std::remove(model_path2.c_str());
  std::remove(ctx_path1.c_str());
  std::remove(ctx_path2.c_str());

  CreateQdqAddModel(model_path1);
  CreateQdqAddModel(model_path2);

  // -----------------------------------------------------------------------
  // Step 1: Generate shared context binary with real HTP backend.
  // -----------------------------------------------------------------------
  {
    ProviderOptions htp_options;
    htp_options["backend_type"] = "htp";
    htp_options["offload_graph_io_quantization"] = "0";

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
    so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, htp_options);

#if defined(_WIN32)
    std::wstring model_path1_w(model_path1.begin(), model_path1.end());
    std::wstring model_path2_w(model_path2.begin(), model_path2.end());
    ScopedOrtSession scoped1(std::move(registered_ep_device),
                             Ort::Session(*ort_env, model_path1_w.c_str(), so));

    so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
    Ort::Session session2(*ort_env, model_path2_w.c_str(), so);
#else
    ScopedOrtSession scoped1(std::move(registered_ep_device),
                             Ort::Session(*ort_env, model_path1.c_str(), so));

    so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
    Ort::Session session2(*ort_env, model_path2.c_str(), so);
#endif
  }

  ASSERT_TRUE(std::filesystem::exists(ctx_path1)) << "Context model 1 not generated.";
  ASSERT_TRUE(std::filesystem::exists(ctx_path2)) << "Context model 2 not generated.";

  // Verify both context models point to the same .bin file.
  std::string bin_name1, bin_name2;
  GetContextBinaryFileName(ctx_path1, bin_name1);
  GetContextBinaryFileName(ctx_path2, bin_name2);
  ASSERT_EQ(bin_name1, bin_name2) << "Weight-sharing models should share the same .bin file.";
  ASSERT_TRUE(std::filesystem::exists(bin_name1));

  // -----------------------------------------------------------------------
  // Step 2: Load context model with weight sharing and verify SSR recovery accuracy.
  // -----------------------------------------------------------------------
  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<int64_t> input_shape{2, 3};
  std::vector<float> input_data(6, 1.0f);

  // Reference run with real HTP backend (without share_ep_contexts to avoid state conflict).
  std::vector<float> reference_output;
  {
    ProviderOptions real_htp_options;
    real_htp_options["backend_type"] = "htp";
    real_htp_options["offload_graph_io_quantization"] = "0";

    Ort::SessionOptions so1;

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so1, kQnnExecutionProvider, real_htp_options);

#if defined(_WIN32)
    std::wstring ctx_path1_w(ctx_path1.begin(), ctx_path1.end());
    ScopedOrtSession scoped1(std::move(registered_ep_device),
                             Ort::Session(*ort_env, ctx_path1_w.c_str(), so1));
#else
    ScopedOrtSession scoped1(std::move(registered_ep_device),
                             Ort::Session(*ort_env, ctx_path1.c_str(), so1));
#endif

    auto in_name = scoped1.session().GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto out_name = scoped1.session().GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto input_tensor = Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                 input_shape.data(), input_shape.size());
    const char* input_names[] = {in_name.get()};
    const char* output_names[] = {out_name.get()};
    auto outputs = scoped1.session().Run(Ort::RunOptions{}, input_names, &input_tensor, 1,
                                         output_names, 1);
    auto* data = outputs[0].GetTensorData<float>();
    auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    reference_output.assign(data, data + count);
  }

  // SSR run with QnnMockSSR.dll — both sessions share the same EP device.
  {
    Ort::SessionOptions so1;
    so1.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so1, kQnnExecutionProvider, provider_options);

    Ort::SessionOptions so2;
    so2.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
    so2.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, provider_options);

#if defined(_WIN32)
    std::wstring ctx_path1_w(ctx_path1.begin(), ctx_path1.end());
    std::wstring ctx_path2_w(ctx_path2.begin(), ctx_path2.end());
    ScopedOrtSession scoped1(std::move(registered_ep_device),
                             Ort::Session(*ort_env, ctx_path1_w.c_str(), so1));
    Ort::Session session2(*ort_env, ctx_path2_w.c_str(), so2);
#else
    ScopedOrtSession scoped1(std::move(registered_ep_device),
                             Ort::Session(*ort_env, ctx_path1.c_str(), so1));
    Ort::Session session2(*ort_env, ctx_path2.c_str(), so2);
#endif

    // Run session 1 — SSR fires on first graphExecute, recovery succeeds.
    auto in_name1 = scoped1.session().GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto out_name1 = scoped1.session().GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto input_tensor1 = Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                  input_shape.data(), input_shape.size());
    const char* input_names1[] = {in_name1.get()};
    const char* output_names1[] = {out_name1.get()};

    auto outputs1 = scoped1.session().Run(Ort::RunOptions{}, input_names1, &input_tensor1, 1,
                                          output_names1, 1);
    ASSERT_EQ(outputs1.size(), 1u);
    ASSERT_TRUE(outputs1[0].IsTensor());

    // Run session 2 — reuses the recovered context via GetQnnContext(0).
    auto in_name2 = session2.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto out_name2 = session2.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto input_tensor2 = Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                  input_shape.data(), input_shape.size());
    const char* input_names2[] = {in_name2.get()};
    const char* output_names2[] = {out_name2.get()};

    auto outputs2 = session2.Run(Ort::RunOptions{}, input_names2, &input_tensor2, 1,
                                 output_names2, 1);
    ASSERT_EQ(outputs2.size(), 1u);
    ASSERT_TRUE(outputs2[0].IsTensor());

    // Verify accuracy for both sessions against reference.
    auto* ssr_data1 = outputs1[0].GetTensorData<float>();
    auto count1 = outputs1[0].GetTensorTypeAndShapeInfo().GetElementCount();
    ASSERT_EQ(count1, reference_output.size());
    for (size_t i = 0; i < count1; ++i) {
      EXPECT_NEAR(ssr_data1[i], reference_output[i], 1e-5f)
          << "Session 1 output mismatch at index " << i;
    }

    auto* ssr_data2 = outputs2[0].GetTensorData<float>();
    auto count2 = outputs2[0].GetTensorTypeAndShapeInfo().GetElementCount();
    ASSERT_EQ(count2, reference_output.size());
    for (size_t i = 0; i < count2; ++i) {
      EXPECT_NEAR(ssr_data2[i], reference_output[i], 1e-5f)
          << "Session 2 output mismatch at index " << i;
    }
  }

  // Cleanup.
  std::remove(model_path1.c_str());
  std::remove(model_path2.c_str());
  std::remove(ctx_path1.c_str());
  std::remove(ctx_path2.c_str());
  std::remove(bin_name1.c_str());
#endif
}

#endif  // defined(_WIN32) && (defined(_M_ARM64) || defined(_M_ARM64EC))
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
