// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdlib.h>

#include <cmath>
#include <filesystem>
#include <string>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_ep_device_ep_metadata_keys.h"
#include "onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "CPU/QnnCpuCommon.h"
#include "HTP/QnnHtpCommon.h"
#include "QnnSdkBuildId.h"

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

using namespace ONNX_NAMESPACE;

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
static void LoadOnnxModelFromFile(const std::string& path, onnx::ModelProto& out_model) {
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  if (!fin) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  if (!out_model.ParseFromIstream(&fin)) {
    throw std::runtime_error("ParseFromIstream() failed for: " + path);
  }
  fin.close();
  return;
}

// from the context cache Onnx model, find the EPContext node with main_context=1,
// and get the QNN context binary file name
static void GetContextBinaryFileName(const std::string& onnx_ctx_file,
                                     std::string& last_ctx_bin_file) {
  // Load model using ONNX protobuf API (avoid onnxruntime::Model).
  ONNX_NAMESPACE::ModelProto ctx_model_proto;
  std::ifstream ifs(onnx_ctx_file, std::ios::in | std::ios::binary);
  ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << onnx_ctx_file;
  ASSERT_TRUE(ctx_model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << onnx_ctx_file;

  // Iterate nodes in the main graph and find EPContext node with main_context == 1.
  for (const auto& node : ctx_model_proto.graph().node()) {
    if (node.op_type() != "EPContext") {
      continue;
    }

    int64_t is_main_context = 0;
    std::string ep_cache_context;

    for (const auto& attr : node.attribute()) {
      if (attr.name() == "main_context") {
        is_main_context = attr.i();
      } else if (attr.name() == "ep_cache_context") {
        ep_cache_context = attr.s();
      }
    }

    if (is_main_context == 1) {
      last_ctx_bin_file = ep_cache_context;
      return;
    }
  }
}

// Get context binary file name from Context model file and remove it with the context model file
void CleanUpCtxFile(std::string context_file_path) {
  std::string qnn_ctx_binary_file_name;
  GetContextBinaryFileName(context_file_path, qnn_ctx_binary_file_name);

  std::filesystem::path ctx_model_path(context_file_path);

  std::string qnn_ctx_binary_file_path = (ctx_model_path.remove_filename().string() + qnn_ctx_binary_file_name);
  ASSERT_EQ(std::remove(qnn_ctx_binary_file_path.c_str()), 0);
  ASSERT_EQ(std::remove(context_file_path.c_str()), 0);
}

// Create a model with FusedGemm + Add (quantized)
// input1 -> Add -> Q -> DQ ----
//                              |
//        input2 -> Q -> DQ -> FusedGemm -> Q -> DQ -> output
static GetTestModelFn BuildGraphWithQAndNonQ(bool single_ep_node = true) {
  return [single_ep_node](ModelTestBuilder& builder) {
    // Create non-quantized FusedGemm node1
    std::vector<float> data(200 * 200, 1.0f);
    MakeTestInput(builder, "input1", TestInputDef<float>({200, 200}, false, data));
    MakeTestInput(builder, "add1_ini_input2", TestInputDef<float>({200, 200}, true, data));
    std::vector<ONNX_NAMESPACE::AttributeProto> fusedgemm_attrs;
    fusedgemm_attrs.push_back(builder.MakeStringAttribute("activation", "Relu"));
    builder.AddNode("FusedGemm_node0",
                    "FusedGemm",
                    {"input1", "add1_ini_input2"},
                    {"add1_out"},
                    kMSDomain,
                    fusedgemm_attrs);

    // Create quantized Add node2
    std::vector<float> add_data(12, 1.0f);
    gsl::span<float> data_range = gsl::make_span(add_data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);
    std::string add2_input1_qdq =
        AddQDQNodePair<uint8_t>(builder, "add2_in1_qdq", "add1_out", q_parameter.scale, q_parameter.zero_point);

    MakeTestInput(builder, "add2_input2", TestInputDef<float>({200, 200}, true, data));
    std::string add2_input2_qdq =
        AddQDQNodePair<uint8_t>(builder, "add2_in2_qdq", "add2_input2", q_parameter.scale, q_parameter.zero_point);

    builder.AddNode("Add_node0",
                    "Add",
                    {add2_input1_qdq, add2_input2_qdq},
                    {"add2_out"});

    if (single_ep_node) {
      // add2_out -> Q -> DQ -> output
      auto final_out = AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "add2_out", q_parameter.scale, q_parameter.zero_point);
    } else {
      std::string add3_input1_qdq =
          AddQDQNodePair<uint8_t>(builder, "add3_in1_qdq", "add2_out", q_parameter.scale, q_parameter.zero_point);

      MakeTestInput(builder, "add3_ini_input2", TestInputDef<float>({200, 200}, true, data));

      std::vector<ONNX_NAMESPACE::AttributeProto> fusedgemm_attrs2;
      fusedgemm_attrs2.push_back(builder.MakeStringAttribute("activation", "Relu"));
      builder.AddNode("FusedGemm_node1",
                      "FusedGemm",
                      {add3_input1_qdq, "add3_ini_input2"},
                      {"add3_out"},
                      kMSDomain,
                      fusedgemm_attrs2);

      // Create quantized Add node4
      std::string add4_input1_qdq =
          AddQDQNodePair<uint8_t>(builder, "add4_in1_qdq", "add3_out", q_parameter.scale, q_parameter.zero_point);

      MakeTestInput(builder, "add4_input2", TestInputDef<float>({200, 200}, true, data));
      std::string add4_input2_qdq =
          AddQDQNodePair<uint8_t>(builder, "add4_in2_qdq", "add4_input2", q_parameter.scale, q_parameter.zero_point);

      builder.AddNode("Add_node1",
                      "Add",
                      {add4_input1_qdq, add4_input2_qdq},
                      {"add4_out"});

      auto final_out = AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "add4_out", q_parameter.scale, q_parameter.zero_point);
    }
  };
}

void QnnContextBinaryMultiPartitionTestBody(bool single_ep_node = true) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  // Build input model via ModelProto directly (avoid onnxruntime::Model/Graph/Resolve).
  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);

  // opset imports
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_model_file = "./testdata/qnn_context_binary_multi_partition_test.onnx";
  std::remove(context_model_file.c_str());

  // Use ONNX protobuf APIs (ModelProto) instead of onnxruntime::Model/Graph APIs.
  ONNX_NAMESPACE::ModelProto ctx_model_proto;

  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

    // Make sure the Qnn context cache binary file is generated
    EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

    // Load generated context model into ModelProto
    std::ifstream ifs(context_model_file, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << context_model_file;
    ASSERT_TRUE(ctx_model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << context_model_file;

    int ep_context_node_count = 0;
    int non_ep_context_node_count = 0;

    for (const auto& node : ctx_model_proto.graph().node()) {
      if (node.op_type() == "EPContext") {
        ++ep_context_node_count;

        // validate the fix for the partition issue relate to QDQ model
        ASSERT_EQ(node.input_size(), 1);
      } else {
        ++non_ep_context_node_count;
      }
    }

    int expected_node_count = single_ep_node ? 1 : 2;
    ASSERT_EQ(ep_context_node_count, expected_node_count);
    ASSERT_EQ(non_ep_context_node_count, expected_node_count);
  }

  Ort::SessionOptions so2;
  // context file path is required if it's non-embed mode and the model is loaded from memory
  so2.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so2, kQnnExecutionProvider, provider_options);

  std::string ctx_model_data;
  ctx_model_proto.SerializeToString(&ctx_model_data);
  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, ctx_model_data.data(), ctx_model_data.size(), so2));

  // clean up
  CleanUpCtxFile(context_model_file);
}

struct TestModel {
  std::unique_ptr<ModelTestBuilder> builder;

  std::string Serialize() const {
    std::string model_data;
    builder->model_.SerializeToString(&model_data);
    return model_data;
  }

  bool Save(const ORTCHAR_T* path) const {
    std::ofstream ofs(path, std::ios::binary);
    return builder->model_.SerializeToOstream(&ofs);
  }
};

// Create a test model from a function that programmatically builds a graph.
// Note: We intentionally avoid onnxruntime::Model + Graph::Resolve() here. These tests only need a valid ONNX
// ModelProto for feeding into Ort::Session / Ort::Compile_extractor.
static void CreateTestModel(test::GetTestModelFn graph_builder,
                            int onnx_opset_version,
                            OrtLoggingLevel log_severity,
                            TestModel& test_model) {
  QNN_TEST_UNUSED_PARAMETER(log_severity);
  const std::unordered_map<std::string, int> domain_to_version = {{"", onnx_opset_version}, {kMSDomain, 1}};

  test_model.builder = std::make_unique<ModelTestBuilder>();
  graph_builder(*test_model.builder);

  // Populate opset imports (similar to RunQnnModelTest).
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{test_model.builder->model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }

  // Keep IR version consistent with other QNN ABI tests.
  test_model.builder->model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
}

// Helper that checks that a compiled model has the expected number of EPContext nodes.
static void CheckEpContextNodeCounts(const ONNX_NAMESPACE::ModelProto& model_proto,
                                     int expected_ep_context_node_count,
                                     int expected_other_node_count) {
  int ep_context_node_count = 0;
  int non_ep_context_node_count = 0;

  // Iterate through nodes in the main graph
  for (const auto& node : model_proto.graph().node()) {
    if (node.op_type() == "EPContext") {
      ++ep_context_node_count;
      // validate the fix for the partition issue relate to QDQ model
      ASSERT_EQ(node.input_size(), 1);
    } else {
      ++non_ep_context_node_count;
    }
  }

  EXPECT_EQ(ep_context_node_count, expected_ep_context_node_count);
  EXPECT_EQ(non_ep_context_node_count, expected_other_node_count);
}

// Helper to check that a compiled model (stored as a file) has the expected number of EPContext nodes.
static void CheckEpContextNodeCounts(const ORTCHAR_T* model_path,
                                     int expected_ep_context_node_count,
                                     int expected_other_node_count) {
  ONNX_NAMESPACE::ModelProto model_proto;
  std::ifstream ifs(model_path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << model_path;
  ASSERT_TRUE(model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << model_path;
  ifs.close();
  CheckEpContextNodeCounts(model_proto, expected_ep_context_node_count, expected_other_node_count);
}

// Helper to check that a compiled model (stored in a buffer) has the expected number of EPContext nodes.
static void CheckEpContextNodeCounts(void* model_buffer, size_t model_buffer_size,
                                     int expected_ep_context_node_count,
                                     int expected_other_node_count) {
  ONNX_NAMESPACE::ModelProto model_proto;
  ASSERT_TRUE(model_proto.ParseFromArray(model_buffer, static_cast<int>(model_buffer_size)))
      << "Failed to parse model from buffer";
  CheckEpContextNodeCounts(model_proto, expected_ep_context_node_count, expected_other_node_count);
}

// Test workflow that:
//   - Creates session that disables EP compilation.
//   - Session creation fails because input model is not pre-compiled.
//   - Uses OrtCompileApi to compile the model.
//   - Recreates session with the compiled model.

TEST_F(QnnHTPBackendTests, CompileApi_DisableEpCompile_ThenCompileExplicitly) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_disable_compile_input.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("./compileapi_disable_compile_output.onnx");
  std::filesystem::remove(input_model_file);
  std::filesystem::remove(output_model_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  ASSERT_TRUE(test_model.Save(input_model_file));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  so.AddConfigEntry(kOrtSessionOptionsDisableModelCompile, "1");  // Disable model compilation!

  // Create an inference session that fails with error ORT_MODEL_REQUIRES_COMPILATION
  try {
    Ort::Session session(*ort_env, input_model_file, so);
    FAIL() << "Expected Session creation to fail but it succeeded";  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    OrtErrorCode error_code = excpt.GetOrtErrorCode();
    std::string_view error_msg = excpt.what();
    ASSERT_EQ(error_code, ORT_MODEL_REQUIRES_COMPILATION);
    ASSERT_THAT(error_msg, testing::HasSubstr(kQnnExecutionProvider));
  }

  // Session creation failed because the model was not pre-compiled.
  // Try to compile it now.

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was generated and has the expected number of EPContext nodes.
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_file, so)));
  std::filesystem::remove(output_model_file);

  std::filesystem::remove(input_model_file);
}

// Test using the CompileModel() API with settings:
//   - input model file
//   - output model file
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_InputModelFromPath) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_fromsessionoptions_inputmodelfrompath.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("./qnn_context_binary_multi_partition_test.onnx");
  std::filesystem::remove(input_model_file);
  std::filesystem::remove(output_model_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  ASSERT_TRUE(test_model.Save(input_model_file));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was generated and has the expected number of EPContext nodes.
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_file, so)));
  std::filesystem::remove(output_model_file);
}

// Test using the CompileModel() API with settings:
//   - input model from buffer
//   - output model file
//   - EPContext nodes in output model use embedded binary blobs.
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_InputModelAsBuffer_Embedded) {
  // Create a test model and serialize it to a buffer.
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  std::string model_data = test_model.Serialize();

  const ORTCHAR_T* output_model_file = ORT_TSTR("./qnn_context_binary_multi_partition_test.onnx");
  std::filesystem::remove(output_model_file);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_data.data()), model_data.size());
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetEpContextEmbedMode(true);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was generated and has the expected number of EPContext nodes.
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_file, so)));
  std::filesystem::remove(output_model_file);
}

// Test using the CompileModel() API with settings:
//   - input model from file
//   - save output model to a buffer
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_OutputModelBuffer) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_fromsessionoptions_inputmodelfrompath.onnx");
  std::filesystem::remove(input_model_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  ASSERT_TRUE(test_model.Save(input_model_file));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Create model compilation options from the session options. Output model is stored in a buffer.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  Ort::AllocatorWithDefaultOptions allocator;
  void* output_model_buffer = nullptr;
  size_t output_model_buffer_size = 0;
  compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was saved to the buffer.
  ASSERT_TRUE(output_model_buffer != nullptr);
  ASSERT_TRUE(output_model_buffer_size > 0);

  // Check that the compiled model has the expected number of EPContext nodes.
  CheckEpContextNodeCounts(output_model_buffer, output_model_buffer_size, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_buffer, output_model_buffer_size, so)));

  allocator.Free(output_model_buffer);
}

// Test using the CompileModel() API with settings:
//   - input model from buffer
//   - save output model to buffer
//   - test enabling AND disabling embed mode for context binary in EPContext node attributes
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_InputAndOutputModelsInBuffers) {
  // Create a test model and serialize it to a buffer.
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  std::string model_data = test_model.Serialize();

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);

  Ort::AllocatorWithDefaultOptions allocator;

  // Test embed mode enabled.
  {
    void* output_model_buffer = nullptr;
    size_t output_model_buffer_size = 0;

    // Create model compilation options from the session options.
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_data.data()), model_data.size());
    compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);
    compile_options.SetEpContextEmbedMode(true);
    compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // Compile the model.
    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    // Make sure the compiled model was saved to the buffer.
    ASSERT_TRUE(output_model_buffer != nullptr);
    ASSERT_TRUE(output_model_buffer_size > 0);

    // Check that the compiled model has the expected number of EPContext nodes.
    CheckEpContextNodeCounts(output_model_buffer, output_model_buffer_size, 2, 2);

    // Should be able to create a session with the compiled model and the original session options.
    EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_buffer, output_model_buffer_size, session_options)));

    allocator.Free(output_model_buffer);
  }

  // Test embed mode disabled.
  {
    void* output_model_buffer = nullptr;
    size_t output_model_buffer_size = 0;

    // Create model compilation options from the session options.
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_data.data()), model_data.size());
    compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);
    std::string target_dir = "./testdata/";
    std::string model_name = "test_model_in_mem.onnx";
    auto pos = model_name.rfind(".onnx");
    std::string bin_file_name = model_name.substr(0, pos) + "_qnn.bin";
#if defined(_WIN32)
    // Use public ORT API surface: `Ort::SessionOptions` / `ORTCHAR_T`-based strings on Windows.
    const std::wstring target_dir_w = std::wstring(target_dir.begin(), target_dir.end());
    const std::wstring model_name_w = std::wstring(model_name.begin(), model_name.end());
    compile_options.SetEpContextBinaryInformation(target_dir_w.c_str(), model_name_w.c_str());
#else
    // On non-Windows platforms ORTCHAR_T is `char`, so std::string is already correct.
    compile_options.SetEpContextBinaryInformation(target_dir.c_str(), model_name.c_str());
#endif
    compile_options.SetEpContextEmbedMode(false);
    compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // Compile the model.
    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    // Make sure the compiled model was saved to the buffer.
    ASSERT_TRUE(output_model_buffer != nullptr);
    ASSERT_TRUE(output_model_buffer_size > 0);

    ASSERT_TRUE(std::filesystem::exists(target_dir + bin_file_name)) << "expected context binary file should exist";

    // Check that the compiled model has the expected number of EPContext nodes.
    CheckEpContextNodeCounts(output_model_buffer, output_model_buffer_size, 2, 2);

    // Add session option "ep.context_file_path" so that the session can use it to locate the [model_name]_qnn.bin file
    std::string ctx_model = target_dir + model_name;
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_model.c_str());
    // Should be able to create a session with the compiled model and the original session options.
    EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_buffer, output_model_buffer_size, session_options)));

    std::filesystem::remove(target_dir + bin_file_name);
    allocator.Free(output_model_buffer);
  }
}

// Test using the CompileModel() API with settings:
//   - input model from file
//   - save output model to a buffer
//   - save initializers (used by CPU EP) to external file.
//   - EPContext nodes in output model use embedded binary blobs.
TEST_F(QnnHTPBackendTests, CompileApi_FromSessionOptions_OutputModelBuffer_OutputInitializersFile) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_fromsessionoptions_outputmodelbuffer_initializers.onnx");
  const ORTCHAR_T* output_initializers_file = ORT_TSTR("./compileapi_initializers.bin");
  std::filesystem::remove(input_model_file);
  std::filesystem::remove(output_initializers_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  ASSERT_TRUE(test_model.Save(input_model_file));

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions so;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Create model compilation options from the session options. Output model is stored in a buffer.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);

  Ort::AllocatorWithDefaultOptions allocator;
  void* output_model_buffer = nullptr;
  size_t output_model_buffer_size = 0;
  compile_options.SetOutputModelBuffer(allocator, &output_model_buffer, &output_model_buffer_size);
  compile_options.SetOutputModelExternalInitializersFile(output_initializers_file, 0);
  compile_options.SetEpContextEmbedMode(true);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Make sure the compiled model was saved to the buffer.
  ASSERT_TRUE(output_model_buffer != nullptr);
  ASSERT_TRUE(output_model_buffer_size > 0);

  // Make sure that the initializers were saved to an external file.
  ASSERT_TRUE(std::filesystem::exists(output_initializers_file));

  // Check that the compiled model has the expected number of EPContext nodes.
  CheckEpContextNodeCounts(output_model_buffer, output_model_buffer_size, 2, 2);

  // Should be able to create a session with the compiled model and the original session options.
  EXPECT_NO_THROW((Ort::Session(*ort_env, output_model_buffer, output_model_buffer_size, so)));

  allocator.Free(output_model_buffer);
  std::filesystem::remove(output_initializers_file);
}

// Test that the explicit compile API can be configured to return an error if the output model does not
// have EPContext nodes.
TEST_F(QnnHTPBackendTests, CompileApi_SetFlags_ErrorIfNoCompiledNodes) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const ORTCHAR_T* output_model_file = ORT_TSTR("should_not_be_generated.onnx");
  std::filesystem::remove(output_model_file);

  // Initialize session options with only CPU EP, which will not be able to compile any nodes.
  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);

  // Call CompileModel() but expect an error status.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_EQ(status.GetErrorCode(), ORT_FAIL);
  ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("Unable to compile any nodes"));

  // Make sure that the output file was *NOT* generated.
  ASSERT_FALSE(std::filesystem::exists(output_model_file));
}

// Test that the explicit compile API can be configured to return an error if the output model already exists and
// would have been overwritten.
TEST_F(QnnHTPBackendTests, CompileApi_SetFlags_ErrorIfOutputFileAlreadyExists) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const ORTCHAR_T* output_model_file = ORT_TSTR("mul_1_ctx_.onnx");
  std::filesystem::remove(output_model_file);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);

  // Compile with QNN EP. Should succeed the first time.
  {
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << "CompileModel() should succeed the first time a model is compiled.";
    ASSERT_TRUE(std::filesystem::exists(output_model_file)) << "compiled model should exist";
  }

  // Compiling the input model again should fail if we disallow overwriting the output file.
  {
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_OUTPUT_FILE_EXISTS);

    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_EQ(status.GetErrorCode(), ORT_FAIL);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("exists already"));
    ASSERT_TRUE(std::filesystem::exists(output_model_file)) << "original compiled model should still exist";

    std::filesystem::remove(output_model_file);
  }
}

// Tests that the explicit compile API returns an error if user tries to compile a compiled model.
// This scenario is silently ignored in the original compilation approach with session option configs.
TEST_F(QnnHTPBackendTests, CompileApi_ErrorIfCompilingACompiledModel) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const ORTCHAR_T* output_model_file = ORT_TSTR("mul_1_ctx_.onnx");
  std::filesystem::remove(output_model_file);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, kQnnExecutionProvider, provider_options);

  // Compile with QNN EP. Should succeed the first time.
  {
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << "CompileModel() should succeed the first time a model is compiled.";
    ASSERT_TRUE(std::filesystem::exists(output_model_file)) << "compiled model should exist";
  }

  // Compiling the compiled model should always fail: it's already compiled!
  {
    const ORTCHAR_T* new_output_model_file = ORT_TSTR("should_not_be_generated.onnx");  // Should not be generated.
    std::filesystem::remove(new_output_model_file);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(output_model_file);  // Set the compiled model as the input!
    compile_options.SetOutputModelPath(new_output_model_file);

    // Currently it would failed at ConvertEpContextNodes in ep_plugin_provider_interfaces.cc.
    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_EQ(status.GetErrorCode(), ORT_FAIL);
    ASSERT_THAT(status.GetErrorMessage(), testing::HasSubstr("OrtEp::Compile() returned a NULL EPContext node"));
    ASSERT_FALSE(std::filesystem::exists(new_output_model_file)) << "new compiled model should not be generated";
    ASSERT_TRUE(std::filesystem::exists(output_model_file)) << "original compiled model should still exist";

    std::filesystem::remove(output_model_file);
  }
}

// Uses the original compiling approach with session option configs (instead of explicit compile API).
// Test that ORT does not generate an output model if the model does not contain EPContext nodes.
// Also, ORT should not return an error.
TEST_F(QnnHTPBackendTests, QnnContextBinary_OriginalCompileApproach_NoCompiledNodesDoesntGenerateOutput) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const char* output_model_file = "should_not_be_generated.onnx";

  // Initialize session options with only CPU EP, which will not be able to compile any nodes.
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file);
  Ort::Session session(*ort_env, input_model_file, so);  // Should not throw an error.

  // Make sure that the output file was *NOT* generated.
  ASSERT_FALSE(std::filesystem::exists(output_model_file));
}

// Uses the original compiling approach with session option configs (instead of explicit compile API).
// Test that ORT does not generate an output model if the input model is already compiled.
// Also, ORT should not return an error.
TEST_F(QnnHTPBackendTests, QnnContextBinary_OriginalCompileApproach_IgnoreCompilingOfCompiledModel) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  const std::string output_model_file = "mul_1_ctx.onnx";
  std::filesystem::remove(output_model_file);

  ProviderOptions qnn_options = {{"backend_type", "htp"}};

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  qnn_options["num_graph_prepare_threads"] = "1";
#endif

  // Compile a model with QNN. This should succeed.
  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));  // check compiled model was generated.
  }

  // Try compiling the compiled model again. ORT should basically ignore it.
  {
    const char* new_output_model_file = "should_not_be_generated.onnx";  // will not be generated!
    std::filesystem::remove(new_output_model_file);

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, new_output_model_file);

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    // Currently it would failed at ConvertEpContextNodes in ep_plugin_provider_interfaces.cc.
    try {
#ifdef _WIN32
      std::wstring out_model_file(output_model_file.begin(), output_model_file.end());
#else
      std::string out_model_file(output_model_file.begin(), output_model_file.end());
#endif
      Ort::Session session(*ort_env, out_model_file.c_str(), so);
    } catch (const Ort::Exception& e) {
      ASSERT_EQ(e.GetOrtErrorCode(), ORT_FAIL);
      ASSERT_THAT(e.what(), testing::HasSubstr("OrtEp::Compile() returned a NULL EPContext node"));
    }

    // Session creation should not throw an error. And a new output model should not have been generated.
    ASSERT_FALSE(std::filesystem::exists(new_output_model_file));
    std::filesystem::remove(output_model_file);
  }
}

// Test that models with 1 non-quantized FusedGemm node and 1 quantized Add node can still generate the context binary
// The generated Onnx model has 1 FusedGemm node and 1 EPContext node
TEST_F(QnnHTPBackendTests, QnnContextBinaryMultiPartitionSupport1) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  QnnContextBinaryMultiPartitionTestBody(true);
}

// Test that models with 2 non-quantized FusedGemm nodes and 2 quantized Add nodes can still generate the context binary
// The generated Onnx model has 2 FusedGemm nodes and 1 EPContext nodes
TEST_F(QnnHTPBackendTests, QnnContextBinaryMultiPartitionSupport2) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  QnnContextBinaryMultiPartitionTestBody(false);
}

static void ExternalizeAllInitializersToFile(ONNX_NAMESPACE::ModelProto& model,
                                             const std::string& external_file_name,
                                             const std::string& external_file_full_path) {
  std::ofstream ext_ofs(external_file_full_path, std::ios::binary);
  ASSERT_TRUE(ext_ofs.good()) << "Failed to open external initializer file: " << external_file_full_path;

  int64_t offset = 0;
  for (auto& initializer : *model.mutable_graph()->mutable_initializer()) {
    // Ensure tensor data is in raw_data so we can write a single contiguous blob.
    if (!initializer.has_raw_data()) {
      // Convert other typed fields to raw_data if needed.
      if (initializer.data_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
        const size_t n = static_cast<size_t>(initializer.float_data_size());
        const float* src = initializer.float_data().data();
        initializer.set_raw_data(reinterpret_cast<const void*>(src), n * sizeof(float));
        initializer.clear_float_data();
      } else if (initializer.data_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
        const size_t n = static_cast<size_t>(initializer.int32_data_size());
        std::string bytes;
        bytes.resize(n);
        for (size_t i = 0; i < n; ++i) {
          bytes[i] = static_cast<char>(static_cast<uint8_t>(initializer.int32_data(static_cast<int>(i))));
        }
        initializer.set_raw_data(bytes);
        initializer.clear_int32_data();
      } else if (initializer.data_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) {
        const size_t n = static_cast<size_t>(initializer.int32_data_size());
        std::string bytes;
        bytes.resize(n);
        for (size_t i = 0; i < n; ++i) {
          bytes[i] = static_cast<char>(static_cast<int8_t>(initializer.int32_data(static_cast<int>(i))));
        }
        initializer.set_raw_data(bytes);
        initializer.clear_int32_data();
      }
    }

    // IMPORTANT: take a copy of raw_data BEFORE we clear it (protobuf string reference would become empty).
    const std::string raw_copy = initializer.raw_data();
    ASSERT_FALSE(raw_copy.empty()) << "Initializer has no data: " << initializer.name();

    ext_ofs.write(raw_copy.data(), raw_copy.size());
    ASSERT_TRUE(ext_ofs.good()) << "Failed writing external initializer bytes";

    // Mark tensor as external and point to the external file.
    initializer.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);

    initializer.clear_raw_data();
    initializer.clear_float_data();
    initializer.clear_int32_data();
    initializer.clear_int64_data();
    initializer.clear_double_data();
    initializer.clear_string_data();
    initializer.clear_uint64_data();

    initializer.clear_external_data();
    auto* kv_loc = initializer.add_external_data();
    kv_loc->set_key("location");
    kv_loc->set_value(external_file_name);

    auto* kv_off = initializer.add_external_data();
    kv_off->set_key("offset");
    kv_off->set_value(std::to_string(offset));

    auto* kv_len = initializer.add_external_data();
    kv_len->set_key("length");
    kv_len->set_value(std::to_string(raw_copy.size()));

    offset += static_cast<int64_t>(raw_copy.size());
  }

  ext_ofs.flush();
  ext_ofs.close();
}

void EpCtxCpuNodeWithExternalIniFileTestBody(bool expect_external_ini_file, bool load_model_from_buffer = false) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  // Build input model via ModelProto directly (avoid onnxruntime::Model/Graph/Resolve).
  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ(true)(helper);

  // opset imports
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Keep ONNX model + external initializer file in testdata folder so it can be resolved via
  // kOrtSessionOptionsModelExternalInitializersFileFolderPath.
  const std::string ext_folder = "./testdata/";
  const std::string model_with_ext = ext_folder + "model_external.onnx";
  const std::string model_ext_file_name = "model_external.bin";  // NOTE: stored in TensorProto.external_data.location
  const std::string model_ext_file = ext_folder + model_ext_file_name;

  // Ensure external initializer folder exists before writing.
  std::filesystem::create_directories(ext_folder);

  std::filesystem::remove(model_with_ext);
  std::filesystem::remove(model_ext_file);

  // Use ONNX protobuf APIs (ModelProto external_data) to save initializers to model_external.bin
  ExternalizeAllInitializersToFile(helper.model_, model_ext_file_name, model_ext_file);

  std::ofstream model_ofs(model_with_ext, std::ios::binary);
  ASSERT_TRUE(model_ofs.good());
  ASSERT_TRUE(helper.model_.SerializeToOstream(&model_ofs));
  model_ofs.close();

  EXPECT_TRUE(std::filesystem::exists(model_with_ext.c_str()));
  EXPECT_TRUE(std::filesystem::exists(model_ext_file.c_str()));

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  const std::string ep_context_model_file = "./qnn_ctx_part_external_ini_ctx.onnx";
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_model_file.c_str());
  const std::string external_ini_file = "./qnn_ctx_part_external_ini.bin";
  if (expect_external_ini_file) {
    // Set the external ini file name will force all initializers to the external file
    so.AddConfigEntry(kOrtSessionOptionsEpContextModelExternalInitializersFileName, external_ini_file.c_str());
  }  // otherwise all initializers are in Onnx file, no external data file generated

  if (load_model_from_buffer) {
    std::vector<char> buffer;
    {
      buffer.resize(std::filesystem::file_size(model_with_ext));
      std::ifstream file(model_with_ext, std::ios::binary);
      if (!file)
        throw std::runtime_error("Error reading model");
      if (!file.read(buffer.data(), buffer.size()))
        throw std::runtime_error("Error reading model");
    }
    so.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath, ext_folder.c_str());
    Ort::Session session(*ort_env, buffer.data(), buffer.size(), so);
  } else {
#if defined(_WIN32)
    const std::wstring model_with_ext_w(model_with_ext.begin(), model_with_ext.end());
    Ort::Session session(*ort_env, model_with_ext_w.c_str(), so);
#else
    Ort::Session session(*ort_env, model_with_ext.c_str(), so);
#endif
  }

  EXPECT_TRUE(std::filesystem::exists(ep_context_model_file.c_str()));
  if (expect_external_ini_file) {
    EXPECT_TRUE(std::filesystem::exists(external_ini_file.c_str()));
    ASSERT_EQ(std::remove(external_ini_file.c_str()), 0);
  } else {
    EXPECT_FALSE(std::filesystem::exists(external_ini_file.c_str()));
  }

  // clean up
  ASSERT_EQ(std::remove(model_with_ext.c_str()), 0);
  ASSERT_EQ(std::remove(model_ext_file.c_str()), 0);
  CleanUpCtxFile(ep_context_model_file);
}

// Set the session option "ep.context_model_external_initializers_file_name" so FusedGemm (which fallback on CPU)
// will dump initializer data to external file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCpuNodeWithExternalWeights) {
  EpCtxCpuNodeWithExternalIniFileTestBody(true);
}

// Without setting the session option "ep.context_model_external_initializers_file_name"
// so FusedGemm (which fallback on CPU) will NOT dump initializer data to external file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCpuNodeWithoutExternalWeights) {
  EpCtxCpuNodeWithExternalIniFileTestBody(false);
}

// Load model from memory
// Without setting the session option "ep.context_model_external_initializers_file_name"
// so FusedGemm (which fallback on CPU) will NOT dump initializer data to external file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCpuNodeWithoutExternalWeightsModelFromMemory) {
  EpCtxCpuNodeWithExternalIniFileTestBody(false, true);
}

// Set ep.context_file_path to folder path which is not a valid option, check the error message
TEST_F(QnnHTPBackendTests, QnnContextBinaryGenerationFolderPathNotExpected) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Serialize the model to a string.
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_onnx_file = "./ep_context_folder_not_expected/";
  std::remove(ep_context_onnx_file.c_str());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_onnx_file.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

// Set ep.context_file_path to invalid file path, check the error message
TEST_F(QnnHTPBackendTests, QnnContextBinaryGenerationFolderPathNotExpected2) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  // Build input model via ModelProto directly (avoid onnxruntime::Model/Graph/Resolve).
  ModelTestBuilder helper;
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);

  // opset imports
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Serialize the model to a string.
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_onnx_file = "./ep_context_folder_not_expected/invalid_file";
  std::remove(ep_context_onnx_file.c_str());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_onnx_file.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

// Create session 1 to generate context binary file
// Create session 2 to do same thing, make sure session 2 failed because file exist already
// Make sure no new file over write from session 2
TEST_F(QnnHTPBackendTests, QnnContextBinaryGenerationNoOverWrite) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  // Build input model via ModelProto directly (avoid onnxruntime::Model/Graph/Resolve).
  ModelTestBuilder helper;
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);

  // opset imports
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Serialize the model to a string.
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_onnx_file = "./ep_context_no_over_write.onnx";
  std::remove(ep_context_onnx_file.c_str());

  const std::string ep_context_binary_file = "./ep_context_no_over_write_qnn.bin";

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_onnx_file.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

  auto modify_time_1 = std::filesystem::last_write_time(ep_context_binary_file);

  try {
    Ort::Session session2(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_FAIL);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("exists already."));
    auto modify_time_2 = std::filesystem::last_write_time(ep_context_binary_file);
    ASSERT_EQ(modify_time_1, modify_time_2);
  }

  ASSERT_EQ(std::remove(ep_context_onnx_file.c_str()), 0);
  ASSERT_EQ(std::remove(ep_context_binary_file.c_str()), 0);
}

// Create a model with Cast + Add (quantized)
// cast_input -> Cast -> Q -> DQ ----+
//                                   |
//             input2 -> Q -> DQ -> Add -> Q -> DQ -> output
static GetTestModelFn BuildCastAddTestCase() {
  return [](ModelTestBuilder& builder) {
    // Create Cast node int32 -> float32
    MakeTestInput(builder, "cast_in", TestInputDef<int32_t>({2, 3}, false, {0, 1, 0, 1, 0, 1}));

    std::vector<ONNX_NAMESPACE::AttributeProto> cast_attrs;
    cast_attrs.push_back(builder.MakeScalarAttribute(
        "to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)));

    builder.AddNode("cast_node",
                    "Cast",
                    {"cast_in"},
                    {"cast_out"},
                    "",
                    cast_attrs);

    // Create Add node
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);

    std::string add_input1_qdq =
        AddQDQNodePair<uint8_t>(builder, "add_in1_qdq", "cast_out", q_parameter.scale, q_parameter.zero_point);

    MakeTestInput(builder, "add_in2", TestInputDef<float>({2, 3}, false, data));
    std::string add_input2_qdq =
        AddQDQNodePair<uint8_t>(builder, "add_in2_qdq", "add_in2", q_parameter.scale, q_parameter.zero_point);

    builder.AddNode("Add_node",
                    "Add",
                    {add_input1_qdq, add_input2_qdq},
                    {"add_out"});

    // add_out -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "add_out", q_parameter.scale, q_parameter.zero_point);
  };
}

// Create a model with Add (quantized)
// input1 -> Q -> DQ ----
//                       |
// input2 -> Q -> DQ -> Add -> Q -> DQ -> output
static GetTestModelFn BuildAddTestCase() {
  return [](ModelTestBuilder& builder) {
    std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    gsl::span<float> data_range = gsl::make_span(data);
    QuantParams<uint8_t> q_parameter = GetDataQuantParams<uint8_t>(data_range);

    MakeTestInput(builder, "add_in1", TestInputDef<float>({2, 3}, false, data));
    std::string add_input1_qdq =
        AddQDQNodePair<uint8_t>(builder, "add_in1_qdq", "add_in1", q_parameter.scale, q_parameter.zero_point);

    MakeTestInput(builder, "add_in2", TestInputDef<float>({2, 3}, true, data));
    std::string add_input2_qdq =
        AddQDQNodePair<uint8_t>(builder, "add_in2_qdq", "add_in2", q_parameter.scale, q_parameter.zero_point);

    builder.AddNode("Add_node",
                    "Add",
                    {add_input1_qdq, add_input2_qdq},
                    {"add_out"});

    // add_out -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, "qdq_out", "add_out", q_parameter.scale, q_parameter.zero_point);
  };
}

// Test that models with 2 inputs which has different data type can still generate the context binary
TEST_F(QnnHTPBackendTests, QnnContextBinaryGeneration2InputTypes) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  // Build input model via ModelProto directly (avoid onnxruntime::Model/Graph/Resolve).
  ModelTestBuilder helper;
  BuildCastAddTestCase()(helper);

  // opset imports
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Serialize the model to a string.
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_model_file = "./qnn_context_binary_int32_fp32_inputs_test.onnx";
  std::remove(context_model_file.c_str());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  // clean up
  CleanUpCtxFile(context_model_file);
}

// Generate context cache model from the ONNX models with 2 inputs.
// The generated model should have same input order.
// The input ONNX model is created in the way that the model inputs order
// is different with the order in the graph (topological order).
// It cause issue if the generated model doesn't set the inputs/outputs explicitly.
TEST_F(QnnHTPBackendTests, QnnContextGeneration2InputsOrderIssue) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  const std::string context_model_file = "./qnn_ctx_2_inputs_order_test_gen.onnx";
  std::remove(context_model_file.c_str());

  const std::string source_model_file = "testdata/qnn_ctx_2_inputs_order_test.onnx";
  ModelTestBuilder helper;
  LoadOnnxModelFromFile(source_model_file, helper.model_);

  // opset imports
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  // Load generated context model using ONNX protobuf API.
  ONNX_NAMESPACE::ModelProto model_proto;
  std::ifstream ifs(context_model_file, std::ios::in | std::ios::binary);
  ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << context_model_file;
  ASSERT_TRUE(model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << context_model_file;
  ifs.close();

  // Graph inputs order must be preserved.
  EXPECT_EQ(model_proto.graph().input_size(), 2);
  EXPECT_TRUE(model_proto.graph().input()[0].name() == "attention_mask");
  EXPECT_TRUE(model_proto.graph().input()[1].name() == "Add_input_0");

  // clean up
  CleanUpCtxFile(context_model_file);
}

// TODO
// kOrtSessionOptionEpContextNodeNamePrefix currently does not work for ABI since naming isn't controlled by EP.
// This test is disabled because the node naming functionality is not implemented in the QNN-ABI EP.
// The standard QNN EP uses context_node_name_prefix_ in its node name generation (see qnn_execution_provider.cc),
// but the QNN-ABI EP only reads this value without using it.
TEST_F(QnnHTPBackendTests, DISABLED_QnnContextGenerationNodeNamePrefix) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  std::string node_name_prefix = "node_name_prefix_test";

  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  const std::string context_model_file = "./qnn_ctx_2_inputs_order_test_gen.onnx";
  std::remove(context_model_file.c_str());

  ModelTestBuilder helper;
  LoadOnnxModelFromFile(context_model_file, helper.model_);

  // opset imports
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());
  so.AddConfigEntry(kOrtSessionOptionEpContextNodeNamePrefix, node_name_prefix.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  // Load generated context model using ONNX protobuf API.
  ONNX_NAMESPACE::ModelProto model_proto;
  std::ifstream ifs(context_model_file, std::ios::in | std::ios::binary);
  ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << context_model_file;
  ASSERT_TRUE(model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << context_model_file;

  for (const auto& node : model_proto.graph().node()) {
    if (node.op_type() == "EPContext") {
      EXPECT_NE(node.name().find(node_name_prefix), std::string::npos);
    }
  }

  // clean up
  CleanUpCtxFile(context_model_file);
}

// Run QDQ model on HTP 3 times
// 1st run will generate the Qnn context cache onnx file
// 2nd run directly loads and run from Qnn context cache model
TEST_F(QnnHTPBackendTests, QnnContextBinaryCacheEmbedModeTest) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::string context_model_file = "./qnn_context_binary_test.onnx";
  std::remove(context_model_file.c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  // 2nd run directly loads and run from Qnn context cache model
  std::unordered_map<std::string, std::string> session_option_pairs2;
  session_option_pairs2.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       context_model_file,
                       session_option_pairs2);
  // Clean up
  CleanUpCtxFile(context_model_file);
}

// Run QDQ model on HTP 3 times
// 1st run will generate the Onnx skeleton file + Qnn context cache binary file
// 2nd run directly loads and run from Onnx skeleton file + Qnn context cache binary file
TEST_F(QnnHTPBackendTests, QnnContextBinaryCacheNonEmbedModeTest) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  const std::string context_binary_file = "./testdata/qnn_context_cache_non_embed.onnx";
  const std::string qnn_ctx_bin = "./testdata/qnn_context_cache_non_embed_qnn.bin";

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  std::remove(context_binary_file.c_str());
  std::remove(qnn_ctx_bin.c_str());

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Check the Onnx skeleton file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));
  // Check the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(qnn_ctx_bin));

  std::unordered_map<std::string, std::string> session_option_pairs2;
  // Need to set the context file path since TestQDQModelAccuracy load the model from memory
  session_option_pairs2.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  // 2nd run directly loads and run from Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       context_binary_file,
                       session_option_pairs2);

  // load the model from file
  std::vector<char> buffer;
  {
    buffer.resize(std::filesystem::file_size(context_binary_file));
    std::ifstream file(context_binary_file, std::ios::binary);
    if (!file)
      throw std::runtime_error("Error reading model");
    if (!file.read(buffer.data(), buffer.size()))
      throw std::runtime_error("Error reading model");
  }

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  Ort::SessionOptions so;  // No need to set the context file path in so since it's load from file
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

#ifdef _WIN32
  std::wstring ctx_model_file(context_binary_file.begin(), context_binary_file.end());
#else
  std::string ctx_model_file(context_binary_file.begin(), context_binary_file.end());
#endif
  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env.get(), ctx_model_file.c_str(), so));

  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
  ASSERT_EQ(std::remove(qnn_ctx_bin.c_str()), 0);
}

// htp_reused_io_limit_mb: a valid numeric value is accepted and does not block AOT context load.
TEST_F(QnnHTPBackendTests, QnnContextBinary_HtpReusedIoLimitMbValid_LoadsSucceeds) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["htp_reused_io_limit_mb"] = "128";
#if defined(__linux__) && !defined(__aarch64__)
  // On the x86_64 Linux HTP simulator, specify SM8850 to enable support for this config
  // option. On real ARM64 hardware, the SoC model is auto-detected by QNN EP.
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

  const std::string context_binary_file = "./testdata/qnn_context_cache_reused_io_limit.onnx";
  const std::string qnn_ctx_bin = "./testdata/qnn_context_cache_reused_io_limit_qnn.bin";
  std::remove(context_binary_file.c_str());
  std::remove(qnn_ctx_bin.c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // 1st run generates the Onnx skeleton file + Qnn context cache binary file with the
  // option applied during context creation.
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",
                       session_option_pairs);
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));
  EXPECT_TRUE(std::filesystem::exists(qnn_ctx_bin));

  // 2nd run loads the cached context binary from disk with the option applied on the
  // contextCreateFromBinary path (LoadCachedQnnContextFromBuffer).
  std::unordered_map<std::string, std::string> session_option_pairs2;
  session_option_pairs2.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       context_binary_file,
                       session_option_pairs2);

  std::remove(context_binary_file.c_str());
  std::remove(qnn_ctx_bin.c_str());
}

// htp_reused_io_limit_mb: malformed values (negative / non-numeric) are ignored with a
// warning rather than failing session creation.
TEST_F(QnnHTPBackendTests, QnnContextBinary_HtpReusedIoLimitMbMalformed_LoadsSucceeds) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  for (const char* bad_value : {"-1", "10abc"}) {
    ProviderOptions provider_options;
    provider_options["backend_type"] = "htp";
    provider_options["offload_graph_io_quantization"] = "0";
    provider_options["htp_reused_io_limit_mb"] = bad_value;
#if defined(__linux__) && !defined(__aarch64__)
    // On the x86_64 Linux HTP simulator, specify SM8850 to enable support for this config
    // option. On real ARM64 hardware, the SoC model is auto-detected by QNN EP.
    provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif

    auto input_defs = {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                       TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)};
    RunQnnModelTest(BuildOpTestCase<float>("Add_node", "Add", input_defs, {}, {}, kOnnxDomain),
                    provider_options,
                    13,
                    EPVerificationParams{ExpectedEPNodeAssignment::All, ElementwiseAbsoluteVerifier(0.008f)});
  }
}

// Run QDQ model on HTP 2 times
// 1st run will generate the Onnx skeleton file + Qnn context cache binary file
// Then delete the context bin file to make the 2nd sesssion.Initialize() return the status with code INVALID_GRAPH
TEST_F(QnnHTPBackendTests, QnnContextBinaryCache_InvalidGraph) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_binary_file = "./qnn_context_cache_non_embed.onnx";
  std::filesystem::path context_bin = "qnn_context_cache_non_embed_qnn.bin";
  std::remove(context_binary_file.c_str());
  std::remove(context_bin.string().c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_binary_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Check the Onnx skeleton file is generated
  EXPECT_TRUE(std::filesystem::exists(context_binary_file.c_str()));
  // Check the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_bin));
  // Delete the Qnn context cache binary file
  EXPECT_TRUE(std::filesystem::remove(context_bin));

  // loads and run from Onnx skeleton file + Qnn context cache binary file
  onnx::ModelProto model_proto;
  {
    std::ifstream ifs(context_binary_file, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << context_binary_file;
    ASSERT_TRUE(model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << context_binary_file;
  }

  std::string qnn_ctx_model_data;
  model_proto.SerializeToString(&qnn_ctx_model_data);

  Ort::RunOptions run_options;
  run_options.SetRunTag("logger0");

  Ort::SessionOptions so;
  so.SetLogId("logger0");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Verify session creation fails with INVALID_GRAPH when the external context binary is missing.
  try {
    Ort::Session session(*ort_env, qnn_ctx_model_data.data(), qnn_ctx_model_data.size(), so);
    FAIL() << "Expected session creation to fail but it succeeded";
  } catch (const Ort::Exception& e) {
    ASSERT_EQ(e.GetOrtErrorCode(), ORT_INVALID_GRAPH);
  }

  // Clean up
  ASSERT_EQ(std::remove(context_binary_file.c_str()), 0);
}

std::string CreateQnnCtxModelWithNonEmbedMode(std::string external_bin_path) {
  ModelTestBuilder builder;
  const std::vector<int64_t> shape = {2, 3};

  MakeTestInput(builder, "X", TestInputDef<float>(shape, true, {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}));
  builder.MakeOutput<float>("Y", shape);

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(builder.MakeScalarAttribute("embed_mode", static_cast<int64_t>(0)));
  attrs.push_back(builder.MakeStringAttribute("ep_cache_context", external_bin_path));
  attrs.push_back(builder.MakeStringAttribute("partition_name", "QNNExecutionProvider_QNN_1110111000111000111_1_0"));
  attrs.push_back(builder.MakeStringAttribute("source", "QNN"));

  builder.AddNode("ep_ctx",
                  "EPContext",
                  {"X"},
                  {"Y"},
                  kMSDomain,
                  attrs);

  // opset imports
  const std::unordered_map<std::string, int> domain_to_version = {{"", 11}, {kMSDomain, 1}};
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{builder.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  builder.model_.SerializeToString(&model_data);
  return model_data;
}

// Create a model with EPContext node. Set the node property ep_cache_context has ".."
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryRelativePathTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("../qnn_context.bin");

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.SetLogId("qnn_ctx_model_logger");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Verify session creation fails with INVALID_GRAPH.
  try {
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
    FAIL() << "Expected session creation to fail but it succeeded";
  } catch (const Ort::Exception& e) {
    ASSERT_EQ(e.GetOrtErrorCode(), ORT_INVALID_GRAPH);
  }
}

// Create a model with EPContext node. Set the node property ep_cache_context has absolute path
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryAbsolutePathTest) {
#if defined(_WIN32)
  std::string external_ctx_bin_path = "D:/qnn_context.bin";
#else
  std::string external_ctx_bin_path = "/data/qnn_context.bin";
#endif
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode(external_ctx_bin_path);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.SetLogId("qnn_ctx_model_logger");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Verify session creation fails with INVALID_GRAPH.
  try {
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
    FAIL() << "Expected session creation to fail but it succeeded";
  } catch (const Ort::Exception& e) {
    ASSERT_EQ(e.GetOrtErrorCode(), ORT_INVALID_GRAPH);
  }
}

// Create a model with EPContext node. Set the node property ep_cache_context to a file not exist
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryFileNotExistTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("qnn_context_not_exist.bin");

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.SetLogId("qnn_ctx_model_logger");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./qnn_context_not_exist.onnx");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Verify session creation fails with INVALID_GRAPH.
  try {
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
    FAIL() << "Expected session creation to fail but it succeeded";
  } catch (const Ort::Exception& e) {
    ASSERT_EQ(e.GetOrtErrorCode(), ORT_INVALID_GRAPH);
  }
}

// Create a model with EPContext node. Set the node property ep_cache_context to empty string
// Verify that it return INVALID_GRAPH status
TEST_F(QnnHTPBackendTests, QnnContextBinaryFileEmptyStringTest) {
  std::string model_data = CreateQnnCtxModelWithNonEmbedMode("");

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  so.SetLogId("qnn_ctx_model_logger");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "./test_ctx.onnx");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Verify session creation fails with INVALID_GRAPH.
  try {
    Ort::Session session(*ort_env, model_data.data(), model_data.size(), so);
    FAIL() << "Expected session creation to fail but it succeeded";
  } catch (const Ort::Exception& e) {
    ASSERT_EQ(e.GetOrtErrorCode(), ORT_INVALID_GRAPH);
  }
}

// Run QDQ model on HTP with 2 inputs
// 1st run will generate the Qnn context cache onnx file
// 2nd run directly loads and run from Qnn context cache model
TEST_F(QnnHTPBackendTests, QnnContextBinary2InputsTest) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_model_file = "./qnn_context_binary_2inputs_test.onnx";
  std::remove(context_model_file.c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);

  const TestInputDef<float> input_def1({1, 2, 3}, false, -10.0f, 10.0f);
  const TestInputDef<float> input_def2({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Add";

  // Runs model with DQ-> Add-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def1, input_def2}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def1, input_def2}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Make sure the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));

  // 2nd run directly loads and run from Qnn context cache model
  std::unordered_map<std::string, std::string> session_option_pairs2;
  session_option_pairs2.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def1, input_def2}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def1, input_def2}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       context_model_file,
                       session_option_pairs2);
  // Clean up
  CleanUpCtxFile(context_model_file);
}

// Context binary only contains a single QNN graph, generated context cache model (detached mode) only has 1 EPContext node
// Create another Onnx model which also reference to the bin file,
// but the node name is not same with the QNN graph name inside the bin file.
// This is to support backward compatible for the models generated before the PR that
// make context generation support multi-partition
TEST_F(QnnHTPBackendTests, QnnContextBinaryCache_SingleNodeNameNotMatchGraphNameInCtx) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  const std::string context_model_file = "./qnn_context_cache_non_embed.onnx";
  std::filesystem::path context_bin = "qnn_context_cache_non_embed_qnn.bin";
  std::remove(context_model_file.c_str());
  std::remove(context_bin.string().c_str());

  std::unordered_map<std::string, std::string> session_option_pairs;
  session_option_pairs.emplace(kOrtSessionOptionEpContextEnable, "1");
  session_option_pairs.emplace(kOrtSessionOptionEpContextFilePath, context_model_file);
  session_option_pairs.emplace(kOrtSessionOptionEpContextEmbedMode, "0");

  const TestInputDef<float> input_def({1, 2, 3}, false, -10.0f, 10.0f);
  const std::string op_type = "Atan";

  // Runs model with DQ-> Atan-> Q and compares the outputs of the CPU and QNN EPs.
  // 1st run will generate the Onnx skeleton file + Qnn context cache binary file
  TestQDQModelAccuracy(BuildOpTestCase<float>(op_type + "_node", op_type, {input_def}, {}, {}),
                       BuildQDQOpTestCase<uint8_t>(op_type + "_node", op_type, {input_def}, {}, {}),
                       provider_options,
                       14,
                       ExpectedEPNodeAssignment::All,
                       QDQTolerance(),
                       OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
                       "",  // context model file path, not required for this inference
                       session_option_pairs);

  // Check the Onnx skeleton file is generated
  EXPECT_TRUE(std::filesystem::exists(context_model_file.c_str()));
  // Check the Qnn context cache binary file is generated
  EXPECT_TRUE(std::filesystem::exists(context_bin));

  // Create another model that references the context bin but uses a different EPContext node name.
  ModelTestBuilder builder;
  std::vector<int64_t> shape = {1, 2, 3};

  MakeTestInput(builder, "quant_input_defs_0", TestInputDef<float>(shape, false, {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f}));
  builder.MakeOutput<float>("qdq_out_dq_out", shape);

  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(builder.MakeScalarAttribute("embed_mode", static_cast<int64_t>(0)));
  attrs.push_back(builder.MakeStringAttribute("ep_cache_context", context_bin.string()));
  attrs.push_back(builder.MakeStringAttribute("partition_name", "QNNExecutionProvider_QNNExecutionProvider_10006633230345173511_0_0"));
  attrs.push_back(builder.MakeStringAttribute("source", "QNNExecutionProvider"));

  builder.AddNode("ep_ctx",
                  "EPContext",
                  {"quant_input_defs_0"},
                  {"qdq_out_dq_out"},
                  kMSDomain,
                  attrs);

  // opset imports
  const std::unordered_map<std::string, int> domain_to_version = {{"", 11}, {kMSDomain, 1}};
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{builder.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  builder.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  builder.model_.SerializeToString(&model_data);

  Ort::SessionOptions so;
  so.SetLogId("qnn_ctx_model_logger");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Session creation should succeed even if EPContext node name does not match graph name in the bin file.
  EXPECT_NO_THROW((Ort::Session(*ort_env, model_data.data(), model_data.size(), so)));

  // Clean up
  ASSERT_EQ(std::remove(context_model_file.c_str()), 0);
  ASSERT_EQ(std::remove(context_bin.string().c_str()), 0);
}

// Model has 2 EPContext nodes, both with main_context=1 and embedded context binary
TEST_F(QnnHTPBackendTests, QnnMultiContextEmbeded) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, ORT_TSTR("testdata/qnn_ctx/qnn_multi_ctx_embed.onnx"), so));
}

// Model has 2 EPContext nodes, both with main_context=1 and external context binary
TEST_F(QnnHTPBackendTests, QnnMultiContextExternal) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, ORT_TSTR("testdata/qnn_ctx/qnn_multi_ctx_external.onnx"), so));
}

static void CreateQdqModel(const std::string& model_file_name) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildAddTestCase()(helper);

  // opset imports
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  // Serialize to file
#if defined(_WIN32)
  const std::wstring model_file_name_w(model_file_name.begin(), model_file_name.end());
  std::ofstream model_ofs(model_file_name_w, std::ios::binary);
#else
  std::ofstream model_ofs(model_file_name, std::ios::binary);
#endif
  ASSERT_TRUE(model_ofs.good());
  ASSERT_TRUE(helper.model_.SerializeToOstream(&model_ofs));
  model_ofs.close();
}

static void DumpModelWithSharedCtx(ProviderOptions provider_options,
                                   const std::string& onnx_model_path1,
                                   const std::string& onnx_model_path2) {
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
  // enable ep.share_ep_contexts so that QNNEP share the QnnBackendManager across sessions
  so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

#ifndef __aarch64__
#ifndef _M_ARM64
  // weight sharing only available for v73 and higher
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SC8380XP);
#endif  // !_M_ARM64
#endif  // !__aarch64__

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Create 2 sessions to generate context binary models, the 1st session will share the QnnBackendManager
  // to the 2nd session, so graphs from these 2 models are all included in the 2nd context binary
#if defined(_WIN32)
  const std::wstring onnx_model_path1_w(onnx_model_path1.begin(), onnx_model_path1.end());
  const std::wstring onnx_model_path2_w(onnx_model_path2.begin(), onnx_model_path2.end());
  ScopedOrtSession scoped1(std::move(registered_ep_device), Ort::Session(*ort_env, onnx_model_path1_w.c_str(), so));

  so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
  Ort::Session session2(*ort_env, onnx_model_path2_w.c_str(), so);
#else
  ScopedOrtSession scoped1(std::move(registered_ep_device), Ort::Session(*ort_env, onnx_model_path1.c_str(), so));

  so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
  Ort::Session session2(*ort_env, onnx_model_path2.c_str(), so);
#endif
}

static void GetModelInputNames(const std::string& model_path,
                               std::vector<std::string>& input_names,
                               std::vector<std::string>& output_names) {
  // Load model using ONNX protobuf API (avoid onnxruntime::Model).
  ONNX_NAMESPACE::ModelProto model_proto;
  std::ifstream ifs(model_path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << model_path;
  ASSERT_TRUE(model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << model_path;

  // Graph inputs/outputs are ValueInfoProto. Name order is as stored in the model.
  for (const auto& input : model_proto.graph().input()) {
    input_names.push_back(input.name());
  }

  for (const auto& output : model_proto.graph().output()) {
    output_names.push_back(output.name());
  }
}

// 1. Create 2 QDQ models
// 2. Initialize 2 Ort sessions which share the same QNN EP from these 2 QDQ models
// with EpContextEnable = 1, to dump the context binary
// so, the 2nd context binary contains the graph from the 1st model
// 3. Start 2 ort session from the dumped context model,
// The 2nd session uses graph from 1st session
// 4. Run the 2nd session
TEST_F(QnnHTPBackendTests, QnnContextShareAcrossSessions) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    !(QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 34))
  GTEST_SKIP() << "HTP weight sharing on ARM64 requires QNN API version >= 2.34.";
#elif defined(__ANDROID__)
  GTEST_SKIP() << "Weight sharing on Android devices is disabled";
#else
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  // cleanup in case some failure test doesn't remove them
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path);
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1);
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2);
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());
  // 2 *_ctx.onn point to same .bin file
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  // only load and run the session on real device
#if defined(__aarch64__) || defined(_M_ARM64)
  Ort::SessionOptions so1;
  so1.SetLogId("so1");
  so1.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so1, kQnnExecutionProvider, provider_options);

  Ort::SessionOptions so2;
  so2.SetLogId("so2");
  so2.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  so2.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, provider_options);

  EXPECT_TRUE(2 == ctx_model_paths.size());
#ifdef _WIN32
  std::wstring ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::wstring ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#else
  std::string ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::string ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#endif
  ScopedOrtSession scoped1(std::move(registered_ep_device), Ort::Session(*ort_env, ctx_model_file1.c_str(), so1));
  Ort::Session session2(*ort_env, ctx_model_file2.c_str(), so2);  // session2 borrows the EP device from scoped1; must be declared after scoped1.

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(ctx_model_paths[1], input_names, output_names);

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{2, 3};
  std::vector<float> input_value(2 * 3, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_outputs1 = scoped1.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                            output_names_c.data(), 1);
#endif

  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }
  std::remove(qnn_ctx_binary_file_name1.c_str());
#endif
}

TEST_F(QnnHTPBackendTests, DISABLED_VTCMBackupBufferSharing) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    !(QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 34))
  GTEST_SKIP() << "HTP weight sharing on ARM64 requires QNN API version >= 2.34.";
#elif defined(__ANDROID__)
  GTEST_SKIP() << "Weight sharing on Android devices is disabled";
#else

  // Disable the test on test-android job in Qualcomm CI here while we investigate
  // but do not upstream this change.
  ProviderOptions provider_options;
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["backend_type"] = "htp";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  // cleanup in case some failure test doesn't remove them
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path);
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1);
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2);
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());
  // 2 *_ctx.onn point to same .bin file
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  provider_options["enable_vtcm_backup_buffer_sharing"] = "1";
  // only load and run the session on real device
#if defined(__aarch64__) || defined(_M_ARM64)
  Ort::SessionOptions so1;
  so1.SetLogId("so1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so1, kQnnExecutionProvider, provider_options);

  Ort::SessionOptions so2;
  so2.SetLogId("so2");
  so2.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, provider_options);

  EXPECT_TRUE(2 == ctx_model_paths.size());
#ifdef _WIN32
  std::wstring ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::wstring ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#else
  std::string ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::string ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#endif
  ScopedOrtSession scoped1(std::move(registered_ep_device), Ort::Session(*ort_env, ctx_model_file1.c_str(), so1));
  Ort::Session session2(*ort_env, ctx_model_file2.c_str(), so2);  // session2 borrows the EP device from scoped1; must be declared after scoped1.

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(ctx_model_paths[1], input_names, output_names);

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{2, 3};
  std::vector<float> input_value(2 * 3, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_outputs1 = scoped1.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                            output_names_c.data(), 1);
#endif

  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }
  std::remove(qnn_ctx_binary_file_name1.c_str());
#endif
}

TEST_F(QnnHTPBackendTests, FileMapping_Off) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    !(QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 34))
  GTEST_SKIP() << "HTP weight sharing on ARM64 requires QNN API version >= 2.34.";
#elif defined(__ANDROID__)
  GTEST_SKIP() << "Weight sharing on Android devices is disabled";
#else

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["disable_file_mapped_weights"] = "1";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  // cleanup in case some failure test doesn't remove them
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path);
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1);
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2);
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());
  // 2 *_ctx.onn point to same .bin file
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  // only load and run the session on real device
#if defined(__aarch64__) || defined(_M_ARM64)
  Ort::SessionOptions so1;
  so1.SetLogId("so1");
  so1.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");
  Ort::SessionOptions so2;

  // Test CreateFromBinaryListAsync path
  so2.SetLogId("so2");
  so2.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

  EXPECT_TRUE(2 == ctx_model_paths.size());
#ifdef _WIN32
  std::wstring ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::wstring ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#else
  std::string ctx_model_file1(ctx_model_paths[0].begin(), ctx_model_paths[0].end());
  std::string ctx_model_file2(ctx_model_paths[1].begin(), ctx_model_paths[1].end());
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so1, kQnnExecutionProvider, provider_options);

  provider_options["enable_vtcm_backup_buffer_sharing"] = "1";
  so2.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, provider_options);
  ScopedOrtSession scoped1(std::move(registered_ep_device), Ort::Session(*ort_env, ctx_model_file1.c_str(), so1));
  Ort::Session session2(*ort_env, ctx_model_file2.c_str(), so2);  // session2 borrows the EP device from scoped1; must be declared after scoped1.

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(ctx_model_paths[1], input_names, output_names);

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{2, 3};
  std::vector<float> input_value(2 * 3, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_outputs1 = scoped1.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                            output_names_c.data(), 1);
  auto ort_outputs2 = session2.Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                   output_names_c.data(), 1);
#endif  // defined(__aarch64__) || defined(_M_ARM64)

  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }
  std::remove(qnn_ctx_binary_file_name1.c_str());
#endif
}

// For Ort sessions to generate the context binary, with session option ep.share_ep_contexts enabled
// Ort sessions will share the QnnBackendManager, so that all graphs from all models compile into the same Qnn context
TEST_F(QnnHTPBackendTests, QnnContextGenWeightSharingSessionAPI) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    !(QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 34))
  GTEST_SKIP() << "HTP weight sharing on ARM64 requires QNN API version >= 2.34.";
#elif defined(__ANDROID__)
  GTEST_SKIP() << "Weight sharing on Android devices is disabled";
#else
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  // Create QDQ models
  std::vector<std::string> onnx_model_paths{"./weight_share1.onnx", "./weight_share2.onnx"};
  // cleanup in case some failure test doesn't remove them
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path);
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1);
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2);
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());

  // 2 *_ctx.onn point to same .bin file
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  // clean up
  for (auto model_path : onnx_model_paths) {
    ASSERT_EQ(std::remove(model_path.c_str()), 0);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    ASSERT_EQ(std::remove(ctx_model_path.c_str()), 0);
  }
  ASSERT_EQ(std::remove(qnn_ctx_binary_file_name1.c_str()), 0);
#endif
}

// Session created from array wth ep.context_enable enabled without ep.context_file_path
// Error message expected
TEST_F(QnnHTPBackendTests, LoadFromArrayWithQnnEpContextGenPathValidation) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  // Graph& graph = model.MainGraph();
  ModelTestBuilder helper;
  bool single_ep_node = true;
  BuildGraphWithQAndNonQ(single_ep_node)(helper);

  // Serialize the model to a string.
  std::string model_data;
  helper.model_.SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_model_file = "./qnn_context_binary_multi_partition_test.onnx";
  std::remove(context_model_file.c_str());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  try {
    Ort::Session session1(*ort_env, model_data_span.data(), model_data_span.size(), so);
  } catch (const std::exception& e) {
    std::string e_message1(std::string(e.what()));
    ASSERT_TRUE(e_message1.find("Please specify a valid ep.context_file_path") != std::string::npos);
  }

  try {
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "");
    Ort::Session session2(*ort_env, model_data_span.data(), model_data_span.size(), so);
  } catch (const std::exception& ex) {
    std::string e_message2(std::string(ex.what()));
    ASSERT_TRUE(e_message2.find("Please specify a valid ep.context_file_path") != std::string::npos);
  }
}

TEST_F(QnnHTPBackendTests, QnnEpDynamicOptions) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  Ort::SessionOptions so;
  so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, ORT_TSTR("testdata/qnn_ctx/qnn_multi_ctx_embed.onnx"), so));

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames("testdata/qnn_ctx/qnn_multi_ctx_embed.onnx", input_names, output_names);

  // Run sessions
  // prepare input
  std::vector<int64_t> input_dim{3, 4};
  std::vector<float> input_value(3 * 4, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  auto ort_output = scoped.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                         output_names_c.data(), 1);

  const char* const workload_type[] = {"ep.dynamic.workload_type"};
  const char* const efficient_type[] = {"Efficient"};
  const char* const default_type[] = {"Default"};

  // Test Efficient & Default options
  scoped.session().SetEpDynamicOptions(workload_type, efficient_type, 1);
  ort_output = scoped.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                    output_names_c.data(), 1);

  scoped.session().SetEpDynamicOptions(workload_type, default_type, 1);
  ort_output = scoped.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                    output_names_c.data(), 1);

  // Test invalid EP dynamic option and invalid workload type
  const char* const dne[] = {"DNE"};
  try {
    scoped.session().SetEpDynamicOptions(workload_type, dne, 1);
    FAIL() << "Expected exception to be thrown for workload type DNE but was set successfully";
  } catch (const std::exception& e) {
    EXPECT_STREQ("Invalid EP Workload Type.", e.what());
  }

  try {
    scoped.session().SetEpDynamicOptions(dne, efficient_type, 1);
    FAIL() << "Expected exception to be thrown for dynamic option DNE but was set successfully";
  } catch (const std::exception& e) {
    EXPECT_STREQ("Unsupported EP Dynamic Option", e.what());
  }

  const char* const htp_perf_mode_type[] = {"ep.dynamic.qnn_htp_performance_mode"};
  const char* const eps_type[] = {"extreme_power_saver"};
  const char* const shp_type[] = {"sustained_high_performance"};
  scoped.session().SetEpDynamicOptions(htp_perf_mode_type, shp_type, 1);
  ort_output = scoped.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                    output_names_c.data(), 1);

  scoped.session().SetEpDynamicOptions(htp_perf_mode_type, eps_type, 1);
  ort_output = scoped.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                    output_names_c.data(), 1);

  scoped.session().SetEpDynamicOptions(htp_perf_mode_type, shp_type, 1);
  ort_output = scoped.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                    output_names_c.data(), 1);
}

// Implementation of OrtOutStreamWriteFunc that writes the compiled model to a file.
static OrtStatus* ORT_API_CALL TestWriteToStream(void* stream_state, const void* buffer, size_t buffer_num_bytes) {
  std::ofstream* outfile = reinterpret_cast<std::ofstream*>(stream_state);
  outfile->write(reinterpret_cast<const char*>(buffer), buffer_num_bytes);
  return nullptr;  // No error
}

// Implementation of OrtOutStreamWriteFunc that directly returns an OrtStatus indicating an error.
static OrtStatus* ORT_API_CALL ReturnStatusFromStream(void* stream_state, const void* buffer, size_t buffer_num_bytes) {
  QNN_TEST_UNUSED_PARAMETER(stream_state);
  QNN_TEST_UNUSED_PARAMETER(buffer);
  QNN_TEST_UNUSED_PARAMETER(buffer_num_bytes);
  return Ort::GetApi().CreateStatus(ORT_FAIL, "Error from OrtOutStreamWriteFunc callback");
}

// Test using the CompileModel() API with settings:
//   - input model comes from a file
//   - write output model to custom write stream
TEST_F(QnnHTPBackendTests, CompileApi_InputFile_WriteOutputModelBytes) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_inputfile_writeoutputmodelbytes.onnx");
  std::filesystem::remove(input_model_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  ASSERT_TRUE(test_model.Save(input_model_file));

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  const ORTCHAR_T* output_model_file = ORT_TSTR("compileapi_inputfile_writeoutputmodelbytes_ctx.onnx");
  std::filesystem::remove(output_model_file);

  // Open an output file. Test will incrementally write the output model to file
  // via calls to our OrtOutStreamWriteFunc callback.
  ASSERT_FALSE(std::filesystem::exists(output_model_file));
  std::ofstream outfile(output_model_file, std::ios::binary);

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelWriteFunc(TestWriteToStream, reinterpret_cast<void*>(&outfile));
  compile_options.SetEpContextEmbedMode(true);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  outfile.flush();
  outfile.close();

  // Check that the compiled model has the expected number of EPContext nodes.
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);
}

// Tests using an OrtOutStreamFunc function that returns an error.
TEST_F(QnnHTPBackendTests, CompileApi_OutputStream_ReturnStatus) {
  // Create a test model (in memory).
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  std::string model_data = test_model.Serialize();

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelFromBuffer(reinterpret_cast<const void*>(model_data.data()), model_data.size());
  compile_options.SetOutputModelWriteFunc(ReturnStatusFromStream, nullptr);  // Set output stream that returns error
  compile_options.SetEpContextEmbedMode(true);

  // Compile the model. Expect a specific error status.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
  EXPECT_EQ(status.GetErrorMessage(), "Error from OrtOutStreamWriteFunc callback");
}

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
// Tests setting num_graph_prepare_threads to compile model
// 1. Compile model with 2 threads
// 2. Check for successful compilation (_ctx.onnx model should exist)
// 3. Execute compiled model successfully
TEST_F(QnnHTPBackendTests, QnnContextBinary_SetNumGraphPrepareThreads_InRange) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  std::filesystem::path output_model_file("mul_1_ctx.onnx");
  std::filesystem::remove(output_model_file);
  {
    // Compile a model with QNN. This should succeed.
    ProviderOptions qnn_options = {{"backend_type", "htp"}, {"num_graph_prepare_threads", "2"}};

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());
    so.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file.c_str()));
  }

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(output_model_file.string(), input_names, output_names);

  // Run session with compiled model
  // prepare input
  std::vector<int64_t> input_dim{3, 2};
  std::vector<float> input_value(3 * 2, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  ProviderOptions qnn_options = {{"backend_type", "htp"}};
  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, output_model_file.native().c_str(), so));
  auto outputs = scoped.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                      output_names_c.data(), 1);

  std::filesystem::remove(output_model_file);
}

// Tests setting num_graph_prepare_threads to an outrageous number to compile model
// 1. Attempt to compile model with 100 threads
//    a. Number of threads should default to something reasonable, and compilation should still be successful
// 2. Check for successful compilation (_ctx.onnx model should exist)
// 3. Execute compiled model successfully
TEST_F(QnnHTPBackendTests, QnnContextBinary_SetNumGraphPrepareThreads_OutOfRange) {
  const ORTCHAR_T* input_model_file = ORT_MODEL_FOLDER "mul_1.onnx";
  std::filesystem::path output_model_file("mul_1_ctx.onnx");
  std::filesystem::remove(output_model_file);
  {
    // Compile a model with QNN. This should succeed.
    // number of threads should default back to 8 or the max supported by platform (whichever is lower)
    ProviderOptions qnn_options = {{"backend_type", "htp"}, {"num_graph_prepare_threads", "100"}};

    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, output_model_file.string().c_str());
    so.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "1");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, input_model_file, so));
    ASSERT_TRUE(std::filesystem::exists(output_model_file.c_str()));
  }

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  GetModelInputNames(output_model_file.string(), input_names, output_names);

  // Run session with compiled model
  // prepare input
  std::vector<int64_t> input_dim{3, 2};
  std::vector<float> input_value(3 * 2, 0.0f);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names_c;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto input_tensor = Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                 input_dim.data(), input_dim.size());
    ort_inputs.push_back(std::move(input_tensor));
    input_names_c.push_back(input_names[i].c_str());
  }
  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < output_names.size(); ++i) {
    output_names_c.push_back(output_names[i].c_str());
  }

  ProviderOptions qnn_options = {{"backend_type", "htp"}};
  Ort::SessionOptions so;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, qnn_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, output_model_file.native().c_str(), so));
  auto outputs = scoped.session().Run(Ort::RunOptions{}, input_names_c.data(), ort_inputs.data(), ort_inputs.size(),
                                      output_names_c.data(), 1);
  std::filesystem::remove(output_model_file);
}
#endif  // _WIN32 && (defined(__aarch64__) || defined(_M_ARM64))

struct CustomInitializerHandlerState {
  const ORTCHAR_T* external_file_path = nullptr;
  std::ofstream* outfile = nullptr;
};

static OrtStatus* ORT_API_CALL TestHandleInitializerDataFunc(void* state,
                                                             const char* initializer_name,
                                                             const OrtValue* c_initializer_value,
                                                             const OrtExternalInitializerInfo* /*c_external_info*/,
                                                             OrtExternalInitializerInfo** c_new_external_info) {
  Ort::Status final_status{nullptr};

  try {
    CustomInitializerHandlerState* custom_state = reinterpret_cast<CustomInitializerHandlerState*>(state);

    if (std::string("constant") == initializer_name) {
      // Keep a specific initializer in the model just to test both scenarios.
      // A real implementation may check the byte size and keep small initializers in the model.
      *c_new_external_info = nullptr;
      return nullptr;
    }

    //
    // Store other initializers in an external file.
    //
    Ort::ConstValue value{c_initializer_value};
    size_t byte_size = value.GetTensorSizeInBytes();
    int64_t offset = custom_state->outfile->tellp();
    const ORTCHAR_T* location = custom_state->external_file_path;

    custom_state->outfile->write(static_cast<const char*>(value.GetTensorRawData()), byte_size);
    custom_state->outfile->flush();

    // Provide caller (ORT) with the new external info.
    Ort::ExternalInitializerInfo new_external_info{nullptr};
    if (Ort::Status status = Ort::ExternalInitializerInfo::Create(location, offset, byte_size, new_external_info);
        !status.IsOK()) {
      return status.release();
    }

    *c_new_external_info = new_external_info.release();
  } catch (const Ort::Exception& ex) {
    final_status = Ort::Status{ex};
  } catch (const std::exception& ex) {
    final_status = Ort::Status(ex.what(), ORT_FAIL);
  }

  return final_status.release();
}

// Test using the CompileModel() API with settings:
//   - input model comes from a file
//   - write output model to a file
//   - Use callback to specify where each initializer is stored (i.e., external file or within model).
TEST_F(QnnHTPBackendTests, CompileApi_InputFile_OutputFile_InitializerHandler) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("./compileapi_inputfile_outputfile_initializerhandler.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("./compileapi_inputfile_outputfile_initializerhandler_ctx.onnx");
  const ORTCHAR_T* initializer_file = ORT_TSTR("./compileapi_inputfile_outputfile_initializerhandler.bin");
  std::filesystem::remove(input_model_file);
  std::filesystem::remove(output_model_file);
  std::filesystem::remove(initializer_file);

  // Create a test model and save it to a file.
  TestModel test_model;
  CreateTestModel(BuildGraphWithQAndNonQ(false), 21, OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, test_model);
  ASSERT_TRUE(test_model.Save(input_model_file));

  // Initialize session options with QNN EP
  Ort::SessionOptions so;
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // Open a file to store external initializers. ORT will call our handler function for every initializer.
  ASSERT_FALSE(std::filesystem::exists(initializer_file));
  std::ofstream outfile(initializer_file, std::ios::binary);
  CustomInitializerHandlerState custom_state = {initializer_file, &outfile};

  // Create model compilation options from the session options.
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetOutputModelGetInitializerLocationFunc(TestHandleInitializerDataFunc,
                                                           reinterpret_cast<void*>(&custom_state));
  compile_options.SetEpContextEmbedMode(true);
  compile_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  outfile.flush();
  outfile.close();

  ASSERT_TRUE(std::filesystem::exists(initializer_file));
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  CheckEpContextNodeCounts(output_model_file, 2, 2);
}

static OrtStatus* ORT_API_CALL ReuseExternalInitializers(void* state,
                                                         const char* /*initializer_name*/,
                                                         const OrtValue* /*initializer_value*/,
                                                         const OrtExternalInitializerInfo* external_info,
                                                         OrtExternalInitializerInfo** new_external_info) {
  Ort::Status final_status{nullptr};

  try {
    // If the original initializer was stored in an external file, keep it there (just for testing).
    if (external_info != nullptr) {
      Ort::ConstExternalInitializerInfo info(external_info);
      auto location = info.GetFilePath();
      int64_t offset = info.GetFileOffset();
      size_t byte_size = info.GetByteSize();

      Ort::ExternalInitializerInfo new_info(nullptr);
      Ort::Status status = Ort::ExternalInitializerInfo::Create(location.c_str(), offset, byte_size, new_info);
      if (!status.IsOK()) {
        return status.release();
      }

      *new_external_info = new_info.release();

      // Keep track of number of reused external initializers so that we can assert
      // that we reused the expected number of initializers.
      // THIS IS TEST CODE. An application would not do this.
      size_t* num_reused_ext_initializers = reinterpret_cast<size_t*>(state);
      *num_reused_ext_initializers += 1;

      return nullptr;
    }

    // If not originally external, save it within the generated compiled model
    *new_external_info = nullptr;
  } catch (const Ort::Exception& ex) {
    final_status = Ort::Status{ex};
  } catch (const std::exception& ex) {
    final_status = Ort::Status(ex.what(), ORT_FAIL);
  }

  return final_status.release();
}

// Test using the CompileModel() API with settings:
//   - input model comes from a file
//   - write output model to a file
//   - Use callback to specify where each initializer is stored. We'll reuse external initializers
//     from original model!
TEST_F(QnnHTPBackendTests, CompileApi_InitializerHandler_ReuseExternalInitializers) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/conv_qdq_external_ini.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("testdata/conv_qdq_external_ini_reuse_ctx.onnx");
  std::filesystem::remove(output_model_file);

  size_t num_reused_ext_initializers = 0;

  // Create model compilation options from the session options.
  Ort::SessionOptions so;
  Ort::ModelCompilationOptions compile_options(*ort_env, so);
  compile_options.SetInputModelPath(input_model_file);
  compile_options.SetOutputModelPath(output_model_file);
  compile_options.SetOutputModelGetInitializerLocationFunc(ReuseExternalInitializers,
                                                           reinterpret_cast<void*>(&num_reused_ext_initializers));
  compile_options.SetEpContextEmbedMode(true);

  // Compile the model.
  Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  ASSERT_TRUE(std::filesystem::exists(output_model_file));
  std::filesystem::remove(output_model_file);

  ASSERT_EQ(num_reused_ext_initializers, 2);  // Reused external conv weight and bias.
}

// ============================================================
// prepare_only tests
// ============================================================

// Helper: build session options with prepare_only enabled.
static void SetPrepareOnlyOptions(Ort::SessionOptions& so,
                                  const std::string& ctx_path,
                                  bool explicit_context_enable = true) {
  if (explicit_context_enable) {
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  }
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_path.c_str());
  so.AddConfigEntry("ep.qnnexecutionprovider.enable_htp_prepare_only", "1");
}

// Test 1: prepare_only writes ctx.onnx and frees backend (session create succeeds).
TEST_F(QnnHTPBackendTests, PrepareOnly_CtxFileWritten) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ctx_path = "./qnn_prepare_only_ctx_written_test.onnx";
  std::remove(ctx_path.c_str());

  Ort::SessionOptions so;
  SetPrepareOnlyOptions(so, ctx_path);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));
  EXPECT_TRUE(std::filesystem::exists(ctx_path));

  CleanUpCtxFile(ctx_path);
}

// Test 2: prepare_only without explicit ep.context_enable errors out at session creation
TEST_F(QnnHTPBackendTests, PrepareOnly_ErrorsWhenContextCacheNotSet) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ctx_path = "./qnn_prepare_only_auto_ctx_test.onnx";
  std::remove(ctx_path.c_str());

  Ort::SessionOptions so;
  // Intentionally omit kOrtSessionOptionEpContextEnable — session creation should fail.
  SetPrepareOnlyOptions(so, ctx_path, /*explicit_context_enable=*/false);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  // ep.context_enable was not set, so prepare_only is invalid: GetCapability returns EP_FAIL
  // instead of silently downgrading to a normal JIT session.
  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), so);
    FAIL() << "Expected session creation to fail when prepare_only is set without ep.context_enable";
  } catch (const std::exception& e) {
    ASSERT_THAT(e.what(), testing::HasSubstr("enable_htp_prepare_only=1 requires ep.context_enable=1"));
  }
}

// Test 3: Session.Run() on a prepare_only session returns EP_FAIL with the expected message.
TEST_F(QnnHTPBackendTests, PrepareOnly_RunReturnsError) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ctx_path = "./qnn_prepare_only_run_error_test.onnx";
  std::remove(ctx_path.c_str());

  Ort::SessionOptions so;
  SetPrepareOnlyOptions(so, ctx_path);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));
  auto& session = scoped.session();
  ASSERT_TRUE(std::filesystem::exists(ctx_path));

  // Get actual input/output names from the session — BuildGraphWithQAndNonQ uses
  // auto-generated output names so we cannot hardcode them.
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
  auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);

  std::vector<int64_t> input_dim{200, 200};
  std::vector<float> input_data(200 * 200, 0.0f);
  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                input_dim.data(), input_dim.size()));
  const char* input_names[] = {input_name_ptr.get()};
  const char* output_names[] = {output_name_ptr.get()};

  try {
    session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), 1, output_names, 1);
    FAIL() << "Expected Session.Run() to fail in prepare_only mode";
  } catch (const std::exception& e) {
    ASSERT_THAT(e.what(), testing::HasSubstr("prepare_only mode"));
  }

  CleanUpCtxFile(ctx_path);
}

// ============================================================
// Graph Splitting tests
// ============================================================

// Helper: build session options with Graph Splittling enabled.
[[maybe_unused]] static void SetGraphSplittingOptions(Ort::SessionOptions& so,
                                                      const std::string& ctx_path) {
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_path.c_str());
  so.AddConfigEntry("ep.qnnexecutionprovider.enable_htp_graph_splitting", "1");
}

// Test 1: Graph Splittling enabled with default thread count — context binary is written.
// On SDK 2.48 the Graph Splittling config block is compiled out; the test still passes because
// QnnContext_create succeeds without option 22.
TEST_F(QnnHTPBackendTests, GraphSplittingEnabled_DefaultThreads_CompileSucceeds) {
#if !(defined(QNN_SDK_VERSION_MAJOR) && QNN_SDK_VERSION_MAJOR == 2 && \
      defined(QNN_SDK_VERSION_MINOR) && QNN_SDK_VERSION_MINOR >= 49)
  GTEST_SKIP() << "Graph splitting requires QAIRT SDK 2.49+. Skipping on this SDK build.";
#else
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ctx_path = "./qnn_graph_splitting_default_threads_test.onnx";
  std::remove(ctx_path.c_str());

  Ort::SessionOptions so;
  SetGraphSplittingOptions(so, ctx_path);

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));
  EXPECT_TRUE(std::filesystem::exists(ctx_path));

  CleanUpCtxFile(ctx_path);
#endif
}

// Test 2: Graph Splittling disabled (key unset) — existing path is unchanged, no regression.
TEST_F(QnnHTPBackendTests, GraphSplittingDisabled_NoRegression) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ctx_path = "./qnn_graph_splitting_disabled_test.onnx";
  std::remove(ctx_path.c_str());

  Ort::SessionOptions so;
  // Graph Splittling key is intentionally NOT set — validate no regression.
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_path.c_str());

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));
  EXPECT_TRUE(std::filesystem::exists(ctx_path));

  CleanUpCtxFile(ctx_path);
}

// ==============================================================================
// GPU weight-sharing tests
// ==============================================================================

// [Case 1] Non-GPU backend (HTP) + share_ep_contexts=true:
// HTP weight sharing is active: both sessions compile into the same .bin.
TEST_F(QnnHTPBackendTests, QnnContextGenHtpBackendNoGpuConfig) {
#if (defined(__aarch64__) || defined(_M_ARM64)) && \
    !(QNN_API_VERSION_MAJOR > 2 || (QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR >= 34))
  GTEST_SKIP() << "HTP weight sharing on ARM64 requires QNN API version >= 2.34.";
#elif defined(__ANDROID__)
  GTEST_SKIP() << "Weight sharing on Android devices is disabled";
#else
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  // By default, 8 is used, which will impact time to run all
  // unit tests due to overhead of thread creation/destruction
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V68);

  std::vector<std::string> onnx_model_paths{"./htp_no_gpu_cfg1.onnx", "./htp_no_gpu_cfg2.onnx"};
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateQdqModel(model_path);
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1);
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2);
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());

  // HTP weight sharing active: both _ctx.onnx models must point to the same .bin.
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  EXPECT_TRUE(std::filesystem::file_size(qnn_ctx_binary_file_name1) > 0);

  // clean up
  for (auto model_path : onnx_model_paths) {
    ASSERT_EQ(std::remove(model_path.c_str()), 0);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    ASSERT_EQ(std::remove(ctx_model_path.c_str()), 0);
  }
  ASSERT_EQ(std::remove(qnn_ctx_binary_file_name1.c_str()), 0);
#endif
}

// Builds a simple fp16 Conv graph (float I/O; enable_htp_fp16_precision runs it in fp16 on HTP).
// Verifies that enable_htp_fp16_clamp_overflow is applied on the EPContext (context-binary) path.
// The option is a runtime graph config applied via graphSetConfig after the graph is restored
// from the binary (QnnModel::ApplyRuntimeGraphConfigs). Input and weight values of 300.0f ensure
// the Conv accumulator reliably overflows fp16 max (~65504): without the clamp the output is NaN;
// with it the output saturates to fp16-max (finite), making the isfinite assertion a real fence.
// Requires QAIRT >= 2.49 (QNN API >= 2.38) for the clamp option to be effective; the overflow
// → NaN behavior only manifests on HTP arch v79+ (Glymur/SM8850 and later).
#ifdef QNN_TEST_HTP_FP16_CLAMP_OVERFLOW_AVAILABLE
TEST_F(QnnHTPBackendTests, EPContext_Fp16ClampOverflow_RoundTrip) {
  SKIP_HTP_TEST_ON_ARCH_LESS_THAN_OR_EQUAL_TO(QNN_HTP_DEVICE_ARCH_V75);

  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  provider_options["enable_htp_fp16_precision"] = "1";
  provider_options["enable_htp_fp16_clamp_overflow"] = "1";
#if defined(__linux__) && !defined(__aarch64__)
  provider_options["soc_model"] = std::to_string(QNN_SOC_MODEL_SM8850);
#endif
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  ModelTestBuilder helper;
  BuildFp16ClampOverflowConvTestCase()(helper);
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string context_model_file = "./testdata/qnn_ctx_fp16_clamp_overflow_test.onnx";
  std::remove(context_model_file.c_str());

  // Phase 1: generate the EPContext model + binary (compose/finalize path).
  ONNX_NAMESPACE::ModelProto ctx_model_proto;
  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

    ScopedOrtSession scoped(std::move(registered_ep_device),
                            Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

    ASSERT_TRUE(std::filesystem::exists(context_model_file.c_str()));

    std::ifstream ifs(context_model_file, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.good()) << "Failed to open ONNX file: " << context_model_file;
    ASSERT_TRUE(ctx_model_proto.ParseFromIstream(&ifs)) << "Failed to parse ONNX file: " << context_model_file;
  }

  // Phase 2: reload from the generated context model (binary-load path, where
  // ApplyRuntimeGraphConfigs sets the fp16-clamp graph config), then run and check finite output.
  {
    Ort::SessionOptions so2;
    so2.AddConfigEntry(kOrtSessionOptionEpContextFilePath, context_model_file.c_str());

    RegisteredEpDeviceUniquePtr registered_ep_device;
    // Same provider_options as Phase 1: enable_htp_fp16_clamp_overflow must be set so
    // ApplyRuntimeGraphConfigs fires on the binary-load path inside CompileContextModel.
    RegisterQnnEpLibrary(registered_ep_device, so2, kQnnExecutionProvider, provider_options);

    std::string ctx_model_data;
    ctx_model_proto.SerializeToString(&ctx_model_data);
    ScopedOrtSession scoped(std::move(registered_ep_device),
                            Ort::Session(*ort_env, ctx_model_data.data(), ctx_model_data.size(), so2));

    std::vector<float> input_value(32, 300.0f);
    std::vector<int64_t> input_dim{1, 2, 4, 4};
    Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(Ort::Value::CreateTensor(info, input_value.data(), input_value.size(),
                                                  input_dim.data(), input_dim.size()));
    const char* input_names_c[] = {"input"};
    const char* output_names_c[] = {"output"};

    auto ort_outputs = scoped.session().Run(Ort::RunOptions{}, input_names_c, ort_inputs.data(), ort_inputs.size(),
                                            output_names_c, 1);

    ASSERT_EQ(ort_outputs.size(), 1u);
    const float* out_data = ort_outputs[0].GetTensorData<float>();
    const size_t out_count = ort_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < out_count; ++i) {
      EXPECT_TRUE(std::isfinite(out_data[i])) << "Output element " << i << " is not finite: " << out_data[i];
    }
  }

  CleanUpCtxFile(context_model_file);
}
#endif  // QNN_TEST_HTP_FP16_CLAMP_OVERFLOW_AVAILABLE

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))

// GPU backend does not support QDQ (quantized) ops. This creates a plain float
// Add model that GPU can compile into a context binary.
static void CreateFloatModel(const std::string& model_file_name) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}};

  ModelTestBuilder helper;
  std::vector<float> data = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  MakeTestInput(helper, "add_in1", TestInputDef<float>({2, 3}, false, data));
  MakeTestInput(helper, "add_in2", TestInputDef<float>({2, 3}, true, data));
  helper.AddNode("Add_node", "Add", {"add_in1", "add_in2"}, {"add_out"});
  helper.MakeOutput<float>("add_out", std::vector<int64_t>{2, 3});

  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  const std::wstring model_file_name_w(model_file_name.begin(), model_file_name.end());
  std::ofstream model_ofs(model_file_name_w, std::ios::binary);

  ASSERT_TRUE(model_ofs.good());
  ASSERT_TRUE(helper.model_.SerializeToOstream(&model_ofs));
  model_ofs.close();
}

// [Case 2] GPU backend + weight sharing enabled (ARM64 + share_ep_contexts):
// Observable effect: both sessions compile into the same context binary.
TEST_F(QnnGPUBackendTests, QnnContextGenGpuWeightSharingSessionAPI) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";

  std::vector<std::string> onnx_model_paths{"./gpu_weight_share1.onnx", "./gpu_weight_share2.onnx"};
  for (auto model_path : onnx_model_paths) {
    std::remove(model_path.c_str());
  }

  std::vector<std::string> ctx_model_paths;
  for (auto model_path : onnx_model_paths) {
    CreateFloatModel(model_path);
    EXPECT_TRUE(std::filesystem::exists(model_path.c_str()));
    auto pos = model_path.find_last_of(".");
    if (pos != std::string::npos) {
      model_path = model_path.substr(0, pos) + "_ctx.onnx";
    } else {
      model_path = model_path + "_ctx.onnx";
    }
    ctx_model_paths.push_back(model_path);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    std::remove(ctx_model_path.c_str());
  }

  DumpModelWithSharedCtx(provider_options, onnx_model_paths[0], onnx_model_paths[1]);

  std::string qnn_ctx_binary_file_name1;
  GetContextBinaryFileName(ctx_model_paths[0], qnn_ctx_binary_file_name1);
  EXPECT_TRUE(!qnn_ctx_binary_file_name1.empty());

  std::string qnn_ctx_binary_file_name2;
  GetContextBinaryFileName(ctx_model_paths[1], qnn_ctx_binary_file_name2);
  EXPECT_TRUE(!qnn_ctx_binary_file_name2.empty());

  // Both _ctx.onnx models must point to the same .bin (weight sharing active).
  EXPECT_TRUE(qnn_ctx_binary_file_name1 == qnn_ctx_binary_file_name2);
  auto file_size_1 = std::filesystem::file_size(qnn_ctx_binary_file_name1);
  EXPECT_TRUE(file_size_1 > 0);

  // clean up
  for (auto model_path : onnx_model_paths) {
    ASSERT_EQ(std::remove(model_path.c_str()), 0);
  }
  for (auto ctx_model_path : ctx_model_paths) {
    ASSERT_EQ(std::remove(ctx_model_path.c_str()), 0);
  }
  ASSERT_EQ(std::remove(qnn_ctx_binary_file_name1.c_str()), 0);
}

// [Case 3] GPU backend + share_ep_contexts=false:
// Observable effect: each session compiles into its own separate context binary.
TEST_F(QnnGPUBackendTests, QnnContextGenGpuNoWeightSharing) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "gpu";

  const std::string onnx_model_path1 = "./gpu_no_share1.onnx";
  const std::string onnx_model_path2 = "./gpu_no_share2.onnx";
  std::remove(onnx_model_path1.c_str());
  std::remove(onnx_model_path2.c_str());

  CreateFloatModel(onnx_model_path1);
  CreateFloatModel(onnx_model_path2);
  ASSERT_TRUE(std::filesystem::exists(onnx_model_path1));
  ASSERT_TRUE(std::filesystem::exists(onnx_model_path2));

  // Derive context model paths.
  auto MakeCtxPath = [](const std::string& p) {
    auto pos = p.find_last_of(".");
    return (pos != std::string::npos ? p.substr(0, pos) : p) + "_ctx.onnx";
  };
  const std::string ctx_path1 = MakeCtxPath(onnx_model_path1);
  const std::string ctx_path2 = MakeCtxPath(onnx_model_path2);
  std::remove(ctx_path1.c_str());
  std::remove(ctx_path2.c_str());

  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");

    RegisteredEpDeviceUniquePtr registered_ep_device;
    RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

    const std::wstring path1_w(onnx_model_path1.begin(), onnx_model_path1.end());
    ScopedOrtSession scoped1(std::move(registered_ep_device), Ort::Session(*ort_env, path1_w.c_str(), so));

    const std::wstring path2_w(onnx_model_path2.begin(), onnx_model_path2.end());
    Ort::Session session2(*ort_env, path2_w.c_str(), so);
  }

  // Each model must have produced its own context binary.
  std::string bin1, bin2;
  GetContextBinaryFileName(ctx_path1, bin1);
  GetContextBinaryFileName(ctx_path2, bin2);
  EXPECT_FALSE(bin1.empty());
  EXPECT_FALSE(bin2.empty());
  // Without weight sharing the two sessions produce distinct binaries.
  EXPECT_NE(bin1, bin2);
  EXPECT_GT(std::filesystem::file_size(bin1), 0u);
  EXPECT_GT(std::filesystem::file_size(bin2), 0u);

  // clean up
  ASSERT_EQ(std::remove(onnx_model_path1.c_str()), 0);
  ASSERT_EQ(std::remove(onnx_model_path2.c_str()), 0);
  ASSERT_EQ(std::remove(ctx_path1.c_str()), 0);
  ASSERT_EQ(std::remove(ctx_path2.c_str()), 0);
  ASSERT_EQ(std::remove(bin1.c_str()), 0);
  ASSERT_EQ(std::remove(bin2.c_str()), 0);
}
#endif  // defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))

// Utility class to help create enviornment using HNRD for testing.
// Expected usage is used along with smart pointer to automatically restore temporarily moved libraries.
class HnrdTestHandle {
 public:
  HnrdTestHandle(uint32_t htp_arch) : htp_arch_(htp_arch) {
    // Move Prepare/Skel/Stub libraries to a temporary directory to trigger HNRD.
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    temp_dir_ = std::string("temp_") + info->test_suite_name() + "-" + info->name();

    std::filesystem::create_directory(temp_dir_);
    for (const std::string& lib : GetRelatedLibs()) {
      if (std::filesystem::exists(lib)) {
        std::filesystem::rename(lib, temp_dir_ / lib);
      }
    }
  }

  ~HnrdTestHandle() {
    // Move libraries back from temporary directory for later testcases.
    for (const std::string& lib : GetRelatedLibs()) {
      if (std::filesystem::exists(temp_dir_ / lib)) {
        std::filesystem::rename(temp_dir_ / lib, lib);
      }
    }

    std::filesystem::remove(temp_dir_);
  }

 private:
  std::vector<std::string> GetRelatedLibs() {
#ifdef _WIN32
    return {"QnnHtpPrepare.dll",
            "libQnnHtpV" + std::to_string(htp_arch_) + "Skel.so",
            "QnnHtpV" + std::to_string(htp_arch_) + "Stub.dll"};
#else
    return {"libQnnHtpPrepare.so",
            "libQnnHtpV" + std::to_string(htp_arch_) + "Skel.so",
            "libQnnHtpV" + std::to_string(htp_arch_) + "Stub.so"};
#endif
  }

  uint32_t htp_arch_;
  std::filesystem::path temp_dir_;
};

// ============================================================
// prepare_and_load tests
// ============================================================
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Test 1: prepare_and_load with context_enable=0 (Path A) — inference works in single session.
TEST_F(QnnHTPBackendTests, PrepareAndLoad_PathA_InferenceWorks) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  Ort::SessionOptions so;
  so.AddConfigEntry("ep.qnnexecutionprovider.enable_htp_prepare_and_load", "1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

  // Verify inference works — get actual names from session (BuildGraphWithQAndNonQ uses auto-generated names)
  Ort::Session& session = scoped.session();
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
  auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);

  std::vector<int64_t> input_dim{200, 200};
  std::vector<float> input_data(200 * 200, 0.0f);
  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                input_dim.data(), input_dim.size()));
  const char* input_names[] = {input_name_ptr.get()};
  const char* output_names[] = {output_name_ptr.get()};

  auto results = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), 1, output_names, 1);
  ASSERT_EQ(results.size(), 1u);
}

// Test 2: prepare_and_load with context_enable=1 (Path B) — persists artifact AND inference works.
TEST_F(QnnHTPBackendTests, PrepareAndLoad_PathB_PersistsAndRuns) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ctx_path = "./qnn_prepare_and_load_path_b_test.onnx";
  std::remove(ctx_path.c_str());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_path.c_str());
  so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
  so.AddConfigEntry("ep.qnnexecutionprovider.enable_htp_prepare_and_load", "1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

  // Verify artifact was written
  EXPECT_TRUE(std::filesystem::exists(ctx_path));

  // Verify inference works — get actual names from session
  Ort::Session& session = scoped.session();
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
  auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);

  std::vector<int64_t> input_dim{200, 200};
  std::vector<float> input_data(200 * 200, 0.0f);
  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(Ort::Value::CreateTensor(mem_info, input_data.data(), input_data.size(),
                                                input_dim.data(), input_dim.size()));
  const char* input_names[] = {input_name_ptr.get()};
  const char* output_names[] = {output_name_ptr.get()};

  auto results = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), 1, output_names, 1);
  ASSERT_EQ(results.size(), 1u);

  CleanUpCtxFile(ctx_path);
}

// Test 3: prepare_and_load + prepare_only are mutually exclusive — throws.
TEST_F(QnnHTPBackendTests, PrepareAndLoad_MutuallyExclusive) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ctx_path = "./qnn_prepare_and_load_mutex_test.onnx";
  std::remove(ctx_path.c_str());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_path.c_str());
  so.AddConfigEntry("ep.qnnexecutionprovider.enable_htp_prepare_only", "1");
  so.AddConfigEntry("ep.qnnexecutionprovider.enable_htp_prepare_and_load", "1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  try {
    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));
    FAIL() << "Expected session creation to throw due to mutually exclusive options";
  } catch (const std::exception& e) {
    ASSERT_THAT(e.what(), testing::HasSubstr("mutually exclusive"));
  }

  std::remove(ctx_path.c_str());
}

// Test 4: prepare_and_load with context_enable=0 + context_file_path — contradictory, throws.
TEST_F(QnnHTPBackendTests, PrepareAndLoad_ContextDisabledWithFilePath_Throws) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "some_path.onnx");
  so.AddConfigEntry("ep.qnnexecutionprovider.enable_htp_prepare_and_load", "1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  try {
    ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));
    FAIL() << "Expected session creation to throw due to contradictory options";
  } catch (const std::exception& e) {
    ASSERT_THAT(e.what(), testing::HasSubstr("Contradictory"));
  }
}

// Test 5: prepare_and_load with embed_mode=1 — binary is embedded in .onnx, no separate .bin.
TEST_F(QnnHTPBackendTests, PrepareAndLoad_EmbedModeRespected) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
#if defined(_WIN32) && (defined(__aarch64__) || defined(_M_ARM64))
  provider_options["num_graph_prepare_threads"] = "1";
#endif

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  ModelTestBuilder helper;
  BuildGraphWithQAndNonQ()(helper);
  for (const auto& [domain, version] : domain_to_version) {
    const gsl::not_null<ONNX_NAMESPACE::OperatorSetIdProto*> opset_id_proto{helper.model_.add_opset_import()};
    opset_id_proto->set_domain(domain);
    opset_id_proto->set_version(version);
  }
  helper.model_.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  std::string model_data;
  helper.model_.SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ctx_path = "./qnn_prepare_and_load_embed_override_test.onnx";
  std::remove(ctx_path.c_str());

  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ctx_path.c_str());
  so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "1");  // should be respected
  so.AddConfigEntry("ep.qnnexecutionprovider.enable_htp_prepare_and_load", "1");

  RegisteredEpDeviceUniquePtr registered_ep_device;
  RegisterQnnEpLibrary(registered_ep_device, so, kQnnExecutionProvider, provider_options);

  ScopedOrtSession scoped(std::move(registered_ep_device), Ort::Session(*ort_env, model_data_span.data(), model_data_span.size(), so));

  // Verify ctx.onnx exists
  EXPECT_TRUE(std::filesystem::exists(ctx_path));

  // Verify embed_mode=1 was respected: binary is embedded, no separate .bin file
  auto bin_path = std::filesystem::path(ctx_path).replace_extension("").string() + "_qnn.bin";
  EXPECT_FALSE(std::filesystem::exists(bin_path));

  // Clean up: only the .onnx exists (no .bin to remove in embed mode)
  ASSERT_EQ(std::remove(ctx_path.c_str()), 0);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
